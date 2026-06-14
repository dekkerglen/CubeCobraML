"""
Usage:  python train.py <epochs> <batch_size> <continue_train:true|false> <primary>
"""

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

cpu_only = os.environ.get("CPU_ONLY", "false").lower() == "true"
if cpu_only:
    # Must be set before `import tensorflow` to suppress CUDA device discovery.
    # The Metal plugin ignores this var, so we additionally hide GPUs via the
    # config API below — that call has to happen before any TF op is created.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

if cpu_only:
    tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.metrics import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    KLDivergence,
    TopKCategoricalAccuracy,
)


from ccml.data import build_dataset, build_validation_dataset
from ccml.losses import PickHistogramCCE
from ccml.model import CubeCobraMLSystem
from ccml.tf_metrics import (
    ArgmaxCollapseCount,
    BombAgreement,
    PickHistDivergence,
    TopRatedPercent,
)
from ccml.utils import DATA_DIR, MODEL_DIR
from ccml.callbacks import PeriodicSnapshot, ValidationProgress, load_stability_subset

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
DATA_DIR = TRAIN_DIR  # rebind to point at train for downstream loads

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
continue_flag = sys.argv[3].lower() == "true"
primary_stream = sys.argv[4].lower()

# Device banner — must check LOGICAL devices, not physical. set_visible_devices
# leaves the physical GPU in list_physical_devices() but removes it from logical
# devices and op placement.
logical_gpus = tf.config.list_logical_devices("GPU")
print("\n" + "=" * 70)
if logical_gpus and not cpu_only:
    print("🚀 GPU DETECTED AND ENABLED FOR TRAINING 🚀")
    for gpu in logical_gpus:
        print(f"   Using: {gpu.name}")
else:
    reason = "CPU_ONLY=true" if cpu_only else "no GPU available"
    print(f"🖥️  CPU TRAINING ({reason})")
print("=" * 70 + "\n")

# =============================================================================
# Hyperparameters
# =============================================================================

quick = os.environ.get("QUICK", "false").lower() == "true"

# Draft head priors (Round 2 fixes that work — kept):
#   DRAFT_BIAS_ALPHA: scale of log-pick-rate prior on the draft output bias.
#       α=1.0 starts the per-card bias AT the empirical log pick rate, so the
#       model only deviates when there is signal to do so. Earlier defaults
#       (0.3, 0.5) under-anchored and let the model collapse niche cards.
#   DRAFT_OUTPUT_L2: L2 on the draft output kernel — suppresses noisy rare-card
#       column directions.
#   WEIGHT_DECAY: AdamW decoupled weight decay (biases excluded).
#   PICK_LABEL_SMOOTHING: ε mass distributed uniformly across the pack on the
#       pick target. Keeps non-picked pack cards with target ≥ ε/K instead of
#       0, preventing their logits from collapsing to −∞. Direct fix for the
#       under-collapse pathology (cards humans pick frequently that the model
#       never picks). Default 0.10.
draft_bias_alpha = float(os.environ.get("DRAFT_BIAS_ALPHA", "1.0"))
draft_output_l2 = float(os.environ.get("DRAFT_OUTPUT_L2", "1e-4"))
weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
pick_label_smoothing = float(os.environ.get("PICK_LABEL_SMOOTHING", "0.10"))

# Intra-batch pick-histogram loss — symmetric KL between the batch-mean
# human and model pick distributions, added to the pick CCE. Two-sided, so
# it punishes BOTH 0%-collapse (model never takes a card humans take) and
# 100%-saturation (model always takes a card humans don't). Replaces the
# old calibration loss, whose KL direction was mathematically blind to
# collapse.
hist_loss_weight = float(os.environ.get("HIST_LOSS_WEIGHT", "0.1"))

# Pack-position sample weights — picks early in a pack carry more signal
# (more options) than late forced picks. w = ((K−1)/(pack_start−1))^power
# per pick, normalized by that draft's own pack size. PACK_WEIGHT_POWER=0.5
# (sqrt) keeps mid-pack picks valuable; forced picks get weight 0.
pack_weighting = os.environ.get("PACK_WEIGHTING", "true").lower() == "true"
pack_weight_power = float(os.environ.get("PACK_WEIGHT_POWER", "0.5"))

# Train pick stream reads INTERLEAVE_FILES files round-robin so the shuffle
# buffer spans many files instead of one — decorrelates batches.
interleave_files = int(os.environ.get("INTERLEAVE_FILES", "16"))

# Collapse-metric cohorts: cards offered 1..COLLAPSE_RARE_MAX times in the
# eval pass are "rare", more is "common".
collapse_rare_max = int(os.environ.get("COLLAPSE_RARE_MAX", "10"))

# Stable val subset size for in-callback diagnostics (argmax_churn).
# These are NOT the same as the main val pass — they're a small fixed subset
# evaluated at each snapshot so we can detect basin flips between consecutive
# best checkpoints.
stability_subset_size = int(os.environ.get("STABILITY_SUBSET_SIZE", "5000"))

# Plateau-triggered LR decay: constant LR_PEAK until val_pick_cce stops
# improving for LR_PLATEAU_PATIENCE epochs, then multiply by
# LR_PLATEAU_FACTOR, floored at LR_END. Replaces cosine decay, whose horizon
# (full single-pass ≈ 1035 epochs) meant the LR never actually annealed in
# runs stopped at ~150 epochs.
lr_peak = float(os.environ.get("LR_PEAK", "3e-4"))
lr_end = float(os.environ.get("LR_END", "1e-5"))
lr_plateau_patience = int(os.environ.get("LR_PLATEAU_PATIENCE", "8"))
lr_plateau_factor = float(os.environ.get("LR_PLATEAU_FACTOR", "0.5"))

# =============================================================================
# Build datasets
# =============================================================================

dataset, num_cards, steps_per_epoch, epochs_final = build_dataset(
    cubes_path=os.path.join(TRAIN_DIR, "cubes"),
    decks_path=os.path.join(TRAIN_DIR, "decks"),
    picks_path=os.path.join(TRAIN_DIR, "picks"),
    freq_path=os.path.join(TRAIN_DIR, "oracleFrequency.json"),
    correlations_path=os.path.join(TRAIN_DIR, "correlations.json"),
    batch_size=batch_size,
    target_epochs=epochs,
    primary=primary_stream,
    respect_target_epochs=quick,
    pick_label_smoothing=pick_label_smoothing,
    pack_weighting=pack_weighting,
    pack_weight_power=pack_weight_power,
    interleave_files=interleave_files,
)

val_dataset, val_steps_per_epoch, val_counts = build_validation_dataset(
    cubes_path=os.path.join(TEST_DIR, "cubes"),
    decks_path=os.path.join(TEST_DIR, "decks"),
    picks_path=os.path.join(TEST_DIR, "picks"),
    train_freq_path=os.path.join(TRAIN_DIR, "oracleFrequency.json"),
    train_correlations_path=os.path.join(TRAIN_DIR, "correlations.json"),
    batch_size=batch_size,
    pick_label_smoothing=pick_label_smoothing,
)
if val_dataset is None:
    print("[info] No validation data found in data/test/ — skipping val_* metrics.")

# =============================================================================
# Build bomb_mask + draft_bias_init from pickCount
# =============================================================================

pick_count_path = os.path.join(DATA_DIR, "pickCount.json")
bomb_indices_path = os.path.join(DATA_DIR, "bomb_card_indices.json")

pick_counts = np.array(json.load(open(pick_count_path)), dtype=np.float32)

# Bomb mask: 1.0 for the top-N cards by pickCount, used by BombAgreement.
bomb_mask = np.zeros(num_cards, dtype=np.float32)
if os.path.exists(bomb_indices_path):
    bomb_indices = json.load(open(bomb_indices_path))
    bomb_mask[bomb_indices] = 1.0
    print(f"Bomb cohort: {len(bomb_indices)} cards from bomb_card_indices.json")
else:
    print(f"[warn] {bomb_indices_path} missing — bomb mask is empty.")

# Draft bias init from log marginal pick rate (priors that work).
freq_path = os.path.join(TRAIN_DIR, "oracleFrequency.json")
oracle_freq = np.array(json.load(open(freq_path)), dtype=np.float32)
if draft_bias_alpha > 0.0:
    eps = 1.0
    log_rate = np.log((pick_counts + eps) / (oracle_freq + eps))
    draft_bias_init = (draft_bias_alpha * log_rate).astype(np.float32)
    print(
        f"Draft bias init: α={draft_bias_alpha}, "
        f"min={draft_bias_init.min():.2f} max={draft_bias_init.max():.2f} "
        f"mean={draft_bias_init.mean():.2f}"
    )
else:
    draft_bias_init = None

print(
    f"Draft output L2: {draft_output_l2}  Weight decay: {weight_decay}  "
    f"Hist loss weight: {hist_loss_weight}  "
    f"Pack weighting: {pack_weighting} (power={pack_weight_power})  "
    f"Interleave files: {interleave_files}  "
    f"Pick label smoothing: {pick_label_smoothing}"
)

# =============================================================================
# Build model
# =============================================================================

print("Creating / loading model …")
model = CubeCobraMLSystem(
    num_cards,
    draft_bias_init=draft_bias_init,
    draft_output_l2=draft_output_l2,
)

losses = [
    "binary_crossentropy",
    "binary_crossentropy",
    PickHistogramCCE(hist_loss_weight),
    "kl_divergence",
]

# AdamW + clipnorm=1.0 at a constant LR; ReduceLROnPlateau (below) handles
# decay when val_pick_cce stops improving. The plateau callback requires a
# variable learning rate, so no schedule object here.
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_peak,
    weight_decay=weight_decay,
    clipnorm=1.0,
)
optimizer.exclude_from_weight_decay(var_names=["bias"])
print(
    f"Optimizer: AdamW(lr={lr_peak:.0e} + ReduceLROnPlateau("
    f"factor={lr_plateau_factor}, patience={lr_plateau_patience}, "
    f"min_lr={lr_end:.0e}), weight_decay={weight_decay}, "
    f"clipnorm=1.0, exclude=['bias'])"
)

# Collapse cohorts by in-pack count over the eval pass.
_BAND_INF = 2**30
_collapse_metrics = [
    ArgmaxCollapseCount(num_cards, 1, collapse_rare_max + 1, "zero", "collapse0_rare"),
    ArgmaxCollapseCount(num_cards, collapse_rare_max + 1, _BAND_INF, "zero", "collapse0_common"),
    ArgmaxCollapseCount(num_cards, 1, collapse_rare_max + 1, "saturated", "sat100_rare"),
    ArgmaxCollapseCount(num_cards, collapse_rare_max + 1, _BAND_INF, "saturated", "sat100_common"),
]

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=[1.0] * 4,
    metrics=[
        # Per-head BCE/CCE/KL metrics duplicate the loss tensor but get their
        # own keys in the logs dict, side-stepping Keras 3's name collision
        # when two outputs share a loss type.
        [TopRatedPercent("trp_cube"), BinaryCrossentropy(name="cube_bce")],
        [BinaryCrossentropy(name="deck_bce")],
        [
            TopKCategoricalAccuracy(k=1, name="top1"),
            TopKCategoricalAccuracy(k=3, name="top3"),
            # bomb_agreement: argmax==target restricted to top-N most-picked
            # cards — directly catches the basin-flip failure mode that
            # cohort averaging cannot see.
            BombAgreement(bomb_mask, name="bomb_agreement"),
            # pick_cce: plain unweighted CCE — the BEST_METRIC selector.
            # The pick LOSS adds the histogram term; this stays comparable
            # across runs with different hist weights.
            CategoricalCrossentropy(name="pick_cce"),
            PickHistDivergence(name="pick_hist_div"),
            *_collapse_metrics,
        ],
        [KLDivergence(name="corr_kl")],
    ],
)

if continue_flag and os.path.isdir(MODEL_DIR):
    model.load_weights(MODEL_DIR)
    print("Weights restored.")

# =============================================================================
# Load fixed val subset for stability diagnostics (argmax_churn)
# =============================================================================




stability_subset = None
if val_dataset is not None:
    stability_subset = load_stability_subset(TEST_DIR / "picks", num_cards, stability_subset_size)
    if stability_subset is not None:
        print(f"Stability subset: {len(stability_subset['targets'])} fixed val picks for argmax_churn diagnostics.")
    else:
        print("[info] No val picks for stability subset; argmax_churn disabled.")


# =============================================================================
# Snapshot config + callback
# =============================================================================
#
# LOG_STEPS > 0: snapshot every LOG_STEPS training batches.
# LOG_STEPS == 0: snapshot at the end of each epoch.
#
# BEST_METRIC defaults to val_pick_cce — the smoothest trustworthy signal
# for the pick head. BEST_MODE defaults via the metric name ("min" here).

log_steps = max(0, int(os.environ.get("LOG_STEPS", "0")))
keep_best_ckpts = int(os.environ.get("KEEP_BEST_N_CKPTS", "10"))
best_metric = os.environ.get("BEST_METRIC", "").strip() or ("val_pick_cce" if val_dataset is not None else "loss")
best_mode = os.environ.get("BEST_MODE", "").strip().lower() or (
    "max" if best_metric in ("val_bomb_agreement", "val_top1", "val_top3") else "min"
)
if best_mode not in ("min", "max"):
    raise ValueError(f"BEST_MODE must be 'min' or 'max', got {best_mode!r}")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


run_id = f"{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%SZ')}-{_git_sha()}"
run_dir = Path("runs") / run_id
ckpts_dir = run_dir / "ckpts"
ckpts_dir.mkdir(parents=True, exist_ok=True)

tracked_env_vars = (
    "CPU_ONLY",
    "QUICK",
    "DRAFT_BIAS_ALPHA",
    "DRAFT_OUTPUT_L2",
    "WEIGHT_DECAY",
    "PICK_LABEL_SMOOTHING",
    "HIST_LOSS_WEIGHT",
    "PACK_WEIGHTING",
    "PACK_WEIGHT_POWER",
    "INTERLEAVE_FILES",
    "COLLAPSE_RARE_MAX",
    "LR_PEAK",
    "LR_END",
    "LR_PLATEAU_PATIENCE",
    "LR_PLATEAU_FACTOR",
    "STABILITY_SUBSET_SIZE",
    "LOG_STEPS",
    "KEEP_BEST_N_CKPTS",
    "BEST_METRIC",
    "BEST_MODE",
    "SHUFFLE_BUFFER",
)
meta = {
    "run_id": run_id,
    "start_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "git_sha": _git_sha(),
    "cli": {
        "epochs": epochs,
        "batch_size": batch_size,
        "continue": continue_flag,
        "primary": primary_stream,
    },
    "env": {k: os.environ[k] for k in tracked_env_vars if k in os.environ},
    "dataset": {
        "num_cards": int(num_cards),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_final": int(epochs_final),
        "bomb_count": int(bomb_mask.sum()),
        "stability_subset_size": int(len(stability_subset["targets"])) if stability_subset else 0,
    },
    "ckpt": {
        "log_steps": log_steps,
        "keep_best_n": keep_best_ckpts,
        "best_metric": best_metric,
        "best_mode": best_mode,
    },
}
with open(run_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"Run dir: {run_dir}  "
    f"(LOG_STEPS={log_steps}, KEEP_BEST={keep_best_ckpts}, "
    f"BEST_METRIC={best_metric}, BEST_MODE={best_mode})"
)




snapshot_cb = PeriodicSnapshot(
    run_dir=run_dir,
    log_every=log_steps,
    keep_best=keep_best_ckpts,
    best_metric=best_metric,
    best_mode=best_mode,
    stability_subset=stability_subset,
    batch_size=batch_size,
    # All four collapse instances accumulate identical vectors; any one
    # serves as the sidecar source.
    collapse_tracker=_collapse_metrics[0],
    has_val=val_dataset is not None,
)


callbacks = [snapshot_cb]
if val_dataset is not None:
    callbacks.append(ValidationProgress(total_steps=val_steps_per_epoch))
    # Plateau-triggered LR decay — the convergence lever. verbose=1 logs
    # every LR cut to stdout so they're visible in train.log.
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pick_cce",
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=lr_end,
            verbose=1,
        )
    )

fit_kwargs = dict(
    epochs=epochs_final,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
)
if val_dataset is not None:
    fit_kwargs["validation_data"] = val_dataset
    fit_kwargs["validation_steps"] = val_steps_per_epoch

try:
    model.fit(dataset, **fit_kwargs)
finally:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_weights(MODEL_DIR)
    print(f"Saved weights to {MODEL_DIR}")
    print(f"Run artifacts: {run_dir}")
