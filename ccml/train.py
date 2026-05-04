"""
Usage:  python train.py <epochs> <batch_size> <continue_train:true|false> <primary_stream>

Environment knobs:
    USE_CUBE_CONTEXT          true|false   default: true
    LAND_PENALTY_WEIGHT       float        default: 1.0
    CHECKPOINT_EVERY_N_EPOCHS int          default: 50
    LOG_DIR                   path         default: <repo>/model/runs/<timestamp>
"""

import datetime
import json
import os
import sys

# TF 2.16+ ships Keras 3 by default; this codebase relies on Keras-2 semantics
# for Model subclassing + custom save_weights/load_weights. Must be set before
# tensorflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from ccml.data import build_dataset
from ccml.model import CubeCobraMLSystem, make_draft_loss
from ccml.tf_metrics import TopRatedPercent
from ccml.training_callbacks import (
    DivergenceGuard,
    EpochCheckpoint,
    MetricsProbe,
    ProbeCallback,
)
from ccml.utils import DATA_DIR, MODEL_DIR

DATA_DIR = DATA_DIR / "train"
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
continue_flag = sys.argv[3].lower() == "true"
primary_stream = sys.argv[4].lower()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


use_cube_context = _env_bool("USE_CUBE_CONTEXT", True)
land_penalty_weight = float(os.environ.get("LAND_PENALTY_WEIGHT", "1.0"))
checkpoint_every_n = int(os.environ.get("CHECKPOINT_EVERY_N_EPOCHS", "50"))

# GPU sanity print
gpus = tf.config.list_physical_devices('GPU')
print("\n" + "=" * 70)
if gpus:
    print("🚀 GPU DETECTED AND ENABLED FOR TRAINING 🚀")
    for gpu in gpus:
        print(f"   Using: {gpu.name}")
    print("=" * 70 + "\n")
else:
    print("⚠️  WARNING: NO GPU FOUND - TRAINING WILL USE CPU ONLY")
    print("   This will be significantly slower than GPU training.")
    print("=" * 70 + "\n")

print(
    f"Config: use_cube_context={use_cube_context}, "
    f"land_penalty_weight={land_penalty_weight}, "
    f"checkpoint_every_n_epochs={checkpoint_every_n}"
)

dataset, num_cards, steps_per_epoch, epochs_final, is_land_mask = build_dataset(
    cubes_path=os.path.join(DATA_DIR, "cubes"),
    decks_path=os.path.join(DATA_DIR, "decks"),
    picks_path=os.path.join(DATA_DIR, "picks"),
    cube_instances_path=os.path.join(DATA_DIR, "cubeInstances"),
    freq_path=os.path.join(DATA_DIR, "oracleFrequency.json"),
    correlations_path=os.path.join(DATA_DIR, "correlations.json"),
    batch_size=batch_size,
    target_epochs=epochs,
    primary=primary_stream,
    use_cube_context=use_cube_context,
)

print("Creating / loading model …")
model = CubeCobraMLSystem(
    num_cards,
    is_land_mask=is_land_mask,
    use_cube_context=use_cube_context,
)

losses = [
    "binary_crossentropy",
    "binary_crossentropy",
    make_draft_loss(is_land_mask, land_penalty_weight),
    "kullback_leibler_divergence",
]

model.compile(
    optimizer="adam",
    loss=losses,
    loss_weights=[1.0] * 4,
    metrics={
        "output_1": TopRatedPercent("trp"),
        "output_2": TopRatedPercent("trp"),
        "output_3": [
            TopKCategoricalAccuracy(k=1, name="top1"),
            TopKCategoricalAccuracy(k=3, name="top3"),
        ],
        "output_4": TopRatedPercent("trp"),
    },
)

if continue_flag and os.path.isdir(MODEL_DIR):
    model.load_weights(MODEL_DIR)
    print("Weights restored.")


# Build the run directory: per-run timestamped dir under model/runs/, plus
# tensorboard logs and checkpoints both nested inside it. This keeps every
# run's artifacts grouped together.
run_id = os.environ.get("RUN_ID") or datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
run_dir = os.environ.get("LOG_DIR") or os.path.join(str(MODEL_DIR), "runs", run_id)
tb_log_dir = os.path.join(run_dir, "tensorboard")
ckpt_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(run_dir, exist_ok=True)

run_meta = {
    "run_id": run_id,
    "epochs_requested": epochs,
    "epochs_final": epochs_final,
    "steps_per_epoch": steps_per_epoch,
    "batch_size": batch_size,
    "primary_stream": primary_stream,
    "use_cube_context": use_cube_context,
    "land_penalty_weight": land_penalty_weight,
    "checkpoint_every_n_epochs": checkpoint_every_n,
    "started_at": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(os.path.join(run_dir, "config.json"), "w") as fh:
    json.dump(run_meta, fh, indent=2)
print(f"Run dir: {run_dir}")
print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")


# Custom probes (e.g. embedding-distance assertions) plug in here. None today;
# add functions of signature `(model, logs) -> dict[str, float]` to the list
# and they'll be logged alongside the standard metrics each epoch.
probes: list[MetricsProbe] = []

# Optional epoch-over-epoch divergence guard. Populate with metric → max-jump
# pairs (e.g. {"loss": 0.5}) to get warning prints when a metric leaps.
divergence_thresholds: dict[str, float] = {}

callbacks = [
    # update_freq="epoch" smooths within-epoch volatility (Keras already
    # accumulates batch-level stats into the epoch summary). Set to a positive
    # int (e.g. 100) for raw batch-level curves at the cost of larger logs.
    tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir,
        update_freq="epoch",
        histogram_freq=0,
        write_graph=False,
        profile_batch=0,  # disables the profiler — has measurable per-step overhead
    ),
    EpochCheckpoint(ckpt_dir, every_n_epochs=checkpoint_every_n, run_meta=run_meta),
    ProbeCallback(probes, tb_log_dir=tb_log_dir),
    DivergenceGuard(divergence_thresholds),
]


# fit the model and save it even if the process gets disrupted
try:
    model.fit(
        dataset,
        epochs=epochs_final,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
finally:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_weights(MODEL_DIR)
    print(f"Saved final weights to {MODEL_DIR}")
