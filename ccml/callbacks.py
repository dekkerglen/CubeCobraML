"""Training callbacks.

Two pieces, both pure run-instrumentation — no effect on the optimization:

- ``ValidationProgress``: a tqdm bar for the per-epoch validation pass.
- ``PeriodicSnapshot``: writes each run's artifacts — weight checkpoints
  (keeping the best N by the selector metric), ``metrics.csv``, the per-card
  collapse sidecars, and the best-N index — periodically during training.

``load_stability_subset`` materializes the fixed val slice ``PeriodicSnapshot``
uses for its snapshot-time ``argmax_churn`` / ``val_subset_top1`` diagnostics.
"""

import csv
import datetime
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def _one_hot(indices: list[int], n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if indices:
        v[indices] = 1.0
    return v


def load_stability_subset(picks_path: Path, num_cards: int, max_count: int):
    """Materialize a fixed slice of val picks for snapshot-time diagnostics."""
    if not picks_path.is_dir():
        return None
    files = sorted(picks_path.iterdir())
    pools, packs, targets = [], [], []
    count = 0
    for fp in files:
        if count >= max_count:
            break
        for rec in json.load(open(fp)):
            if count >= max_count:
                break
            pools.append(_one_hot(rec["pool"], num_cards))
            packs.append(_one_hot(rec["pack"], num_cards))
            targets.append(int(rec["pick"]))
            count += 1
    if count == 0:
        return None
    return {
        "pools": np.stack(pools),
        "packs": np.stack(packs),
        "targets": np.array(targets, dtype=np.int64),
    }


class ValidationProgress(tf.keras.callbacks.Callback):
    """Show a tqdm bar for the per-epoch validation pass.

    Keras's default progbar emits one carriage-return-overwritten line and
    interleaves badly with train progress on slow per-step val. tqdm renders
    crisp progress with ETA and step time, which makes it obvious whether
    the val pass is making progress vs. hung.
    """

    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = total_steps
        self._bar = None

    def on_test_begin(self, logs=None):
        self._bar = tqdm(total=self.total_steps, desc="val", unit="batch", leave=True)

    def on_test_batch_end(self, batch, logs=None):
        if self._bar is not None:
            self._bar.update(1)

    def on_test_end(self, logs=None):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class PeriodicSnapshot(tf.keras.callbacks.Callback):
    """Snapshot metrics row + weight checkpoint periodically; retain top-N.

    Adds diagnostics computed on a fixed val subset, NOT the Keras val
    pipeline (so they're stable across snapshots):
      - argmax_churn: fraction of examples whose argmax flipped since the
        previous snapshot. Catches basin-flips directly; the convergence
        gauge (should fall toward the ~5-10% ambiguity floor as LR anneals).
      - val_subset_top1: top1 on the fixed subset.
    """

    def __init__(
        self,
        run_dir: Path,
        log_every: int,
        keep_best: int,
        best_metric: str,
        best_mode: str,
        stability_subset: dict | None,
        batch_size: int,
        collapse_tracker=None,
        has_val: bool = False,
    ):
        super().__init__()
        self.run_dir = run_dir
        self.collapse_tracker = collapse_tracker
        self.has_val = has_val
        self.ckpts_dir = run_dir / "ckpts"
        self.csv_path = run_dir / "metrics.csv"
        self.log_every = log_every
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.global_step = 0
        self.epoch = 0
        self.last_batch_idx = 0
        self.ranked: list[dict] = []
        self.index_path = self.ckpts_dir / "best_index.json"
        self.best_link = self.ckpts_dir / "best"
        self._warned_missing = False
        self._fh = None
        self._writer = None
        self._fieldnames: list[str] | None = None
        self.stability_subset = stability_subset
        self.batch_size = batch_size
        self.last_argmax = None

    # --- Keras hooks -----------------------------------------------------
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        self.last_batch_idx = batch
        if self.log_every <= 0:
            return
        if self.global_step % self.log_every != 0:
            return
        self._snapshot(batch + 1, logs or {})

    def on_epoch_end(self, epoch, logs=None):
        self._save_collapse_sidecar()
        if self.log_every > 0:
            return
        self._snapshot(self.last_batch_idx + 1, logs or {})

    def on_train_end(self, logs=None):
        if self._fh is not None:
            self._fh.close()

    # --- Collapse sidecar -------------------------------------------------
    def _save_collapse_sidecar(self) -> None:
        """Dump the per-card collapse accumulators (offers / model-argmax /
        human-pick over the full pass) for train AND val to one npz per
        epoch. Raw counts only — model-only collapse, cumulative train pick
        rates, and val flip logs are all offline derivations from these.
        """
        if self.collapse_tracker is None:
            return
        current = self.collapse_tracker.current_vectors()
        stashed = self.collapse_tracker.last_pass_vectors()
        if self.has_val:
            # Metrics were reset at val start, stashing the train vectors;
            # the variables now hold the val pass.
            train_vecs, val_vecs = stashed, current
        else:
            train_vecs, val_vecs = current, None

        arrays = {}
        if train_vecs is not None:
            arrays.update({f"train_{k}": v for k, v in train_vecs.items()})
        if val_vecs is not None:
            arrays.update({f"val_{k}": v for k, v in val_vecs.items()})
        if not arrays:
            return
        out_dir = self.run_dir / "collapse"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"epoch_{self.epoch:03d}.npz", **arrays)

    # --- Stability diagnostics ------------------------------------------
    def _compute_stability_extras(self) -> dict:
        """Run inference on the fixed val subset, return
        {argmax_churn, val_subset_top1}."""
        if self.stability_subset is None:
            return {}
        pools = self.stability_subset["pools"]
        packs = self.stability_subset["packs"]
        targets = self.stability_subset["targets"]

        bs = self.batch_size
        argmaxes = []
        for i in range(0, len(targets), bs):
            j = min(i + bs, len(targets))
            preds = self.model.draft(pools[i:j], packs[i:j], training=False).numpy()
            argmaxes.append(preds.argmax(axis=-1))
        argmaxes = np.concatenate(argmaxes)

        top1 = float((argmaxes == targets).mean())

        if self.last_argmax is not None:
            argmax_churn = float((argmaxes != self.last_argmax).mean())
        else:
            argmax_churn = 0.0
        self.last_argmax = argmaxes

        return {
            "argmax_churn": argmax_churn,
            "val_subset_top1": top1,
        }

    # --- Snapshot pipeline ----------------------------------------------
    def _snapshot(self, step_in_epoch: int, logs: dict) -> None:
        wall = datetime.datetime.utcnow().isoformat() + "Z"
        extras = self._compute_stability_extras()
        combined_logs = {**logs, **extras}
        self._write_row(step_in_epoch, combined_logs, wall)
        # Always overwrite latest/ regardless of metric — this is the canonical
        # "most recent" checkpoint and is independent of the best-N ranking. It
        # lets you (a) resume from latest after a crash, (b) compare current vs
        # best in analysis tools without losing recent state to keep-best
        # eviction. Distinct from runs/<id>/ckpts/best which tracks best metric.
        self._save_latest()
        metric_val = self._extract_metric(combined_logs)
        if metric_val is None:
            return
        self._maybe_save_ckpt(metric_val, wall)

    def _save_latest(self) -> None:
        latest_dir = self.ckpts_dir / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(latest_dir))
        # Sidecar metadata so diff tools know which epoch the latest is from.
        (latest_dir / "_meta.json").write_text(
            json.dumps(
                {
                    "epoch": self.epoch,
                    "step": self.global_step,
                    "wall_utc": datetime.datetime.utcnow().isoformat() + "Z",
                }
            )
        )

    def _extract_metric(self, logs: dict) -> float | None:
        v = logs.get(self.best_metric)
        if v is None:
            if not self._warned_missing:
                print(
                    f"[snapshot] BEST_METRIC={self.best_metric!r} not in logs "
                    f"({sorted(logs.keys())[:6]}...). Skipping checkpoint until it appears."
                )
                self._warned_missing = True
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    def _maybe_save_ckpt(self, metric_val: float, wall: str) -> None:
        if self.keep_best <= 0:
            self._save_ckpt(metric_val, wall)
            return
        if len(self.ranked) < self.keep_best:
            self._save_ckpt(metric_val, wall)
            return
        worst = self.ranked[-1]["metric"]
        if self._is_better(metric_val, worst):
            self._save_ckpt(metric_val, wall)
            evicted = self.ranked.pop()
            self._delete_ckpt(evicted["dirname"])
            self._sort_ranked()
            self._persist_index()
            self._update_best_link()

    def _is_better(self, a: float, b: float) -> bool:
        return a < b if self.best_mode == "min" else a > b

    def _save_ckpt(self, metric_val: float, wall: str) -> None:
        dirname = f"step_{self.global_step:09d}_epoch_{self.epoch:03d}"
        ckpt_path = self.ckpts_dir / dirname
        self.model.save_weights(str(ckpt_path))
        self.ranked.append(
            {
                "metric": metric_val,
                "step": self.global_step,
                "epoch": self.epoch,
                "dirname": dirname,
                "wall_utc": wall,
            }
        )
        self._sort_ranked()
        self._persist_index()
        self._update_best_link()

    def _delete_ckpt(self, dirname: str) -> None:
        shutil.rmtree(self.ckpts_dir / dirname, ignore_errors=True)

    def _sort_ranked(self) -> None:
        reverse = self.best_mode == "max"
        self.ranked.sort(key=lambda e: e["metric"], reverse=reverse)

    def _persist_index(self) -> None:
        payload = {
            "best_metric": self.best_metric,
            "best_mode": self.best_mode,
            "ranked": self.ranked,
        }
        with open(self.index_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _update_best_link(self) -> None:
        if not self.ranked:
            return
        target = self.ranked[0]["dirname"]
        if self.best_link.is_symlink() or self.best_link.exists():
            self.best_link.unlink()
        os.symlink(target, self.best_link)

    # --- CSV ------------------------------------------------------------
    # Keras duplicates each compiled loss as a metric column named after the
    # loss class; the named per-head metrics (cube_bce, deck_bce, pick_cce,
    # corr_kl) already cover these, so the duplicates are dropped.
    _CSV_DROP = frozenset(
        name
        for base in (
            "binary_crossentropy_loss",
            "categorical_crossentropy_loss",
            "kl_divergence_loss",
        )
        for name in (base, f"val_{base}")
    )

    def _write_row(self, step_in_epoch: int, logs: dict, wall_utc: str) -> None:
        row = {
            "run_id": self.run_dir.name,
            "epoch": self.epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": self.global_step,
            "wall_utc": wall_utc,
            # Read LR off the optimizer directly: ReduceLROnPlateau only puts
            # `learning_rate` into logs AFTER this callback runs (list order),
            # so relying on logs would lock the CSV header without it.
            "learning_rate": float(self.model.optimizer.learning_rate),
        }
        for k, v in logs.items():
            if k in self._CSV_DROP:
                continue
            row[k] = float(v) if v is not None else ""
        if self._writer is None:
            self._fh = open(self.csv_path, "w", newline="")
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._fh.flush()
