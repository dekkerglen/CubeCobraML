"""Training callbacks: epoch checkpointing, metric persistence, custom probes.

Kept in one file because they all hang off the same `model.fit` callback list
and share a small amount of state (the run directory).
"""

import datetime
import json
import os
from typing import Callable, Iterable

import tensorflow as tf


class EpochCheckpoint(tf.keras.callbacks.Callback):
    """Save full sub-model weights every N epochs into a versioned directory.

    Plays nicely with `CubeCobraMLSystem.save_weights`, which writes one
    sub-model per directory (encoder + 4 decoders). Each checkpoint also gets
    a `meta.json` next to it so we can tell runs apart later.
    """

    def __init__(self, base_dir: str, every_n_epochs: int, run_meta: dict):
        super().__init__()
        self.base_dir = base_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.run_meta = dict(run_meta)
        os.makedirs(self.base_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-indexed; humanize by +1.
        e = epoch + 1
        if e % self.every_n_epochs != 0:
            return
        ckpt_dir = os.path.join(self.base_dir, f"epoch_{e:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[checkpoint] saving epoch {e} → {ckpt_dir}")
        # Lightweight weights-only save; ~MB instead of ~GB of full SavedModel.
        self.model.save_checkpoint(ckpt_dir)
        meta = dict(self.run_meta)
        meta.update({
            "epoch": e,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "logs": {k: float(v) for k, v in (logs or {}).items()},
        })
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)


# A probe is a function that, given the model and the current epoch logs,
# returns a dict of scalar metrics to log alongside the rest. Keeping the
# interface as a plain callable means future probes (e.g. cosine similarity
# between two card embeddings, or rank-of-card-X-in-pack-Y) can be one
# function each — no class hierarchy needed.
MetricsProbe = Callable[[tf.keras.Model, dict], dict[str, float]]


class ProbeCallback(tf.keras.callbacks.Callback):
    """Runs each registered probe at the end of every epoch and writes the
    results into the TensorBoard summary for that epoch.

    Probes are also surfaced on stdout for human-readable training logs.
    """

    def __init__(self, probes: Iterable[MetricsProbe], tb_log_dir: str):
        super().__init__()
        self.probes = list(probes)
        self._writer = tf.summary.create_file_writer(os.path.join(tb_log_dir, "probes")) if self.probes else None

    def on_epoch_end(self, epoch, logs=None):
        if not self.probes or self._writer is None:
            return
        merged: dict[str, float] = {}
        for probe in self.probes:
            try:
                out = probe(self.model, dict(logs or {}))
            except Exception as exc:  # noqa: BLE001 — probes must never crash training
                print(f"[probe-error] {getattr(probe, '__name__', repr(probe))}: {exc}")
                continue
            for k, v in (out or {}).items():
                merged[k] = float(v)
        if not merged:
            return
        print("[probes] " + ", ".join(f"{k}={v:.4f}" for k, v in merged.items()))
        with self._writer.as_default():
            for k, v in merged.items():
                tf.summary.scalar(f"probe/{k}", v, step=epoch)
        self._writer.flush()


# Optional simple "did this metric jump too far in one epoch" guard. Empty
# config = no-op. Wired in train.py so future runs can populate it without
# touching the callback code.
class DivergenceGuard(tf.keras.callbacks.Callback):
    def __init__(self, thresholds: dict[str, float]):
        super().__init__()
        self.thresholds = dict(thresholds)
        self._prev: dict[str, float] = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, max_jump in self.thresholds.items():
            if k not in logs:
                continue
            cur = float(logs[k])
            prev = self._prev.get(k)
            if prev is not None and abs(cur - prev) > max_jump:
                print(f"[warn] {k} jumped {prev:.4f} → {cur:.4f} (>|{max_jump}|) at epoch {epoch + 1}")
            self._prev[k] = cur
