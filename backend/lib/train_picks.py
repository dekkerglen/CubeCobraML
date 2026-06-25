"""Lazy loader for train pick records.

Train picks are stored in `data/train/picks/{NNNN}.json` batched files (~10K
picks per batch, ~30 MB parsed each). 135M picks total ≈ 60 GB if loaded
fully — that's why val_data.picks works (329K picks ≈ 150 MB) and the train
path can't follow the same pattern.

This module exposes `get_pick_record(global_idx)` which loads the right batch
on demand and returns one pick. `_load_batch` is `@lru_cache`d so popular
batches stay hot.

Cache sizing: 16 × ~30 MB = ~480 MB ceiling — large but bounded; suitable for
the dashboard's interactive query patterns where the user usually drills into
a handful of drafts at a time.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache

from backend.lib.config import TRAIN_DIR


# The pipeline writes batches of WRITE_BATCH_SIZE picks (10000 — see
# scripts/process_data.js). Read the first batch at first call to confirm and
# cache. Saves us hard-coding a number that could drift if the pipeline ever
# changes its WRITE_BATCH_SIZE.
_BATCH_SIZE: int | None = None
_BATCH_SIZE_LOCK = threading.Lock()


def _infer_batch_size() -> int:
    global _BATCH_SIZE
    if _BATCH_SIZE is not None:
        return _BATCH_SIZE
    with _BATCH_SIZE_LOCK:
        if _BATCH_SIZE is not None:
            return _BATCH_SIZE
        first = TRAIN_DIR / "picks" / "0000.json"
        if not first.is_file():
            # No train picks on disk — return the documented default so the
            # caller can fail gracefully on the file open downstream.
            _BATCH_SIZE = 10000
        else:
            _BATCH_SIZE = len(json.loads(first.read_text()))
        return _BATCH_SIZE


@lru_cache(maxsize=16)
def _load_batch(batch_idx: int) -> list[dict]:
    path = TRAIN_DIR / "picks" / f"{batch_idx:04d}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())


_TOTAL: int | None = None
_TOTAL_LOCK = threading.Lock()


def estimated_total() -> int:
    """Total number of train pick records on disk.

    Counts batch files × WRITE_BATCH_SIZE and reads only the final batch to
    pick up its (possibly partial) length. Cached for the lifetime of the
    process. Returns 0 if no train picks are present.
    """
    global _TOTAL
    if _TOTAL is not None:
        return _TOTAL
    with _TOTAL_LOCK:
        if _TOTAL is not None:
            return _TOTAL
        picks_dir = TRAIN_DIR / "picks"
        if not picks_dir.is_dir():
            _TOTAL = 0
            return _TOTAL
        files = sorted(p for p in picks_dir.iterdir() if p.suffix == ".json")
        if not files:
            _TOTAL = 0
            return _TOTAL
        batch = _infer_batch_size()
        # All batches but the last are guaranteed full; the last one may be partial.
        full_batches = len(files) - 1
        last = json.loads(files[-1].read_text())
        _TOTAL = full_batches * batch + len(last)
        return _TOTAL


def get_pick_record(global_idx: int) -> dict | None:
    """Return the pick record at the given global train index, or None if
    the batch file is missing / index is out of range. Record shape mirrors
    val_data.picks entries: `{pool, pack, pick, cube_cards_idx}`."""
    batch_size = _infer_batch_size()
    batch_idx = global_idx // batch_size
    within = global_idx % batch_size
    batch = _load_batch(batch_idx)
    if within >= len(batch):
        return None
    return batch[within]


def get_pick_records(global_idxs: list[int]) -> list[dict]:
    """Bulk variant — orders results to match the input. Same-batch indices
    coalesce into a single batch read, so a 42-pick draft is one disk read."""
    out: list[dict | None] = [None] * len(global_idxs)
    # Group by batch_idx to amortize file reads.
    batch_size = _infer_batch_size()
    by_batch: dict[int, list[tuple[int, int]]] = {}
    for i, gi in enumerate(global_idxs):
        bi = gi // batch_size
        by_batch.setdefault(bi, []).append((i, gi % batch_size))
    for bi, positions in by_batch.items():
        batch = _load_batch(bi)
        for out_i, pos in positions:
            if pos < len(batch):
                out[out_i] = batch[pos]
    return [r for r in out if r is not None]
