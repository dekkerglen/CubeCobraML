"""Per-card model vs human pick rate from val (and optionally train) picks.

Two access patterns:

  - **Synchronous** `GET /api/collapse?...`. Small `sample_n` (≤ a few thousand)
    is the use case — per-cube Collapse panels on the cube workspace. Cached
    per `(ckpt, sample_n, cube_filter_key)` via lru_cache for instant repeats.
  - **Job** via `POST /api/jobs/collapse` (Phase N + Q). Big runs — full val
    (329K picks) ± a train sample. Emits a partial CollapseOut every ~10% so
    the frontend can update the bucket histogram live, supports cancellation,
    and survives the synchronous request timeout window.

Both paths funnel through `_compute_on_records` so they stay in lockstep.
"""

from __future__ import annotations

import random
import threading
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.lib import cards as cards_lib
from backend.lib import train_picks
from backend.lib.inference import draft_logits_batch, one_hot
from backend.services.cube_directory import cube_directory
from backend.services.draft_sessions import draft_sessions
from backend.services.jobs import JobHandle
from backend.services.model_registry import registry
from backend.services.val_data import val_data
from backend.routers import jobs as jobs_router


router = APIRouter(prefix="/collapse", tags=["collapse"])


BATCH = 32


_lock = threading.Lock()


Source = Literal["all", "val", "train"]


class CollapseOut(BaseModel):
    ckpt: str
    sample_n: int
    sample_n_val: int = 0
    sample_n_train: int = 0
    appearances: list[int]
    model_picks: list[int]
    human_picks: list[int]
    # Per-pick chosen card-idx lists (length = sample_n). Lets the frontend
    # build attribute (color/CMC/type) distribution charts off the same job
    # without a second inference run.
    human_pick_idxs: list[int] = []
    model_pick_idxs: list[int] = []


# ----- core compute ---------------------------------------------------------


def _compute_on_records(
    ckpt: str,
    records: list[dict],
    handle: JobHandle | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Inference loop over a pre-assembled list of pick records. Each record
    must have `pool`, `pack`, and `pick`.

    Returns `(appearances, model_picks_count, human_picks_count,
    human_pick_idxs, model_pick_idxs)`. The two idx-lists are per-record
    (length = len(records) scored before cancellation) and let the caller
    build attribute distributions off the same job result.

    If `handle` is set, emits a partial CollapseOut every ~10% of records and
    bails on `handle.cancel.is_set()` at every batch boundary.
    """
    obj = registry.get(ckpt)
    n_cards = cards_lib.num_cards()
    appearances = np.zeros(n_cards, dtype=np.int64)
    model_picks = np.zeros(n_cards, dtype=np.int64)
    human_picks = np.zeros(n_cards, dtype=np.int64)
    human_pick_idxs: list[int] = []
    model_pick_idxs: list[int] = []

    n = len(records)
    if n == 0:
        return appearances, model_picks, human_picks, human_pick_idxs, model_pick_idxs
    # Aim for ~20 progress reports per run regardless of size, but never less
    # often than every batch (small cubes need updates *more* often, not less).
    report_every = max(BATCH, n // 20)
    next_report = BATCH  # always fire after the very first batch

    for i in range(0, n, BATCH):
        if handle is not None and handle.cancel.is_set():
            break
        batch = records[i : i + BATCH]
        pools = np.stack([one_hot(r["pool"]) for r in batch])
        packs = np.stack([one_hot(r["pack"]) for r in batch])
        probs = draft_logits_batch(obj, pools, packs)
        chosen = probs.argmax(axis=1)
        for j, r in enumerate(batch):
            mi = int(chosen[j])
            hi = int(r["pick"])
            model_picks[mi] += 1
            human_picks[hi] += 1
            model_pick_idxs.append(mi)
            human_pick_idxs.append(hi)
            for c in r["pack"]:
                appearances[c] += 1
        done = i + len(batch)
        if handle is not None and done >= next_report:
            partial = CollapseOut(
                ckpt=ckpt,
                sample_n=done,
                appearances=appearances.tolist(),
                model_picks=model_picks.tolist(),
                human_picks=human_picks.tolist(),
                human_pick_idxs=list(human_pick_idxs),
                model_pick_idxs=list(model_pick_idxs),
            ).model_dump()
            handle.report(progress=done / n, partial=partial)
            next_report = done + report_every
    return appearances, model_picks, human_picks, human_pick_idxs, model_pick_idxs


# ----- record assembly ------------------------------------------------------


def _val_records_filtered(cube_uuids: list[str] | None) -> list[dict]:
    val_data.ensure_loaded()
    if not cube_uuids:
        return list(val_data.picks)
    wanted: set[int] = set()
    for uuid in cube_uuids:
        wanted.update(draft_sessions.pick_indices_by_cube(uuid, split="val"))
    return [val_data.picks[i] for i in sorted(wanted)]


def _train_records_for_cubes(cube_uuids: list[str]) -> list[dict]:
    """Resolve train pick records for the given cubes."""
    records: list[dict] = []
    for uuid in cube_uuids:
        train_idxs = draft_sessions.pick_indices_by_cube(uuid, split="train")
        if not train_idxs:
            continue
        for rec in train_picks.get_pick_records(sorted(train_idxs)):
            records.append(
                {
                    "pool": rec.get("pool") or [],
                    "pack": rec.get("pack") or [],
                    "pick": rec.get("pick", -1),
                }
            )
    return records


def _records_filtered_for_cubes(
    cube_uuids: list[str] | None,
    source: Source,
) -> tuple[list[dict], int, int]:
    """Records for the given cubes (or all if None), filtered by source.

    Returns `(records, n_val, n_train)`. Val records come first to keep cache
    behavior in the unfiltered case bit-identical to the pre-refactor path.
    """
    val_records: list[dict] = []
    train_records: list[dict] = []
    if source in ("val", "all"):
        val_records = _val_records_filtered(cube_uuids)
    if source in ("train", "all") and cube_uuids:
        # Train-side without a cube filter would mean iterating tens of millions
        # of picks. We only support the per-cube case here; cross-cube train
        # sampling stays in `_train_sample_records` (used by the job runner).
        train_records = _train_records_for_cubes(cube_uuids)
    return val_records + train_records, len(val_records), len(train_records)


def _train_sample_records(n_total: int, cube_uuids: list[str] | None, seed: int = 42) -> list[dict]:
    """Cube-fair train sample: per cube with val drafts, take a share of
    train picks proportional to that cube's val pick share. Keeps the train
    distribution comparable to val so they can be aggregated meaningfully.

    `cube_uuids` is the *filter* set (if any); when None we use all val cubes.
    """
    if n_total <= 0:
        return []
    val_data.ensure_loaded()
    cube_directory.ensure_loaded()
    # Build val pick counts per cube — the proportion key.
    pool_uuids: list[str] = (
        cube_uuids if cube_uuids else [e.cube_uuid for e in cube_directory.entries if e.has_val_drafts]
    )
    val_share: dict[str, int] = {}
    total_val = 0
    for u in pool_uuids:
        idxs = draft_sessions.pick_indices_by_cube(u, split="val")
        if idxs:
            val_share[u] = len(idxs)
            total_val += len(idxs)
    if total_val == 0:
        return []

    rng = random.Random(seed)
    picked: list[int] = []
    for uuid, share in val_share.items():
        want = max(1, round(n_total * share / total_val))
        train_idxs = draft_sessions.pick_indices_by_cube(uuid, split="train")
        if not train_idxs:
            continue
        if len(train_idxs) <= want:
            picked.extend(train_idxs)
        else:
            picked.extend(rng.sample(train_idxs, want))

    records = train_picks.get_pick_records(picked)
    return [
        {
            "pool": rec.get("pool") or [],
            "pack": rec.get("pack") or [],
            "pick": rec.get("pick", -1),
        }
        for rec in records
    ]


# ----- sync (lru_cache'd) entry ---------------------------------------------


@lru_cache(maxsize=16)
def _run_sync(
    ckpt: str,
    sample_n: int,
    cube_filter_key: str | None,
    source: Source,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], int, int]:
    """Returns `(appearances, model_picks, human_picks, human_pick_idxs,
    model_pick_idxs, n_val, n_train)`."""
    cube_uuids = cube_filter_key.split(",") if cube_filter_key else None
    records, n_val, n_train = _records_filtered_for_cubes(cube_uuids, source)
    if sample_n and sample_n < len(records):
        records = records[:sample_n]
        # Val records come first, so truncation eats val before train. Adjust
        # the breakdown so it reflects what was actually scored.
        if sample_n <= n_val:
            n_val, n_train = sample_n, 0
        else:
            n_train = sample_n - n_val
    ap, mp, hp, hpi, mpi = _compute_on_records(ckpt, records, handle=None)
    return ap, mp, hp, hpi, mpi, n_val, n_train


@router.get("", response_model=CollapseOut)
def collapse(
    ckpt: str,
    sample_n: int = Query(10_000, ge=500, le=400_000),
    cube_uuids: str | None = Query(
        None, description="Comma-separated cube UUIDs; restrict picks to drafts of these cubes"
    ),
    source: Source = Query(
        "all", description="Pick source: 'all' (train+val), 'val', or 'train'. Train requires cube_uuids."
    ),
) -> CollapseOut:
    if registry.find(ckpt) is None:
        raise HTTPException(404, f"unknown checkpoint: {ckpt!r}")
    key = None
    if cube_uuids and cube_uuids.strip():
        parts = sorted({u.strip() for u in cube_uuids.split(",") if u.strip()})
        if parts:
            key = ",".join(parts)
    if source == "train" and key is None:
        raise HTTPException(400, "source='train' requires cube_uuids — cross-cube train sampling uses the job runner.")
    with _lock:
        ap, mp, hp, hpi, mpi, n_val, n_train = _run_sync(ckpt, sample_n, key, source)
    return CollapseOut(
        ckpt=ckpt,
        sample_n=n_val + n_train,
        sample_n_val=n_val,
        sample_n_train=n_train,
        appearances=ap.tolist(),
        model_picks=mp.tolist(),
        human_picks=hp.tolist(),
        human_pick_idxs=list(hpi),
        model_pick_idxs=list(mpi),
    )


# ----- job runner -----------------------------------------------------------


def _collapse_runner(handle: JobHandle, body: dict) -> dict:
    """Body shape:
      `{ckpt, sample_n?, cube_uuids?: [str], source?: 'all'|'val'|'train',
        include_train_sample?: int}`.

    Per-cube case (`cube_uuids` non-empty): defaults `source='all'` and ignores
    `include_train_sample` — the train side comes in via `source` instead.
    Cross-cube case (`cube_uuids` empty): always val + optional train sample
    (via `_train_sample_records`); `source` is ignored.
    """
    ckpt = body.get("ckpt")
    if not ckpt or registry.find(ckpt) is None:
        raise ValueError(f"unknown or missing checkpoint: {ckpt!r}")
    sample_n = body.get("sample_n")
    raw_uuids = body.get("cube_uuids") or []
    cube_uuids = [u.strip() for u in raw_uuids if u and u.strip()] or None
    include_train_sample = int(body.get("include_train_sample") or 0)
    source: Source = body.get("source") or ("all" if cube_uuids else "val")

    if cube_uuids:
        records, n_val, n_train = _records_filtered_for_cubes(cube_uuids, source)
        if sample_n and sample_n < len(records):
            records = records[:sample_n]
            if sample_n <= n_val:
                n_val, n_train = sample_n, 0
            else:
                n_train = sample_n - n_val
    else:
        val_records = _val_records_filtered(None)
        if sample_n and sample_n < len(val_records):
            val_records = val_records[:sample_n]
        train_records = _train_sample_records(include_train_sample, None)
        records = val_records + train_records
        n_val, n_train = len(val_records), len(train_records)

    handle.report(progress=0.0, partial=None, eta_seconds=None)
    ap, mp, hp, hpi, mpi = _compute_on_records(ckpt, records, handle=handle)
    return CollapseOut(
        ckpt=ckpt,
        sample_n=n_val + n_train,
        sample_n_val=n_val,
        sample_n_train=n_train,
        appearances=ap.tolist(),
        model_picks=mp.tolist(),
        human_picks=hp.tolist(),
        human_pick_idxs=list(hpi),
        model_pick_idxs=list(mpi),
    ).model_dump()


jobs_router.register("collapse", _collapse_runner)
