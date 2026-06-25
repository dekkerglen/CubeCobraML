"""Unified cube metrics — one inference pass produces headline KPIs,
attribute pick lists, and per-card collapse counts for a single cube.

Replaces the older scorecard runner. Frontend reads from one job result
across Headline / Distribution / Collapse subtabs of the cube workspace,
so switching subtabs never re-fires inference.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from backend.lib import cards as cards_lib
from backend.lib.inference import draft_logits_batch, one_hot
from backend.routers import jobs as jobs_router
from backend.routers.collapse import _records_filtered_for_cubes
from backend.services.jobs import JobHandle
from backend.services.model_registry import registry


# Smaller than collapse.py's BATCH — cube samples are typically small (dozens
# of picks), so a tiny batch is the only way the user sees the progress bar
# move at all before completion.
BATCH = 16


Source = Literal["all", "val", "train"]


class CubeMetricsOut(BaseModel):
    ckpt: str
    cube_uuid: str
    source: Source
    n_picks: int
    n_val: int
    n_train: int
    # headline
    top1: float
    top3: float
    avg_top1_p: float
    avg_human_p: float
    # distribution (per-pick chosen card)
    human_pick_idxs: list[int]
    model_pick_idxs: list[int]
    # collapse (per-card counts over the cube vocabulary)
    appearances: list[int]
    model_picks_count: list[int]
    human_picks_count: list[int]


def _cube_metrics_runner(handle: JobHandle, body: dict) -> dict:
    ckpt = body.get("ckpt")
    cube_uuid = body.get("cube_uuid")
    source: Source = body.get("source") or "all"
    if source not in ("all", "val", "train"):
        raise ValueError(f"bad source: {source!r}")
    if not ckpt or registry.find(ckpt) is None:
        raise ValueError(f"unknown or missing checkpoint: {ckpt!r}")
    if not cube_uuid:
        raise ValueError("missing cube_uuid")

    records, n_val, n_train = _records_filtered_for_cubes([cube_uuid], source)
    n = len(records)
    n_cards = cards_lib.num_cards()

    appearances = np.zeros(n_cards, dtype=np.int64)
    model_count = np.zeros(n_cards, dtype=np.int64)
    human_count = np.zeros(n_cards, dtype=np.int64)
    human_pick_idxs: list[int] = []
    model_pick_idxs: list[int] = []

    def emit(top1_hits: int, top3_hits: int, sum_top1_p: float, sum_human_p: float, done: int, final: bool) -> dict:
        denom = max(1, done)
        return CubeMetricsOut(
            ckpt=ckpt,
            cube_uuid=cube_uuid,
            source=source,
            n_picks=done,
            n_val=min(done, n_val),
            n_train=max(0, done - n_val),
            top1=top1_hits / denom,
            top3=top3_hits / denom,
            avg_top1_p=sum_top1_p / denom,
            avg_human_p=sum_human_p / denom,
            human_pick_idxs=list(human_pick_idxs),
            model_pick_idxs=list(model_pick_idxs),
            appearances=appearances.tolist(),
            model_picks_count=model_count.tolist(),
            human_picks_count=human_count.tolist(),
        ).model_dump()

    if n == 0:
        return emit(0, 0, 0.0, 0.0, 0, final=True)

    obj = registry.get(ckpt)
    top1_hits = 0
    top3_hits = 0
    sum_top1_p = 0.0
    sum_human_p = 0.0
    # Aim for ~20 reports total. For tiny cubes (n < 20·BATCH) this fires
    # every batch — exactly what the user wants when the whole run takes <1s.
    report_every = max(BATCH, n // 20)
    next_report = BATCH  # always fire after the very first batch

    for i in range(0, n, BATCH):
        if handle.cancel.is_set():
            break
        batch = records[i : i + BATCH]
        pools = np.stack([one_hot(r["pool"]) for r in batch])
        packs = np.stack([one_hot(r["pack"]) for r in batch])
        probs = draft_logits_batch(obj, pools, packs)
        for j, r in enumerate(batch):
            row = probs[j]
            human = int(r["pick"])
            argmax = int(row.argmax())
            top3 = np.argpartition(-row, 3)[:3]
            human_pick_idxs.append(human)
            model_pick_idxs.append(argmax)
            human_count[human] += 1
            model_count[argmax] += 1
            for c in r["pack"]:
                appearances[c] += 1
            if argmax == human:
                top1_hits += 1
            if human in top3:
                top3_hits += 1
            sum_top1_p += float(row[argmax])
            sum_human_p += float(row[human]) if 0 <= human < row.shape[0] else 0.0
        done = i + len(batch)
        if done >= next_report:
            handle.report(
                progress=done / n, partial=emit(top1_hits, top3_hits, sum_top1_p, sum_human_p, done, final=False)
            )
            next_report = done + report_every

    return emit(top1_hits, top3_hits, sum_top1_p, sum_human_p, n, final=True)


jobs_router.register("cube_metrics", _cube_metrics_runner)
