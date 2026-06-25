"""Run-level discovery for the runs-first model selector.

Treats each training run as a first-class entity (one experiment). The frontend
header selector lists runs, not individual ckpts; per-run drill-down to specific
ckpts goes through `/runs/{run_id}/ckpts`.

The flat `/checkpoints` endpoint is kept alive for lookup-by-key by components
like CardDrawer and Metrics's RunPicker that already work fine with the flat list.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.lib.ckpt import list_runs_rich, list_ckpts, list_runs
from backend.models import CheckpointOut, RunCollapseEpochsOut, RunCollapseOut, RunOut
from backend.services.model_registry import registry


router = APIRouter(prefix="/runs", tags=["runs"])


def _run_to_out(r) -> RunOut:
    return RunOut(
        run_id=r.run_id,
        label=r.label,
        short=r.short,
        start_utc=r.start_utc,
        git_sha=r.git_sha,
        archived=r.archived,
        best_key=r.best_key,
        latest_key=r.latest_key,
        best_metric=r.best_metric,
        best_mode=r.best_mode,  # type: ignore[arg-type]
        best_metric_name=r.best_metric_name,
        n_ckpts=r.n_ckpts,
    )


@router.get("", response_model=list[RunOut])
def list_runs_endpoint() -> list[RunOut]:
    return [_run_to_out(r) for r in list_runs_rich()]


# ----- per-card collapse sidecars --------------------------------------------
#
# Training dumps runs/<id>/collapse/epoch_NNN.npz with per-card accumulators
# (offers / model-argmax / human-pick) for both passes. Serving them in the
# same vector shape as /collapse lets the frontend reuse the per-card
# explorer with zero inference cost.

_EPOCH_RE = re.compile(r"^epoch_(\d+)\.npz$")

CollapseSplit = Literal["val", "train", "train_cum"]


def _run_dir(run_id: str) -> Path:
    target = next((p for p in list_runs() if p.name == run_id), None)
    if target is None:
        raise HTTPException(404, f"unknown run: {run_id!r}")
    return target


def _sidecar_epochs(run_dir: Path) -> list[int]:
    collapse_dir = run_dir / "collapse"
    if not collapse_dir.is_dir():
        return []
    epochs = []
    for p in collapse_dir.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


@lru_cache(maxsize=64)
def _load_sidecar_vectors(
    npz_path: str,
    split: Literal["val", "train"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Past-epoch sidecars are immutable, so caching by path is safe."""
    with np.load(npz_path) as z:
        keys = (f"{split}_in_pack", f"{split}_model_argmax", f"{split}_human_pick")
        if any(k not in z for k in keys):
            return None
        return tuple(z[k].astype(np.int64) for k in keys)


@router.get("/{run_id}/collapse", response_model=RunCollapseEpochsOut)
def run_collapse_epochs(run_id: str) -> RunCollapseEpochsOut:
    return RunCollapseEpochsOut(run_id=run_id, epochs=_sidecar_epochs(_run_dir(run_id)))


@router.get("/{run_id}/collapse/{epoch}", response_model=RunCollapseOut)
def run_collapse_epoch(
    run_id: str,
    epoch: int,
    split: CollapseSplit = "val",
) -> RunCollapseOut:
    run_dir = _run_dir(run_id)
    epochs = _sidecar_epochs(run_dir)
    if epoch not in epochs:
        raise HTTPException(404, f"no collapse sidecar for epoch {epoch}")

    base_split: Literal["val", "train"] = "train" if split.startswith("train") else "val"
    use_epochs = [e for e in epochs if e <= epoch] if split == "train_cum" else [epoch]

    totals: list[np.ndarray] | None = None
    for e in use_epochs:
        vecs = _load_sidecar_vectors(str(run_dir / "collapse" / f"epoch_{e:03d}.npz"), base_split)
        if vecs is None:
            continue
        if totals is None:
            totals = [v.copy() for v in vecs]
        else:
            for t, v in zip(totals, vecs):
                t += v
    if totals is None:
        raise HTTPException(404, f"sidecar has no {base_split} vectors")

    in_pack, model_argmax, human_pick = totals
    return RunCollapseOut(
        run_id=run_id,
        epoch=epoch,
        split=split,
        epochs_aggregated=len(use_epochs),
        appearances=in_pack.tolist(),
        model_picks=model_argmax.tolist(),
        human_picks=human_pick.tolist(),
    )


@router.get("/{run_id}/ckpts", response_model=list[CheckpointOut])
def run_ckpts(run_id: str) -> list[CheckpointOut]:
    """All checkpoints under one run, in epoch order."""
    target = next((p for p in list_runs() if p.name == run_id), None)
    if target is None:
        raise HTTPException(404, f"unknown run: {run_id!r}")
    out: list[CheckpointOut] = []
    for e in list_ckpts(target):
        out.append(
            CheckpointOut(
                key=e.key,
                label=e.label,
                kind=e.kind,  # type: ignore[arg-type]
                run_id=e.run_id,
                epoch=e.epoch,
                metric=e.metric,
                is_best=False,  # not meaningful in a per-run view
                is_loaded=registry.is_loaded(e.key),
            )
        )
    return out
