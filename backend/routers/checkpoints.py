"""Checkpoint listing + headline KPI snapshots."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.models import CheckpointOut, CheckpointSnapshot
from backend.services.metrics_service import metrics_service
from backend.services.model_registry import registry


router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])


def _to_out(entry, *, best_key: str | None) -> CheckpointOut:
    return CheckpointOut(
        key=entry.key,
        label=entry.label,
        kind=entry.kind,  # type: ignore
        run_id=entry.run_id,
        epoch=entry.epoch,
        metric=entry.metric,
        is_best=(entry.key == best_key),
        is_loaded=registry.is_loaded(entry.key),
    )


@router.get("", response_model=list[CheckpointOut])
def list_checkpoints() -> list[CheckpointOut]:
    from backend.lib.ckpt import find_best  # circular-safe import

    entries = registry.list()
    best = find_best(entries)
    best_key = best.key if best else None
    return [_to_out(e, best_key=best_key) for e in entries]


@router.get("/{key:path}/snapshot", response_model=CheckpointSnapshot)
def snapshot(key: str) -> CheckpointSnapshot:
    entry = registry.find(key)
    if entry is None:
        raise HTTPException(404, f"unknown checkpoint: {key!r}")
    if not entry.run_id:
        return CheckpointSnapshot(ckpt=key, run_id=None, epoch=entry.epoch)
    row = metrics_service.last_completed_row(entry.run_id)
    if row is None:
        return CheckpointSnapshot(ckpt=key, run_id=entry.run_id, epoch=entry.epoch)
    return CheckpointSnapshot(
        ckpt=key,
        run_id=entry.run_id,
        epoch=entry.epoch,
        val_bomb_agreement=row.get("val_bomb_agreement"),
        val_top1=row.get("val_top1"),
        val_top3=row.get("val_top3"),
        val_loss=row.get("val_loss"),
        val_pick_cce=row.get("val_pick_cce"),
        argmax_churn=row.get("argmax_churn"),
        val_collapse0_common=row.get("val_collapse0_common"),
        val_sat100_common=row.get("val_sat100_common"),
    )


@router.post("/{key:path}/preload", status_code=202)
def preload(key: str, bg: BackgroundTasks) -> dict:
    """Trigger a background load so the frontend can fire-and-forget warmup."""
    if registry.find(key) is None:
        raise HTTPException(404, f"unknown checkpoint: {key!r}")
    bg.add_task(lambda: registry.get(key))
    return {"accepted": True}
