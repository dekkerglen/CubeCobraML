"""Health check + dataset stats."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.lib import train_picks
from backend.services.val_data import val_data


router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    status: str
    version: str


class DataStatsOut(BaseModel):
    val_total: int
    train_total: int


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", version="0.1.0")


@router.get("/data/stats", response_model=DataStatsOut)
def data_stats() -> DataStatsOut:
    """Pick counts across val + train, used by the Metrics Distribution
    sliders so % maps to a real total."""
    val_data.ensure_loaded()
    return DataStatsOut(
        val_total=len(val_data.picks),
        train_total=train_picks.estimated_total(),
    )
