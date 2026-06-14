"""Training metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.metrics_service import metrics_service


router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/runs")
def runs() -> dict:
    return {"runs": metrics_service.list_runs()}


@router.get("/runs/{run_id}")
def run_rows(run_id: str) -> dict:
    rows = metrics_service.rows(run_id)
    if not rows:
        raise HTTPException(404, f"unknown run or empty metrics: {run_id!r}")
    return {"run_id": run_id, "rows": rows}
