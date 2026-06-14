"""Generic async-job endpoints. Job kinds register their runner in
`_REGISTRY` below; the router dispatches by URL kind segment.

Frontend pattern: `POST /api/jobs/<kind>` with the kind-specific body → `{id}`;
poll `GET /api/jobs/{id}` at ~750ms until `status` is terminal; `partial` shape
is whatever the kind emits and the frontend understands.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException

from backend.models import JobCreatedOut, JobOut
from backend.services.jobs import JobHandle, job_store


router = APIRouter(prefix="/jobs", tags=["jobs"])


# Runner registry: kind → (runner, doc). Runners are populated by other modules
# at import time. Phase Q wires up "collapse" inside backend/routers/collapse.py.
JobRunner = Callable[[JobHandle, dict], dict | None]
_REGISTRY: dict[str, JobRunner] = {}


def register(kind: str, runner: JobRunner) -> None:
    """Register a job kind. Idempotent — re-registering overwrites (handy for
    hot-reload during dev)."""
    _REGISTRY[kind] = runner


# ----- built-in "noop" kind for smoke testing the runner ---------------------


def _noop_runner(handle: JobHandle, body: dict) -> dict:
    """Sleeps for `seconds` (default 5), emits 10 progress reports. Lets the
    frontend ProgressBar be exercised without any TF dependency."""
    seconds = float(body.get("seconds", 5))
    steps = 10
    for i in range(1, steps + 1):
        if handle.cancel.is_set():
            return {"done_at_step": i}
        time.sleep(seconds / steps)
        handle.report(progress=i / steps, partial={"step": i, "of": steps})
    return {"completed_steps": steps, "seconds": seconds}


register("noop", _noop_runner)


# ----- routes ----------------------------------------------------------------


@router.post("/{kind}", response_model=JobCreatedOut)
def start(kind: str, body: dict[str, Any] = Body(default_factory=dict)) -> JobCreatedOut:
    runner = _REGISTRY.get(kind)
    if runner is None:
        raise HTTPException(404, f"unknown job kind: {kind!r}. registered: {sorted(_REGISTRY.keys())}")
    job_id = job_store.create(kind, body, runner)
    return JobCreatedOut(id=job_id)


@router.get("/{job_id}", response_model=JobOut)
def get(job_id: str) -> JobOut:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or evicted job_id")
    return JobOut(**job.to_out())


@router.post("/{job_id}/cancel", status_code=202)
def cancel(job_id: str) -> dict:
    ok = job_store.cancel(job_id)
    if not ok:
        # Either unknown or already terminal — 404 in either case is fine
        # because there's nothing left to act on.
        raise HTTPException(404, "job not found or already terminal")
    return {"cancelled": True}
