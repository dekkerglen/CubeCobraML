"""Generic async-job primitive.

Workers expose a `JobHandle` API (`cancel: Event`, `report(progress, partial, eta)`)
and run inside a single-threaded `ThreadPoolExecutor` — TF Keras inference is the
target workload and isn't thread-safe per session, so serial execution is the
right default. The pool sizing is config knob; bump to 2+ only for non-TF jobs.

Jobs are addressed by an opaque string `id`. Terminal jobs survive an hour
(TTL) before eviction to give the frontend time to fetch the final result.

Usage from a router:

    @router.post("/api/jobs/collapse")
    def start_collapse(body: CollapseJobBody) -> JobCreatedOut:
        job_id = job_store.create("collapse", body.model_dump(),
                                  runner=_run_collapse)
        return JobCreatedOut(id=job_id)

    @router.get("/api/jobs/{job_id}")
    def get(job_id: str) -> JobOut:
        job = job_store.get(job_id)
        if job is None: raise HTTPException(404)
        return job.to_out()

The runner receives `(handle, body) -> dict | None`. Returning a dict marks the
result; raising marks error; calling `handle.cancel.is_set()` mid-loop lets the
worker exit early (status flips to cancelled).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Literal


log = logging.getLogger(__name__)


JobStatus = Literal["queued", "running", "done", "error", "cancelled"]

TerminalStatuses: frozenset[JobStatus] = frozenset(("done", "error", "cancelled"))


# Max parallel workers. TF Keras inference is the primary load; running two
# inference jobs in parallel just contends for GPU/CPU. Single worker means
# jobs queue serially — predictable, no thrash.
MAX_WORKERS = 1

# How long terminal jobs survive in memory after completion (seconds).
TTL_SECONDS = 3600


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    status: JobStatus = "queued"
    progress: float = 0.0
    eta_seconds: float | None = None
    partial: dict | None = None
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def to_out(self) -> dict:
        """Pydantic-friendly dict (Job dataclass has non-serializable fields)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "eta_seconds": self.eta_seconds,
            "partial": self.partial,
            "result": self.result,
            "error": self.error,
        }


class JobHandle:
    """Worker-facing API. Workers report progress and check for cancellation
    through this — they never touch the Job dataclass directly."""

    def __init__(self, store: "JobStore", job: Job) -> None:
        self._store = store
        self._job = job
        self.cancel = job.cancel
        self._started = time.monotonic()

    def report(
        self,
        progress: float,
        partial: dict | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        """Push a progress update. `progress` ∈ [0, 1]. `partial` is opaque
        to the store; frontends use whatever shape makes sense per job kind.
        If `eta_seconds` is None and progress > 0, we estimate from elapsed."""
        self._store._update(
            self._job.id, progress=progress, partial=partial, eta_seconds=eta_seconds, auto_eta_started=self._started
        )


class JobStore:
    """In-memory job store. Singleton via the module-level `job_store` below."""

    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="jobs",
        )
        self._eviction_timer: threading.Timer | None = None

    # ----- public ----------------------------------------------------------

    def create(
        self,
        kind: str,
        body: dict,
        runner: Callable[[JobHandle, dict], dict | None],
    ) -> str:
        """Mint a job_id, submit the worker, return immediately.

        `runner(handle, body) -> dict | None` is the work function. Returning a
        dict marks the job done with that result; raising marks it error.
        """
        job_id = uuid.uuid4().hex[:16]
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
            handle = JobHandle(self, job)
            job.future = self._executor.submit(self._run, job, handle, body, runner)
        self._arm_eviction()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Signal cancellation. Returns True if the job exists and is non-terminal.
        The worker exits at the next batch boundary; status is flipped to
        `cancelled` at that point (not immediately)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in TerminalStatuses:
                return False
            job.cancel.set()
            return True

    # ----- internals -------------------------------------------------------

    def _run(self, job: Job, handle: JobHandle, body: dict, runner: Callable[[JobHandle, dict], dict | None]) -> None:
        with self._lock:
            job.status = "running"
            job.updated_at = time.monotonic()
        try:
            result = runner(handle, body)
            with self._lock:
                if job.cancel.is_set():
                    job.status = "cancelled"
                else:
                    job.status = "done"
                    job.result = result
                    job.progress = 1.0
                job.updated_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001 — surface every worker failure
            log.exception("Job %s (%s) failed", job.id, job.kind)
            with self._lock:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.updated_at = time.monotonic()

    def _update(
        self,
        job_id: str,
        *,
        progress: float,
        partial: dict | None,
        eta_seconds: float | None,
        auto_eta_started: float,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(job.progress, progress)
            if partial is not None:
                job.partial = partial
            if eta_seconds is not None:
                job.eta_seconds = eta_seconds
            elif progress > 0.0:
                elapsed = time.monotonic() - auto_eta_started
                # remaining = elapsed * (1 - p) / p
                job.eta_seconds = elapsed * max(0.0, (1.0 - progress)) / progress
            job.updated_at = time.monotonic()

    def _arm_eviction(self) -> None:
        with self._lock:
            if self._eviction_timer is not None:
                self._eviction_timer.cancel()
            self._eviction_timer = threading.Timer(TTL_SECONDS, self._evict_terminal)
            self._eviction_timer.daemon = True
            self._eviction_timer.start()

    def _evict_terminal(self) -> None:
        now = time.monotonic()
        with self._lock:
            stale = [
                jid
                for jid, j in self._jobs.items()
                if j.status in TerminalStatuses and (now - j.updated_at) > TTL_SECONDS
            ]
            for jid in stale:
                self._jobs.pop(jid, None)
            if self._jobs:
                # re-arm if anything is still around so we keep sweeping
                self._eviction_timer = threading.Timer(TTL_SECONDS, self._evict_terminal)
                self._eviction_timer.daemon = True
                self._eviction_timer.start()
            else:
                self._eviction_timer = None


job_store = JobStore()
