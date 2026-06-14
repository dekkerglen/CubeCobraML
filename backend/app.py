"""FastAPI app factory.

Single-process layout: this app serves both the JSON API at `/api/*` and the
built React SPA from `backend/static/` (mounted at `/`). During development,
the React dev server (vite on :5173) proxies its `/api` calls back here on
:8000, so both sides hot-reload independently.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import (
    cards,
    checkpoints,
    collapse,
    cube_metrics,
    cubes,
    decks,
    drafts,
    health,
    jobs,
    metrics,
    picks,
    predict,
    runs,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
# TFJS export produced by scripts/export_tfjs.py. Served at /model/* so
# CubeCobra's browser bot can load a local checkpoint with one URL override.
TFJS_EXPORT_DIR = Path(__file__).resolve().parents[1] / "tfjs_export"


def create_app() -> FastAPI:
    app = FastAPI(
        title="CubeCobraML Dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Vite dev server runs on :5173 and proxies /api → here, but allow CORS
    # for direct browser hits during dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://0.0.0.0:5173",
            # CubeCobra dev server — for local-model draft-sim swap.
            # 8080 is the webpack dev server (where the browser actually hits);
            # 5000 is the Node server. Both included for completeness.
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5000",
            "http://127.0.0.1:5000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(checkpoints.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(metrics.router, prefix="/api")
    app.include_router(cards.router, prefix="/api")
    app.include_router(picks.router, prefix="/api")
    app.include_router(decks.router, prefix="/api")
    app.include_router(cubes.router, prefix="/api")
    app.include_router(drafts.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(collapse.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    # cube_metrics has no HTTP routes of its own — importing it registers
    # the "cube_metrics" job-runner kind via jobs_router.register at module
    # load time. Same pattern as collapse.py.
    _ = cube_metrics  # keep the import alive for linters

    # /model/* — CubeCobra's draft bot fetches `/model/encoder/model.json` etc.;
    # mounting tfjs_export here lets the browser client point at this server
    # and use our local v2 checkpoint instead of the prod model. Mounted before
    # the SPA so /model paths win over the wildcard.
    if TFJS_EXPORT_DIR.is_dir():
        app.mount("/model", StaticFiles(directory=str(TFJS_EXPORT_DIR)), name="tfjs_model")

    # Mount the built SPA last so /api/* routes always win. The directory may
    # be empty in dev (vite serves directly from frontend/) — that's fine.
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="spa")

    return app


app = create_app()
