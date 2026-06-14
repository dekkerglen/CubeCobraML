# Dashboard backend

A FastAPI service that exposes the trained model + processed data for the
React dashboard (`frontend/`). It is an **optional analysis surface** — none
of it is required to train or export a model; it reads the same `data/` and
`runs/` artifacts the training pipeline produces.

## Running

See the root `README.md` "Dashboard" section. In short:

```bash
uv sync --extra dashboard
uv run uvicorn backend.main:app --reload          # API on :8000
cd frontend && npm run dev                         # UI on :5173 (proxies /api)
```

`backend.main:app` just re-exports the app built by `backend/app.py:create_app()`,
which mounts every router under `/api` and serves the built SPA from
`backend/static/` (populated by `npm run build`). When a `tfjs_export/`
directory exists it is also mounted at `/model/*` for the CubeCobra local-model
swap (see `scripts/README.md`).

## Layout

- `app.py` — app factory: CORS, router mounts, SPA + `/model` static mounts.
- `routers/` — HTTP endpoints, one module per concern (all under `/api`):

  | Prefix | Purpose |
  |---|---|
  | `/health` | Liveness + dataset presence. |
  | `/cards` | Card search, detail, reverse-lookup (in cubes / decks / packs). |
  | `/checkpoints` | Flat checkpoint list + per-ckpt KPI snapshot. |
  | `/runs` | Runs-first selector; per-run ckpts + per-card collapse sidecars. |
  | `/metrics` | A run's `metrics.csv` rows. |
  | `/picks`, `/decks`, `/cubes`, `/drafts` | Browse the processed data + draft replay. |
  | `/predict` | Live draft / deck-build / recommend / encode against a checkpoint. |
  | `/collapse` | Per-card model-vs-human pick-rate jobs (the collapse explorer). |
  | `/cube_metrics` | Unified per-cube metrics job. |
  | `/jobs` | Generic async-job status/streaming for the long-running endpoints above. |

- `services/` — stateful singletons behind the routers: model registry (loads +
  caches checkpoints), val-data loader, draft-session index, cube directory,
  the async job runner, etc.
- `lib/` — stateless helpers: checkpoint discovery (`ckpt.py`), inference
  wrappers (`inference.py`, `deckbuild.py`), the prod-TFJS loader
  (`tfjs_load.py`), config (`config.py`).
- `models.py` — Pydantic response models (mirrored by `frontend/src/lib/types.ts`).
- `smoke.py` — `uv run python -m backend.smoke` exercises every major endpoint
  in-process and reports OK/FAIL per step.

## Checkpoint compatibility

The model registry loads any run's `ckpts/<dir>` via `CubeCobraMLSystem.load_weights`,
including older checkpoints: the encoder's `load_weights` shims a legacy dense
first-layer kernel into the current `MultiHotEmbedding` table, so runs from
before the sparse rewrite stay selectable for comparison. The prod TFJS model
is loaded through `lib/tfjs_load.py` and exposes the same `draft`/`deck_build`/
`recommend` surface as a trained checkpoint.
