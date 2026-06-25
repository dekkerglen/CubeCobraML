# Scripts

Tooling around the model. Python scripts run under `uv`
(`uv run python scripts/<name>.py`); the data-pipeline scripts are Node.

## Data pipeline (run on a data refresh)

| Script | What it does |
|---|---|
| `process_data.js` | Splits `raw_data/{picks,cubes,decks}` into `data/{train,test}/`, applies the cube-size filter, samples the validation split, and writes the per-card sidecars (frequencies, pick counts, bomb indices). `node --max-old-space-size=24576 --expose-gc scripts/process_data.js` |
| `create_metadata.js` | Counts the processed cubes/decks/picks and writes `metadata.json`. |

`lib/` holds the pieces `process_data.js` composes: `val_sampler.js` (the
distribution-matched + coverage-guaranteed validation-split sampler) and
`random.js` (its seedable PRNG).

## Model export & CubeCobra integration

| Script | What it does |
|---|---|
| `export_tfjs.py` | Converts a Keras checkpoint (`BEST_CKPT=runs/<id>/ckpts/best`) to TFJS graph models + `indexToOracleMap.json`, laid out as CubeCobra's browser draft bot expects. Output dir via `OUT_DIR` (default `tfjs_export/`). Self-manages its env via PEP 723 inline metadata. |

The local-model draft-sim swap: run `export_tfjs.py`, start the backend (it
mounts the export at `/model/*`), then on a localhost CubeCobra origin the
patched `draftBot.ts` auto-loads from `http://localhost:8000`. The client-side
edits live in the sibling CubeCobra repo and are out of scope for this repo.
