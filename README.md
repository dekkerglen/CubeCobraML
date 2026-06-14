# Magic the Gathering Cube Multi-Purpose Model

This repo contains a model that simultaneously trains a deckbuilder, drafter, and recommender system with a **shared encoder**. The encoder maps a *set of cards* into a 128-dimensional latent space; four decoder heads read that embedding for four different tasks.

The draft and deck-build heads operate on the encoded **pool only** (128-dim). Earlier versions concatenated an encoded "cube context" (a 256-dim input) into these heads, but that was removed — it contributed nothing measurable and was the source of the training instability we were chasing.

## Setup

### 1. Download Raw Data

All exported data is available from CubeCobra's public S3 bucket. Data exports are updated every three months.

```bash
aws s3 sync s3://cubecobra-public/export/ ./raw_data/ --no-sign-request
```

This downloads the following structure:

```
raw_data/
├── indexToOracleMap.json      # Index → Oracle ID mapping
├── simpleCardDict.json        # Oracle ID → Card metadata
├── cubes.json                 # All cube lists
├── decks/
│   └── {n}.json               # Completed draft decks (batched)
├── picks/
│   └── {n}.json               # Individual draft picks (batched)
└── cubeInstances/
    └── {n}.json               # Draft card pools (batched, parallel with picks/)
```

**Key concepts:**
- Cards are represented as numeric indexes throughout. Use `indexToOracleMap.json` to convert an index to an Oracle ID, then look it up in `simpleCardDict.json` for card details.
- `picks/{n}.json` and `cubeInstances/{n}.json` are parallel batches. Each pick has a `cubeCards` field that indexes into the corresponding `cubeInstances` array, giving the full card pool for that draft.
- `decks/{n}.json` entries include a `cube` UUID field that links back to `cubes.json` for the cube's card list.

### 2. Process Data

After downloading, process the raw data into the training format. The picks pass loads millions of records into memory; bump Node's heap accordingly (`--expose-gc` lets the script free large intermediate structures between phases):

```bash
node --max-old-space-size=32768 --expose-gc scripts/process_data.js
```

This always produces both a `data/train/` and a `data/test/` split — there is no `TEST_PERCENT` knob. The split is **per-draft**: a draft is identified by `(file index, owner, cube UUID, cube-snapshot index)`, and a two-phase sampler chooses which drafts become validation:

- **Phase A — random baseline.** Drafts are shuffled with a seeded PRNG and accepted until `RANDOM_TARGET_PICKS` picks are collected, so the per-card pick rate in val matches the full dataset within sampling noise.
- **Phase B — coverage top-up.** Every card with `totalPicks ≥ COVERAGE_THRESHOLD` that Phase A missed is force-covered by adding the draft that newly covers the most still-uncovered cards (rarest first), capped at `HARD_CAP_PICKS`.

The cube head split is **asymmetric**: `train/cubes/` contains *all* valid cubes, while `test/cubes/` contains only the cubes referenced by val drafts. Decks are routed to the split of their content-matched source draft. Training-time statistics (`oracleFrequency`, `pickCount`, `correlations`) are computed over train data and copied to `data/test/` so both splits share the same vocabulary and priors.

Before any of this runs, a **cube-size filter** drops every cube — and all of its picks/decks/instances — whose card count falls outside `[CUBE_SIZE_MIN, CUBE_SIZE_MAX]`. Multi-thousand-card "all sets" dumps and tiny stubs destabilize the encoder.

This produces (abridged):

```
data/
├── train/
│   ├── cubes/                 # ALL valid cube card lists (batched)
│   ├── decks/                 # Batched decks (mainboard/sideboard inline)
│   ├── picks/                 # Batched picks ({pool, pick, pack, cube_cards_idx})
│   ├── cubeInstances/         # Parallel to picks/ — deduplicated cube card pools
│   ├── correlations.json      # Card co-occurrence matrix (train-only)
│   ├── oracleFrequency.json
│   ├── oracleDict.json
│   ├── pickCount.json         # Per-card pick counts (train-only)
│   ├── bomb_card_indices.json # Top-N cards by pickCount (drives val_bomb_agreement)
│   ├── elos.json
│   └── metadata.json
└── test/
    ├── cubes/, decks/, picks/, cubeInstances/
    ├── (mirrors of the train-only stats for vocab parity)
    ├── cube_uuids.json        # Positional cube UUIDs aligned with test/cubes/
    └── val_manifest.json      # Sampler stats (coverage, phase A/B counts)
```

Processed picks store a `cube_cards_idx` integer that indexes into the parallel `cubeInstances/{n}.json` file (same batch number), avoiding duplication of the cube card list across the ~45 picks sharing a draft. Processed decks carry their mainboard/sideboard inline. (Pick records no longer carry `landCount`/`nonlandCount` and no `isLand.json` mask is written — the land penalty was removed.)

#### `process_data.js` configuration

| Var | Default | What it does |
|---|---|---|
| `CUBE_SIZE_MIN` | `180` | Cubes with fewer cards are dropped. |
| `CUBE_SIZE_MAX` | `1080` | Cubes with more cards are dropped. |
| `RANDOM_TARGET_PICKS` | `300000` | Phase-A target: accept random drafts until val reaches this many picks. |
| `HARD_CAP_PICKS` | `500000` | Safety ceiling on total val picks across both phases. |
| `COVERAGE_THRESHOLD` | `400` | Only cards with `totalPicks ≥` this are force-covered in Phase B. |
| `BOMB_COUNT` | `100` | Size of the top-by-`pickCount` "bomb" cohort used by `val_bomb_agreement`. |
| `RANDOM_SEED` | `42` | Seeds the deterministic draft shuffle so the split is reproducible. |

## Training a New Model

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv
uv sync
CPU_ONLY=true HIST_LOSS_WEIGHT=0.1 LR_PEAK=3e-4 LR_END=1e-5 uv run python -m ccml.train 300 1024 false cube
```

That is the **recommended canonical recipe**. Positional arguments are `<epochs> <batch_size> <continue:true|false> <primary>`. `<primary>` is one of `cube` / `card` / `pick` / `deck` and selects which stream defines an epoch; by default the script *extends* `epochs` until the largest stream has been seen at least once (`QUICK=true` honors the literal count, for smoke tests).

Batch `4096` with `LR_PEAK=6e-4` was tested and reaches an equivalent accuracy ceiling with lower argmax churn — a fine alternative to `1024` @ `3e-4`.

**To continue training after interruption:** the model auto-saves weights to `model/` even on Ctrl+C. Set the third positional arg to `true` to resume:

```bash
CPU_ONLY=true uv run python -m ccml.train 300 1024 true cube
```

### How training works

- **Sparse input pipeline.** Generators yield *padded int32 card-index arrays* (~1 KB/record) rather than dense ~37k-float one-hots; records stay sparse through repeat/shuffle/padded-batch, and a single post-batch step densifies the targets and the draft pack mask with scatter ops. The encoder consumes the index arrays directly via `MultiHotEmbedding` (see Architecture).
- **Pack-position sample weights.** Early-pack picks carry more signal than late forced picks: `w = ((K−1)/(pack_start−1)) ** PACK_WEIGHT_POWER`, sqrt by default; the forced last pick gets weight `0`. Applied to the pick head only.
- **Pack-targeted label smoothing.** `PICK_LABEL_SMOOTHING` (default `0.10`) spreads `ε` mass uniformly across the cards in the pack, so non-picked pack cards keep target `≥ ε/K` — preventing their logits from collapsing to −∞.
- **`PickHistogramCCE` loss.** Per-example categorical crossentropy plus `HIST_LOSS_WEIGHT × symmetric-KL` between the batch-mean human and model pick distributions. Being two-sided, it punishes *both* 0%-collapse (model never takes a card humans take) and 100%-saturation.
- **Draft-head priors.** The draft output bias is initialized to `DRAFT_BIAS_ALPHA × log(marginal pick rate)` so each card starts at its empirical log pick rate; `DRAFT_OUTPUT_L2` applies L2 to the draft output kernel to suppress noisy rare-card directions.
- **Optimizer + plateau LR.** AdamW (`clipnorm=1.0`, biases excluded from weight decay) at a *constant* `LR_PEAK`, with `ReduceLROnPlateau` on `val_pick_cce` (factor `LR_PLATEAU_FACTOR`, patience `LR_PLATEAU_PATIENCE`, floored at `LR_END`).
- **File interleaving.** The pick stream reads `INTERLEAVE_FILES` (default 16) files round-robin so the shuffle buffer spans many drafts instead of one, decorrelating batches.
- **Collapse diagnostics.** `ArgmaxCollapseCount` / `PickHistDivergence` surface per-cohort collapse and saturation each epoch, and per-card accumulators (offers / model-argmax / human-pick) are dumped to `runs/<id>/collapse/epoch_NNN.npz` for offline analysis. A fixed val subset (`STABILITY_SUBSET_SIZE`, default 5000) is re-scored each snapshot to report `argmax_churn`.

### Environment-variable knobs

All defaults are sensible — most runs need none beyond the canonical recipe above.

| Var | Default | What it does |
|---|---|---|
| `CPU_ONLY` | `false` | Hide GPUs before TF imports for clean CPU runs. |
| `QUICK` | `false` | Honor the CLI `<epochs>` literally instead of extending to cover the largest stream. |
| `DRAFT_BIAS_ALPHA` | `1.0` | Scale of the log-pick-rate prior on the draft output bias; `0` disables it. |
| `DRAFT_OUTPUT_L2` | `1e-4` | L2 on the draft output kernel. |
| `WEIGHT_DECAY` | `1e-4` | AdamW decoupled weight decay (biases excluded). |
| `PICK_LABEL_SMOOTHING` | `0.10` | ε mass spread across pack cards on the pick target. |
| `HIST_LOSS_WEIGHT` | `0.1` | Weight on the symmetric-KL pick-histogram term. |
| `PACK_WEIGHTING` | `true` | Enable pack-position sample weights. |
| `PACK_WEIGHT_POWER` | `0.5` | Exponent on the pack-position weight (sqrt). |
| `INTERLEAVE_FILES` | `16` | Pick files read round-robin to decorrelate batches. |
| `COLLAPSE_RARE_MAX` | `10` | Cards offered 1..N times in the eval pass count as the "rare" collapse cohort. |
| `STABILITY_SUBSET_SIZE` | `5000` | Fixed val picks re-scored each snapshot for `argmax_churn`. |
| `LR_PEAK` | `3e-4` | Constant peak learning rate. |
| `LR_END` | `1e-5` | Floor for the plateau LR decay. |
| `LR_PLATEAU_PATIENCE` | `8` | Epochs without `val_pick_cce` improvement before cutting LR. |
| `LR_PLATEAU_FACTOR` | `0.5` | Multiplier applied to LR on plateau. |
| `SHUFFLE_BUFFER` | `16384` | Shuffle-buffer size (holds cheap index arrays — fine to raise). |
| `LOG_STEPS` | `0` | Snapshot every N training batches; `0` = snapshot at epoch end. |
| `KEEP_BEST_N_CKPTS` | `10` | Top-N checkpoints to retain by `BEST_METRIC` (beats-worst-or-skip; `0` = unlimited). |
| `BEST_METRIC` | `val_pick_cce` | Keras-log key to rank checkpoints by (falls back to `loss` with no val data). |
| `BEST_MODE` | auto | `min` / `max`; auto-`max` for accuracy-style metrics, else `min`. |

### Validation

Because `process_data.js` always emits a small, stable `data/test/` split, training auto-detects it and runs a full deterministic validation pass each epoch (no shuffling, no augmentation). `val_*` metrics appear in the tqdm bar and in `metrics.csv` (e.g. `val_pick_cce`, `val_top1`, `val_bomb_agreement`). `val_bomb_agreement` — argmax accuracy restricted to the top-`BOMB_COUNT` most-picked cards — is the headline deployment gate.

### Run artifacts

Each training invocation creates a fresh directory under `runs/`:

```
runs/20260531-170337Z-a2314c6b/
├── meta.json            # CLI args, tracked env vars, dataset shape, git SHA, start time
├── metrics.csv          # One row per snapshot; columns = all Keras metrics (+ val_*)
├── collapse/
│   └── epoch_NNN.npz    # Per-epoch per-card offer/argmax/pick accumulators (train + val)
└── ckpts/
    ├── step_NNNNNNNNN_epoch_XXX/   # Per-snapshot weight directories
    ├── latest                      # Always the most recent snapshot (resume after crash)
    ├── best                        # Symlink → current #1 checkpoint by BEST_METRIC
    └── best_index.json             # Ranked manifest (best first) with {step, epoch, metric}
```

`model.load_weights("runs/<id>/ckpts/best")` loads the best checkpoint of a run; `ckpts/latest` is the most recent. `model/` is also overwritten with the latest weights at training end (or on Ctrl+C) for the export / demo pipelines.

### Results

Representative numbers from a `1024` / `LR_PEAK=3e-4` run on this recipe (the
in-flight canonical run reproduces the same trajectory):

| Metric | Value | What it means |
|---|---|---|
| `val_top1` | ~0.50 | Model argmax equals the human pick. |
| `val_top3` | ~0.83 | Human pick is in the model's top 3. |
| `val_bomb_agreement` | ~0.68 | Top-1 restricted to the top-`100` most-picked cards — the deployment gate. |
| `argmax_churn` (converged) | ~5% | Per-snapshot argmax flips at the LR floor; near the intrinsic pick-ambiguity floor. |
| `collapse0_common` | −61% vs. pre-histogram-loss baseline | Common cards humans pick but the model never does — the two-sided histogram loss + pack label smoothing drive this down. |

`top1 ≈ 0.50 / top3 ≈ 0.83` holds across batch `1024` and `4096`, so the
asymptote is **data/capacity-bound, not an optimization artifact** — batch size
and LR schedule move churn and collapse, not the accuracy ceiling.

### Export to TF.js (CubeCobra integration)

Convert a checkpoint to the TFJS graph-model layout CubeCobra's browser draft bot expects:

```bash
BEST_CKPT=runs/<id>/ckpts/best uv run python scripts/export_tfjs.py
```

`export_tfjs.py` writes the TFJS models plus `indexToOracleMap.json` into `OUT_DIR` (default `tfjs_export/`) and self-manages its environment via PEP 723 inline metadata. See `scripts/README.md` for the full local-model draft-sim swap (start the backend, point a localhost CubeCobra origin at `http://localhost:8000`).

## Architecture

A shared `Encoder` maps a *set of cards* into a 128-dimensional latent space, and four decoders read that embedding.

The encoder's first layer is a sparse **`MultiHotEmbedding`** — the gather-sum equivalent of `relu(multi_hot @ W + b)`. For a multi-hot input the dense matmul is just the sum of the kernel rows for the cards present (>99.8% of the FLOPs multiply zeros), so the layer instead takes padded int32 card-index arrays and gathers those rows directly; the math is identical to the old `Dense(37238→512)`. It exposes a `dense_call` so the inference / dashboard / TFJS-export paths can keep passing dense multi-hot floats — the `Encoder` dispatches on input dtype. The stack is `MultiHotEmbedding(→512) → Dense(256, relu) → Dense(128, linear)`.

The four decoder heads:

- **Cube decoder** (recsys) — input: encoded cube (128-dim) → sigmoid over the vocabulary.
- **Draft decoder** — input: encoded **pool only** (128-dim) → linear logits, then a pack-masked softmax. No cube context.
- **Deck-build decoder** — input: encoded **pool only** (128-dim) → sigmoid. No cube context.
- **Correlation decoder** — input: a single encoded card (128-dim) → softmax.

The `draft` and `deck_build` inference methods still *accept* a `cube_context` argument so existing callers don't break, but it is explicitly ignored.

## Demo

The legacy `demo/` server reads two artifacts that are **not** produced by `scripts/export_tfjs.py`:

- `demo/server/indexToOracleMap.json` — must match the current data export (`raw_data/indexToOracleMap.json`). If the card catalog grew between exports, copy the new file over.
- `demo/server/embeddings.json` — precomputed encoder outputs for every oracle, used by the **Synergy** page. This file is **stale after every retrain** and must be regenerated against the freshly exported encoder. Symptom of a stale file: the Synergy page returns "No results yet" for any card whose index exceeds the embedding count.

To regenerate after a retrain + export:

```bash
cd demo/server
cp ../../raw_data/indexToOracleMap.json .   # only if the catalog changed
node compile_embeddings.js                  # rebuilds embeddings.json from the exported encoder
```

Then run the server from `/demo/server` with `npm run start`, run the client from `/demo/client` with `npm start`, and after a minute a browser window should open the demo at `localhost:3000`.

## Uploading data to S3

To recursively upload the `data` folder to S3, run
`aws s3 cp data s3://cubecobra-private/training-2025/july/data --recursive`

And to download it, run
`aws s3 cp s3://cubecobra-private/training-2025/july/data data --recursive`

To upload the exported model to S3, run
`aws s3 cp tfjs_export s3://cubecobra-data-production/model --recursive`
MAKE SURE to get the `indexToOracleMap.json` file as well, or the model won't work.

## Dashboard

A FastAPI + React UI for inspecting training, debugging pick collapse, A/B comparing checkpoints, and asking "tell me everything about *this* card" from anywhere in the app. See `backend/README.md` for the route map and architecture.

### One-time setup

```bash
uv sync --extra dashboard      # installs FastAPI / uvicorn / httpx
cd frontend && npm install && cd ..
```

To make the prod TensorFlow.js model selectable in the dashboard, point at your local CubeCobra checkout:

```bash
echo "CUBECOBRA_PROD_MODEL=$HOME/Projects/CubeCobra/packages/recommenderService/model" >> .env
```

### Running

```bash
# Build the React bundle (writes to backend/static/):
cd frontend && npm run build && cd ..

# Serve API + SPA from one port:
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 (or the host's Tailscale IP from your phone).

### Hot-reload dev mode

```bash
# Terminal A: backend with reload
uv run uvicorn backend.main:app --reload

# Terminal B: vite dev server (proxies /api → :8000)
cd frontend && npm run dev
```

Open http://localhost:5173.

### Smoke test

`uv run python -m backend.smoke` runs an in-process FastAPI client against every major endpoint (cards / checkpoints / picks / metrics / predict / deckbuilder) and reports OK / FAIL per step.
