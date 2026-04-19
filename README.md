# Magic the Gathering Cube Multi-Purpose Model

This repo contains a model that simultaneously trains a deckbuilder, drafter, and recommender system with a shared encoder.

The draft and deck-build decoders receive **cube context** — the full set of cards available in a draft — concatenated with the pool/pack encoding. This lets the model condition its decisions on what cards the drafter could expect to see.

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

After downloading, process the raw data into the training format. This script needs a large heap because it holds cube maps and correlation matrices in memory:

```bash
node --max-old-space-size=8192 scripts/process_data.js
```

This produces:

```
data/
└── train/
    ├── cubes/              # Batched cube card lists
    ├── decks/              # Batched decks with cube context
    ├── picks/              # Batched picks (cube_cards_idx reference)
    ├── cubeInstances/      # Parallel to picks/ — deduplicated cube card pools
    ├── correlations.json
    ├── oracleFrequency.json
    ├── oracleDict.json
    ├── elos.json
    └── metadata.json
```

Processed picks store a `cube_cards_idx` integer that indexes into the parallel `cubeInstances/{n}.json` file (same batch number). This avoids duplicating the ~360-card cube context array across all ~45 picks sharing the same draft. Processed decks include `cube_cards` inline (the full cube card list from `cubes.json`).

A trained model is committed to the repo under `tfjs_model`. You can use this model to run the demo.

## Training a New Model

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv
uv sync
source .venv/bin/activate
python ccml/train.py 500 256 false cube
```

The above command will train a model with a batch size of 256 such that training will stop after it has iterated over each cube 500 times. The last parameter can be any of `cube`, `card`, `pick`, `deck` — it controls which dataset determines the epoch size. If the requested epoch count would not cover the full dataset, the script extends training until every data point is seen at least once.

Once your model is done training (or you manually kill the process), your model will be saved in the `model` directory. Execute the following script to convert the model to TF.js for deployment or the demo. The conversion requires `tensorflowjs` binaries (not supported on Windows).

```bash
sh scripts/convert.sh
```

To evaluate on the holdout set:

```bash
python ccml/test.py
```

## Architecture

The system uses a shared encoder that maps a set of cards (represented as a binary vector) into a 128-dimensional latent space. Four decoders operate on these embeddings:

- **Cube decoder** — predicts which cards belong in a cube (input: encoded cube, 128-dim)
- **Draft decoder** — predicts the best card to pick from a pack (input: encoded pool ‖ encoded cube context, 256-dim)
- **Deck-build decoder** — predicts mainboard cards from a card pool (input: encoded pool ‖ encoded cube context, 256-dim)
- **Correlation decoder** — predicts card co-occurrence (input: encoded card, 128-dim)

The draft and deck-build decoders receive twice-wide input: the first 128 dimensions are the encoded pool/pack state, and the second 128 dimensions are the encoded cube card list (all cards available in that draft). This cube context helps the model understand the draft environment.

## Demo

First run the server from `/demo/server` with `npm run start`

Then run the client from `/demo/client` with `npm start`

Wait a minute, and then a browser window should open with the demo at `localhost:3000`

## Uploading data to S3

To recursively upload the `data` folder to S3, run
`aws s3 cp data s3://cubecobra-private/training-2025/july/data --recursive`

And to download it, run
`aws s3 cp s3://cubecobra-private/training-2025/july/data data --recursive`

To upload the tfjs model to S3, run
`aws s3 cp tfjs_model s3://cubecobra-data-production/model --recursive`
MAKE SURE to get the indexToOracleMap file as well, or the model won't work.
