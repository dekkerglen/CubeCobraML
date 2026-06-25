#!/usr/bin/env -S uv run -q --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "tensorflow==2.17.1",
#     "tensorflowjs==4.22.0",
#     "jax",
#     "flax",
#     "setuptools<81",
#     "numpy<2",
# ]
#
# [tool.uv]
# # tensorflowjs hard-requires tensorflow-decision-forests, which has no arm64
# # wheel. The conversion paths we use never touch TFDF, so we override it to
# # a marker that resolves to nothing. Combined with the sys.modules stub below,
# # this lets the script install cleanly on Apple Silicon.
# override-dependencies = ["tensorflow-decision-forests ; platform_machine == 'never'"]
# ///
"""Export a CubeCobraML v2 checkpoint as TFJS graph models laid out the way
CubeCobra's browser draft bot expects.

Why this exists: CubeCobra's draftBot.ts fetches
  /model/encoder/model.json
  /model/draft_decoder/model.json
  /model/cube_decoder/model.json
  /model/deck_build_decoder/model.json
  /model/indexToOracleMap.json

v3 heads are prod-shaped (draft 128-dim in, deck_build 128-dim in, no cube
context anywhere), so the client's no-ctx forward path matches exactly.

Run:
  ./scripts/export_tfjs.py                          # latest run's best ckpt
  BEST_CKPT=runs/<id>/ckpts/best ./scripts/export_tfjs.py
  OUT_DIR=/some/path ./scripts/export_tfjs.py

Workarounds baked in:
  - tensorflow-decision-forests has no arm64 wheel but tensorflowjs declares
    it as a hard dep; stubbed before tensorflowjs imports.
  - Newer setuptools removed pkg_resources, which tensorflow-hub still uses;
    pinned <81 above.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Stub tensorflow_decision_forests BEFORE tensorflowjs imports it (TFDF has no
# arm64 wheel; the import path inside tensorflowjs we don't touch).
sys.modules.setdefault("tensorflow_decision_forests", types.ModuleType("tensorflow_decision_forests"))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import tensorflowjs as tfjs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Register custom Keras classes (e.g. MultiHotEmbedding) so load_model can
# deserialize checkpoints that contain them.
sys.path.insert(0, str(REPO))
import ccml.model  # noqa: F401, E402

DATA_TRAIN = REPO / "data" / "train"


def _resolve_ckpt() -> Path:
    if env := os.environ.get("BEST_CKPT"):
        p = Path(env)
        if not p.is_absolute():
            p = REPO / p
        return p
    runs = REPO / "runs"
    for run_dir in sorted(runs.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name == "archive":
            continue
        cand = run_dir / "ckpts" / "best"
        if (cand / "encoder" / "model.keras").exists():
            return cand.resolve()
    raise FileNotFoundError("No usable ckpt under runs/; set BEST_CKPT.")


def _convert_keras_to_graph(model: keras.Model, out_dir: Path) -> None:
    """SavedModel → TFJS graph_model. The browser uses tf.loadGraphModel
    everywhere — this is the format prod expects, not layers_model."""
    saved = out_dir.with_suffix(".saved")
    shutil.rmtree(saved, ignore_errors=True)
    model.export(str(saved))
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tfjs.converters.convert_tf_saved_model(str(saved), str(out_dir))
    shutil.rmtree(saved, ignore_errors=True)


class _DenseEncoder(keras.Model):
    """Encoder that takes a dense multi-hot instead of padded card indices.

    The checkpoint encoder is Sequential[MultiHotEmbedding, Dense, Dense] whose
    forward expects padded int32 card-index arrays. CubeCobra's draft bot feeds
    a dense multi-hot, so without this the exported encoder reads each of the
    vocab-wide values as an index and returns a frozen constant. This runs the
    embedding's dense_call path then the same Dense stack — identical weights,
    dense input contract.
    """

    def __init__(self, embed, dense_layers, **kwargs):
        super().__init__(**kwargs)
        self.embed = embed
        self.dense_layers = dense_layers

    def call(self, x):
        y = self.embed.dense_call(x)
        for layer in self.dense_layers:
            y = layer(y)
        return y


def _to_dense_encoder(m: keras.Model) -> keras.Model:
    embed, *tail = m.layers
    dense = _DenseEncoder(embed, tail, name="encoder_dense")
    dense(keras.ops.zeros((1, embed.vocab)))  # build so weights are concrete
    return dense


def main() -> None:
    ckpt = _resolve_ckpt()
    out_dir = Path(os.environ.get("OUT_DIR", REPO / "tfjs_export"))
    print(f"[ckpt] {ckpt}")
    print(f"[out]  {out_dir}")

    heads = {
        "encoder": ckpt / "encoder" / "model.keras",
        "draft_decoder": ckpt / "draft_decoder" / "model.keras",
        "cube_decoder": ckpt / "cube_decoder" / "model.keras",
        "deck_build_decoder": ckpt / "deck_build_decoder" / "model.keras",
    }
    missing = [n for n, p in heads.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ckpt is missing head subdirs: {missing}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    for head, src in heads.items():
        print(f"[convert] {head}")
        m = keras.models.load_model(src)
        if head == "encoder":
            m = _to_dense_encoder(m)
            print("    [encoder] exporting with a dense multi-hot input signature")
        else:
            print(f"    in={m.input_shape}  out={m.output_shape}")
        _convert_keras_to_graph(m, out_dir / head)

    # indexToOracleMap.json — CubeCobra's client loads it as
    # `{"<int>": "<oracle_id>"}` and uses string-int keys.
    oracle_path = DATA_TRAIN / "oracleDict.json"
    oracles = json.load(open(oracle_path))
    idx_map = {str(i): o for i, o in enumerate(oracles)}
    num_oracles = len(idx_map)
    (out_dir / "indexToOracleMap.json").write_text(json.dumps(idx_map))
    print(f"[meta] indexToOracleMap.json ({num_oracles} entries)")

    # NOTE: no cube_context_encoder. v3 deleted cube context from the deck
    # head (128-dim input, matching prod), so the CubeCobra client needs no
    # ctx model. Exporting a v2-era ckpt (256-dim deck head) requires the
    # old export logic — check out commit 0d90388.

    # Manifest so the server (or you) can sanity-check what was exported.
    (out_dir / "_export_manifest.json").write_text(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "num_oracles": num_oracles,
                "heads_exported": list(heads),
                "note": (
                    "On a localhost CubeCobra origin, draftBot.ts auto-detects and "
                    "loads /model/* from http://localhost:8000. v3 heads are all "
                    "prod-shaped (draft 128 in, deck_build 128 in) — no cube context "
                    "anywhere, no client-side ctx threading needed."
                ),
            },
            indent=2,
        )
    )

    print(f"\n[done] {out_dir}")
    print("       contents:", sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main()
