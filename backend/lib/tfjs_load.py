"""Load a TFJS graph-model export into a Keras model directly.

The tensorflowjs Python package dropped `tfjs_graph_model` as an input format
in recent versions, but the on-disk format is simple:

    model.json: {format, modelTopology (GraphDef as JSON), weightsManifest}
    group1-shardXofN.bin: raw little-endian bytes for the concatenated weights
        in manifest order.

We don't actually need the GraphDef — the prod model is a stack of Dense
layers per head. We map weight names to (kernel, bias) per Dense layer using
the canonical name pattern from a Sequential model, rebuild a Keras Sequential
with the right shapes, and assign weights.

Heads in a CubeCobra recommenderService/model export:
    encoder:                  37238 -> 512 -> 256 -> 128         (relu, relu, linear)
    cube_decoder:             128 -> 256 -> 512 -> 37238          (relu, relu, sigmoid)
    deck_build_decoder:       similar shape; different name suffixes
    draft_decoder:            similar shape; softmax output
    correlation_decoder:      identity-style head
    cube_context_encoder:     small MLP

We infer activations from the modelTopology (the JSON has the fused-ops list)
and the output shape, falling back to "linear" if unknown — sigmoid/softmax
output activations matter for matching prod behavior.

Usage:
    from tfjs_load import load_tfjs_head
    encoder = load_tfjs_head("/path/to/prod/encoder")
    out = encoder(input_tensor)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


DTYPE_BYTES = {"float32": 4, "float16": 2, "int32": 4}
DTYPE_NUMPY = {"float32": np.float32, "float16": np.float16, "int32": np.int32}


def _b64_decode(s: str) -> str:
    """Some TFJS exports base64-encode op suffixes in fused_ops."""
    try:
        return base64.b64decode(s).decode("utf-8")
    except Exception:
        return s


def _read_all_shards(model_dir: Path, manifest: list[dict]) -> bytes:
    """Concatenate all shard bytes in manifest order."""
    chunks = []
    for grp in manifest:
        for fname in grp["paths"]:
            chunks.append((model_dir / fname).read_bytes())
    return b"".join(chunks)


def _parse_weights(manifest: list[dict], raw: bytes) -> dict[str, np.ndarray]:
    """Slice `raw` into named ndarrays per the manifest."""
    weights: dict[str, np.ndarray] = {}
    offset = 0
    for grp in manifest:
        for w in grp["weights"]:
            shape = tuple(w["shape"])
            dtype = w["dtype"]
            n_elems = 1
            for d in shape:
                n_elems *= d
            nbytes = n_elems * DTYPE_BYTES[dtype]
            arr = np.frombuffer(raw, dtype=DTYPE_NUMPY[dtype], count=n_elems, offset=offset)
            weights[w["name"]] = arr.reshape(shape).copy()
            offset += nbytes
    return weights


def _infer_dense_layers(topology: dict, weights: dict[str, np.ndarray]) -> list[dict]:
    """Walk the graph nodes, find Dense (MatMul + BiasAdd, possibly fused), recover
    activation and the kernel/bias name pair. Returns layers in topological order.
    """
    # Map fused-matmul op nodes -> their fused_ops list (BiasAdd, optional activation).
    layers: list[dict] = []
    for node in topology["node"]:
        op = node.get("op", "")
        if op not in ("_FusedMatMul", "MatMul"):
            continue
        inputs = node.get("input", [])
        # Inputs typically: [input_tensor, kernel_var, bias_var(if fused)]
        if len(inputs) < 2:
            continue
        kernel_name = inputs[1]
        bias_name = inputs[2] if len(inputs) > 2 else None
        if kernel_name not in weights:
            continue
        # Activation: read fused_ops list. Base64-encoded strings in `s`.
        fused_ops_attr = node.get("attr", {}).get("fused_ops", {}).get("list", {}).get("s", [])
        fused_ops = [_b64_decode(s) for s in fused_ops_attr]
        if "Relu" in fused_ops:
            act = "relu"
        elif "Sigmoid" in fused_ops:
            act = "sigmoid"
        elif "Tanh" in fused_ops:
            act = "tanh"
        else:
            act = "linear"
        layers.append(
            {
                "name": node["name"],
                "kernel_name": kernel_name,
                "bias_name": bias_name,
                "activation": act,
            }
        )

    # Heuristic for final activation: look for Softmax / Sigmoid Identity-tail ops
    # not picked up in fused_ops.
    for node in topology["node"]:
        if node.get("op") == "Softmax":
            if layers:
                layers[-1]["activation"] = "softmax"
        elif node.get("op") == "Sigmoid" and layers:
            # Trailing post-Dense sigmoid (not fused into the MatMul).
            if layers[-1]["activation"] == "linear":
                layers[-1]["activation"] = "sigmoid"
    return layers


def load_tfjs_head(model_dir: str | Path) -> tf.keras.Model:
    """Load a TFJS graph-model head as a Keras Sequential of Dense layers."""
    model_dir = Path(model_dir)
    spec = json.loads((model_dir / "model.json").read_text())
    if spec.get("format") != "graph-model":
        raise ValueError(f"Expected format=graph-model in {model_dir}/model.json")

    weights = _parse_weights(spec["weightsManifest"], _read_all_shards(model_dir, spec["weightsManifest"]))
    layers_meta = _infer_dense_layers(spec["modelTopology"], weights)

    if not layers_meta:
        raise ValueError(f"No Dense layers detected in {model_dir}/model.json")

    # Build Keras Sequential. First layer's input shape from kernel's first dim.
    first_kernel = weights[layers_meta[0]["kernel_name"]]
    input_dim = first_kernel.shape[0]

    keras_layers = []
    for i, meta in enumerate(layers_meta):
        kernel = weights[meta["kernel_name"]]
        bias = weights[meta["bias_name"]] if meta["bias_name"] in weights else None
        units = kernel.shape[1]
        dense = tf.keras.layers.Dense(
            units,
            activation=meta["activation"],
            use_bias=bias is not None,
            input_shape=(input_dim,) if i == 0 else None,
            name=f"dense_{i}",
        )
        keras_layers.append(dense)
        input_dim = units

    model = tf.keras.Sequential(keras_layers)
    # Build the model to instantiate weights.
    model.build((None, first_kernel.shape[0]))

    # Assign weights in order.
    for dense, meta in zip(keras_layers, layers_meta):
        kernel = weights[meta["kernel_name"]]
        if meta["bias_name"] and meta["bias_name"] in weights:
            dense.set_weights([kernel, weights[meta["bias_name"]]])
        else:
            dense.set_weights([kernel])

    return model


def load_prod_model(prod_root: str | Path, current_oracle_list: list[str] | None = None) -> "ProdModel":
    """Load every head under prod_root/<head_name>/model.json into a ProdModel
    wrapper that mimics CubeCobraMLSystem's `.draft / .deck_build / .recommend`
    interface so dashboard pages don't have to branch on architecture.

    If `current_oracle_list` is provided (recommended — pass
    `train/oracleDict.json` content), the wrapper auto-detects index-space
    mismatches between prod's `indexToOracleMap.json` and the current
    pipeline and remaps inputs/outputs transparently. When the two index
    maps are identical, the remap is a no-op and incurs no overhead.
    """
    prod_root = Path(prod_root)
    heads = {}
    for child in sorted(prod_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "model.json").exists():
            continue
        try:
            heads[child.name] = load_tfjs_head(child)
            print(f"[ok] {child.name}: {heads[child.name].count_params():,} params")
        except Exception as e:
            print(f"[skip] {child.name}: {e}")
    return ProdModel(heads, prod_root=prod_root, current_oracle_list=current_oracle_list)


class ProdModel:
    """Wraps a dict of prod TFJS heads behind the CubeCobraMLSystem API.

    The prod model predates the additive-perturbation architecture: its draft
    head concatenates pool_emb (128) with cube_ctx_emb (32) to form a 160-d
    input, and its deck_build head takes only pool_emb (128). This wrapper
    encapsulates those differences so callers can pretend prod is current.

    Also handles **index-space remapping** between prod's
    `indexToOracleMap.json` (frozen at prod training time) and the current
    pipeline's `train/oracleDict.json`. If the user reprocessed `raw_data`
    after prod was exported, the same integer index can mean different cards
    in the two spaces, and we'd silently mix them up. The remapping is built
    once at load and is a no-op when the two index maps match (the common
    case).

    Pass `prod_root` so we can read `indexToOracleMap.json` from it, and
    `current_oracle_list` (the list from `train/oracleDict.json`) for the
    current-space ordering.
    """

    def __init__(self, heads: dict, prod_root: Path | None = None, current_oracle_list: list[str] | None = None):
        self.heads = heads
        # Raw heads — internal use only. Their input space is prod's.
        self._encoder_raw = heads.get("encoder")
        self._cube_ctx_enc_raw = heads.get("cube_context_encoder")
        # Other heads receive embeddings, not card vectors, so no remap.
        self.cube_decoder = heads.get("cube_decoder")
        self.draft_decoder = heads.get("draft_decoder")
        self.deck_build_decoder = heads.get("deck_build_decoder")
        self.correlation_decoder = heads.get("correlation_decoder")

        self._needs_remap = False
        self._cur_to_prod: np.ndarray | None = None  # cur_idx → prod_idx, -1 if missing
        self._prod_to_cur: np.ndarray | None = None  # prod_idx → cur_idx, -1 if missing
        self._prod_num_cards = 0

        if prod_root is not None and current_oracle_list is not None:
            self._build_remap(Path(prod_root), current_oracle_list)

    def _build_remap(self, prod_root: Path, current_oracle_list: list[str]) -> None:
        map_path = prod_root / "indexToOracleMap.json"
        if not map_path.is_file():
            return
        raw = json.loads(map_path.read_text())
        prod_list = [raw[str(i)] for i in range(len(raw))]
        self._prod_num_cards = len(prod_list)

        # Identical → leave remap off, cheap path stays.
        if prod_list == current_oracle_list:
            return

        prod_idx_by_oracle = {o: i for i, o in enumerate(prod_list)}
        cur_idx_by_oracle = {o: i for i, o in enumerate(current_oracle_list)}

        self._cur_to_prod = np.full(len(current_oracle_list), -1, dtype=np.int64)
        for i, o in enumerate(current_oracle_list):
            j = prod_idx_by_oracle.get(o, -1)
            self._cur_to_prod[i] = j

        self._prod_to_cur = np.full(len(prod_list), -1, dtype=np.int64)
        for j, o in enumerate(prod_list):
            self._prod_to_cur[j] = cur_idx_by_oracle.get(o, -1)

        n_dropped = int((self._cur_to_prod == -1).sum())
        n_extra = int((self._prod_to_cur == -1).sum())
        print(
            f"[ProdModel] index remap enabled: "
            f"{n_dropped} current cards have no prod equivalent, "
            f"{n_extra} prod cards have no current equivalent."
        )
        self._needs_remap = True

    # ----- public callable views that expect CURRENT-space inputs -----------

    def encoder(self, x):
        """Encode current-space card vector(s) → embedding."""
        x_prod = self._cur_to_prod_vec(np.asarray(x))
        return np.asarray(self._encoder_raw(x_prod))

    def cube_context_encoder(self, x):
        """Encode current-space cube context vector(s) → ctx embedding."""
        x_prod = self._cur_to_prod_vec(np.asarray(x))
        return np.asarray(self._cube_ctx_enc_raw(x_prod))

    # ----- index-space translation helpers ----------------------------------

    def _cur_to_prod_vec(self, v_cur: np.ndarray) -> np.ndarray:
        """Map a [..., num_cards_current] one-hot/multi-hot to prod-space."""
        if not self._needs_remap:
            return v_cur
        out_shape = list(v_cur.shape)
        out_shape[-1] = self._prod_num_cards
        out = np.zeros(out_shape, dtype=v_cur.dtype)
        # Only positions whose mapping is >= 0 get copied.
        valid = self._cur_to_prod >= 0
        out[..., self._cur_to_prod[valid]] = v_cur[..., valid]
        return out

    def _prod_to_cur_vec(self, v_prod: np.ndarray) -> np.ndarray:
        """Map a [..., num_cards_prod] output back to current-space."""
        if not self._needs_remap:
            return v_prod
        out_shape = list(v_prod.shape)
        out_shape[-1] = len(self._cur_to_prod)
        out = np.zeros(out_shape, dtype=v_prod.dtype)
        # current_idx i gets value from prod index cur_to_prod[i] (if valid).
        valid = self._cur_to_prod >= 0
        out[..., valid] = v_prod[..., self._cur_to_prod[valid]]
        return out

    # CubeCobraMLSystem-style public methods ---------------------------------
    # Inputs/outputs are in CURRENT index space. We remap on the way in/out
    # if the prod index map differs from the current one.

    def draft(self, pools, packs, cube_context, training=False):
        pools_p = self._cur_to_prod_vec(np.asarray(pools))
        packs_p = self._cur_to_prod_vec(np.asarray(packs))
        ctx_p = self._cur_to_prod_vec(np.asarray(cube_context))
        pool_emb = np.asarray(self._encoder_raw(pools_p))
        ctx_emb = np.asarray(self._cube_ctx_enc_raw(ctx_p))
        combined = np.concatenate([pool_emb, ctx_emb], axis=-1)
        logits = np.asarray(self.draft_decoder(combined))
        # Mask + softmax in prod-space using the prod-space pack.
        mask = 1e9 * (1 - packs_p)
        z = logits * packs_p - mask
        z = z - z.max(axis=-1, keepdims=True)
        probs_prod = np.exp(z)
        probs_prod /= probs_prod.sum(axis=-1, keepdims=True)
        return self._prod_to_cur_vec(probs_prod)

    def deck_build(self, pools, cube_context, training=False):
        pools_p = self._cur_to_prod_vec(np.asarray(pools))
        pool_emb = np.asarray(self._encoder_raw(pools_p))
        probs_prod = np.asarray(self.deck_build_decoder(pool_emb))
        return self._prod_to_cur_vec(probs_prod)

    def recommend(self, cubes, training=False):
        cubes_p = self._cur_to_prod_vec(np.asarray(cubes))
        emb = np.asarray(self._encoder_raw(cubes_p))
        probs_prod = np.asarray(self.cube_decoder(emb))
        return self._prod_to_cur_vec(probs_prod)


if __name__ == "__main__":
    import sys

    from backend.lib.config import prod_model_path

    prod = sys.argv[1] if len(sys.argv) > 1 else prod_model_path()
    if prod is None:
        raise SystemExit("pass a prod-model dir or set CUBECOBRA_PROD_MODEL")
    load_prod_model(str(prod))
