"""Thin uniform wrappers over the model heads.

`CubeCobraMLSystem` exposes `.draft / .deck_build / .recommend / .encoder`,
all of which take only pool/pack card-set vectors — the model is context-free.
"""

from __future__ import annotations

import numpy as np

from .cards import num_cards


# ----- one-hot helpers -------------------------------------------------------


def one_hot(indices) -> np.ndarray:
    v = np.zeros(num_cards(), dtype=np.float32)
    if indices:
        v[list(indices)] = 1.0
    return v


def empty_vec() -> np.ndarray:
    return np.zeros(num_cards(), dtype=np.float32)


def _as_2d(v: np.ndarray) -> np.ndarray:
    return v[None, :].astype(np.float32) if v.ndim == 1 else v.astype(np.float32)


# ----- per-head dispatch (current + prod use the same API) -------------------


def encode(obj, vec: np.ndarray) -> np.ndarray:
    """Project a card-set one-hot vector through the encoder to an embedding."""
    out = obj.encoder(_as_2d(vec))
    if hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out)[0]


def draft_logits(obj, pool: np.ndarray, pack: np.ndarray) -> np.ndarray:
    """Pack-masked softmax over cards."""
    result = obj.draft(_as_2d(pool), _as_2d(pack), training=False)
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)[0]


def draft_logits_batch(obj, pools: np.ndarray, packs: np.ndarray) -> np.ndarray:
    """Batched pack-masked softmax over cards. Inputs are (batch, num_cards)."""
    result = obj.draft(pools.astype(np.float32), packs.astype(np.float32), training=False)
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)


def deck_build(obj, pool: np.ndarray) -> np.ndarray:
    """Per-card mainboard inclusion probabilities."""
    result = obj.deck_build(_as_2d(pool), training=False)
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)[0]


def cube_recon(obj, cube_vec: np.ndarray) -> np.ndarray:
    """Cube reconstruction probabilities (autoencoder output)."""
    result = obj.recommend(_as_2d(cube_vec), training=False)
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)[0]


# ----- card embeddings -------------------------------------------------------


def card_embedding(obj, card_idx: int) -> np.ndarray:
    """Single-card embedding (one-hot of just that card through encoder)."""
    v = empty_vec()
    v[card_idx] = 1.0
    return encode(obj, v)


def all_card_embeddings(obj, batch: int = 256) -> np.ndarray:
    """Embed every card with its own one-hot. Returns (num_cards, emb_dim).

    Heavy. Pages should cache the result via `st.cache_data`.
    """
    N = num_cards()
    # Probe one card to discover the embedding dim (current=128, prod=128).
    probe = card_embedding(obj, 0)
    out = np.zeros((N, probe.shape[0]), dtype=np.float32)
    out[0] = probe
    for i in range(1, N, batch):
        j = min(i + batch, N)
        x = np.zeros((j - i, N), dtype=np.float32)
        for k in range(j - i):
            x[k, i + k] = 1.0
        emb = obj.encoder(x)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        out[i:j] = np.asarray(emb)
    return out


def cosine_neighbors(embeddings: np.ndarray, card_idx: int, k: int = 20):
    """Top-k nearest cards to card_idx by cosine similarity, excluding self."""
    q = embeddings[card_idx]
    qn = q / (np.linalg.norm(q) + 1e-9)
    en = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sims = en @ qn
    sims[card_idx] = -2.0
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
