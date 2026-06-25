"""Card-embedding cache per checkpoint.

`all_card_embeddings` projects every card through the encoder (one-hot per
card). Heavy — ~5s on CPU. We cache the result per ckpt key in memory.

Used by `/api/cards/:idx/neighbors` for cosine nearest-neighbor lookup in the
encoder space.
"""

from __future__ import annotations

import threading

import numpy as np

from backend.lib.inference import all_card_embeddings, cosine_neighbors
from backend.services.model_registry import registry


class EmbeddingService:
    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def get(self, ckpt: str) -> np.ndarray:
        with self._lock:
            if ckpt in self._cache:
                return self._cache[ckpt]
        obj = registry.get(ckpt)
        emb = all_card_embeddings(obj)
        with self._lock:
            self._cache[ckpt] = emb
        return emb

    def neighbors(self, ckpt: str, card_idx: int, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
        emb = self.get(ckpt)
        return cosine_neighbors(emb, card_idx, k=k)


embedding_service = EmbeddingService()
