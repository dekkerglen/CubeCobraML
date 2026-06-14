"""Model registry — LRU cache of loaded checkpoints.

The first request for a given checkpoint pays the ~3-5s Keras load cost; the
result stays in memory until evicted. Default capacity is 2 so the user can
A/B compare without re-loading per request.

Thread-safe (FastAPI's threadpool runs sync endpoints in parallel; the same
key arriving twice serialises behind the lock and only loads once).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from backend.lib import ckpt as ckpt_lib
from backend.lib.ckpt import CkptEntry


class ModelRegistry:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()
        self._per_key_lock: dict[str, threading.Lock] = {}

    def list(self) -> list[CkptEntry]:
        return ckpt_lib.all_selectable()

    def find(self, key: str) -> CkptEntry | None:
        for e in self.list():
            if e.key == key:
                return e
        return None

    def is_loaded(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def get(self, key: str) -> object:
        """Return the loaded model object for `key`, loading if needed.

        Raises KeyError if the key isn't selectable.
        """
        # Fast path: already cached, refresh LRU order.
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            kl = self._per_key_lock.setdefault(key, threading.Lock())

        # Serialise concurrent loaders of the SAME key.
        with kl:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]

            entry = self.find(key)
            if entry is None:
                raise KeyError(f"unknown checkpoint key: {key!r}")
            obj = ckpt_lib.load_entry(entry)

            with self._lock:
                self._cache[key] = obj
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    evicted, _ = self._cache.popitem(last=False)
                    # Drop per-key lock so it doesn't leak.
                    self._per_key_lock.pop(evicted, None)
            return obj


registry = ModelRegistry()
