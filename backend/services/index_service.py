"""Reverse-lookup indexes for the val split.

For each card idx, we want to answer four questions fast:
  - which cubes contain it?
  - which decks include it in the mainboard?
  - which val picks have it in the pack?
  - which val picks did a human take it?

Naive scans over 329 k picks per request are too slow for an interactive UI;
we precompute the inverted indexes on first access and persist them to
`data/.dashboard_cache/*.npz` so cold-starts stay under a few seconds.

Memory budget (val split, ~37k vocab):
  card→picks(in-pack)        ~470 MB
  card→picks(picked-by-human) ~10 MB
  card→cubes                  ~200 MB
  card→decks                   ~100 MB
  total                        ~770 MB

Indexes are stored as numpy arrays for compact representation (int32). Each
lookup returns a numpy ndarray; routers paginate on top.
"""

from __future__ import annotations

import threading
from functools import cached_property

import numpy as np

from backend.lib.config import DATA_DIR
from backend.services.val_data import val_data


CACHE_DIR = DATA_DIR / ".dashboard_cache"


def _save(name: str, idx: list[np.ndarray]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Save as a sparse representation: concatenate + offsets.
    lens = np.array([len(a) for a in idx], dtype=np.int64)
    concat = np.concatenate(idx) if idx and any(len(a) for a in idx) else np.array([], dtype=np.int32)
    np.savez(CACHE_DIR / f"{name}.npz", lens=lens, concat=concat)


def _load(name: str) -> list[np.ndarray] | None:
    p = CACHE_DIR / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    lens = z["lens"]
    concat = z["concat"]
    out: list[np.ndarray] = []
    start = 0
    for n in lens:
        out.append(concat[start : start + int(n)])
        start += int(n)
    return out


class IndexService:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _build_pack_indexes(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        val_data.ensure_loaded()
        from backend.lib.cards import num_cards

        N = num_cards()
        in_pack: list[list[int]] = [[] for _ in range(N)]
        by_human: list[list[int]] = [[] for _ in range(N)]
        for i, r in enumerate(val_data.picks):
            for c in r["pack"]:
                in_pack[c].append(i)
            by_human[r["pick"]].append(i)
        return (
            [np.asarray(x, dtype=np.int32) for x in in_pack],
            [np.asarray(x, dtype=np.int32) for x in by_human],
        )

    def _build_cube_index(self) -> list[np.ndarray]:
        val_data.ensure_loaded()
        from backend.lib.cards import num_cards

        N = num_cards()
        out: list[list[int]] = [[] for _ in range(N)]
        for i, cube in enumerate(val_data.cubes):
            for c in cube:
                out[c].append(i)
        return [np.asarray(x, dtype=np.int32) for x in out]

    def _build_deck_index(self) -> list[np.ndarray]:
        val_data.ensure_loaded()
        from backend.lib.cards import num_cards

        N = num_cards()
        out: list[list[int]] = [[] for _ in range(N)]
        for i, d in enumerate(val_data.decks):
            mb = d.get("mainboard") or []
            for c in mb:
                out[c].append(i)
        return [np.asarray(x, dtype=np.int32) for x in out]

    @cached_property
    def card_to_picks_in_pack(self) -> list[np.ndarray]:
        with self._lock:
            cached = _load("card_to_picks_in_pack")
            if cached is not None:
                # We may have also persisted by_human as a sibling; load it now.
                return cached
            in_pack, by_human = self._build_pack_indexes()
            _save("card_to_picks_in_pack", in_pack)
            _save("card_to_picks_human", by_human)
            self.__dict__["card_to_picks_human"] = by_human
            return in_pack

    @cached_property
    def card_to_picks_human(self) -> list[np.ndarray]:
        with self._lock:
            cached = _load("card_to_picks_human")
            if cached is not None:
                return cached
            in_pack, by_human = self._build_pack_indexes()
            _save("card_to_picks_in_pack", in_pack)
            _save("card_to_picks_human", by_human)
            self.__dict__["card_to_picks_in_pack"] = in_pack
            return by_human

    @cached_property
    def card_to_cubes(self) -> list[np.ndarray]:
        with self._lock:
            cached = _load("card_to_cubes")
            if cached is not None:
                return cached
            out = self._build_cube_index()
            _save("card_to_cubes", out)
            return out

    @cached_property
    def card_to_decks(self) -> list[np.ndarray]:
        with self._lock:
            cached = _load("card_to_decks")
            if cached is not None:
                return cached
            out = self._build_deck_index()
            _save("card_to_decks", out)
            return out


index_service = IndexService()
