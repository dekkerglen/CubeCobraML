"""Cube directory — UUID, name, owner, and cube card lookup.

Loads `data/cube_directory.json` (produced by `scripts/process_data.js`'s
additive sidecar emission) once at first access. Builds:
  - uuid → entry            for O(1) lookups by stable UUID
  - tokenized name index    for fuzzy substring search by name/owner

Card lists for a single cube are read lazily from the appropriate batch in
`data/train/cubes/{NNNN}.json` to keep startup memory small (~58 MB JSON
loaded; ~250 MB Python dicts).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from functools import cached_property

from backend.lib.config import DATA_DIR, TRAIN_DIR


CUBES_PER_BATCH = 10000  # matches WRITE_BATCH_SIZE in process_data.js


@dataclass(slots=True)
class CubeEntry:
    cube_uuid: str
    name: str
    owner: str
    owner_id: str
    image_uri: str
    card_count: int
    has_val_drafts: bool
    train_idx: int


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class CubeDirectoryService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ = self.entries
            _ = self._by_uuid
            _ = self._by_token
            self._loaded = True

    @cached_property
    def entries(self) -> list[CubeEntry]:
        path = DATA_DIR / "cube_directory.json"
        if not path.is_file():
            return []
        raw = json.loads(path.read_text())
        return [CubeEntry(**r) for r in raw]

    @cached_property
    def _by_uuid(self) -> dict[str, CubeEntry]:
        return {e.cube_uuid: e for e in self.entries}

    @cached_property
    def _by_token(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            for tok in {*_tokenize(e.name), *_tokenize(e.owner)}:
                idx.setdefault(tok, []).append(i)
        return idx

    @cached_property
    def _by_owner(self) -> dict[str, list[int]]:
        """owner (lower-cased exact match) → entry indices. Phase V replaces
        the old "search and filter client-side" My Cubes shelf with a real
        index — name collisions across owners no longer leak."""
        idx: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            if e.owner:
                idx.setdefault(e.owner.lower(), []).append(i)
        return idx

    # ----- queries ----------------------------------------------------------

    def get(self, cube_uuid: str) -> CubeEntry | None:
        self.ensure_loaded()
        return self._by_uuid.get(cube_uuid)

    def search(self, q: str, split: str = "all", limit: int = 25) -> list[CubeEntry]:
        """Token-AND substring search across name + owner.

        `split`: "train" (all), "val" (has_val_drafts only), or "all". The
        val/test cubes are a strict subset of train cubes in our pipeline.
        """
        self.ensure_loaded()
        toks = _tokenize(q)
        if not toks:
            return []

        # AND-merge: candidate sets per token, intersect.
        candidate_sets: list[set[int]] = []
        for t in toks:
            hits: set[int] = set()
            for tok, ids in self._by_token.items():
                if t in tok:
                    hits.update(ids)
            if not hits:
                return []
            candidate_sets.append(hits)
        merged = candidate_sets[0]
        for s in candidate_sets[1:]:
            merged &= s
            if not merged:
                return []

        results = [self.entries[i] for i in merged]
        if split == "val":
            results = [e for e in results if e.has_val_drafts]
        # Rank by: exact name prefix match > shorter name > val cubes first
        q_lower = q.lower()
        results.sort(
            key=lambda e: (
                not e.name.lower().startswith(q_lower),
                not e.has_val_drafts,
                len(e.name),
            )
        )
        return results[:limit]

    def by_owner(self, owner: str, limit: int = 50, offset: int = 0) -> tuple[list[CubeEntry], int]:
        """Paginated exact-match (case-insensitive) lookup of cubes by owner.

        Sorted by `has_val_drafts` first (matches the user mental model of
        "show the ones I can actually analyze first"), then by name length
        ascending so the canonical / shorter-named cube floats to the top.
        Returns (page, total) so the caller can render pagination.
        """
        self.ensure_loaded()
        ids = self._by_owner.get(owner.strip().lower(), [])
        total = len(ids)
        rows = [self.entries[i] for i in ids]
        rows.sort(key=lambda e: (not e.has_val_drafts, len(e.name), e.name.lower()))
        return rows[offset : offset + limit], total

    def cards_for(self, cube_uuid: str) -> list[int] | None:
        """Lazily read the cube's card list from the train/cubes/ batch."""
        entry = self.get(cube_uuid)
        if entry is None:
            return None
        batch_idx = entry.train_idx // CUBES_PER_BATCH
        within = entry.train_idx % CUBES_PER_BATCH
        batch_path = TRAIN_DIR / "cubes" / f"{batch_idx:04d}.json"
        if not batch_path.is_file():
            return None
        batch = json.loads(batch_path.read_text())
        if within >= len(batch):
            return None
        return batch[within]


cube_directory = CubeDirectoryService()
