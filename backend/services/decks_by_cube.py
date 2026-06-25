"""Index of deck indices grouped by cube UUID, per split.

Sidecar files (`decks_by_cube_uuid.json`) live under `data/{train,test}/`.
The canonical pipeline emits both; `--rebuild-drafts` also emits both via the
val deck-matching pass. Each file is ~50–75 MB JSON — fully boot-loaded.
"""

from __future__ import annotations

import json
import threading
from functools import cached_property
from pathlib import Path
from typing import Literal

from backend.lib.config import TEST_DIR, TRAIN_DIR


class DecksByCubeService:
    def __init__(self, dir_path: Path) -> None:
        self._dir = dir_path
        self._lock = threading.Lock()
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ = self.by_cube
            self._loaded = True

    @cached_property
    def by_cube(self) -> dict[str, list[int]]:
        path = self._dir / "decks_by_cube_uuid.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text())

    def get(self, cube_uuid: str) -> list[int]:
        self.ensure_loaded()
        return self.by_cube.get(cube_uuid, [])


val_decks_by_cube = DecksByCubeService(TEST_DIR)
train_decks_by_cube = DecksByCubeService(TRAIN_DIR)


# Backward-compatible alias — existing callers (Phase J/M cubes router) use
# `decks_by_cube` as the val singleton. Keep the name pointing at val.
decks_by_cube = val_decks_by_cube


def get_by_split(cube_uuid: str, split: Literal["train", "val", "all"] = "val") -> list[tuple[int, str]]:
    """Return a list of (deck_idx, split_tag) for a cube. Train idxs and val
    idxs occupy disjoint integer spaces (each starts from 0 in its own split),
    so callers must use the tag to look up the right deck."""
    out: list[tuple[int, str]] = []
    if split in ("val", "all"):
        for idx in val_decks_by_cube.get(cube_uuid):
            out.append((idx, "val"))
    if split in ("train", "all"):
        for idx in train_decks_by_cube.get(cube_uuid):
            out.append((idx, "train"))
    return out
