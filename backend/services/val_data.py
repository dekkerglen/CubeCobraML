"""Val-data preloader.

Loads every val pick / deck / cube / cubeInstance into memory at startup
(~150 MB total) so all downstream endpoints are pure RAM reads. Triggered
on first access — `app.py`'s lifespan handler warms it during boot.
"""

from __future__ import annotations

import json
import threading
from functools import cached_property

from backend.lib.config import TEST_DIR


class ValDataService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ = self.picks
            _ = self.decks
            _ = self.cubes
            _ = self.cube_instances
            self._loaded = True

    # ------ collections (lazy via cached_property) --------------------------

    @cached_property
    def picks(self) -> list[dict]:
        out: list[dict] = []
        picks_dir = TEST_DIR / "picks"
        ci_dir = TEST_DIR / "cubeInstances"
        if not picks_dir.is_dir():
            return out
        for fname in sorted(picks_dir.iterdir()):
            cubes = []
            ci_path = ci_dir / fname.name
            if ci_path.exists():
                cubes = json.loads(ci_path.read_text())
            for rec in json.loads(fname.read_text()):
                cci = rec.get("cube_cards_idx")
                cube_cards = cubes[cci] if cci is not None and cci < len(cubes) else []
                out.append(
                    {
                        "pool": rec["pool"],
                        "pack": rec["pack"],
                        "pick": rec["pick"],
                        "cube_ctx": cube_cards,
                        "cube_ctx_idx": cci,
                    }
                )
        return out

    @cached_property
    def decks(self) -> list[dict]:
        out: list[dict] = []
        decks_dir = TEST_DIR / "decks"
        if not decks_dir.is_dir():
            return out
        for fname in sorted(decks_dir.iterdir()):
            out.extend(json.loads(fname.read_text()))
        return out

    @cached_property
    def cubes(self) -> list[list[int]]:
        out: list[list[int]] = []
        cubes_dir = TEST_DIR / "cubes"
        if not cubes_dir.is_dir():
            return out
        for fname in sorted(cubes_dir.iterdir()):
            out.extend(json.loads(fname.read_text()))
        return out

    @cached_property
    def cube_instances(self) -> list[list[int]]:
        out: list[list[int]] = []
        ci_dir = TEST_DIR / "cubeInstances"
        if not ci_dir.is_dir():
            return out
        for fname in sorted(ci_dir.iterdir()):
            out.extend(json.loads(fname.read_text()))
        return out


val_data = ValDataService()
