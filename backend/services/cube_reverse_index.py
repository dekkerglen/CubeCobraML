"""Reverse lookups from positional val indices back to cube_uuid + draft info.

Built at first access by inverting three on-disk sidecars produced by
`scripts/process_data.js`:

  - `data/test/cube_uuids.json` (Phase S) — list of cube UUIDs positionally
    aligned with `val_data.cubes`. Trivial lookup.
  - `data/test/decks_by_cube_uuid.json` — `{cube_uuid: [val_deck_idx, ...]}`,
    inverted to `{val_deck_idx: cube_uuid}` (~50ms, ~5MB dict).
  - `data/test/draft_sessions.json` — `{draft_id: {cube_uuid, picks: [...]}}`,
    inverted to `{val_pick_idx: (draft_id, step, cube_uuid)}` (~50ms, ~30MB
    dict).

These power the clickable CardDrawer rows (Phase U) — a val pick / deck /
cube positional index resolves to the canonical UUID + draft_id needed to
construct a navigation URL.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from functools import cached_property

from backend.lib.config import TEST_DIR


log = logging.getLogger(__name__)


@dataclass(slots=True)
class PickLocator:
    """Where a positional val pick lives in the draft graph."""

    draft_id: str
    step: int
    cube_uuid: str | None


class CubeReverseIndexService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ = self._cube_uuids
            _ = self._deck_to_cube
            _ = self._pick_to_locator
            self._loaded = True

    # ----- sidecar loaders (cached) ----------------------------------------

    @cached_property
    def _cube_uuids(self) -> list[str]:
        path = TEST_DIR / "cube_uuids.json"
        if not path.is_file():
            log.info("cube_uuids.json missing — val cube reverse lookups unavailable.")
            return []
        return json.loads(path.read_text())

    @cached_property
    def _deck_to_cube(self) -> dict[int, str]:
        path = TEST_DIR / "decks_by_cube_uuid.json"
        if not path.is_file():
            return {}
        forward: dict[str, list[int]] = json.loads(path.read_text())
        inverted: dict[int, str] = {}
        for cube_uuid, idxs in forward.items():
            for idx in idxs:
                inverted[idx] = cube_uuid
        log.info("Deck → cube_uuid reverse index: %d entries.", len(inverted))
        return inverted

    @cached_property
    def _pick_to_locator(self) -> dict[int, PickLocator]:
        path = TEST_DIR / "draft_sessions.json"
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text())
        out: dict[int, PickLocator] = {}
        for draft_id, body in raw.get("sessions", {}).items():
            cube_uuid = body.get("cube_uuid")
            for step, pick_idx in enumerate(body.get("picks", [])):
                out[int(pick_idx)] = PickLocator(
                    draft_id=draft_id,
                    step=step,
                    cube_uuid=cube_uuid,
                )
        log.info("Pick → (draft_id, step, cube_uuid) reverse index: %d entries.", len(out))
        return out

    # ----- queries ---------------------------------------------------------

    def cube_uuid_for_val_cube_idx(self, val_cube_idx: int) -> str | None:
        self.ensure_loaded()
        if 0 <= val_cube_idx < len(self._cube_uuids):
            return self._cube_uuids[val_cube_idx]
        return None

    def cube_uuid_for_val_deck_idx(self, val_deck_idx: int) -> str | None:
        self.ensure_loaded()
        return self._deck_to_cube.get(val_deck_idx)

    def locator_for_val_pick_idx(self, val_pick_idx: int) -> PickLocator | None:
        self.ensure_loaded()
        return self._pick_to_locator.get(val_pick_idx)


cube_reverse_index = CubeReverseIndexService()
