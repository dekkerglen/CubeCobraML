"""Per-card statistics (train pick/cube counts + val equivalents computed on
the fly from val_data)."""

from __future__ import annotations

import json
from collections import Counter
from functools import cached_property

from backend.lib.config import TRAIN_DIR
from backend.services.val_data import val_data


class CardStatsService:
    @cached_property
    def pick_count(self) -> dict[int, int]:
        p = TRAIN_DIR / "pickCount.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text())
        if isinstance(raw, list):
            return {i: int(v) for i, v in enumerate(raw)}
        return {int(k): int(v) for k, v in raw.items()}

    @cached_property
    def oracle_frequency(self) -> dict[int, int]:
        """Card idx → number of train cubes containing the card."""
        p = TRAIN_DIR / "oracleFrequency.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text())
        if isinstance(raw, list):
            return {i: int(v) for i, v in enumerate(raw)}
        return {int(k): int(v) for k, v in raw.items()}

    @cached_property
    def is_land(self) -> dict[int, bool]:
        p = TRAIN_DIR / "isLand.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text())
        if isinstance(raw, list):
            return {i: bool(v) for i, v in enumerate(raw)}
        return {int(k): bool(v) for k, v in raw.items()}

    @cached_property
    def val_pick_count(self) -> dict[int, int]:
        """Card idx → number of val picks where the human chose this card."""
        val_data.ensure_loaded()
        c: Counter[int] = Counter()
        for rec in val_data.picks:
            c[int(rec["pick"])] += 1
        return dict(c)

    @cached_property
    def val_cube_count(self) -> dict[int, int]:
        """Card idx → number of distinct val cubes containing this card.
        Uses val_data.cubes (canonical val cubes, deduped)."""
        val_data.ensure_loaded()
        c: Counter[int] = Counter()
        for cube in val_data.cubes:
            for idx in set(cube):
                c[int(idx)] += 1
        return dict(c)


card_stats = CardStatsService()
