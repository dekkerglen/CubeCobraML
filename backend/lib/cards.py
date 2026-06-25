"""Card index ↔ name / image / metadata helpers.

The training pipeline uses integer indices (0..num_cards-1) to identify cards.
Mapping back to human-readable names + Scryfall images for the dashboard is a
two-step lookup:

    index → oracle_id (via train/oracleDict.json)
    oracle_id → {name, image, type, cmc, elo} (via raw_data/simpleCardDict.json)

This module also caches a couple of structures for fast UI lookups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import RAW_DATA_DIR, TRAIN_DIR


@dataclass(slots=True)
class CardInfo:
    idx: int
    oracle_id: str
    name: str
    image: str
    image_normal: str  # upgraded from "small/" to "normal/" — bigger zoomed image
    type: str
    cmc: float
    elo: float
    # Phase L: from data/train/cardDictEnriched.json (CubeCobra carddict join).
    # Default-empty when the sidecar isn't present.
    set: str = ""
    set_name: str = ""
    colors: tuple[str, ...] = ()
    color_identity: tuple[str, ...] = ()
    rarity: str = ""

    def is_real(self) -> bool:
        """False for placeholder/custom-card entries with no image."""
        return bool(self.image) and self.oracle_id not in ("custom-card", "voucher")


@lru_cache(maxsize=1)
def _oracle_index() -> list[str]:
    return json.load(open(TRAIN_DIR / "oracleDict.json"))


@lru_cache(maxsize=1)
def _card_dict() -> dict:
    """Prefer the enriched dict (set, colors, rarity, ...) when its sidecar
    exists; fall back to simpleCardDict.json so old data still loads cleanly."""
    enriched = TRAIN_DIR / "cardDictEnriched.json"
    if enriched.is_file():
        return json.load(open(enriched))
    return json.load(open(RAW_DATA_DIR / "simpleCardDict.json"))


@lru_cache(maxsize=1)
def num_cards() -> int:
    return len(_oracle_index())


def card(idx: int) -> CardInfo:
    """Build a CardInfo for the given training index."""
    oracle = _oracle_index()
    cd = _card_dict()
    if idx < 0 or idx >= len(oracle):
        return CardInfo(idx, "", f"<unknown #{idx}>", "", "", "", 0.0, 1200.0)
    oid = oracle[idx]
    raw = cd.get(oid, {})
    img = raw.get("image", "") or ""
    img_normal = img.replace("/small/", "/normal/") if img else ""
    return CardInfo(
        idx=idx,
        oracle_id=oid,
        name=raw.get("name") or oid,
        image=img,
        image_normal=img_normal,
        type=raw.get("type") or "",
        cmc=float(raw.get("cmc") or 0),
        elo=float(raw.get("elo") or 1200),
        set=raw.get("set") or "",
        set_name=raw.get("set_name") or "",
        colors=tuple(raw.get("colors") or ()),
        color_identity=tuple(raw.get("color_identity") or ()),
        rarity=raw.get("rarity") or "",
    )


def cards(indices) -> list[CardInfo]:
    return [card(int(i)) for i in indices]


def name(idx: int) -> str:
    return card(idx).name


@lru_cache(maxsize=1)
def name_to_idx() -> dict[str, int]:
    """Reverse map for typeahead lookup. Case-insensitive lookups should use
    name.lower() against keys() copy."""
    oracle = _oracle_index()
    cd = _card_dict()
    out: dict[str, int] = {}
    for i, oid in enumerate(oracle):
        nm = (cd.get(oid, {}).get("name") or oid).strip()
        if nm:
            out[nm] = i
    return out


def search_by_name(query: str, limit: int = 20) -> list[CardInfo]:
    """Case-insensitive substring search across card names.

    Ranked so the most recognizable / expected match shows up first:
      1. exact (case-insensitive)
      2. starts-with
      3. word-boundary substring
      4. anywhere substring
    Within each tier, sorted by elo descending (so "Lightning Bolt" beats
    "Lightning Bolt Tribal Common" etc.)."""
    q = (query or "").strip().lower()
    if not q:
        return []
    tiers: list[list[CardInfo]] = [[], [], [], []]
    for nm, idx in name_to_idx().items():
        nm_low = nm.lower()
        if nm_low == q:
            tiers[0].append(card(idx))
        elif nm_low.startswith(q):
            tiers[1].append(card(idx))
        elif f" {q}" in f" {nm_low}":
            tiers[2].append(card(idx))
        elif q in nm_low:
            tiers[3].append(card(idx))
    out: list[CardInfo] = []
    for tier in tiers:
        tier.sort(key=lambda c: -c.elo)
        for c in tier:
            out.append(c)
            if len(out) >= limit:
                return out
    return out


def real_card_indices() -> np.ndarray:
    """Indices that correspond to actual Magic cards (not placeholders)."""
    return np.array([i for i in range(num_cards()) if card(i).is_real()], dtype=np.int64)
