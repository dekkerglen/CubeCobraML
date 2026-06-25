"""Val deck browsing endpoints."""

from __future__ import annotations

import random

from fastapi import APIRouter, HTTPException, Query

from backend.models import DeckOut
from backend.services.val_data import val_data


router = APIRouter(prefix="/decks", tags=["decks"])


@router.get("", response_model=list[DeckOut])
def list_decks(limit: int = Query(50, ge=1, le=500)) -> list[DeckOut]:
    val_data.ensure_loaded()
    return [
        DeckOut(
            idx=i,
            mainboard=d.get("mainboard") or [],
            sideboard=d.get("sideboard") or [],
            cube_cards=d.get("cube_cards") or [],
        )
        for i, d in enumerate(val_data.decks[:limit])
    ]


@router.get("/random/one", response_model=DeckOut)
def random_one() -> DeckOut:
    val_data.ensure_loaded()
    n = len(val_data.decks)
    if not n:
        raise HTTPException(404, "no decks loaded")
    return by_idx(random.randrange(n))


@router.get("/{idx}", response_model=DeckOut)
def by_idx(idx: int) -> DeckOut:
    val_data.ensure_loaded()
    n = len(val_data.decks)
    if idx < 0 or idx >= n:
        raise HTTPException(404, "deck idx out of range")
    d = val_data.decks[idx]
    return DeckOut(
        idx=idx,
        mainboard=d.get("mainboard") or [],
        sideboard=d.get("sideboard") or [],
        cube_cards=d.get("cube_cards") or [],
    )
