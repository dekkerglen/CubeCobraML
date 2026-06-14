"""Card search + detail + reverse-lookup endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.lib import cards as cards_lib
from backend.models import CardLookupItem, CardOut, CardReverseLookupOut, CardSearchOut
from backend.services.card_stats import card_stats
from backend.services.cube_reverse_index import cube_reverse_index
from backend.services.embedding_service import embedding_service
from backend.services.index_service import index_service
from backend.services.model_registry import registry


router = APIRouter(prefix="/cards", tags=["cards"])


def _to_card_out(idx: int) -> CardOut:
    c = cards_lib.card(idx)
    return CardOut(
        idx=c.idx,
        oracle_id=c.oracle_id,
        name=c.name,
        image=c.image,
        image_normal=c.image_normal,
        type=c.type,
        cmc=c.cmc,
        elo=c.elo,
        pick_count=card_stats.pick_count.get(idx, 0),
        cube_count=card_stats.oracle_frequency.get(idx, 0),
        val_pick_count=card_stats.val_pick_count.get(idx, 0),
        val_cube_count=card_stats.val_cube_count.get(idx, 0),
        is_land=card_stats.is_land.get(
            idx, "land" in (c.type or "").lower() and "creature" not in (c.type or "").lower()
        ),
        set=c.set,
        set_name=c.set_name,
        colors=list(c.colors),
        color_identity=list(c.color_identity),
        rarity=c.rarity,
    )


@router.get("/search", response_model=CardSearchOut)
def search(
    q: str = Query("", description="Name substring (case-insensitive)"),
    type: str = Query("", description="Filter by substring of card type line"),
    cmc_min: float | None = None,
    cmc_max: float | None = None,
    elo_min: float | None = None,
    elo_max: float | None = None,
    sets: str | None = Query(None, description="Comma-separated set codes (e.g. sos,mh3)"),
    colors: str | None = Query(None, description="Comma-separated color letters; matches color_identity"),
    rarity: str | None = Query(None, description="Comma-separated rarities (common,uncommon,rare,mythic)"),
    sort: str = Query("elo_desc", description="elo_desc | elo_asc | name | cmc_asc | cmc_desc | pick_desc"),
    limit: int = Query(60, ge=1, le=500),
) -> CardSearchOut:
    """Ranked card search with light filters.

    If `q` is empty, walks the full vocabulary; otherwise leverages tiered
    name matching from `lib.cards.search_by_name`.
    """
    pick = card_stats.pick_count
    set_filter = {s.strip().lower() for s in sets.split(",")} if sets else None
    rarity_filter = {r.strip().lower() for r in rarity.split(",")} if rarity else None
    color_filter = {c.strip().upper() for c in colors.split(",")} if colors else None

    def passes(c) -> bool:
        if type and type.lower() not in (c.type or "").lower():
            return False
        if cmc_min is not None and c.cmc < cmc_min:
            return False
        if cmc_max is not None and c.cmc > cmc_max:
            return False
        if elo_min is not None and c.elo < elo_min:
            return False
        if elo_max is not None and c.elo > elo_max:
            return False
        if set_filter is not None and (c.set or "").lower() not in set_filter:
            return False
        if rarity_filter is not None and (c.rarity or "").lower() not in rarity_filter:
            return False
        if color_filter is not None:
            ci = set(c.color_identity or ())
            # Treat empty color_identity as colorless ("C"). Match if any
            # selected color is in the card's identity, or if "C" is selected
            # and the card is colorless.
            if "C" in color_filter and not ci:
                pass
            elif ci & color_filter:
                pass
            else:
                return False
        return True

    if q:
        candidates = [c for c in cards_lib.search_by_name(q, limit=4000) if passes(c)]
    else:
        candidates = []
        for i in cards_lib.real_card_indices():
            ci = cards_lib.card(int(i))
            if not ci.is_real():
                continue
            if passes(ci):
                candidates.append(ci)

    key_map = {
        "elo_desc": (lambda c: -c.elo),
        "elo_asc": (lambda c: c.elo),
        "name": (lambda c: c.name.lower()),
        "cmc_asc": (lambda c: (c.cmc, -c.elo)),
        "cmc_desc": (lambda c: (-c.cmc, -c.elo)),
        "pick_desc": (lambda c: -pick.get(c.idx, 0)),
    }
    candidates.sort(key=key_map.get(sort, key_map["elo_desc"]))
    total = len(candidates)
    items = [_to_card_out(c.idx) for c in candidates[:limit]]
    return CardSearchOut(total=total, items=items)


@router.get("/many")
def many(ids: str = Query(..., description="comma-separated card indices")) -> list[CardOut]:
    """Batch-fetch CardOut for many indices. Used by deck/pool views that need
    metadata for every card at once."""
    try:
        idxs = [int(x) for x in ids.split(",") if x]
    except ValueError:
        raise HTTPException(400, "invalid ids")
    N = cards_lib.num_cards()
    out: list[CardOut] = []
    for i in idxs:
        if 0 <= i < N:
            out.append(_to_card_out(i))
    return out


@router.get("/{idx}", response_model=CardOut)
def detail(idx: int) -> CardOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    return _to_card_out(idx)


# ----- reverse lookups ------------------------------------------------------


def _paginate(arr, page: int, per_page: int) -> tuple[list[int], int]:
    total = int(len(arr))
    start = page * per_page
    end = start + per_page
    return [int(x) for x in arr[start:end]], total


def _resolve_cube_items(positional: list[int]) -> list[CardLookupItem]:
    return [CardLookupItem(idx=i, cube_uuid=cube_reverse_index.cube_uuid_for_val_cube_idx(i)) for i in positional]


def _resolve_deck_items(positional: list[int]) -> list[CardLookupItem]:
    return [
        CardLookupItem(
            idx=i,
            deck_idx=i,
            cube_uuid=cube_reverse_index.cube_uuid_for_val_deck_idx(i),
        )
        for i in positional
    ]


def _resolve_pick_items(positional: list[int]) -> list[CardLookupItem]:
    out: list[CardLookupItem] = []
    for i in positional:
        loc = cube_reverse_index.locator_for_val_pick_idx(i)
        if loc is None:
            out.append(CardLookupItem(idx=i))
        else:
            out.append(
                CardLookupItem(
                    idx=i,
                    cube_uuid=loc.cube_uuid,
                    draft_id=loc.draft_id,
                    step=loc.step,
                )
            )
    return out


@router.get("/{idx}/in-cubes", response_model=CardReverseLookupOut)
def in_cubes(idx: int, page: int = 0, per_page: int = Query(40, ge=1, le=200)) -> CardReverseLookupOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    arr = index_service.card_to_cubes[idx]
    raw, total = _paginate(arr, page, per_page)
    return CardReverseLookupOut(
        card_idx=idx, kind="cubes", total=total, page=page, per_page=per_page, items=_resolve_cube_items(raw)
    )


@router.get("/{idx}/in-decks", response_model=CardReverseLookupOut)
def in_decks(idx: int, page: int = 0, per_page: int = Query(40, ge=1, le=200)) -> CardReverseLookupOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    arr = index_service.card_to_decks[idx]
    raw, total = _paginate(arr, page, per_page)
    return CardReverseLookupOut(
        card_idx=idx, kind="decks", total=total, page=page, per_page=per_page, items=_resolve_deck_items(raw)
    )


@router.get("/{idx}/in-packs", response_model=CardReverseLookupOut)
def in_packs(idx: int, page: int = 0, per_page: int = Query(40, ge=1, le=200)) -> CardReverseLookupOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    arr = index_service.card_to_picks_in_pack[idx]
    raw, total = _paginate(arr, page, per_page)
    return CardReverseLookupOut(
        card_idx=idx, kind="in-packs", total=total, page=page, per_page=per_page, items=_resolve_pick_items(raw)
    )


@router.get("/{idx}/picked-by-humans", response_model=CardReverseLookupOut)
def picked_by_humans(idx: int, page: int = 0, per_page: int = Query(40, ge=1, le=200)) -> CardReverseLookupOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    arr = index_service.card_to_picks_human[idx]
    raw, total = _paginate(arr, page, per_page)
    return CardReverseLookupOut(
        card_idx=idx, kind="picked-by-humans", total=total, page=page, per_page=per_page, items=_resolve_pick_items(raw)
    )


class CardNeighbor(BaseModel):
    idx: int
    similarity: float


class CardNeighborsOut(BaseModel):
    card_idx: int
    ckpt: str
    neighbors: list[CardNeighbor]


@router.get("/{idx}/neighbors", response_model=CardNeighborsOut)
def neighbors(idx: int, ckpt: str, k: int = Query(24, ge=1, le=100)) -> CardNeighborsOut:
    if idx < 0 or idx >= cards_lib.num_cards():
        raise HTTPException(404, "card index out of range")
    if registry.find(ckpt) is None:
        raise HTTPException(404, f"unknown checkpoint: {ckpt!r}")
    nbr_idx, sims = embedding_service.neighbors(ckpt, idx, k=k)
    return CardNeighborsOut(
        card_idx=idx,
        ckpt=ckpt,
        neighbors=[CardNeighbor(idx=int(i), similarity=float(s)) for i, s in zip(nbr_idx, sims)],
    )
