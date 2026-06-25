"""Val pick browsing endpoints."""

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from backend.models import CubeGroupEntry, PickIndexEntry, PickOut
from backend.services.val_data import val_data


router = APIRouter(prefix="/picks", tags=["picks"])


@router.get("/{idx}", response_model=PickOut)
def by_idx(idx: int) -> PickOut:
    val_data.ensure_loaded()
    picks = val_data.picks
    if idx < 0 or idx >= len(picks):
        raise HTTPException(404, "pick idx out of range")
    rec = picks[idx]
    return PickOut(
        idx=idx,
        pool=rec["pool"],
        pack=rec["pack"],
        pick=rec["pick"],
        cube_ctx=rec["cube_ctx"],
        cube_ctx_idx=rec["cube_ctx_idx"],
    )


@router.get("/random/one", response_model=PickOut)
def random_one() -> PickOut:
    val_data.ensure_loaded()
    n = len(val_data.picks)
    if not n:
        raise HTTPException(404, "no picks loaded")
    idx = random.randrange(n)
    return by_idx(idx)


@lru_cache(maxsize=1)
def _picks_by_cube_idx() -> dict[int, list[int]]:
    """Pick-record indices grouped by cube_ctx_idx (rebuilt once)."""
    val_data.ensure_loaded()
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(val_data.picks):
        cci = r.get("cube_ctx_idx")
        if cci is not None:
            groups[cci].append(i)
    return dict(groups)


@router.get("/by-cube/{cube_ctx_idx}/list", response_model=list[PickIndexEntry])
def by_cube(cube_ctx_idx: int, limit: int = Query(200, ge=1, le=2000)) -> list[PickIndexEntry]:
    val_data.ensure_loaded()
    pick_idxs = _picks_by_cube_idx().get(cube_ctx_idx, [])
    out: list[PickIndexEntry] = []
    for i in pick_idxs[:limit]:
        r = val_data.picks[i]
        out.append(
            PickIndexEntry(
                idx=i,
                pool_size=len(r["pool"]),
                pack_size=len(r["pack"]),
                pick=r["pick"],
                cube_ctx_idx=r.get("cube_ctx_idx"),
            )
        )
    return out


@router.get("/cubes/list", response_model=list[CubeGroupEntry])
def list_cube_groups(limit: int = Query(120, ge=1, le=500)) -> list[CubeGroupEntry]:
    """Cubes available in val data, ranked by # of val picks against them."""
    val_data.ensure_loaded()
    groups = _picks_by_cube_idx()
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    out: list[CubeGroupEntry] = []
    for cci, pick_idxs in ordered[:limit]:
        ctx = val_data.picks[pick_idxs[0]]["cube_ctx"] if pick_idxs else []
        out.append(
            CubeGroupEntry(
                cube_ctx_idx=cci,
                cube_size=len(ctx),
                n_picks=len(pick_idxs),
                sample_cards=ctx[:3],
            )
        )
    return out
