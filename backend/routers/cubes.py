"""Val cube browsing + global cube directory (search/by-uuid/resolve)."""

from __future__ import annotations

import random
import re
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query

from backend.models import (
    CubeDirectoryDetailOut,
    CubeDirectoryEntryOut,
    CubeDirectorySearchOut,
    CubeOut,
    DraftSessionListOut,
    DraftSessionSummaryOut,
)
from backend.lib import train_picks
from backend.services.cube_directory import CubeEntry, cube_directory
from backend.services.decks_by_cube import get_by_split
from backend.services.draft_sessions import draft_sessions
from backend.services.val_data import val_data


router = APIRouter(prefix="/cubes", tags=["cubes"])

# CubeCobra exposes cube JSON for any UUID or shortid at this URL. Public,
# unauthenticated, used here only to resolve human-typed identifiers back to a
# canonical UUID we can look up in our offline directory.
CUBECOBRA_API_URL = "https://cubecobra.com/cube/api/cubeJSON/{slug}"


def _to_entry(e: CubeEntry) -> CubeDirectoryEntryOut:
    return CubeDirectoryEntryOut(
        cube_uuid=e.cube_uuid,
        name=e.name,
        owner=e.owner,
        owner_id=e.owner_id,
        image_uri=e.image_uri,
        card_count=e.card_count,
        has_val_drafts=e.has_val_drafts,
    )


# ----- positional (val-only) — preserved for existing dashboard pages -------


@router.get("", response_model=list[CubeOut])
def list_cubes(
    limit: int = Query(50, ge=1, le=500), kind: str = Query("cube", description="cube | cubeInstance")
) -> list[CubeOut]:
    val_data.ensure_loaded()
    src = val_data.cubes if kind == "cube" else val_data.cube_instances
    return [CubeOut(idx=i, cards=c, kind=kind) for i, c in enumerate(src[:limit])]  # type: ignore


@router.get("/random/one", response_model=CubeOut)
def random_one() -> CubeOut:
    val_data.ensure_loaded()
    n = len(val_data.cubes)
    if not n:
        raise HTTPException(404, "no cubes loaded")
    return by_idx(random.randrange(n))


# ----- directory (Phase I) --------------------------------------------------


@router.get("/search", response_model=CubeDirectorySearchOut)
def search_cubes(
    q: str = Query(..., min_length=1),
    split: str = Query("all", pattern="^(all|val)$"),
    limit: int = Query(25, ge=1, le=100),
) -> CubeDirectorySearchOut:
    hits = cube_directory.search(q, split=split, limit=limit)
    return CubeDirectorySearchOut(items=[_to_entry(e) for e in hits])


@router.get("/by-owner", response_model=CubeDirectorySearchOut)
def cubes_by_owner(
    owner: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> CubeDirectorySearchOut:
    """Exact-match (case-insensitive) cubes for a given owner — Phase V
    replacement for the old My Cubes shelf hack."""
    rows, _total = cube_directory.by_owner(owner, limit=limit, offset=offset)
    return CubeDirectorySearchOut(items=[_to_entry(e) for e in rows])


@router.get("/by-uuid/{cube_uuid}", response_model=CubeDirectoryDetailOut)
def cube_by_uuid(cube_uuid: str) -> CubeDirectoryDetailOut:
    entry = cube_directory.get(cube_uuid)
    if entry is None:
        raise HTTPException(404, "cube uuid not in directory")
    cards = cube_directory.cards_for(cube_uuid) or []
    base = _to_entry(entry)
    return CubeDirectoryDetailOut(**base.model_dump(), cards=cards)


def _parse_card_idxs(raw: str | None) -> set[int] | None:
    """Parse a comma-separated string into a card-idx set. None / empty → no filter."""
    if not raw or not raw.strip():
        return None
    out: set[int] = set()
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.add(int(s))
        except ValueError:
            continue
    return out or None


@router.get("/by-uuid/{cube_uuid}/drafts", response_model=DraftSessionListOut)
def cube_drafts(
    cube_uuid: str,
    include_suspect: bool = Query(False),
    split: str = Query("all", pattern="^(all|train|val)$"),
    card_idxs: str | None = Query(None, description="Comma-sep card idxs; keep drafts where any pick is in the set"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DraftSessionListOut:
    sessions = draft_sessions.list_by_cube(cube_uuid, split=split)  # type: ignore[arg-type]
    if not include_suspect:
        sessions = [s for s in sessions if not s.suspect]

    wanted = _parse_card_idxs(card_idxs)
    if wanted is not None:
        sessions = _filter_drafts_by_card_idxs(sessions, wanted)

    sessions = sessions[offset : offset + limit]
    return DraftSessionListOut(
        items=[
            DraftSessionSummaryOut(
                draft_id=s.draft_id,
                cube_uuid=s.cube_uuid,
                owner=s.owner,
                n_picks=len(s.picks),
                synthetic=s.synthetic,
                suspect=s.suspect,
                split=s.split,
            )
            for s in sessions
        ]
    )


def _filter_drafts_by_card_idxs(sessions: list, wanted: set[int]) -> list:
    """Keep drafts whose human-picked cards intersect `wanted`. Val + train
    resolved separately to amortize the train per-batch JSON reads."""
    val_data.ensure_loaded()
    val_keep: set[str] = set()
    train_pick_indices: list[int] = []
    train_to_draft: dict[int, str] = {}

    for s in sessions:
        if s.split == "val":
            for i in s.picks:
                rec = val_data.picks[i]
                if int(rec.get("pick", -1)) in wanted:
                    val_keep.add(s.draft_id)
                    break
        else:
            # Defer train: bulk-resolve below to share batch reads.
            for i in s.picks:
                train_pick_indices.append(i)
                train_to_draft[i] = s.draft_id

    train_keep: set[str] = set()
    if train_pick_indices:
        records = train_picks.get_pick_records(sorted(set(train_pick_indices)))
        for i, rec in zip(sorted(set(train_pick_indices)), records):
            if int(rec.get("pick", -1)) in wanted:
                train_keep.add(train_to_draft[i])

    return [s for s in sessions if (s.draft_id in val_keep or s.draft_id in train_keep)]


@router.get("/by-uuid/{cube_uuid}/decks")
def cube_decks(
    cube_uuid: str,
    split: str = Query("all", pattern="^(all|train|val)$"),
    card_idxs: str | None = Query(
        None,
        description="Comma-sep card idxs; keep decks whose mainboard intersects the set (val only — train decks pass through unfiltered)",
    ),
    limit: int = Query(120, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    items = get_by_split(cube_uuid, split=split)  # type: ignore[arg-type]

    wanted = _parse_card_idxs(card_idxs)
    if wanted is not None:
        items = _filter_decks_by_card_idxs(items, wanted)

    total = len(items)
    page = items[offset : offset + limit]
    return {
        "cube_uuid": cube_uuid,
        "total": total,
        # Each item: {idx, split}. Frontend uses split to know which deck endpoint to hit.
        "items": [{"idx": idx, "split": tag} for (idx, tag) in page],
    }


def _filter_decks_by_card_idxs(items: list, wanted: set[int]) -> list:
    """Keep val decks whose mainboard intersects `wanted`. Train decks pass
    through unfiltered for now — the split-aware deck reader isn't wired yet,
    so we can't read their mainboards here."""
    val_data.ensure_loaded()
    out = []
    for idx, tag in items:
        if tag == "train":
            out.append((idx, tag))
            continue
        if idx < 0 or idx >= len(val_data.decks):
            continue
        mb = val_data.decks[idx].get("mainboard") or []
        if any(int(c) in wanted for c in mb):
            out.append((idx, tag))
    return out


# Match either a cubecobra.com URL or a bare slug/uuid.
_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _extract_slug(url_or_slug: str) -> str | None:
    s = url_or_slug.strip()
    if not s:
        return None
    if "//" in s or s.startswith("cubecobra.com") or "/" in s:
        # treat as URL
        if "//" not in s:
            s = "https://" + s
        try:
            parts = urlparse(s).path.strip("/").split("/")
        except Exception:
            return None
        # cubecobra.com/cube/overview/<slug> | /cube/list/<slug> | /cube/playtest/<slug> | ...
        for i, p in enumerate(parts):
            if p == "cube" and i + 2 < len(parts):
                cand = parts[i + 2]
                if _SLUG_RE.match(cand):
                    return cand
        return None
    return s if _SLUG_RE.match(s) else None


@router.get("/resolve", response_model=CubeDirectoryDetailOut)
def resolve_cube(url_or_slug: str = Query(..., min_length=1)) -> CubeDirectoryDetailOut:
    """Resolve a CubeCobra URL or human-readable slug to a directory entry.

    Calls cubecobra.com once to translate slug → UUID (since slugs aren't
    stored locally), then looks the UUID up in our directory.
    """
    slug = _extract_slug(url_or_slug)
    if slug is None:
        raise HTTPException(400, "could not extract cube slug/uuid from input")

    # Fast path: if the user pasted a UUID directly and it's in our directory.
    if cube_directory.get(slug) is not None:
        return cube_by_uuid(slug)

    # Slug path: ask CubeCobra to resolve it.
    try:
        r = httpx.get(CUBECOBRA_API_URL.format(slug=slug), timeout=15.0)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"cubecobra.com unreachable: {exc}") from exc
    if r.status_code != 200:
        raise HTTPException(404, f"cubecobra.com returned {r.status_code} for slug={slug!r}")
    try:
        data = r.json()
    except ValueError as exc:
        raise HTTPException(502, "cubecobra.com returned non-JSON") from exc
    # CubeCobra's cubeJSON puts the UUID at `cube.id` or `id` depending on the
    # endpoint version. Check both.
    uuid = (data.get("cube") or {}).get("id") or data.get("id")
    if not uuid:
        raise HTTPException(404, "cubecobra.com response missing cube id")
    if cube_directory.get(uuid) is None:
        raise HTTPException(404, "cube exists on cubecobra.com but is not in our offline corpus")
    return cube_by_uuid(uuid)


# ----- positional fallback — must stay below the named routes ---------------


@router.get("/{idx}", response_model=CubeOut)
def by_idx(idx: int) -> CubeOut:
    val_data.ensure_loaded()
    n = len(val_data.cubes)
    if idx < 0 or idx >= n:
        raise HTTPException(404, "cube idx out of range")
    return CubeOut(idx=idx, cards=val_data.cubes[idx], kind="cube")
