"""Model inference endpoints — draft / deck_build / recommend / encode /
deckbuilder.

Each request takes a checkpoint key + card-index inputs (not one-hots; the
backend constructs those itself, smaller payloads + safer API surface).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.lib import inference as inf_lib
from backend.lib.deckbuild import build_deck
from backend.models import (
    DeckBuildOut,
    DeckBuildRequest,
    DeckBuilderOut,
    DeckBuilderRequest,
    DraftPredictionOut,
    DraftRequest,
    EncodeOut,
    EncodeRequest,
    RecommendOut,
    RecommendRequest,
)
from backend.routers import jobs as jobs_router
from backend.services.jobs import JobHandle
from backend.services.model_registry import registry


router = APIRouter(prefix="/predict", tags=["predict"])


def _get_obj(ckpt: str):
    if registry.find(ckpt) is None:
        raise HTTPException(404, f"unknown checkpoint: {ckpt!r}")
    return registry.get(ckpt)


@router.post("/draft", response_model=DraftPredictionOut)
def draft(req: DraftRequest) -> DraftPredictionOut:
    obj = _get_obj(req.ckpt)
    pool_oh = inf_lib.one_hot(req.pool)
    pack_oh = inf_lib.one_hot(req.pack)
    probs = inf_lib.draft_logits(obj, pool_oh, pack_oh)
    pack_set = set(req.pack)
    ranked = sorted(
        ((int(i), float(probs[i])) for i in pack_set),
        key=lambda x: -x[1],
    )
    top1, top1_p = ranked[0] if ranked else (0, 0.0)
    return DraftPredictionOut(ckpt=req.ckpt, ranked=ranked, top1=top1, top1_p=top1_p)


@router.post("/deck_build", response_model=DeckBuildOut)
def deck_build(req: DeckBuildRequest) -> DeckBuildOut:
    obj = _get_obj(req.ckpt)
    pool_oh = inf_lib.one_hot(req.pool)
    probs = inf_lib.deck_build(obj, pool_oh)
    # Only return probabilities for cards in the pool — that's the meaningful
    # decision space; everything else is just noise to the FE.
    pool_unique = list(dict.fromkeys(req.pool))
    ranked = sorted(
        ((int(i), float(probs[i])) for i in pool_unique),
        key=lambda x: -x[1],
    )
    return DeckBuildOut(ckpt=req.ckpt, ranked=ranked)


@router.post("/recommend", response_model=RecommendOut)
def recommend(req: RecommendRequest) -> RecommendOut:
    obj = _get_obj(req.ckpt)
    cube_oh = inf_lib.one_hot(req.cube)
    probs = inf_lib.cube_recon(obj, cube_oh)
    cube_set = set(req.cube)
    adds = []
    cuts = []
    # Adds: highest probability not in cube
    cand = [(i, p) for i, p in enumerate(probs) if i not in cube_set]
    cand.sort(key=lambda x: -x[1])
    adds = [(int(i), float(p)) for i, p in cand[:50]]
    # Cuts: lowest probability in cube
    in_cube = [(i, float(probs[i])) for i in req.cube]
    in_cube.sort(key=lambda x: x[1])
    cuts = [(int(i), p) for i, p in in_cube[:50]]
    return RecommendOut(ckpt=req.ckpt, adds=adds, cuts=cuts)


@router.post("/encode", response_model=EncodeOut)
def encode(req: EncodeRequest) -> EncodeOut:
    obj = _get_obj(req.ckpt)
    vec = inf_lib.one_hot(req.cards)
    out = inf_lib.encode(obj, vec).tolist()
    return EncodeOut(ckpt=req.ckpt, embedding=out)


@router.post("/deckbuilder", response_model=DeckBuilderOut)
def deckbuilder(req: DeckBuilderRequest) -> DeckBuilderOut:
    obj = _get_obj(req.ckpt)
    result = build_deck(
        obj,
        req.pool,
        max_spells=req.max_spells,
        max_lands=req.max_lands,
        seed_count=req.seed_count,
    )
    return DeckBuilderOut(
        ckpt=req.ckpt,
        deck=result["deck"],
        phase1=result["phase1"],
        phase2=result["phase2"],
        rejected=result["rejected"],
        n_spells=result["n_spells"],
        n_lands=result["n_lands"],
        implied_basics=result["implied_basics"],
    )


# ----- job runners ----------------------------------------------------------


def _recommend_runner(handle: JobHandle, body: dict) -> dict:
    """Body: `{ckpt, cube: [card_idx]}`. Single forward pass, so progress is
    just 0 → 1 — but we still go through the job runner so the frontend
    surfaces a ProgressBar on every model call."""
    ckpt = body.get("ckpt")
    if not ckpt or registry.find(ckpt) is None:
        raise ValueError(f"unknown or missing checkpoint: {ckpt!r}")
    cube = body.get("cube") or []
    handle.report(progress=0.0, partial=None)
    cube_oh = inf_lib.one_hot(cube)
    probs = inf_lib.cube_recon(registry.get(ckpt), cube_oh)
    cube_set = set(cube)
    cand = [(i, p) for i, p in enumerate(probs) if i not in cube_set]
    cand.sort(key=lambda x: -x[1])
    adds = [(int(i), float(p)) for i, p in cand[:50]]
    in_cube = [(int(i), float(probs[i])) for i in cube]
    in_cube.sort(key=lambda x: x[1])
    cuts = in_cube[:50]
    return RecommendOut(ckpt=ckpt, adds=adds, cuts=cuts).model_dump()


def _deckbuilder_runner(handle: JobHandle, body: dict) -> dict:
    """Body: same shape as DeckBuilderRequest. Reports per-spell progress."""
    ckpt = body.get("ckpt")
    if not ckpt or registry.find(ckpt) is None:
        raise ValueError(f"unknown or missing checkpoint: {ckpt!r}")
    pool = body.get("pool") or []
    obj = registry.get(ckpt)

    def on_progress(done: int, total: int, deck: list[int]) -> None:
        if handle.cancel.is_set():
            return
        handle.report(progress=done / max(1, total), partial={"deck_so_far": deck, "done": done, "of": total})

    result = build_deck(
        obj,
        pool,
        max_spells=int(body.get("max_spells") or 23),
        max_lands=int(body.get("max_lands") or 17),
        seed_count=int(body.get("seed_count") or 10),
        progress_cb=on_progress,
    )
    return DeckBuilderOut(
        ckpt=ckpt,
        deck=result["deck"],
        phase1=result["phase1"],
        phase2=result["phase2"],
        rejected=result["rejected"],
        n_spells=result["n_spells"],
        n_lands=result["n_lands"],
        implied_basics=result["implied_basics"],
    ).model_dump()


jobs_router.register("recommend", _recommend_runner)
jobs_router.register("deckbuilder", _deckbuilder_runner)
