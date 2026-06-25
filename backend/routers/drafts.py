"""Draft session listing + per-pick replay (Phase J)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.lib import inference as inf_lib
from backend.lib import train_picks
from backend.models import (
    DraftReplayOut,
    DraftReplayStepOut,
    DraftSessionListOut,
    DraftSessionOut,
    DraftSessionSummaryOut,
    DraftStepOut,
)
from backend.routers import jobs as jobs_router
from backend.services.draft_sessions import DraftSession, draft_sessions
from backend.services.jobs import JobHandle
from backend.services.model_registry import registry
from backend.services.val_data import val_data


router = APIRouter(prefix="/drafts", tags=["drafts"])


REPLAY_BATCH = 8


def _summary(s: DraftSession) -> DraftSessionSummaryOut:
    return DraftSessionSummaryOut(
        draft_id=s.draft_id,
        cube_uuid=s.cube_uuid,
        owner=s.owner,
        n_picks=len(s.picks),
        synthetic=s.synthetic,
        suspect=s.suspect,
        split=s.split,
    )


def _steps(s: DraftSession) -> list[DraftStepOut]:
    """Resolve a draft's picks into full step records. Val pulls from the
    in-memory val_data.picks; train lazily reads the per-batch JSON files."""
    out: list[DraftStepOut] = []
    if s.split == "val":
        val_data.ensure_loaded()
        for step, pick_idx in enumerate(s.picks):
            rec = val_data.picks[pick_idx]
            out.append(
                DraftStepOut(
                    step=step,
                    pool=rec["pool"],
                    pack=rec["pack"],
                    human_pick=rec["pick"],
                    cube_ctx=rec.get("cube_ctx") or [],
                )
            )
        return out

    # Train — bulk-load the picks; same-batch picks share one file read.
    records = train_picks.get_pick_records(s.picks)
    for step, rec in enumerate(records):
        out.append(
            DraftStepOut(
                step=step,
                pool=rec.get("pool") or [],
                pack=rec.get("pack") or [],
                human_pick=rec.get("pick", -1),
                # Train pick records carry cube_cards_idx (per-batch). We don't
                # have a per-pick cube_ctx materialized for train; the cube card
                # list can be resolved via cube_directory at render time.
                cube_ctx=[],
            )
        )
    return out


@router.get("", response_model=DraftSessionListOut)
def list_drafts(
    cube_uuid: str | None = Query(None),
    include_suspect: bool = Query(False),
    split: str = Query("all", pattern="^(all|train|val)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DraftSessionListOut:
    draft_sessions.ensure_loaded()
    if cube_uuid:
        sessions = draft_sessions.list_by_cube(cube_uuid, split=split)  # type: ignore[arg-type]
        if not include_suspect:
            sessions = [s for s in sessions if not s.suspect]
        sessions = sessions[offset : offset + limit]
    else:
        sessions = draft_sessions.list_all(
            include_suspect=include_suspect,
            limit=limit,
            offset=offset,
            split=split,  # type: ignore[arg-type]
        )
    return DraftSessionListOut(items=[_summary(s) for s in sessions])


@router.get("/{draft_id}", response_model=DraftSessionOut)
def get_draft(draft_id: str) -> DraftSessionOut:
    s = draft_sessions.get(draft_id)
    if s is None:
        raise HTTPException(404, "unknown draft_id")
    return DraftSessionOut(
        draft_id=s.draft_id,
        cube_uuid=s.cube_uuid,
        owner=s.owner,
        synthetic=s.synthetic,
        suspect=s.suspect,
        split=s.split,
        steps=_steps(s),
    )


def _replay_step(st: DraftStepOut, probs_row: np.ndarray) -> DraftReplayStepOut:
    pack_set = set(st.pack)
    ranked = sorted(
        ((int(i), float(probs_row[i])) for i in pack_set),
        key=lambda x: -x[1],
    )
    top1, top1_p = ranked[0] if ranked else (0, 0.0)
    return DraftReplayStepOut(
        step=st.step,
        pool=st.pool,
        pack=st.pack,
        human_pick=st.human_pick,
        cube_ctx=st.cube_ctx,
        ranked=ranked,
        top1=top1,
        top1_p=top1_p,
        agrees_with_human=(top1 == st.human_pick),
    )


@router.get("/{draft_id}/replay", response_model=DraftReplayOut)
def replay_draft(draft_id: str, ckpt: str) -> DraftReplayOut:
    """Run the model against every pick of this draft in one batched inference.

    Sync path retained for callers that don't need progress reporting; the job
    runner below is the default UI path so the replay panel can stream steps
    as they finish.
    """
    s = draft_sessions.get(draft_id)
    if s is None:
        raise HTTPException(404, "unknown draft_id")
    if registry.find(ckpt) is None:
        raise HTTPException(404, f"unknown checkpoint: {ckpt!r}")
    obj = registry.get(ckpt)

    steps_plain = _steps(s)
    if not steps_plain:
        return DraftReplayOut(draft_id=draft_id, cube_uuid=s.cube_uuid, ckpt=ckpt, steps=[])

    pools = np.stack([inf_lib.one_hot(st.pool) for st in steps_plain]).astype(np.float32)
    packs = np.stack([inf_lib.one_hot(st.pack) for st in steps_plain]).astype(np.float32)
    probs = inf_lib.draft_logits_batch(obj, pools, packs)

    replay_steps = [_replay_step(st, probs[i]) for i, st in enumerate(steps_plain)]
    return DraftReplayOut(draft_id=draft_id, cube_uuid=s.cube_uuid, ckpt=ckpt, steps=replay_steps)


def _draft_replay_runner(handle: JobHandle, body: dict) -> dict:
    """Streaming replay. Body: `{draft_id, ckpt}`. Emits a partial DraftReplayOut
    every batch so the frontend can fill in steps as they're computed."""
    draft_id = body.get("draft_id")
    ckpt = body.get("ckpt")
    if not draft_id or not ckpt:
        raise ValueError("missing draft_id or ckpt")
    s = draft_sessions.get(draft_id)
    if s is None:
        raise ValueError(f"unknown draft_id: {draft_id!r}")
    if registry.find(ckpt) is None:
        raise ValueError(f"unknown checkpoint: {ckpt!r}")
    obj = registry.get(ckpt)

    steps_plain = _steps(s)
    n = len(steps_plain)
    if n == 0:
        return DraftReplayOut(draft_id=draft_id, cube_uuid=s.cube_uuid, ckpt=ckpt, steps=[]).model_dump()

    replay_steps: list[DraftReplayStepOut] = []
    for i in range(0, n, REPLAY_BATCH):
        if handle.cancel.is_set():
            break
        batch = steps_plain[i : i + REPLAY_BATCH]
        pools = np.stack([inf_lib.one_hot(st.pool) for st in batch]).astype(np.float32)
        packs = np.stack([inf_lib.one_hot(st.pack) for st in batch]).astype(np.float32)
        probs = inf_lib.draft_logits_batch(obj, pools, packs)
        for j, st in enumerate(batch):
            replay_steps.append(_replay_step(st, probs[j]))
        done = len(replay_steps)
        handle.report(
            progress=done / n,
            partial=DraftReplayOut(
                draft_id=draft_id,
                cube_uuid=s.cube_uuid,
                ckpt=ckpt,
                steps=replay_steps,
            ).model_dump(),
        )

    return DraftReplayOut(
        draft_id=draft_id,
        cube_uuid=s.cube_uuid,
        ckpt=ckpt,
        steps=replay_steps,
    ).model_dump()


jobs_router.register("draft_replay", _draft_replay_runner)
