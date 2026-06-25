"""Two-phase deck builder, modeled on CubeCobra's `localBatchDeckbuild`.

Algorithm:
  Phase 1 — seed (first 10 cards):
    score each pool card with `deck_build_decoder`(current_deck) and apply a
    duplicate penalty of `0.9 ** existing_count`. Pick the top scoring card
    that doesn't blow the land/spell cap. Repeat until 10 cards picked.

  Phase 2 — fill remaining slots:
    treat the not-yet-picked pool as a "pack" and score with
    `draft_decoder`(current_deck, pack). Apply same duplicate penalty.
    Continue until the deck reaches `max_spells + max_lands`.

This intentionally doesn't add basic lands (val deck records don't carry them).
If the pool has fewer than `max_lands` lands, the deck ends with fewer cards
and the caller can note that basics are implied.

Mirrors the reference TypeScript implementation in CubeCobra's
`packages/client/src/utils/draftBot.ts` (localBatchDeckbuild).
"""

from __future__ import annotations

import numpy as np

from . import cards as cards_mod
from .inference import deck_build, draft_logits


DUP_PENALTY = 0.9
DEFAULT_MAX_SPELLS = 23
DEFAULT_MAX_LANDS = 17
DEFAULT_SEED_COUNT = 10


def is_land(idx: int) -> bool:
    c = cards_mod.card(idx)
    t = (c.type or "").lower()
    return "land" in t and "creature" not in t


def _counts(deck: list[int]) -> tuple[int, int, dict[int, int]]:
    """Return (spells_in_deck, lands_in_deck, per_card_count)."""
    spells, lands = 0, 0
    per: dict[int, int] = {}
    for c in deck:
        per[c] = per.get(c, 0) + 1
        if is_land(c):
            lands += 1
        else:
            spells += 1
    return spells, lands, per


def _eligible_mask(
    pool_unique: list[int], deck: list[int], pool_counts: dict[int, int], max_spells: int, max_lands: int
) -> np.ndarray:
    """Boolean mask over pool_unique: True if the card can still be added."""
    spells, lands, in_deck = _counts(deck)
    mask = np.ones(len(pool_unique), dtype=bool)
    for i, c in enumerate(pool_unique):
        if in_deck.get(c, 0) >= pool_counts.get(c, 0):
            mask[i] = False
            continue
        if is_land(c) and lands >= max_lands:
            mask[i] = False
        elif (not is_land(c)) and spells >= max_spells:
            mask[i] = False
    return mask


def _duplicate_factor(idx: int, deck: list[int]) -> float:
    count = sum(1 for c in deck if c == idx)
    return DUP_PENALTY**count


def build_deck(
    obj,
    pool: list[int],
    *,
    max_spells: int = DEFAULT_MAX_SPELLS,
    max_lands: int = DEFAULT_MAX_LANDS,
    seed_count: int = DEFAULT_SEED_COUNT,
    return_steps: bool = False,
    progress_cb=None,
) -> dict:
    """Run two-phase deck building. Returns dict with keys: deck, phase1, phase2,
    rejected, steps (only if return_steps)."""
    pool_counts: dict[int, int] = {}
    for c in pool:
        pool_counts[c] = pool_counts.get(c, 0) + 1
    pool_unique = list(pool_counts.keys())
    pool_arr = np.array(pool_unique, dtype=np.int64)

    deck: list[int] = []
    steps: list[dict] = []
    target = max_spells + max_lands

    N = cards_mod.num_cards()
    deck_oh = np.zeros(N, dtype=np.float32)

    # ---- Phase 1: seed --------------------------------------------------------
    while len(deck) < seed_count and len(deck) < target:
        probs = deck_build(obj, deck_oh)
        elig = _eligible_mask(pool_unique, deck, pool_counts, max_spells, max_lands)
        if not elig.any():
            break
        scores = probs[pool_arr]
        # Duplicate penalty
        for i, c in enumerate(pool_unique):
            scores[i] *= _duplicate_factor(c, deck)
        scores = np.where(elig, scores, -np.inf)
        pick_i = int(np.argmax(scores))
        pick = int(pool_unique[pick_i])
        deck.append(pick)
        deck_oh[pick] = min(deck_oh[pick] + 1.0, pool_counts[pick])
        if return_steps:
            steps.append(
                {
                    "phase": 1,
                    "step": len(deck),
                    "pick": pick,
                    "score": float(scores[pick_i]),
                }
            )
        if progress_cb is not None:
            progress_cb(len(deck), target, list(deck))

    phase1 = list(deck)

    # ---- Phase 2: fill --------------------------------------------------------
    while len(deck) < target:
        elig = _eligible_mask(pool_unique, deck, pool_counts, max_spells, max_lands)
        if not elig.any():
            break
        eligible_idx = [c for i, c in enumerate(pool_unique) if elig[i]]
        pack_oh = np.zeros(N, dtype=np.float32)
        pack_oh[eligible_idx] = 1.0
        probs = draft_logits(obj, deck_oh, pack_oh)
        scores = probs[pool_arr]
        for i, c in enumerate(pool_unique):
            scores[i] *= _duplicate_factor(c, deck)
        scores = np.where(elig, scores, -np.inf)
        pick_i = int(np.argmax(scores))
        pick = int(pool_unique[pick_i])
        deck.append(pick)
        deck_oh[pick] = min(deck_oh[pick] + 1.0, pool_counts[pick])
        if return_steps:
            steps.append(
                {
                    "phase": 2,
                    "step": len(deck),
                    "pick": pick,
                    "score": float(scores[pick_i]),
                }
            )
        if progress_cb is not None:
            progress_cb(len(deck), target, list(deck))

    phase2 = deck[len(phase1) :]
    in_deck = {c: 0 for c in deck}
    for c in deck:
        in_deck[c] += 1
    rejected: list[int] = []
    for c in pool:
        if in_deck.get(c, 0) > 0:
            in_deck[c] -= 1
        else:
            rejected.append(c)

    result = {
        "deck": deck,
        "phase1": phase1,
        "phase2": phase2,
        "rejected": rejected,
        "n_spells": sum(1 for c in deck if not is_land(c)),
        "n_lands": sum(1 for c in deck if is_land(c)),
        "implied_basics": max(0, max_lands - sum(1 for c in deck if is_land(c))),
    }
    if return_steps:
        result["steps"] = steps
    return result


def pile_by_cmc(deck: list[int], *, lands_separate: bool = True) -> tuple[list[list[int]], list[str]]:
    """Group deck into CMC piles. Returns (piles, labels). Lands optionally
    bucketed in a separate trailing pile."""
    spell_buckets: dict[int, list[int]] = {}
    lands: list[int] = []
    for c in deck:
        if is_land(c):
            lands.append(c)
            continue
        cmc = int(cards_mod.card(c).cmc or 0)
        key = 6 if cmc >= 6 else cmc
        spell_buckets.setdefault(key, []).append(c)
    piles: list[list[int]] = []
    labels: list[str] = []
    for k in sorted(spell_buckets.keys()):
        piles.append(spell_buckets[k])
        labels.append("6+" if k == 6 else str(k))
    if lands_separate and lands:
        piles.append(lands)
        labels.append("L")
    return piles, labels
