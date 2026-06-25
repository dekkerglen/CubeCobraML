"""Pydantic response models shared across routers.

Frontend `src/lib/types.ts` mirrors these. They are intentionally hand-synced
(no openapi codegen) because we change them rarely and the manual mirroring is
clearer for a small team.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ----- cards ----------------------------------------------------------------


class CardOut(BaseModel):
    idx: int
    oracle_id: str
    name: str
    image: str = ""
    image_normal: str = ""
    type: str = ""
    cmc: float = 0.0
    elo: float = 1200.0
    pick_count: int = 0
    cube_count: int = 0
    val_pick_count: int = 0
    val_cube_count: int = 0
    is_land: bool = False
    # Phase L: from cardDictEnriched.json. Empty when sidecar missing.
    set: str = ""
    set_name: str = ""
    colors: list[str] = Field(default_factory=list)
    color_identity: list[str] = Field(default_factory=list)
    rarity: str = ""


class CardSearchOut(BaseModel):
    total: int
    items: list[CardOut]


# ----- checkpoints ----------------------------------------------------------


class HeadlineMetric(BaseModel):
    name: str
    value: float
    mode: Literal["min", "max"]


class CheckpointOut(BaseModel):
    key: str
    label: str
    kind: Literal["ckpt", "latest", "best", "current", "prod"]
    run_id: str | None = None
    epoch: int | None = None
    metric: float | None = None
    is_best: bool = False
    is_loaded: bool = False


class RunOut(BaseModel):
    """A training run as seen by the runs-first model selector."""

    run_id: str
    label: str
    short: str
    start_utc: str
    git_sha: str
    archived: bool
    best_key: str | None = None
    latest_key: str | None = None
    best_metric: float | None = None
    best_mode: Literal["min", "max"] = "max"
    best_metric_name: str = "val_bomb_agreement"
    n_ckpts: int


class CheckpointSnapshot(BaseModel):
    """KPI summary for a checkpoint — drawn from the run's metrics.csv if the
    checkpoint belongs to a run, or empty otherwise."""

    ckpt: str
    run_id: str | None
    epoch: int | None
    val_bomb_agreement: float | None = None
    val_top1: float | None = None
    val_top3: float | None = None
    val_loss: float | None = None
    val_pick_cce: float | None = None
    argmax_churn: float | None = None
    val_collapse0_common: float | None = None
    val_sat100_common: float | None = None


class RunCollapseEpochsOut(BaseModel):
    """Epochs with a per-card collapse sidecar under runs/<id>/collapse/."""

    run_id: str
    epochs: list[int]


class RunCollapseOut(BaseModel):
    """Per-card collapse vectors from a training sidecar — same shape as the
    inference-based /collapse endpoint so the frontend explorer is shared.
    `train_cum` sums train vectors over all epochs ≤ `epoch` (per-epoch train
    samples are tiny; the cumulative view converges to true train pick rates)."""

    run_id: str
    epoch: int
    split: Literal["val", "train", "train_cum"]
    epochs_aggregated: int
    appearances: list[int]
    model_picks: list[int]
    human_picks: list[int]


# ----- picks / decks / cubes -------------------------------------------------


class PickOut(BaseModel):
    idx: int
    pool: list[int]
    pack: list[int]
    pick: int
    cube_ctx: list[int] = Field(default_factory=list)
    cube_ctx_idx: int | None = None


class DeckOut(BaseModel):
    idx: int
    mainboard: list[int]
    sideboard: list[int] = Field(default_factory=list)
    cube_cards: list[int] = Field(default_factory=list)


class CubeOut(BaseModel):
    idx: int
    cards: list[int]
    kind: Literal["cube", "cubeInstance"] = "cube"


class PickIndexEntry(BaseModel):
    """Compact entry for the Draft Explorer pick list."""

    idx: int
    pool_size: int
    pack_size: int
    pick: int
    cube_ctx_idx: int | None


class CubeGroupEntry(BaseModel):
    """A cube context with the number of val picks against it."""

    cube_ctx_idx: int
    cube_size: int
    n_picks: int
    sample_cards: list[int]  # first few card indices for the cube label


# ----- cube directory (Phase I) ---------------------------------------------


class CubeDirectoryEntryOut(BaseModel):
    cube_uuid: str
    name: str
    owner: str
    owner_id: str
    image_uri: str
    card_count: int
    has_val_drafts: bool


class CubeDirectoryDetailOut(CubeDirectoryEntryOut):
    cards: list[int]


class CubeDirectorySearchOut(BaseModel):
    items: list[CubeDirectoryEntryOut]


# ----- draft sessions (Phase J) ---------------------------------------------


class DraftSessionSummaryOut(BaseModel):
    draft_id: str
    cube_uuid: str | None
    owner: str | None
    n_picks: int
    synthetic: bool
    suspect: bool
    split: Literal["train", "val"] = "val"


class DraftSessionListOut(BaseModel):
    items: list[DraftSessionSummaryOut]


class DraftStepOut(BaseModel):
    step: int
    pool: list[int]
    pack: list[int]
    human_pick: int
    cube_ctx: list[int]


class DraftSessionOut(BaseModel):
    draft_id: str
    cube_uuid: str | None
    owner: str | None
    synthetic: bool
    suspect: bool
    split: Literal["train", "val"] = "val"
    steps: list[DraftStepOut]


class DraftReplayStepOut(DraftStepOut):
    # Model's ranked predictions for this step's pack. Sorted by descending
    # probability; first entry is the model's top pick.
    ranked: list[tuple[int, float]]
    top1: int
    top1_p: float
    agrees_with_human: bool


class DraftReplayOut(BaseModel):
    draft_id: str
    cube_uuid: str | None
    ckpt: str
    steps: list[DraftReplayStepOut]


# ----- prediction -----------------------------------------------------------


class DraftRequest(BaseModel):
    ckpt: str
    pool: list[int]
    pack: list[int]


class DraftPredictionOut(BaseModel):
    ckpt: str
    ranked: list[tuple[int, float]]
    top1: int
    top1_p: float


class DeckBuildRequest(BaseModel):
    ckpt: str
    pool: list[int]


class DeckBuildOut(BaseModel):
    ckpt: str
    ranked: list[tuple[int, float]]


class RecommendRequest(BaseModel):
    ckpt: str
    cube: list[int]


class RecommendOut(BaseModel):
    ckpt: str
    adds: list[tuple[int, float]]
    cuts: list[tuple[int, float]]


class EncodeRequest(BaseModel):
    ckpt: str
    cards: list[int]


class EncodeOut(BaseModel):
    ckpt: str
    embedding: list[float]


class DeckBuilderRequest(BaseModel):
    ckpt: str
    pool: list[int]
    max_spells: int = 23
    max_lands: int = 17
    seed_count: int = 10


class DeckBuilderOut(BaseModel):
    ckpt: str
    deck: list[int]
    phase1: list[int]
    phase2: list[int]
    rejected: list[int]
    n_spells: int
    n_lands: int
    implied_basics: int


# ----- metrics --------------------------------------------------------------


class MetricsRowOut(BaseModel):
    epoch: float | None = None
    step_in_epoch: float | None = None
    fields: dict[str, float | None]


class RunMetricsOut(BaseModel):
    run_id: str
    rows: list[dict]  # raw dicts; FE will read fields by name


# ----- reverse-lookup -------------------------------------------------------


class CardLookupItem(BaseModel):
    """One reverse-lookup row. `idx` is the positional index in the respective
    collection (the original API shape); the other fields are resolved from
    Phase S/U reverse indexes so the frontend can build navigation Links
    without further lookups. All resolved fields are kind-conditional and
    optional — present when the sidecar makes them available."""

    idx: int
    cube_uuid: str | None = None
    deck_idx: int | None = None  # mirrors `idx` for "decks" kind (convenience)
    draft_id: str | None = None
    step: int | None = None


class CardReverseLookupOut(BaseModel):
    """Paginated reverse lookup. The router routes by 'kind' to one of:
    cubes / decks / picks-in-pack / picks-by-human."""

    card_idx: int
    kind: Literal["cubes", "decks", "in-packs", "picked-by-humans"]
    total: int
    page: int
    per_page: int
    items: list[CardLookupItem]


# ----- async jobs (Phase N) -------------------------------------------------


JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


class JobCreatedOut(BaseModel):
    id: str


class JobOut(BaseModel):
    id: str
    kind: str
    status: JobStatus
    progress: float
    eta_seconds: float | None = None
    # `partial` / `result` shapes vary per job kind; the frontend reads them
    # per-kind. Dict for max forward-compat.
    partial: dict | None = None
    result: dict | None = None
    error: str | None = None
