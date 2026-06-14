/** Hand-synced mirror of backend/models.py. Keep in sync on schema changes. */

export type CkptKind = "ckpt" | "latest" | "best" | "current" | "prod";

export interface Card {
  idx: number;
  oracle_id: string;
  name: string;
  image: string;
  image_normal: string;
  type: string;
  cmc: number;
  elo: number;
  pick_count: number;
  cube_count: number;
  val_pick_count: number;
  val_cube_count: number;
  is_land: boolean;
  // Phase L — present when the enriched sidecar exists, else default empty.
  set: string;
  set_name: string;
  colors: string[];
  color_identity: string[];
  rarity: string;
}

export interface CardSearchResult {
  total: number;
  items: Card[];
}

export interface Checkpoint {
  key: string;
  label: string;
  kind: CkptKind;
  run_id: string | null;
  epoch: number | null;
  metric: number | null;
  is_best: boolean;
  is_loaded: boolean;
}

/** A training run as a first-class entity. Drives the runs-first selector. */
export interface Run {
  run_id: string;
  label: string;
  short: string;
  start_utc: string;
  git_sha: string;
  archived: boolean;
  best_key: string | null;
  latest_key: string | null;
  best_metric: number | null;
  best_mode: "min" | "max";
  best_metric_name: string;
  n_ckpts: number;
}

export interface CheckpointSnapshot {
  ckpt: string;
  run_id: string | null;
  epoch: number | null;
  val_bomb_agreement: number | null;
  val_top1: number | null;
  val_top3: number | null;
  val_loss: number | null;
  val_pick_cce: number | null;
  argmax_churn: number | null;
  val_collapse0_common: number | null;
  val_sat100_common: number | null;
}

export interface Pick {
  idx: number;
  pool: number[];
  pack: number[];
  pick: number;
  cube_ctx: number[];
  cube_ctx_idx: number | null;
}

export interface PickIndexEntry {
  idx: number;
  pool_size: number;
  pack_size: number;
  pick: number;
  cube_ctx_idx: number | null;
}

export interface CubeGroupEntry {
  cube_ctx_idx: number;
  cube_size: number;
  n_picks: number;
  sample_cards: number[];
}

export interface Deck {
  idx: number;
  mainboard: number[];
  sideboard: number[];
  cube_cards: number[];
}

export interface Cube {
  idx: number;
  cards: number[];
  kind: "cube" | "cubeInstance";
}

export interface DraftPrediction {
  ckpt: string;
  ranked: Array<[number, number]>;
  top1: number;
  top1_p: number;
}

export interface DeckBuilderResult {
  ckpt: string;
  deck: number[];
  phase1: number[];
  phase2: number[];
  rejected: number[];
  n_spells: number;
  n_lands: number;
  implied_basics: number;
}

export interface MetricsRow {
  epoch?: number | null;
  step_in_epoch?: number | null;
  [k: string]: number | null | undefined;
}

export interface CubeDirectoryEntry {
  cube_uuid: string;
  name: string;
  owner: string;
  owner_id: string;
  image_uri: string;
  card_count: number;
  has_val_drafts: boolean;
}

export interface CubeDirectoryDetail extends CubeDirectoryEntry {
  cards: number[];
}

// ----- draft sessions (Phase J) -------------------------------------------

export interface DraftSessionSummary {
  draft_id: string;
  cube_uuid: string | null;
  owner: string | null;
  n_picks: number;
  synthetic: boolean;
  suspect: boolean;
  split: "train" | "val";
}

export interface DraftStep {
  step: number;
  pool: number[];
  pack: number[];
  human_pick: number;
  cube_ctx: number[];
}

export interface DraftSession {
  draft_id: string;
  cube_uuid: string | null;
  owner: string | null;
  synthetic: boolean;
  suspect: boolean;
  steps: DraftStep[];
}

export interface DraftReplayStep extends DraftStep {
  ranked: Array<[number, number]>;
  top1: number;
  top1_p: number;
  agrees_with_human: boolean;
}

export interface DraftReplay {
  draft_id: string;
  cube_uuid: string | null;
  ckpt: string;
  steps: DraftReplayStep[];
}

// ----- async jobs (Phase N) -----------------------------------------------

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface JobOut {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;        // 0..1
  eta_seconds: number | null;
  partial: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  error: string | null;
}
