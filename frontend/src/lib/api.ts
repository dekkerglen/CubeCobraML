/** Typed fetch wrappers over the FastAPI backend. */
import type {
  Card,
  CardSearchResult,
  Checkpoint,
  CheckpointSnapshot,
  Cube,
  CubeDirectoryDetail,
  CubeDirectoryEntry,
  CubeGroupEntry,
  Deck,
  DeckBuilderResult,
  DraftPrediction,
  DraftReplay,
  DraftSession,
  DraftSessionSummary,
  JobOut,
  MetricsRow,
  Pick,
  PickIndexEntry,
  Run,
} from "./types";

const API = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${path} — ${text}`);
  }
  return r.json();
}

// ----- checkpoints ---------------------------------------------------------

export const apiCheckpoints = () =>
  get<Checkpoint[]>("/checkpoints");

export const apiCheckpointSnapshot = (key: string) =>
  get<CheckpointSnapshot>(`/checkpoints/${encodeURIComponent(key)}/snapshot`);

export const apiPreloadCheckpoint = (key: string) =>
  fetch(`${API}/checkpoints/${encodeURIComponent(key)}/preload`, { method: "POST" });

// ----- runs (runs-first selector) ------------------------------------------

export const apiRuns = () =>
  get<Run[]>("/runs");

export const apiRunCkpts = (run_id: string) =>
  get<Checkpoint[]>(`/runs/${encodeURIComponent(run_id)}/ckpts`);

// ----- cards ---------------------------------------------------------------

export const apiCard = (idx: number) =>
  get<Card>(`/cards/${idx}`);

export const apiCardsMany = (idxs: number[]) =>
  idxs.length
    ? get<Card[]>(`/cards/many?ids=${idxs.join(",")}`)
    : Promise.resolve<Card[]>([]);

// ----- card reverse-lookups -----------------------------------------------

export interface CardLookupItem {
  idx: number;
  cube_uuid?: string | null;
  deck_idx?: number | null;
  draft_id?: string | null;
  step?: number | null;
}

export interface CardLookupOut {
  card_idx: number;
  kind: "cubes" | "decks" | "in-packs" | "picked-by-humans";
  total: number;
  page: number;
  per_page: number;
  items: CardLookupItem[];
}

export const apiCardInCubes = (idx: number, page = 0, perPage = 40) =>
  get<CardLookupOut>(`/cards/${idx}/in-cubes?page=${page}&per_page=${perPage}`);
export const apiCardInDecks = (idx: number, page = 0, perPage = 40) =>
  get<CardLookupOut>(`/cards/${idx}/in-decks?page=${page}&per_page=${perPage}`);
export const apiCardInPacks = (idx: number, page = 0, perPage = 40) =>
  get<CardLookupOut>(`/cards/${idx}/in-packs?page=${page}&per_page=${perPage}`);
export const apiCardPickedByHumans = (idx: number, page = 0, perPage = 40) =>
  get<CardLookupOut>(`/cards/${idx}/picked-by-humans?page=${page}&per_page=${perPage}`);

export interface CardNeighborsOut {
  card_idx: number;
  ckpt: string;
  neighbors: Array<{ idx: number; similarity: number }>;
}
export const apiCardNeighbors = (idx: number, ckpt: string, k = 24) =>
  get<CardNeighborsOut>(`/cards/${idx}/neighbors?ckpt=${encodeURIComponent(ckpt)}&k=${k}`);

export interface CardSearchParams {
  q?: string;
  type?: string;
  cmc_min?: number;
  cmc_max?: number;
  elo_min?: number;
  elo_max?: number;
  sets?: string;
  colors?: string;
  rarity?: string;
  sort?: "elo_desc" | "elo_asc" | "name" | "cmc_asc" | "cmc_desc" | "pick_desc";
  limit?: number;
}

export const apiSearchCards = (params: CardSearchParams) => {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  return get<CardSearchResult>(`/cards/search?${qs.toString()}`);
};

// ----- picks / decks / cubes ----------------------------------------------

export const apiPick = (idx: number) => get<Pick>(`/picks/${idx}`);
export const apiRandomPick = () => get<Pick>("/picks/random/one");
export const apiCubeGroups = (limit = 120) =>
  get<CubeGroupEntry[]>(`/picks/cubes/list?limit=${limit}`);
export const apiPicksByCube = (cci: number, limit = 200) =>
  get<PickIndexEntry[]>(`/picks/by-cube/${cci}/list?limit=${limit}`);

export const apiDeck = (idx: number) => get<Deck>(`/decks/${idx}`);
export const apiDecks = (limit = 50) => get<Deck[]>(`/decks?limit=${limit}`);

export const apiCube = (idx: number) => get<Cube>(`/cubes/${idx}`);
export const apiCubes = (limit = 50) => get<Cube[]>(`/cubes?limit=${limit}`);

// ----- cube directory (Phase I) -------------------------------------------

export interface CubeDirectorySearchResult {
  items: CubeDirectoryEntry[];
}

export const apiSearchCubes = (params: { q: string; split?: "all" | "val"; limit?: number }) => {
  const qs = new URLSearchParams({ q: params.q });
  if (params.split) qs.set("split", params.split);
  if (params.limit != null) qs.set("limit", String(params.limit));
  return get<CubeDirectorySearchResult>(`/cubes/search?${qs.toString()}`);
};

export const apiCubeByUuid = (uuid: string) =>
  get<CubeDirectoryDetail>(`/cubes/by-uuid/${encodeURIComponent(uuid)}`);

export const apiResolveCube = (urlOrSlug: string) =>
  get<CubeDirectoryDetail>(`/cubes/resolve?url_or_slug=${encodeURIComponent(urlOrSlug)}`);

// ----- draft sessions (Phase J) -------------------------------------------

export interface DraftListParams {
  cube_uuid?: string;
  include_suspect?: boolean;
  limit?: number;
  offset?: number;
}

export const apiListDrafts = (params: DraftListParams = {}) => {
  const qs = new URLSearchParams();
  if (params.cube_uuid) qs.set("cube_uuid", params.cube_uuid);
  if (params.include_suspect) qs.set("include_suspect", "true");
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  return get<{ items: DraftSessionSummary[] }>(`/drafts?${qs.toString()}`);
};

export const apiGetDraft = (draftId: string) =>
  get<DraftSession>(`/drafts/${encodeURIComponent(draftId)}`);

export const apiReplayDraft = (draftId: string, ckpt: string) =>
  get<DraftReplay>(`/drafts/${encodeURIComponent(draftId)}/replay?ckpt=${encodeURIComponent(ckpt)}`);

export type SplitFilter = "all" | "train" | "val";

export const apiCubeDrafts = (
  uuid: string,
  params: {
    include_suspect?: boolean;
    split?: SplitFilter;
    limit?: number;
    offset?: number;
    cardIdxs?: number[];
  } = {},
) => {
  const qs = new URLSearchParams();
  if (params.include_suspect) qs.set("include_suspect", "true");
  if (params.split) qs.set("split", params.split);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  if (params.cardIdxs && params.cardIdxs.length) qs.set("card_idxs", params.cardIdxs.join(","));
  return get<{ items: DraftSessionSummary[] }>(`/cubes/by-uuid/${encodeURIComponent(uuid)}/drafts?${qs.toString()}`);
};

export interface CubeDeckItem {
  idx: number;
  split: "train" | "val";
}

export const apiCubeDecks = (
  uuid: string,
  params: {
    split?: SplitFilter;
    limit?: number;
    offset?: number;
    cardIdxs?: number[];
  } = {},
) => {
  const qs = new URLSearchParams();
  if (params.split) qs.set("split", params.split);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  if (params.cardIdxs && params.cardIdxs.length) qs.set("card_idxs", params.cardIdxs.join(","));
  return get<{ cube_uuid: string; total: number; items: CubeDeckItem[] }>(
    `/cubes/by-uuid/${encodeURIComponent(uuid)}/decks?${qs.toString()}`,
  );
};

// ----- metrics -------------------------------------------------------------

export const apiMetricsRunIds = () => get<{ runs: string[] }>("/metrics/runs");
export const apiRunRows = (runId: string) =>
  get<{ run_id: string; rows: MetricsRow[] }>(`/metrics/runs/${encodeURIComponent(runId)}`);

export const apiCubesByOwner = (owner: string, limit = 24) =>
  get<{ items: CubeDirectoryEntry[] }>(
    `/cubes/by-owner?owner=${encodeURIComponent(owner)}&limit=${limit}`,
  );


// ----- collapse ------------------------------------------------------------

export type CollapseSource = "all" | "val" | "train";

export interface CollapseOut {
  ckpt: string;
  sample_n: number;
  sample_n_val: number;
  sample_n_train: number;
  appearances: number[];
  model_picks: number[];
  human_picks: number[];
  // Per-pick chosen card-idx lists (length = sample_n). Populated by the
  // job runner so the attribute distribution charts can run off the same
  // job without a second inference pass. Older partials may omit these.
  human_pick_idxs?: number[];
  model_pick_idxs?: number[];
}

export const apiCollapse = (
  ckpt: string,
  sample_n = 10_000,
  cube_uuids?: string[],
  source: CollapseSource = "all",
) => {
  const qs = new URLSearchParams({ ckpt, sample_n: String(sample_n), source });
  if (cube_uuids && cube_uuids.length) qs.set("cube_uuids", cube_uuids.join(","));
  return get<CollapseOut>(`/collapse?${qs.toString()}`);
};

// Per-card collapse vectors recorded during training (sidecar npz per epoch).
// Same vector shape as CollapseOut but zero inference cost.
export type RunCollapseSplit = "val" | "train" | "train_cum";

export interface RunCollapseOut {
  run_id: string;
  epoch: number;
  split: RunCollapseSplit;
  epochs_aggregated: number;
  appearances: number[];
  model_picks: number[];
  human_picks: number[];
}

export const apiRunCollapseEpochs = (runId: string) =>
  get<{ run_id: string; epochs: number[] }>(`/runs/${encodeURIComponent(runId)}/collapse`);

export const apiRunCollapse = (runId: string, epoch: number, split: RunCollapseSplit) =>
  get<RunCollapseOut>(`/runs/${encodeURIComponent(runId)}/collapse/${epoch}?split=${split}`);

// ----- predict -------------------------------------------------------------

export const apiPredictDraft = (body: {
  ckpt: string;
  pool: number[];
  pack: number[];
}) => post<DraftPrediction>("/predict/draft", body);

export const apiPredictDeckbuilder = (body: {
  ckpt: string;
  pool: number[];
  max_spells?: number;
  max_lands?: number;
  seed_count?: number;
}) => post<DeckBuilderResult>("/predict/deckbuilder", body);


export interface RecommendOut {
  ckpt: string;
  adds: Array<[number, number]>;
  cuts: Array<[number, number]>;
}

export const apiPredictRecommend = (body: { ckpt: string; cube: number[] }) =>
  post<RecommendOut>("/predict/recommend", body);


// ----- unified cube metrics (Phase 2 — replaces scorecard) -----------------

export type CubeMetricsSource = "all" | "val" | "train";

export interface CubeMetricsOut {
  ckpt: string;
  cube_uuid: string;
  source: CubeMetricsSource;
  n_picks: number;
  n_val: number;
  n_train: number;
  // headline
  top1: number;
  top3: number;
  avg_top1_p: number;
  avg_human_p: number;
  // distribution (per-pick chosen card)
  human_pick_idxs: number[];
  model_pick_idxs: number[];
  // collapse (per-card over cube vocabulary)
  appearances: number[];
  model_picks_count: number[];
  human_picks_count: number[];
}

// ----- dataset stats ------------------------------------------------------

export interface DataStats {
  val_total: number;
  train_total: number;
}

export const apiDataStats = () => get<DataStats>("/data/stats");


// ----- async jobs (Phase N) -----------------------------------------------

export const apiStartJob = (kind: string, body: Record<string, unknown> = {}) =>
  post<{ id: string }>(`/jobs/${encodeURIComponent(kind)}`, body);

export const apiGetJob = (jobId: string) =>
  get<JobOut>(`/jobs/${encodeURIComponent(jobId)}`);

export const apiCancelJob = (jobId: string) =>
  fetch(`${API}/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
