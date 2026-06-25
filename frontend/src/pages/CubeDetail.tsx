/** Cube workspace — analyze the model from a single cube's perspective.
 *
 * Tabs:
 *   Cards     — the cube's contents
 *   Drafts    — train+val draft sessions of this cube
 *   Decks     — train+val decks built from this cube (clicking opens
 *               /cube/:uuid/deck/:idx with human vs bot side-by-side)
 *   Collapse  — pick-rate aggregation scoped to this cube
 *   Recommend — model's add/cut recommendations for this cube
 */
import * as Tabs from "@radix-ui/react-tabs";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ExternalLink, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";

import { CardImage } from "@/components/CardImage";
import { CardGrid, type CardGridItem } from "@/components/CardGrid";
import { ComparePanels } from "@/components/ComparePanels";
import {
  CubeCardFilter,
  useCubeCardFilter,
  useFilteredCubeCardIdxs,
} from "@/components/CubeCardFilter";
import { DistributionCharts } from "@/components/DistributionCharts";
import { KpiRow, KpiTile } from "@/components/KpiTile";
import { ProgressBar } from "@/components/ProgressBar";
import { useCard } from "@/hooks/useCards";
import { useCardDrawer } from "@/hooks/useCardDrawer";
import { useJob, useStartJob } from "@/hooks/useJob";
import { useCubeMetrics } from "@/hooks/useCubeMetrics";
import {
  apiCardsMany,
  apiCubeByUuid,
  apiCubeDecks,
  apiCubeDrafts,
  apiDeck,
  apiResolveCube,
  type CubeMetricsOut,
  type CubeMetricsSource,
} from "@/lib/api";
import type { Card } from "@/lib/types";
import {
  CHART_THEME,
  CHART_TOOLTIP_STYLE,
  HUMAN_COLOR,
  MODEL_COLOR,
} from "@/lib/chartTheme";
import { cn } from "@/lib/cn";
import { pushRecentCube } from "@/pages/Cubes";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function CubeDetailPage() {
  const { uuid = "" } = useParams<{ uuid: string }>();
  // Try direct UUID lookup first; fall back to /cubes/resolve so a pasted
  // slug like /cube/buildaround translates via cubecobra.com → canonical UUID.
  const {
    data: cube,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["cube-by-uuid-or-slug", uuid],
    queryFn: async () => {
      try {
        return await apiCubeByUuid(uuid);
      } catch (e) {
        if (!(e as Error).message?.includes("404")) throw e;
        return await apiResolveCube(uuid);
      }
    },
    enabled: !!uuid,
    retry: false,
  });

  // Add to recently-viewed (localStorage-backed shelf on /cubes landing).
  useEffect(() => {
    if (cube?.cube_uuid) pushRecentCube(cube.cube_uuid);
  }, [cube?.cube_uuid]);

  // If we resolved a slug, redirect the URL to the canonical /cube/<uuid> so
  // refreshes are stable and shareable.
  if (cube && cube.cube_uuid !== uuid) {
    return <Navigate to={`/cube/${cube.cube_uuid}`} replace />;
  }

  return (
    <div className="container py-8">
      <div className="mb-4">
        <Link
          to="/cubes"
          className="inline-flex items-center gap-1 text-sm text-fg-2 hover:text-fg-0"
        >
          <ChevronLeft className="size-4" /> Back to Cubes
        </Link>
      </div>

      {isLoading && <div className="text-fg-2">Loading cube…</div>}
      {error && (
        <div className="text-bad">
          Could not load cube: {(error as Error).message}
        </div>
      )}

      {cube && (
        <>
          <header className="border-b border-border pb-5 mb-6 flex items-start gap-5">
            {cube.image_uri && (
              <img
                src={cube.image_uri}
                alt=""
                className="w-40 h-24 object-cover rounded-md border border-border shrink-0"
              />
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h1 className="text-2xl font-bold text-fg-0 truncate">
                  {cube.name || "(unnamed cube)"}
                </h1>
                {cube.has_val_drafts && (
                  <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full bg-good text-white">
                    val
                  </span>
                )}
              </div>
              <div className="text-fg-2 text-sm mt-1">
                by{" "}
                <span className="font-medium text-fg-1">
                  {cube.owner || "—"}
                </span>
                <span className="text-fg-3"> · </span>
                {cube.card_count} cards
                <span className="text-fg-3"> · </span>
                <a
                  href={`https://cubecobra.com/cube/overview/${cube.cube_uuid}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-link hover:underline"
                >
                  cubecobra.com <ExternalLink className="size-3" />
                </a>
              </div>
              <div className="text-fg-3 text-xs font-mono mt-1 truncate">
                {cube.cube_uuid}
              </div>
            </div>
          </header>

          <CubeCardFilter />

          <CubeTabs cubeCards={cube.cards} uuid={uuid} />
        </>
      )}
    </div>
  );
}

const TOP_TABS = ["cards", "drafts", "decks", "metrics", "recommend"] as const;
type TopTab = (typeof TOP_TABS)[number];

function CubeTabs({ cubeCards, uuid }: { cubeCards: number[]; uuid: string }) {
  const [params, setParams] = useSearchParams();
  const tab = (TOP_TABS as readonly string[]).includes(params.get("tab") ?? "")
    ? (params.get("tab") as TopTab)
    : "cards";
  const setTab = (next: string) => {
    const p = new URLSearchParams(params);
    if (next === "cards") p.delete("tab");
    else p.set("tab", next);
    setParams(p, { replace: true });
  };
  return (
    <Tabs.Root value={tab} onValueChange={setTab} className="space-y-5">
      <Tabs.List className="flex border-b border-border flex-wrap">
        <TabTrigger value="cards">Cards ({cubeCards.length})</TabTrigger>
        <TabTrigger value="drafts">Drafts</TabTrigger>
        <TabTrigger value="decks">Decks</TabTrigger>
        <TabTrigger value="metrics">Metrics</TabTrigger>
        <TabTrigger value="recommend">Recommend</TabTrigger>
      </Tabs.List>
      <Tabs.Content value="cards">
        <CardsTab cards={cubeCards} />
      </Tabs.Content>
      <Tabs.Content value="drafts">
        <DraftsTab uuid={uuid} cubeCards={cubeCards} />
      </Tabs.Content>
      <Tabs.Content value="decks">
        <DecksTab uuid={uuid} cubeCards={cubeCards} />
      </Tabs.Content>
      <Tabs.Content value="metrics">
        <MetricsTab uuid={uuid} cubeCards={cubeCards} />
      </Tabs.Content>
      <Tabs.Content value="recommend">
        <RecommendTab cards={cubeCards} />
      </Tabs.Content>
    </Tabs.Root>
  );
}

function TabTrigger({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}) {
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        "px-4 py-2 text-sm font-medium border-b-2 border-transparent",
        "text-fg-2 hover:text-fg-0 data-[state=active]:text-fg-0",
        "data-[state=active]:border-accent transition-colors",
      )}
    >
      {children}
    </Tabs.Trigger>
  );
}

function CardsTab({ cards }: { cards: number[] }) {
  const { predicate, isActive } = useCubeCardFilter();
  const { data: meta } = useQuery({
    queryKey: [
      "cards-many",
      "cube",
      cards.length,
      cards[0] ?? -1,
      cards[cards.length - 1] ?? -1,
    ],
    queryFn: () => apiCardsMany(cards),
    enabled: cards.length > 0,
    staleTime: Infinity,
  });
  const filtered = useMemo(() => {
    if (!isActive || !meta) return cards;
    const ok = new Set(meta.filter(predicate).map((c) => c.idx));
    return cards.filter((idx) => ok.has(idx));
  }, [cards, meta, isActive, predicate]);
  return (
    <>
      {isActive && (
        <div className="text-fg-3 text-xs mb-3">
          Showing {filtered.length} of {cards.length} (filter active)
        </div>
      )}
      <CardGrid items={filtered.map((idx) => ({ idx }))} size="sm" />
    </>
  );
}

function DraftsTab({ uuid, cubeCards }: { uuid: string; cubeCards: number[] }) {
  const [split, setSplit] = useState<"all" | "train" | "val">("all");
  const filterIdxs = useFilteredCubeCardIdxs(cubeCards);
  const { data, isLoading } = useQuery({
    queryKey: ["cube-drafts", uuid, split, filterIdxs?.join(",") ?? "all"],
    queryFn: () =>
      apiCubeDrafts(uuid, {
        split,
        limit: 200,
        ...(filterIdxs ? { cardIdxs: filterIdxs } : {}),
      }),
    enabled: !!uuid,
  });
  const drafts = data?.items ?? [];
  return (
    <div className="space-y-4">
      <SplitToggle
        value={split}
        onChange={setSplit}
        counts={{ all: drafts.length }}
      />
      {isLoading && (
        <div className="text-fg-3 text-sm py-6">Loading drafts…</div>
      )}
      {!isLoading && drafts.length === 0 && (
        <div className="text-fg-3 text-sm py-6">
          No {split === "all" ? "" : `${split} `}drafts indexed for this cube.
          {split !== "val" && " Train sidecars need a re-run of "}
          {split !== "val" && (
            <code className="font-mono text-xs">
              node scripts/process_data.js --rebuild-drafts
            </code>
          )}
        </div>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {drafts.map((d) => (
          <Link
            key={d.draft_id}
            to={`/drafts/${d.draft_id}`}
            className="rounded-md border border-border bg-bg-1 p-3 hover:border-accent hover:bg-bg-2 transition-colors flex items-start gap-3"
          >
            <Play className="size-5 text-accent shrink-0 mt-0.5" />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-fg-0 text-sm font-medium font-mono truncate">
                  {d.draft_id}
                </span>
                <SplitBadge split={d.split} />
              </div>
              <div className="text-fg-3 text-xs nums mt-0.5">
                {d.n_picks} picks
                {d.owner && (
                  <span className="ml-2 text-fg-2">
                    · {d.owner.slice(0, 10)}…
                  </span>
                )}
                {d.suspect && <span className="ml-2 text-bad">suspect</span>}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

function SplitBadge({ split }: { split: "train" | "val" }) {
  return (
    <span
      className={cn(
        "text-[9px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded",
        split === "val" ? "bg-good text-white" : "bg-bg-3 text-fg-2",
      )}
    >
      {split}
    </span>
  );
}

function SplitToggle({
  value,
  onChange,
}: {
  value: "all" | "train" | "val";
  onChange: (v: "all" | "train" | "val") => void;
  counts?: { all?: number };
}) {
  return (
    <div className="inline-flex gap-1 p-1 bg-bg-3 rounded-md">
      {(["all", "val", "train"] as const).map((k) => (
        <button
          key={k}
          type="button"
          onClick={() => onChange(k)}
          className={cn(
            "px-3 py-1 rounded text-xs font-medium capitalize",
            value === k
              ? "bg-bg-1 text-fg-0 shadow-sm"
              : "text-fg-3 hover:text-fg-1",
          )}
        >
          {k}
        </button>
      ))}
    </div>
  );
}

function DecksTab({ uuid, cubeCards }: { uuid: string; cubeCards: number[] }) {
  const [split, setSplit] = useState<"all" | "train" | "val">("all");
  const filterIdxs = useFilteredCubeCardIdxs(cubeCards);
  const { data, isLoading } = useQuery({
    queryKey: ["cube-decks", uuid, split, filterIdxs?.join(",") ?? "all"],
    queryFn: () =>
      apiCubeDecks(uuid, {
        split,
        limit: 60,
        ...(filterIdxs ? { cardIdxs: filterIdxs } : {}),
      }),
    enabled: !!uuid,
  });
  const items = data?.items ?? [];
  return (
    <div className="space-y-4">
      <SplitToggle value={split} onChange={setSplit} />
      {isLoading && (
        <div className="text-fg-3 text-sm py-6">Loading decks…</div>
      )}
      {!isLoading && items.length === 0 && (
        <div className="text-fg-3 text-sm py-6">
          No {split === "all" ? "" : `${split} `}decks indexed for this cube.
        </div>
      )}
      {items.length > 0 && (
        <>
          <div className="text-fg-3 text-xs">
            Showing {items.length} of {data?.total ?? items.length}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {items.map((it) => (
              <DeckCard
                key={`${it.split}-${it.idx}`}
                uuid={uuid}
                idx={it.idx}
                split={it.split}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function DeckCard({
  uuid,
  idx,
  split,
}: {
  uuid: string;
  idx: number;
  split: "train" | "val";
}) {
  // Val decks load a 6-card preview; train decks render label-only until the
  // split-aware deck reader lands. The whole card links to the deck workspace.
  // The cube-anchor filter is applied server-side via card_idxs, so this row
  // doesn't need to re-check anything client-side.
  const { data: deck } = useQuery({
    queryKey: ["deck", split, idx],
    queryFn: () => apiDeck(idx),
    enabled: split === "val",
  });
  const preview: CardGridItem[] = (deck?.mainboard ?? [])
    .slice(0, 6)
    .map((cardIdx) => ({ idx: cardIdx }));
  return (
    <Link
      to={`/cube/${uuid}/deck/${idx}?split=${split}`}
      className="block rounded-md border border-border bg-bg-1 p-4 hover:border-accent transition-colors"
    >
      <div className="text-sm font-semibold text-fg-0 mb-2 flex items-center gap-2">
        Deck #{idx}
        <SplitBadge split={split} />
        {deck && (
          <span className="text-fg-3 font-normal">
            · {deck.mainboard.length} cards
          </span>
        )}
      </div>
      {split === "val" && deck && <CardGrid items={preview} size="xs" />}
      {split === "train" && (
        <div className="text-fg-3 text-xs">
          Train deck inspection coming with the split-aware endpoint.
        </div>
      )}
    </Link>
  );
}

// ----- new tabs (Phase P) --------------------------------------------------

/** Cube workspace Metrics tab: Headline + attribute Distribution + per-card
 * Collapse. All three subtabs read from a single `cube_metrics` job per ckpt
 * — switching subtabs never re-fires inference. The source pill (all/train/
 * val) at the top drives that single job.
 */
const METRICS_SUBTABS = ["headline", "distribution", "pickrate"] as const;
type MetricsSubtab = (typeof METRICS_SUBTABS)[number];

function MetricsTab({
  uuid,
  cubeCards,
}: {
  uuid: string;
  cubeCards: number[];
}) {
  const [params, setParams] = useSearchParams();
  const sourceParam = params.get("source");
  const source: CubeMetricsSource =
    sourceParam === "val" || sourceParam === "train" ? sourceParam : "all";
  const setSource = (next: CubeMetricsSource) => {
    const p = new URLSearchParams(params);
    if (next === "all") p.delete("source");
    else p.set("source", next);
    setParams(p, { replace: true });
  };
  const subtab: MetricsSubtab = (METRICS_SUBTABS as readonly string[]).includes(
    params.get("subtab") ?? "",
  )
    ? (params.get("subtab") as MetricsSubtab)
    : "headline";
  const setSubtab = (next: string) => {
    const p = new URLSearchParams(params);
    if (next === "headline") p.delete("subtab");
    else p.set("subtab", next);
    setParams(p, { replace: true });
  };

  return (
    <Tabs.Root value={subtab} onValueChange={setSubtab} className="space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <Tabs.List className="flex gap-1 border-b border-border">
          {[
            ["headline", "Headline"],
            ["distribution", "Distribution"],
            ["pickrate", "Pick percentage"],
          ].map(([key, label]) => (
            <Tabs.Trigger
              key={key}
              value={key}
              className={cn(
                "px-3 py-1.5 text-xs font-medium text-fg-2 hover:text-fg-0",
                "data-[state=active]:text-fg-0 data-[state=active]:border-b-2 data-[state=active]:border-accent",
                "transition-colors -mb-px",
              )}
            >
              {label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        <SourcePill source={source} onChange={setSource} />
      </div>

      <ComparePanels
        render={(ckpt) => (
          <MetricsTabPerCkpt
            ckpt={ckpt}
            uuid={uuid}
            cubeCards={cubeCards}
            source={source}
          />
        )}
      />
    </Tabs.Root>
  );
}

function MetricsTabPerCkpt({
  ckpt,
  uuid,
  cubeCards,
  source,
}: {
  ckpt: string;
  uuid: string;
  cubeCards: number[];
  source: CubeMetricsSource;
}) {
  const metrics = useCubeMetrics(ckpt, uuid, source);
  const data = metrics.data;
  return (
    <div className="space-y-4">
      {metrics.job && metrics.job.status !== "done" && (
        <ProgressBar
          job={metrics.job}
          label={`Cube metrics (${ckpt}, ${source})`}
        />
      )}
      {data && (
        <div className="text-fg-3 text-xs nums">
          {data.n_picks.toLocaleString()} picks scored ·{" "}
          {data.n_val.toLocaleString()} val + {data.n_train.toLocaleString()}{" "}
          train
        </div>
      )}
      <Tabs.Content value="headline">
        <HeadlineSubtab data={data} />
      </Tabs.Content>
      <Tabs.Content value="distribution">
        <DistributionSubtab data={data} cubeCards={cubeCards} />
      </Tabs.Content>
      <Tabs.Content value="pickrate">
        <CollapseSubtabUnified data={data} />
      </Tabs.Content>
    </div>
  );
}

function HeadlineSubtab({ data }: { data: CubeMetricsOut | null }) {
  if (!data)
    return <div className="text-fg-3 text-sm py-6">Running cube metrics…</div>;
  if (data.n_picks === 0) {
    return (
      <div className="text-fg-3 text-sm py-6">
        No picks indexed for this cube under the selected source.
      </div>
    );
  }
  return (
    <KpiRow>
      <KpiTile label="picks scored" value={data.n_picks.toLocaleString()} />
      <KpiTile
        label="top1 agreement"
        value={(data.top1 * 100).toFixed(1) + "%"}
      />
      <KpiTile
        label="top3 agreement"
        value={(data.top3 * 100).toFixed(1) + "%"}
      />
      <KpiTile
        label="avg top1 conf"
        value={(data.avg_top1_p * 100).toFixed(1) + "%"}
      />
      <KpiTile
        label="avg P(human)"
        value={(data.avg_human_p * 100).toFixed(1) + "%"}
      />
    </KpiRow>
  );
}

function DistributionSubtab({
  data,
  cubeCards,
}: {
  data: CubeMetricsOut | null;
  cubeCards: number[];
}) {
  const { predicate, isActive } = useCubeCardFilter();
  const needed = useMemo(() => {
    const s = new Set<number>(cubeCards);
    if (data) {
      for (const i of data.human_pick_idxs) s.add(i);
      for (const i of data.model_pick_idxs) s.add(i);
    }
    return Array.from(s);
  }, [data, cubeCards]);

  const { data: cards } = useQuery({
    queryKey: ["cards-many", needed.length, needed[0] ?? -1],
    queryFn: () => apiCardsMany(needed),
    enabled: needed.length > 0,
    staleTime: Infinity,
  });

  const cardsByIdx = useMemo(() => {
    const m = new Map<number, Card>();
    for (const c of cards ?? []) m.set(c.idx, c);
    return m;
  }, [cards]);

  const series = useMemo(() => {
    if (!data) return null;
    const filterIdxs = (idxs: number[]) => {
      if (!isActive) return idxs;
      return idxs.filter((i) => {
        const c = cardsByIdx.get(i);
        return c ? predicate(c) : false;
      });
    };
    return {
      human: filterIdxs(data.human_pick_idxs),
      model: filterIdxs(data.model_pick_idxs),
      cube: filterIdxs(cubeCards),
    };
  }, [data, cubeCards, cardsByIdx, predicate, isActive]);

  if (!data)
    return <div className="text-fg-3 text-sm py-6">Running cube metrics…</div>;
  if (data.n_picks === 0) {
    return (
      <div className="text-fg-3 text-sm py-6">
        No picks indexed for this cube under the selected source.
      </div>
    );
  }

  return series ? (
    <DistributionCharts series={series} cardsByIdx={cardsByIdx} />
  ) : null;
}

/** Reads the unified cube_metrics result and adapts it to CollapseBody's
 * existing prop shape, so we get the per-card pick-percentage view with
 * zero extra inference cost. */
function CollapseSubtabUnified({ data }: { data: CubeMetricsOut | null }) {
  if (!data)
    return <div className="text-fg-3 text-sm py-6">Running cube metrics…</div>;
  if (data.n_picks === 0) {
    return (
      <div className="text-fg-3 text-sm py-6">
        No picks indexed for this cube under the selected source.
      </div>
    );
  }
  return (
    <CollapseBody
      data={{
        sample_n: data.n_picks,
        appearances: data.appearances,
        model_picks: data.model_picks_count,
        human_picks: data.human_picks_count,
      }}
    />
  );
}

function SourcePill({
  source,
  onChange,
}: {
  source: CubeMetricsSource;
  onChange: (s: CubeMetricsSource) => void;
}) {
  return (
    <div className="inline-flex gap-1 p-1 bg-bg-3 rounded-md">
      {(["all", "train", "val"] as const).map((k) => (
        <button
          key={k}
          type="button"
          onClick={() => onChange(k)}
          className={cn(
            "px-3 py-1 rounded text-xs font-medium capitalize",
            source === k
              ? "bg-bg-1 text-fg-0 shadow-sm"
              : "text-fg-3 hover:text-fg-1",
          )}
        >
          {k}
        </button>
      ))}
    </div>
  );
}

/** Per-cube outliers + a folded-away global histogram. Headline is the
 * outlier table because for one cube the actionable question is which
 * specific cards the model disagrees on — the histogram shape, while still
 * useful, is secondary.
 */
function CollapseBody({
  data,
}: {
  data: {
    sample_n: number;
    appearances: number[];
    model_picks: number[];
    human_picks: number[];
  };
}) {
  const [minApps, setMinApps] = useState(1);
  const [topN, setTopN] = useState(30);
  const [modelRange, setModelRange] = useState<[number, number]>([0, 100]);
  const [humanRange, setHumanRange] = useState<[number, number]>([0, 100]);
  const { predicate, isActive } = useCubeCardFilter();

  // Collect every idx that hits the appearance floor; metadata fetch uses
  // this set so the predicate filter has something to evaluate against.
  const eligibleIdxs = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < data.appearances.length; i++) {
      if (data.appearances[i] >= minApps) out.push(i);
    }
    return out;
  }, [data, minApps]);

  const { data: meta } = useQuery({
    queryKey: [
      "cards-many",
      "pickrate",
      eligibleIdxs.length,
      eligibleIdxs[0] ?? -1,
    ],
    queryFn: () => apiCardsMany(eligibleIdxs),
    enabled: eligibleIdxs.length > 0,
    staleTime: Infinity,
  });
  const keep = useMemo(() => {
    if (!isActive) return null;
    if (!meta) return new Set<number>();
    return new Set(meta.filter(predicate).map((c) => c.idx));
  }, [isActive, meta, predicate]);

  const { overpicks, underpicks, totalAboveThreshold } = useMemo(() => {
    type Row = {
      idx: number;
      modelRate: number;
      humanRate: number;
      appearances: number;
      delta: number;
    };
    const rows: Row[] = [];
    const mLo = modelRange[0] / 100,
      mHi = modelRange[1] / 100;
    const hLo = humanRange[0] / 100,
      hHi = humanRange[1] / 100;
    for (const i of eligibleIdxs) {
      if (keep && !keep.has(i)) continue;
      const ap = data.appearances[i];
      const mr = data.model_picks[i] / ap;
      const hr = data.human_picks[i] / ap;
      if (mr < mLo || mr > mHi) continue;
      if (hr < hLo || hr > hHi) continue;
      rows.push({
        idx: i,
        modelRate: mr,
        humanRate: hr,
        appearances: ap,
        delta: mr - hr,
      });
    }
    const byDelta = [...rows].sort((a, b) => b.delta - a.delta);
    return {
      overpicks: byDelta.slice(0, topN),
      underpicks: byDelta.slice(-topN).reverse(),
      totalAboveThreshold: rows.length,
    };
  }, [data, eligibleIdxs, keep, topN, modelRange, humanRange]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 flex-wrap text-xs">
        <label className="flex items-center gap-1 text-fg-3">
          Min appearances
          <input
            type="number"
            min={1}
            max={200}
            value={minApps}
            onChange={(e) =>
              setMinApps(Math.max(1, Number(e.target.value) || 1))
            }
            className="w-16 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 nums text-fg-0"
          />
        </label>
        <label className="flex items-center gap-1 text-fg-3">
          Show top
          <input
            type="number"
            min={5}
            max={200}
            value={topN}
            onChange={(e) => setTopN(Math.max(5, Number(e.target.value) || 30))}
            className="w-16 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 nums text-fg-0"
          />
        </label>
        <RatePill
          label="Model %"
          tone="model"
          range={modelRange}
          onChange={setModelRange}
        />
        <RatePill
          label="Human %"
          tone="human"
          range={humanRange}
          onChange={setHumanRange}
        />
        <span className="text-fg-3">
          {totalAboveThreshold.toLocaleString()} eligible cards
          {isActive && " (filtered)"}
        </span>
      </div>

      {totalAboveThreshold === 0 ? (
        <div className="rounded-lg border border-border bg-bg-2 p-6 text-fg-2 text-sm">
          No cards match the current filter + thresholds.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <OutlierTable
              title="Model over-picks vs humans"
              subtitle={`Top ${topN} cards where the model picks more often than humans (Δ = model − human rate)`}
              rows={overpicks}
              tone="model"
            />
            <OutlierTable
              title="Model under-picks vs humans"
              subtitle={`Top ${topN} cards where humans pick more often than the model`}
              rows={underpicks}
              tone="human"
            />
          </div>

          <details className="rounded-lg border border-border bg-bg-2 p-4 group">
            <summary className="cursor-pointer text-sm font-semibold text-fg-0 select-none">
              Pick-rate distribution histogram
              <span className="text-fg-3 font-normal ml-2">
                ({totalAboveThreshold.toLocaleString()} cards · minApps{" "}
                {minApps})
              </span>
            </summary>
            <div className="mt-3">
              <HistogramChart data={data} minApps={minApps} keep={keep} />
            </div>
          </details>
        </>
      )}
    </div>
  );
}

function RatePill({
  label,
  tone,
  range,
  onChange,
}: {
  label: string;
  tone: "model" | "human";
  range: [number, number];
  onChange: (r: [number, number]) => void;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-1 rounded border border-border-subtle",
        tone === "model" ? "text-model" : "text-human",
      )}
    >
      {label}
      <input
        type="number"
        min={0}
        max={100}
        value={range[0]}
        onChange={(e) =>
          onChange([clamp(Number(e.target.value) || 0), range[1]])
        }
        className="w-12 bg-bg-3 border border-border-subtle rounded px-1 py-0.5 nums text-fg-0"
      />
      <span className="text-fg-3">–</span>
      <input
        type="number"
        min={0}
        max={100}
        value={range[1]}
        onChange={(e) =>
          onChange([range[0], clamp(Number(e.target.value) || 100)])
        }
        className="w-12 bg-bg-3 border border-border-subtle rounded px-1 py-0.5 nums text-fg-0"
      />
    </span>
  );
}

function clamp(n: number): number {
  return Math.max(0, Math.min(100, n));
}

function OutlierTable({
  title,
  subtitle,
  rows,
  tone,
}: {
  title: string;
  subtitle: string;
  rows: Array<{
    idx: number;
    modelRate: number;
    humanRate: number;
    appearances: number;
    delta: number;
  }>;
  tone: "model" | "human";
}) {
  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-fg-0">{title}</h3>
        <p className="text-fg-3 text-xs mt-0.5">{subtitle}</p>
      </div>
      <ul className="divide-y divide-border-subtle">
        {rows.map((r) => (
          <OutlierRow key={r.idx} row={r} tone={tone} />
        ))}
      </ul>
    </section>
  );
}

function OutlierRow({
  row,
  tone,
}: {
  row: {
    idx: number;
    modelRate: number;
    humanRate: number;
    appearances: number;
    delta: number;
  };
  tone: "model" | "human";
}) {
  const { data: card } = useCard(row.idx);
  const openDrawer = useCardDrawer((s) => s.open);
  const deltaPct = (row.delta * 100).toFixed(1);
  const deltaClass = tone === "model" ? "text-model" : "text-human";
  return (
    <li>
      <button
        type="button"
        onClick={() => openDrawer(row.idx)}
        className="w-full flex items-center gap-3 py-2 text-left hover:bg-bg-2 rounded-sm px-2 -mx-2"
      >
        <div className="w-10 shrink-0">
          <CardImage idx={row.idx} disableDrawer />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-fg-0 text-sm font-medium truncate">
            {card?.name ?? "…"}
          </div>
          <div className="text-fg-3 text-xs nums">
            model {(row.modelRate * 100).toFixed(1)}%{" · "}human{" "}
            {(row.humanRate * 100).toFixed(1)}%{" · "}
            {row.appearances} pack{row.appearances === 1 ? "" : "s"}
          </div>
        </div>
        <div className={cn("text-sm font-bold nums shrink-0", deltaClass)}>
          {row.delta > 0 ? "+" : ""}
          {deltaPct}%
        </div>
      </button>
    </li>
  );
}

function HistogramChart({
  data,
  minApps,
  keep,
}: {
  data: { appearances: number[]; model_picks: number[]; human_picks: number[] };
  minApps: number;
  keep: Set<number> | null;
}) {
  const BUCKETS: Array<[number, number, string]> = [
    [0.0, 0.005, "0%"],
    [0.005, 0.05, "<5%"],
    [0.05, 0.1, "5–10%"],
    [0.1, 0.2, "10–20%"],
    [0.2, 0.3, "20–30%"],
    [0.3, 0.5, "30–50%"],
    [0.5, 0.7, "50–70%"],
    [0.7, 0.9, "70–90%"],
    [0.9, 0.95, "90–95%"],
    [0.95, 1.001, "≥95%"],
  ];
  const histogram = BUCKETS.map(([lo, hi, label]) => {
    let m = 0,
      h = 0;
    for (let i = 0; i < data.appearances.length; i++) {
      const ap = data.appearances[i];
      if (ap < minApps) continue;
      if (keep && !keep.has(i)) continue;
      const mr = data.model_picks[i] / ap;
      const hr = data.human_picks[i] / ap;
      if (mr >= lo && mr < hi) m++;
      if (hr >= lo && hr < hi) h++;
    }
    return { label, model: m, human: h };
  });
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart
        data={histogram}
        margin={{ top: 8, right: 12, left: -8, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
        <XAxis
          dataKey="label"
          stroke={CHART_THEME.axis}
          tick={{ fontSize: 11 }}
        />
        <YAxis stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} />
        <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="model" fill={MODEL_COLOR} name="Model" />
        <Bar dataKey="human" fill={HUMAN_COLOR} name="Human" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function RecommendTab({ cards }: { cards: number[] }) {
  return (
    <ComparePanels
      render={(ckpt) => <RecommendForCkpt ckpt={ckpt} cards={cards} />}
    />
  );
}

function RecommendForCkpt({ ckpt, cards }: { ckpt: string; cards: number[] }) {
  const start = useStartJob("recommend");
  const [jobId, setJobId] = useState<string | null>(null);
  const lastKey = useRef<string | null>(null);
  const key = `${ckpt}::${cards.length}`;
  useEffect(() => {
    if (!ckpt || cards.length === 0 || lastKey.current === key) return;
    lastKey.current = key;
    setJobId(null);
    start.mutate(
      { ckpt, cube: cards },
      { onSuccess: ({ id }) => setJobId(id) },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const job = useJob(jobId);
  const data = (job.data?.result ?? null) as {
    adds: Array<[number, number]>;
    cuts: Array<[number, number]>;
  } | null;

  const { predicate, isActive } = useCubeCardFilter();
  // Fetch metadata for every recommended idx so we can filter both adds and
  // cuts by the predicate. Bounded (max 100 cards) so this is cheap.
  const allIdxs = useMemo(() => {
    if (!data) return [] as number[];
    const s = new Set<number>();
    for (const [i] of data.adds) s.add(i);
    for (const [i] of data.cuts) s.add(i);
    return Array.from(s);
  }, [data]);
  const { data: meta } = useQuery({
    queryKey: ["cards-many", "recommend", allIdxs.length, allIdxs[0] ?? -1],
    queryFn: () => apiCardsMany(allIdxs),
    enabled: allIdxs.length > 0,
    staleTime: Infinity,
  });
  const keep = useMemo(() => {
    if (!isActive) return null;
    if (!meta) return new Set<number>();
    return new Set(meta.filter(predicate).map((c) => c.idx));
  }, [isActive, meta, predicate]);

  if (job.data && job.data.status !== "done") {
    return (
      <div className="space-y-3">
        <ProgressBar job={job.data} label={`Recommend (${ckpt})`} />
        <div className="text-fg-3 text-sm">Running cube recommender…</div>
      </div>
    );
  }
  if (!data) return null;

  const adds = keep ? data.adds.filter(([i]) => keep.has(i)) : data.adds;
  const cuts = keep ? data.cuts.filter(([i]) => keep.has(i)) : data.cuts;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      <section className="rounded-lg border border-border bg-bg-2 p-4">
        <h3 className="text-sm font-semibold text-fg-0 mb-3">
          Adds (highest recon, not in cube)
        </h3>
        <CardGrid
          items={adds.slice(0, 24).map(
            ([idx, p]): CardGridItem => ({
              idx,
              caption: (
                <span className="nums text-good">{(p * 100).toFixed(1)}%</span>
              ),
            }),
          )}
          size="xs"
        />
      </section>
      <section className="rounded-lg border border-border bg-bg-2 p-4">
        <h3 className="text-sm font-semibold text-fg-0 mb-3">
          Cuts (lowest recon, in cube)
        </h3>
        <CardGrid
          items={cuts.slice(0, 24).map(
            ([idx, p]): CardGridItem => ({
              idx,
              caption: (
                <span className="nums text-bad">{(p * 100).toFixed(1)}%</span>
              ),
            }),
          )}
          size="xs"
        />
      </section>
    </div>
  );
}
