/** Distribution — global model behavior over a configurable val + train sample.
 *
 * Two stacked views off the same collapse job:
 *   - Attribute distribution (color / CMC / type) of human vs model picks
 *   - Pick-rate collapse: per-card model vs human pick rate, bucket histogram,
 *     and outlier tables (over/under-picked).
 *
 * Sample size is dialed by % of val (default 100) + % of train (default 0).
 * The collapse job streams partials so both views fill in as inference runs.
 */
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

import { CardGrid, type CardGridItem } from "@/components/CardGrid";
import { ComparePanels } from "@/components/ComparePanels";
import { CubeFilterBar, useCubeFilter } from "@/components/CubeFilterBar";
import { DistributionCharts } from "@/components/DistributionCharts";
import { KpiRow, KpiTile } from "@/components/KpiTile";
import { ProgressBar } from "@/components/ProgressBar";
import { useJob, useStartJob } from "@/hooks/useJob";
import { apiCardsMany, apiDataStats, type CollapseOut } from "@/lib/api";
import { CHART_THEME, CHART_TOOLTIP_STYLE, HUMAN_COLOR, MODEL_COLOR } from "@/lib/chartTheme";
import { cn } from "@/lib/cn";
import type { Card } from "@/lib/types";




const BUCKETS: Array<[number, number, string]> = [
  [0.00, 0.005, "0%"],
  [0.005, 0.05, "<5%"],
  [0.05, 0.10, "5-10%"],
  [0.10, 0.20, "10-20%"],
  [0.20, 0.30, "20-30%"],
  [0.30, 0.50, "30-50%"],
  [0.50, 0.70, "50-70%"],
  [0.70, 0.90, "70-90%"],
  [0.90, 0.95, "90-95%"],
  [0.95, 1.001, "≥95%"],
];


export function CollapsePage() {
  return (
    <div className="container py-8 space-y-6">
      <header className="border-b border-border pb-4">
        <h1 className="text-2xl font-bold text-fg-0">Distribution — global model behavior</h1>
        <p className="text-fg-2 text-sm mt-1">
          Attribute distribution (color / CMC / type) on top, per-card
          pick-rate collapse below. One job feeds both. Default scores the
          full val set; dial train up to push further.
        </p>
      </header>
      <DistributionPanel />
    </div>
  );
}


/** The body of the Distribution view, sans page chrome. Used at /collapse
 * (legacy) and as the Distribution tab of /metrics. */
export function DistributionPanel() {
  const [valPct, setValPct] = useState(100);
  const [trainPct, setTrainPct] = useState(0);
  // Bumping runKey re-fires the job with the latest sliders.
  const [runKey, setRunKey] = useState(0);

  // Real dataset totals so % maps to an honest absolute count.
  const { data: stats } = useQuery({
    queryKey: ["data-stats"],
    queryFn: apiDataStats,
    staleTime: Infinity,
  });
  const valTotal = stats?.val_total ?? 0;
  const trainTotal = stats?.train_total ?? 0;
  const valCount = Math.round((valPct / 100) * valTotal);
  const trainCount = Math.round((trainPct / 100) * trainTotal);

  return (
    <div className="space-y-6">
      <CubeFilterBar />

      <div className="rounded-lg border border-border bg-bg-2 p-4 flex flex-wrap items-end gap-6">
        <Field label={`% val: ${valPct}%`}>
          <input type="range" min={1} max={100} step={1}
            value={valPct} onChange={(e) => setValPct(Number(e.target.value))}
            className="w-72 accent-accent" />
          <span className="text-fg-3 text-xs nums">
            ~{valCount.toLocaleString()} / {valTotal.toLocaleString()} picks
          </span>
        </Field>
        <Field label={`% train: ${trainPct}%`}>
          <input type="range" min={0} max={100} step={1}
            value={trainPct} onChange={(e) => setTrainPct(Number(e.target.value))}
            className="w-72 accent-accent" />
          <span className="text-fg-3 text-xs nums">
            ~{trainCount.toLocaleString()} / {trainTotal.toLocaleString()} picks
          </span>
        </Field>
        <button
          type="button"
          onClick={() => setRunKey((k) => k + 1)}
          className="px-4 py-1.5 rounded-md bg-accent text-white font-medium text-sm hover:bg-accent-hover"
        >
          Run
        </button>
      </div>

      <ComparePanels render={(ckpt) => (
        <CollapseForCkpt
          ckpt={ckpt} valCount={valCount} trainCount={trainCount} runKey={runKey}
        />
      )} />
    </div>
  );
}


function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] uppercase tracking-wider font-bold text-fg-3">{label}</span>
      {children}
    </div>
  );
}


function CollapseForCkpt({
  ckpt, valCount, trainCount, runKey,
}: {
  ckpt: string;
  valCount: number;
  trainCount: number;
  runKey: number;
}) {
  const { selectedUuids } = useCubeFilter();
  const start = useStartJob("collapse");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const lastRunKey = useRef(0);

  useEffect(() => {
    if (runKey === 0 || runKey === lastRunKey.current) return;
    lastRunKey.current = runKey;
    start.mutate(
      {
        ckpt,
        sample_n: valCount,
        cube_uuids: selectedUuids,
        include_train_sample: trainCount,
      },
      { onSuccess: ({ id }) => setJobId(id) },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey, ckpt]);

  const live: CollapseOut | null = useMemo(() => {
    const src = job.data?.result ?? job.data?.partial;
    return (src as unknown as CollapseOut) ?? null;
  }, [job.data?.result, job.data?.partial]);

  return (
    <div className="space-y-4">
      {jobId && job.data && (
        <ProgressBar job={job.data} label={`Distribution (${ckpt})`} />
      )}
      {!jobId && (
        <div className="rounded-md border border-border-subtle bg-bg-1 p-6 text-center text-fg-3 text-sm">
          Click <span className="text-fg-1 font-medium">Run</span> above to start the inference job.
          Full val (329K picks) takes ~30s warm, longer cold.
        </div>
      )}
      {live ? <DistributionBody data={live} /> : null}
    </div>
  );
}


function DistributionBody({ data }: { data: CollapseOut }) {
  return (
    <div className="space-y-6">
      <AttributeCharts data={data} />
      <CollapseBody data={data} />
    </div>
  );
}


/** Color / CMC / type distribution from the collapse job's per-pick idx lists.
 * Caps unique-card metadata fetch at 2000 to keep the URL sane. */
function AttributeCharts({ data }: { data: CollapseOut }) {
  const uniqueIdxs = useMemo(() => {
    const s = new Set<number>();
    for (const i of data.human_pick_idxs ?? []) s.add(i);
    for (const i of data.model_pick_idxs ?? []) s.add(i);
    return Array.from(s).slice(0, 2000);
  }, [data]);

  const { data: cards } = useQuery({
    queryKey: ["cards-many", "dist-global", uniqueIdxs.length, uniqueIdxs[0] ?? -1],
    queryFn: () => apiCardsMany(uniqueIdxs),
    enabled: uniqueIdxs.length > 0,
    staleTime: Infinity,
  });

  const cardsByIdx = useMemo(() => {
    const m = new Map<number, Card>();
    for (const c of cards ?? []) m.set(c.idx, c);
    return m;
  }, [cards]);

  if (!data.human_pick_idxs?.length) {
    return null;  // older partial without idx lists
  }

  return (
    <section className="rounded-lg border border-border bg-bg-2 p-4 space-y-3">
      <div className="text-sm font-semibold text-fg-0">Attribute distribution</div>
      <DistributionCharts
        series={{ human: data.human_pick_idxs, model: data.model_pick_idxs }}
        cardsByIdx={cardsByIdx}
      />
    </section>
  );
}


/** Per-card pick-rate explorer. Exported so the Metrics page's Collapse tab
 * can reuse it on training-sidecar vectors (no inference job needed). */
export function CollapseBody({ data }: { data: CollapseOut }) {
  const [minApps, setMinApps] = useState(1);
  const [topN, setTopN] = useState(30);
  const [modelRange, setModelRange] = useState<[number, number]>([0, 100]);
  const [humanRange, setHumanRange] = useState<[number, number]>([0, 100]);
  const stats = useMemo(
    () => computeStats(data, minApps, modelRange, humanRange),
    [data, minApps, modelRange, humanRange],
  );
  const [bucketIdx, setBucketIdx] = useState<number>(BUCKETS.length - 1);

  const bucketCards = useMemo(() => {
    const [lo, hi] = BUCKETS[bucketIdx];
    const idxs: number[] = [];
    for (let i = 0; i < stats.modelRate.length; i++) {
      if (!stats.eligible[i]) continue;
      const r = stats.modelRate[i];
      if (r >= lo && r < hi) idxs.push(i);
    }
    idxs.sort((a, b) => stats.modelRate[b] - stats.modelRate[a]);
    return idxs.slice(0, topN);
  }, [bucketIdx, stats, topN]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 flex-wrap text-xs">
        <label className="flex items-center gap-1 text-fg-3">
          Min appearances
          <input type="number" min={1} max={500} value={minApps}
            onChange={(e) => setMinApps(Math.max(1, Number(e.target.value) || 1))}
            className="w-16 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 nums text-fg-0" />
        </label>
        <label className="flex items-center gap-1 text-fg-3">
          Show top
          <input type="number" min={5} max={200} value={topN}
            onChange={(e) => setTopN(Math.max(5, Number(e.target.value) || 30))}
            className="w-16 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 nums text-fg-0" />
        </label>
        <RatePill label="Model %" tone="model" range={modelRange} onChange={setModelRange} />
        <RatePill label="Human %" tone="human" range={humanRange} onChange={setHumanRange} />
      </div>

      <KpiRow>
        <KpiTile label="eligible cards" value={stats.nEligible.toLocaleString()} />
        <KpiTile label="model 100%" value={String(stats.nFull)} />
        <KpiTile label="model 0%" value={String(stats.nZero)} />
        <KpiTile label="strong overpick" value={String(stats.nStrongOver)} />
        <KpiTile label="strong underpick" value={String(stats.nStrongUnder)} />
      </KpiRow>

      <div className="rounded-lg border border-border bg-bg-2 p-4">
        <div className="text-sm font-semibold text-fg-0 mb-3">Pick-rate distribution</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={stats.histogram} margin={{ top: 8, right: 12, left: -8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
            <XAxis dataKey="label" stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} />
            <YAxis stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="model" fill={MODEL_COLOR} name="Model" />
            <Bar dataKey="human" fill={HUMAN_COLOR} name="Human" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-fg-0 font-semibold">Bucket explorer</h3>
          <span className="text-fg-3 text-xs">click a bucket to see the cards in it</span>
        </div>
        <div className="flex flex-wrap gap-1 mb-4">
          {BUCKETS.map(([, , label], i) => {
            const count = stats.bucketCounts[i];
            const active = i === bucketIdx;
            return (
              <button key={label}
                onClick={() => setBucketIdx(i)}
                className={cn(
                  "px-3 py-1.5 rounded-md text-sm font-medium border transition-colors",
                  active
                    ? "bg-accent-subtle text-accent border-accent"
                    : "bg-bg-2 text-fg-2 border-border hover:bg-bg-3",
                )}
              >
                {label} <span className="nums text-fg-3 ml-1">{count}</span>
              </button>
            );
          })}
        </div>
        <CardGrid
          items={bucketCards.map((idx): CardGridItem => ({
            idx,
            caption: (
              <>
                <span className="nums text-model font-semibold">m {(stats.modelRate[idx] * 100).toFixed(0)}%</span>
                <span className="text-fg-3"> / </span>
                <span className="nums text-human font-semibold">h {(stats.humanRate[idx] * 100).toFixed(0)}%</span>
                <span className="nums text-fg-3"> · {stats.appearances[idx]} packs</span>
              </>
            ),
          }))}
          size="xs"
        />
      </section>

      {stats.alwaysPicked.length > 0 && (
        <Detail title="Always picked (model 100%)" idxs={stats.alwaysPicked.slice(0, topN)} stats={stats} />
      )}
      {stats.neverPicked.length > 0 && (
        <Detail title="Never picked (model 0%, humans pick ≥5x)" idxs={stats.neverPicked.slice(0, topN)} stats={stats} />
      )}
      <Detail title="Top overpicked (Δ > 0)" idxs={stats.topOverpicked.slice(0, topN)} stats={stats} highlightDelta="over" />
      <Detail title="Top underpicked (Δ < 0)" idxs={stats.topUnderpicked.slice(0, topN)} stats={stats} highlightDelta="under" />
    </div>
  );
}


function Detail({
  title, idxs, stats, highlightDelta,
}: {
  title: string;
  idxs: number[];
  stats: ReturnType<typeof computeStats>;
  highlightDelta?: "over" | "under";
}) {
  if (idxs.length === 0) return null;
  return (
    <section>
      <h3 className="text-fg-0 font-semibold mb-3">{title}</h3>
      <CardGrid
        items={idxs.map((idx): CardGridItem => {
          const m = stats.modelRate[idx];
          const h = stats.humanRate[idx];
          const d = m - h;
          const sign = d >= 0 ? "+" : "";
          return {
            idx,
            caption: (
              <>
                <span className="nums text-model">m {(m * 100).toFixed(0)}%</span>
                <span className="text-fg-3"> / </span>
                <span className="nums text-human">h {(h * 100).toFixed(0)}%</span>
                <span className="nums text-fg-3"> · {stats.appearances[idx]} packs</span>
                <br />
                <span className={cn(
                  "nums font-semibold",
                  highlightDelta === "over" && "text-bad",
                  highlightDelta === "under" && "text-accent",
                )}>
                  Δ {sign}{(d * 100).toFixed(1)}%
                </span>
              </>
            ),
          };
        })}
        size="xs"
      />
    </section>
  );
}


function computeStats(
  data: CollapseOut,
  minApps: number,
  modelRange: [number, number] = [0, 100],
  humanRange: [number, number] = [0, 100],
) {
  const N = data.appearances.length;
  const eligible = new Uint8Array(N);
  const modelRate = new Float32Array(N);
  const humanRate = new Float32Array(N);
  const mLo = modelRange[0] / 100, mHi = modelRange[1] / 100;
  const hLo = humanRange[0] / 100, hHi = humanRange[1] / 100;
  let nEligible = 0;
  let nFull = 0;
  let nZero = 0;
  for (let i = 0; i < N; i++) {
    const ap = data.appearances[i];
    if (ap < minApps) continue;
    const mr = data.model_picks[i] / ap;
    const hr = data.human_picks[i] / ap;
    modelRate[i] = mr;
    humanRate[i] = hr;
    if (mr < mLo || mr > mHi) continue;
    if (hr < hLo || hr > hHi) continue;
    eligible[i] = 1;
    nEligible++;
    if (data.model_picks[i] === ap) nFull++;
    if (data.model_picks[i] === 0) nZero++;
  }
  let nStrongOver = 0;
  let nStrongUnder = 0;
  for (let i = 0; i < N; i++) {
    if (!eligible[i]) continue;
    if (modelRate[i] > 0.5 && humanRate[i] < 0.1) nStrongOver++;
    if (modelRate[i] < 0.05 && humanRate[i] > 0.3) nStrongUnder++;
  }
  const histogram = BUCKETS.map(([lo, hi, label]) => {
    let m = 0, h = 0;
    for (let i = 0; i < N; i++) {
      if (!eligible[i]) continue;
      if (modelRate[i] >= lo && modelRate[i] < hi) m++;
      if (humanRate[i] >= lo && humanRate[i] < hi) h++;
    }
    return { label, model: m, human: h };
  });
  const bucketCounts = histogram.map((r) => r.model);
  const alwaysPicked: number[] = [];
  const neverPicked: number[] = [];
  for (let i = 0; i < N; i++) {
    if (!eligible[i]) continue;
    if (data.model_picks[i] === data.appearances[i]) alwaysPicked.push(i);
    if (data.model_picks[i] === 0 && data.human_picks[i] >= 5) neverPicked.push(i);
  }
  alwaysPicked.sort((a, b) => data.appearances[b] - data.appearances[a]);
  neverPicked.sort((a, b) => data.human_picks[b] - data.human_picks[a]);

  // Top over/underpicked with a stricter eligibility for stable estimates.
  const stableMin = Math.max(minApps, 30);
  const overUnder: number[] = [];
  for (let i = 0; i < N; i++) {
    if (data.appearances[i] >= stableMin) overUnder.push(i);
  }
  const topOverpicked = [...overUnder].sort(
    (a, b) => (modelRate[b] - humanRate[b]) - (modelRate[a] - humanRate[a]),
  );
  const topUnderpicked = [...overUnder].sort(
    (a, b) => (modelRate[a] - humanRate[a]) - (modelRate[b] - humanRate[b]),
  );

  return {
    nEligible, nFull, nZero, nStrongOver, nStrongUnder,
    eligible, modelRate, humanRate,
    appearances: data.appearances,
    histogram, bucketCounts,
    alwaysPicked, neverPicked,
    topOverpicked, topUnderpicked,
  };
}


function RatePill({
  label, tone, range, onChange,
}: {
  label: string;
  tone: "model" | "human";
  range: [number, number];
  onChange: (r: [number, number]) => void;
}) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1 px-2 py-1 rounded border border-border-subtle",
      tone === "model" ? "text-model" : "text-human",
    )}>
      {label}
      <input type="number" min={0} max={100} value={range[0]}
        onChange={(e) => onChange([clamp(Number(e.target.value) || 0), range[1]])}
        className="w-12 bg-bg-3 border border-border-subtle rounded px-1 py-0.5 nums text-fg-0" />
      <span className="text-fg-3">–</span>
      <input type="number" min={0} max={100} value={range[1]}
        onChange={(e) => onChange([range[0], clamp(Number(e.target.value) || 100)])}
        className="w-12 bg-bg-3 border border-border-subtle rounded px-1 py-0.5 nums text-fg-0" />
    </span>
  );
}


function clamp(n: number): number {
  return Math.max(0, Math.min(100, n));
}
