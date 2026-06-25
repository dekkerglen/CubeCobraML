/** Training metrics page — KPI deltas + tabbed plots. */
import * as Tabs from "@radix-ui/react-tabs";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useQuery } from "@tanstack/react-query";

import { ComparePanels } from "@/components/ComparePanels";
import { KpiRow, KpiTile } from "@/components/KpiTile";
import { MetricChart, type SeriesSpec } from "@/components/MetricChart";
import { useCheckpoints } from "@/hooks/useCheckpoints";
import { useModelSelection } from "@/hooks/useModelSelection";
import { CollapseBody, DistributionPanel } from "@/pages/Collapse";
import { useMetricsRunIds, useRunRows } from "@/hooks/useMetrics";
import {
  apiRunCollapse, apiRunCollapseEpochs, type RunCollapseSplit,
} from "@/lib/api";
import { SERIES_PALETTE } from "@/lib/chartTheme";
import type { Checkpoint, MetricsRow } from "@/lib/types";
import { cn } from "@/lib/cn";


const COLORS = SERIES_PALETTE;


export function MetricsPage() {
  const { data: ckpts = [] } = useCheckpoints();
  const { data: runsData } = useMetricsRunIds();
  const runs = runsData?.runs ?? [];
  const { ckpt, hasCompare } = useModelSelection();

  // Default to the run of the active ckpt, else the most recent run.
  const ckptRunId = ckpts.find((c) => c.key === ckpt)?.run_id ?? null;
  const [selected, setSelected] = useState<string[]>([]);
  // Re-sync to the global ckpt's run whenever the header selection changes.
  useEffect(() => {
    if (ckptRunId) {
      setSelected([ckptRunId]);
    } else if (runs.length > 0 && selected.length === 0) {
      setSelected([runs[0]]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ckptRunId, runs]);

  if (runs.length === 0) {
    return (
      <div className="container py-16 text-center">
        <p className="text-fg-2">No training runs found.</p>
      </div>
    );
  }

  return (
    <div className="container py-8 space-y-6">
      <header className="flex items-end justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-2xl font-bold text-fg-0">Training metrics</h1>
          <p className="text-fg-2 text-sm mt-1">
            KPI snapshot + curves from <code className="font-mono text-xs">runs/&lt;id&gt;/metrics.csv</code>.
          </p>
        </div>
        {!hasCompare && (
          <RunPicker runs={runs} selected={selected} onChange={setSelected} ckpts={ckpts} />
        )}
      </header>

      <ComparePanels render={(ckptKey) => {
        // In compare mode each panel pins to its own ckpt's run, ignoring the
        // RunPicker so the two sides actually differ. In single mode `selected`
        // drives so the user can overlay extra runs.
        const ownRun = ckpts.find((c) => c.key === ckptKey)?.run_id ?? null;
        const runIds = hasCompare ? (ownRun ? [ownRun] : []) : selected;
        return (
          <div className="space-y-6">
            {runIds[0] && <Snapshot rowsRunId={runIds[0]} />}
            <TabbedPlots selected={runIds} />
          </div>
        );
      }} />
    </div>
  );
}


function RunPicker({
  runs, selected, onChange, ckpts,
}: {
  runs: string[];
  selected: string[];
  onChange: (s: string[]) => void;
  ckpts: Checkpoint[];
}) {
  const toggle = (r: string) => {
    onChange(selected.includes(r) ? selected.filter((x) => x !== r) : [...selected, r]);
  };
  return (
    <details className="relative">
      <summary className="list-none inline-flex items-center gap-2 cursor-pointer rounded-md border border-border bg-bg-2 px-3 py-1.5 text-sm hover:bg-bg-3">
        <span className="text-fg-3 uppercase tracking-wide text-xs font-semibold">Runs</span>
        <span className="text-fg-0">{selected.length} selected</span>
      </summary>
      <div className="absolute right-0 mt-2 z-20 rounded-lg border border-border bg-bg-2 shadow-2xl w-[480px] max-h-[60vh] overflow-y-auto scrollbar-thin">
        {runs.map((r, i) => {
          const isBest = ckpts.some((c) => c.run_id === r && c.is_best);
          return (
            <label key={r}
                   className="flex items-center gap-3 px-3 py-2 hover:bg-bg-3 cursor-pointer border-b border-border-subtle/40 last:border-b-0">
              <input type="checkbox" checked={selected.includes(r)}
                     onChange={() => toggle(r)}
                     className="accent-accent" />
              <div className="flex-1 min-w-0">
                <div className="text-fg-0 text-sm font-medium truncate">
                  {isBest && <span className="text-warn mr-1.5">★</span>}
                  {r}
                </div>
              </div>
              <span className="size-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            </label>
          );
        })}
      </div>
    </details>
  );
}


function Snapshot({ rowsRunId }: { rowsRunId: string }) {
  const { data } = useRunRows(rowsRunId);
  const rows = data?.rows ?? [];
  const completed = rows.filter((r) => (r.val_bomb_agreement ?? 0) > 0);
  const last = completed[completed.length - 1] ?? rows[rows.length - 1];
  const prev = completed[completed.length - 2];

  function delta(k: string): number | null {
    if (!prev || !last) return null;
    const v0 = (prev as MetricsRow)[k];
    const v1 = (last as MetricsRow)[k];
    if (v0 == null || v1 == null) return null;
    return v1 - v0;
  }

  function v(k: string): string {
    const x = last?.[k];
    if (x == null) return "—";
    return Math.abs(x) >= 1 ? x.toFixed(3) : x.toFixed(4);
  }

  return (
    <div className="space-y-3">
      <KpiRow>
        <KpiTile label="epochs" value={String(rows.length)} />
        <KpiTile label="bomb agreement" value={v("val_bomb_agreement")} delta={delta("val_bomb_agreement")} />
        <KpiTile label="top1 (val)" value={v("val_top1")} delta={delta("val_top1")} />
        <KpiTile label="loss (val)" value={v("val_loss")} delta={delta("val_loss")} good="down" />
        <KpiTile label="argmax churn" value={v("argmax_churn")} delta={delta("argmax_churn")} good="down" />
      </KpiRow>
      <KpiRow>
        <KpiTile label="collapse0 (common)" value={v("val_collapse0_common")} delta={delta("val_collapse0_common")} good="down" />
        <KpiTile label="sat100 (common)" value={v("val_sat100_common")} delta={delta("val_sat100_common")} good="down" />
        <KpiTile label="hist div" value={v("val_pick_hist_div")} delta={delta("val_pick_hist_div")} good="down" />
        <KpiTile label="pick_cce (val)" value={v("val_pick_cce")} delta={delta("val_pick_cce")} good="down" />
      </KpiRow>
    </div>
  );
}


const METRICS_TABS = ["headline", "collapse", "heads", "stab", "distribution"] as const;
type MetricsTab = typeof METRICS_TABS[number];


function TabbedPlots({ selected }: { selected: string[] }) {
  const [params, setParams] = useSearchParams();
  const tab: MetricsTab = (METRICS_TABS as readonly string[]).includes(params.get("tab") ?? "")
    ? (params.get("tab") as MetricsTab)
    : "headline";
  const setTab = (next: string) => {
    const p = new URLSearchParams(params);
    if (next === "headline") p.delete("tab");
    else p.set("tab", next);
    setParams(p, { replace: true });
  };
  return (
    <Tabs.Root value={tab} onValueChange={setTab}>
      <Tabs.List className="flex gap-1 border-b border-border mb-4">
        {[
          ["headline", "Headline"],
          ["collapse", "Collapse"],
          ["heads", "Per-head losses"],
          ["stab", "Stability"],
          ["distribution", "Distribution"],
        ].map(([key, label]) => (
          <Tabs.Trigger key={key} value={key}
            className={cn(
              "px-3 py-2 text-sm font-medium text-fg-2 hover:text-fg-0",
              "data-[state=active]:text-fg-0 data-[state=active]:border-b-2 data-[state=active]:border-accent",
              "transition-colors -mb-px",
            )}
          >{label}</Tabs.Trigger>
        ))}
      </Tabs.List>

      <Tabs.Content value="headline">
        <PlotGroup selected={selected} group="headline" />
      </Tabs.Content>
      <Tabs.Content value="collapse">
        <PlotGroup selected={selected} group="collapse" />
      </Tabs.Content>
      <Tabs.Content value="heads">
        <PlotGroup selected={selected} group="heads" />
      </Tabs.Content>
      <Tabs.Content value="stab">
        <PlotGroup selected={selected} group="stab" />
      </Tabs.Content>
      <Tabs.Content value="distribution">
        <DistributionPanel />
      </Tabs.Content>
    </Tabs.Root>
  );
}


type GroupKey = "headline" | "collapse" | "heads" | "stab";

function PlotGroup({ selected, group }: { selected: string[]; group: GroupKey }) {
  // Hooks-into-loops would be wrong; load each run with its own hook.
  return (
    <div className="space-y-4">
      {group === "headline" && <>
        <Combined selected={selected} title="BombAgreement (val)" pairs={[["bomb_agreement", "val_bomb_agreement"]]} domain={[0, 1]} />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Combined selected={selected} title="top1" pairs={[["top1", "val_top1"]]} domain={[0, 1]} />
          <Combined selected={selected} title="top3" pairs={[["top3", "val_top3"]]} domain={[0, 1]} />
        </div>
        <Combined selected={selected} title="Total loss" pairs={[["loss", "val_loss"]]} />
      </>}
      {group === "collapse" && <>
        <p className="text-fg-2 text-sm">
          Counts of cards the model takes <span className="text-good">0%</span> (humans sometimes take) or <span className="text-good">100%</span> (humans don&apos;t always take), split by how often the card was offered in val. All four should trend toward zero.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Combined selected={selected} title="collapse0 (model 0%, humans &gt; 0%)" pairs={[["val_collapse0_common", null], ["val_collapse0_rare", null]]} />
          <Combined selected={selected} title="sat100 (model 100%, humans &lt; 100%)" pairs={[["val_sat100_common", null], ["val_sat100_rare", null]]} />
        </div>
        <Combined selected={selected} title="pick_hist_div (sym-KL of batch pick histograms)" pairs={[["pick_hist_div", "val_pick_hist_div"]]} />
        {selected[0] && <SidecarExplorer runId={selected[0]} />}
      </>}
      {group === "heads" && <>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Combined selected={selected} title="pick_cce (draft head)" pairs={[["pick_cce", "val_pick_cce"]]} />
          <Combined selected={selected} title="deck_bce" pairs={[["deck_bce", "val_deck_bce"]]} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Combined selected={selected} title="cube_bce" pairs={[["cube_bce", "val_cube_bce"]]} />
          <Combined selected={selected} title="corr_kl"  pairs={[["corr_kl",  "val_corr_kl"]]}  />
        </div>
      </>}
      {group === "stab" && <>
        <p className="text-fg-2 text-sm">
          Spikes in argmax_churn = basin flip — the model swapped which top-cohort cards it commits to.
        </p>
        <Combined selected={selected} title="argmax_churn" pairs={[["argmax_churn", null]]} />
      </>}
    </div>
  );
}


const SPLITS: Array<[RunCollapseSplit, string]> = [
  ["val", "val"],
  ["train", "train (epoch)"],
  ["train_cum", "train Σ"],
];

/** Per-card collapse explorer fed by the training sidecars — which exact
 * cards are collapsed/saturated at a given epoch, no inference job needed. */
function SidecarExplorer({ runId }: { runId: string }) {
  const [epoch, setEpoch] = useState<number | null>(null);
  const [split, setSplit] = useState<RunCollapseSplit>("val");

  const { data: epochsData } = useQuery({
    queryKey: ["run-collapse-epochs", runId],
    queryFn: () => apiRunCollapseEpochs(runId),
    refetchInterval: 120_000,  // live runs grow new sidecars
  });
  const epochs = epochsData?.epochs ?? [];
  const activeEpoch = epoch !== null && epochs.includes(epoch)
    ? epoch
    : epochs[epochs.length - 1];

  const { data, isFetching } = useQuery({
    queryKey: ["run-collapse", runId, activeEpoch, split],
    queryFn: () => apiRunCollapse(runId, activeEpoch!, split),
    enabled: activeEpoch !== undefined,
    staleTime: Infinity,
  });

  if (epochs.length === 0) {
    return (
      <p className="text-fg-3 text-sm">
        No per-card collapse sidecars for this run (pre-sidecar run, or epoch 1 not finished).
      </p>
    );
  }

  return (
    <section className="space-y-4 pt-2 border-t border-border">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-fg-0 font-semibold">Per-card explorer (training sidecar)</h3>
          <p className="text-fg-3 text-xs mt-0.5">
            Exact offers / model-argmax / human-pick counts recorded during the run — no inference job.
            {split === "train" && " Train epochs sample ~0.1% of picks; expect noise."}
            {split === "train_cum" && data && ` Summed over ${data.epochs_aggregated} epochs — converges to true train rates.`}
          </p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <label className="flex items-center gap-1.5 text-fg-3 text-xs">
            epoch
            <select
              value={activeEpoch}
              onChange={(e) => setEpoch(Number(e.target.value))}
              className="bg-bg-3 border border-border-subtle rounded px-1.5 py-1 nums text-fg-0"
            >
              {epochs.map((e) => <option key={e} value={e}>{e}</option>)}
            </select>
          </label>
          <div className="flex rounded-md border border-border overflow-hidden">
            {SPLITS.map(([key, label]) => (
              <button key={key} type="button" onClick={() => setSplit(key)}
                className={cn(
                  "px-2.5 py-1 text-xs font-medium transition-colors",
                  split === key ? "bg-accent-subtle text-accent" : "bg-bg-2 text-fg-2 hover:bg-bg-3",
                )}
              >{label}</button>
            ))}
          </div>
        </div>
      </div>
      {isFetching && !data && <p className="text-fg-3 text-sm">Loading sidecar…</p>}
      {data && (
        <CollapseBody data={{
          ckpt: runId,
          sample_n: 0, sample_n_val: 0, sample_n_train: 0,
          appearances: data.appearances,
          model_picks: data.model_picks,
          human_picks: data.human_picks,
          human_pick_idxs: [], model_pick_idxs: [],
        }} />
      )}
    </section>
  );
}


function RunRowsLoader({
  run, onLoad,
}: {
  run: string;
  onLoad: (run: string, rows: MetricsRow[]) => void;
}) {
  const q = useRunRows(run);
  const rows = q.data?.rows;
  useEffect(() => { if (rows) onLoad(run, rows); }, [run, rows, onLoad]);
  return null;
}


function Combined({
  selected, title, pairs, domain,
}: {
  selected: string[];
  title: string;
  pairs: Array<[string, string | null]>;
  domain?: [number, number];
}) {
  const [rowsByRun, setRowsByRun] = useState<Record<string, MetricsRow[]>>({});
  const onLoad = (run: string, rows: MetricsRow[]) =>
    setRowsByRun((prev) => (prev[run] === rows ? prev : { ...prev, [run]: rows }));

  const merged = useMemo(() => {
    const out: Array<MetricsRow & { x: number }> = [];
    selected.forEach((run, runIdx) => {
      const rows = rowsByRun[run] ?? [];
      for (const r of rows) {
        const x = (r.epoch ?? 0) + ((r.step_in_epoch ?? 0) / 1000);
        let target = out.find((o) => o.x === x);
        if (!target) {
          target = { x, epoch: r.epoch, step_in_epoch: r.step_in_epoch };
          out.push(target);
        }
        for (const [tk, vk] of pairs) {
          target[`${runIdx}__${tk}`] = r[tk];
          if (vk) target[`${runIdx}__${vk}`] = r[vk];
        }
      }
    });
    return out.sort((a, b) => (a.x ?? 0) - (b.x ?? 0));
  }, [selected, rowsByRun, pairs]);

  const series: SeriesSpec[] = [];
  selected.forEach((run, runIdx) => {
    pairs.forEach(([tk, vk], pairIdx) => {
      // With multiple pairs per chart, vary color per pair and label by
      // metric key so the series are distinguishable.
      const color = COLORS[(runIdx + pairIdx) % COLORS.length];
      const trainLabel = pairs.length > 1 ? `${shortRun(run)} ${tk}` : `${shortRun(run)} (train)`;
      series.push({ key: `${runIdx}__${tk}`, label: trainLabel, color });
      if (vk) series.push({ key: `${runIdx}__${vk}`, label: `${shortRun(run)} (val)`, color, dashed: true });
    });
  });

  return (
    <>
      {selected.map((run) => (
        <RunRowsLoader key={run} run={run} onLoad={onLoad} />
      ))}
      <MetricChart title={title} rows={merged} series={series} yDomain={domain} />
    </>
  );
}


function shortRun(r: string): string {
  // Take 20260605-163405Z-f7bb20e2 → 06-05 16:34
  const m = r.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  return m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : r;
}
