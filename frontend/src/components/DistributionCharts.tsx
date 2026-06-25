/** Color / CMC / type distribution charts for a list of picks.
 *
 * Pure presentation — caller supplies the card indices that make up each
 * series (human picks, optional model picks, optional cube composition) plus
 * the card metadata to read attributes from. Renders three side-by-side
 * bar charts. Missing/unknown cards are silently dropped.
 */
import { useMemo } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

import { CHART_THEME, CHART_TOOLTIP_STYLE, HUMAN_COLOR, MODEL_COLOR } from "@/lib/chartTheme";
import type { Card } from "@/lib/types";


export type CardIdx = number;


export interface DistributionSeries {
  human: CardIdx[];
  model?: CardIdx[];
  cube?: CardIdx[];
}


const COLOR_ORDER = ["W", "U", "B", "R", "G", "C"] as const;
const CMC_BUCKETS = ["0", "1", "2", "3", "4", "5", "6", "7+"] as const;
const TYPES = [
  "Creature", "Instant", "Sorcery", "Artifact",
  "Enchantment", "Planeswalker", "Land",
] as const;


export function DistributionCharts({
  series, cardsByIdx,
}: {
  series: DistributionSeries;
  cardsByIdx: Map<CardIdx, Card>;
}) {
  const colorData = useMemo(() => buildBuckets(series, cardsByIdx, colorBuckets), [series, cardsByIdx]);
  const cmcData   = useMemo(() => buildBuckets(series, cardsByIdx, cmcBuckets),   [series, cardsByIdx]);
  const typeData  = useMemo(() => buildBuckets(series, cardsByIdx, typeBuckets),  [series, cardsByIdx]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <Chart title="Color" data={colorData} hasModel={!!series.model} hasCube={!!series.cube} />
      <Chart title="CMC"   data={cmcData}   hasModel={!!series.model} hasCube={!!series.cube} />
      <Chart title="Type"  data={typeData}  hasModel={!!series.model} hasCube={!!series.cube} />
    </div>
  );
}


// ----- chart -----

interface Row { label: string; human: number; model?: number; cube?: number }

function Chart({
  title, data, hasModel, hasCube,
}: {
  title: string;
  data: Row[];
  hasModel: boolean;
  hasCube: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg-2 p-4">
      <div className="text-sm font-semibold text-fg-0 mb-3">{title}</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
          <XAxis dataKey="label" stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} />
          <YAxis stroke={CHART_THEME.axis} tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip contentStyle={CHART_TOOLTIP_STYLE}
            formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v ?? "")} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="human" fill={HUMAN_COLOR} name="Human" />
          {hasModel && <Bar dataKey="model" fill={MODEL_COLOR} name="Model" />}
          {hasCube && <Bar dataKey="cube" fill={CHART_THEME.axis} name="Cube" />}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}


// ----- bucketing -----

type Bucketer = (card: Card) => readonly string[];


function colorBuckets(card: Card): readonly string[] {
  const ci = card.color_identity ?? [];
  if (ci.length === 0) return ["C"];
  // Multicolor cards count once per color, matching how pick-color bias is
  // usually read ("how often did each color end up in the picked deck").
  return ci.filter((c) => COLOR_ORDER.includes(c as (typeof COLOR_ORDER)[number]));
}


function cmcBuckets(card: Card): readonly string[] {
  if (card.is_land) return [];
  const cmc = Math.max(0, Math.floor(card.cmc ?? 0));
  return [cmc >= 7 ? "7+" : String(cmc)];
}


function typeBuckets(card: Card): readonly string[] {
  const t = (card.type ?? "").toLowerCase();
  for (const cat of TYPES) {
    if (t.includes(cat.toLowerCase())) return [cat];
  }
  return [];
}


function buildBuckets(
  series: DistributionSeries,
  cardsByIdx: Map<CardIdx, Card>,
  bucketer: Bucketer,
): Row[] {
  const labels = bucketer === colorBuckets ? COLOR_ORDER
                : bucketer === cmcBuckets   ? CMC_BUCKETS
                : TYPES;

  const human = countShare(series.human, cardsByIdx, bucketer, labels);
  const model = series.model ? countShare(series.model, cardsByIdx, bucketer, labels) : null;
  const cube  = series.cube  ? countShare(series.cube,  cardsByIdx, bucketer, labels) : null;

  return labels.map((label) => ({
    label,
    human: human[label] ?? 0,
    ...(model ? { model: model[label] ?? 0 } : {}),
    ...(cube  ? { cube:  cube[label]  ?? 0 } : {}),
  }));
}


function countShare(
  idxs: CardIdx[],
  cardsByIdx: Map<CardIdx, Card>,
  bucketer: Bucketer,
  labels: readonly string[],
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const l of labels) counts[l] = 0;
  let total = 0;
  for (const idx of idxs) {
    const card = cardsByIdx.get(idx);
    if (!card) continue;
    const buckets = bucketer(card);
    for (const b of buckets) {
      counts[b] = (counts[b] ?? 0) + 1;
      total += 1;
    }
  }
  if (total === 0) return counts;
  const out: Record<string, number> = {};
  for (const l of labels) out[l] = counts[l] / total;
  return out;
}
