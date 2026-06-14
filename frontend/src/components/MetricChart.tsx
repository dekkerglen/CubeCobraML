/** Wrapper around Recharts LineChart with dashboard-style theming. */
import {
  Brush, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

import { CHART_THEME, CHART_TOOLTIP_STYLE } from "@/lib/chartTheme";
import type { MetricsRow } from "@/lib/types";


export interface SeriesSpec {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
}


export function MetricChart({
  title, rows, series, yDomain, height = 280,
}: {
  title?: string;
  rows: MetricsRow[];
  series: SeriesSpec[];
  yDomain?: [number, number];
  height?: number;
}) {
  // Add a synthetic `x` field combining epoch + step_in_epoch fraction
  const data = rows.map((r) => ({
    ...r,
    x: (r.epoch ?? 0) + ((r.step_in_epoch ?? 0) / 1000),
  }));

  return (
    <div className="rounded-lg border border-border bg-bg-2 p-4">
      {title && (
        <div className="text-sm font-semibold text-fg-0 mb-3">{title}</div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
          <XAxis dataKey="x" stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} />
          <YAxis stroke={CHART_THEME.axis} tick={{ fontSize: 11 }} domain={yDomain ?? ["auto", "auto"]} />
          <Tooltip
            contentStyle={CHART_TOOLTIP_STYLE}
            cursor={{ stroke: CHART_THEME.cursor, strokeWidth: 1 }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
            iconType="line"
          />
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.dashed ? "4 3" : undefined}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
          {data.length > 12 && (
            <Brush
              dataKey="x"
              height={20}
              stroke={CHART_THEME.axis}
              fill={CHART_THEME.grid}
              travellerWidth={8}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
