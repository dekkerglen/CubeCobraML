/** KPI tile — label, value, optional delta. */
import { cn } from "@/lib/cn";

export interface KpiProps {
  label: string;
  value: string;
  delta?: number | null;
  /** Larger-is-better metrics (default). Set to false for "smaller is better"
   * (e.g. loss) so the color coding flips. */
  good?: "up" | "down";
  className?: string;
}

export function KpiTile({ label, value, delta, good = "up", className }: KpiProps) {
  let deltaCls = "text-fg-3";
  let arrow = "";
  if (delta != null && delta !== 0) {
    const positive = delta > 0;
    arrow = positive ? "↑" : "↓";
    if (good === "up") deltaCls = positive ? "text-good" : "text-bad";
    else deltaCls = positive ? "text-bad" : "text-good";
  } else if (delta === 0) {
    arrow = "→";
  }
  return (
    <div className={cn(
      "rounded-md border border-border bg-bg-2 px-4 py-3",
      className,
    )}>
      <div className="text-[11px] text-fg-3 uppercase tracking-wider font-semibold">
        {label}
      </div>
      <div className="mt-1 text-2xl font-bold text-fg-0 nums leading-tight">
        {value}
      </div>
      {delta != null && (
        <div className={cn("mt-0.5 text-xs nums", deltaCls)}>
          {arrow} {Math.abs(delta).toFixed(4)}
        </div>
      )}
    </div>
  );
}

export function KpiRow({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn(
      "grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3",
      className,
    )}>
      {children}
    </div>
  );
}
