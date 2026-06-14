/** Job progress UI — fills a bar, formats ETA, exposes a Cancel button.
 *
 * Used everywhere we drive a long-running job via `useJob`. Renders nothing
 * for terminal statuses so the parent can swap in the result view at 100%.
 */
import { X } from "lucide-react";

import { useCancelJob } from "@/hooks/useJob";
import { cn } from "@/lib/cn";
import type { JobOut } from "@/lib/types";


function formatEta(seconds: number | null): string {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  return `${m}m ${s}s`;
}


export function ProgressBar({
  job, label, className,
}: {
  job: JobOut | undefined;
  label?: string;
  className?: string;
}) {
  const cancel = useCancelJob();
  if (!job) return null;
  const pct = Math.max(0, Math.min(100, job.progress * 100));
  const cancellable = job.status === "running" || job.status === "queued";
  return (
    <div className={cn(
      "rounded-md border border-border bg-bg-1 p-3 space-y-2",
      className,
    )}>
      <div className="flex items-center gap-3 text-xs">
        <span className="text-fg-2 font-medium">
          {label ?? job.kind}
        </span>
        <span className={cn(
          "uppercase tracking-wider font-bold px-1.5 py-0.5 rounded text-[10px]",
          job.status === "running" && "bg-accent text-white",
          job.status === "queued" && "bg-bg-3 text-fg-2",
          job.status === "done" && "bg-good text-white",
          job.status === "error" && "bg-bad text-white",
          job.status === "cancelled" && "bg-warn text-white",
        )}>{job.status}</span>
        <span className="text-fg-3 nums ml-auto">
          {pct.toFixed(0)}%
          {job.status === "running" && (
            <span className="ml-2">eta {formatEta(job.eta_seconds)}</span>
          )}
        </span>
        {cancellable && (
          <button
            type="button"
            onClick={() => cancel.mutate(job.id)}
            disabled={cancel.isPending}
            className="inline-flex items-center gap-1 text-fg-3 hover:text-bad disabled:opacity-50"
            aria-label="cancel job"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>
      <div className="h-1.5 bg-bg-3 rounded overflow-hidden">
        <div
          className={cn(
            "h-full transition-[width] duration-200",
            job.status === "error" ? "bg-bad"
              : job.status === "cancelled" ? "bg-warn"
              : "bg-accent",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      {job.error && (
        <div className="text-bad text-xs font-mono break-all">{job.error}</div>
      )}
    </div>
  );
}
