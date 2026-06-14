/** Pack-by-pack stepper for the draft replay view.
 *
 * Prev / next buttons, jump-to-step input, autoplay toggle (advances every
 * ~1.5s). Stateless: parent owns `step` and `onChange`.
 */
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/cn";


export function PackStepper({
  step, total, onChange, className,
}: {
  step: number;
  total: number;
  onChange: (s: number) => void;
  className?: string;
}) {
  const [autoplay, setAutoplay] = useState(false);

  useEffect(() => {
    if (!autoplay) return;
    if (step >= total - 1) { setAutoplay(false); return; }
    const t = setTimeout(() => onChange(step + 1), 1500);
    return () => clearTimeout(t);
  }, [autoplay, step, total, onChange]);

  const atStart = step <= 0;
  const atEnd = step >= total - 1;

  return (
    <div className={cn("flex items-center gap-3 flex-wrap", className)}>
      <button
        type="button"
        onClick={() => onChange(Math.max(0, step - 1))}
        disabled={atStart}
        className={cn(
          "inline-flex items-center gap-1 px-3 py-1.5 rounded-md border border-border text-sm font-medium",
          "bg-bg-0 hover:bg-bg-2 disabled:opacity-40 disabled:cursor-not-allowed",
        )}
      >
        <ChevronLeft className="size-4" /> Prev
      </button>
      <button
        type="button"
        onClick={() => onChange(Math.min(total - 1, step + 1))}
        disabled={atEnd}
        className={cn(
          "inline-flex items-center gap-1 px-3 py-1.5 rounded-md border border-border text-sm font-medium",
          "bg-bg-0 hover:bg-bg-2 disabled:opacity-40 disabled:cursor-not-allowed",
        )}
      >
        Next <ChevronRight className="size-4" />
      </button>
      <button
        type="button"
        onClick={() => setAutoplay((a) => !a)}
        disabled={atEnd}
        className={cn(
          "inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium border",
          autoplay
            ? "border-accent bg-accent text-white"
            : "border-border bg-bg-0 hover:bg-bg-2",
          "disabled:opacity-40 disabled:cursor-not-allowed",
        )}
      >
        {autoplay ? <Pause className="size-4" /> : <Play className="size-4" />}
        {autoplay ? "Pause" : "Auto"}
      </button>
      <span className="text-fg-2 text-sm nums">
        pick <span className="text-fg-0 font-semibold">{step + 1}</span> / {total}
      </span>
      <input
        type="range"
        min={0}
        max={Math.max(0, total - 1)}
        value={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 min-w-[160px] accent-accent"
      />
    </div>
  );
}
