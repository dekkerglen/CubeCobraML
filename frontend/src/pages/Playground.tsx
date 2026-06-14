/** Custom-query playground for the draft head.
 *
 * Build an arbitrary (Pack, Pool) pair by hand and run the model on it.
 * With header compare-mode on, renders the prediction side-by-side across
 * the selected checkpoints.
 */
import { useMutation } from "@tanstack/react-query";
import { Play, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { CardImage } from "@/components/CardImage";
import { CardMultiPicker } from "@/components/CardMultiPicker";
import { RankedPickList, type RankedPick } from "@/components/RankedPickList";
import { useModelSelection } from "@/hooks/useModelSelection";
import { apiPredictDraft } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { DraftPrediction } from "@/lib/types";


/** Card preview grid with a per-card × button. Click the card opens the
 * drawer (existing CardImage behavior); click the × removes it from the
 * list. */
function RemovableCardGrid({
  cards, onChange,
}: {
  cards: number[];
  onChange: (next: number[]) => void;
}) {
  if (cards.length === 0) return null;
  const remove = (idx: number) => onChange(cards.filter((c) => c !== idx));
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(90px,1fr))] gap-2">
      {cards.map((idx) => (
        <div key={idx} className="relative group">
          <CardImage idx={idx} />
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); remove(idx); }}
            className="absolute -top-1 -right-1 z-10 size-5 rounded-full bg-bad text-white text-xs flex items-center justify-center shadow-md opacity-0 group-hover:opacity-100 transition-opacity hover:scale-110"
            aria-label="Remove card"
            title="Remove from list"
          >
            <X className="size-3" />
          </button>
        </div>
      ))}
    </div>
  );
}


function ClearButton({ count, onClear }: { count: number; onClear: () => void }) {
  if (count === 0) return null;
  return (
    <button
      type="button"
      onClick={onClear}
      className="text-xs text-fg-3 hover:text-bad inline-flex items-center gap-1"
      title="Remove all cards"
    >
      <X className="size-3" /> clear all
    </button>
  );
}


function parseIdxList(s: string | null): number[] {
  if (!s) return [];
  return s.split(",").map((x) => Number(x.trim())).filter((n) => Number.isFinite(n));
}


export function PlaygroundPage() {
  const [params] = useSearchParams();
  const initial = {
    pack: parseIdxList(params.get("pack")),
    pool: parseIdxList(params.get("pool")),
    autoRun: params.get("run") === "1",
  };

  const [pack, setPack] = useState<number[]>(initial.pack);
  const [pool, setPool] = useState<number[]>(initial.pool);
  // Bumping runKey kicks each cell mutation in PredictionCell.
  const [runKey, setRunKey] = useState(0);
  const { activeKeys } = useModelSelection();

  const canRun = pack.length > 0 && activeKeys.length > 0;

  // Auto-run once if URL says so and we have everything we need.
  useEffect(() => {
    if (initial.autoRun && initial.pack.length > 0) {
      setRunKey(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="container py-8 space-y-6">
      <header className="border-b border-border pb-4">
        <h1 className="text-2xl font-bold text-fg-0">Playground</h1>
        <p className="text-fg-2 text-sm mt-1">
          Build a pack and pool by hand and run the model on it. Compare how
          the pick changes across checkpoints.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <InputBox
          title={`Pack (${pack.length})`}
          help="Cards offered this pick. Required."
          headerExtra={<ClearButton count={pack.length} onClear={() => setPack([])} />}
        >
          <CardMultiPicker value={pack} onChange={setPack} placeholder="add to pack…" />
          <RemovableCardGrid cards={pack} onChange={setPack} />
        </InputBox>

        <InputBox
          title={`Pool (${pool.length})`}
          help="Cards already drafted. Optional."
          headerExtra={<ClearButton count={pool.length} onClear={() => setPool([])} />}
        >
          <CardMultiPicker value={pool} onChange={setPool} placeholder="add to pool…" />
          <RemovableCardGrid cards={pool} onChange={setPool} />
        </InputBox>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setRunKey((k) => k + 1)}
          disabled={!canRun}
          className={cn(
            "inline-flex items-center gap-2 px-4 py-2 rounded-md font-medium text-sm",
            "bg-accent text-white hover:bg-accent-hover",
            "disabled:opacity-50 disabled:cursor-not-allowed",
          )}
        >
          <Play className="size-4" />
          Run
        </button>
        {!canRun && (
          <span className="text-fg-3 text-sm">
            {activeKeys.length === 0 ? "Select a model from the header." : "Add at least one card to the pack."}
          </span>
        )}
      </div>

      {runKey > 0 && (
        <ResultsGrid runKey={runKey} ckpts={activeKeys} pack={pack} pool={pool} />
      )}
    </div>
  );
}


function InputBox({
  title, help, headerExtra, children,
}: {
  title: string;
  help: string;
  headerExtra?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4 space-y-3">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-fg-0">{title}</h2>
          <p className="text-fg-3 text-xs mt-0.5">{help}</p>
        </div>
        {headerExtra}
      </header>
      {children}
    </section>
  );
}


function ResultsGrid({
  runKey, ckpts, pack, pool,
}: {
  runKey: number;
  ckpts: string[];
  pack: number[];
  pool: number[];
}) {
  return (
    <div className="space-y-4">
      <h2 className="text-fg-0 font-semibold">Results</h2>
      <div className={cn(
        "grid gap-4",
        ckpts.length === 2 ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1",
      )}>
        {ckpts.map((ckpt) => (
          <PredictionCell
            key={`${ckpt}::${runKey}`}
            ckpt={ckpt}
            pack={pack}
            pool={pool}
          />
        ))}
      </div>
    </div>
  );
}


function PredictionCell({
  ckpt, pack, pool,
}: {
  ckpt: string;
  pack: number[];
  pool: number[];
}) {
  const mut = useMutation({
    mutationFn: () => apiPredictDraft({ ckpt, pack, pool }),
  });
  // Fire once on mount — parent re-mounts via `key` when runKey bumps.
  useEffect(() => { mut.mutate(); /* eslint-disable-next-line */ }, []);

  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4 space-y-3">
      <header className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-fg-0 truncate">{ckpt}</h3>
        {mut.data && (
          <span className="text-fg-3 text-xs nums">
            top1 {(mut.data.top1_p * 100).toFixed(1)}%
          </span>
        )}
      </header>
      {mut.isPending && <div className="text-fg-3 text-sm py-4">Running model…</div>}
      {mut.error && <div className="text-bad text-sm">{(mut.error as Error).message}</div>}
      {mut.data && <RankedRows data={mut.data} />}
    </section>
  );
}


function RankedRows({ data }: { data: DraftPrediction }) {
  const rows: RankedPick[] = data.ranked.slice(0, 15).map(([idx, p], i) => ({
    idx, p, rank: i + 1, isTop: i === 0,
  }));
  return <RankedPickList rows={rows} />;
}
