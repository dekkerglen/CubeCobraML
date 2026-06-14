/** Custom-query playground for the draft head.
 *
 * Build an arbitrary (Pack, Pool, Context) tuple and run the model on it.
 * Context A is required; Context B is optional and enables side-by-side
 * comparison of two contexts. Combined with header compare-mode, the page
 * renders up to a 2×2 grid: (ckpt A | ckpt B) × (context A | context B).
 *
 * Use case: "if I have Splinter Twin in my context, does the model raise
 * Pestermite's pick rate?" Set both contexts the same except for one card,
 * then read the ranked list deltas.
 */
import { useMutation } from "@tanstack/react-query";
import { Play, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { CardImage } from "@/components/CardImage";
import { CardMultiPicker } from "@/components/CardMultiPicker";
import { CubeSearchPicker } from "@/components/CubeSearchPicker";
import { RankedPickList, type RankedPick } from "@/components/RankedPickList";
import { useModelSelection } from "@/hooks/useModelSelection";
import { apiCubeByUuid, apiPredictDraft } from "@/lib/api";
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
    ctxA: parseIdxList(params.get("ctxA")),
    ctxB: parseIdxList(params.get("ctxB")),
    autoRun: params.get("run") === "1",
  };

  const [pack, setPack] = useState<number[]>(initial.pack);
  const [pool, setPool] = useState<number[]>(initial.pool);
  const [contextA, setContextA] = useState<number[]>(initial.ctxA);
  const [contextB, setContextB] = useState<number[]>(initial.ctxB);
  const [enableB, setEnableB] = useState(initial.ctxB.length > 0);
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
          Build a pack, pool, and context vector by hand. Compare how the
          model's pick changes with the context or with the checkpoint.
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

        <ContextBox title={`Context A (${contextA.length})`} cards={contextA} setCards={setContextA} />

        <div>
          {enableB ? (
            <ContextBox title={`Context B (${contextB.length})`} cards={contextB} setCards={setContextB}
              onRemove={() => { setEnableB(false); setContextB([]); }} />
          ) : (
            <button
              type="button"
              onClick={() => setEnableB(true)}
              className="w-full h-full min-h-[200px] rounded-lg border border-dashed border-border text-fg-3 hover:border-accent hover:text-accent"
            >
              + Add Context B (compare two contexts)
            </button>
          )}
        </div>
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
        <ResultsGrid
          runKey={runKey}
          ckpts={activeKeys}
          contexts={enableB ? [{ label: "A", cards: contextA }, { label: "B", cards: contextB }]
                            : [{ label: "A", cards: contextA }]}
          pack={pack}
          pool={pool}
        />
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


function ContextBox({
  title, cards, setCards, onRemove,
}: {
  title: string;
  cards: number[];
  setCards: (v: number[]) => void;
  onRemove?: () => void;
}) {
  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4 space-y-3">
      <header className="flex items-center justify-between gap-2 flex-wrap">
        <div>
          <h2 className="text-sm font-semibold text-fg-0">{title}</h2>
          <p className="text-fg-3 text-xs mt-0.5">Pick a cube to load its cards, or add cards directly.</p>
        </div>
        <div className="flex items-center gap-3">
          <ClearButton count={cards.length} onClear={() => setCards([])} />
          {onRemove && (
            <button type="button" onClick={onRemove} className="text-fg-3 hover:text-bad text-xs">
              remove
            </button>
          )}
        </div>
      </header>
      <CubeSearchPicker
        placeholder="Load a cube as context…"
        onPickOverride={async (uuid) => {
          const cube = await apiCubeByUuid(uuid);
          setCards(cube.cards);
        }}
      />
      <CardMultiPicker value={cards} onChange={setCards} placeholder="or add cards directly…" />
      {cards.length > 0 && (
        <details>
          <summary className="text-fg-3 text-xs cursor-pointer hover:text-fg-1">
            preview {cards.length} cards
          </summary>
          <div className="mt-2">
            <RemovableCardGrid
              cards={cards.slice(0, 24)}
              onChange={(next) => setCards([...next, ...cards.slice(24)])}
            />
          </div>
        </details>
      )}
    </section>
  );
}


function ResultsGrid({
  runKey, ckpts, contexts, pack, pool,
}: {
  runKey: number;
  ckpts: string[];
  contexts: Array<{ label: string; cards: number[] }>;
  pack: number[];
  pool: number[];
}) {
  return (
    <div className="space-y-4">
      <h2 className="text-fg-0 font-semibold">Results</h2>
      <div
        className={cn(
          "grid gap-4",
          contexts.length === 2 ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1",
        )}
      >
        {contexts.map((ctx) => (
          <div key={ctx.label} className="space-y-3">
            <div className="text-xs uppercase tracking-wider font-bold text-fg-3">
              Context {ctx.label} ({ctx.cards.length} cards)
            </div>
            <div className={cn(
              "grid gap-4",
              ckpts.length === 2 ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1",
            )}>
              {ckpts.map((ckpt) => (
                <PredictionCell
                  key={`${ckpt}::${ctx.label}::${runKey}`}
                  ckpt={ckpt}
                  pack={pack}
                  pool={pool}
                  cubeCtx={ctx.cards}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


function PredictionCell({
  ckpt, pack, pool, cubeCtx,
}: {
  ckpt: string;
  pack: number[];
  pool: number[];
  cubeCtx: number[];
}) {
  const mut = useMutation({
    mutationFn: () => apiPredictDraft({ ckpt, pack, pool, cube_ctx: cubeCtx }),
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
