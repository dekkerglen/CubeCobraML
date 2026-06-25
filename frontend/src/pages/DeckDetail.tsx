/** Single deck workspace.
 *
 * Left: the human-built mainboard for this val deck.
 * Right: the bot-built deck given the same card pool (the mainboard
 * itself is fed in as the pool — "if you had these cards, what would
 * the model assemble?"). Wrapped in ComparePanels so compare mode
 * shows the bot deck per checkpoint.
 */
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { CardGrid, type CardGridItem } from "@/components/CardGrid";
import { ComparePanels } from "@/components/ComparePanels";
import { ProgressBar } from "@/components/ProgressBar";
import { useJob, useStartJob } from "@/hooks/useJob";
import { useModelSelection } from "@/hooks/useModelSelection";
import { apiDeck } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { DeckBuilderResult } from "@/lib/types";


type Split = "train" | "val";


export function DeckDetailPage() {
  const { uuid = "", idx = "" } = useParams<{ uuid: string; idx: string }>();
  const [params] = useSearchParams();
  const split: Split = params.get("split") === "train" ? "train" : "val";
  const deckIdx = Number(idx);
  const { hasCompare } = useModelSelection();

  const { data: deck, isLoading } = useQuery({
    queryKey: ["deck", split, deckIdx],
    queryFn: () => apiDeck(deckIdx),
    enabled: split === "val" && Number.isFinite(deckIdx),
  });

  return (
    <div className="container py-8 space-y-6">
      <div>
        <Link
          to={`/cube/${uuid}?tab=decks`}
          className="inline-flex items-center gap-1 text-sm text-fg-2 hover:text-fg-0"
        >
          <ChevronLeft className="size-4" />
          Back to cube
        </Link>
      </div>

      <header className="border-b border-border pb-4 flex items-center gap-3 flex-wrap">
        <h1 className="text-2xl font-bold text-fg-0">Deck #{deckIdx}</h1>
        <SplitBadge split={split} />
        {deck && (
          <span className="text-fg-3 text-xs nums ml-auto">
            {deck.mainboard.length} main · {deck.sideboard?.length ?? 0} side
          </span>
        )}
      </header>

      {split === "train" && (
        <p className="text-fg-3 text-sm py-6">
          Train deck inspection coming with the split-aware deck endpoint.
        </p>
      )}

      {isLoading && <div className="text-fg-3 text-sm">Loading deck…</div>}

      {deck && (
        <DeckDiffContainer deck={deck} hasCompare={hasCompare} />
      )}
    </div>
  );
}


/** Owns the bot-deck registry so HumanPanel can ring cards the model didn't
 * pick (in either active checkpoint) and BotPanel can ring cards only it
 * picked. */
function DeckDiffContainer({
  deck, hasCompare,
}: {
  deck: { mainboard: number[]; sideboard?: number[]; cube_cards: number[] };
  hasCompare: boolean;
}) {
  const [botDecks, setBotDecks] = useState<Record<string, number[]>>({});
  const setBotDeck = useCallback((ckpt: string, cards: number[]) => {
    setBotDecks((prev) => {
      const cur = prev[ckpt];
      if (cur && cur.length === cards.length && cur.every((v, i) => v === cards[i])) return prev;
      return { ...prev, [ckpt]: cards };
    });
  }, []);

  const allBotPicks = useMemo(() => {
    const s = new Set<number>();
    for (const arr of Object.values(botDecks)) for (const i of arr) s.add(i);
    return s;
  }, [botDecks]);

  return (
    <div className="space-y-3">
      <DiffLegend hasCompare={hasCompare} />
      <div className={cn(
        "grid gap-6",
        hasCompare ? "grid-cols-1" : "grid-cols-1 lg:grid-cols-2",
      )}>
        <HumanPanel deck={deck} allBotPicks={allBotPicks} />
        <ComparePanels render={(ckpt) => (
          <BotPanel
            ckpt={ckpt}
            pool={deck.mainboard}
            humanMainboard={deck.mainboard}
            onDeck={setBotDeck}
          />
        )} />
      </div>
    </div>
  );
}


function DiffLegend({ hasCompare }: { hasCompare: boolean }) {
  return (
    <div className="flex items-center gap-4 flex-wrap text-xs text-fg-3">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-sm bg-human" />
        only in human{hasCompare ? " (vs both bots)" : ""}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-sm bg-model" />
        only in bot
      </span>
    </div>
  );
}


function HumanPanel({
  deck, allBotPicks,
}: {
  deck: { mainboard: number[]; sideboard?: number[] };
  allBotPicks: Set<number>;
}) {
  // Ring human-only when at least one bot has rendered (otherwise everything
  // would falsely look "human-only").
  const hasBots = allBotPicks.size > 0;
  const main: CardGridItem[] = deck.mainboard.map((idx) => ({
    idx,
    ring: hasBots && !allBotPicks.has(idx) ? "human" : null,
  }));
  const side: CardGridItem[] = (deck.sideboard ?? []).map((idx) => ({ idx }));
  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4 space-y-4">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-fg-0">Human-built</h2>
        <span className="text-fg-3 text-xs nums">{main.length} cards</span>
      </header>
      <CardGrid items={main} size="xs" />
      {side.length > 0 && (
        <div>
          <div className="text-fg-3 text-xs uppercase font-bold tracking-wider mb-2">
            Sideboard
          </div>
          <CardGrid items={side} size="xs" />
        </div>
      )}
    </section>
  );
}


function BotPanel({
  ckpt, pool, humanMainboard, onDeck,
}: {
  ckpt: string;
  pool: number[];
  humanMainboard: number[];
  onDeck: (ckpt: string, cards: number[]) => void;
}) {
  // Pool input is the human deck's mainboard — i.e. given the same cards a
  // human assembled with, what subset would this model build?
  const start = useStartJob("deckbuilder");
  const [jobId, setJobId] = useState<string | null>(null);
  const lastKey = useRef<string | null>(null);
  const key = `${ckpt}::${pool.join(",")}`;

  useEffect(() => {
    if (!ckpt || pool.length === 0 || lastKey.current === key) return;
    lastKey.current = key;
    setJobId(null);
    start.mutate(
      { ckpt, pool },
      { onSuccess: ({ id }) => setJobId(id) },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const job = useJob(jobId);
  const partial = job.data?.partial as { deck_so_far?: number[] } | null;
  const result = job.data?.result as DeckBuilderResult | null;
  const deckCards = result?.deck ?? partial?.deck_so_far ?? [];

  // Publish the current deck so HumanPanel can ring cards no bot picked.
  useEffect(() => {
    if (deckCards.length > 0) onDeck(ckpt, deckCards);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ckpt, deckCards.join(",")]);

  const humanSet = useMemo(() => new Set(humanMainboard), [humanMainboard]);
  const items: CardGridItem[] = deckCards.map((idx) => ({
    idx,
    ring: humanSet.has(idx) ? null : "model",
  }));

  return (
    <section className="rounded-lg border border-border bg-bg-1 p-4 space-y-4">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-fg-0">Bot-built</h2>
        {result && (
          <span className="text-fg-3 text-xs nums">
            {items.length} cards · {result.n_spells} spells · {result.n_lands} lands
          </span>
        )}
      </header>
      {job.data && job.data.status !== "done" && (
        <ProgressBar job={job.data} label={`Deck build (${ckpt})`} />
      )}
      {items.length > 0 && <CardGrid items={items} size="xs" />}
    </section>
  );
}


function SplitBadge({ split }: { split: Split }) {
  return (
    <span className={cn(
      "text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full",
      split === "val" ? "bg-good text-white" : "bg-bg-3 text-fg-2",
    )}>{split}</span>
  );
}
