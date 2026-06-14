/** Card workspace — first-class page parity with /cube/:uuid.
 *
 * The CardDrawer keeps its role as a quick-peek panel from anywhere; this
 * page is the full home for everything we know about one card. Each tab
 * re-renders the same data the drawer shows, but at workspace scale (bigger
 * image, grid layouts where the drawer used lists).
 *
 * Reuses the drawer's `PaginatedList`, `CubeRow`, `DeckRow`, `PickRow` rows
 * so the deep-linked content stays in sync with the drawer rendering.
 */
import * as Tabs from "@radix-ui/react-tabs";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import {
  CubeRow, DeckRow, PaginatedList, PickRow,
} from "@/components/CardDrawer";
import { CardGrid } from "@/components/CardGrid";
import { useCard } from "@/hooks/useCards";
import { useModelSelection } from "@/hooks/useModelSelection";
import {
  apiCardInCubes, apiCardInDecks, apiCardInPacks,
  apiCardNeighbors, apiCardPickedByHumans,
} from "@/lib/api";
import { cn } from "@/lib/cn";


export function CardDetailPage() {
  const { idx: rawIdx = "" } = useParams<{ idx: string }>();
  const cardIdx = Number(rawIdx);
  const { data: card, isLoading, error } = useCard(Number.isFinite(cardIdx) ? cardIdx : null);

  if (!Number.isFinite(cardIdx) || cardIdx < 0) {
    return (
      <div className="container py-8">
        <BackLink />
        <div className="text-bad">Invalid card index.</div>
      </div>
    );
  }

  return (
    <div className="container py-8 space-y-6">
      <BackLink />
      {isLoading && <div className="text-fg-2">Loading card…</div>}
      {error && <div className="text-bad">Could not load card: {(error as Error).message}</div>}
      {card && (
        <>
          <header className="border-b border-border pb-5 flex items-start gap-6 flex-wrap">
            {card.image_normal && (
              <img
                src={card.image_normal}
                alt={card.name}
                className="w-48 rounded-lg shadow-lg border border-border"
              />
            )}
            <div className="flex-1 min-w-0 space-y-2">
              <h1 className="text-3xl font-bold text-fg-0">{card.name}</h1>
              {card.type && <div className="text-fg-2 text-sm">{card.type}</div>}
              <div className="flex flex-wrap gap-2 items-center text-xs">
                {card.set && (
                  <span className="uppercase font-mono px-1.5 py-0.5 rounded border border-border bg-bg-3">
                    {card.set}
                    {card.set_name && <span className="ml-1 text-fg-3">· {card.set_name}</span>}
                  </span>
                )}
                {card.rarity && <span className="capitalize font-medium text-fg-2">{card.rarity}</span>}
                {(card.color_identity ?? []).length > 0 && (
                  <span className="font-mono text-fg-2">{(card.color_identity ?? []).join("")}</span>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-w-md pt-2">
                <Stat label="CMC" value={String(card.cmc)} />
                <Stat label="Elo" value={card.elo.toFixed(0)} />
                <Stat label="Train picks" value={card.pick_count.toLocaleString()} />
                <Stat label="In cubes" value={card.cube_count.toLocaleString()} />
              </div>
              <div className="pt-1 text-fg-3 text-xs font-mono break-all">
                oracle id: {card.oracle_id} · vocab idx: {card.idx}
              </div>
            </div>
          </header>

          <Tabs.Root defaultValue="in-cubes" className="space-y-5">
            <Tabs.List className="flex border-b border-border flex-wrap">
              <TabTrigger value="in-cubes">In cubes</TabTrigger>
              <TabTrigger value="in-decks">In decks</TabTrigger>
              <TabTrigger value="in-packs">In packs</TabTrigger>
              <TabTrigger value="neighbors">Neighbors</TabTrigger>
            </Tabs.List>
            <Tabs.Content value="in-cubes">
              <InCubesPanel cardIdx={cardIdx} />
            </Tabs.Content>
            <Tabs.Content value="in-decks">
              <InDecksPanel cardIdx={cardIdx} />
            </Tabs.Content>
            <Tabs.Content value="in-packs">
              <InPacksPanel cardIdx={cardIdx} />
            </Tabs.Content>
            <Tabs.Content value="neighbors">
              <NeighborsPanel cardIdx={cardIdx} />
            </Tabs.Content>
          </Tabs.Root>
        </>
      )}
    </div>
  );
}


function BackLink() {
  return (
    <Link to="/browse" className="inline-flex items-center gap-1 text-sm text-fg-2 hover:text-fg-0">
      <ChevronLeft className="size-4" /> Back to cards
    </Link>
  );
}


function TabTrigger({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        "px-4 py-2 text-sm font-medium border-b-2 border-transparent",
        "text-fg-2 hover:text-fg-0 data-[state=active]:text-fg-0",
        "data-[state=active]:border-accent transition-colors",
      )}
    >
      {children}
    </Tabs.Trigger>
  );
}


function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-bg-2 border border-border-subtle px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-fg-3 font-bold">{label}</div>
      <div className="text-lg font-bold text-fg-0 nums">{value}</div>
    </div>
  );
}


function InCubesPanel({ cardIdx }: { cardIdx: number }) {
  return (
    <PaginatedList
      fetcher={(p) => apiCardInCubes(cardIdx, p, 24)}
      label="val cubes"
      render={(items) => (
        <ul className="divide-y divide-border-subtle/40">
          {items.map((it) => <CubeRow key={it.idx} item={it} />)}
        </ul>
      )}
    />
  );
}


function InDecksPanel({ cardIdx }: { cardIdx: number }) {
  return (
    <PaginatedList
      fetcher={(p) => apiCardInDecks(cardIdx, p, 24)}
      label="val decks"
      render={(items) => (
        <ul className="divide-y divide-border-subtle/40">
          {items.map((it) => <DeckRow key={it.idx} item={it} />)}
        </ul>
      )}
    />
  );
}


function InPacksPanel({ cardIdx }: { cardIdx: number }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <section>
        <h3 className="text-sm font-semibold text-fg-0 mb-3">Offered in pack</h3>
        <PaginatedList
          fetcher={(p) => apiCardInPacks(cardIdx, p, 12)}
          label="picks where this card was offered"
          render={(items) => (
            <ul className="space-y-2">
              {items.map((it) => <PickRow key={it.idx} item={it} cardIdx={cardIdx} />)}
            </ul>
          )}
        />
      </section>
      <section>
        <h3 className="text-sm font-semibold text-fg-0 mb-3">Humans picked this</h3>
        <PaginatedList
          fetcher={(p) => apiCardPickedByHumans(cardIdx, p, 12)}
          label="picks where humans took this card"
          render={(items) => (
            <ul className="space-y-2">
              {items.map((it) => <PickRow key={it.idx} item={it} cardIdx={cardIdx} pickedByHuman />)}
            </ul>
          )}
        />
      </section>
    </div>
  );
}


function NeighborsPanel({ cardIdx }: { cardIdx: number }) {
  const { activeKeys } = useModelSelection();
  if (activeKeys.length === 0) {
    return <div className="text-fg-3 text-sm py-6">Select a model from the header to view embedding neighbors.</div>;
  }
  return (
    <div className="space-y-6">
      {activeKeys.map((ckpt, i) => (
        <NeighborsForCkpt key={ckpt} cardIdx={cardIdx} ckpt={ckpt} idx={i} />
      ))}
    </div>
  );
}


function NeighborsForCkpt({ cardIdx, ckpt, idx }: { cardIdx: number; ckpt: string; idx: number }) {
  const { data, isLoading } = useQuery({
    queryKey: ["neighbors", cardIdx, ckpt],
    queryFn: () => apiCardNeighbors(cardIdx, ckpt, 40),
  });
  return (
    <section>
      <h3 className="text-sm font-semibold text-fg-0 mb-3">
        Model {String.fromCharCode(65 + idx)} · <span className="font-mono text-fg-2 text-xs">{ckpt}</span>
      </h3>
      {isLoading && !data ? (
        <div className="text-fg-3 text-sm">building embeddings (first hit only, ~5s)…</div>
      ) : data?.neighbors.length ? (
        <CardGrid
          items={data.neighbors.map((n) => ({
            idx: n.idx,
            caption: <span className="nums text-fg-2">{n.similarity.toFixed(3)}</span>,
          }))}
          size="sm"
        />
      ) : null}
    </section>
  );
}
