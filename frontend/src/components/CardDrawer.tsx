/** Global card drawer — Radix Dialog slid in from the right.
 *
 * Tabs:
 *   Overview        - art + stats
 *   In cubes        - val cubes containing this card
 *   In decks        - val decks mainboarding this card
 *   In packs        - val pick records where this card was in the pack
 *   Neighbors       - encoder NN per loaded ckpt
 *   Recommendations - (later) cube_decoder thinks add this to which cubes
 */
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { CardGrid } from "@/components/CardGrid";
import { useCard } from "@/hooks/useCards";
import { useCardDrawer, type DrawerTab } from "@/hooks/useCardDrawer";
import { useModelSelection } from "@/hooks/useModelSelection";
import {
  apiCardInCubes, apiCardInDecks, apiCardInPacks, apiCardNeighbors,
  apiCardPickedByHumans, apiCube, apiDeck, apiPick,
  type CardLookupItem, type CardLookupOut,
} from "@/lib/api";
import { cn } from "@/lib/cn";


const TABS: Array<{ key: DrawerTab; label: string }> = [
  { key: "overview",  label: "Overview" },
  { key: "in-cubes",  label: "In cubes" },
  { key: "in-decks",  label: "In decks" },
  { key: "in-packs",  label: "In packs" },
  { key: "neighbors", label: "Neighbors" },
];


export function CardDrawer() {
  const { cardIdx, tab, setTab, close } = useCardDrawer();
  const open = cardIdx != null;
  const { data: card } = useCard(cardIdx);
  // Closes the drawer when the user clicks "Open full card workspace" so the
  // page nav isn't fighting the drawer overlay on the destination page.
  const dismissAndNavigate = () => close();

  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && close()}>
      <Dialog.Portal>
        <Dialog.Overlay className={cn(
          "fixed inset-0 z-40 bg-black/30 backdrop-blur-sm",
          "data-[state=open]:animate-in data-[state=closed]:animate-out",
          "data-[state=open]:fade-in data-[state=closed]:fade-out",
        )} />
        <Dialog.Content
          className={cn(
            "fixed right-0 top-0 h-screen z-50 bg-bg-1 border-l border-border",
            "w-full sm:w-[560px] md:w-[640px]",
            "flex flex-col",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=open]:slide-in-from-right data-[state=closed]:slide-out-to-right",
            "duration-200",
          )}
        >
          <Dialog.Title className="sr-only">
            {card?.name ?? "Card details"}
          </Dialog.Title>
          <header className="flex items-start justify-between p-5 border-b border-border-subtle">
            <div className="min-w-0 flex-1">
              <Dialog.Description className="text-fg-3 text-xs uppercase tracking-wider font-bold">
                Card
              </Dialog.Description>
              <h2 className="text-2xl font-bold text-fg-0 truncate">
                {card?.name ?? "…"}
              </h2>
              {card?.type && (
                <div className="mt-1 text-fg-2 text-sm">{card.type}</div>
              )}
              {cardIdx != null && (
                <Link
                  to={`/card/${cardIdx}`}
                  onClick={dismissAndNavigate}
                  className="mt-2 inline-flex items-center gap-1 text-xs text-link hover:underline"
                >
                  Open full card workspace
                  <ExternalLink className="size-3" />
                </Link>
              )}
            </div>
            <Dialog.Close className="ml-3 p-1.5 rounded-md text-fg-3 hover:bg-bg-3 hover:text-fg-0">
              <X className="size-5" />
            </Dialog.Close>
          </header>

          <Tabs.Root value={tab} onValueChange={(v) => setTab(v as DrawerTab)}
                     className="flex-1 flex flex-col min-h-0">
            <Tabs.List className="flex gap-1 px-3 pt-3 border-b border-border-subtle overflow-x-auto scrollbar-thin">
              {TABS.map((t) => (
                <Tabs.Trigger
                  key={t.key} value={t.key}
                  className={cn(
                    "px-3 py-1.5 text-sm rounded-md whitespace-nowrap font-medium transition-colors",
                    "text-fg-2 hover:text-fg-0",
                    "data-[state=active]:bg-accent-subtle data-[state=active]:text-accent",
                  )}
                >
                  {t.label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            <div className="flex-1 overflow-y-auto scrollbar-thin">
              {cardIdx != null && (
                <>
                  <Tabs.Content value="overview" className="p-5 outline-none">
                    <OverviewTab cardIdx={cardIdx} />
                  </Tabs.Content>
                  <Tabs.Content value="in-cubes" className="p-5 outline-none">
                    <InCubesTab cardIdx={cardIdx} />
                  </Tabs.Content>
                  <Tabs.Content value="in-decks" className="p-5 outline-none">
                    <InDecksTab cardIdx={cardIdx} />
                  </Tabs.Content>
                  <Tabs.Content value="in-packs" className="p-5 outline-none">
                    <InPacksTab cardIdx={cardIdx} />
                  </Tabs.Content>
                  <Tabs.Content value="neighbors" className="p-5 outline-none">
                    <NeighborsTab cardIdx={cardIdx} />
                  </Tabs.Content>
                </>
              )}
            </div>
          </Tabs.Root>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}


function OverviewTab({ cardIdx }: { cardIdx: number }) {
  const { data: card } = useCard(cardIdx);
  if (!card) return null;
  return (
    <div className="space-y-5">
      {card.image_normal && (
        <div className="flex justify-center">
          <img
            src={card.image_normal}
            alt={card.name}
            className="w-full max-w-[320px] rounded-lg shadow-2xl"
          />
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="CMC" value={String(card.cmc)} />
        <Stat label="Elo" value={card.elo.toFixed(0)} />
        <Stat label="Train pick count" value={card.pick_count.toLocaleString()} />
        <Stat label="Val pick count"   value={card.val_pick_count.toLocaleString()} />
        <Stat label="In train cubes"   value={card.cube_count.toLocaleString()} />
        <Stat label="In val cubes"     value={card.val_cube_count.toLocaleString()} />
      </div>
      {(card.set || card.rarity || card.color_identity?.length) && (
        <div className="text-xs text-fg-2 flex items-center gap-3 flex-wrap">
          {card.set && (
            <span className="uppercase font-mono px-1.5 py-0.5 rounded border border-border bg-bg-3">
              {card.set}
              {card.set_name && <span className="ml-1 text-fg-3">· {card.set_name}</span>}
            </span>
          )}
          {card.rarity && (
            <span className="capitalize font-medium">{card.rarity}</span>
          )}
          {(card.color_identity ?? []).length > 0 && (
            <span className="font-mono">{(card.color_identity ?? []).join("")}</span>
          )}
        </div>
      )}
      <div className="space-y-1 text-xs">
        <div className="text-fg-3 font-mono break-all">
          oracle id: <span className="text-fg-2">{card.oracle_id}</span>
        </div>
        <div className="text-fg-3 font-mono">
          vocab idx: <span className="text-fg-2">{card.idx}</span>
        </div>
      </div>
    </div>
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


export function PaginatedList({
  fetcher, label, render,
}: {
  fetcher: (page: number) => Promise<CardLookupOut>;
  label: string;
  render: (items: CardLookupItem[]) => React.ReactNode;
}) {
  const [page, setPage] = useState(0);
  const { data, isLoading } = useQuery({
    queryKey: ["lookup", label, page],
    queryFn: () => fetcher(page),
  });
  if (isLoading && !data) return <div className="text-fg-3 py-6 text-center text-sm">loading…</div>;
  if (!data || data.total === 0) return (
    <div className="py-10 text-center">
      <p className="text-fg-2 font-medium">No {label}.</p>
    </div>
  );
  const pages = Math.ceil(data.total / data.per_page);
  return (
    <div className="space-y-4">
      <div className="text-fg-3 text-xs">
        {data.total.toLocaleString()} {label} · page {data.page + 1} of {pages}
      </div>
      {render(data.items)}
      {pages > 1 && (
        <div className="flex items-center justify-between pt-2">
          <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}
            className="text-xs text-fg-2 hover:text-fg-0 disabled:text-fg-3 disabled:cursor-not-allowed px-3 py-1 rounded-md hover:bg-bg-3">◀ prev</button>
          <span className="text-fg-3 text-xs nums">{page + 1} / {pages}</span>
          <button onClick={() => setPage((p) => Math.min(pages - 1, p + 1))} disabled={page >= pages - 1}
            className="text-xs text-fg-2 hover:text-fg-0 disabled:text-fg-3 disabled:cursor-not-allowed px-3 py-1 rounded-md hover:bg-bg-3">next ▶</button>
        </div>
      )}
    </div>
  );
}


function InCubesTab({ cardIdx }: { cardIdx: number }) {
  return (
    <PaginatedList
      fetcher={(p) => apiCardInCubes(cardIdx, p, 12)}
      label="val cubes"
      render={(items) => (
        <ul className="divide-y divide-border-subtle/40">
          {items.map((it) => <CubeRow key={it.idx} item={it} />)}
        </ul>
      )}
    />
  );
}


export function CubeRow({ item }: { item: CardLookupItem }) {
  const { data } = useQuery({ queryKey: ["cube", item.idx], queryFn: () => apiCube(item.idx) });
  const inner = (
    <div className="py-2 flex items-center justify-between gap-3 -mx-2 px-2 rounded hover:bg-bg-2">
      <div>
        <div className="text-fg-0 font-medium flex items-center gap-2">
          {item.cube_uuid ? `Cube ${item.cube_uuid.slice(0, 8)}…` : `Cube #${item.idx}`}
          <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-good text-white">val</span>
        </div>
        <div className="text-fg-3 text-xs nums">
          {data ? `${data.cards.length} cards` : "…"}
        </div>
      </div>
    </div>
  );
  return (
    <li>
      {item.cube_uuid
        ? <Link to={`/cube/${item.cube_uuid}`}>{inner}</Link>
        : inner /* fallback when cube_uuids.json sidecar isn't present yet */}
    </li>
  );
}


function InDecksTab({ cardIdx }: { cardIdx: number }) {
  return (
    <PaginatedList
      fetcher={(p) => apiCardInDecks(cardIdx, p, 12)}
      label="val decks"
      render={(items) => (
        <ul className="divide-y divide-border-subtle/40">
          {items.map((it) => <DeckRow key={it.idx} item={it} />)}
        </ul>
      )}
    />
  );
}


export function DeckRow({ item }: { item: CardLookupItem }) {
  const { data } = useQuery({ queryKey: ["deck", item.idx], queryFn: () => apiDeck(item.idx) });
  const inner = (
    <div className="py-2 flex items-center justify-between -mx-2 px-2 rounded hover:bg-bg-2">
      <div>
        <div className="text-fg-0 font-medium">Deck #{item.idx}</div>
        <div className="text-fg-3 text-xs nums">
          {data ? `${data.mainboard.length} MB · ${data.sideboard.length} SB · cube ${data.cube_cards.length}` : "…"}
        </div>
      </div>
    </div>
  );
  return (
    <li>
      {item.cube_uuid
        ? <Link to={`/cube/${item.cube_uuid}/deck/${item.idx}?split=val`}>{inner}</Link>
        : inner}
    </li>
  );
}


function InPacksTab({ cardIdx }: { cardIdx: number }) {
  return (
    <div className="space-y-6">
      <section>
        <h3 className="text-xs font-bold uppercase tracking-wider text-fg-3 mb-2">In packs</h3>
        <PaginatedList
          fetcher={(p) => apiCardInPacks(cardIdx, p, 8)}
          label="picks where this card was offered"
          render={(items) => (
            <ul className="space-y-2">
              {items.map((it) => <PickRow key={it.idx} item={it} cardIdx={cardIdx} />)}
            </ul>
          )}
        />
      </section>

      <section>
        <h3 className="text-xs font-bold uppercase tracking-wider text-fg-3 mb-2">Humans picked this</h3>
        <PaginatedList
          fetcher={(p) => apiCardPickedByHumans(cardIdx, p, 8)}
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


export function PickRow({
  item, cardIdx, pickedByHuman,
}: {
  item: CardLookupItem;
  cardIdx: number;
  pickedByHuman?: boolean;
}) {
  const { data } = useQuery({ queryKey: ["pick", item.idx], queryFn: () => apiPick(item.idx) });
  if (!data) {
    return <li className="text-fg-3 text-sm">…</li>;
  }
  const human = data.pick;
  const isThis = human === cardIdx;
  // Prefer the nested cube-scoped draft URL (Phase V). Falls back to the
  // legacy /drafts/:id route if a cube_uuid isn't resolved yet (sidecar
  // missing) — that route still resolves the same DraftReplay page.
  const draftHref = item.draft_id
    ? (item.cube_uuid
        ? `/cube/${item.cube_uuid}/draft/${item.draft_id}?step=${item.step ?? 0}`
        : `/drafts/${item.draft_id}?step=${item.step ?? 0}`)
    : null;
  const card = (
    <div className="rounded-md border border-border-subtle bg-bg-2 px-3 py-2 text-sm hover:border-accent transition-colors">
      <div className="flex items-center gap-2">
        <span className="font-mono text-fg-3 text-xs">pick #{item.idx}</span>
        {item.draft_id && (
          <>
            <span className="text-fg-3 text-xs">·</span>
            <span className="font-mono text-fg-2 text-xs">draft {item.draft_id.slice(0, 8)}… step {item.step}</span>
          </>
        )}
        <span className="text-fg-3 text-xs">·</span>
        <span className="text-fg-2 text-xs nums">{data.pool.length} pool · {data.pack.length} pack</span>
      </div>
      <div className="mt-1.5 flex items-center gap-2">
        <span className={cn(
          "text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded",
          isThis ? "bg-human text-white" : "bg-bg-3 text-fg-2",
        )}>
          human: {isThis ? "this card" : `#${human}`}
        </span>
        {pickedByHuman && (
          <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-good text-white">
            picked
          </span>
        )}
      </div>
    </div>
  );
  return (
    <li>
      {draftHref ? <Link to={draftHref}>{card}</Link> : card}
    </li>
  );
}


function NeighborsTab({ cardIdx }: { cardIdx: number }) {
  const { activeKeys } = useModelSelection();
  if (activeKeys.length === 0) {
    return <div className="text-fg-3 text-sm py-6 text-center">Select a model from the header.</div>;
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
    queryFn: () => apiCardNeighbors(cardIdx, ckpt, 20),
  });
  return (
    <section>
      <div className="text-[11px] uppercase font-bold tracking-wider text-fg-3 mb-2">
        Model {String.fromCharCode(65 + idx)} · {ckpt}
      </div>
      {isLoading && !data ? (
        <div className="text-fg-3 text-sm">building embeddings (first hit only, ~5s)…</div>
      ) : data?.neighbors.length ? (
        <CardGrid
          items={data.neighbors.map((n) => ({
            idx: n.idx,
            caption: <span className="nums text-fg-2">{n.similarity.toFixed(3)}</span>,
          }))}
          size="xs"
        />
      ) : null}
    </section>
  );
}


