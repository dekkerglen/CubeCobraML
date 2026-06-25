/** Deck rendered as vertical piles stacked by CMC. */
import { useCard } from "@/hooks/useCards";
import { useCardDrawer } from "@/hooks/useCardDrawer";
import { cn } from "@/lib/cn";


/** Group a deck's card indices into spell-piles by CMC + a single land pile. */
export function pileByCmc(deck: number[], cards: Map<number, { cmc: number; is_land: boolean; name: string }>): {
  piles: number[][]; labels: string[];
} {
  const spell: Record<number, number[]> = {};
  const lands: number[] = [];
  for (const c of deck) {
    const meta = cards.get(c);
    if (!meta) continue;
    if (meta.is_land) { lands.push(c); continue; }
    const cmc = Math.min(6, Math.floor(meta.cmc || 0));
    spell[cmc] ??= [];
    spell[cmc].push(c);
  }
  const piles: number[][] = [];
  const labels: string[] = [];
  for (const k of Object.keys(spell).map(Number).sort((a, b) => a - b)) {
    piles.push(spell[k]);
    labels.push(k === 6 ? "6+" : String(k));
  }
  if (lands.length > 0) { piles.push(lands); labels.push("L"); }
  return { piles, labels };
}


export function DeckPiles({ piles, labels }: { piles: number[][]; labels: string[] }) {
  if (piles.length === 0) {
    return <div className="text-fg-3 text-sm py-6 text-center">No cards in deck.</div>;
  }
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${piles.length}, 1fr)` }}>
      {piles.map((pile, i) => (
        <Pile key={i} cards={pile} label={labels[i]} />
      ))}
    </div>
  );
}


function Pile({ cards, label }: { cards: number[]; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-xs font-bold text-fg-3 uppercase tracking-wider">
        {label} <span className="text-fg-2 nums ml-1">{cards.length}</span>
      </div>
      <div className="relative w-full" style={{ paddingTop: `${100 + Math.max(0, cards.length - 1) * 22}%` }}>
        <div className="absolute inset-0">
          {cards.map((idx, i) => (
            <PileCard key={`${idx}-${i}`} idx={idx} top={i * 22} />
          ))}
        </div>
      </div>
    </div>
  );
}


function PileCard({ idx, top }: { idx: number; top: number }) {
  const { data: card } = useCard(idx);
  const open = useCardDrawer((s) => s.open);
  if (!card?.image) {
    return <div className="absolute w-full" style={{ top: `${top}%` }}>
      <div className="w-full aspect-[5/7] rounded-sm bg-bg-3" />
    </div>;
  }
  return (
    <button
      type="button"
      onClick={() => open(idx)}
      style={{ top: `${top}%` }}
      className={cn(
        "absolute w-full block",
        "transition-transform duration-150 hover:z-10 hover:scale-110",
      )}
    >
      <img
        src={card.image}
        alt={card.name}
        className="w-full rounded-sm shadow-md ring-0 hover:ring-2 hover:ring-accent/60"
      />
    </button>
  );
}
