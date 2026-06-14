/** Ranked pick list — left rail card + name + tags + probability bar.
 * Each row is itself clickable into the card drawer.
 */
import { useCard } from "@/hooks/useCards";
import { useCardDrawer } from "@/hooks/useCardDrawer";
import { cn } from "@/lib/cn";


export interface RankedPick {
  idx: number;
  p: number;
  rank: number;
  isHuman?: boolean;
  isTop?: boolean;
}


export function RankedPickList({ rows }: { rows: RankedPick[] }) {
  if (!rows.length) return <div className="text-fg-3 text-sm">No predictions.</div>;
  return (
    <ul className="space-y-2">
      {rows.map((r) => <Row key={r.idx} row={r} />)}
    </ul>
  );
}


function Row({ row }: { row: RankedPick }) {
  const { data: card } = useCard(row.idx);
  const open = useCardDrawer((s) => s.open);
  const cls = row.isHuman && row.isTop ? "border-good"
            : row.isHuman ? "border-human"
            : row.isTop ? "border-model"
            : "border-border";
  const numCls = row.isHuman && row.isTop ? "text-good"
              : row.isHuman ? "text-human"
              : row.isTop ? "text-model"
              : "text-fg-3";
  return (
    <li>
      <button
        type="button"
        onClick={() => open(row.idx)}
        className={cn(
          "w-full grid grid-cols-[40px_64px_1fr] gap-3 items-center",
          "px-3 py-2 rounded-md border-2 bg-bg-2 hover:bg-bg-3 transition-colors",
          "text-left",
          cls,
        )}
      >
        <div className={cn("text-xl font-extrabold nums text-center", numCls)}>
          #{row.rank}
        </div>
        <div>
          {card?.image && (
            <img src={card.image} alt={card.name} className="w-full rounded-sm" />
          )}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-fg-0 font-semibold truncate">{card?.name ?? "…"}</span>
            {row.isTop && <Tag tone="model">model · {(row.p * 100).toFixed(1)}%</Tag>}
            {row.isHuman && <Tag tone="human">human pick</Tag>}
          </div>
          <div className="mt-1.5 flex items-center gap-2">
            <div className="flex-1 h-2 bg-bg-3 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-model to-accent-hover transition-all"
                style={{ width: `${Math.max(0, Math.min(100, row.p * 100))}%` }}
              />
            </div>
            <span className="nums text-sm text-fg-1 font-semibold w-14 text-right">
              {(row.p * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </button>
    </li>
  );
}


function Tag({ tone, children }: { tone: "human" | "model"; children: React.ReactNode }) {
  return (
    <span className={cn(
      "text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded",
      tone === "human" ? "bg-human text-white" : "bg-model text-white",
    )}>{children}</span>
  );
}
