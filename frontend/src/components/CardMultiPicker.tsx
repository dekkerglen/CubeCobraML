/** Compact card multi-selector: search → results → chip list.
 *
 * Each "+" adds the card to the local selection. Selected cards render as
 * removable chips (× to drop). Returns the selection via `onChange`.
 */
import { useEffect, useState } from "react";
import { Search, X } from "lucide-react";

import { useCard } from "@/hooks/useCards";
import { apiSearchCards } from "@/lib/api";
import type { Card } from "@/lib/types";
import { cn } from "@/lib/cn";


export function CardMultiPicker({
  value, onChange, placeholder = "type to search…",
}: {
  value: number[];
  onChange: (v: number[]) => void;
  placeholder?: string;
}) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Card[]>([]);

  useEffect(() => {
    if (!q.trim()) { setHits([]); return; }
    const handle = setTimeout(() => {
      apiSearchCards({ q, limit: 8 }).then((r) => setHits(r.items)).catch(() => setHits([]));
    }, 120);
    return () => clearTimeout(handle);
  }, [q]);

  function add(idx: number) {
    if (!value.includes(idx)) onChange([...value, idx]);
  }
  function remove(idx: number) {
    onChange(value.filter((v) => v !== idx));
  }

  return (
    <div className="space-y-2">
      <div className="relative">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-fg-3" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={placeholder}
          className={cn(
            "w-full bg-bg-3 border border-border-subtle rounded-md pl-8 pr-3 py-1.5 text-sm",
            "outline-none focus:ring-2 focus:ring-accent placeholder:text-fg-3",
          )}
        />
      </div>
      {hits.length > 0 && (
        <div className="rounded-md border border-border-subtle bg-bg-2 max-h-56 overflow-y-auto scrollbar-thin">
          {hits.map((h) => (
            <button key={h.idx} type="button" onClick={() => { add(h.idx); setQ(""); setHits([]); }}
              className="w-full text-left px-2 py-1.5 text-sm text-fg-1 hover:bg-bg-3 flex items-center gap-2">
              {h.image && <img src={h.image} alt="" className="h-8 w-6 object-cover rounded-sm" />}
              <span className="truncate">{h.name}</span>
            </button>
          ))}
        </div>
      )}
      {value.length > 0 && (
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-fg-3 font-bold">
            {value.length} selected
          </div>
          <ul className="flex flex-col gap-0.5 max-h-72 overflow-y-auto scrollbar-thin">
            {value.map((idx) => <Chip key={idx} idx={idx} onRemove={() => remove(idx)} />)}
          </ul>
          <button onClick={() => onChange([])}
            className="text-xs text-fg-3 hover:text-fg-1">clear all</button>
        </div>
      )}
    </div>
  );
}


function Chip({ idx, onRemove }: { idx: number; onRemove: () => void }) {
  const { data: card } = useCard(idx);
  return (
    <li className="flex items-center gap-2 text-sm py-0.5 group">
      <span className="text-fg-1 truncate flex-1">{card?.name ?? "…"}</span>
      <button onClick={onRemove}
        className="p-0.5 text-fg-3 hover:text-bad opacity-0 group-hover:opacity-100 transition-opacity">
        <X className="size-3.5" />
      </button>
    </li>
  );
}
