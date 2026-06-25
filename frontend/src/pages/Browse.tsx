/** Card browser — global search & filter over the model vocabulary. */
import { useState } from "react";
import { Search } from "lucide-react";

import { CardGrid, type CardGridItem } from "@/components/CardGrid";
import { KpiRow, KpiTile } from "@/components/KpiTile";
import { useCardSearch } from "@/hooks/useCards";
import type { CardSearchParams } from "@/lib/api";
import { cn } from "@/lib/cn";


const TYPES = ["", "Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land"];
const SORTS = [
  { v: "elo_desc",  l: "Elo (desc)" },
  { v: "elo_asc",   l: "Elo (asc)" },
  { v: "name",      l: "Name" },
  { v: "cmc_asc",   l: "CMC (asc)" },
  { v: "cmc_desc",  l: "CMC (desc)" },
  { v: "pick_desc", l: "Pick count (desc)" },
] as const;


const COLOR_OPTIONS = ["W", "U", "B", "R", "G", "C"] as const;
const RARITY_OPTIONS = ["common", "uncommon", "rare", "mythic"] as const;

export function BrowsePage() {
  const [q, setQ] = useState("");
  const [type, setType] = useState("");
  const [cmc, setCmc] = useState<[number, number]>([0, 12]);
  const [elo, setElo] = useState<[number, number]>([1000, 2200]);
  const [sets, setSets] = useState("");
  const [colors, setColors] = useState<Set<string>>(new Set());
  const [rarities, setRarities] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<CardSearchParams["sort"]>("elo_desc");
  const [limit, setLimit] = useState(60);

  const params: CardSearchParams = {
    q, type: type || undefined,
    cmc_min: cmc[0], cmc_max: cmc[1],
    elo_min: elo[0], elo_max: elo[1],
    sets: sets.trim() || undefined,
    colors: colors.size ? [...colors].join(",") : undefined,
    rarity: rarities.size ? [...rarities].join(",") : undefined,
    sort, limit,
  };
  const { data, isLoading } = useCardSearch(params);

  const items: CardGridItem[] = (data?.items ?? []).map((c) => ({
    idx: c.idx,
    caption: (
      <>
        <strong className="block text-fg-1 truncate">{c.name}</strong>
        <span className="text-fg-3 nums">elo {c.elo.toFixed(0)} · cmc {c.cmc}</span>
      </>
    ),
  }));

  return (
    <div className="container py-8 space-y-6">
      <header className="border-b border-border pb-4">
        <h1 className="text-2xl font-bold text-fg-0">Card browser</h1>
        <p className="text-fg-2 text-sm mt-1">Filter the full model vocabulary.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
        <aside className="space-y-4 rounded-lg border border-border bg-bg-2 p-4 h-fit sticky top-20">
          <Field label="Name contains">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-fg-3" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="e.g. Lightning"
                className={cn(
                  "w-full bg-bg-3 border border-border-subtle rounded-md pl-8 pr-3 py-1.5 text-sm",
                  "outline-none focus:ring-2 focus:ring-accent placeholder:text-fg-3",
                )}
              />
            </div>
          </Field>

          <Field label="Type">
            <select value={type} onChange={(e) => setType(e.target.value)}
              className="w-full bg-bg-3 border border-border-subtle rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent">
              {TYPES.map((t) => (
                <option key={t} value={t}>{t || "Any"}</option>
              ))}
            </select>
          </Field>

          <Field label={`CMC: ${cmc[0]}–${cmc[1]}`}>
            <RangeRow vals={cmc} setVals={setCmc} min={0} max={12} step={1} />
          </Field>

          <Field label={`Elo: ${elo[0]}–${elo[1]}`}>
            <RangeRow vals={elo} setVals={setElo} min={1000} max={2200} step={50} />
          </Field>

          <Field label="Sets (codes, comma-separated)">
            <input
              value={sets}
              onChange={(e) => setSets(e.target.value)}
              placeholder="e.g. sos, mh3"
              className="w-full bg-bg-3 border border-border-subtle rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent placeholder:text-fg-3"
            />
          </Field>

          <Field label="Colors">
            <div className="flex flex-wrap gap-1.5">
              {COLOR_OPTIONS.map((c) => {
                const active = colors.has(c);
                return (
                  <button
                    key={c}
                    type="button"
                    onClick={() => setColors((prev) => {
                      const n = new Set(prev);
                      if (n.has(c)) n.delete(c); else n.add(c);
                      return n;
                    })}
                    className={cn(
                      "h-7 w-7 rounded-full border text-xs font-bold",
                      active
                        ? "bg-accent text-white border-accent"
                        : "bg-bg-3 text-fg-2 border-border hover:border-accent",
                    )}
                  >{c}</button>
                );
              })}
            </div>
          </Field>

          <Field label="Rarity">
            <div className="flex flex-wrap gap-1">
              {RARITY_OPTIONS.map((r) => {
                const active = rarities.has(r);
                return (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setRarities((prev) => {
                      const n = new Set(prev);
                      if (n.has(r)) n.delete(r); else n.add(r);
                      return n;
                    })}
                    className={cn(
                      "px-2 py-1 rounded text-xs font-medium border capitalize",
                      active
                        ? "bg-accent text-white border-accent"
                        : "bg-bg-3 text-fg-2 border-border hover:border-accent",
                    )}
                  >{r}</button>
                );
              })}
            </div>
          </Field>

          <Field label="Sort">
            <select value={sort} onChange={(e) => setSort(e.target.value as CardSearchParams["sort"])}
              className="w-full bg-bg-3 border border-border-subtle rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent">
              {SORTS.map((s) => (
                <option key={s.v} value={s.v}>{s.l}</option>
              ))}
            </select>
          </Field>

          <Field label="Results">
            <input type="number" value={limit} min={12} max={300} step={12}
              onChange={(e) => setLimit(Number(e.target.value) || 60)}
              className="w-full bg-bg-3 border border-border-subtle rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent" />
          </Field>
        </aside>

        <div className="space-y-4">
          <KpiRow>
            <KpiTile label="Total in vocab" value="37,238" />
            <KpiTile label="Matches" value={(data?.total ?? 0).toLocaleString()} />
            <KpiTile label="Showing" value={(data?.items.length ?? 0).toLocaleString()} />
          </KpiRow>

          {isLoading && data == null ? (
            <div className="py-16 text-center text-fg-3 text-sm">searching…</div>
          ) : (
            <CardGrid items={items} size="sm" />
          )}
        </div>
      </div>
    </div>
  );
}


function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider font-semibold text-fg-3 mb-1.5">{label}</div>
      {children}
    </div>
  );
}


function RangeRow({
  vals, setVals, min, max, step,
}: {
  vals: [number, number];
  setVals: (v: [number, number]) => void;
  min: number; max: number; step: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <input type="number" value={vals[0]} min={min} max={vals[1]} step={step}
        onChange={(e) => setVals([Number(e.target.value), vals[1]])}
        className="w-20 bg-bg-3 border border-border-subtle rounded-md px-2 py-1 text-sm nums" />
      <span className="text-fg-3">–</span>
      <input type="number" value={vals[1]} min={vals[0]} max={max} step={step}
        onChange={(e) => setVals([vals[0], Number(e.target.value)])}
        className="w-20 bg-bg-3 border border-border-subtle rounded-md px-2 py-1 text-sm nums" />
    </div>
  );
}
