/** Anchored card filter for the Cube workspace.
 *
 * Sits above the cube's tabs and exposes the same filter vocabulary as the
 * /browse page (name contains, type, CMC range, colors) plus a "pinned card"
 * autocomplete. State lives in URL search params so it persists across tab
 * switches and survives a refresh.
 *
 * Other parts of the cube workspace call `useCubeCardFilter()` to get the
 * resolved predicate + active flag and apply it to their data.
 */
import { useQuery } from "@tanstack/react-query";
import { Search, X } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { CardMultiPicker } from "@/components/CardMultiPicker";
import { apiCardsMany } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { Card } from "@/lib/types";


const LS_PREFIX = "cubecobra-ml.cube-filter.";
const FILTER_KEYS = ["fq", "ft", "fcmcmin", "fcmcmax", "fc", "fpin"] as const;


function readLS(uuid: string | undefined): Record<string, string> {
  if (!uuid || typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(LS_PREFIX + uuid);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}


function writeLS(uuid: string | undefined, params: URLSearchParams): void {
  if (!uuid || typeof window === "undefined") return;
  const snap: Record<string, string> = {};
  for (const k of FILTER_KEYS) {
    const v = params.get(k);
    if (v) snap[k] = v;
  }
  try {
    if (Object.keys(snap).length === 0) {
      window.localStorage.removeItem(LS_PREFIX + uuid);
    } else {
      window.localStorage.setItem(LS_PREFIX + uuid, JSON.stringify(snap));
    }
  } catch {
    // ignore
  }
}


const TYPES = ["", "Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land"];
const COLOR_OPTIONS = ["W", "U", "B", "R", "G", "C"] as const;


export interface CubeCardFilterState {
  q: string;
  type: string;
  cmcMin: number;
  cmcMax: number;
  colors: Set<string>;
  pinned: number | null;
}


export type CardPredicate = (card: Card) => boolean;


export function useCubeCardFilter() {
  const [params, setParams] = useSearchParams();
  const { uuid } = useParams<{ uuid: string }>();
  const hydratedFor = useRef<string | null>(null);

  // Hydrate URL from localStorage when entering a cube with bare filter params.
  // Once URL has anything we own, we stop hydrating until the cube changes.
  useEffect(() => {
    if (!uuid || hydratedFor.current === uuid) return;
    const hasAny = FILTER_KEYS.some((k) => params.get(k));
    if (hasAny) {
      hydratedFor.current = uuid;
      return;
    }
    const stored = readLS(uuid);
    if (Object.keys(stored).length > 0) {
      const p = new URLSearchParams(params);
      for (const [k, v] of Object.entries(stored)) p.set(k, v);
      setParams(p, { replace: true });
    }
    hydratedFor.current = uuid;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uuid]);

  const filter: CubeCardFilterState = useMemo(() => ({
    q: params.get("fq") || "",
    type: params.get("ft") || "",
    cmcMin: numOr(params.get("fcmcmin"), 0),
    cmcMax: numOr(params.get("fcmcmax"), 12),
    colors: new Set((params.get("fc") || "").split(",").filter(Boolean)),
    pinned: params.get("fpin") ? Number(params.get("fpin")) : null,
  }), [params]);

  const setFilter = (next: Partial<CubeCardFilterState>) => {
    const p = new URLSearchParams(params);
    const merged = { ...filter, ...next };
    setIf(p, "fq", merged.q);
    setIf(p, "ft", merged.type);
    setIf(p, "fcmcmin", merged.cmcMin !== 0 ? String(merged.cmcMin) : "");
    setIf(p, "fcmcmax", merged.cmcMax !== 12 ? String(merged.cmcMax) : "");
    setIf(p, "fc", [...merged.colors].join(","));
    setIf(p, "fpin", merged.pinned != null ? String(merged.pinned) : "");
    setParams(p, { replace: true });
    writeLS(uuid, p);
  };

  const isActive = !!(
    filter.q || filter.type ||
    filter.cmcMin !== 0 || filter.cmcMax !== 12 ||
    filter.colors.size > 0 || filter.pinned != null
  );

  const predicate: CardPredicate = useMemo(() => {
    if (!isActive) return () => true;
    const qLower = filter.q.toLowerCase();
    return (card: Card) => {
      if (filter.pinned != null && card.idx !== filter.pinned) return false;
      if (qLower && !card.name.toLowerCase().includes(qLower)) return false;
      if (filter.type && !card.type.toLowerCase().includes(filter.type.toLowerCase())) return false;
      const cmc = card.cmc ?? 0;
      if (cmc < filter.cmcMin || cmc > filter.cmcMax) return false;
      if (filter.colors.size > 0) {
        const ci = card.color_identity ?? [];
        const isColorless = ci.length === 0;
        const wantsC = filter.colors.has("C");
        const matched = ci.some((c) => filter.colors.has(c)) || (isColorless && wantsC);
        if (!matched) return false;
      }
      return true;
    };
  }, [filter, isActive]);

  return { filter, setFilter, predicate, isActive };
}


/** Resolve the active filter to the matching subset of the given cube cards.
 * Fetches card metadata via /cards/many once and re-uses the predicate. Returns
 * `null` when the filter is inactive (caller should not send a filter param). */
export function useFilteredCubeCardIdxs(cubeCards: number[]): number[] | null {
  const { predicate, isActive } = useCubeCardFilter();
  const { data: meta } = useQuery({
    queryKey: ["cards-many", "cube", cubeCards.length, cubeCards[0] ?? -1, cubeCards[cubeCards.length - 1] ?? -1],
    queryFn: () => apiCardsMany(cubeCards),
    enabled: cubeCards.length > 0,
    staleTime: Infinity,
  });
  return useMemo(() => {
    if (!isActive) return null;
    if (!meta) return [];
    return meta.filter(predicate).map((c) => c.idx);
  }, [isActive, meta, predicate]);
}


export function CubeCardFilter() {
  const { filter, setFilter, isActive } = useCubeCardFilter();

  const toggleColor = (c: string) => {
    const next = new Set(filter.colors);
    if (next.has(c)) next.delete(c);
    else next.add(c);
    setFilter({ colors: next });
  };

  return (
    <div className="rounded-md border border-border bg-bg-1 p-3 flex items-center gap-3 flex-wrap">
      <div className="text-xs uppercase tracking-wider font-bold text-fg-3 shrink-0">
        Filter
      </div>

      <div className="relative">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-fg-3" />
        <input
          value={filter.q}
          onChange={(e) => setFilter({ q: e.target.value })}
          placeholder="name…"
          className={cn(
            "bg-bg-3 border border-border-subtle rounded-md pl-8 pr-3 py-1.5 text-sm w-44",
            "outline-none focus:ring-2 focus:ring-accent placeholder:text-fg-3",
          )}
        />
      </div>

      <select
        value={filter.type}
        onChange={(e) => setFilter({ type: e.target.value })}
        className="bg-bg-3 border border-border-subtle rounded-md px-2 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent"
      >
        {TYPES.map((t) => (
          <option key={t} value={t}>{t || "Any type"}</option>
        ))}
      </select>

      <label className="text-xs text-fg-3 flex items-center gap-1">
        CMC
        <input
          type="number"
          min={0}
          max={12}
          value={filter.cmcMin}
          onChange={(e) => setFilter({ cmcMin: Number(e.target.value) || 0 })}
          className="w-12 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 text-sm nums"
        />
        –
        <input
          type="number"
          min={0}
          max={12}
          value={filter.cmcMax}
          onChange={(e) => setFilter({ cmcMax: Number(e.target.value) || 12 })}
          className="w-12 bg-bg-3 border border-border-subtle rounded px-1.5 py-1 text-sm nums"
        />
      </label>

      <div className="inline-flex gap-1">
        {COLOR_OPTIONS.map((c) => {
          const on = filter.colors.has(c);
          return (
            <button
              key={c}
              type="button"
              onClick={() => toggleColor(c)}
              className={cn(
                "size-7 rounded-full border text-xs font-bold",
                on
                  ? "bg-accent text-white border-accent"
                  : "bg-bg-3 text-fg-2 border-border hover:text-fg-0",
              )}
            >{c}</button>
          );
        })}
      </div>

      <div className="w-56 shrink-0">
        <CardMultiPicker
          value={filter.pinned != null ? [filter.pinned] : []}
          onChange={(v) => setFilter({ pinned: v.length ? v[v.length - 1] : null })}
          placeholder="pin a card…"
        />
      </div>

      {isActive && (
        <button
          type="button"
          onClick={() => setFilter({
            q: "", type: "", cmcMin: 0, cmcMax: 12,
            colors: new Set(), pinned: null,
          })}
          className="inline-flex items-center gap-1 text-xs text-fg-3 hover:text-fg-1"
        >
          <X className="size-3" />
          clear
        </button>
      )}
    </div>
  );
}


function numOr(s: string | null, d: number): number {
  if (!s) return d;
  const n = Number(s);
  return Number.isFinite(n) ? n : d;
}

function setIf(p: URLSearchParams, key: string, val: string) {
  if (val) p.set(key, val);
  else p.delete(key);
}
