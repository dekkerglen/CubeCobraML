/** Debounced cube directory search. */
import { useEffect, useState } from "react";

import { apiSearchCubes } from "@/lib/api";
import type { CubeDirectoryEntry } from "@/lib/types";


export function useCubeSearch(query: string, split: "all" | "val" = "all", limit = 12) {
  const [items, setItems] = useState<CubeDirectoryEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setItems([]);
      return;
    }
    setLoading(true);
    let cancelled = false;
    const handle = setTimeout(async () => {
      try {
        const r = await apiSearchCubes({ q, split, limit });
        if (!cancelled) setItems(r.items);
      } catch {
        if (!cancelled) setItems([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 120);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [query, split, limit]);

  return { items, loading };
}
