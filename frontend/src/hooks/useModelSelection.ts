/** URL-synced model selection: primary `ckpt` + optional `compare` for A/B mode.
 *
 * Sticky across navigation via localStorage: if the URL doesn't carry `ckpt`
 * or `compare`, hydrate from localStorage. Selecting from the header writes
 * both URL and localStorage, so:
 *   - sharable links (URL wins when present)
 *   - new-page navigations preserve the last selection (localStorage wins
 *     when URL is bare)
 *   - hard refresh preserves both
 */
import { useCallback, useEffect, useMemo } from "react";
import { useUrlState } from "@/lib/url";


const LS_KEY = "cubecobra-ml.model-selection";


interface Persisted {
  ckpt: string | null;
  compare: string | null;
}


function readLS(): Persisted {
  if (typeof window === "undefined") return { ckpt: null, compare: null };
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return { ckpt: null, compare: null };
    const parsed = JSON.parse(raw) as Persisted;
    return { ckpt: parsed.ckpt ?? null, compare: parsed.compare ?? null };
  } catch {
    return { ckpt: null, compare: null };
  }
}


function writeLS(next: Persisted): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LS_KEY, JSON.stringify(next));
  } catch {
    // quota / private-mode — ignore; URL is still source of truth in-session
  }
}


export function useModelSelection() {
  const url = useUrlState();
  const urlCkpt = url.get("ckpt");
  const urlCompare = url.get("compare");

  // Hydrate URL from localStorage on first mount when bare. Done in an effect
  // so we don't write during render. After hydration, URL drives.
  useEffect(() => {
    if (urlCkpt != null || urlCompare != null) return;
    const stored = readLS();
    if (stored.ckpt || stored.compare) {
      url.set({ ckpt: stored.ckpt, compare: stored.compare });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ckpt = urlCkpt;
  const compare = urlCompare;

  const setCkpt = useCallback(
    (k: string | null) => {
      url.set({ ckpt: k });
      writeLS({ ckpt: k, compare });
    },
    [url, compare],
  );
  const setCompare = useCallback(
    (k: string | null) => {
      url.set({ compare: k });
      writeLS({ ckpt, compare: k });
    },
    [url, ckpt],
  );

  return useMemo(
    () => ({
      ckpt,
      compare,
      hasCompare: Boolean(compare),
      activeKeys: compare ? [ckpt, compare].filter(Boolean) as string[] : ckpt ? [ckpt] : [],
      setCkpt,
      setCompare,
    }),
    [ckpt, compare, setCkpt, setCompare],
  );
}
