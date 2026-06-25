/** URL search-param state helpers — single source of truth for ckpt selection,
 * comparison mode, and the active card drawer. Pages read these via the
 * matching zustand stores (see `hooks/useModelSelection.ts`, `useCardDrawer.ts`).
 */
import { useSearchParams } from "react-router-dom";

export type UrlKey = "ckpt" | "compare" | "card" | "tab" | "cube_uuids";

export function useUrlState() {
  const [params, setParams] = useSearchParams();
  return {
    get(k: UrlKey): string | null {
      return params.get(k);
    },
    set(patch: Partial<Record<UrlKey, string | null>>) {
      const next = new URLSearchParams(params);
      Object.entries(patch).forEach(([k, v]) => {
        if (v == null || v === "") next.delete(k);
        else next.set(k, v);
      });
      setParams(next, { replace: true });
    },
  };
}
