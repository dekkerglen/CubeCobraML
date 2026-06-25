/** Single-select cube search input.
 *
 * Accepts either a fuzzy name/owner query or a pasted cubecobra.com URL/slug.
 * On URL paste, calls `/cubes/resolve` to translate slug → UUID and navigates
 * to the cube detail page.
 */
import { Loader2, Search } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { apiResolveCube } from "@/lib/api";
import { useCubeSearch } from "@/hooks/useCubeSearch";
import { cn } from "@/lib/cn";


function looksLikeUrlOrSlug(q: string): boolean {
  const s = q.trim();
  if (!s) return false;
  if (s.includes("//") || s.includes("/")) return true;
  if (s.startsWith("cubecobra.com")) return true;
  if (/^[A-Za-z0-9_-]{8,}$/.test(s) && !/\s/.test(s)) return true;
  return false;
}


export function CubeSearchPicker({
  placeholder = "Search cubes by name, or paste a cubecobra.com URL…",
  splitFilter = "all",
  onPickOverride,
}: {
  placeholder?: string;
  splitFilter?: "all" | "val";
  /** When provided, picking a cube calls this instead of navigating to /cube/:uuid. */
  onPickOverride?: (uuid: string) => void;
}) {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [resolving, setResolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { items, loading } = useCubeSearch(q, splitFilter, 10);

  const pick = (uuid: string) => {
    if (onPickOverride) onPickOverride(uuid);
    else navigate(`/cube/${uuid}`);
    setQ("");
  };

  async function tryResolve() {
    setError(null);
    setResolving(true);
    try {
      const r = await apiResolveCube(q.trim());
      pick(r.cube_uuid);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setResolving(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      if (looksLikeUrlOrSlug(q)) {
        void tryResolve();
      } else if (items[0]) {
        pick(items[0].cube_uuid);
      }
    }
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-2 border border-border rounded-md bg-bg-0 px-3 py-2 focus-within:border-accent">
        {resolving || loading ? (
          <Loader2 className="size-4 text-fg-3 animate-spin" />
        ) : (
          <Search className="size-4 text-fg-3" />
        )}
        <input
          value={q}
          onChange={(e) => { setQ(e.target.value); setError(null); }}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          className="flex-1 bg-transparent outline-none text-fg-0 placeholder:text-fg-3 text-sm"
        />
      </div>
      {error && (
        <div className="mt-1 text-xs text-bad">{error}</div>
      )}
      {q.trim() && items.length > 0 && (
        <ul className="absolute z-10 mt-1 w-full max-h-80 overflow-y-auto rounded-md border border-border bg-bg-0 shadow-lg scrollbar-thin">
          {items.map((c) => (
            <li key={c.cube_uuid}>
              <button
                type="button"
                onClick={() => pick(c.cube_uuid)}
                className={cn(
                  "w-full text-left px-3 py-2 flex items-center gap-3 hover:bg-bg-2",
                  "border-b border-border-subtle last:border-0",
                )}
              >
                {c.image_uri && (
                  <img src={c.image_uri} alt="" className="h-8 w-12 object-cover rounded shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-fg-0 truncate text-sm font-medium">{c.name || "(unnamed)"}</div>
                  <div className="text-fg-3 text-xs truncate">
                    {c.owner || "—"} · {c.card_count} cards
                    {c.has_val_drafts && <span className="ml-2 text-good">val</span>}
                  </div>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
