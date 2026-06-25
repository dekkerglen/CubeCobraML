/** Cubes landing — the entry point for picking a cube to analyze.
 *
 * Phase R: the old Cube Recommender content moved into the Cube workspace's
 * Recommend tab. This page now hosts the hero search + a few quick-pick
 * shelves so users can grab a cube without typing.
 */
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { CubeSearchPicker } from "@/components/CubeSearchPicker";
import { apiCubesByOwner, apiSearchCubes } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { CubeDirectoryEntry } from "@/lib/types";


// User's CubeCobra owner name — what shows up on /cube/<uuid> headers.
// Configurable via Vite env var; defaults to the maintainer.
const MY_OWNER = (import.meta.env.VITE_OWNER as string | undefined) || "RyanSaxe";

const RECENT_KEY = "ccml-recent-cubes";
const RECENT_MAX = 20;


export function CubesPage() {
  const recent = useRecentCubes();

  return (
    <div className="container py-8 space-y-8">
      <header className="border-b border-border pb-6">
        <h1 className="text-3xl font-bold text-fg-0">Pick a cube</h1>
        <p className="text-fg-2 text-sm mt-2">
          Cube-first analysis: search a name, paste a <code className="font-mono text-xs">cubecobra.com</code>
          {" "}URL, or click a shelf below.
        </p>
        <div className="mt-4 max-w-xl">
          <CubeSearchPicker placeholder="Search any cube by name or paste a cubecobra URL…" />
        </div>
      </header>

      {recent.length > 0 && (
        <Shelf title="Recently viewed" cubes={recent} />
      )}
      <MyCubesShelf />
      <PopularValShelf />
    </div>
  );
}


// ----- shelves -------------------------------------------------------------


function Shelf({ title, cubes }: { title: string; cubes: CubeDirectoryEntry[] }) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-fg-0 mb-3">{title}</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {cubes.map((c) => <CubeCard key={c.cube_uuid} cube={c} />)}
      </div>
    </section>
  );
}


function MyCubesShelf() {
  // Phase V: exact-owner endpoint replaces the old search-and-filter hack.
  const { data } = useQuery({
    queryKey: ["cubes-by-owner", MY_OWNER],
    queryFn: () => apiCubesByOwner(MY_OWNER, 24),
    staleTime: Infinity,
  });
  const cubes = data?.items ?? [];
  if (cubes.length === 0) return null;
  return <Shelf title={`${MY_OWNER}'s cubes`} cubes={cubes} />;
}


function PopularValShelf() {
  // "Vintage" hits a stable set of well-trafficked vintage-style cubes that
  // are virtually guaranteed to have val drafts indexed. Cheap discovery
  // bait for someone exploring without a specific cube in mind.
  const { data } = useQuery({
    queryKey: ["cube-search", "popular-val"],
    queryFn: () => apiSearchCubes({ q: "vintage", limit: 8, split: "val" }),
    staleTime: Infinity,
  });
  const cubes = data?.items ?? [];
  if (cubes.length === 0) return null;
  return <Shelf title="Try a popular val cube" cubes={cubes} />;
}


function CubeCard({ cube }: { cube: CubeDirectoryEntry }) {
  return (
    <Link
      to={`/cube/${cube.cube_uuid}`}
      className={cn(
        "block rounded-lg border border-border bg-bg-1 hover:border-accent hover:bg-bg-2",
        "transition-colors overflow-hidden",
      )}
    >
      {cube.image_uri && (
        <img src={cube.image_uri} alt="" className="w-full h-24 object-cover" />
      )}
      <div className="p-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-fg-0 font-medium text-sm truncate">
            {cube.name || "(unnamed)"}
          </span>
          {cube.has_val_drafts && (
            <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-good text-white">val</span>
          )}
        </div>
        <div className="text-fg-3 text-xs mt-0.5 truncate">
          {cube.owner || "—"} · {cube.card_count} cards
        </div>
      </div>
    </Link>
  );
}


// ----- localStorage-backed recent cubes ------------------------------------


function useRecentCubes(): CubeDirectoryEntry[] {
  const [uuids, setUuids] = useState<string[]>([]);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      if (raw) setUuids(JSON.parse(raw));
    } catch { /* ignore */ }
  }, []);
  // Hydrate metadata for the first few uuids via a parallel search.
  const { data } = useQuery({
    queryKey: ["recent-cubes", uuids.join(",")],
    queryFn: async () => {
      if (uuids.length === 0) return [];
      // Hit cube directory by-uuid in parallel; deduped via Set up the call chain.
      const items = await Promise.all(uuids.slice(0, 8).map(async (u) => {
        try {
          const r = await fetch(`/api/cubes/by-uuid/${encodeURIComponent(u)}`).then((x) => x.json());
          // Map the by-uuid shape (extends entry with `cards`) back to entry only.
          const { cards, ...entry } = r;
          void cards;
          return entry as CubeDirectoryEntry;
        } catch { return null; }
      }));
      return items.filter(Boolean) as CubeDirectoryEntry[];
    },
    enabled: uuids.length > 0,
    staleTime: Infinity,
  });
  return data ?? [];
}


/** Append a UUID to the recently-viewed list. Call from CubeDetail mount. */
export function pushRecentCube(uuid: string): void {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const cur: string[] = raw ? JSON.parse(raw) : [];
    const next = [uuid, ...cur.filter((u) => u !== uuid)].slice(0, RECENT_MAX);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  } catch { /* ignore */ }
}
