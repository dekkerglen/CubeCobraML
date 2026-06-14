/** Multi-cube filter chip-bar — sits under a page header.
 *
 * Each chip represents one cube_uuid (resolved to its human name via the cube
 * directory). Backed by URL state so the filter survives reloads. Pages that
 * accept the filter call `selectedUuids` from `useCubeFilter()` and pass it
 * to their data hook.
 *
 * Right side hosts a CubeSearchPicker for adding more cubes.
 */
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { CubeSearchPicker } from "@/components/CubeSearchPicker";
import { apiCubeByUuid } from "@/lib/api";
import { useUrlState } from "@/lib/url";
import { cn } from "@/lib/cn";


export function useCubeFilter() {
  const url = useUrlState();
  const raw = url.get("cube_uuids") || "";
  const selectedUuids = useMemo(
    () => raw.split(",").map((s) => s.trim()).filter(Boolean),
    [raw],
  );
  const setSelectedUuids = useCallback(
    (next: string[]) => url.set({ cube_uuids: next.length ? next.join(",") : null }),
    [url],
  );
  return { selectedUuids, setSelectedUuids };
}


export function CubeFilterBar({ className }: { className?: string }) {
  const { selectedUuids, setSelectedUuids } = useCubeFilter();
  // Click intercept on the picker isn't needed — the picker navigates to
  // /cube/:uuid on selection. We override that by mounting a lightweight
  // local picker variant inside the bar.
  return (
    <div className={cn(
      "rounded-md border border-border bg-bg-1 p-3 flex items-start gap-3 flex-wrap",
      className,
    )}>
      <div className="text-xs uppercase tracking-wider font-bold text-fg-3 mt-2 shrink-0">
        Filter by cube
      </div>
      <div className="flex flex-wrap gap-1.5 flex-1 min-w-[200px]">
        {selectedUuids.length === 0 && (
          <span className="text-fg-3 text-sm mt-1.5">No filter — showing all val data.</span>
        )}
        {selectedUuids.map((uuid) => (
          <CubeChip
            key={uuid}
            uuid={uuid}
            onRemove={() => setSelectedUuids(selectedUuids.filter((u) => u !== uuid))}
          />
        ))}
      </div>
      <div className="w-64 shrink-0">
        <FilterAddPicker
          onAdd={(uuid) => {
            if (!selectedUuids.includes(uuid)) {
              setSelectedUuids([...selectedUuids, uuid]);
            }
          }}
        />
      </div>
      {selectedUuids.length > 0 && (
        <button
          type="button"
          onClick={() => setSelectedUuids([])}
          className="text-xs text-fg-3 hover:text-fg-1 px-2 py-1 self-center"
        >
          clear
        </button>
      )}
    </div>
  );
}


function CubeChip({ uuid, onRemove }: { uuid: string; onRemove: () => void }) {
  const { data } = useQuery({
    queryKey: ["cube-by-uuid", uuid],
    queryFn: () => apiCubeByUuid(uuid),
    staleTime: Infinity,
  });
  const label = data?.name || uuid.slice(0, 8);
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full bg-accent-subtle text-accent border border-accent/40">
      {label}
      <button onClick={onRemove} className="hover:text-accent-hover" aria-label="remove">
        <X className="size-3" />
      </button>
    </span>
  );
}


// Local picker that calls back instead of navigating — so the filter bar can
// accumulate chips. Reuses CubeSearchPicker's search hook via a wrapper.
function FilterAddPicker({ onAdd }: { onAdd: (uuid: string) => void }) {
  const [open, setOpen] = useState(false);
  return open ? (
    <InlinePicker onAdd={(uuid) => { onAdd(uuid); setOpen(false); }} onCancel={() => setOpen(false)} />
  ) : (
    <button
      type="button"
      onClick={() => setOpen(true)}
      className="w-full px-3 py-2 border border-dashed border-border rounded-md text-sm text-fg-2 hover:border-accent hover:text-accent text-left"
    >
      + Add cube
    </button>
  );
}


function InlinePicker({ onAdd, onCancel }: { onAdd: (uuid: string) => void; onCancel: () => void }) {
  // Re-uses CubeSearchPicker which navigates. To get callback behavior, we
  // listen on history changes — but simpler: import the underlying useCubeSearch.
  return (
    <div onKeyDown={(e) => e.key === "Escape" && onCancel()} className="space-y-1">
      <CubeSearchPickerCallback onPick={onAdd} />
      <button onClick={onCancel} className="text-xs text-fg-3 hover:text-fg-1">cancel</button>
    </div>
  );
}


// Variant of CubeSearchPicker that calls onPick instead of navigating.
function CubeSearchPickerCallback({ onPick }: { onPick: (uuid: string) => void }) {
  return <CubeSearchPicker placeholder="Search a cube…" onPickOverride={onPick} />;
}
