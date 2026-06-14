/** Global cmd+K palette. Live-searches cards and offers page nav. */
import { Command } from "cmdk";
import * as Dialog from "@radix-ui/react-dialog";
import { Search } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { apiSearchCards, apiSearchCubes } from "@/lib/api";
import type { Card, CubeDirectoryEntry } from "@/lib/types";
import { useCardDrawer } from "@/hooks/useCardDrawer";
import { cn } from "@/lib/cn";


const NAV_PAGES = [
  { to: "/cubes",                         label: "Cubes" },
  { to: "/metrics",                       label: "Metrics" },
  { to: "/metrics?tab=distribution",      label: "Metrics — Distribution" },
  { to: "/browse",                        label: "Browse" },
  { to: "/playground",                    label: "Playground" },
];


export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [cards, setCards] = useState<Card[]>([]);
  const [cubes, setCubes] = useState<CubeDirectoryEntry[]>([]);
  const openDrawer = useCardDrawer((s) => s.open);
  const navigate = useNavigate();

  // Toggle on cmd/ctrl+K + "/" + Esc
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "/" && !open && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Debounced card + cube search (parallel, independent failure)
  useEffect(() => {
    if (!q.trim()) {
      setCards([]);
      setCubes([]);
      return;
    }
    const handle = setTimeout(async () => {
      const [cardR, cubeR] = await Promise.allSettled([
        apiSearchCards({ q, limit: 8 }),
        apiSearchCubes({ q, limit: 5 }),
      ]);
      setCards(cardR.status === "fulfilled" ? cardR.value.items : []);
      setCubes(cubeR.status === "fulfilled" ? cubeR.value.items : []);
    }, 100);
    return () => clearTimeout(handle);
  }, [q]);

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/30 backdrop-blur-sm data-[state=open]:animate-in data-[state=open]:fade-in" />
        <Dialog.Content
          className={cn(
            "fixed left-1/2 top-[15vh] -translate-x-1/2 z-50 w-[640px] max-w-[92vw]",
            "rounded-xl border border-border bg-bg-1 shadow-2xl overflow-hidden",
            "data-[state=open]:animate-in data-[state=open]:fade-in data-[state=open]:zoom-in-95",
          )}
        >
          <Dialog.Title className="sr-only">Search</Dialog.Title>
          <Command shouldFilter={false} loop>
            <div className="flex items-center gap-2 px-4 border-b border-border-subtle">
              <Search className="size-4 text-fg-3" />
              <Command.Input
                autoFocus
                value={q}
                onValueChange={setQ}
                placeholder="Search cards or pages…"
                className="flex-1 bg-transparent py-3 outline-none text-fg-0 placeholder:text-fg-3"
              />
              <kbd className="text-[10px] text-fg-3 font-mono border border-border-subtle rounded px-1.5 py-0.5">ESC</kbd>
            </div>
            <Command.List className="max-h-[400px] overflow-y-auto p-2 scrollbar-thin">
              {cubes.length > 0 && (
                <Command.Group heading="Cubes" className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-fg-3 [&_[cmdk-group-heading]]:font-bold [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                  {cubes.map((c) => (
                    <Command.Item
                      key={`cube-${c.cube_uuid}`}
                      value={`cube-${c.cube_uuid}-${c.name}`}
                      onSelect={() => {
                        setOpen(false);
                        setQ("");
                        navigate(`/cube/${c.cube_uuid}`);
                      }}
                      className={cn(
                        "flex items-center gap-3 px-2 py-2 rounded-md cursor-pointer",
                        "aria-selected:bg-bg-3",
                      )}
                    >
                      {c.image_uri && (
                        <img src={c.image_uri} alt="" className="h-7 w-10 object-cover rounded shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-fg-0 truncate font-medium">{c.name || "(unnamed)"}</div>
                        <div className="text-fg-3 text-xs truncate">
                          {c.owner || "—"} · {c.card_count} cards
                          {c.has_val_drafts && <span className="ml-2 text-good">val</span>}
                        </div>
                      </div>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}
              {cards.length > 0 && (
                <Command.Group heading="Cards" className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-fg-3 [&_[cmdk-group-heading]]:font-bold [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                  {cards.map((c) => (
                    <Command.Item
                      key={`card-${c.idx}`}
                      value={`card-${c.idx}-${c.name}`}
                      onSelect={() => {
                        setOpen(false);
                        setQ("");
                        openDrawer(c.idx);
                      }}
                      className={cn(
                        "flex items-center gap-3 px-2 py-2 rounded-md cursor-pointer",
                        "aria-selected:bg-bg-3",
                      )}
                    >
                      {c.image && (
                        <img src={c.image} alt="" className="h-10 w-7 object-cover rounded shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-fg-0 truncate font-medium">{c.name}</div>
                        <div className="text-fg-3 text-xs truncate">{c.type} · elo {c.elo.toFixed(0)} · cmc {c.cmc}</div>
                      </div>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}
              <Command.Group heading="Pages" className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-fg-3 [&_[cmdk-group-heading]]:font-bold [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                {NAV_PAGES.map((p) => (
                  <Command.Item
                    key={`page-${p.to}`}
                    value={`page-${p.label}`}
                    onSelect={() => {
                      setOpen(false);
                      navigate(p.to);
                    }}
                    className={cn(
                      "px-2 py-2 rounded-md cursor-pointer aria-selected:bg-bg-3 text-fg-1",
                    )}
                  >
                    {p.label}
                  </Command.Item>
                ))}
              </Command.Group>
              {q && cards.length === 0 && cubes.length === 0 && (
                <Command.Empty className="text-center text-fg-3 text-sm py-6">No matches.</Command.Empty>
              )}
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
