/** Global card drawer state — any `<CardImage>` click opens this. */
import { create } from "zustand";

export type DrawerTab =
  | "overview"
  | "in-cubes"
  | "in-decks"
  | "in-packs"
  | "neighbors";

interface DrawerState {
  cardIdx: number | null;
  tab: DrawerTab;
  open: (idx: number, tab?: DrawerTab) => void;
  close: () => void;
  setTab: (tab: DrawerTab) => void;
}

export const useCardDrawer = create<DrawerState>((set) => ({
  cardIdx: null,
  tab: "overview",
  open: (idx, tab = "overview") => set({ cardIdx: idx, tab }),
  close: () => set({ cardIdx: null }),
  setTab: (tab) => set({ tab }),
}));
