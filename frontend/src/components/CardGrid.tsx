/** Responsive card grid using CSS grid auto-fill. */
import { CardImage, type CardRing, type CardBadge } from "./CardImage";
import { cn } from "@/lib/cn";

export type CardGridSize = "xs" | "sm" | "md" | "lg";

const sizeClass: Record<CardGridSize, string> = {
  xs: "grid-cols-[repeat(auto-fill,minmax(90px,1fr))]  gap-2",
  sm: "grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3",
  md: "grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3",
  lg: "grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4",
};

export interface CardGridItem {
  idx: number;
  ring?: CardRing;
  badge?: CardBadge;
  caption?: React.ReactNode;
}

export function CardGrid({
  items, size = "sm", className,
}: {
  items: CardGridItem[];
  size?: CardGridSize;
  className?: string;
}) {
  if (!items.length) {
    return <div className="text-fg-3 text-sm py-6 text-center">No cards.</div>;
  }
  return (
    <div className={cn("grid", sizeClass[size], className)}>
      {items.map((it) => (
        <CardImage
          key={it.idx}
          idx={it.idx}
          ring={it.ring ?? null}
          badge={it.badge ?? null}
          caption={it.caption}
        />
      ))}
    </div>
  );
}
