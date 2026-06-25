/** Hover-zoom card thumbnail. Clicking opens the global card drawer.
 *
 * Uses Radix HoverCard for the floating preview so it's portal-rendered
 * (escapes any parent `overflow:hidden`) and properly z-indexed above the
 * page. Loading state is a shimmering skeleton with the card aspect ratio.
 */
import * as HoverCard from "@radix-ui/react-hover-card";
import { useState } from "react";

import { useCard } from "@/hooks/useCards";
import { useCardDrawer } from "@/hooks/useCardDrawer";
import { cn } from "@/lib/cn";

export type CardRing = "human" | "model" | "both" | null;
export type CardBadge = { text: string; tone: "human" | "model" | "both" | "neutral" } | null;

export interface CardImageProps {
  idx: number;
  ring?: CardRing;
  badge?: CardBadge;
  className?: string;
  /** Show name + meta under the card. */
  caption?: React.ReactNode;
  /** Disable the click-to-drawer behavior (useful inside the drawer itself). */
  disableDrawer?: boolean;
}

const ringClass: Record<NonNullable<CardRing>, string> = {
  human: "ring-2 ring-human ring-offset-1 ring-offset-bg-1",
  model: "ring-2 ring-model ring-offset-1 ring-offset-bg-1",
  both: "ring-2 ring-good ring-offset-1 ring-offset-bg-1",
};

const badgeClass: Record<NonNullable<CardBadge>["tone"], string> = {
  human: "bg-human text-white",
  model: "bg-model text-white",
  both: "bg-good text-white",
  neutral: "bg-bg-3/90 text-fg-0",
};

export function CardImage({
  idx, ring = null, badge = null, className, caption, disableDrawer = false,
}: CardImageProps) {
  const { data: card, isLoading } = useCard(idx);
  const openDrawer = useCardDrawer((s) => s.open);
  const [loaded, setLoaded] = useState(false);

  const handleClick = (e: React.MouseEvent) => {
    if (disableDrawer) return;
    e.preventDefault();
    openDrawer(idx);
  };

  if (isLoading || !card) {
    return <CardSkeleton className={className} caption={caption} />;
  }

  return (
    <HoverCard.Root openDelay={120} closeDelay={50}>
      <HoverCard.Trigger asChild>
        <button
          type="button"
          onClick={handleClick}
          className={cn(
            "group block w-full text-left outline-none focus-visible:ring-2",
            "focus-visible:ring-accent rounded-md",
            className,
          )}
          aria-label={card.name}
        >
          <div className="relative">
            {!loaded && <SkeletonInner />}
            <img
              src={card.image}
              alt={card.name}
              loading="lazy"
              onLoad={() => setLoaded(true)}
              className={cn(
                "w-full rounded-md transition-all duration-150",
                "shadow-[0_1px_3px_rgba(0,0,0,0.08)]",
                "group-hover:shadow-[0_6px_16px_rgba(0,0,0,0.18)]",
                ring && ringClass[ring],
                !loaded && "opacity-0",
              )}
            />
            {badge && (
              <span
                className={cn(
                  "absolute bottom-1.5 left-1/2 -translate-x-1/2",
                  "px-2 py-0.5 rounded-full text-[11px] font-bold backdrop-blur-sm",
                  badgeClass[badge.tone],
                )}
              >
                {badge.text}
              </span>
            )}
          </div>
          {caption && (
            <div className="mt-1 text-[11px] text-fg-2 text-center truncate">
              {caption}
            </div>
          )}
        </button>
      </HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="right"
          align="start"
          sideOffset={8}
          className="z-50 pointer-events-none animate-in fade-in zoom-in-95 duration-150"
        >
          <img
            src={card.image_normal || card.image}
            alt=""
            className="w-[280px] rounded-lg shadow-2xl ring-1 ring-black/10"
          />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}

function CardSkeleton({ className, caption }: { className?: string; caption?: React.ReactNode }) {
  return (
    <div className={cn("block", className)}>
      <SkeletonInner />
      {caption && <div className="mt-1 text-[11px] text-fg-3 text-center truncate">{caption}</div>}
    </div>
  );
}

function SkeletonInner() {
  return (
    <div
      className="w-full aspect-[5/7] rounded-md bg-bg-3 animate-shimmer
        [background-image:linear-gradient(110deg,#E9E9E9_0%,#FAFAFA_40%,#E9E9E9_80%)]
        [background-size:200%_100%]"
    />
  );
}
