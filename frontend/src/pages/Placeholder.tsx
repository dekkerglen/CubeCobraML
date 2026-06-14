/** Generic placeholder for pages not yet built. Replaced phase by phase. */
import { Sparkles } from "lucide-react";

export function Placeholder({ title, phase }: { title: string; phase: string }) {
  return (
    <div className="container py-16 text-center space-y-3">
      <Sparkles className="size-8 text-fg-3 mx-auto" />
      <h1 className="text-2xl font-bold text-fg-0">{title}</h1>
      <p className="text-fg-2 text-sm">Coming in {phase}.</p>
    </div>
  );
}
