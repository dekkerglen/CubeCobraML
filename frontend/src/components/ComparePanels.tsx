/** A/B compare-aware container.
 *
 * Render-prop pattern: pass a function that receives a single ckpt key and
 * returns the panel content. In single-model mode you get one full-width
 * column; in A/B mode you get two side-by-side panels with sticky badges.
 */
import { useCheckpoints } from "@/hooks/useCheckpoints";
import { useModelSelection } from "@/hooks/useModelSelection";
import { cn } from "@/lib/cn";


export function ComparePanels({
  render, className,
}: {
  render: (ckptKey: string) => React.ReactNode;
  className?: string;
}) {
  const { activeKeys } = useModelSelection();
  const { data: ckpts = [] } = useCheckpoints();
  const labels = activeKeys.map((k) => ckpts.find((c) => c.key === k)?.label ?? k);

  if (activeKeys.length === 0) {
    return (
      <div className="text-fg-3 text-sm py-10 text-center">
        Select a model from the header to continue.
      </div>
    );
  }
  if (activeKeys.length === 1) {
    return <div className={className}>{render(activeKeys[0])}</div>;
  }
  return (
    <div className={cn("grid grid-cols-1 lg:grid-cols-2 gap-6", className)}>
      {activeKeys.map((k, i) => (
        <section key={k} className="min-w-0">
          <div className="sticky top-0 z-10 -mt-2 mb-3 pb-2 pt-2 bg-bg-1/95 backdrop-blur">
            <span className="text-[11px] uppercase tracking-wider font-bold text-fg-3">
              Model {String.fromCharCode(65 + i)}
            </span>
            <div className="text-fg-0 font-medium truncate">{labels[i]}</div>
          </div>
          {render(k)}
        </section>
      ))}
    </div>
  );
}
