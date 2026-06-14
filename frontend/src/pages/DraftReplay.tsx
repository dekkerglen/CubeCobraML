/** Draft replay — step through a real val draft pick-by-pick.
 *
 * For each pick: show the pack with ring markers for human/model picks,
 * the drafter's pool so far, the cube context, and the model's full ranked
 * predictions for that step. Optionally compares two checkpoints via the
 * standard ComparePanels.
 */
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, FlaskConical } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { CardGrid, type CardGridItem } from "@/components/CardGrid";
import { ComparePanels } from "@/components/ComparePanels";
import { PackStepper } from "@/components/PackStepper";
import { ProgressBar } from "@/components/ProgressBar";
import { RankedPickList, type RankedPick } from "@/components/RankedPickList";
import { useJob, useStartJob } from "@/hooks/useJob";
import { apiGetDraft } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { DraftReplay, DraftReplayStep } from "@/lib/types";


export function DraftReplayPage() {
  const { draftId = "" } = useParams<{ draftId: string }>();
  const [params, setParams] = useSearchParams();
  const step = Number(params.get("step") ?? 0) || 0;
  const setStep = (next: number) => {
    const p = new URLSearchParams(params);
    if (next <= 0) p.delete("step");
    else p.set("step", String(next));
    setParams(p, { replace: true });
  };

  // Base draft (pack/pool/human_pick per step). Loaded first; replay (with
  // model predictions per ckpt) is loaded lazily for active checkpoints.
  const { data: draft, isLoading, error } = useQuery({
    queryKey: ["draft", draftId],
    queryFn: () => apiGetDraft(draftId),
    enabled: !!draftId,
  });

  const total = draft?.steps.length ?? 0;
  const safeStep = Math.min(Math.max(0, step), Math.max(0, total - 1));

  return (
    <div className="container py-8 space-y-6">
      <div>
        <Link
          to={draft?.cube_uuid ? `/cube/${draft.cube_uuid}` : "/cubes"}
          className="inline-flex items-center gap-1 text-sm text-fg-2 hover:text-fg-0"
        >
          <ChevronLeft className="size-4" />
          {draft?.cube_uuid ? "Back to cube" : "Back to cubes"}
        </Link>
      </div>

      {isLoading && <div className="text-fg-2">Loading draft…</div>}
      {error && <div className="text-bad">Could not load draft: {(error as Error).message}</div>}

      {draft && (
        <>
          <header className="border-b border-border pb-4">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-2xl font-bold text-fg-0">
                {draft.cube_uuid ? (
                  <Link to={`/cube/${draft.cube_uuid}`} className="hover:underline text-accent">
                    Cube {draft.cube_uuid.slice(0, 8)}…
                  </Link>
                ) : "Draft Replay"}
              </h1>
              {draft.synthetic && (
                <Badge tone="warn">synthetic id</Badge>
              )}
              {draft.suspect && (
                <Badge tone="bad">suspect</Badge>
              )}
              <span className="text-fg-3 text-xs font-mono ml-2">{draft.draft_id}</span>
            </div>
            <p className="text-fg-2 text-sm mt-1">
              {total} picks · {draft.owner ? <>drafter <span className="font-medium text-fg-1">{draft.owner.slice(0, 12)}…</span></> : "(drafter unknown)"}
            </p>
          </header>

          <PackStepper step={safeStep} total={total} onChange={setStep} />

          <ComparePanels render={(ckpt) => (
            <ReplayPanel draftId={draftId} ckpt={ckpt} step={safeStep} />
          )} />
        </>
      )}
    </div>
  );
}


function Badge({ tone, children }: { tone: "warn" | "bad"; children: React.ReactNode }) {
  return (
    <span className={cn(
      "text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full",
      tone === "warn" ? "bg-warn text-white" : "bg-bad text-white",
    )}>{children}</span>
  );
}


function ReplayPanel({ draftId, ckpt, step }: { draftId: string; ckpt: string; step: number }) {
  const start = useStartJob("draft_replay");
  const [jobId, setJobId] = useState<string | null>(null);
  const lastKey = useRef<string | null>(null);
  const key = `${draftId}::${ckpt}`;

  useEffect(() => {
    if (!draftId || !ckpt || lastKey.current === key) return;
    lastKey.current = key;
    setJobId(null);
    start.mutate(
      { draft_id: draftId, ckpt },
      { onSuccess: ({ id }) => setJobId(id) },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const job = useJob(jobId);
  // Both partial (mid-run) and result (final) carry the same DraftReplay shape.
  const replay = (job.data?.result ?? job.data?.partial) as DraftReplay | null;
  const cur = replay?.steps[step] ?? null;

  return (
    <div className="space-y-4">
      {job.data && job.data.status !== "done" && (
        <ProgressBar job={job.data} label={`Replay (${ckpt})`} />
      )}
      {!cur && (
        <div className="text-fg-2 text-sm py-12 text-center">
          Waiting on step {step + 1}…
        </div>
      )}
      {cur && <StepView step={cur} />}
    </div>
  );
}


function playgroundHref(step: DraftReplayStep): string {
  const qs = new URLSearchParams();
  qs.set("pack", step.pack.join(","));
  if (step.pool.length) qs.set("pool", step.pool.join(","));
  qs.set("run", "1");
  return `/playground?${qs.toString()}`;
}


function StepView({ step }: { step: DraftReplayStep }) {
  const packItems: CardGridItem[] = useMemo(() => {
    const seen = new Set<number>();
    return step.pack
      .filter((idx) => { if (seen.has(idx)) return false; seen.add(idx); return true; })
      .map((idx) => {
        const isHuman = idx === step.human_pick;
        const isModel = idx === step.top1;
        const ring = isHuman && isModel ? "both" : isHuman ? "human" : isModel ? "model" : null;
        const badge = isHuman ? { tone: "human" as const, text: "human" }
                  : isModel ? { tone: "model" as const, text: "model" }
                  : null;
        return { idx, ring, badge };
      });
  }, [step]);

  const ranked: RankedPick[] = useMemo(() => (
    step.ranked.map(([idx, p], i) => ({
      idx, p, rank: i + 1,
      isHuman: idx === step.human_pick,
      isTop: i === 0,
    }))
  ), [step]);

  const poolItems: CardGridItem[] = step.pool.map((idx) => ({ idx }));

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-border bg-bg-1 p-4">
        <div className="text-sm font-semibold text-fg-0 mb-3 flex items-center gap-2 flex-wrap">
          Pack ({step.pack.length})
          <span className={cn(
            "text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full",
            step.agrees_with_human ? "bg-good text-white" : "bg-warn text-white",
          )}>
            {step.agrees_with_human ? "✓ model agrees" : "✗ model disagrees"}
          </span>
          <span className="text-fg-3 text-xs nums ml-auto">
            top1 {(step.top1_p * 100).toFixed(1)}%
          </span>
          <Link
            to={playgroundHref(step)}
            className="inline-flex items-center gap-1 text-xs text-fg-2 hover:text-accent border border-border-subtle rounded px-2 py-1"
            title="Open this pick in the Playground"
          >
            <FlaskConical className="size-3" /> See in playground
          </Link>
        </div>
        <CardGrid items={packItems} size="sm" />
      </div>

      <div className="rounded-lg border border-border bg-bg-1 p-4">
        <div className="text-sm font-semibold text-fg-0 mb-3">Ranked predictions</div>
        <RankedPickList rows={ranked} />
      </div>

      {poolItems.length > 0 && (
        <div className="rounded-lg border border-border bg-bg-1 p-4">
          <div className="text-sm font-semibold text-fg-0 mb-3">Pool so far ({poolItems.length})</div>
          <CardGrid items={poolItems} size="xs" />
        </div>
      )}
    </div>
  );
}
