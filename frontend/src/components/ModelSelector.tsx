/** Header model selector — runs-first, with best/latest chip toggle and
 *  ckpt drill-down.
 *
 *  Two-tier UX:
 *   - Top level: pick a RUN (not a ckpt). Each run shows best/latest chips.
 *   - "Show all ckpts" disclosure within a run reveals every snapshot.
 *
 *  Both `?ckpt=<key>` and `?compare=<key>` URL params drive selection. The
 *  picker writes through them via useModelSelection, so deep links survive.
 *
 *  Default selection: the most recent non-archived run's `latest_key` (the
 *  user's literal current model state — favors "what am I training now" over
 *  "what's historically best by metric").
 *
 *  Archived runs (under `runs/archive/`) appear in a collapsed section.
 */
import * as Popover from "@radix-ui/react-popover";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Folder,
  Rocket,
  Star,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useCheckpoints } from "@/hooks/useCheckpoints";
import { useRunCkpts, useRuns } from "@/hooks/useRuns";
import { useModelSelection } from "@/hooks/useModelSelection";
import { apiPreloadCheckpoint } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { Checkpoint, Run } from "@/lib/types";

export function ModelSelector() {
  const { data: runs = [] } = useRuns();
  // Flat ckpt list is still useful for prod + lookup-by-key for the trigger label.
  const { data: ckpts = [] } = useCheckpoints();
  const { ckpt, compare, hasCompare, setCkpt, setCompare } =
    useModelSelection();

  // Default to latest of the most recent non-archived run when nothing is set.
  useEffect(() => {
    if (ckpt || runs.length === 0) return;
    const mostRecent = runs.find((r) => !r.archived) ?? runs[0];
    if (mostRecent?.latest_key) {
      setCkpt(mostRecent.latest_key);
      return;
    }
    if (mostRecent?.best_key) setCkpt(mostRecent.best_key);
  }, [ckpt, runs, setCkpt]);

  const primary = describeSelection(ckpt, runs, ckpts);
  const secondary = describeSelection(compare, runs, ckpts);

  return (
    <div className="flex items-center gap-2 min-w-0">
      <Picker
        label="Model"
        value={primary}
        runs={runs}
        ckpts={ckpts}
        selectedKey={ckpt}
        onChange={(k) => {
          setCkpt(k);
          if (k) apiPreloadCheckpoint(k);
        }}
      />
      {hasCompare ? (
        <>
          <span className="text-fg-3 text-sm font-mono">vs</span>
          <Picker
            label="Compare"
            value={secondary}
            runs={runs}
            ckpts={ckpts}
            selectedKey={compare}
            disabledKey={ckpt}
            withRemove
            onRemove={() => setCompare(null)}
            onChange={(k) => {
              setCompare(k);
              if (k) apiPreloadCheckpoint(k);
            }}
          />
        </>
      ) : (
        <button
          type="button"
          onClick={() => {
            // Default the comparison to a different run's latest.
            const otherRun = runs.find(
              (r) => r.latest_key && r.latest_key !== ckpt,
            );
            if (otherRun?.latest_key) setCompare(otherRun.latest_key);
          }}
          className="hidden sm:block text-xs text-fg-3 hover:text-fg-1 font-medium px-2 py-1 rounded-md hover:bg-bg-3 transition-colors whitespace-nowrap"
        >
          + Compare
        </button>
      )}
    </div>
  );
}

// ----- trigger label -------------------------------------------------------

interface Selection {
  text: string;
  isBest: boolean;
}

function describeSelection(
  key: string | null,
  runs: Run[],
  ckpts: Checkpoint[],
): Selection | null {
  if (!key) return null;
  const prodCkpt = ckpts.find((c) => c.kind === "prod");
  if (prodCkpt && key === prodCkpt.key) {
    return { text: "Prod TFJS", isBest: false };
  }
  const ckpt = ckpts.find((c) => c.key === key);
  const runId = ckpt?.run_id ?? null;
  const run = runs.find((r) => r.run_id === runId);
  if (!run) {
    // Standalone or unknown — fall back to the raw ckpt label.
    return { text: ckpt?.label ?? key, isBest: !!ckpt?.is_best };
  }
  const isBest = key === run.best_key;
  const isLatest = key === run.latest_key;
  const ckptKind = isBest ? "best" : isLatest ? "latest" : "ckpt";
  const epStr = ckpt?.epoch != null ? ` ep ${ckpt.epoch}` : "";
  const metricStr = ckpt?.metric != null ? ` · ${ckpt.metric.toFixed(3)}` : "";
  return {
    text: `${run.label} · ${ckptKind}${epStr}${metricStr}`,
    isBest,
  };
}

// ----- the popover ---------------------------------------------------------

function Picker({
  label,
  value,
  runs,
  ckpts,
  selectedKey,
  disabledKey,
  onChange,
  withRemove,
  onRemove,
}: {
  label: string;
  value: Selection | null;
  runs: Run[];
  ckpts: Checkpoint[];
  selectedKey: string | null;
  /** Compare picker: forbids picking the same ckpt as primary. */
  disabledKey?: string | null;
  onChange: (k: string) => void;
  withRemove?: boolean;
  onRemove?: () => void;
}) {
  const [archiveOpen, setArchiveOpen] = useState(false);

  const prodCkpt = useMemo(() => ckpts.find((c) => c.kind === "prod"), [ckpts]);
  const nonArchived = useMemo(() => runs.filter((r) => !r.archived), [runs]);
  const archived = useMemo(() => runs.filter((r) => r.archived), [runs]);

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
            "inline-flex items-center gap-2 rounded-md border border-border",
            "bg-bg-2 hover:bg-bg-3 px-3 py-1.5 text-sm transition-colors",
            "min-w-0 max-w-[180px] sm:max-w-[380px]",
          )}
        >
          <span className="text-fg-3 text-xs uppercase tracking-wide font-semibold shrink-0">
            {label}
          </span>
          <span className="text-fg-0 font-medium truncate flex items-center gap-1.5">
            {value?.isBest && (
              <Star className="size-3 text-warn fill-warn shrink-0" />
            )}
            {value?.text ?? "Select…"}
          </span>
          {withRemove && (
            <span
              role="button"
              onClick={(e) => {
                e.stopPropagation();
                onRemove?.();
              }}
              className="ml-1 p-0.5 rounded hover:bg-bg-3"
              aria-label="Remove comparison"
            >
              <X className="size-3 text-fg-3" />
            </span>
          )}
          <ChevronDown className="size-3.5 text-fg-3 shrink-0" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          className={cn(
            "z-50 w-[min(460px,calc(100vw-16px))] rounded-lg border border-border bg-bg-1 shadow-2xl",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=open]:fade-in data-[state=closed]:fade-out",
            "max-h-[640px] overflow-y-auto",
          )}
        >
          <div className="divide-y divide-border-subtle">
            {prodCkpt && (
              <ProdCard
                selected={selectedKey === prodCkpt.key}
                disabled={prodCkpt.key === disabledKey}
                onSelect={() => onChange(prodCkpt.key)}
              />
            )}
            {nonArchived.length === 0 && !prodCkpt && (
              <div className="px-3 py-6 text-center text-fg-3 text-sm">
                No runs found.
              </div>
            )}
            {nonArchived.map((r) => (
              <RunCard
                key={r.run_id}
                run={r}
                selectedKey={selectedKey}
                disabledKey={disabledKey ?? null}
                onSelect={onChange}
              />
            ))}
            {archived.length > 0 && (
              <div>
                <button
                  type="button"
                  onClick={() => setArchiveOpen((v) => !v)}
                  className="w-full px-3 py-2 flex items-center gap-2 text-sm font-medium text-fg-2 hover:bg-bg-3"
                >
                  {archiveOpen ? (
                    <ChevronDown className="size-3.5" />
                  ) : (
                    <ChevronRight className="size-3.5" />
                  )}
                  <Folder className="size-3.5 text-fg-3" />
                  <span>archive</span>
                  <span className="text-fg-3 text-xs ml-1">
                    ({archived.length} run{archived.length === 1 ? "" : "s"})
                  </span>
                </button>
                {archiveOpen &&
                  archived.map((r) => (
                    <RunCard
                      key={r.run_id}
                      run={r}
                      selectedKey={selectedKey}
                      disabledKey={disabledKey ?? null}
                      onSelect={onChange}
                    />
                  ))}
              </div>
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

// ----- prod (pinned single) ------------------------------------------------

function ProdCard({
  selected,
  disabled,
  onSelect,
}: {
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onSelect()}
      disabled={disabled}
      className={cn(
        "w-full text-left px-3 py-3 flex items-start gap-3",
        "hover:bg-bg-3 disabled:opacity-40 disabled:cursor-not-allowed",
        selected && "bg-accent-subtle",
      )}
    >
      <Rocket className="size-4 text-accent shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <div className="text-fg-0 font-medium truncate">Prod TFJS</div>
        <div className="text-fg-3 text-xs truncate">
          deployed recommender model
        </div>
      </div>
      {selected && <Check className="size-4 text-accent shrink-0 mt-0.5" />}
    </button>
  );
}

// ----- one run (the meat of the picker) ------------------------------------

function RunCard({
  run,
  selectedKey,
  disabledKey,
  onSelect,
}: {
  run: Run;
  selectedKey: string | null;
  disabledKey: string | null;
  onSelect: (k: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);

  const bestActive = !!run.best_key && selectedKey === run.best_key;
  const latestActive = !!run.latest_key && selectedKey === run.latest_key;
  const otherCkptActive = !!(
    selectedKey &&
    selectedKey.startsWith(run.run_id + "/") &&
    !bestActive &&
    !latestActive
  );
  const anyActive = bestActive || latestActive || otherCkptActive;

  return (
    <div className={cn("px-3 py-3", anyActive && "bg-accent-subtle/40")}>
      <div className="flex items-start gap-2 mb-1">
        {anyActive ? (
          <Star className="size-3.5 text-warn fill-warn shrink-0 mt-1" />
        ) : (
          <span className="size-3.5 shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="text-fg-0 font-medium truncate">{run.label}</div>
          <div className="text-fg-3 text-xs font-mono truncate">
            {run.short}
          </div>
        </div>
      </div>
      <div className="text-fg-2 text-xs mb-2 pl-[1.4rem]">
        {run.best_metric != null && (
          <>
            {run.best_metric_name.replace(/^val_/, "")}{" "}
            <span className="font-mono">{run.best_metric.toFixed(4)}</span>
            {" · "}
          </>
        )}
        {run.n_ckpts} ckpt{run.n_ckpts === 1 ? "" : "s"}
      </div>
      <div className="flex items-center gap-1.5 pl-[1.4rem]">
        <Chip
          label="best"
          active={bestActive}
          disabled={!run.best_key || run.best_key === disabledKey}
          onClick={() => run.best_key && onSelect(run.best_key)}
        />
        <Chip
          label="latest"
          active={latestActive}
          disabled={!run.latest_key || run.latest_key === disabledKey}
          onClick={() => run.latest_key && onSelect(run.latest_key)}
        />
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="ml-auto text-xs text-fg-3 hover:text-fg-1 font-medium px-1.5 py-0.5 rounded transition-colors flex items-center gap-1"
        >
          {showAll ? (
            <ChevronDown className="size-3" />
          ) : (
            <ChevronRight className="size-3" />
          )}
          {showAll ? "hide ckpts" : "show all ckpts"}
        </button>
      </div>
      {showAll && (
        <CkptList
          runId={run.run_id}
          selectedKey={selectedKey}
          disabledKey={disabledKey}
          onSelect={onSelect}
        />
      )}
    </div>
  );
}

function Chip({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "px-2 py-0.5 rounded text-xs font-medium transition-colors",
        active
          ? "bg-accent text-bg-0"
          : "bg-bg-3 text-fg-2 hover:bg-bg-2 hover:text-fg-0",
        disabled &&
          "opacity-40 cursor-not-allowed hover:bg-bg-3 hover:text-fg-2",
      )}
    >
      {label}
    </button>
  );
}

// ----- drill-down: all ckpts in a run --------------------------------------

function CkptList({
  runId,
  selectedKey,
  disabledKey,
  onSelect,
}: {
  runId: string;
  selectedKey: string | null;
  disabledKey: string | null;
  onSelect: (k: string) => void;
}) {
  const { data: ckpts = [], isLoading } = useRunCkpts(runId);
  if (isLoading) {
    return <div className="mt-2 px-2 py-1.5 text-xs text-fg-3">loading…</div>;
  }
  if (ckpts.length === 0) {
    return <div className="mt-2 px-2 py-1.5 text-xs text-fg-3">no ckpts</div>;
  }
  return (
    <div className="mt-2 space-y-0.5 pl-[1.4rem]">
      {ckpts.map((c) => {
        const selected = c.key === selectedKey;
        const disabled = c.key === disabledKey;
        return (
          <button
            key={c.key}
            type="button"
            onClick={() => !disabled && onSelect(c.key)}
            disabled={disabled}
            className={cn(
              "w-full text-left px-2 py-1 rounded text-xs flex items-center gap-2",
              "hover:bg-bg-3 disabled:opacity-40 disabled:cursor-not-allowed",
              selected && "bg-accent-subtle text-fg-0",
              !selected && "text-fg-2",
            )}
          >
            {selected ? (
              <Check className="size-3 text-accent shrink-0" />
            ) : (
              <span className="size-3 shrink-0" />
            )}
            <span className="font-mono">
              {c.kind === "latest" ? "latest" : `ep ${c.epoch ?? "?"}`}
            </span>
            {c.metric != null && (
              <span className="text-fg-3 ml-auto font-mono">
                {c.metric.toFixed(4)}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
