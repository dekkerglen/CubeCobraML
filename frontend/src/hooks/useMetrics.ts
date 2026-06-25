/** Metrics hooks. */
import { useQuery } from "@tanstack/react-query";
import { apiMetricsRunIds, apiRunRows } from "@/lib/api";

/** Metrics-page run list — fast, csv-only scan that can include runs without
 *  a complete ckpts/ dir. The runs-first selector (`useRuns` from
 *  `@/hooks/useRuns`) is the canonical rich source for everything else. */
export function useMetricsRunIds() {
  return useQuery({
    queryKey: ["metrics", "runs"],
    queryFn: apiMetricsRunIds,
  });
}

export function useRunRows(runId: string | null | undefined) {
  return useQuery({
    queryKey: ["metrics", "runs", runId],
    queryFn: () => apiRunRows(runId!),
    enabled: !!runId,
    staleTime: 10_000,
  });
}
