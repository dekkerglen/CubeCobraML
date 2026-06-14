/** Run-level discovery hooks for the runs-first model selector. */
import { useQuery } from "@tanstack/react-query";

import { apiRunCkpts, apiRuns } from "@/lib/api";


export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: apiRuns,
  });
}

export function useRunCkpts(run_id: string | null) {
  return useQuery({
    queryKey: ["runs", run_id, "ckpts"],
    queryFn: () => apiRunCkpts(run_id!),
    enabled: !!run_id,
  });
}
