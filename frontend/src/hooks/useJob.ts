/** Polling hook for the async job runner (Phase N).
 *
 * Usage:
 *   const { data: { id } } = useStartJob("collapse", body);
 *   const job = useJob(id);  // polls every 750ms while running; idle when done
 *
 * Two helpers (one to start, one to poll) so a component can hold onto a
 * `job_id` across re-renders without re-submitting.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiCancelJob, apiGetJob, apiStartJob } from "@/lib/api";
import type { JobOut } from "@/lib/types";


const TERMINAL = new Set<JobOut["status"]>(["done", "error", "cancelled"]);


/** Fire-and-forget submit. Returns a mutation that resolves to `{id}`. */
export function useStartJob(kind: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => apiStartJob(kind, body),
    onSuccess: ({ id }) => {
      // Pre-seed the cache so the first poll has a placeholder.
      qc.setQueryData<JobOut>(["job", id], {
        id, kind, status: "queued", progress: 0,
        eta_seconds: null, partial: null, result: null, error: null,
      });
    },
  });
}


/** Poll `/api/jobs/{id}` while the job is running. Returns the live JobOut. */
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => apiGetJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const data = q.state.data as JobOut | undefined;
      return data && !TERMINAL.has(data.status) ? 750 : false;
    },
    staleTime: 0,
  });
}


export function useCancelJob() {
  return useMutation({
    mutationFn: (jobId: string) => apiCancelJob(jobId),
  });
}
