/** Auto-start + poll the unified "cube_metrics" job for (ckpt, cube_uuid, source).
 *
 * Replaces the older useScorecard. The single job feeds Headline / Distribution
 * / Collapse subtabs of the Cube workspace so switching subtabs doesn't re-fire
 * inference. Source toggles (all/train/val) trigger a fresh job; everything
 * else (subtab swaps) reads off the cached result.
 */
import { useEffect, useRef, useState } from "react";

import { useJob, useStartJob } from "@/hooks/useJob";
import type { CubeMetricsOut, CubeMetricsSource } from "@/lib/api";


export function useCubeMetrics(
  ckpt: string | null,
  cubeUuid: string | null,
  source: CubeMetricsSource,
) {
  const start = useStartJob("cube_metrics");
  const [jobId, setJobId] = useState<string | null>(null);
  const lastKey = useRef<string | null>(null);
  const key = ckpt && cubeUuid ? `${ckpt}::${cubeUuid}::${source}` : null;

  useEffect(() => {
    if (!ckpt || !cubeUuid || !key) return;
    if (lastKey.current === key) return;
    lastKey.current = key;
    setJobId(null);
    start.mutate(
      { ckpt, cube_uuid: cubeUuid, source },
      { onSuccess: ({ id }) => setJobId(id) },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const job = useJob(jobId);
  // Show partial results as inference progresses so subtabs aren't blank.
  const result = (job.data?.result ?? job.data?.partial ?? null) as CubeMetricsOut | null;
  return { jobId, job: job.data, data: result };
}
