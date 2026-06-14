/** Checkpoint listing hook. */
import { useQuery } from "@tanstack/react-query";
import { apiCheckpoints, apiCheckpointSnapshot } from "@/lib/api";

export function useCheckpoints() {
  return useQuery({
    queryKey: ["checkpoints"],
    queryFn: apiCheckpoints,
  });
}

export function useCheckpointSnapshot(key: string | null | undefined) {
  return useQuery({
    queryKey: ["checkpoints", "snapshot", key],
    queryFn: () => apiCheckpointSnapshot(key!),
    enabled: !!key,
  });
}
