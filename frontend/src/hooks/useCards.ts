/** Card data hooks. */
import { useQuery } from "@tanstack/react-query";
import {
  apiCard,
  apiSearchCards,
  type CardSearchParams,
} from "@/lib/api";

export function useCard(idx: number | null) {
  return useQuery({
    queryKey: ["card", idx],
    queryFn: () => apiCard(idx!),
    enabled: idx != null,
  });
}

export function useCardSearch(params: CardSearchParams) {
  return useQuery({
    queryKey: ["cards", "search", params],
    queryFn: () => apiSearchCards(params),
    placeholderData: (prev) => prev,
  });
}
