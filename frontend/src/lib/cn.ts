import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware class composer. Resolves conflicts so the LAST class wins. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
