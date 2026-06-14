/** Recharts theme constants — kept in sync with tailwind.config.js tokens.
 *  Recharts can't read Tailwind classes, so chart styles get explicit hex. */

export const CHART_THEME = {
  grid: "#D4D4D4",
  axis: "#808080",
  tooltipBg: "#FFFFFF",
  tooltipBorder: "#D4D4D4",
  tooltipText: "#212529",
  cursor: "#808080",
} as const;

export const CHART_TOOLTIP_STYLE = {
  backgroundColor: CHART_THEME.tooltipBg,
  border: `1px solid ${CHART_THEME.tooltipBorder}`,
  borderRadius: 8,
  fontSize: 12,
  color: CHART_THEME.tooltipText,
} as const;

/** Series palette for multi-line metrics charts. Picked for legibility on a
 *  light background, with reasonable color-blind separation. */
export const SERIES_PALETTE = [
  "#0366D6", // model blue
  "#BC1525", // red
  "#BF6900", // human orange
  "#087715", // green
  "#7B2CBF", // purple
  "#0E7C66", // teal
] as const;

export const HUMAN_COLOR = "#BF6900";
export const MODEL_COLOR = "#0366D6";
