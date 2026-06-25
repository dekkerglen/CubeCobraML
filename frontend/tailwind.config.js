/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
      screens: { "2xl": "1480px" },
    },
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        // Surfaces — light CubeCobra palette
        bg: {
          0: "#FFFFFF",
          1: "#FAFAFA",
          2: "#F0F0F0",
          3: "#E9E9E9",
        },
        border: { DEFAULT: "#D4D4D4", subtle: "#E9E9E9" },
        // Text — darker = stronger
        fg: {
          0: "#212529",
          1: "#343A40",
          2: "#495057",
          3: "#808080",
        },
        // Accent — CubeCobra green for CTAs and brand
        accent: {
          DEFAULT: "#087715",
          hover: "#004B0D",
          subtle: "#E6F4E8",
        },
        good: "#087715",
        warn: "#BF6900",
        bad: "#BC1525",
        link: "#0366D6",
        // Human vs model badges — orange vs blue for chart distinction
        human: "#BF6900",
        model: "#0366D6",
      },
      borderRadius: {
        lg: "12px",
        md: "8px",
        sm: "4px",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
