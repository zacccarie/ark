import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./*.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ARK tokens (mirror ark/ui/design_tokens.py)
        bg: {
          deep: "#09090B",
          panel: "#18181B",
          raised: "#27272A",
          overlay: "#3F3F46",
          input: "#0E0E11",
        },
        fg: {
          DEFAULT: "#FAFAFA",
          secondary: "#E4E4E7",
          tertiary: "#B8B8C0",
          disabled: "#71717A",
        },
        border: {
          subtle: "#27272A",
          DEFAULT: "#3F3F46",
          strong: "#52525B",
        },
        primary: {
          DEFAULT: "#00FF85",
          hover: "#00E676",
          press: "#00CC6E",
          dim: "#052E1A",
        },
        accent: {
          blue: "#3B82F6",
          red: "#E11D2A",
        },
        success: "#00FF85",
        warn: "#FBBF24",
        error: "#F87171",
        info: "#38BDF8",
      },
      borderRadius: {
        sm: "6px",
        md: "10px",
        lg: "18px",
        xl: "24px",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Menlo', 'JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.4)",
        md: "0 4px 12px rgba(0,0,0,0.5)",
        lg: "0 12px 32px rgba(0,0,0,0.6)",
        glow: "0 0 0 2px rgba(0,255,133,0.25)",
      },
      transitionDuration: {
        fast: "150ms",
        normal: "240ms",
        slow: "360ms",
      },
    },
  },
  plugins: [],
} satisfies Config;
