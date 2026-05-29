import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  base: "./",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      input: {
        palette: path.resolve(__dirname, "palette.html"),
        inspector: path.resolve(__dirname, "inspector.html"),
        terminal: path.resolve(__dirname, "terminal.html"),
        toolbar: path.resolve(__dirname, "toolbar.html"),
      },
    },
  },
});
