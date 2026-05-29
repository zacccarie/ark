import { useEffect, useState } from "react";
import { getBridge } from "@/lib/bridge";
import {
  Play, Square, Boxes, PanelRight, Cog,
  Maximize2, Map, Grid3x3,
} from "lucide-react";
import { cn } from "@/lib/utils";

const ZOOM_OPTS = ["25%", "50%", "75%", "100%", "150%", "200%", "400%"];
const TOOL_OPTS = ["Select", "Pan", "Box"];

export function ToolbarPanel() {
  const [bridge, setBridge] = useState<any>(null);
  const [mode, setMode] = useState<"Compiler" | "Entraîner" | "Bench">("Compiler");
  const [batch, setBatch] = useState(1);
  const [active, setActive] = useState(0);
  const [zoom, setZoom] = useState("100%");
  const [mapOn, setMapOn] = useState(true);
  const [snapOn, setSnapOn] = useState(false);

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      setBridge(b);
    })();
  }, []);

  const run = () => {
    if (!bridge) return;
    if (mode === "Compiler") bridge.compile?.(batch);
    if (mode === "Entraîner") bridge.train?.(batch);
    if (mode === "Bench") bridge.bench?.(batch);
    setActive((n) => n + 1);
  };
  const stop = () => bridge?.stop?.();

  return (
    <div className="flex items-center gap-2 h-full px-3 bg-bg-panel border-b border-border-subtle">
      <button
        onClick={() => bridge?.open_extensions?.()}
        className="ark-btn flex items-center gap-1.5 text-xs"
      >
        <Boxes className="w-3.5 h-3.5" />
        Extensions
      </button>
      <div className="flex items-center gap-1.5 ml-2">
        <label className="text-[10px] uppercase tracking-widest text-fg-tertiary font-semibold">
          Batch
        </label>
        <input
          type="number"
          min={1}
          value={batch}
          onChange={(e) => setBatch(parseInt(e.target.value) || 1)}
          className="ark-input w-14 text-center"
        />
      </div>
      <button
        onClick={run}
        className="ark-btn-primary flex items-center gap-1.5 text-xs ml-2"
      >
        <Play className="w-3.5 h-3.5 fill-current" />
        {mode}
      </button>
      <select
        value={mode}
        onChange={(e) => setMode(e.target.value as any)}
        className="ark-input w-32"
      >
        <option>Compiler</option>
        <option>Entraîner</option>
        <option>Bench</option>
      </select>
      <button
        onClick={stop}
        className="ark-btn text-accent-red border-accent-red/40 hover:bg-accent-red hover:text-white flex items-center gap-1.5 text-xs"
      >
        <Square className="w-3.5 h-3.5 fill-current" />
        Stop
      </button>
      <div className="flex-1" />
      <span className="font-mono text-xs text-fg-secondary bg-bg-raised border border-border rounded-sm px-2 py-0.5">
        {active} actif
      </span>
      <button
        onClick={() => bridge?.toggle_inspector?.()}
        className="ark-btn flex items-center gap-1.5 text-xs"
      >
        <PanelRight className="w-3.5 h-3.5" />
        Inspector
      </button>
      <button
        onClick={() => bridge?.open_settings?.()}
        className="ark-btn flex items-center gap-1 text-xs"
      >
        <Cog className="w-3.5 h-3.5" />
      </button>

      {/* Divider */}
      <div className="w-px h-6 bg-border mx-1" />

      {/* View controls */}
      <select
        value={zoom}
        onChange={(e) => {
          setZoom(e.target.value);
          const pct = parseInt(e.target.value);
          bridge?.set_zoom?.(pct / 100);
        }}
        className="ark-input w-16"
      >
        {ZOOM_OPTS.map((z) => (
          <option key={z}>{z}</option>
        ))}
      </select>
      <button
        onClick={() => bridge?.fit_view?.()}
        className="ark-btn flex items-center gap-1 text-xs"
        title="Centrer + ajuster"
      >
        <Maximize2 className="w-3.5 h-3.5" />
        Fit
      </button>
      <button
        onClick={() => {
          const n = !mapOn;
          setMapOn(n);
          bridge?.toggle_minimap?.(n);
        }}
        className={cn(
          "ark-btn flex items-center gap-1 text-xs",
          mapOn && "bg-accent-blue/15 text-accent-blue border-accent-blue/50"
        )}
        title="Minimap"
      >
        <Map className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => {
          const n = !snapOn;
          setSnapOn(n);
          bridge?.toggle_snap?.(n);
        }}
        className={cn(
          "ark-btn flex items-center gap-1 text-xs",
          snapOn && "bg-accent-blue/15 text-accent-blue border-accent-blue/50"
        )}
        title="Snap to grid"
      >
        <Grid3x3 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
