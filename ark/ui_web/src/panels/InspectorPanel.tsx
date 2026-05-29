import { useEffect, useState } from "react";
import { getBridge } from "@/lib/bridge";
import { Info, Play, Download } from "lucide-react";
import { cn } from "@/lib/utils";

type ParamSpec = {
  name: string;
  kind: string;
  default: any;
  options?: string[];
  min?: number;
  max?: number;
};

type NodeData = {
  id: string;
  type: string;
  title: string;
  category: string;
  description: string;
  role: string;
  function: string;
  params: ParamSpec[];
  values: Record<string, any>;
  last_text?: string;
  last_value?: string;
  last_audio_path?: string;
  preview_image?: string;
};

export function InspectorPanel() {
  const [node, setNode] = useState<NodeData | null>(null);
  const [bridge, setBridge] = useState<any>(null);

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      setBridge(b);
      if (b.selection_changed && b.selection_changed.connect) {
        b.selection_changed.connect((nd: NodeData | null) => setNode(nd));
      }
      const cur = await b.get_selected_node?.();
      if (cur) setNode(cur);
    })();
  }, []);

  if (!node) {
    return (
      <div className="flex flex-col h-full bg-bg-panel">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
          <Info className="w-4 h-4 text-primary" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
            Inspector
          </h2>
        </div>
        <div className="flex-1 flex items-center justify-center text-fg-disabled text-sm">
          Aucune sélection
        </div>
      </div>
    );
  }

  const setParam = (name: string, value: any) => {
    bridge?.set_node_param?.(node.id, name, value);
    setNode({ ...node, values: { ...node.values, [name]: value } });
  };

  return (
    <div className="flex flex-col h-full bg-bg-panel">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <Info className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          Inspector
        </h2>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <div>
          <h1 className="text-base font-bold text-fg">{node.title}</h1>
          <p className="text-[10px] text-fg-tertiary uppercase tracking-widest mt-0.5">
            {node.category} · id {node.id.slice(0, 12)}
          </p>
        </div>

        {node.role && (
          <p className="text-sm font-semibold text-fg leading-snug">
            {node.role}
          </p>
        )}
        {node.function && (
          <p className="text-xs text-fg-secondary leading-relaxed">
            {node.function}
          </p>
        )}

        {node.preview_image && (
          <div className="ark-card p-1 flex items-center justify-center bg-bg-deep">
            <img src={node.preview_image} alt="" className="max-h-60 object-contain" />
          </div>
        )}

        {node.last_audio_path && (
          <div className="ark-card p-3 space-y-2 border-primary/50">
            <p className="text-[10px] uppercase tracking-widest text-primary font-bold">
              Audio Player
            </p>
            <p className="font-mono text-[10px] text-fg-tertiary break-all">
              {node.last_audio_path}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => bridge?.play_audio?.(node.last_audio_path)}
                className="flex-1 px-3 py-2 bg-primary text-bg-deep rounded-sm font-semibold text-sm flex items-center justify-center gap-1.5 hover:bg-primary-hover"
              >
                <Play className="w-3.5 h-3.5" /> Play
              </button>
              <button
                onClick={() => bridge?.download_audio?.(node.last_audio_path)}
                className="px-3 py-2 ark-btn flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5" /> Save
              </button>
            </div>
          </div>
        )}

        {(node.last_text || node.last_value) && (
          <div className="ark-card p-3 border-primary/50">
            <p className="text-[10px] uppercase tracking-widest text-primary font-bold mb-2">
              Live Output
            </p>
            <pre className="font-mono text-xs text-fg whitespace-pre-wrap">
              {node.last_text || node.last_value}
            </pre>
          </div>
        )}

        <div className="space-y-2 pt-2 border-t border-border-subtle">
          {node.params.map((p) => (
            <ParamRow
              key={p.name}
              spec={p}
              value={node.values[p.name] ?? p.default}
              onChange={(v) => setParam(p.name, v)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ParamRow({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: any;
  onChange: (v: any) => void;
}) {
  if (spec.kind === "bool") {
    return (
      <label className="flex items-center justify-between gap-3 cursor-pointer">
        <span className="text-xs text-fg-secondary">{spec.name}</span>
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
          className="accent-primary w-4 h-4"
        />
      </label>
    );
  }
  if (spec.kind === "choice") {
    return (
      <div>
        <label className="text-xs text-fg-tertiary block mb-1">{spec.name}</label>
        <select
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          className="ark-input"
        >
          {(spec.options || []).map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </div>
    );
  }
  if (spec.kind === "multiline") {
    return (
      <div>
        <label className="text-xs text-fg-tertiary block mb-1">{spec.name}</label>
        <textarea
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          rows={4}
          className={cn("ark-input font-mono", "resize-y")}
        />
      </div>
    );
  }
  return (
    <div>
      <label className="text-xs text-fg-tertiary block mb-1">{spec.name}</label>
      <input
        type={spec.kind === "int" || spec.kind === "float" ? "number" : "text"}
        value={value ?? ""}
        step={spec.kind === "float" ? "0.001" : undefined}
        min={spec.min}
        max={spec.max}
        onChange={(e) => {
          const v =
            spec.kind === "int"
              ? parseInt(e.target.value)
              : spec.kind === "float"
              ? parseFloat(e.target.value)
              : e.target.value;
          onChange(v);
        }}
        className="ark-input"
      />
    </div>
  );
}
