import { useEffect, useState, useCallback } from "react";
import { getBridge } from "@/lib/bridge";
import {
  Blocks, Cpu, Activity, CheckCircle2, Rocket, RefreshCw,
  Check, Circle, Play, ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Check = { label: string; done: boolean };
type Phase = {
  key: string;
  title: string;
  icon: string;
  done: boolean;
  current: boolean;
  desc: string;
  checks: Check[];
  action: string | null;
};
type State = {
  modelType: string;
  modelTypeLabel: string;
  nNodes: number;
  phases: Phase[];
};

const ICONS: Record<string, any> = {
  blocks: Blocks, cpu: Cpu, activity: Activity,
  "check-circle": CheckCircle2, rocket: Rocket,
};

export function WorkflowPanel() {
  const [bridge, setBridge] = useState<any>(null);
  const [state, setState] = useState<State | null>(null);

  const refresh = useCallback(async () => {
    const b = bridge || (await getBridge());
    if (!bridge) setBridge(b);
    try {
      const s = JSON.parse(await b.workflow_get_state());
      setState(s);
    } catch {
      /* noop */
    }
  }, [bridge]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500); // live update
    return () => clearInterval(t);
  }, [refresh]);

  const doAction = (action: string | null) => {
    if (!bridge || !action) return;
    if (action === "compile") bridge.compile?.(1);
    if (action === "train") bridge.raise_compute?.(); // Modal panel for train
    setTimeout(refresh, 600);
  };

  if (!state) {
    return (
      <div className="flex items-center justify-center h-full text-fg-disabled text-sm">
        Chargement…
      </div>
    );
  }

  const curIdx = state.phases.findIndex((p) => p.current);

  return (
    <div className="flex flex-col h-full bg-bg-panel">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <Rocket className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          Workflow
        </h2>
        <span className="text-[10px] text-fg-disabled ml-1">
          {state.modelTypeLabel} · {state.nNodes} nodes
        </span>
        <button
          onClick={refresh}
          className="ml-auto text-fg-tertiary hover:text-fg p-1"
          title="Rafraîchir"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Timeline */}
      <div className="flex items-center px-3 py-3 border-b border-border-subtle overflow-x-auto">
        {state.phases.map((p, i) => {
          const Icon = ICONS[p.icon] || Circle;
          return (
            <div key={p.key} className="flex items-center shrink-0">
              <div className="flex flex-col items-center gap-1" style={{ width: 56 }}>
                <div
                  className={cn(
                    "w-9 h-9 rounded-full flex items-center justify-center border-2 transition-colors",
                    p.done
                      ? "bg-primary/20 border-primary text-primary"
                      : p.current
                      ? "bg-accent-blue/20 border-accent-blue text-accent-blue animate-pulse"
                      : "bg-bg-raised border-border text-fg-disabled"
                  )}
                >
                  {p.done ? <Check className="w-4 h-4" /> : <Icon className="w-4 h-4" />}
                </div>
                <span
                  className={cn(
                    "text-[8px] text-center leading-tight",
                    p.current ? "text-accent-blue font-bold" :
                    p.done ? "text-primary" : "text-fg-disabled"
                  )}
                >
                  {p.title.split("·")[1]?.trim() || p.title}
                </span>
              </div>
              {i < state.phases.length - 1 && (
                <ChevronRight
                  className={cn(
                    "w-3 h-3 shrink-0 -mt-3",
                    state.phases[i].done ? "text-primary" : "text-border-strong"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Phases detail */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {state.phases.map((p, i) => (
          <div
            key={p.key}
            className={cn(
              "ark-card p-3 transition-colors",
              p.current && "border-accent-blue/60",
              p.done && "border-primary/40"
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                className={cn(
                  "text-sm font-bold",
                  p.done ? "text-primary" :
                  p.current ? "text-accent-blue" : "text-fg-secondary"
                )}
              >
                {p.title}
              </span>
              {p.current && (
                <span className="text-[9px] uppercase tracking-wider bg-accent-blue/20 text-accent-blue px-1.5 py-0.5 rounded-sm font-bold">
                  en cours
                </span>
              )}
              {p.done && (
                <Check className="w-3.5 h-3.5 text-primary ml-auto" />
              )}
            </div>
            <p className="text-[11px] text-fg-tertiary leading-relaxed mb-2">
              {p.desc}
            </p>
            <div className="space-y-1">
              {p.checks.map((c, ci) => (
                <div key={ci} className="flex items-center gap-2">
                  {c.done ? (
                    <Check className="w-3.5 h-3.5 text-primary shrink-0" />
                  ) : (
                    <Circle className="w-3.5 h-3.5 text-fg-disabled shrink-0" />
                  )}
                  <span
                    className={cn(
                      "text-[11px]",
                      c.done ? "text-fg-secondary line-through opacity-70" :
                      "text-fg-secondary"
                    )}
                  >
                    {c.label}
                  </span>
                </div>
              ))}
            </div>
            {p.action && p.current && (
              <button
                onClick={() => doAction(p.action)}
                className="ark-btn-primary mt-2 flex items-center gap-1.5 text-xs"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                {p.action === "compile" ? "Compiler maintenant" :
                 p.action === "train" ? "Ouvrir entraînement (Modal)" :
                 "Lancer"}
              </button>
            )}
          </div>
        ))}
        <p className="text-[10px] text-fg-disabled text-center pt-2">
          Le workflow se met à jour automatiquement selon ta scène.
        </p>
      </div>
    </div>
  );
}
