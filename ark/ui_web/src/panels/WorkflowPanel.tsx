import { useEffect, useState, useCallback } from "react";
import { getBridge } from "@/lib/bridge";
import {
  Blocks, Cpu, Activity, CheckCircle2, Rocket, RefreshCw,
  Check, Circle, Play, ChevronRight, FlaskConical, Loader2,
  Gauge, BarChart3,
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

type ModelState = {
  state: "empty" | "blank" | "trained" | "validated" | "operational";
  val_acc?: number;
  n_params?: number;
  validated?: boolean;
  test_metric?: number | null;
  test_metric_name?: string;
  test_gap?: number | null;
  kind?: string;
  kindLabel?: string;
  metricName?: string;
};

type TestResult = {
  status: string;
  metric?: number | null;
  metric_name?: string;
  accuracy?: number;
  gap?: number | null;
  passed?: boolean;
  note?: string;
  error?: string;
  kindLabel?: string;
};

// Formate une métrique selon son nom
function fmtMetric(name?: string, v?: number | null): string {
  if (v == null) return "—";
  if (name === "accuracy") return `${(v * 100).toFixed(1)}%`;
  return v.toFixed(3);
}

// Bench (Phase 82)
type BenchYou = {
  name: string; n_params: number; params_str: string; param_mb: number;
  flops: number; flops_str: string; latency_ms: number | null;
  throughput: number | null; device: string; kind?: string;
  kindLabel?: string; metric?: string; value?: number | null;
  validated?: boolean;
};
type BenchRef = {
  name: string; params: number; params_str: string; metric: string;
  value: number; dataset: string; year: number; type: string;
};
type BenchData = {
  status: string; you?: BenchYou; references?: BenchRef[]; error?: string;
};

// Métrique de bench : accuracy/top1 en %, perplexity brut
function fmtBench(metric?: string, v?: number | null): string {
  if (v == null) return "—";
  if (metric === "perplexity") return v.toFixed(1);
  return `${(v * 100).toFixed(1)}%`;
}

const STATE_META: Record<
  string,
  { label: string; box: string; dot: string }
> = {
  operational: {
    label: "OPÉRATIONNEL",
    box: "bg-primary/10 border-primary/50",
    dot: "bg-primary",
  },
  validated: {
    label: "VALIDÉ · généralise",
    box: "bg-accent-blue/10 border-accent-blue/50",
    dot: "bg-accent-blue",
  },
  trained: {
    label: "Entraîné · à valider",
    box: "bg-warn/10 border-warn/50",
    dot: "bg-warn",
  },
  blank: {
    label: "Modèle vierge",
    box: "bg-bg-raised border-border",
    dot: "bg-fg-disabled",
  },
  empty: {
    label: "Scène vide",
    box: "bg-bg-raised border-border",
    dot: "bg-fg-disabled",
  },
};

export function WorkflowPanel() {
  const [bridge, setBridge] = useState<any>(null);
  const [state, setState] = useState<State | null>(null);
  const [model, setModel] = useState<ModelState | null>(null);
  const [testing, setTesting] = useState(false);
  const [testRes, setTestRes] = useState<TestResult | null>(null);
  const [bench, setBench] = useState<BenchData | null>(null);
  const [benching, setBenching] = useState(false);

  const refresh = useCallback(async () => {
    const b = bridge || (await getBridge());
    if (!bridge) setBridge(b);
    try {
      const s = JSON.parse(await b.workflow_get_state());
      setState(s);
    } catch {
      /* noop */
    }
    try {
      setModel(JSON.parse(await b.model_state()));
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

  const doTest = async () => {
    if (!bridge || testing) return;
    setTesting(true);
    try {
      const r = JSON.parse(await bridge.evaluate_model());
      setTestRes(r);
    } catch {
      /* noop */
    }
    setTesting(false);
    setTimeout(refresh, 400);
  };

  const doBench = async () => {
    if (!bridge || benching) return;
    setBenching(true);
    try {
      setBench(JSON.parse(await bridge.benchmark_model()));
    } catch {
      /* noop */
    }
    setBenching(false);
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

      {/* Model state badge — Phase 81 : vierge→entraîné→validé→opérationnel */}
      {model &&
        (() => {
          const sm = STATE_META[model.state] || STATE_META.blank;
          const canTest =
            model.state === "trained" ||
            model.state === "validated" ||
            model.state === "operational";
          return (
            <div className="px-3 py-2 border-b border-border-subtle">
              <div
                className={cn(
                  "flex items-center gap-2 px-2.5 py-1.5 rounded-md border",
                  sm.box
                )}
              >
                <span className={cn("w-2.5 h-2.5 rounded-full", sm.dot)} />
                <span className="text-xs font-bold text-fg">{sm.label}</span>
                {model.kindLabel && (
                  <span className="text-[9px] uppercase tracking-wider text-fg-tertiary border border-border-subtle rounded-sm px-1 py-0.5">
                    {model.kindLabel}
                  </span>
                )}
                {model.state === "validated" && model.test_metric != null && (
                  <span className="text-[11px] text-accent-blue font-mono ml-1">
                    {model.test_metric_name || "test"}{" "}
                    {fmtMetric(model.test_metric_name, model.test_metric)}
                  </span>
                )}
                {model.state !== "validated" &&
                  model.val_acc != null &&
                  model.val_acc > 0 && (
                    <span className="text-[11px] text-fg-secondary font-mono ml-1">
                      val {(model.val_acc * 100).toFixed(1)}%
                    </span>
                  )}
                <div className="flex-1" />
                {model.state !== "empty" && (
                  <button
                    onClick={() => {
                      bridge?.train_and_keep?.();
                      setTimeout(refresh, 800);
                    }}
                    className="text-[10px] px-2 py-1 rounded-sm border border-border text-fg-secondary hover:bg-bg-raised flex items-center gap-1"
                    title="Entraîne puis persiste les poids"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    {model.state === "blank" ? "Entraîner" : "Ré-entraîner"}
                  </button>
                )}
                {canTest && (
                  <button
                    onClick={doTest}
                    disabled={testing}
                    className="ark-btn-primary text-[10px] px-2 py-1 flex items-center gap-1 disabled:opacity-50"
                    title="Test sur données held-out (jamais vues) → valide la généralisation"
                  >
                    {testing ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <FlaskConical className="w-3 h-3" />
                    )}
                    Tester
                  </button>
                )}
              </div>

              {/* Ligne contextuelle selon l'état */}
              {model.state === "operational" && (
                <p className="text-[10px] text-primary mt-1 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> En production · poids
                  rechargés à chaque compile.
                </p>
              )}
              {model.state === "validated" && (
                <p className="text-[10px] text-accent-blue mt-1 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> Généralise sur held-out
                  {model.test_gap != null &&
                    ` · gap train−test ${(model.test_gap * 100).toFixed(1)}pt`}
                  .
                </p>
              )}
              {model.state === "trained" && (
                <p className="text-[10px] text-warn mt-1">
                  Entraîné mais pas prouvé. Clique <b>Tester</b> (données jamais
                  vues) pour valider la généralisation.
                </p>
              )}
              {model.state === "blank" && (
                <p className="text-[10px] text-fg-disabled mt-1">
                  Poids aléatoires. Entraîne pour des prédictions réelles.
                </p>
              )}

              {/* Résultat de test détaillé */}
              {testRes && (
                <div className="mt-1.5 text-[10px] rounded-sm border border-border-subtle bg-bg-raised px-2 py-1">
                  {testRes.status === "ok" ? (
                    <span
                      className={testRes.passed ? "text-primary" : "text-warn"}
                    >
                      {testRes.passed ? "✓ validé" : "✗ sous le seuil"} ·{" "}
                      {testRes.metric_name}{" "}
                      {fmtMetric(testRes.metric_name, testRes.metric)}
                      {testRes.gap != null &&
                        ` · gap ${(testRes.gap * 100).toFixed(1)}pt`}
                    </span>
                  ) : testRes.status === "needs_reference_set" ? (
                    <span className="text-fg-tertiary">
                      {testRes.note ||
                        "Harnais d'évaluation dédié requis pour ce type."}
                    </span>
                  ) : testRes.status === "not_trained" ? (
                    <span className="text-fg-tertiary">
                      Modèle vierge — entraîne d'abord.
                    </span>
                  ) : (
                    <span className="text-warn">
                      {testRes.error || testRes.status}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })()}

      {/* Benchmark — Phase 82 : params/FLOPs/latence + comparaison */}
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="flex items-center gap-2 mb-1.5">
          <Gauge className="w-3.5 h-3.5 text-fg-tertiary" />
          <span className="text-[10px] uppercase tracking-wider font-semibold text-fg-secondary">
            Benchmark
          </span>
          <div className="flex-1" />
          <button
            onClick={doBench}
            disabled={benching}
            className="text-[10px] px-2 py-1 rounded-sm border border-border text-fg-secondary hover:bg-bg-raised flex items-center gap-1 disabled:opacity-50"
            title="Mesure params, FLOPs, latence sur le device courant"
          >
            {benching ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <BarChart3 className="w-3 h-3" />
            )}
            Mesurer
          </button>
        </div>

        {bench?.status === "error" && (
          <p className="text-[10px] text-warn">{bench.error}</p>
        )}

        {bench?.status === "ok" && bench.you && (
          <>
            {/* Métriques brutes */}
            <div className="grid grid-cols-4 gap-1 mb-2">
              {[
                ["params", bench.you.params_str],
                ["FLOPs", bench.you.flops_str],
                ["latence", bench.you.latency_ms != null ? `${bench.you.latency_ms}ms` : "—"],
                ["débit", bench.you.throughput != null ? `${bench.you.throughput}/s` : "—"],
              ].map(([k, v]) => (
                <div key={k} className="rounded-sm bg-bg-raised border border-border-subtle px-1.5 py-1">
                  <div className="text-[8px] uppercase tracking-wider text-fg-disabled">{k}</div>
                  <div className="text-[11px] font-mono text-fg font-bold truncate">{v}</div>
                </div>
              ))}
            </div>
            <div className="text-[9px] text-fg-disabled mb-1.5">
              {bench.you.param_mb} MB · {bench.you.device}
            </div>

            {/* Table de comparaison (Pareto params↔précision) */}
            {(() => {
              const rows = [
                ...(bench.references || []).map((r) => ({
                  name: r.name, params: r.params, params_str: r.params_str,
                  metric: r.metric, value: r.value as number | null,
                  type: r.type, you: false,
                })),
                {
                  name: bench.you!.name, params: bench.you!.n_params,
                  params_str: bench.you!.params_str,
                  metric: bench.you!.metric || "accuracy",
                  value: bench.you!.value ?? null,
                  type: bench.you!.kindLabel || "—", you: true,
                },
              ].sort((a, b) => a.params - b.params);
              return (
                <div className="rounded-sm border border-border-subtle overflow-hidden">
                  <div className="grid grid-cols-[1fr_auto_auto] gap-2 px-2 py-1 bg-bg-raised text-[8px] uppercase tracking-wider text-fg-disabled">
                    <span>modèle</span>
                    <span className="text-right">params</span>
                    <span className="text-right">score</span>
                  </div>
                  {rows.map((r, i) => (
                    <div
                      key={i}
                      className={cn(
                        "grid grid-cols-[1fr_auto_auto] gap-2 px-2 py-1 text-[10px] items-center border-t border-border-subtle",
                        r.you ? "bg-primary/10" : ""
                      )}
                    >
                      <span className={cn("truncate", r.you ? "text-primary font-bold" : "text-fg-secondary")}>
                        {r.you && "▸ "}{r.name}
                        <span className="text-fg-disabled ml-1">· {r.type}</span>
                      </span>
                      <span className="text-right font-mono text-fg-tertiary">{r.params_str}</span>
                      <span className={cn("text-right font-mono", r.you ? "text-primary" : "text-fg-tertiary")}>
                        {fmtBench(r.metric, r.value)}
                      </span>
                    </div>
                  ))}
                </div>
              );
            })()}
            <p className="text-[9px] text-fg-disabled mt-1">
              Trié par taille. Score = métrique de validité du type (acc% / perplexity).
              Ton modèle a besoin du Test (held-out) pour afficher son score réel.
            </p>
          </>
        )}
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
