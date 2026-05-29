import { useEffect, useRef, useState } from "react";
import { getBridge } from "@/lib/bridge";
import {
  Brain, Send, RotateCcw, Settings2, Sparkles, Check, KeyRound,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Msg = { role: "user" | "assistant"; text: string };
type Provider = { name: string; default_model: string };

export function ArchitectPanel() {
  const [bridge, setBridge] = useState<any>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [pendingArch, setPendingArch] = useState<any>(null);
  const [pendingActions, setPendingActions] = useState<any[] | null>(null);
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("claude-3-5-sonnet-latest");
  const [apiKey, setApiKey] = useState("");
  const [providers, setProviders] = useState<Provider[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [keyConfigured, setKeyConfigured] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      setBridge(b);
      const provs: Provider[] = JSON.parse(await b.architect_providers());
      setProviders(provs);
      const cfg = JSON.parse(await b.architect_get_config());
      setProvider(cfg.provider);
      setModel(cfg.model);
      setKeyConfigured(!!cfg.secrets_present?.[cfg.provider]);
      if (b.architect_chunk?.connect) {
        b.architect_chunk.connect((chunk: string) => {
          setMsgs((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            if (last && last.role === "assistant") {
              copy[copy.length - 1] = { ...last, text: last.text + chunk };
            } else {
              copy.push({ role: "assistant", text: chunk });
            }
            return copy;
          });
        });
      }
      if (b.architect_done?.connect) {
        b.architect_done.connect(() => setStreaming(false));
      }
      if (b.architect_arch_proposed?.connect) {
        b.architect_arch_proposed.connect((s: string) =>
          setPendingArch(JSON.parse(s))
        );
      }
      if (b.architect_actions_proposed?.connect) {
        b.architect_actions_proposed.connect((s: string) =>
          setPendingActions(JSON.parse(s))
        );
      }
    })();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [msgs]);

  const send = () => {
    if (!input.trim() || !bridge || streaming) return;
    setMsgs((m) => [...m, { role: "user", text: input }]);
    bridge.architect_chat?.(input);
    setStreaming(true);
    setPendingArch(null);
    setPendingActions(null);
    setInput("");
  };

  const apply = () => {
    if (!pendingArch || !bridge) return;
    bridge.architect_apply_arch?.(JSON.stringify(pendingArch));
    setPendingArch(null);
  };

  const applyActions = async () => {
    if (!pendingActions || !bridge) return;
    const resStr = await bridge.architect_apply_actions?.(
      JSON.stringify(pendingActions)
    );
    const res = resStr ? JSON.parse(resStr) : null;
    setPendingActions(null);
    if (res) {
      setMsgs((m) => [
        ...m,
        {
          role: "assistant",
          text: `✓ Appliqué : ${res.ok} ops réussies${
            res.fail ? `, ${res.fail} erreurs : ${res.errors.join("; ")}` : ""
          }`,
        },
      ]);
    }
  };

  const reset = () => {
    bridge?.architect_reset?.();
    setMsgs([]);
    setPendingArch(null);
  };

  const saveConfig = () => {
    bridge?.architect_set_provider?.(provider, model, apiKey);
    setApiKey("");
    setKeyConfigured(true);
    setShowSettings(false);
  };

  return (
    <div className="flex flex-col h-full bg-bg-panel">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <Brain className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          Architect
        </h2>
        <span className="text-[10px] text-fg-disabled ml-1">
          {provider} · {model.split("/").pop()}
        </span>
        <div className="flex-1" />
        <button
          onClick={reset}
          className="text-fg-tertiary hover:text-fg p-1"
          title="Reset conversation"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => setShowSettings((s) => !s)}
          className={cn(
            "p-1",
            showSettings ? "text-primary" : "text-fg-tertiary hover:text-fg"
          )}
          title="Provider settings"
        >
          <Settings2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {showSettings && (
        <div className="px-3 py-2 border-b border-border-subtle space-y-2 bg-bg-deep">
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider">
              Provider
            </label>
            <select
              value={provider}
              onChange={(e) => {
                const p = e.target.value;
                setProvider(p);
                const def = providers.find((x) => x.name === p);
                if (def) setModel(def.default_model);
              }}
              className="ark-input mt-0.5"
            >
              {providers.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider">
              Model
            </label>
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="ark-input mt-0.5 font-mono"
            />
          </div>
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider flex items-center gap-1">
              <KeyRound className="w-3 h-3" />
              {provider === "ollama" ? "Base URL" : "API key"}
              {keyConfigured && (
                <span className="ml-1 text-primary">· configuré</span>
              )}
            </label>
            <input
              type={provider === "ollama" ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                provider === "ollama"
                  ? "http://localhost:11434"
                  : "sk-..."
              }
              className="ark-input mt-0.5 font-mono"
            />
          </div>
          <button
            onClick={saveConfig}
            className="ark-btn-primary w-full flex items-center justify-center gap-1.5"
          >
            <Check className="w-3.5 h-3.5" />
            Sauvegarder
          </button>
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {msgs.length === 0 ? (
          <div className="text-center text-fg-disabled text-xs py-12 space-y-2">
            <Sparkles className="w-8 h-8 mx-auto text-primary/40" />
            <p>Expert ML/DL. Pose-moi une question.</p>
            <div className="text-[10px] space-y-0.5 mt-3 text-fg-tertiary">
              <p>« Architecture pour OCR manuscrit ? »</p>
              <p>« Améliore mon LLM tiny »</p>
              <p>« Diffusion CIFAR-10 from scratch »</p>
            </div>
          </div>
        ) : (
          msgs.map((m, i) => (
            <div
              key={i}
              className={cn(
                "p-3 rounded-md text-sm leading-relaxed whitespace-pre-wrap",
                m.role === "user"
                  ? "bg-accent-blue/10 border border-accent-blue/30 ml-6"
                  : "bg-bg-raised border border-border-subtle mr-6"
              )}
            >
              {m.text}
              {streaming && i === msgs.length - 1 && m.role === "assistant" && (
                <span className="inline-block w-1 h-3 ml-1 bg-primary animate-pulse" />
              )}
            </div>
          ))
        )}
        {pendingArch && (
          <div className="ark-card p-3 border-primary/50">
            <p className="text-[10px] uppercase tracking-widest text-primary font-bold mb-1">
              Architecture proposée
            </p>
            <p className="text-sm font-semibold text-fg mb-1">
              {pendingArch.name}
            </p>
            <p className="text-xs text-fg-tertiary mb-3">
              {pendingArch.nodes?.length || 0} nodes ·{" "}
              {pendingArch.connections?.length || 0} connexions
            </p>
            <button
              onClick={apply}
              className="ark-btn-primary w-full flex items-center justify-center gap-1.5"
            >
              <Check className="w-3.5 h-3.5" />
              Charger dans nouvel onglet
            </button>
          </div>
        )}

        {pendingActions && (
          <div className="ark-card p-3 border-accent-blue/50">
            <p className="text-[10px] uppercase tracking-widest text-accent-blue font-bold mb-2">
              {pendingActions.length} action(s) sur scène courante
            </p>
            <div className="space-y-1 mb-3 max-h-40 overflow-y-auto">
              {pendingActions.map((a, i) => (
                <div
                  key={i}
                  className="font-mono text-[10px] text-fg-secondary bg-bg-deep px-2 py-1 rounded-sm"
                >
                  <span className="text-accent-blue font-bold">{a.op}</span>{" "}
                  {a.type && <span className="text-primary">{a.type}</span>}
                  {a.id && <span className="text-fg-tertiary"> #{a.id}</span>}
                  {a.name && <span className="text-primary"> {a.name}</span>}
                  {a.params && (
                    <span className="text-fg-tertiary ml-1">
                      {Object.entries(a.params)
                        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                        .join(", ")}
                    </span>
                  )}
                  {a.from && a.to && (
                    <span className="text-fg-tertiary">
                      {" "}
                      {a.from}:{a.from_socket || 0} → {a.to}:{a.to_socket || 0}
                    </span>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={applyActions}
              className="ark-btn-primary w-full flex items-center justify-center gap-1.5"
            >
              <Check className="w-3.5 h-3.5" />
              Appliquer sur scène
            </button>
          </div>
        )}
      </div>

      <div className="border-t border-border-subtle p-2 flex items-end gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Demande, critique, architecture…"
          rows={2}
          disabled={streaming}
          className="ark-input flex-1 resize-none text-sm"
        />
        <button
          onClick={send}
          disabled={!input.trim() || streaming}
          className="ark-btn-primary px-2 py-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
