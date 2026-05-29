import { useEffect, useRef, useState } from "react";
import { getBridge } from "@/lib/bridge";
import { Cloud, KeyRound, Cpu, Play, Check, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export function ComputePanel() {
  const [bridge, setBridge] = useState<any>(null);
  const [available, setAvailable] = useState(false);
  const [tokenOk, setTokenOk] = useState(false);
  const [gpus, setGpus] = useState<string[]>([]);
  const [gpu, setGpu] = useState("A10G");
  const [epochs, setEpochs] = useState(3);
  const [dataset, setDataset] = useState("mnist");
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      setBridge(b);
      const cfg = JSON.parse(await b.modal_get_config());
      setAvailable(cfg.available);
      setTokenOk(cfg.token_configured);
      setGpus(cfg.gpus || []);
      if (b.modal_log?.connect) {
        b.modal_log.connect((m: string) =>
          setLogs((l) => [...l.slice(-300), m])
        );
      }
      if (b.modal_done?.connect) {
        b.modal_done.connect((s: string) => {
          setRunning(false);
          try {
            setResult(JSON.parse(s));
          } catch {
            setResult(null);
          }
        });
      }
    })();
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs]);

  const saveToken = () => {
    if (!bridge || !tokenId || !tokenSecret) return;
    bridge.modal_set_token?.(tokenId, tokenSecret);
    setTokenOk(true);
    setTokenId("");
    setTokenSecret("");
  };

  const train = () => {
    if (!bridge || running) return;
    setLogs([]);
    setResult(null);
    setRunning(true);
    bridge.modal_train?.("", epochs, gpu, dataset);
  };

  return (
    <div className="flex flex-col h-full bg-bg-panel">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <Cloud className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          Compute · Modal
        </h2>
        {available ? (
          tokenOk ? (
            <span className="text-[10px] text-primary flex items-center gap-1">
              <Check className="w-3 h-3" /> connecté
            </span>
          ) : (
            <span className="text-[10px] text-warn flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> token requis
            </span>
          )
        ) : (
          <span className="text-[10px] text-error flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> modal non installé
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {!available && (
          <div className="ark-card p-3 text-xs text-fg-tertiary">
            <code className="text-primary">pip install modal</code> dans le venv
            ARK, puis relance. Compte Modal gratuit : modal.com.
          </div>
        )}

        {/* Token config */}
        {available && !tokenOk && (
          <div className="ark-card p-3 space-y-2">
            <p className="text-[10px] uppercase tracking-widest text-fg-tertiary font-bold flex items-center gap-1">
              <KeyRound className="w-3 h-3" /> Modal token
            </p>
            <p className="text-[10px] text-fg-disabled">
              modal.com → Settings → API Tokens → New token
            </p>
            <input
              value={tokenId}
              onChange={(e) => setTokenId(e.target.value)}
              placeholder="token_id (ak-…)"
              className="ark-input font-mono"
            />
            <input
              type="password"
              value={tokenSecret}
              onChange={(e) => setTokenSecret(e.target.value)}
              placeholder="token_secret (as-…)"
              className="ark-input font-mono"
            />
            <button
              onClick={saveToken}
              className="ark-btn-primary w-full flex items-center justify-center gap-1.5"
            >
              <Check className="w-3.5 h-3.5" /> Sauvegarder token
            </button>
          </div>
        )}

        {/* Train config */}
        {available && (
          <div className="ark-card p-3 space-y-2">
            <p className="text-[10px] uppercase tracking-widest text-fg-tertiary font-bold flex items-center gap-1">
              <Cpu className="w-3 h-3" /> Job config
            </p>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-fg-tertiary">GPU</label>
                <select
                  value={gpu}
                  onChange={(e) => setGpu(e.target.value)}
                  className="ark-input"
                >
                  {gpus.map((g) => (
                    <option key={g}>{g}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[10px] text-fg-tertiary">Dataset</label>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  className="ark-input"
                >
                  <option value="mnist">MNIST</option>
                  <option value="cifar10">CIFAR-10</option>
                </select>
              </div>
            </div>
            <div>
              <label className="text-[10px] text-fg-tertiary">Epochs</label>
              <input
                type="number"
                min={1}
                max={100}
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                className="ark-input"
              />
            </div>
            <button
              onClick={train}
              disabled={running || !tokenOk}
              className="ark-btn-primary w-full flex items-center justify-center gap-1.5 disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {running ? "Training…" : "Train scène courante sur Modal"}
            </button>
          </div>
        )}

        {/* Result */}
        {result && (
          <div
            className={cn(
              "ark-card p-3",
              result.ok ? "border-primary/50" : "border-error/50"
            )}
          >
            {result.ok ? (
              <>
                <p className="text-[10px] uppercase tracking-widest text-primary font-bold mb-1">
                  ✓ Training terminé
                </p>
                {result.history?.length > 0 && (
                  <p className="text-xs text-fg-secondary">
                    Final : loss{" "}
                    {result.history[result.history.length - 1].loss.toFixed(4)} ·
                    acc{" "}
                    {(
                      result.history[result.history.length - 1].acc * 100
                    ).toFixed(1)}
                    %
                  </p>
                )}
                {result.checkpoint && (
                  <p className="font-mono text-[10px] text-fg-tertiary mt-1 break-all">
                    {result.checkpoint}
                  </p>
                )}
              </>
            ) : (
              <p className="text-xs text-error">✗ {result.error}</p>
            )}
          </div>
        )}

        {/* Live log */}
        {logs.length > 0 && (
          <div
            ref={logRef}
            className="bg-bg-deep rounded-md p-2 font-mono text-[10px] text-fg-secondary max-h-48 overflow-y-auto border border-border-subtle"
          >
            {logs.map((l, i) => (
              <div key={i} className="whitespace-pre-wrap">
                {l}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
