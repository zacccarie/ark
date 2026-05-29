import { useEffect, useRef, useState } from "react";
import { getBridge } from "@/lib/bridge";
import { TerminalSquare, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Line = { ts: number; text: string; kind?: "info" | "warn" | "err" };

export function TerminalPanel() {
  const [lines, setLines] = useState<Line[]>([]);
  const [filter, setFilter] = useState<"log" | "python" | "metrics">("log");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      if (b.log_appended && b.log_appended.connect) {
        b.log_appended.connect((msg: string) => {
          setLines((ls) =>
            [...ls.slice(-499), { ts: Date.now(), text: msg }]
          );
        });
      }
    })();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div className="flex flex-col h-full bg-bg-deep">
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border-subtle bg-bg-panel">
        <TerminalSquare className="w-4 h-4 text-primary" />
        {(["log", "python", "metrics"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={cn(
              "text-[10px] uppercase tracking-wider font-bold px-2 py-1 rounded-sm transition-colors",
              filter === f
                ? "text-primary border-b-2 border-primary"
                : "text-fg-tertiary hover:text-fg"
            )}
          >
            {f}
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={() => setLines([])}
          className="text-fg-tertiary hover:text-fg p-1"
          title="Clear"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-2 font-mono text-[11px] leading-relaxed">
        {lines.length === 0 ? (
          <p className="text-fg-disabled italic">No output yet…</p>
        ) : (
          lines.map((l, i) => (
            <div key={i} className="text-fg-secondary whitespace-pre-wrap">
              {l.text}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
