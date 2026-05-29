import { useEffect, useState } from "react";
import { getBridge } from "@/lib/bridge";
import { Search, ChevronRight, ChevronDown, Box } from "lucide-react";
import { cn } from "@/lib/utils";

type Category = { name: string; nodes: { name: string; title: string }[] };

export function PalettePanel() {
  const [cats, setCats] = useState<Category[]>([]);
  const [query, setQuery] = useState("");
  const [openCats, setOpenCats] = useState<Record<string, boolean>>({});
  const [colors, setColors] = useState<Record<string, string>>({});

  useEffect(() => {
    (async () => {
      const b = await getBridge();
      try {
        setColors(JSON.parse(await b.category_colors()));
      } catch {
        /* noop */
      }
      const catNames: string[] = await b.list_categories();
      const all: Category[] = await Promise.all(
        catNames.map(async (cn) => ({
          name: cn,
          nodes: await b.list_nodes_in_category(cn),
        }))
      );
      setCats(all);
    })();
  }, []);

  const filtered = cats
    .map((c) => ({
      ...c,
      nodes: c.nodes.filter(
        (n) =>
          n.title.toLowerCase().includes(query.toLowerCase()) ||
          n.name.toLowerCase().includes(query.toLowerCase())
      ),
    }))
    .filter((c) => c.nodes.length > 0);

  const toggleCat = (name: string) =>
    setOpenCats((o) => ({ ...o, [name]: !o[name] }));

  const handleDragStart = (e: React.DragEvent, nodeName: string) => {
    e.dataTransfer.setData("application/x-ark-node", nodeName);
    e.dataTransfer.effectAllowed = "copy";
  };

  const handleClick = async (nodeName: string) => {
    const b = await getBridge();
    b.spawn_node?.(nodeName);
  };

  return (
    <div className="flex flex-col h-full bg-bg-panel">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <Box className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          Palette
        </h2>
      </div>
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-fg-disabled" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filtrer nodes…"
            className="ark-input pl-7 text-xs"
          />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-1.5 py-2 space-y-0.5">
        {filtered.length === 0 ? (
          <p className="text-xs text-fg-disabled text-center py-8">
            {cats.length === 0 ? "Chargement…" : "Aucun résultat"}
          </p>
        ) : (
          filtered.map((c) => {
            const open = openCats[c.name] !== false; // open by default
            return (
              <div key={c.name}>
                <button
                  onClick={() => toggleCat(c.name)}
                  className={cn(
                    "flex items-center gap-1.5 w-full px-2 py-1 text-xs font-semibold",
                    "uppercase tracking-wider text-fg-tertiary hover:text-fg",
                    "transition-colors duration-fast"
                  )}
                >
                  {open ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  <span
                    className="w-2.5 h-2.5 rounded-sm shrink-0"
                    style={{ backgroundColor: colors[c.name] || "#71717A" }}
                  />
                  {c.name}
                  <span className="ml-auto text-[10px] text-fg-disabled font-normal">
                    {c.nodes.length}
                  </span>
                </button>
                {open && (
                  <div className="pl-3 space-y-px py-0.5">
                    {c.nodes.map((n) => (
                      <div
                        key={n.name}
                        draggable
                        onDragStart={(e) => handleDragStart(e, n.name)}
                        onDoubleClick={() => handleClick(n.name)}
                        onClick={() => handleClick(n.name)}
                        className={cn(
                          "flex items-center gap-2 px-2 py-1.5 text-xs rounded-sm cursor-pointer",
                          "text-fg-secondary hover:bg-bg-raised hover:text-fg",
                          "active:bg-primary/20 transition-colors duration-fast",
                          "border-l-2"
                        )}
                        style={{ borderLeftColor: colors[c.name] || "#71717A" }}
                        title={`${n.name} — click to add`}
                      >
                        {n.title}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
