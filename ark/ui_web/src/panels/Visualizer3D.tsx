import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { getBridge } from "@/lib/bridge";
import {
  buildGptScene, buildGenericScene, GptConfig, ModuleRef, GenericLayer,
  ModuleMeta, LEGEND,
} from "@/lib/gptScene";
import { buildWalkthrough, Step } from "@/lib/walkthrough";
import { Box, ChevronLeft, ChevronRight, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

const PRESETS: Record<string, GptConfig> = {
  "nano-GPT (demo)": { nLayers: 3, nHeads: 3, nEmbd: 48, seqLen: 6, vocab: 3,
    name: "nano-GPT" },
  "GPT-2 small": { nLayers: 12, nHeads: 12, nEmbd: 768, seqLen: 1024,
    vocab: 50257, name: "GPT-2" },
  "Shakespeare (Karpathy)": { nLayers: 6, nHeads: 6, nEmbd: 384, seqLen: 256,
    vocab: 65, name: "char-GPT" },
};

export function Visualizer3D() {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<any>({});
  const [presetName, setPresetName] = useState("nano-GPT (demo)");
  const [config, setConfig] = useState<GptConfig>(PRESETS["nano-GPT (demo)"]);
  const [steps, setSteps] = useState<Step[]>(buildWalkthrough(3));
  const [stepIdx, setStepIdx] = useState(0);
  const [bridge, setBridge] = useState<any>(null);
  const [fromArk, setFromArk] = useState(false);
  const [genericLayers, setGenericLayers] = useState<GenericLayer[] | null>(null);
  const [toast, setToast] = useState<string>("");
  const [hover, setHover] = useState<
    { meta: ModuleMeta; x: number; y: number } | null
  >(null);
  const [showLegend, setShowLegend] = useState(true);

  // ---- Three.js init ----
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    scene.fog = new THREE.Fog(0x09090b, 80, 260);

    const camera = new THREE.PerspectiveCamera(
      50, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    camera.position.set(40, 10, 60);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(30, 50, 40);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x00ff85, 0.3);
    rim.position.set(-40, 20, -30);
    scene.add(rim);

    // Manual orbit (no OrbitControls dep)
    let dragging = false;
    let px = 0, py = 0;
    const orbit = { theta: 0.9, phi: 1.3, radius: 70,
      target: new THREE.Vector3(0, 0, 0) };
    const camTarget = {
      theta: 0.9, phi: 1.3, radius: 70,
      target: new THREE.Vector3(0, 0, 0),
    };
    const onDown = (e: PointerEvent) => {
      dragging = true; px = e.clientX; py = e.clientY;
    };
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const onMove = (e: PointerEvent) => {
      if (dragging) {
        camTarget.theta -= (e.clientX - px) * 0.008;
        camTarget.phi = Math.max(0.15, Math.min(Math.PI - 0.15,
          camTarget.phi - (e.clientY - py) * 0.008));
        px = e.clientX; py = e.clientY;
        setHover(null);
        return;
      }
      // Hover raycast (only over canvas)
      const rect = renderer.domElement.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right ||
          e.clientY < rect.top || e.clientY > rect.bottom) {
        return;
      }
      const st = stateRef.current;
      if (!st.root || !st.modules?.length) return;
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObject(st.root, true);
      let key: string | null = null;
      for (const h of hits) {
        let o: THREE.Object3D | null = h.object;
        while (o) {
          if (o.userData && o.userData.moduleKey) {
            key = o.userData.moduleKey; break;
          }
          o = o.parent;
        }
        if (key) break;
      }
      if (key) {
        const mod = st.modules.find((m: ModuleRef) => m.key === key);
        if (mod?.meta) {
          setHover({ meta: mod.meta, x: e.clientX - rect.left,
            y: e.clientY - rect.top });
          renderer.domElement.style.cursor = "pointer";
          return;
        }
      }
      setHover(null);
      renderer.domElement.style.cursor = "grab";
    };
    const onUp = () => { dragging = false; };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camTarget.radius = Math.max(15, Math.min(200,
        camTarget.radius + e.deltaY * 0.06));
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      // Lerp orbit
      orbit.theta += (camTarget.theta - orbit.theta) * 0.08;
      orbit.phi += (camTarget.phi - orbit.phi) * 0.08;
      orbit.radius += (camTarget.radius - orbit.radius) * 0.08;
      orbit.target.lerp(camTarget.target, 0.08);
      const sinP = Math.sin(orbit.phi);
      camera.position.set(
        orbit.target.x + orbit.radius * sinP * Math.sin(orbit.theta),
        orbit.target.y + orbit.radius * Math.cos(orbit.phi),
        orbit.target.z + orbit.radius * sinP * Math.cos(orbit.theta));
      camera.lookAt(orbit.target);
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mount.clientWidth) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);

    stateRef.current = { scene, camera, renderer, camTarget, orbit,
      root: null as THREE.Group | null, modules: [] as ModuleRef[] };

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // ---- (re)build scene on config / generic change ----
  useEffect(() => {
    const st = stateRef.current;
    if (!st.scene) return;
    if (st.root) {
      st.scene.remove(st.root);
      st.root.traverse((o: any) => {
        o.geometry?.dispose?.();
        o.material?.dispose?.();
      });
    }
    if (genericLayers && genericLayers.length) {
      const { root, modules } = buildGenericScene(genericLayers);
      st.scene.add(root);
      st.root = root;
      st.modules = modules;
      // Generic walkthrough: overview + 1 step per layer
      const gsteps: Step[] = [
        {
          title: "Vue d'ensemble",
          focus: "__all__",
          body: `Architecture ARK : ${genericLayers.length} couches. Le flux ` +
            "va de gauche (entrée) à droite (sortie). Hauteur/profondeur des " +
            "blocs ∝ taille (neurones/canaux). Glisse pour parcourir chaque couche.",
        },
        ...genericLayers.map((l) => ({
          title: l.label,
          focus: l.id,
          body: `${l.label}${l.sub ? " · " + l.sub : ""}\nCatégorie : ` +
            `${l.category}. Type node : ${l.type}.`,
        })),
      ];
      setSteps(gsteps);
    } else {
      const { root, modules } = buildGptScene(config);
      st.scene.add(root);
      st.root = root;
      st.modules = modules;
      setSteps(buildWalkthrough(config.nLayers));
    }
    setStepIdx(0);
  }, [config, genericLayers]);

  // ---- focus camera on step ----
  const focusStep = useCallback((idx: number) => {
    const st = stateRef.current;
    if (!st.modules || !st.root) return;
    const step = steps[idx];
    if (!step) return;
    const ct = st.camTarget;
    if (step.focus === "__all__" || step.focus === "__blocks__") {
      ct.target.set(0, 0, 0);
      ct.radius = step.focus === "__all__" ? 90 : 55;
      ct.theta = 0.9; ct.phi = 1.2;
      return;
    }
    const mod: ModuleRef | undefined =
      st.modules.find((m: ModuleRef) => m.key === step.focus) ||
      st.modules.find((m: ModuleRef) => m.key.startsWith(step.focus));
    if (!mod) return;
    // world center (root may have x/y offset)
    const world = mod.center.clone().add(st.root.position);
    ct.target.copy(world);
    ct.radius = Math.max(16, mod.radius * 2.6);
    ct.theta = 0.6; ct.phi = 1.35;
  }, [steps]);

  useEffect(() => { focusStep(stepIdx); }, [stepIdx, focusStep]);

  // ---- bridge: load ARK model config ----
  useEffect(() => {
    (async () => {
      const b = await getBridge();
      setBridge(b);
    })();
  }, []);

  const loadFromArk = async () => {
    if (!bridge) return;
    try {
      const cfgStr = await bridge.viz_get_model_config?.();
      if (!cfgStr) {
        setToast("Scène vide — ajoute des nodes d'abord.");
        setTimeout(() => setToast(""), 3000);
        return;
      }
      const c = JSON.parse(cfgStr);
      if (c && c.generic) {
        // Generic architecture mode (CNN/ResNet/MLP/etc)
        if (!c.layers || !c.layers.length) {
          setToast("Aucune couche détectée dans la scène.");
          setTimeout(() => setToast(""), 3000);
          return;
        }
        setGenericLayers(c.layers);
        setPresetName(`ARK · ${c.name || "scène"}`);
        setFromArk(true);
        setToast(`${c.layers.length} couches chargées.`);
        setTimeout(() => setToast(""), 2500);
      } else if (c && c.nEmbd) {
        // Transformer/GPT mode
        setGenericLayers(null);
        setConfig(c);
        setPresetName(c.name ? `ARK · ${c.name}` : "ARK scène");
        setFromArk(true);
        setToast("Modèle transformer chargé (mode GPT).");
        setTimeout(() => setToast(""), 2500);
      }
    } catch (e) {
      setToast("Erreur chargement scène.");
      setTimeout(() => setToast(""), 3000);
    }
  };

  const step = steps[stepIdx];

  return (
    <div className="flex flex-col h-full bg-bg-deep relative">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle bg-bg-panel z-10">
        <Box className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-fg-secondary">
          LLM 3D
        </h2>
        <select
          value={presetName}
          onChange={(e) => {
            const p = PRESETS[e.target.value];
            if (p) {
              setPresetName(e.target.value);
              setGenericLayers(null); // exit generic mode → rebuild GPT
              setConfig({ ...p }); // new ref forces effect even if same preset
              setFromArk(false);
            }
          }}
          className="ark-input w-44 ml-1"
        >
          {Object.keys(PRESETS).map((k) => (
            <option key={k}>{k}</option>
          ))}
          {fromArk && <option>{presetName}</option>}
        </select>
        <button
          onClick={loadFromArk}
          className="ark-btn flex items-center gap-1 text-xs"
          title="Visualiser le tiny_llm_model de la scène ARK"
        >
          <Cpu className="w-3.5 h-3.5" />
          Depuis ARK
        </button>
        <span className="text-[10px] text-fg-disabled ml-auto font-mono">
          {config.nLayers}L · {config.nHeads}H · {config.nEmbd}d · vocab{" "}
          {config.vocab}
        </span>
      </div>

      {/* 3D canvas */}
      <div ref={mountRef} className="flex-1 min-h-0 cursor-grab active:cursor-grabbing" />

      {/* Toast */}
      {toast && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-20 bg-bg-overlay border border-primary/50 text-fg text-xs px-3 py-2 rounded-md shadow-lg">
          {toast}
        </div>
      )}

      {/* Hover tooltip */}
      {hover && (
        <div
          className="absolute z-30 pointer-events-none bg-bg-overlay/95 backdrop-blur border border-border-strong rounded-md shadow-lg px-3 py-2 max-w-[240px]"
          style={{
            left: Math.min(hover.x + 14, 9999),
            top: hover.y + 14,
            transform:
              hover.x > 300 ? "translateX(-100%) translateX(-28px)" : "none",
          }}
        >
          <div className="flex items-center gap-1.5 mb-1">
            <span
              className="w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: hover.meta.colorHex }}
            />
            <span className="text-xs font-bold text-fg">
              {hover.meta.label}
            </span>
          </div>
          <p className="text-[10px] text-fg-tertiary uppercase tracking-wider mb-1">
            {hover.meta.category}
          </p>
          {hover.meta.info && (
            <p className="text-[11px] text-fg-secondary leading-snug">
              {hover.meta.info}
            </p>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-14 right-3 z-20">
        <button
          onClick={() => setShowLegend((s) => !s)}
          className="ark-btn text-[10px] px-2 py-1 mb-1 w-full"
        >
          {showLegend ? "Masquer légende" : "Légende"}
        </button>
        {showLegend && (
          <div className="bg-bg-panel/90 backdrop-blur border border-border-subtle rounded-md p-2 space-y-1 w-52">
            <p className="text-[9px] uppercase tracking-widest text-fg-disabled font-bold mb-1">
              Couleur = catégorie
            </p>
            {LEGEND.map((l) => (
              <div key={l.hex} className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-sm shrink-0"
                  style={{ backgroundColor: l.hex }}
                />
                <span className="text-[10px] text-fg-secondary">
                  {l.label}
                </span>
              </div>
            ))}
            <p className="text-[9px] text-fg-disabled mt-1.5 leading-tight border-t border-border-subtle pt-1.5">
              Taille bloc ∝ dimension (neurones / canaux). Survole un bloc
              pour le détail.
            </p>
          </div>
        )}
      </div>

      {/* Didactic card */}
      <div className="absolute bottom-0 left-0 right-0 bg-bg-panel/95 backdrop-blur border-t border-border-subtle p-3 z-10">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-mono text-primary">
                {stepIdx + 1}/{steps.length}
              </span>
              <h3 className="text-sm font-bold text-fg">{step?.title}</h3>
            </div>
            <p className="text-xs text-fg-secondary leading-relaxed">
              {step?.body}
            </p>
          </div>
          <div className="flex flex-col gap-1">
            <button
              onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
              disabled={stepIdx === 0}
              className="ark-btn p-1.5 disabled:opacity-30"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() =>
                setStepIdx((i) => Math.min(steps.length - 1, i + 1))
              }
              disabled={stepIdx === steps.length - 1}
              className="ark-btn p-1.5 disabled:opacity-30"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
        {/* Slider */}
        <input
          type="range"
          min={0}
          max={steps.length - 1}
          value={stepIdx}
          onChange={(e) => setStepIdx(parseInt(e.target.value))}
          className="w-full mt-2 accent-primary"
        />
        <div className="flex justify-between mt-1">
          {steps.map((s, i) => (
            <button
              key={i}
              onClick={() => setStepIdx(i)}
              className={cn(
                "text-[9px] truncate px-0.5 transition-colors",
                i === stepIdx
                  ? "text-primary font-bold"
                  : "text-fg-disabled hover:text-fg-tertiary"
              )}
              style={{ maxWidth: `${100 / steps.length}%` }}
              title={s.title}
            >
              {s.title.split(" ")[0]}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
