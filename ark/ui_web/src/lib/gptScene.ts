// Phase 69 — 3D GPT scene builder (Three.js). Inspired by bbycroft/llm-viz.
// Vertical tower: token embedding bottom → transformer blocks → output top.
import * as THREE from "three";

export type GptConfig = {
  nLayers: number;
  nHeads: number;
  nEmbd: number;
  seqLen: number;
  vocab: number;
  name?: string;
};

export type ModuleRef = {
  key: string;
  center: THREE.Vector3;
  radius: number;
  meshes: THREE.Object3D[];
};

const COL = {
  embed: 0x00ff85,
  pos: 0x00cc6e,
  ln: 0x38bdf8,
  attn: 0xa78bfa,
  qkv: 0x8b5cf6,
  softmax: 0xfbbf24,
  mlp: 0xf472b6,
  proj: 0x60a5fa,
  output: 0xe11d2a,
  residual: 0xffffff,
};

function labelSprite(text: string, color = "#E4E4E7"): THREE.Sprite {
  const cv = document.createElement("canvas");
  cv.width = 256;
  cv.height = 64;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = color;
  ctx.font = "bold 30px Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true,
    depthTest: false });
  const sp = new THREE.Sprite(mat);
  sp.scale.set(8, 2, 1);
  return sp;
}

function block(
  w: number, h: number, d: number, color: number, opacity = 0.9
): THREE.Mesh {
  const geo = new THREE.BoxGeometry(w, h, d);
  const mat = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.35,
    metalness: 0.2,
    roughness: 0.55,
    transparent: opacity < 1,
    opacity,
  });
  return new THREE.Mesh(geo, mat);
}

// Grid of cells (instanced) — for matrices like embedding / attention
function cellGrid(
  cols: number, rows: number, color: number, cell = 0.32, gap = 0.06
): THREE.InstancedMesh {
  const n = cols * rows;
  const geo = new THREE.BoxGeometry(cell, cell, cell);
  const mat = new THREE.MeshStandardMaterial({
    color, emissive: color, emissiveIntensity: 0.25,
    metalness: 0.1, roughness: 0.6,
  });
  const inst = new THREE.InstancedMesh(geo, mat, n);
  const m = new THREE.Matrix4();
  const step = cell + gap;
  const w = cols * step;
  const hh = rows * step;
  let i = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      m.setPosition(c * step - w / 2, r * step - hh / 2, 0);
      inst.setMatrixAt(i++, m);
    }
  }
  inst.instanceMatrix.needsUpdate = true;
  return inst;
}

export function buildGptScene(cfg: GptConfig): {
  root: THREE.Group;
  modules: ModuleRef[];
  height: number;
} {
  const root = new THREE.Group();
  const modules: ModuleRef[] = [];
  const C = cfg.nEmbd;
  const H = cfg.nHeads;
  const T = cfg.seqLen;
  const V = cfg.vocab;

  // Visual scale helpers (cap dims so huge models stay viewable)
  const cCells = Math.min(C, 24);
  const vCells = Math.min(V, 16);
  let y = 0;
  const LAYER_GAP = 16;
  const add = (key: string, objs: THREE.Object3D[], yc: number) => {
    const box = new THREE.Box3();
    objs.forEach((o) => box.expandByObject(o));
    const center = new THREE.Vector3();
    box.getCenter(center);
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    modules.push({ key, center, radius: Math.max(sphere.radius, 6),
      meshes: objs });
  };

  // ---- 1. Token + position embedding (bottom) ----
  {
    const g = new THREE.Group();
    const emb = cellGrid(cCells, vCells, COL.embed);
    emb.position.set(-7, 0, 0);
    const pos = cellGrid(cCells, Math.min(T, 8), COL.pos);
    pos.position.set(7, 0, 0);
    const lblE = labelSprite("token embed", "#00FF85");
    lblE.position.set(-7, vCells * 0.2 + 2, 0);
    const lblP = labelSprite("pos embed", "#00CC6E");
    lblP.position.set(7, Math.min(T, 8) * 0.2 + 2, 0);
    g.add(emb, pos, lblE, lblP);
    g.position.y = y;
    root.add(g);
    add("embed", [g], y);
    y += LAYER_GAP;
  }

  // ---- residual stream column (central highway) ----
  const totalBlocks = cfg.nLayers;
  const towerTop = y + totalBlocks * LAYER_GAP * 2 + LAYER_GAP;
  {
    const geo = new THREE.CylinderGeometry(0.6, 0.6, towerTop, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: COL.residual, emissive: COL.residual, emissiveIntensity: 0.08,
      transparent: true, opacity: 0.12,
    });
    const cyl = new THREE.Mesh(geo, mat);
    cyl.position.set(0, towerTop / 2 - LAYER_GAP, 0);
    root.add(cyl);
  }

  // ---- transformer blocks ----
  for (let b = 0; b < cfg.nLayers; b++) {
    // LN1
    {
      const g = new THREE.Group();
      const ln = block(10, 0.8, 3, COL.ln);
      const lbl = labelSprite(`LN1 · block ${b}`, "#38BDF8");
      lbl.position.set(0, 1.6, 0);
      g.add(ln, lbl);
      g.position.y = y;
      root.add(g);
      add(`ln1_${b}`, [g], y);
      y += LAYER_GAP * 0.5;
    }
    // Attention — heads side by side, each with Q/K/V + attn matrix
    {
      const g = new THREE.Group();
      const headW = 3.2;
      const spread = (H - 1) * (headW + 1.2);
      for (let h = 0; h < H; h++) {
        const hx = h * (headW + 1.2) - spread / 2;
        const A = Math.max(2, Math.min(Math.floor(C / H), 8));
        const q = cellGrid(A, Math.min(T, 6), COL.qkv, 0.26, 0.05);
        q.position.set(hx - 1, 0, 0);
        const k = cellGrid(A, Math.min(T, 6), COL.qkv, 0.26, 0.05);
        k.position.set(hx, 0, -1.2);
        const v = cellGrid(A, Math.min(T, 6), COL.qkv, 0.26, 0.05);
        v.position.set(hx + 1, 0, -2.4);
        // attention matrix T×T above
        const attnM = cellGrid(Math.min(T, 6), Math.min(T, 6), COL.attn,
          0.3, 0.04);
        attnM.position.set(hx, 4, 0);
        g.add(q, k, v, attnM);
      }
      const lbl = labelSprite(`Self-Attention · ${H} heads`, "#A78BFA");
      lbl.position.set(0, 7, 0);
      g.add(lbl);
      g.position.y = y;
      root.add(g);
      add(`attn_${b}`, [g], y);
      y += LAYER_GAP;
    }
    // Softmax (slim layer on attn)
    {
      const g = new THREE.Group();
      const sm = block(8, 0.6, 2.5, COL.softmax);
      const lbl = labelSprite("softmax", "#FBBF24");
      lbl.position.set(0, 1.4, 0);
      g.add(sm, lbl);
      g.position.y = y;
      root.add(g);
      add(`softmax_${b}`, [g], y);
      y += LAYER_GAP * 0.5;
    }
    // LN2
    {
      const g = new THREE.Group();
      const ln = block(10, 0.8, 3, COL.ln);
      const lbl = labelSprite("LN2", "#38BDF8");
      lbl.position.set(0, 1.6, 0);
      g.add(ln, lbl);
      g.position.y = y;
      root.add(g);
      add(`ln2_${b}`, [g], y);
      y += LAYER_GAP * 0.5;
    }
    // MLP — fc up (4C) then down
    {
      const g = new THREE.Group();
      const up = cellGrid(Math.min(C, 16), 4, COL.mlp, 0.3, 0.05);
      up.position.set(-5, 0, 0);
      const down = cellGrid(4, Math.min(C, 16), COL.mlp, 0.3, 0.05);
      down.position.set(5, 0, 0);
      const lbl = labelSprite(`MLP · 4×${C}`, "#F472B6");
      lbl.position.set(0, 4, 0);
      g.add(up, down, lbl);
      g.position.y = y;
      root.add(g);
      add(`mlp_${b}`, [g], y);
      y += LAYER_GAP;
    }
  }

  // ---- final LN + output ----
  {
    const g = new THREE.Group();
    const ln = block(10, 0.8, 3, COL.ln);
    const lbl = labelSprite("LN final", "#38BDF8");
    lbl.position.set(0, 1.6, 0);
    g.add(ln, lbl);
    g.position.y = y;
    root.add(g);
    add("ln_f", [g], y);
    y += LAYER_GAP * 0.6;
  }
  {
    const g = new THREE.Group();
    const logits = cellGrid(vCells, Math.min(T, 6), COL.output, 0.32, 0.05);
    const lbl = labelSprite(`logits → ${V} vocab`, "#F87171");
    lbl.position.set(0, vCells * 0.2 + 2, 0);
    g.add(logits, lbl);
    g.position.y = y;
    root.add(g);
    add("output", [g], y);
    y += LAYER_GAP;
  }

  // Center tower vertically around origin
  root.position.y = -y / 2;

  return { root, modules, height: y };
}

// ============================================================
// Generic architecture (any ARK model: CNN, ResNet, MLP, diffusion…)
// ============================================================

export type GenericLayer = {
  id: string;
  type: string;
  label: string;
  sub: string;
  category: string;
  kind: keyof typeof COL;
  size: number;
  nIn: number;
};

export function buildGenericScene(layers: GenericLayer[]): {
  root: THREE.Group;
  modules: ModuleRef[];
  height: number;
} {
  const root = new THREE.Group();
  const modules: ModuleRef[] = [];
  if (!layers.length) return { root, modules, height: 0 };

  // Horizontal flow left→right. Block size encodes `size` (log-scaled).
  const GAP = 9;
  let x = 0;
  const prevCenters: THREE.Vector3[] = [];
  const maxSize = Math.max(...layers.map((l) => l.size));

  layers.forEach((layer, i) => {
    const g = new THREE.Group();
    const norm = Math.log2(layer.size + 1) / Math.log2(maxSize + 1 || 2);
    const h = 2 + norm * 10; // height encodes width/channels
    const w = 5;
    const d = 2 + norm * 8;
    const color = COL[layer.kind] ?? COL.proj;
    const mesh = block(w, h, d, color, 0.92);
    g.add(mesh);
    const lbl = labelSprite(layer.label, "#E4E4E7");
    lbl.position.set(0, h / 2 + 1.6, 0);
    lbl.scale.set(10, 2.5, 1);
    g.add(lbl);
    if (layer.sub) {
      const sub = labelSprite(layer.sub, "#B8B8C0");
      sub.position.set(0, h / 2 + 0.2, 0);
      sub.scale.set(9, 1.8, 1);
      g.add(sub);
    }
    g.position.set(x, 0, 0);
    root.add(g);

    const center = new THREE.Vector3(x, 0, 0);
    modules.push({ key: layer.id, center, radius: Math.max(h, w, d),
      meshes: [g] });
    prevCenters.push(center);

    // connection line to previous
    if (i > 0) {
      const prev = prevCenters[i - 1];
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(prev.x + w / 2, 0, 0),
        new THREE.Vector3(x - w / 2, 0, 0),
      ]);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x52525b, transparent: true, opacity: 0.6 });
      root.add(new THREE.Line(lineGeo, lineMat));
    }
    x += GAP;
  });

  // center horizontally
  root.position.x = -(x - GAP) / 2;
  return { root, modules, height: x };
}

