"""Phase 75 — Data conversion & adaptation toolkit.

Inspiré Geometry Nodes (Blender) / TouchDesigner / Houdini : convertir
tout en n'importe quoi via nodes. Catégorie "Data".

Familles :
- Type      : to_list, to_array, to_tensor, reshape, flatten, transpose
- Format    : parse_json, dump_json, parse_csv, to_csv, json_to_csv,
              csv_to_json
- Extract   : extract_numbers, to_int, to_float, to_string
- Array ops : map_range, normalize, standardize, clamp, reduce, slice_data,
              resample
- Image     : image_to_pixels, pixels_to_image
- 3D glTF   : load_gltf, mesh_to_json, gltf_separate, classify_geometry
- Inspect   : data_inspect (lecteur intelligent : type/shape/dtype/aperçu)

Chaque node = DataNode(op). forward(x) transforme + stocke last_text
(aperçu pour Inspector). Tolérant : try/except, ne casse pas le graphe.
"""
from __future__ import annotations
import csv as _csv
import io
import json
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_np(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _preview(obj, maxlen: int = 400) -> str:
    try:
        if isinstance(obj, torch.Tensor):
            return f"<tensor {tuple(obj.shape)} {obj.dtype}>"
        s = repr(obj)
    except Exception:
        s = str(type(obj))
    return s[:maxlen]


class DataNode(nn.Module):
    """Node générique de transformation de données. Dispatch sur self.op."""

    def __init__(self, op: str, params: dict | None = None):
        super().__init__()
        self.op = op
        self.p = params or {}
        self.last_text: str = ""
        self.last_value: Any = None

    # -- helpers --
    def _pf(self, k, d):  # param float
        try:
            return float(self.p.get(k, d))
        except Exception:
            return d

    def _pi(self, k, d):
        try:
            return int(self.p.get(k, d))
        except Exception:
            return d

    def _ps(self, k, d):
        return str(self.p.get(k, d))

    def forward(self, x=None):
        try:
            out = self._dispatch(x)
        except Exception as e:
            self.last_text = f"err [{self.op}]: {e}"
            return x
        self.last_value = out
        self.last_text = _preview(out)
        return out

    def _dispatch(self, x):
        import numpy as np
        op = self.op

        # ---------- Type ----------
        if op == "to_list":
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]
        if op == "to_array":
            return _to_np(x)
        if op == "to_tensor":
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(_to_np(x))
        if op == "reshape":
            shape = self._parse_shape(self._ps("shape", "-1"))
            t = self._as_tensor(x)
            return t.reshape(*shape)
        if op == "flatten":
            return self._as_tensor(x).flatten()
        if op == "transpose":
            t = self._as_tensor(x)
            if t.dim() == 2:
                return t.t()
            dims = list(range(t.dim()))[::-1]
            return t.permute(*dims)

        # ---------- Format ----------
        if op == "parse_json":
            txt = x if isinstance(x, str) else _preview(x, 10**7)
            return json.loads(txt)
        if op == "dump_json":
            return json.dumps(self._jsonable(x), indent=self._pi("indent", 2),
                                ensure_ascii=False)
        if op == "parse_csv":
            sep = self._ps("sep", ",")
            txt = x if isinstance(x, str) else str(x)
            rows = list(_csv.reader(io.StringIO(txt), delimiter=sep))
            if self.p.get("numeric", False):
                rows = [[self._num(c) for c in r] for r in rows]
            return rows
        if op == "to_csv":
            sep = self._ps("sep", ",")
            rows = self._as_rows(x)
            buf = io.StringIO()
            w = _csv.writer(buf, delimiter=sep)
            w.writerows(rows)
            return buf.getvalue()
        if op == "json_to_csv":
            data = x if isinstance(x, (list, dict)) else json.loads(str(x))
            if isinstance(data, dict):
                data = [data]
            keys = list(data[0].keys()) if data else []
            buf = io.StringIO()
            w = _csv.DictWriter(buf, fieldnames=keys,
                                  delimiter=self._ps("sep", ","))
            w.writeheader()
            for row in data:
                w.writerow(row)
            return buf.getvalue()
        if op == "csv_to_json":
            sep = self._ps("sep", ",")
            txt = x if isinstance(x, str) else str(x)
            rd = _csv.DictReader(io.StringIO(txt), delimiter=sep)
            return [dict(r) for r in rd]

        # ---------- Extract ----------
        if op == "extract_numbers":
            txt = x if isinstance(x, str) else _preview(x, 10**6)
            nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", txt)
            return [self._num(n) for n in nums]
        if op == "to_int":
            return self._cast_scalar(x, int)
        if op == "to_float":
            return self._cast_scalar(x, float)
        if op == "to_string":
            if isinstance(x, torch.Tensor):
                return _preview(x.detach().cpu().tolist(), 10**6)
            return str(x)

        # ---------- Array ops ----------
        if op == "map_range":
            t = self._as_tensor(x).float()
            i0 = self._pf("in_min", 0.0); i1 = self._pf("in_max", 1.0)
            o0 = self._pf("out_min", 0.0); o1 = self._pf("out_max", 1.0)
            denom = (i1 - i0) or 1e-9
            r = (t - i0) / denom * (o1 - o0) + o0
            if self.p.get("clamp", False):
                lo, hi = min(o0, o1), max(o0, o1)
                r = r.clamp(lo, hi)
            return r
        if op == "normalize":
            t = self._as_tensor(x).float()
            mn, mx = t.min(), t.max()
            return (t - mn) / ((mx - mn) or 1e-9)
        if op == "standardize":
            t = self._as_tensor(x).float()
            return (t - t.mean()) / (t.std() + 1e-9)
        if op == "clamp":
            t = self._as_tensor(x).float()
            return t.clamp(self._pf("min", 0.0), self._pf("max", 1.0))
        if op == "reduce":
            t = self._as_tensor(x).float()
            mode = self._ps("mode", "mean")
            fns = {"sum": t.sum, "mean": t.mean, "min": t.min,
                     "max": t.max, "prod": t.prod, "std": t.std}
            v = fns.get(mode, t.mean)()
            return float(v.item()) if hasattr(v, "item") else v
        if op == "slice_data":
            s = self._pi("start", 0); e = self.p.get("end", None)
            step = self._pi("step", 1)
            e = int(e) if e not in (None, "", "none") else None
            if isinstance(x, torch.Tensor):
                return x[s:e:step]
            return list(x)[s:e:step]
        if op == "resample":
            n = self._pi("n", 16)
            t = self._as_tensor(x).float().flatten()
            t = t.view(1, 1, -1)
            r = F.interpolate(t, size=n, mode="linear", align_corners=False)
            return r.flatten()

        # ---------- Image ----------
        if op == "image_to_pixels":
            return self._image_to_pixels(x)
        if op == "pixels_to_image":
            return self._pixels_to_image(x)

        # ---------- 3D glTF ----------
        if op == "load_gltf":
            return self._load_gltf()
        if op == "mesh_to_json":
            return self._mesh_to_json(x)
        if op == "gltf_separate":
            return self._gltf_separate(x)
        if op == "classify_geometry":
            return self._classify_geometry(x)

        # ---------- Inspect ----------
        if op == "data_inspect":
            return self._inspect(x)

        raise ValueError(f"op inconnue: {op}")

    # ===================== helpers détaillés =====================

    def _as_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(_to_np(x))

    def _parse_shape(self, s: str):
        parts = re.findall(r"-?\d+", s)
        return [int(p) for p in parts] or [-1]

    def _num(self, s):
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except Exception:
            return s

    def _cast_scalar(self, x, typ):
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return typ(x.item())
            return [typ(v) for v in x.flatten().tolist()]
        if isinstance(x, (list, tuple)):
            return [typ(v) for v in x]
        return typ(x)

    def _jsonable(self, x):
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [self._jsonable(v) for v in x]
        if isinstance(x, dict):
            return {k: self._jsonable(v) for k, v in x.items()}
        return x

    def _as_rows(self, x):
        import numpy as np
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return [[v] for v in x.tolist()]
            return x.tolist()
        if isinstance(x, (list, tuple)):
            if x and isinstance(x[0], (list, tuple)):
                return [list(r) for r in x]
            return [[v] for v in x]
        return [[x]]

    def _image_to_pixels(self, x):
        """Image (C,H,W) ou (B,C,H,W) → matrice (H*W, 5) [x,y,r,g,b]
        (ou format param). Aussi sortie 'matrix' = array (H,W,C)."""
        t = self._as_tensor(x).float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        C, H, W = t.shape
        fmt = self._ps("format", "xyrgb")
        if fmt == "matrix":
            return t.permute(1, 2, 0)  # (H,W,C)
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W),
                                  indexing="ij")
        xs = xs.flatten().float(); ys = ys.flatten().float()
        chans = [t[c].flatten() for c in range(C)]
        if C == 1:
            chans = chans * 3
        cols = [xs, ys] + chans[:3]
        return torch.stack(cols, dim=1)  # (H*W, 5)

    def _pixels_to_image(self, x):
        t = self._as_tensor(x).float()
        H = self._pi("height", 0); W = self._pi("width", 0)
        if t.dim() == 3:  # (H,W,C) matrix
            return t.permute(2, 0, 1)
        # (N,5) xyrgb → reconstruct
        if t.dim() == 2 and t.shape[1] >= 5:
            if not (H and W):
                side = int(t.shape[0] ** 0.5)
                H = W = side
            rgb = t[:, 2:5].reshape(H, W, 3).permute(2, 0, 1)
            return rgb
        return t

    def _inspect(self, x):
        import numpy as np
        lines = []
        tn = type(x).__name__
        lines.append(f"type: {tn}")
        if isinstance(x, torch.Tensor):
            lines.append(f"shape: {tuple(x.shape)}")
            lines.append(f"dtype: {x.dtype}")
            lines.append(f"device: {x.device}")
            if x.numel():
                lines.append(f"min/max: {float(x.min()):.4g} / "
                               f"{float(x.max()):.4g}")
                lines.append(f"mean: {float(x.float().mean()):.4g}")
        elif isinstance(x, np.ndarray):
            lines.append(f"shape: {x.shape} dtype: {x.dtype}")
        elif isinstance(x, (list, tuple)):
            lines.append(f"len: {len(x)}")
            if x:
                lines.append(f"elem[0] type: {type(x[0]).__name__}")
        elif isinstance(x, dict):
            lines.append(f"keys: {list(x.keys())[:20]}")
        elif isinstance(x, str):
            lines.append(f"len: {len(x)}")
            lines.append(f"aperçu: {x[:120]!r}")
        txt = "\n".join(lines)
        self.last_text = txt
        return txt

    # 3D
    def _load_gltf(self):
        path = self._ps("path", "")
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"glTF introuvable: {path}")
        import trimesh
        scene = trimesh.load(str(p), force="scene")
        return scene

    def _scene_meshes(self, scene):
        import trimesh
        if isinstance(scene, trimesh.Trimesh):
            return {"mesh_0": scene}
        if hasattr(scene, "geometry"):
            return dict(scene.geometry)
        return {}

    def _mesh_to_json(self, x):
        import trimesh
        mesh = x
        if not isinstance(mesh, trimesh.Trimesh):
            meshes = self._scene_meshes(x)
            if not meshes:
                raise ValueError("pas de mesh exploitable")
            mesh = next(iter(meshes.values()))
        return {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "vertex_normals": (mesh.vertex_normals.tolist()
                                 if mesh.vertex_normals is not None else []),
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "bounds": mesh.bounds.tolist(),
        }

    def _gltf_separate(self, x):
        meshes = self._scene_meshes(x)
        out = []
        for name, m in meshes.items():
            out.append({
                "name": name,
                "type": "mesh",
                "n_vertices": int(len(m.vertices)),
                "n_faces": int(len(m.faces)),
            })
        # animations / cameras si dispo
        if hasattr(x, "graph"):
            try:
                out.append({"name": "scene_graph",
                              "nodes": list(x.graph.nodes)})
            except Exception:
                pass
        return out

    def _classify_geometry(self, x):
        import trimesh
        meshes = self._scene_meshes(x)
        res = {}
        for name, m in meshes.items():
            if isinstance(m, trimesh.Trimesh):
                if len(m.faces) == 0:
                    res[name] = "pointcloud"
                elif m.is_watertight:
                    res[name] = "mesh (watertight/topology)"
                else:
                    res[name] = "mesh (surface)"
            elif isinstance(m, trimesh.PointCloud):
                res[name] = "pointcloud"
            elif isinstance(m, trimesh.path.Path3D):
                res[name] = "spline/curve"
            else:
                res[name] = type(m).__name__
        return res


# ============================================================
# Registry des ops : name → (title, category-sub, [ParamSpec args], desc,
#                            in_type, out_type)
# ============================================================

def _ops_table():
    # (name, title, params[list of (pname, kind, default, opts)], desc)
    return [
        # Type
        ("to_list", "→ List", [], "Tensor/array → liste Python."),
        ("to_array", "→ Array (numpy)", [], "Liste/tensor → numpy array."),
        ("to_tensor", "→ Tensor", [], "Liste/array → torch tensor."),
        ("reshape", "Reshape", [("shape", "str", "-1", None)],
         "Redimensionne en forme donnée (ex '3,-1')."),
        ("flatten", "Flatten", [], "Aplati en vecteur 1D."),
        ("transpose", "Transpose", [], "Transpose (2D) / inverse axes."),
        # Format
        ("parse_json", "Parse JSON", [],
         "Texte JSON → objet (dict/list)."),
        ("dump_json", "Dump JSON", [("indent", "int", 2, None)],
         "Objet → texte JSON."),
        ("parse_csv", "Parse CSV", [("sep", "str", ","),
                                       ("numeric", "bool", False)],
         "Texte CSV → liste de lignes."),
        ("to_csv", "→ CSV", [("sep", "str", ",")],
         "Array/liste → texte CSV."),
        ("json_to_csv", "JSON → CSV", [("sep", "str", ",")],
         "Liste d'objets JSON → CSV."),
        ("csv_to_json", "CSV → JSON", [("sep", "str", ",")],
         "CSV (avec header) → liste d'objets."),
        # Extract
        ("extract_numbers", "Extract Numbers", [],
         "Extrait tous les nombres d'un texte → liste."),
        ("to_int", "→ Int", [], "Cast en entier(s)."),
        ("to_float", "→ Float", [], "Cast en décimal(aux)."),
        ("to_string", "→ String", [], "Cast en texte."),
        # Array ops
        ("map_range", "Map Range (remap)",
         [("in_min", "float", 0.0), ("in_max", "float", 1.0),
          ("out_min", "float", 0.0), ("out_max", "float", 1.0),
          ("clamp", "bool", False)],
         "Remappe valeurs d'un intervalle vers un autre (Houdini fit)."),
        ("normalize", "Normalize 0-1", [], "(x-min)/(max-min)."),
        ("standardize", "Standardize", [], "(x-mean)/std (z-score)."),
        ("clamp", "Clamp", [("min", "float", 0.0), ("max", "float", 1.0)],
         "Borne les valeurs."),
        ("reduce", "Reduce", [("mode", "choice", "mean",
                                 ["sum", "mean", "min", "max", "prod", "std"])],
         "Agrège (somme/moyenne/min/max...)."),
        ("slice_data", "Slice",
         [("start", "int", 0), ("end", "str", ""), ("step", "int", 1)],
         "Découpe [start:end:step]."),
        ("resample", "Resample", [("n", "int", 16)],
         "Ré-échantillonne en n points (interpolation)."),
        # Image
        ("image_to_pixels", "Image → Pixels",
         [("format", "choice", "xyrgb", ["xyrgb", "matrix"])],
         "Image → matrice pixels. xyrgb=(N,5)[x,y,R,G,B], "
         "matrix=(H,W,C)."),
        ("pixels_to_image", "Pixels → Image",
         [("height", "int", 0), ("width", "int", 0)],
         "Reconstruit image depuis pixels/matrice."),
        # 3D
        ("load_gltf", "Load glTF/GLB",
         [("path", "filepath", "")],
         "Charge scène glTF/GLB (trimesh). Sortie = scène 3D."),
        ("mesh_to_json", "Mesh → JSON", [],
         "Mesh → JSON {vertices, faces, normals, bounds}."),
        ("gltf_separate", "glTF Separate", [],
         "Sépare composants scène (objets/meshes + graph)."),
        ("classify_geometry", "Classify Geometry", [],
         "Qualifie géométrie : mesh/pointcloud/spline/topologie."),
        # Inspect
        ("data_inspect", "Data Inspect", [],
         "Lecteur intelligent : type/shape/dtype/min-max/aperçu."),
    ]


def register_data_handlers(handlers_dict: dict) -> bool:
    for name, *_ in _ops_table():
        def make(op):
            def handler(node, in_shapes):
                m = DataNode(op, dict(node.params))
                return m, in_shapes[0] if in_shapes else (1,)
            return handler
        handlers_dict[name] = make(name)
    return True


def register_data_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec
    CAT = "Data"
    for name, title, params, desc in _ops_table():
        pspecs = []
        for pdef in params:
            pname, kind, default = pdef[0], pdef[1], pdef[2]
            opts = pdef[3] if len(pdef) > 3 and pdef[3] else []
            if kind == "choice":
                pspecs.append(ParamSpec(pname, "choice", default,
                                          options=opts))
            elif kind == "filepath":
                pspecs.append(ParamSpec(pname, "filepath", default,
                                          file_filter="3D (*.gltf *.glb);;All (*)"))
            else:
                pspecs.append(ParamSpec(pname, kind, default))
        registry.register(NodeSpec(
            name=name, title=title, category=CAT,
            inputs=[SocketSpec("in", "any")],
            outputs=[SocketSpec("out", "any")],
            params=pspecs,
            description=desc,
            role="Conversion / adaptation de données.",
            function=f"DataNode op='{name}'.",
            has_thumbnail=False,
        ))
    return True
