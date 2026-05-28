"""Phase 32 — Loaders média Input : image, video, sound, multi-image.

Tous lazy imports. Fallback déterministe si lib absente.
- Image : PIL (déjà installé)
- Video : torchvision.io.read_video ou cv2 fallback
- Sound : torchaudio ou wave (stdlib)
- Multi-image : PIL + glob

Output shapes :
- LoadImageFile      : (3, H, W) ou (1, H, W) selon mode
- LoadVideo          : (T, 3, H, W)
- LoadSound          : (channels, T) ou (T,) si mono
- MultiImageLoader   : (B, 3, H, W) — B = nombre d'images dans dir
"""
from __future__ import annotations
import glob
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Load Image (single file)
# ============================================================

class LoadImageFile(nn.Module):
    def __init__(self, path: str = "", size: int = 0, mode: str = "RGB"):
        super().__init__()
        self.path = path
        self.size = size  # 0 = pas de resize
        self.mode = mode  # "RGB" | "L" (grayscale)

    def forward(self, x=None):
        try:
            from PIL import Image
            import numpy as np
            p = Path(self.path).expanduser()
            if p.exists():
                img = Image.open(p).convert(self.mode)
                if self.size > 0:
                    img = img.resize((self.size, self.size))
                arr = np.array(img).astype("float32") / 255.0
                if arr.ndim == 2:
                    arr = arr[None, ...]  # (1, H, W)
                else:
                    arr = arr.transpose(2, 0, 1)  # (C, H, W)
                return torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
        except Exception:
            pass
        # Fallback : noise déterministe par hash(path)
        seed = int(hashlib.sha1((self.path or "x").encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        C = 3 if self.mode == "RGB" else 1
        s = self.size if self.size > 0 else 256
        return torch.rand(1, C, s, s, generator=g)


# ============================================================
# Load Video
# ============================================================

class LoadVideo(nn.Module):
    def __init__(self, path: str = "", fps: int = 0, max_frames: int = 0,
                 resize: int = 0):
        super().__init__()
        self.path = path
        self.fps = fps  # 0 = original
        self.max_frames = max_frames  # 0 = tous
        self.resize = resize  # 0 = original

    def _load_torchvision(self) -> Optional[torch.Tensor]:
        try:
            from torchvision.io import read_video
            video, _audio, _meta = read_video(self.path, pts_unit="sec")
            # video : (T, H, W, C) uint8
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
            return video
        except Exception:
            return None

    def _load_cv2(self) -> Optional[torch.Tensor]:
        try:
            import cv2
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                return None
            frames = []
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # BGR → RGB
                frame = frame[..., ::-1]
                frames.append(frame)
                n += 1
                if self.max_frames > 0 and n >= self.max_frames:
                    break
            cap.release()
            if not frames:
                return None
            import numpy as np
            arr = np.stack(frames).astype("float32") / 255.0  # (T, H, W, C)
            arr = arr.transpose(0, 3, 1, 2)  # (T, C, H, W)
            return torch.from_numpy(arr)
        except Exception:
            return None

    def forward(self, x=None):
        p = Path(self.path).expanduser()
        if p.exists() and self.path:
            t = self._load_torchvision()
            if t is None:
                t = self._load_cv2()
            if t is not None:
                # Resize si demandé
                if self.resize > 0:
                    t = torch.nn.functional.interpolate(
                        t, size=(self.resize, self.resize),
                        mode="bilinear", align_corners=False)
                # Sub-sample fps si demandé
                if self.fps > 0 and t.shape[0] > self.fps:
                    idx = torch.linspace(0, t.shape[0] - 1, self.fps).long()
                    t = t[idx]
                # Limite max_frames
                if self.max_frames > 0 and t.shape[0] > self.max_frames:
                    t = t[:self.max_frames]
                return t.unsqueeze(0)  # (1, T, C, H, W)
        # Fallback : noise (1, T=30, 3, 64, 64)
        seed = int(hashlib.sha1((self.path or "v").encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        T = self.max_frames if self.max_frames > 0 else 30
        S = self.resize if self.resize > 0 else 64
        return torch.rand(1, T, 3, S, S, generator=g)


# ============================================================
# Load Sound
# ============================================================

class LoadSound(nn.Module):
    def __init__(self, path: str = "", sample_rate: int = 16000,
                 max_duration: float = 0.0, mono: bool = True):
        super().__init__()
        self.path = path
        self.sample_rate = sample_rate
        self.max_duration = max_duration  # 0 = tout
        self.mono = mono

    def _load_torchaudio(self) -> Optional[torch.Tensor]:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(self.path)
            # waveform : (C, T)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        except Exception:
            return None

    def _load_scipy(self) -> Optional[torch.Tensor]:
        try:
            from scipy.io import wavfile
            import numpy as np
            sr, data = wavfile.read(self.path)
            # data : (T,) ou (T, C)
            data = data.astype("float32")
            if data.dtype != "float32" or data.max() > 1.0:
                data = data / 32768.0
            if data.ndim == 1:
                data = data[None, :]
            else:
                data = data.T
            t = torch.from_numpy(data)
            # Resample basic via linear interp si besoin
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                new_len = int(t.shape[-1] * ratio)
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=new_len, mode="linear",
                    align_corners=False).squeeze(0)
            return t
        except Exception:
            return None

    def forward(self, x=None):
        p = Path(self.path).expanduser()
        if p.exists() and self.path:
            t = self._load_torchaudio()
            if t is None:
                t = self._load_scipy()
            if t is not None:
                if self.mono and t.shape[0] > 1:
                    t = t.mean(dim=0, keepdim=True)
                if self.max_duration > 0:
                    max_samples = int(self.max_duration * self.sample_rate)
                    t = t[..., :max_samples]
                return t.unsqueeze(0)  # (1, C, T)
        # Fallback : sine 440Hz, 1s
        seed = int(hashlib.sha1((self.path or "s").encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        T = int((self.max_duration if self.max_duration > 0 else 1.0)
                  * self.sample_rate)
        sr = self.sample_rate
        t = torch.arange(T, dtype=torch.float32) / sr
        wave = torch.sin(2 * 3.14159 * 440 * t) * 0.3
        wave = wave + torch.rand(T, generator=g) * 0.05  # léger noise
        wave = wave.unsqueeze(0)  # (C=1, T)
        return wave.unsqueeze(0)  # (1, 1, T)


# ============================================================
# Multi Image Loader (sequence from dir)
# ============================================================

class MultiImageLoader(nn.Module):
    """Charge tous les images d'un dir/glob → (B, C, H, W)."""

    def __init__(self, path: str = "", pattern: str = "*.png",
                 size: int = 256, max_n: int = 0, sort: bool = True):
        super().__init__()
        self.path = path  # dir or glob
        self.pattern = pattern
        self.size = size
        self.max_n = max_n  # 0 = tous
        self.sort = sort

    def forward(self, x=None):
        try:
            from PIL import Image
            import numpy as np
            p = Path(self.path).expanduser()
            if p.is_dir():
                files = list(p.glob(self.pattern))
                # Ajoute jpg/jpeg si pattern par défaut
                if self.pattern == "*.png" and not files:
                    for ext in ("*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                        files.extend(p.glob(ext))
            else:
                files = [Path(f) for f in glob.glob(str(p))]
            if self.sort:
                files = sorted(files)
            if self.max_n > 0:
                files = files[:self.max_n]
            if not files:
                raise FileNotFoundError("aucune image trouvée")

            arrs = []
            for f in files:
                img = Image.open(f).convert("RGB")
                if self.size > 0:
                    img = img.resize((self.size, self.size))
                arr = np.array(img).astype("float32") / 255.0
                arr = arr.transpose(2, 0, 1)  # (C, H, W)
                arrs.append(arr)
            stack = np.stack(arrs)  # (B, C, H, W)
            return torch.from_numpy(stack)
        except Exception:
            pass
        # Fallback : 4 noise images
        seed = int(hashlib.sha1((self.path or "m").encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        S = self.size if self.size > 0 else 256
        return torch.rand(4, 3, S, S, generator=g)


# ============================================================
# Handlers
# ============================================================

def h_load_image_file(node, in_shapes):
    m = LoadImageFile(
        path=node.params.get("path", ""),
        size=int(node.params.get("size", 0)),
        mode=node.params.get("mode", "RGB"),
    )
    m._ark_node_id = getattr(node, "id", "lim")
    C = 3 if node.params.get("mode", "RGB") == "RGB" else 1
    S = int(node.params.get("size", 256))
    return m, (C, S, S)


def h_load_video(node, in_shapes):
    m = LoadVideo(
        path=node.params.get("path", ""),
        fps=int(node.params.get("fps", 0)),
        max_frames=int(node.params.get("max_frames", 0)),
        resize=int(node.params.get("resize", 0)),
    )
    m._ark_node_id = getattr(node, "id", "vid")
    T = int(node.params.get("max_frames", 30) or 30)
    S = int(node.params.get("resize", 64) or 64)
    return m, (T, 3, S, S)


def h_load_sound(node, in_shapes):
    m = LoadSound(
        path=node.params.get("path", ""),
        sample_rate=int(node.params.get("sample_rate", 16000)),
        max_duration=float(node.params.get("max_duration", 0.0)),
        mono=bool(node.params.get("mono", True)),
    )
    m._ark_node_id = getattr(node, "id", "snd")
    sr = int(node.params.get("sample_rate", 16000))
    dur = float(node.params.get("max_duration", 1.0))
    T_samples = int(sr * dur) if dur > 0 else sr
    return m, (1, T_samples)


def h_multi_image_loader(node, in_shapes):
    m = MultiImageLoader(
        path=node.params.get("path", ""),
        pattern=node.params.get("pattern", "*.png"),
        size=int(node.params.get("size", 256)),
        max_n=int(node.params.get("max_n", 0)),
        sort=bool(node.params.get("sort", True)),
    )
    m._ark_node_id = getattr(node, "id", "mi")
    S = int(node.params.get("size", 256))
    return m, (3, S, S)


def register_media_loader_handlers(handlers_dict: dict) -> bool:
    handlers_dict["load_image_file"] = h_load_image_file
    handlers_dict["load_video"] = h_load_video
    handlers_dict["load_sound"] = h_load_sound
    handlers_dict["multi_image_loader"] = h_multi_image_loader
    return True


def register_media_loader_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    # Filtres Qt courants
    IMG_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All (*)"
    VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All (*)"
    SOUND_FILTER = "Audio (*.mp3 *.wav *.flac *.ogg *.m4a);;All (*)"

    registry.register(NodeSpec(
        name="load_image_file", title="Load Image (file)",
        category="Input",
        outputs=[SocketSpec("image", "image")],
        params=[
            ParamSpec("path", "filepath", "", file_filter=IMG_FILTER),
            ParamSpec("size", "int", 0, min=0, max=4096),
            ParamSpec("mode", "choice", "RGB", options=["RGB", "L"]),
        ],
        description="Charge fichier image (PNG/JPG/WEBP/BMP) → tenseur (1, C, H, W). "
                       "size=0 garde taille originale.",
        role="Charge image disque → tenseur normalisé [0,1].",
        function="PIL.Image.open + convert(mode) + resize(size) + numpy → torch.Tensor. "
                    "Fallback : noise déterministe si fichier absent.",
        param_help={
            "path": "Fichier image PNG/JPG/WEBP/BMP. Cliquer … pour parcourir.",
            "size": "Resize carré (0 = garde original).",
            "mode": "RGB (3 channels) ou L (grayscale 1 channel).",
        },
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="load_video", title="Load Video (.mp4/.mov)",
        category="Input",
        outputs=[SocketSpec("frames", "tensor")],
        params=[
            ParamSpec("path", "filepath", "", file_filter=VIDEO_FILTER),
            ParamSpec("fps", "int", 0, min=0, max=120),
            ParamSpec("max_frames", "int", 30, min=0, max=10000),
            ParamSpec("resize", "int", 256, min=0, max=4096),
        ],
        description="Charge vidéo (mp4/mov/avi/webm) → tenseur (1, T, 3, H, W). "
                       "Lazy via torchvision.io.read_video, fallback cv2.",
        role="Vidéo → séquence frames tensor.",
        function="torchvision.io.read_video (préféré) ou cv2.VideoCapture. "
                    "fps=0 → original. max_frames=0 → tout. resize=0 → original.",
        param_help={
            "path": "Fichier vidéo .mp4/.mov/.avi/.webm. … pour parcourir.",
            "fps": "Sub-sample à N fps (0 = original).",
            "max_frames": "Limite N frames (0 = tout).",
            "resize": "Resize carré (0 = original).",
        },
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="load_sound", title="Load Sound (.mp3/.wav)",
        category="Input",
        outputs=[SocketSpec("audio", "tensor")],
        params=[
            ParamSpec("path", "filepath", "", file_filter=SOUND_FILTER),
            ParamSpec("sample_rate", "int", 16000, min=8000, max=192000),
            ParamSpec("max_duration", "float", 0.0, min=0.0, max=600.0),
            ParamSpec("mono", "bool", True),
        ],
        description="Charge fichier audio → tenseur (1, C, T). "
                       "Lazy torchaudio, fallback scipy.io.wavfile.",
        role="Audio → waveform tensor.",
        function="torchaudio.load (préféré) ou scipy.io.wavfile. "
                    "Resample auto. Mono = mix channels. max_duration=0 = tout.",
        param_help={
            "path": "Fichier audio .mp3/.wav/.flac/.ogg. … pour parcourir.",
            "sample_rate": "Taux échantillonnage cible (Hz). Resample auto.",
            "max_duration": "Durée max en secondes (0 = tout).",
            "mono": "Convertit en mono (moyenne channels).",
        },
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="multi_image_loader", title="Multi Image Loader (dir)",
        category="Input",
        outputs=[SocketSpec("batch", "tensor")],
        params=[
            ParamSpec("path", "dirpath", ""),
            ParamSpec("pattern", "str", "*.png"),
            ParamSpec("size", "int", 256, min=16, max=4096),
            ParamSpec("max_n", "int", 0, min=0, max=10000),
            ParamSpec("sort", "bool", True),
        ],
        description="Charge tous les images d'un dossier → tenseur (B, 3, H, W). "
                       "Auto-fallback jpg/jpeg/webp/bmp si pattern défaut.",
        role="Dossier images → batch tensor.",
        function="PIL + glob pattern. Sort alpha. max_n=0 = tous. "
                    "Resize uniforme size×size.",
        param_help={
            "path": "Dossier contenant images. … pour parcourir.",
            "pattern": "Glob (ex. *.png, frame_*.jpg).",
            "size": "Resize carré uniforme.",
            "max_n": "Limite N images (0 = toutes).",
            "sort": "Trier alphabétique avant chargement.",
        },
        has_thumbnail=True,
    ))
    return True
