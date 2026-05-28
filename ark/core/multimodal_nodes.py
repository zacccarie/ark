"""Phase 56 — Multimodal nodes (text2video, text2audio, img2text).

Nodes :
- text_to_frame_seq   : prompt str → tensor (T, 3, H, W) animé seeded by hash
- video_writer        : tensor (T, 3, H, W) → .gif / .mp4 sur disque
- text_to_melody      : prompt str → mapping syllables → notes MIDI freqs
- audio_synth         : melody/tensor → waveform tensor (1, N) (sin synth ADSR)
- audio_writer        : waveform → .wav file
- imagenet_classifier : tensor (1, 3, 224, 224) → logits (1, 1000)
- caption_formatter   : top_k dict list → caption "I see X, Y, Z"

Tous lazy-importent imageio / scipy.io.wavfile / torchvision pretrained.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


# ============================================================
# Text → frame sequence (animated, deterministic from prompt)
# ============================================================

class TextToFrameSeq(nn.Module):
    """Génère séquence frames (T, 3, H, W) seedée par hash(prompt).

    Pas de modèle. Animation : gradient sinusoïdal + bruit modulé.
    Sert demo text2video sans 5GB de checkpoints.
    """

    def __init__(self, n_frames: int = 24, height: int = 128, width: int = 128,
                 prompt: str = "ocean sunset"):
        super().__init__()
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.prompt = prompt
        self.last_video_tensor: Optional[torch.Tensor] = None

    def _hash_seed(self, s: str) -> int:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def forward(self, x=None):
        # Si x est string, override prompt
        prompt = x if isinstance(x, str) else self.prompt
        seed = self._hash_seed(prompt)
        g = torch.Generator().manual_seed(seed)
        T, H, W = self.n_frames, self.height, self.width

        # Base color from hash : 3 floats [0..1]
        base_r = ((seed >> 0) & 0xFF) / 255.0
        base_g = ((seed >> 8) & 0xFF) / 255.0
        base_b = ((seed >> 16) & 0xFF) / 255.0

        # Coords grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
            indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2)

        frames = []
        for t in range(T):
            phase = 2 * math.pi * t / T
            wave = torch.sin(4 * r + phase) * 0.5 + 0.5
            noise = torch.randn(H, W, generator=g) * 0.08
            ch_r = (base_r * wave + noise * base_r).clamp(0, 1)
            ch_g = (base_g * (1 - wave) + noise * base_g).clamp(0, 1)
            ch_b = (base_b * torch.sin(2 * r + phase + 1.0) * 0.5 + 0.5 +
                       noise).clamp(0, 1)
            frame = torch.stack([ch_r, ch_g, ch_b], dim=0)  # (3, H, W)
            frames.append(frame)
        video = torch.stack(frames, dim=0)  # (T, 3, H, W)
        self.last_video_tensor = video
        return video


# ============================================================
# Video writer
# ============================================================

class VideoWriter(nn.Module):
    """Tensor (T, 3, H, W) ∈ [0,1] → .gif ou .mp4."""

    def __init__(self, output_path: str = "./output/video.gif", fps: int = 12):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.last_path: Optional[str] = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            import imageio.v2 as imageio
            import numpy as np
            video = x
            if video.dim() == 5:  # (B, T, 3, H, W)
                video = video[0]
            if video.dim() != 4:
                return x
            # Convert to uint8 (T, H, W, 3)
            vid = video.detach().cpu().clamp(0, 1).mul(255).byte()
            vid = vid.permute(0, 2, 3, 1).numpy()
            out = Path(self.output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            suffix = out.suffix.lower()
            if suffix == ".gif":
                imageio.mimsave(str(out), vid, fps=self.fps, loop=0)
            else:
                # mp4 via ffmpeg
                try:
                    imageio.mimsave(str(out), vid, fps=self.fps,
                                       codec="libx264")
                except Exception:
                    # fallback gif
                    out = out.with_suffix(".gif")
                    imageio.mimsave(str(out), vid, fps=self.fps, loop=0)
            self.last_path = str(out)
        except Exception as e:
            self.last_path = f"err: {e}"
        return x


# ============================================================
# Text → melody (syllable → note frequency)
# ============================================================

# Pentatonic A minor : A C D E G (Hz)
_NOTES = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33]


class TextToMelody(nn.Module):
    """Tokenise prompt → liste fréquences (Hz) + durées (s).

    Map char/word → note via hash modulo. Voyelle = note longue.
    Output : tensor (N, 2) [freq, dur]
    """

    def __init__(self, prompt: str = "hello world", note_dur: float = 0.25,
                 base_octave: int = 4):
        super().__init__()
        self.prompt = prompt
        self.note_dur = note_dur
        self.base_octave = base_octave
        self.last_melody: list = []

    def forward(self, x=None):
        prompt = x if isinstance(x, str) else self.prompt
        words = prompt.lower().split()
        if not words:
            words = ["a"]
        notes = []
        for w in words:
            # syllable count = vowels
            syll = max(1, sum(1 for c in w if c in "aeiouy"))
            for i in range(syll):
                seed = (hash(w) + i) & 0xFFFFFFFF
                note_idx = seed % len(_NOTES)
                freq = _NOTES[note_idx] * (2 ** (self.base_octave - 4))
                notes.append([freq, self.note_dur])
        self.last_melody = notes
        return torch.tensor(notes, dtype=torch.float32)


# ============================================================
# Audio synth (waveform from melody)
# ============================================================

class AudioSynth(nn.Module):
    """Tensor (N, 2) [freq, dur] → waveform tensor (1, samples)."""

    def __init__(self, sample_rate: int = 22050, envelope: str = "adsr",
                 wave: str = "sine"):
        super().__init__()
        self.sample_rate = sample_rate
        self.envelope = envelope
        self.wave = wave
        self.last_waveform: Optional[torch.Tensor] = None

    def _adsr(self, n: int) -> torch.Tensor:
        a = int(n * 0.05)  # attack
        d = int(n * 0.10)  # decay
        s_lvl = 0.7
        r = int(n * 0.15)  # release
        s = n - a - d - r
        if s < 0:
            s = 0
            r = n - a - d
        env = torch.cat([
            torch.linspace(0, 1, a) if a > 0 else torch.tensor([]),
            torch.linspace(1, s_lvl, d) if d > 0 else torch.tensor([]),
            torch.full((s,), s_lvl) if s > 0 else torch.tensor([]),
            torch.linspace(s_lvl, 0, r) if r > 0 else torch.tensor([]),
        ])
        return env[:n]

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        if x.dim() != 2 or x.shape[1] != 2:
            return x
        sr = self.sample_rate
        chunks = []
        for note in x:
            freq, dur = float(note[0]), float(note[1])
            n = max(1, int(sr * dur))
            t = torch.arange(n).float() / sr
            if self.wave == "square":
                w = torch.sign(torch.sin(2 * math.pi * freq * t))
            elif self.wave == "triangle":
                w = 2 * torch.abs(2 * (t * freq - torch.floor(t * freq + 0.5))) - 1
            elif self.wave == "saw":
                w = 2 * (t * freq - torch.floor(t * freq + 0.5))
            else:  # sine
                w = torch.sin(2 * math.pi * freq * t)
            if self.envelope == "adsr":
                w = w * self._adsr(n)
            chunks.append(w)
        wave = torch.cat(chunks) if chunks else torch.zeros(sr)
        # Normalize
        peak = wave.abs().max().clamp(min=1e-6)
        wave = wave / peak * 0.9
        wave = wave.unsqueeze(0)  # (1, samples)
        self.last_waveform = wave
        return wave


# ============================================================
# Audio writer (.wav)
# ============================================================

class AudioWriter(nn.Module):
    """Waveform (C, samples) or (samples,) → .wav file."""

    def __init__(self, output_path: str = "./output/audio.wav",
                 sample_rate: int = 22050):
        super().__init__()
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.last_path: Optional[str] = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            import numpy as np
            from scipy.io import wavfile
            w = x.detach().cpu()
            if w.dim() == 2:
                w = w[0]  # take channel 0
            w_np = w.clamp(-1, 1).mul(32767).short().numpy()
            out = Path(self.output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(out), self.sample_rate, w_np)
            self.last_path = str(out)
        except Exception as e:
            self.last_path = f"err: {e}"
        return x


# ============================================================
# ImageNet classifier (real pretrained resnet18)
# ============================================================

_RESNET_CACHE: dict = {}


def _load_resnet(name: str = "resnet18") -> nn.Module:
    if name in _RESNET_CACHE:
        return _RESNET_CACHE[name]
    import torchvision.models as M
    if name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
    elif name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    elif name == "mobilenet_v3":
        m = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
    m.eval()
    _RESNET_CACHE[name] = m
    return m


# ImageNet 1000 classes — load lazy
_IMAGENET_LABELS: list = []


def _imagenet_labels() -> list:
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS:
        return _IMAGENET_LABELS
    try:
        from torchvision.models import ResNet18_Weights
        meta = ResNet18_Weights.DEFAULT.meta
        _IMAGENET_LABELS = list(meta.get("categories", []))
    except Exception:
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_LABELS


class ImageNetClassifier(nn.Module):
    """Tensor (B, 3, 224, 224) → logits (B, 1000). Pretrained ResNet18."""

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone
        self.last_logits: Optional[torch.Tensor] = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            net = _load_resnet(self.backbone_name)
            # Resize to 224×224 if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                import torch.nn.functional as F
                x = F.interpolate(x, size=(224, 224),
                                       mode="bilinear", align_corners=False)
            # Normalize ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x_n = (x - mean) / std
            with torch.no_grad():
                logits = net(x_n)
            self.last_logits = logits.detach()
            return logits
        except Exception:
            return x


# ============================================================
# Caption formatter (top_k predictions → string)
# ============================================================

class CaptionFormatter(nn.Module):
    """Reçoit logits tensor. Compute top-K + ImageNet labels → caption.

    Output : string "I see {top1} ({p1%}), maybe {top2} ({p2%}) or {top3}."
    """

    def __init__(self, k: int = 3, template: str = "default",
                 use_imagenet_labels: bool = True):
        super().__init__()
        self.k = k
        self.template = template
        self.use_imagenet_labels = use_imagenet_labels
        self.last_caption: str = ""
        self.last_predictions: list = []

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            logits = x
            if logits.dim() > 1:
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            k = min(self.k, probs.numel())
            top_v, top_i = torch.topk(probs, k)
            labels = _imagenet_labels() if self.use_imagenet_labels else []
            preds = []
            for v, i in zip(top_v.tolist(), top_i.tolist()):
                lbl = labels[i] if i < len(labels) else f"class_{i}"
                preds.append({"rank": len(preds) + 1, "index": int(i),
                                  "label": lbl, "prob": float(v)})
            self.last_predictions = preds
            if not preds:
                self.last_caption = ""
                return x
            if self.template == "json":
                import json
                self.last_caption = json.dumps(preds, indent=2)
            elif self.template == "list":
                self.last_caption = "\n".join(
                    f"{p['rank']}. {p['label']} ({p['prob']*100:.1f}%)"
                    for p in preds)
            else:  # default narrative
                p1 = preds[0]
                parts = [
                    f"I see a {p1['label']} ({p1['prob']*100:.1f}%)"
                ]
                if len(preds) > 1:
                    parts.append(
                        f"maybe a {preds[1]['label']} "
                        f"({preds[1]['prob']*100:.1f}%)")
                if len(preds) > 2:
                    parts.append(
                        f"or possibly a {preds[2]['label']} "
                        f"({preds[2]['prob']*100:.1f}%)")
                self.last_caption = ", ".join(parts) + "."
        except Exception as e:
            self.last_caption = f"err: {e}"
        return self.last_caption


# ============================================================
# Handlers
# ============================================================

def h_text_to_frame_seq(node, in_shapes):
    m = TextToFrameSeq(
        n_frames=int(node.params.get("n_frames", 24)),
        height=int(node.params.get("height", 128)),
        width=int(node.params.get("width", 128)),
        prompt=node.params.get("prompt", "ocean sunset"),
    )
    H, W = m.height, m.width
    return m, (m.n_frames, 3, H, W)


def h_video_writer(node, in_shapes):
    m = VideoWriter(
        output_path=node.params.get("output_path", "./output/video.gif"),
        fps=int(node.params.get("fps", 12)),
    )
    return m, in_shapes[0] if in_shapes else (1,)


def h_text_to_melody(node, in_shapes):
    m = TextToMelody(
        prompt=node.params.get("prompt", "hello world"),
        note_dur=float(node.params.get("note_dur", 0.25)),
        base_octave=int(node.params.get("base_octave", 4)),
    )
    return m, (8, 2)  # approximate


def h_audio_synth(node, in_shapes):
    m = AudioSynth(
        sample_rate=int(node.params.get("sample_rate", 22050)),
        envelope=node.params.get("envelope", "adsr"),
        wave=node.params.get("wave", "sine"),
    )
    return m, (1, m.sample_rate * 2)


def h_audio_writer(node, in_shapes):
    m = AudioWriter(
        output_path=node.params.get("output_path", "./output/audio.wav"),
        sample_rate=int(node.params.get("sample_rate", 22050)),
    )
    return m, in_shapes[0] if in_shapes else (1,)


def h_imagenet_classifier(node, in_shapes):
    m = ImageNetClassifier(
        backbone=node.params.get("backbone", "resnet18"),
    )
    return m, (1, 1000)


def h_caption_formatter(node, in_shapes):
    m = CaptionFormatter(
        k=int(node.params.get("k", 3)),
        template=node.params.get("template", "default"),
        use_imagenet_labels=bool(node.params.get("use_imagenet_labels", True)),
    )
    return m, (1,)


def register_multimodal_handlers(handlers_dict: dict) -> bool:
    handlers_dict["text_to_frame_seq"] = h_text_to_frame_seq
    handlers_dict["video_writer"] = h_video_writer
    handlers_dict["text_to_melody"] = h_text_to_melody
    handlers_dict["audio_synth"] = h_audio_synth
    handlers_dict["audio_writer"] = h_audio_writer
    handlers_dict["imagenet_classifier"] = h_imagenet_classifier
    handlers_dict["caption_formatter"] = h_caption_formatter
    return True


def register_multimodal_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Multimodal"

    registry.register(NodeSpec(
        name="text_to_frame_seq", title="Text → Frame Sequence",
        category=CAT,
        inputs=[SocketSpec("prompt", "string")],
        outputs=[SocketSpec("video", "tensor")],
        params=[
            ParamSpec("prompt", "str", "ocean sunset"),
            ParamSpec("n_frames", "int", 24, min=4, max=300),
            ParamSpec("height", "int", 128, min=32, max=512),
            ParamSpec("width", "int", 128, min=32, max=512),
        ],
        description="Génère séquence frames (T,3,H,W) seedée par hash(prompt). "
                       "Animation sin radial + bruit. Demo text2video sans modèle lourd.",
        role="Procedural video frames from text prompt.",
        function="hash(prompt) → seed → palette + animation sinusoïdale.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="video_writer", title="Video Writer (.gif/.mp4)",
        category=CAT,
        inputs=[SocketSpec("video", "tensor")],
        outputs=[SocketSpec("video", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/video.gif"),
            ParamSpec("fps", "int", 12, min=1, max=60),
        ],
        description="Écrit tensor (T,3,H,W) ∈ [0,1] sur disque (.gif ou .mp4). "
                       "imageio backend, fallback gif si codec mp4 absent.",
        role="Save video tensor to file.",
        function="permute(T,H,W,3) × 255 uint8 → imageio.mimsave.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="text_to_melody", title="Text → Melody",
        category=CAT,
        inputs=[SocketSpec("prompt", "string")],
        outputs=[SocketSpec("melody", "tensor")],
        params=[
            ParamSpec("prompt", "str", "hello world"),
            ParamSpec("note_dur", "float", 0.25, min=0.05, max=2.0),
            ParamSpec("base_octave", "int", 4, min=2, max=6),
        ],
        description="Tokenise text → syllables → notes pentatonique A minor. "
                       "Output tensor (N,2) [freq Hz, duration s].",
        role="Procedural melody from text.",
        function="word → vowel count = syllables → hash%notes (A C D E G).",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="audio_synth", title="Audio Synth (Sine/Saw)",
        category=CAT,
        inputs=[SocketSpec("melody", "tensor")],
        outputs=[SocketSpec("waveform", "tensor")],
        params=[
            ParamSpec("sample_rate", "int", 22050, min=8000, max=48000),
            ParamSpec("envelope", "choice", "adsr",
                          options=["adsr", "none"]),
            ParamSpec("wave", "choice", "sine",
                          options=["sine", "square", "triangle", "saw"]),
        ],
        description="Mélodie (N,2) → waveform tensor (1, samples). "
                       "Synth additive + envelope ADSR. Output peak-normalized 0.9.",
        role="Synth audio from frequencies+durations.",
        function="sin(2π·f·t) × ADSR + concat + normalize.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="audio_writer", title="Audio Writer (.wav)",
        category=CAT,
        inputs=[SocketSpec("waveform", "tensor")],
        outputs=[SocketSpec("waveform", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/audio.wav"),
            ParamSpec("sample_rate", "int", 22050, min=8000, max=48000),
        ],
        description="Waveform tensor ∈ [-1,1] → .wav int16 via scipy.io.wavfile.",
        role="Save waveform to .wav.",
        function="clamp(-1,1)·32767 → int16 → wavfile.write.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="imagenet_classifier", title="ImageNet Classifier",
        category=CAT,
        inputs=[SocketSpec("image", "tensor")],
        outputs=[SocketSpec("logits", "tensor")],
        params=[
            ParamSpec("backbone", "choice", "resnet18",
                          options=["resnet18", "resnet50", "mobilenet_v3"]),
        ],
        description="Tensor (B,3,H,W) → logits (B,1000). Pretrained ImageNet. "
                       "Auto-resize 224x224 + normalize (0.485,0.456,0.406).",
        role="Image → 1000-class logits.",
        function="torchvision pretrained net.eval() → forward sans grad.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="caption_formatter", title="Caption Formatter",
        category=CAT,
        inputs=[SocketSpec("logits", "tensor")],
        outputs=[SocketSpec("caption", "string")],
        params=[
            ParamSpec("k", "int", 3, min=1, max=20),
            ParamSpec("template", "choice", "default",
                          options=["default", "list", "json"]),
            ParamSpec("use_imagenet_labels", "bool", True),
        ],
        description="Logits → top-K labels + caption narrative. "
                       "Output string : 'I see a {top1} ({p%}), maybe {top2}...'.",
        role="Format predictions as human caption.",
        function="softmax → topk → ImageNet labels lookup → narrative string.",
        has_thumbnail=True,
    ))
    return True
