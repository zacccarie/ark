"""Phase 59 — Real neural multimodal nodes.

- musicgen_audio  : facebook/musicgen-small → text → music (32kHz mono)
- bark_audio      : suno/bark-small → text → speech + sfx
- text2video_diff : damo-vilab/text-to-video-ms-1.7b → text → video tensor
- audioldm_audio  : cvssp/audioldm2 → text → ambient sound (fallback)

Tous lazy-import (transformers/diffusers/scipy) — modèles téléchargés au
1er forward via HuggingFace cache.

Warning : modèles 300M-2GB. 30-120s par génération MPS.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


_MODEL_CACHE: dict = {}


def _pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# MusicGen — facebook/musicgen-small (300M params)
# ============================================================

class MusicGenAudio(nn.Module):
    """Text → music waveform via MusicGen small (transformers).

    Output : tensor (1, samples) at 32000 Hz.
    Latency MPS : ~30s pour 5s d'audio.
    """

    def __init__(self, model_id: str = "facebook/musicgen-small",
                 duration_s: float = 5.0, guidance: float = 3.0,
                 sampling_rate: int = 32000):
        super().__init__()
        self.model_id = model_id
        self.duration_s = duration_s
        self.guidance = guidance
        self.sampling_rate = sampling_rate
        self.last_waveform: Optional[torch.Tensor] = None
        self.last_audio_path: Optional[str] = None
        self.last_status: str = "idle"
        # Inspector tag
        self._is_audio_player = True

    def _load(self):
        key = ("musicgen", self.model_id)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)
        dev = _pick_device()
        try:
            model = model.to(dev)
        except Exception:
            dev = "cpu"
            model = model.to(dev)
        model.eval()
        _MODEL_CACHE[key] = (processor, model, dev)
        return _MODEL_CACHE[key]

    def forward(self, x):
        prompt = x if isinstance(x, str) else getattr(self, "prompt", "")
        if not prompt:
            prompt = "electronic ambient music"
        try:
            self.last_status = "loading"
            processor, model, dev = self._load()
            self.last_status = "generating"
            inputs = processor(text=[prompt], padding=True,
                                  return_tensors="pt").to(dev)
            # Max new tokens ≈ duration_s * frame_rate (model.config.audio_encoder.frame_rate ~50)
            try:
                fr = model.config.audio_encoder.frame_rate
            except Exception:
                fr = 50
            max_new = int(self.duration_s * fr)
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=self.guidance,
                    max_new_tokens=max_new,
                )
            # (B, C, T) → (1, T) mono
            wave = audio_values[0].detach().cpu().float()
            if wave.dim() == 2:
                wave = wave.mean(dim=0, keepdim=True)
            elif wave.dim() == 1:
                wave = wave.unsqueeze(0)
            # Peak normalize
            peak = wave.abs().max().clamp(min=1e-6)
            wave = wave / peak * 0.95
            self.last_waveform = wave
            # Write cache for Inspector audio player
            self._write_cache(wave)
            self.last_status = "ok"
            return wave
        except Exception as e:
            self.last_status = f"err: {e}"
            return x

    def _write_cache(self, wave):
        try:
            import numpy as np
            from scipy.io import wavfile
            cache_dir = Path.home() / ".ark" / "audio_player"
            cache_dir.mkdir(parents=True, exist_ok=True)
            node_id = getattr(self, "_ark_node_id", "musicgen")
            out = cache_dir / f"{node_id}.wav"
            w = wave[0].clamp(-1, 1).mul(32767).short().numpy()
            wavfile.write(str(out), self.sampling_rate, w)
            self.last_audio_path = str(out)
        except Exception:
            pass


# ============================================================
# Bark — suno/bark-small (text-to-speech + sfx)
# ============================================================

class BarkAudio(nn.Module):
    """Text → speech/music via Bark (suno).

    Output : tensor (1, samples) at 24000 Hz.
    """

    def __init__(self, model_id: str = "suno/bark-small",
                 voice_preset: str = "v2/en_speaker_6",
                 sampling_rate: int = 24000):
        super().__init__()
        self.model_id = model_id
        self.voice_preset = voice_preset
        self.sampling_rate = sampling_rate
        self.last_waveform: Optional[torch.Tensor] = None
        self.last_audio_path: Optional[str] = None
        self.last_status: str = "idle"
        self._is_audio_player = True

    def _load(self):
        key = ("bark", self.model_id)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from transformers import AutoProcessor, BarkModel
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = BarkModel.from_pretrained(self.model_id)
        dev = _pick_device()
        try:
            model = model.to(dev)
        except Exception:
            dev = "cpu"
            model = model.to(dev)
        model.eval()
        _MODEL_CACHE[key] = (processor, model, dev)
        return _MODEL_CACHE[key]

    def forward(self, x):
        prompt = x if isinstance(x, str) else getattr(self, "prompt", "")
        if not prompt:
            prompt = "Hello, this is Bark speaking."
        try:
            self.last_status = "loading"
            processor, model, dev = self._load()
            self.last_status = "generating"
            inputs = processor(prompt,
                                  voice_preset=self.voice_preset,
                                  return_tensors="pt").to(dev)
            with torch.no_grad():
                audio = model.generate(**inputs)
            wave = audio[0].detach().cpu().float()
            if wave.dim() == 1:
                wave = wave.unsqueeze(0)
            peak = wave.abs().max().clamp(min=1e-6)
            wave = wave / peak * 0.95
            self.last_waveform = wave
            self._write_cache(wave)
            self.last_status = "ok"
            return wave
        except Exception as e:
            self.last_status = f"err: {e}"
            return x

    def _write_cache(self, wave):
        try:
            from scipy.io import wavfile
            cache_dir = Path.home() / ".ark" / "audio_player"
            cache_dir.mkdir(parents=True, exist_ok=True)
            node_id = getattr(self, "_ark_node_id", "bark")
            out = cache_dir / f"{node_id}.wav"
            w = wave[0].clamp(-1, 1).mul(32767).short().numpy()
            wavfile.write(str(out), self.sampling_rate, w)
            self.last_audio_path = str(out)
        except Exception:
            pass


# ============================================================
# Text2Video Diffusion — damo-vilab/text-to-video-ms-1.7b
# ============================================================

class Text2VideoDiffusion(nn.Module):
    """Text → video tensor (T, 3, H, W) via DiffusionPipeline.

    Output : (T, 3, H, W) ∈ [0, 1].
    Latency MPS : ~2min pour 16 frames 256×256.
    """

    def __init__(self, model_id: str = "damo-vilab/text-to-video-ms-1.7b",
                 num_frames: int = 16, height: int = 256, width: int = 256,
                 num_inference_steps: int = 25, guidance: float = 9.0):
        super().__init__()
        self.model_id = model_id
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.last_video_tensor: Optional[torch.Tensor] = None
        self.last_status: str = "idle"

    def _load(self):
        key = ("t2v", self.model_id)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32)
        dev = _pick_device()
        try:
            pipe = pipe.to(dev)
        except Exception:
            dev = "cpu"
            pipe = pipe.to(dev)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _MODEL_CACHE[key] = (pipe, dev)
        return _MODEL_CACHE[key]

    def forward(self, x):
        prompt = x if isinstance(x, str) else getattr(self, "prompt", "")
        if not prompt:
            prompt = "A panda is eating bamboo"
        try:
            self.last_status = "loading"
            pipe, dev = self._load()
            self.last_status = "generating"
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_frames=self.num_frames,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance,
                )
            # result.frames = list[PIL] OR np.ndarray (T, H, W, 3) [0,1]
            frames = result.frames
            if isinstance(frames, list) and len(frames) > 0:
                # PIL list (1 per frame)
                import numpy as np
                arr = []
                for f in frames:
                    if hasattr(f, "convert"):  # PIL
                        arr.append(np.array(f.convert("RGB")))
                    else:
                        arr.append(np.array(f))
                arr = np.stack(arr, axis=0)  # (T, H, W, 3)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                # (T, H, W, 3) → (T, 3, H, W)
                video = torch.from_numpy(arr).float().permute(0, 3, 1, 2)
            else:
                # ndarray shape (B, T, H, W, 3) or (T, H, W, 3)
                import numpy as np
                arr = frames if isinstance(frames, np.ndarray) else np.array(frames)
                if arr.ndim == 5:
                    arr = arr[0]
                if arr.max() > 1.5:
                    arr = arr / 255.0
                video = torch.from_numpy(arr).float().permute(0, 3, 1, 2)
            video = video.clamp(0, 1)
            self.last_video_tensor = video
            self.last_status = "ok"
            return video
        except Exception as e:
            self.last_status = f"err: {e}"
            # Return placeholder noise
            T = self.num_frames
            return torch.zeros(T, 3, self.height, self.width)


# ============================================================
# Handlers
# ============================================================

def h_musicgen_audio(node, in_shapes):
    m = MusicGenAudio(
        model_id=node.params.get("model_id", "facebook/musicgen-small"),
        duration_s=float(node.params.get("duration_s", 5.0)),
        guidance=float(node.params.get("guidance", 3.0)),
        sampling_rate=int(node.params.get("sampling_rate", 32000)),
    )
    m._ark_node_id = getattr(node, "id", "musicgen")
    sr = m.sampling_rate
    return m, (1, int(sr * m.duration_s))


def h_bark_audio(node, in_shapes):
    m = BarkAudio(
        model_id=node.params.get("model_id", "suno/bark-small"),
        voice_preset=node.params.get("voice_preset", "v2/en_speaker_6"),
        sampling_rate=int(node.params.get("sampling_rate", 24000)),
    )
    m._ark_node_id = getattr(node, "id", "bark")
    return m, (1, int(m.sampling_rate * 6))  # ~6s typical


def h_text2video_diffusion(node, in_shapes):
    m = Text2VideoDiffusion(
        model_id=node.params.get("model_id",
                                       "damo-vilab/text-to-video-ms-1.7b"),
        num_frames=int(node.params.get("num_frames", 16)),
        height=int(node.params.get("height", 256)),
        width=int(node.params.get("width", 256)),
        num_inference_steps=int(node.params.get("steps", 25)),
        guidance=float(node.params.get("guidance", 9.0)),
    )
    return m, (m.num_frames, 3, m.height, m.width)


def register_real_multimodal_handlers(handlers_dict: dict) -> bool:
    handlers_dict["musicgen_audio"] = h_musicgen_audio
    handlers_dict["bark_audio"] = h_bark_audio
    handlers_dict["text2video_diffusion"] = h_text2video_diffusion
    return True


def register_real_multimodal_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Multimodal (Neural)"

    registry.register(NodeSpec(
        name="musicgen_audio", title="MusicGen Audio (Neural)",
        category=CAT,
        inputs=[SocketSpec("prompt", "string")],
        outputs=[SocketSpec("waveform", "tensor")],
        params=[
            ParamSpec("model_id", "str", "facebook/musicgen-small"),
            ParamSpec("duration_s", "float", 5.0, min=1.0, max=30.0),
            ParamSpec("guidance", "float", 3.0, min=1.0, max=15.0),
            ParamSpec("sampling_rate", "int", 32000),
        ],
        description="REAL text→music via MusicGen 300M. Download ~1.5GB au 1er run. "
                       "~30s par 5s d'audio MPS. Output (1, samples) @ 32kHz.",
        role="Neural text-to-music generation.",
        function="transformers.MusicgenForConditionalGeneration → generate.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="bark_audio", title="Bark Speech/Music (Neural)",
        category=CAT,
        inputs=[SocketSpec("prompt", "string")],
        outputs=[SocketSpec("waveform", "tensor")],
        params=[
            ParamSpec("model_id", "str", "suno/bark-small"),
            ParamSpec("voice_preset", "choice", "v2/en_speaker_6",
                          options=["v2/en_speaker_0", "v2/en_speaker_6",
                                       "v2/fr_speaker_1", "v2/fr_speaker_5",
                                       "v2/en_speaker_9"]),
            ParamSpec("sampling_rate", "int", 24000),
        ],
        description="REAL text→speech/music via Bark Suno. Download ~700MB. "
                       "Output (1, samples) @ 24kHz. ~30s génération MPS.",
        role="Neural text-to-speech (with music capability via [MUSIC] tag).",
        function="transformers.BarkModel.generate avec voice preset.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="text2video_diffusion", title="Text2Video Diffusion (Neural)",
        category=CAT,
        inputs=[SocketSpec("prompt", "string")],
        outputs=[SocketSpec("video", "tensor")],
        params=[
            ParamSpec("model_id", "str", "damo-vilab/text-to-video-ms-1.7b"),
            ParamSpec("num_frames", "int", 16, min=4, max=64),
            ParamSpec("height", "int", 256, min=128, max=512),
            ParamSpec("width", "int", 256, min=128, max=512),
            ParamSpec("steps", "int", 25, min=5, max=100),
            ParamSpec("guidance", "float", 9.0, min=1.0, max=20.0),
        ],
        description="REAL text→video via ModelScope 1.7B diffusers. "
                       "Download ~4GB. ~2min pour 16 frames 256x256 MPS. "
                       "Output (T, 3, H, W) ∈ [0, 1].",
        role="Neural text-to-video generation.",
        function="diffusers.DiffusionPipeline avec attention_slicing MPS.",
        has_thumbnail=True,
    ))
    return True
