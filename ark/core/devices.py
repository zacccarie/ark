"""Périph handlers : webcam (cv2 optionnel) + microphone (sounddevice optionnel).

Deps non installées par défaut. Import paresseux + erreur claire si manque.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class WebcamModule(nn.Module):
    """Capture une frame webcam à chaque forward.

    Demande `opencv-python` (cv2). Le forward retourne tensor (1, 3, H, W).
    """
    def __init__(self, device_id: int = 0, fps: int = 30):
        super().__init__()
        self.device_id = int(device_id)
        self.fps = int(fps)
        self._cap = None

    def _ensure_cap(self):
        if self._cap is None:
            try:
                import cv2
            except ImportError as e:
                raise RuntimeError(
                    "webcam node requires opencv-python: "
                    "uv pip install opencv-python"
                ) from e
            self._cap = cv2.VideoCapture(self.device_id)
            if not self._cap.isOpened():
                raise RuntimeError(f"webcam device {self.device_id} cannot open")

    def forward(self, x=None):
        self._ensure_cap()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        # BGR (cv2) → RGB → (3, H, W) tensor float [0, 1]
        import numpy as np
        frame = frame[:, :, ::-1].copy()  # BGR → RGB
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)  # (1, 3, H, W)

    def __del__(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


class MicrophoneModule(nn.Module):
    """Capture buffer audio via sounddevice.

    Demande `sounddevice` + `numpy`. Le forward retourne tensor (channels, n_samples).
    """
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 buffer_seconds: float = 0.5):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.buffer_seconds = float(buffer_seconds)

    def forward(self, x=None):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError(
                "microphone node requires sounddevice: "
                "uv pip install sounddevice"
            ) from e
        import numpy as np
        n = int(self.sample_rate * self.buffer_seconds)
        rec = sd.rec(n, samplerate=self.sample_rate, channels=self.channels,
                       blocking=True, dtype="float32")
        # (n, channels) → (channels, n)
        return torch.from_numpy(rec.T)


def h_webcam(node, in_shapes):
    dev = int(node.params.get("device_id", 0))
    fps = int(node.params.get("fps", 30))
    return WebcamModule(dev, fps), (3, -1, -1)  # H, W runtime-resolved


def h_microphone(node, in_shapes):
    sr = int(node.params.get("sample_rate", 16000))
    ch = int(node.params.get("channels", 1))
    return MicrophoneModule(sr, ch), (ch, -1)
