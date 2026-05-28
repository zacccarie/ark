"""Tests Phase 32 — Media loaders + filepath ParamSpec + Inspector picker."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import wave
import struct
from pathlib import Path
import pytest
import torch
from types import SimpleNamespace

from ark.app import create_app
from ark.nodes.registry import default_registry, ParamSpec
from ark.core.media_loaders import (
    LoadImageFile, LoadVideo, LoadSound, MultiImageLoader,
    h_load_image_file, h_load_video, h_load_sound, h_multi_image_loader,
)
from ark.core.graph_compiler import HANDLERS
from ark.ui.panels.inspector_panel import InspectorPanel
from ark.ui.node_editor.node_item import NodeItem


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_paramspec_filepath_kind():
    p = ParamSpec(name="path", kind="filepath", default="",
                    file_filter="Images (*.png)")
    assert p.kind == "filepath"
    assert p.file_filter == "Images (*.png)"


def test_paramspec_dirpath_kind():
    p = ParamSpec(name="folder", kind="dirpath", default="")
    assert p.kind == "dirpath"


def test_load_image_file_fallback(app):
    """Sans fichier réel → fallback noise déterministe."""
    m = LoadImageFile(path="/nonexistent/x.png", size=64, mode="RGB")
    out = m()
    assert out.shape == (1, 3, 64, 64)
    out2 = m()
    assert torch.allclose(out, out2)  # déterministe


def test_load_image_file_grayscale(app):
    m = LoadImageFile(path="", size=32, mode="L")
    out = m()
    assert out.shape == (1, 1, 32, 32)


def test_load_image_file_real(tmp_path, app):
    """Crée vraie image PNG + charge."""
    from PIL import Image
    import numpy as np
    arr = np.zeros((48, 48, 3), dtype="uint8")
    arr[10:30, 10:30, 0] = 255  # carré rouge
    p = tmp_path / "test.png"
    Image.fromarray(arr).save(p)
    m = LoadImageFile(path=str(p), size=0, mode="RGB")
    out = m()
    assert out.shape == (1, 3, 48, 48)
    assert out[0, 0, 15, 15].item() > 0.9  # rouge ≈ 1.0


def test_load_video_fallback(app):
    m = LoadVideo(path="/nonexistent/x.mp4", max_frames=10, resize=32)
    out = m()
    assert out.shape == (1, 10, 3, 32, 32)


def test_load_sound_fallback(app):
    m = LoadSound(path="", sample_rate=8000, max_duration=0.5, mono=True)
    out = m()
    assert out.shape == (1, 1, 4000)


def test_load_sound_real_wav(tmp_path, app):
    """Crée vraie WAV + charge via scipy fallback."""
    p = tmp_path / "tone.wav"
    sr = 8000
    duration = 0.25
    n = int(sr * duration)
    with wave.open(str(p), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        samples = [int(0.3 * 32767 * (i % 100 - 50) / 50) for i in range(n)]
        data = b"".join(struct.pack("<h", s) for s in samples)
        w.writeframes(data)
    m = LoadSound(path=str(p), sample_rate=sr, max_duration=0, mono=True)
    out = m()
    assert out.shape[0] == 1
    assert out.shape[1] == 1  # mono
    assert out.shape[2] > 0


def test_multi_image_loader_fallback(app):
    m = MultiImageLoader(path="/nonexistent_dir", pattern="*.png",
                              size=32, max_n=4)
    out = m()
    assert out.shape == (4, 3, 32, 32)


def test_multi_image_loader_real(tmp_path, app):
    """Crée 3 images PNG → charge batch."""
    from PIL import Image
    import numpy as np
    for i in range(3):
        arr = np.full((20, 20, 3), i * 80, dtype="uint8")
        Image.fromarray(arr).save(tmp_path / f"frame_{i:03d}.png")
    m = MultiImageLoader(path=str(tmp_path), pattern="*.png",
                              size=20, max_n=0, sort=True)
    out = m()
    assert out.shape == (3, 3, 20, 20)
    # Premier frame (i=0) → noir, dernier (i=2) → gris clair
    assert out[0].mean() < out[2].mean()


def test_handlers_registered(app):
    for name in ("load_image_file", "load_video", "load_sound",
                  "multi_image_loader"):
        assert name in HANDLERS, f"handler {name} absent"


def test_specs_registered(app):
    r = default_registry()
    for name in ("load_image_file", "load_video", "load_sound",
                  "multi_image_loader"):
        spec = r.get(name)
        assert spec is not None
        assert spec.category == "Input"
        # Vérifie qu'au moins 1 param est filepath/dirpath
        kinds = {p.kind for p in spec.params}
        assert "filepath" in kinds or "dirpath" in kinds


def test_load_image_spec_uses_filepath(app):
    """L'ancien load_image (Diffusion) utilise maintenant filepath."""
    r = default_registry()
    spec = r.get("load_image")
    p = next(p for p in spec.params if p.name == "path")
    assert p.kind == "filepath"
    assert "Images" in p.file_filter


def test_inspector_filepath_widget(app):
    """Inspector construit composite QLineEdit + button pour filepath."""
    r = default_registry()
    spec = r.get("load_video")
    panel = InspectorPanel()
    node = NodeItem(spec)
    panel.set_node(node)
    # Pas de crash + form contient widgets
    assert panel._form.rowCount() >= 1
