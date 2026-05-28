"""Tests ARK Phase 56 — Multimodal nodes (text2video, text2audio, img2text)."""
import os
import torch
import pytest

from ark.core.multimodal_nodes import (
    TextToFrameSeq, VideoWriter,
    TextToMelody, AudioSynth, AudioWriter,
    ImageNetClassifier, CaptionFormatter,
    register_multimodal_handlers, register_multimodal_specs,
)


# ============================================================
# Specs + handlers registration
# ============================================================

def test_specs_registered():
    from ark.nodes.registry import default_registry
    r = default_registry(force_rebuild=True)
    mm = [s for s in r.all() if s.category == "Multimodal"]
    names = {s.name for s in mm}
    assert names == {
        "text_to_frame_seq", "video_writer",
        "text_to_melody", "audio_synth", "audio_writer",
        "imagenet_classifier", "caption_formatter",
    }


def test_handlers_registered():
    from ark.core.graph_compiler import HANDLERS
    for name in ("text_to_frame_seq", "video_writer", "text_to_melody",
                 "audio_synth", "audio_writer", "imagenet_classifier",
                 "caption_formatter"):
        assert name in HANDLERS, f"Missing handler: {name}"


# ============================================================
# Text2Video
# ============================================================

def test_text_to_frame_seq_deterministic():
    """Same prompt → same frames (hash seeded)."""
    a = TextToFrameSeq(n_frames=4, height=32, width=32)
    b = TextToFrameSeq(n_frames=4, height=32, width=32)
    v1 = a("test prompt")
    v2 = b("test prompt")
    assert v1.shape == (4, 3, 32, 32)
    assert torch.allclose(v1, v2)


def test_text_to_frame_seq_different_prompts():
    tfs = TextToFrameSeq(n_frames=4, height=32, width=32)
    v1 = tfs("cat")
    v2 = tfs("dog")
    # Different prompts → different videos
    assert not torch.allclose(v1, v2)


def test_video_writer_gif(tmp_path):
    pytest.importorskip("imageio")
    tfs = TextToFrameSeq(n_frames=4, height=32, width=32)
    vid = tfs("test")
    out = tmp_path / "video.gif"
    vw = VideoWriter(output_path=str(out), fps=8)
    vw(vid)
    assert out.exists()
    assert out.stat().st_size > 100  # Real gif data


# ============================================================
# Text2Audio
# ============================================================

def test_text_to_melody():
    mel = TextToMelody(prompt="hello world", note_dur=0.2)
    out = mel(None)
    assert out.dim() == 2
    assert out.shape[1] == 2  # (freq, dur)
    assert out.shape[0] > 0  # at least 1 note
    # All freqs positive
    assert (out[:, 0] > 0).all()
    # All durations equal to note_dur
    assert torch.allclose(out[:, 1], torch.full((out.shape[0],), 0.2))


def test_audio_synth_shape():
    melody = torch.tensor([[440.0, 0.1], [880.0, 0.1]])
    syn = AudioSynth(sample_rate=8000, wave="sine")
    wave = syn(melody)
    assert wave.dim() == 2
    # 0.2 s × 8000 Hz = 1600 samples
    assert wave.shape[1] == 1600
    # Peak normalized
    assert wave.abs().max() <= 1.0


def test_audio_synth_envelope_types():
    melody = torch.tensor([[440.0, 0.1]])
    for env in ("adsr", "none"):
        syn = AudioSynth(sample_rate=8000, envelope=env)
        wave = syn(melody)
        assert wave.shape[1] == 800


def test_audio_synth_wave_types():
    melody = torch.tensor([[440.0, 0.05]])
    waves = {}
    for w in ("sine", "square", "triangle", "saw"):
        syn = AudioSynth(sample_rate=8000, wave=w, envelope="none")
        waves[w] = syn(melody)
    # Each wave type produces different output
    assert not torch.allclose(waves["sine"], waves["square"])


def test_audio_writer(tmp_path):
    pytest.importorskip("scipy")
    wave = torch.sin(torch.arange(8000).float() / 8000 * 2 * 3.14159 * 440)
    wave = wave.unsqueeze(0)
    out = tmp_path / "test.wav"
    aw = AudioWriter(output_path=str(out), sample_rate=8000)
    aw(wave)
    assert out.exists()
    assert out.stat().st_size > 100


# ============================================================
# Img2Text
# ============================================================

@pytest.mark.slow
def test_imagenet_classifier_smoke():
    """Pretrained ResNet18 download may take time, skip if no internet."""
    img = torch.rand(1, 3, 224, 224)
    clf = ImageNetClassifier(backbone="resnet18")
    try:
        logits = clf(img)
    except Exception as e:
        pytest.skip(f"resnet weights unavailable: {e}")
    assert logits.shape == (1, 1000)


def test_caption_formatter_no_labels():
    # No real ResNet, just fake logits
    logits = torch.zeros(1, 1000)
    logits[0, 42] = 5.0  # strong activation index 42
    cap = CaptionFormatter(k=3, template="default",
                              use_imagenet_labels=False)
    out = cap(logits)
    assert isinstance(out, str)
    assert "class_42" in out or "42" in out
    assert len(cap.last_predictions) == 3
    assert cap.last_predictions[0]["index"] == 42


def test_caption_formatter_templates():
    logits = torch.zeros(1, 100)
    logits[0, 5] = 3.0
    for tpl in ("default", "list", "json"):
        cap = CaptionFormatter(k=2, template=tpl, use_imagenet_labels=False)
        out = cap(logits)
        assert isinstance(out, str) and len(out) > 0


# ============================================================
# Templates
# ============================================================

def test_template_text2video_tiny():
    from ark.core.templates import template_text2video_tiny
    t = template_text2video_tiny()
    assert t["version"] == "ark-arch-v1"
    types = [n["type"] for n in t["nodes"]]
    assert "text_to_frame_seq" in types
    assert "video_writer" in types
    assert len(t["connections"]) >= 2


def test_template_text2audio_tiny():
    from ark.core.templates import template_text2audio_tiny
    t = template_text2audio_tiny()
    types = [n["type"] for n in t["nodes"]]
    assert "text_to_melody" in types
    assert "audio_synth" in types
    assert "audio_writer" in types


def test_template_img2text_resnet():
    from ark.core.templates import template_img2text_resnet
    t = template_img2text_resnet()
    types = [n["type"] for n in t["nodes"]]
    assert "imagenet_classifier" in types
    assert "caption_formatter" in types
    assert "text_viewer" in types


def test_template_img2img_style():
    from ark.core.templates import template_img2img_style
    t = template_img2img_style()
    types = [n["type"] for n in t["nodes"]]
    assert "load_image_file" in types
    assert types.count("conv2d") == 2
    assert "save_image" in types


def test_all_4_templates_in_builders():
    from ark.core.templates import _BUILDERS, TEMPLATES_LIST
    for name in ("text2video_tiny", "text2audio_tiny",
                 "img2text_resnet", "img2img_style"):
        assert name in _BUILDERS
        assert any(n == name for n, _desc in TEMPLATES_LIST)
