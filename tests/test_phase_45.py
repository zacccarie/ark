"""Tests Phase 45 — Dataset nodes."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
from types import SimpleNamespace

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.graph_compiler import HANDLERS
from ark.core.dataset_nodes import (
    TorchvisionDataset, ImageNetLocal, COCODetection,
    AudioDataset, VideoDataset, Word2VecEmbeddings,
    HFDataset, SyntheticDataset,
)


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Synthetic (toujours dispo) ============

def test_synthetic_moons(app):
    m = SyntheticDataset(kind="moons", n_samples=200, noise=0.1)
    out = m()
    assert out.shape == (200, 2)
    assert m.last_result.ok


def test_synthetic_circles(app):
    m = SyntheticDataset(kind="circles", n_samples=100, noise=0.05)
    out = m()
    assert out.shape == (100, 2)


def test_synthetic_blobs(app):
    m = SyntheticDataset(kind="blobs", n_samples=300, n_classes=4,
                              n_features=3)
    out = m()
    assert out.shape == (300, 3)
    assert m.last_result.n_classes == 4


def test_synthetic_classification(app):
    m = SyntheticDataset(kind="classification", n_samples=500,
                              n_features=10, n_classes=5)
    out = m()
    assert out.shape == (500, 10)


# ============ Fallback paths (no real download in tests) ============

def test_imagenet_local_fallback(app):
    """Path absent → fallback noise."""
    m = ImageNetLocal(root="/nonexistent_imagenet", size=64,
                          batch_preview=4)
    out = m()
    assert out.shape == (4, 3, 64, 64)
    assert not m.last_result.ok


def test_coco_fallback(app):
    m = COCODetection(root="/nonexistent_coco", ann_file="/nonexistent.json",
                          size=128, batch_preview=2)
    out = m()
    assert out.shape == (2, 3, 128, 128)


def test_audio_dataset_fallback(app):
    """YESNO peut être absent → fallback waveform."""
    m = AudioDataset(name="YESNO", root="/nonexistent_audio",
                          download=False)
    out = m()
    # Soit waveform réel, soit fallback (1, 16000)
    assert out.dim() >= 1


def test_video_fallback(app):
    m = VideoDataset(name="UCF101", root="/nonexistent",
                          frames_per_clip=8, batch_preview=2)
    out = m()
    assert out.shape == (2, 8, 3, 112, 112)


def test_word2vec_fallback(app):
    """gensim peut être absent → fallback random (6, 100)."""
    m = Word2VecEmbeddings(model_name="glove-twitter-25",
                                query_words="hello,world")
    out = m()
    assert out.dim() == 2
    assert out.shape[0] >= 2


def test_hf_dataset_fallback(app):
    m = HFDataset(name="nonexistent_dataset_xyz", split="train[:5]")
    out = m()
    # Fallback : list[str]
    assert isinstance(out, list)


# ============ Registration ============

def test_all_handlers_registered(app):
    for name in ("torchvision_dataset", "imagenet_local", "coco_detection",
                  "audio_dataset", "video_dataset", "word2vec_embeddings",
                  "hf_dataset", "synthetic_dataset"):
        assert name in HANDLERS, f"handler {name} absent"


def test_all_specs_registered(app):
    r = default_registry()
    for name in ("torchvision_dataset", "imagenet_local", "coco_detection",
                  "audio_dataset", "video_dataset", "word2vec_embeddings",
                  "hf_dataset", "synthetic_dataset"):
        spec = r.get(name)
        assert spec is not None
        assert spec.category == "Datasets"


def test_torchvision_spec_options(app):
    """Vérifie liste choice options pour torchvision."""
    r = default_registry()
    spec = r.get("torchvision_dataset")
    name_param = next(p for p in spec.params if p.name == "name")
    assert "MNIST" in name_param.options
    assert "CIFAR10" in name_param.options
    assert "Caltech101" in name_param.options


def test_audio_spec_options(app):
    r = default_registry()
    spec = r.get("audio_dataset")
    name_param = next(p for p in spec.params if p.name == "name")
    assert "SPEECHCOMMANDS" in name_param.options
    assert "LIBRISPEECH" in name_param.options


def test_w2v_spec_options(app):
    r = default_registry()
    spec = r.get("word2vec_embeddings")
    name_param = next(p for p in spec.params if p.name == "model_name")
    assert "glove-wiki-gigaword-100" in name_param.options
    assert "word2vec-google-news-300" in name_param.options


def test_handler_dispatch_synthetic(app):
    node = SimpleNamespace(id="x", params={
        "kind": "moons", "n_samples": 100, "noise": 0.1,
        "n_features": 2, "n_classes": 2, "random_state": 42,
    })
    m, shape = HANDLERS["synthetic_dataset"](node, [])
    out = m()
    assert out.shape == (100, 2)
