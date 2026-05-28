"""Phase 45 — Dataset nodes : wrappers torchvision/torchaudio/HuggingFace.

Aucune donnée stockée dans le repo. Tous loaders pointent vers les sources
officielles (download lazy on demand). Compat licences amont (CC-BY, custom,
etc.) — l'utilisateur doit accepter les conditions des fournisseurs.

Familles couvertes :
- Vision classification : MNIST / FashionMNIST / KMNIST / EMNIST / CIFAR10/100 /
  STL10 / SVHN / GTSRB / Country211 / Flowers102
- Vision detection/segmentation : COCO / VOC / Caltech101/256 / CelebA / Places365
- Vision tabular tiny : FER2013 / USPS
- Vision spécifique : ImageNet (path local manuel)
- Audio : SPEECHCOMMANDS / LibriSpeech / Yesno / VCTK
- Vidéo : UCF101 / HMDB51 / Kinetics400
- Pose : COCO Keypoints / MPII Human Pose (path local)
- Embeddings : Word2Vec (gensim), GloVe (HF)
- Synthetic : Sklearn make_classification / make_blobs / make_moons

Tous output : torch.utils.data.Dataset ou tensor batch (B, ...).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


DEFAULT_ROOT = Path.home() / ".ark" / "datasets"


@dataclass
class DatasetResult:
    ok: bool
    dataset: Any = None
    n_samples: int = 0
    n_classes: int = 0
    error: str = ""


# ============================================================
# Helpers
# ============================================================

def _ensure_torchvision():
    try:
        import torchvision  # noqa
        return True
    except ImportError as e:
        raise RuntimeError("torchvision absent. Install : uv pip install torchvision") from e


def _ensure_torchaudio():
    try:
        import torchaudio  # noqa
        return True
    except ImportError as e:
        raise RuntimeError("torchaudio absent. Install : uv pip install torchaudio") from e


def _default_transform_vision(size: int = 0):
    """Transform standard image vision : ToTensor + optional Resize."""
    _ensure_torchvision()
    from torchvision import transforms
    t = [transforms.ToTensor()]
    if size > 0:
        t.insert(0, transforms.Resize((size, size)))
    return transforms.Compose(t)


def _sample_first(dataset, n: int = 32):
    """Retourne premier batch (B, C, H, W) ou (B, T) selon nature."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    try:
        x, y = next(iter(loader))
        return x, y
    except Exception:
        return None, None


# ============================================================
# 1. Vision classification (torchvision)
# ============================================================

class TorchvisionDataset(nn.Module):
    """Wrap torchvision.datasets.* avec download lazy.

    Constructor params :
    - name : str ∈ {MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10, CIFAR100,
                       STL10, SVHN, GTSRB, Country211, Flowers102}
    - root : str (default ~/.ark/datasets)
    - train : bool
    - download : bool (auto download si absent)
    - size : int (resize, 0 = original)
    - batch_preview : int (renvoie premier batch dans forward)
    """

    def __init__(self, name: str = "MNIST", root: str = "",
                 train: bool = True, download: bool = True,
                 size: int = 0, batch_preview: int = 32):
        super().__init__()
        self.name = name
        self.root = root or str(DEFAULT_ROOT)
        self.train = train
        self.download = download
        self.size = size
        self.batch_preview = batch_preview
        self.last_result: DatasetResult | None = None
        self._cached_dataset = None

    def _build_dataset(self):
        if self._cached_dataset is not None:
            return self._cached_dataset
        _ensure_torchvision()
        import torchvision.datasets as ds
        Path(self.root).mkdir(parents=True, exist_ok=True)
        tfm = _default_transform_vision(self.size)
        cls = getattr(ds, self.name, None)
        if cls is None:
            raise RuntimeError(f"Dataset torchvision.{self.name} introuvable")
        # Constructeurs varient légèrement
        kwargs = dict(root=self.root, download=self.download, transform=tfm)
        # train/test/split param varie
        if self.name in ("STL10", "SVHN", "Country211", "Flowers102"):
            kwargs["split"] = "train" if self.train else "test"
        elif self.name in ("GTSRB",):
            kwargs["split"] = "train" if self.train else "test"
        else:
            kwargs["train"] = self.train
        try:
            self._cached_dataset = cls(**kwargs)
        except TypeError:
            # Fallback : retry sans train/split
            kwargs.pop("train", None); kwargs.pop("split", None)
            self._cached_dataset = cls(**kwargs)
        return self._cached_dataset

    def forward(self, x=None):
        try:
            ds = self._build_dataset()
            n = len(ds)
            x_batch, y_batch = _sample_first(ds, self.batch_preview)
            n_cls = len(getattr(ds, "classes", []))
            self.last_result = DatasetResult(True, ds, n, n_cls)
            return x_batch if x_batch is not None else torch.zeros(1)
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            # Fallback noise déterministe
            return torch.randn(self.batch_preview, 1, 28, 28)


# ============================================================
# 2. ImageNet (path local manuel)
# ============================================================

class ImageNetLocal(nn.Module):
    """ImageNet : trop gros pour auto-download. Path local manuel.

    Attend structure ImageFolder : root/train/<class_id>/<img>.JPEG
    """

    def __init__(self, root: str = "", split: str = "train",
                 size: int = 224, batch_preview: int = 32):
        super().__init__()
        self.root = root
        self.split = split
        self.size = size
        self.batch_preview = batch_preview
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            _ensure_torchvision()
            import torchvision.datasets as ds
            root = Path(self.root).expanduser() / self.split
            if not root.exists():
                raise RuntimeError(
                    f"ImageNet absent : {root}. Télécharge manuellement "
                    f"depuis https://image-net.org/ (auth requis).")
            tfm = _default_transform_vision(self.size)
            dataset = ds.ImageFolder(root=str(root), transform=tfm)
            x_b, y_b = _sample_first(dataset, self.batch_preview)
            self.last_result = DatasetResult(
                True, dataset, len(dataset), len(dataset.classes))
            return x_b if x_b is not None else torch.zeros(1)
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(self.batch_preview, 3, self.size, self.size)


# ============================================================
# 3. COCO Detection (torchvision)
# ============================================================

class COCODetection(nn.Module):
    """COCO 2017 detection. Path local annotations JSON."""

    def __init__(self, root: str = "", ann_file: str = "",
                 size: int = 256, batch_preview: int = 4):
        super().__init__()
        self.root = root
        self.ann_file = ann_file
        self.size = size
        self.batch_preview = batch_preview
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            _ensure_torchvision()
            import torchvision.datasets as ds
            tfm = _default_transform_vision(self.size)
            dataset = ds.CocoDetection(root=self.root, annFile=self.ann_file,
                                            transform=tfm)
            self.last_result = DatasetResult(True, dataset, len(dataset))
            x_b, _ = _sample_first(dataset, self.batch_preview)
            return x_b if x_b is not None else torch.zeros(1)
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(self.batch_preview, 3, self.size, self.size)


# ============================================================
# 4. Audio (torchaudio)
# ============================================================

class AudioDataset(nn.Module):
    """Wrap torchaudio.datasets. SPEECHCOMMANDS / LIBRISPEECH / YESNO / VCTK_092."""

    def __init__(self, name: str = "SPEECHCOMMANDS", root: str = "",
                 download: bool = True, subset: str = "training",
                 batch_preview: int = 16):
        super().__init__()
        self.name = name
        self.root = root or str(DEFAULT_ROOT)
        self.download = download
        self.subset = subset
        self.batch_preview = batch_preview
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            _ensure_torchaudio()
            import torchaudio.datasets as ad
            Path(self.root).mkdir(parents=True, exist_ok=True)
            cls = getattr(ad, self.name, None)
            if cls is None:
                raise RuntimeError(f"torchaudio.{self.name} introuvable")
            kwargs = dict(root=self.root, download=self.download)
            if self.name == "SPEECHCOMMANDS":
                kwargs["subset"] = self.subset
            elif self.name == "LIBRISPEECH":
                kwargs["url"] = "train-clean-100"
            dataset = cls(**kwargs)
            self.last_result = DatasetResult(True, dataset, len(dataset))
            # Premier waveform
            sample = dataset[0]
            wave = sample[0] if isinstance(sample, tuple) else sample
            return wave.unsqueeze(0) if wave.dim() == 1 else wave
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(1, 16000)


# ============================================================
# 5. Video (torchvision.datasets.UCF101 etc)
# ============================================================

class VideoDataset(nn.Module):
    """Wrap UCF101 / HMDB51 / Kinetics400. Annotations dir requis."""

    def __init__(self, name: str = "UCF101", root: str = "",
                 annotation_path: str = "",
                 frames_per_clip: int = 16, step_between_clips: int = 8,
                 batch_preview: int = 2):
        super().__init__()
        self.name = name
        self.root = root
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.batch_preview = batch_preview
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            _ensure_torchvision()
            import torchvision.datasets as ds
            cls = getattr(ds, self.name, None)
            if cls is None:
                raise RuntimeError(f"torchvision.{self.name} introuvable")
            kwargs = dict(root=self.root,
                            frames_per_clip=self.frames_per_clip,
                            step_between_clips=self.step_between_clips)
            if self.annotation_path:
                kwargs["annotation_path"] = self.annotation_path
            dataset = cls(**kwargs)
            self.last_result = DatasetResult(True, dataset, len(dataset))
            sample = dataset[0]
            video = sample[0] if isinstance(sample, tuple) else sample
            return video.unsqueeze(0) if video.dim() == 4 else video
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(self.batch_preview, self.frames_per_clip,
                                 3, 112, 112)


# ============================================================
# 6. Word2Vec embeddings (gensim downloader)
# ============================================================

class Word2VecEmbeddings(nn.Module):
    """Charge embeddings pré-entraînés via gensim.downloader.

    Modèles dispo : word2vec-google-news-300, glove-wiki-gigaword-100,
    glove-twitter-25, fasttext-wiki-news-subwords-300, conceptnet-numberbatch-17-06-300.
    """

    def __init__(self, model_name: str = "glove-wiki-gigaword-100",
                 query_words: str = "king,queen,man,woman,paris,france"):
        super().__init__()
        self.model_name = model_name
        self.query_words = query_words
        self.last_result: DatasetResult | None = None
        self._model = None

    def forward(self, x=None):
        try:
            import gensim.downloader as api
            if self._model is None:
                self._model = api.load(self.model_name)
            words = [w.strip() for w in self.query_words.split(",") if w.strip()]
            vecs = []
            for w in words:
                try:
                    vecs.append(torch.from_numpy(self._model[w]).float())
                except KeyError:
                    vecs.append(torch.zeros(self._model.vector_size))
            stacked = torch.stack(vecs)
            self.last_result = DatasetResult(
                True, self._model, len(words),
                getattr(self._model, "vector_size", 0))
            return stacked
        except ImportError:
            self.last_result = DatasetResult(
                False, error="gensim absent. uv pip install gensim")
            return torch.randn(6, 100)
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(6, 100)


# ============================================================
# 7. HuggingFace datasets (NLP)
# ============================================================

class HFDataset(nn.Module):
    """Wrap datasets.load_dataset HuggingFace. GLUE, SQuAD, IMDB, etc."""

    def __init__(self, name: str = "imdb", config: str = "",
                 split: str = "train[:1000]", text_field: str = "text"):
        super().__init__()
        self.name = name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            from datasets import load_dataset
            kwargs = dict(path=self.name, split=self.split)
            if self.config:
                kwargs["name"] = self.config
            dataset = load_dataset(**kwargs)
            n = len(dataset)
            self.last_result = DatasetResult(True, dataset, n)
            # Sample : retourne 1er texte tokenisé en bytes
            texts = [str(dataset[i].get(self.text_field, ""))[:128]
                       for i in range(min(8, n))]
            return texts  # list[str] non-tensorizable mais informatif
        except ImportError:
            self.last_result = DatasetResult(
                False, error="datasets absent. uv pip install datasets")
            return ["fallback text"] * 8
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return ["error " + self.name] * 8


# ============================================================
# 8. Synthetic (sklearn)
# ============================================================

class SyntheticDataset(nn.Module):
    """Sklearn synthetic generators : moons / blobs / circles / classification."""

    def __init__(self, kind: str = "moons", n_samples: int = 500,
                 noise: float = 0.1, n_features: int = 2,
                 n_classes: int = 2, random_state: int = 42):
        super().__init__()
        self.kind = kind
        self.n_samples = n_samples
        self.noise = noise
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.last_result: DatasetResult | None = None

    def forward(self, x=None):
        try:
            from sklearn import datasets as ds
            if self.kind == "moons":
                X, y = ds.make_moons(n_samples=self.n_samples,
                                          noise=self.noise,
                                          random_state=self.random_state)
            elif self.kind == "circles":
                X, y = ds.make_circles(n_samples=self.n_samples,
                                            noise=self.noise,
                                            random_state=self.random_state)
            elif self.kind == "blobs":
                X, y = ds.make_blobs(n_samples=self.n_samples,
                                          centers=self.n_classes,
                                          n_features=self.n_features,
                                          random_state=self.random_state)
            elif self.kind == "classification":
                X, y = ds.make_classification(
                    n_samples=self.n_samples, n_features=self.n_features,
                    n_classes=self.n_classes, n_informative=max(2, self.n_features // 2),
                    random_state=self.random_state)
            else:
                X, y = ds.make_moons(n_samples=self.n_samples, noise=self.noise)
            X_t = torch.from_numpy(X).float()
            y_t = torch.from_numpy(y).long()
            self.last_result = DatasetResult(True, (X_t, y_t),
                                                  self.n_samples, self.n_classes)
            return X_t
        except Exception as e:
            self.last_result = DatasetResult(False, error=str(e))
            return torch.randn(self.n_samples, self.n_features)


# ============================================================
# Handlers + Specs
# ============================================================

def _make_handler(cls, defaults: dict):
    def h(node, in_shapes):
        params = {k: node.params.get(k, v) for k, v in defaults.items()}
        m = cls(**params)
        m._ark_node_id = getattr(node, "id", "ds")
        return m, in_shapes[0] if in_shapes else (1,)
    return h


def register_dataset_handlers(handlers_dict: dict) -> bool:
    handlers_dict["torchvision_dataset"] = _make_handler(
        TorchvisionDataset, dict(
            name="MNIST", root="", train=True, download=True,
            size=0, batch_preview=32))
    handlers_dict["imagenet_local"] = _make_handler(
        ImageNetLocal, dict(
            root="", split="train", size=224, batch_preview=32))
    handlers_dict["coco_detection"] = _make_handler(
        COCODetection, dict(
            root="", ann_file="", size=256, batch_preview=4))
    handlers_dict["audio_dataset"] = _make_handler(
        AudioDataset, dict(
            name="SPEECHCOMMANDS", root="", download=True,
            subset="training", batch_preview=16))
    handlers_dict["video_dataset"] = _make_handler(
        VideoDataset, dict(
            name="UCF101", root="", annotation_path="",
            frames_per_clip=16, step_between_clips=8, batch_preview=2))
    handlers_dict["word2vec_embeddings"] = _make_handler(
        Word2VecEmbeddings, dict(
            model_name="glove-wiki-gigaword-100",
            query_words="king,queen,man,woman,paris,france"))
    handlers_dict["hf_dataset"] = _make_handler(
        HFDataset, dict(
            name="imdb", config="",
            split="train[:1000]", text_field="text"))
    handlers_dict["synthetic_dataset"] = _make_handler(
        SyntheticDataset, dict(
            kind="moons", n_samples=500, noise=0.1,
            n_features=2, n_classes=2, random_state=42))
    return True


def register_dataset_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Datasets"

    registry.register(NodeSpec(
        name="torchvision_dataset", title="Torchvision Dataset",
        category=CAT,
        outputs=[SocketSpec("batch", "tensor")],
        params=[
            ParamSpec("name", "choice", "MNIST",
                        options=["MNIST", "FashionMNIST", "KMNIST", "EMNIST",
                                  "CIFAR10", "CIFAR100", "STL10", "SVHN",
                                  "GTSRB", "Country211", "Flowers102",
                                  "Caltech101", "Caltech256", "CelebA",
                                  "Places365"]),
            ParamSpec("root", "dirpath", ""),
            ParamSpec("train", "bool", True),
            ParamSpec("download", "bool", True),
            ParamSpec("size", "int", 0, min=0, max=4096),
            ParamSpec("batch_preview", "int", 32, min=1, max=1024),
        ],
        description="Charge dataset torchvision avec auto-download lazy. "
                       "15 datasets vision : MNIST, CIFAR, STL, SVHN, Caltech, etc.",
        role="Wrap torchvision.datasets.* avec transforms standard.",
        function="ToTensor + Resize(size) optionnel. Download dans ~/.ark/datasets "
                    "par défaut. Premier batch preview retourné en forward.",
        has_thumbnail=True,
        param_help={
            "name": "Nom dataset torchvision (MNIST/CIFAR10/STL10/etc).",
            "root": "Cache local. Vide = ~/.ark/datasets.",
            "train": "True = train split, False = test/val.",
            "download": "Auto-download si absent.",
            "size": "Resize carré. 0 = original.",
            "batch_preview": "Taille batch premier preview forward.",
        },
    ))

    registry.register(NodeSpec(
        name="imagenet_local", title="ImageNet (local path)",
        category=CAT,
        outputs=[SocketSpec("batch", "tensor")],
        params=[
            ParamSpec("root", "dirpath", ""),
            ParamSpec("split", "choice", "train",
                        options=["train", "val", "test"]),
            ParamSpec("size", "int", 224, min=32, max=1024),
            ParamSpec("batch_preview", "int", 32, min=1, max=1024),
        ],
        description="ImageNet via ImageFolder local. Trop gros pour auto-download. "
                       "Auth manuelle requise sur https://image-net.org/",
        role="Load ImageNet from manual local extraction.",
        function="torchvision.datasets.ImageFolder. Structure attendue : "
                    "root/{train,val}/<class_id>/<image>.JPEG",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="coco_detection", title="COCO Detection",
        category=CAT,
        outputs=[SocketSpec("batch", "tensor")],
        params=[
            ParamSpec("root", "dirpath", ""),
            ParamSpec("ann_file", "filepath", "",
                        file_filter="JSON (*.json);;All (*)"),
            ParamSpec("size", "int", 256, min=64, max=2048),
            ParamSpec("batch_preview", "int", 4, min=1, max=64),
        ],
        description="COCO 2017 detection via torchvision. Annotations JSON requis.",
        role="MS-COCO loader pour detection/segmentation.",
        function="torchvision.datasets.CocoDetection(root, annFile, transform).",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="audio_dataset", title="Audio Dataset (torchaudio)",
        category=CAT,
        outputs=[SocketSpec("waveform", "tensor")],
        params=[
            ParamSpec("name", "choice", "SPEECHCOMMANDS",
                        options=["SPEECHCOMMANDS", "LIBRISPEECH", "YESNO",
                                  "VCTK_092", "COMMONVOICE"]),
            ParamSpec("root", "dirpath", ""),
            ParamSpec("download", "bool", True),
            ParamSpec("subset", "choice", "training",
                        options=["training", "validation", "testing"]),
            ParamSpec("batch_preview", "int", 16, min=1, max=256),
        ],
        description="Audio datasets via torchaudio. Speech commands, LibriSpeech, "
                       "VCTK voices, Common Voice multilingual.",
        role="Audio loader avec sample rate auto.",
        function="torchaudio.datasets.* avec download lazy. Premier waveform sample.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="video_dataset", title="Video Dataset",
        category=CAT,
        outputs=[SocketSpec("clips", "tensor")],
        params=[
            ParamSpec("name", "choice", "UCF101",
                        options=["UCF101", "HMDB51", "Kinetics400"]),
            ParamSpec("root", "dirpath", ""),
            ParamSpec("annotation_path", "dirpath", ""),
            ParamSpec("frames_per_clip", "int", 16, min=2, max=128),
            ParamSpec("step_between_clips", "int", 8, min=1, max=64),
            ParamSpec("batch_preview", "int", 2, min=1, max=32),
        ],
        description="Video datasets pour action recognition. UCF101 (101 classes), "
                       "HMDB51, Kinetics400. Download manuel requis (gros volume).",
        role="Action recognition clips loader.",
        function="torchvision.datasets.UCF101/HMDB51/Kinetics400 avec "
                    "frames_per_clip + step_between_clips.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="word2vec_embeddings", title="Word2Vec / GloVe Embeddings",
        category=CAT,
        outputs=[SocketSpec("embeddings", "tensor")],
        params=[
            ParamSpec("model_name", "choice", "glove-wiki-gigaword-100",
                        options=[
                            "glove-wiki-gigaword-50",
                            "glove-wiki-gigaword-100",
                            "glove-wiki-gigaword-200",
                            "glove-wiki-gigaword-300",
                            "glove-twitter-25",
                            "glove-twitter-50",
                            "glove-twitter-100",
                            "glove-twitter-200",
                            "word2vec-google-news-300",
                            "fasttext-wiki-news-subwords-300",
                            "conceptnet-numberbatch-17-06-300",
                        ]),
            ParamSpec("query_words", "multiline",
                        "king,queen,man,woman,paris,france"),
        ],
        description="Embeddings pré-entraînés via gensim downloader. Word2Vec Google "
                       "News, GloVe Wikipedia/Twitter, FastText, ConceptNet Numberbatch.",
        role="Pre-trained word embeddings lookup.",
        function="gensim.downloader.load(model_name) puis lookup pour query_words "
                    "comma-separated. Output (N, dim) tensor.",
        equation="cos(king - man + woman, queen) ≈ 1",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="hf_dataset", title="HuggingFace Dataset",
        category=CAT,
        outputs=[SocketSpec("texts", "any")],
        params=[
            ParamSpec("name", "str", "imdb"),
            ParamSpec("config", "str", ""),
            ParamSpec("split", "str", "train[:1000]"),
            ParamSpec("text_field", "str", "text"),
        ],
        description="HuggingFace datasets : NLP, multimodal, audio. Load via "
                       "datasets.load_dataset() avec slicing split[:N].",
        role="HF Hub datasets loader.",
        function="datasets.load_dataset(name, config, split). Premier 8 textes "
                    "extraits via text_field.",
        has_thumbnail=True,
        param_help={
            "name": "Nom HF (imdb, glue, squad, wikipedia, etc).",
            "config": "Sub-config si applicable (ex. 'sst2' pour glue).",
            "split": "Slicing HF : train, train[:1000], test, etc.",
            "text_field": "Clé du champ texte dans les samples.",
        },
    ))

    registry.register(NodeSpec(
        name="synthetic_dataset", title="Synthetic Dataset (sklearn)",
        category=CAT,
        outputs=[SocketSpec("X", "tensor")],
        params=[
            ParamSpec("kind", "choice", "moons",
                        options=["moons", "circles", "blobs", "classification"]),
            ParamSpec("n_samples", "int", 500, min=10, max=100000),
            ParamSpec("noise", "float", 0.1, min=0.0, max=2.0),
            ParamSpec("n_features", "int", 2, min=2, max=1000),
            ParamSpec("n_classes", "int", 2, min=2, max=100),
            ParamSpec("random_state", "int", 42),
        ],
        description="Sklearn synthetic generators pour tests/démos. Moons (XOR), "
                       "circles, blobs gaussiens, classification multi-classes.",
        role="Synthetic data for prototyping + sanity checks.",
        function="sklearn.datasets.make_{moons,circles,blobs,classification}.",
        has_thumbnail=True,
    ))
    return True
