"""Pinned author architectures with project-specific BraTS I/O adapters.

Exact upstream source files and their licenses are bundled in third_party so
AutoDL training does not require network access to the authors' GitHub repos.
See docs/PAPER_BASELINES.md for provenance and comparison details.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import subprocess
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCES = {
    "doubleblock": (
        "https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment.git",
        "5b617d72608986f38e602f4dd69b95d45443f5c7",
        "models/DB_MaxViT.py",
    ),
    "superlight": (
        "https://github.com/WTU-MIS-Laboratory/SuperLightNet.git",
        "0d6532434586dd68750bf417b158511f46f4dd75",
        "Jnetworks/superlightnet.py",
    ),
}

BUNDLED_SOURCES = {
    "doubleblock": (
        "third_party/paper_baselines/doubleblock/DB_MaxViT.py",
        "1b9caf37be4b0c1b4fd00d16dd9ef4dfd71610bcf2e009c1bc4124aa5176c8b2",
    ),
    "superlight": (
        "third_party/paper_baselines/superlight/superlightnet.py",
        "b613d7e09f5795274e04bb71c66096fc9477aec03b80d3c570cbf9a5a278b303",
    ),
}


def ensure_source(name: str, source_dir: Path) -> Path:
    """Prefer the exact bundled source; never fetch during a training run."""
    relative_bundle, expected_sha256 = BUNDLED_SOURCES[name]
    bundled = Path(__file__).resolve().parents[1] / relative_bundle
    if bundled.is_file():
        actual_sha256 = hashlib.sha256(bundled.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{name} bundled source checksum mismatch: {bundled}")
        return bundled

    _, commit, relative_module = SOURCES[name]
    checkout = source_dir / name
    if not checkout.exists():
        raise FileNotFoundError(
            f"{name} bundled source missing: {bundled}; pull the latest codex/paper-baselines branch"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"{name} upstream commit mismatch: {actual} != {commit}")
    module_path = checkout / relative_module
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    expected_blob = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", f"HEAD:{relative_module}"], text=True
    ).strip()
    actual_blob = subprocess.check_output(
        ["git", "-C", str(checkout), "hash-object", str(module_path)], text=True
    ).strip()
    if actual_blob != expected_blob:
        raise RuntimeError(f"{name} upstream source file was modified: {module_path}")
    return module_path


def _load_doubleblock(path: Path):
    # The authors' module creates a CUDA model at import time on its last two
    # lines.  Remove only those top-level demo statements; all definitions and
    # weights remain exactly those in the pinned source.
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    if not (
        len(tree.body) >= 2
        and isinstance(tree.body[-2], ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model" for t in tree.body[-2].targets)
        and isinstance(tree.body[-1], ast.Expr)
        and isinstance(tree.body[-1].value, ast.Call)
        and isinstance(tree.body[-1].value.func, ast.Name)
        and tree.body[-1].value.func.id == "print"
    ):
        raise RuntimeError("DoubleBlock upstream module layout changed; inspect before use")
    tree.body = tree.body[:-2]
    module = types.ModuleType("pinned_doubleblock")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def _load_superlight(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_superlight", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperBaseline(nn.Module):
    """Expose author architecture as logits in project order [WT, TC, ET]."""

    def __init__(self, name: str, source_dir: Path):
        super().__init__()
        if name not in SOURCES:
            raise ValueError(name)
        self.name = name
        path = ensure_source(name, source_dir)
        if name == "doubleblock":
            module = _load_doubleblock(path)
            self.architecture = module.Unet(in_channels=4, n_classes=3, n_channels=16)
            self.divisor = 32  # three MaxViT stages, 4-voxel attention windows
        else:
            module = _load_superlight(path)
            self.architecture = module.NormalU_Net(
                init_channels=4, n_channels=24, class_nums=4,
                depths_unidirectional="small",
            )
            self.divisor = 16  # four 2x downsampling stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 4:
            raise ValueError(f"expected [B, 4, D, H, W], got {tuple(x.shape)}")
        original = x.shape[2:]
        padded = tuple(math.ceil(dim / self.divisor) * self.divisor for dim in original)
        # F.pad uses W,H,D order.  Pad at the high end so the project's fixed
        # crop origin and lesion coordinates are unchanged.
        x = F.pad(x, (0, padded[2]-original[2], 0, padded[1]-original[1],
                      0, padded[0]-original[0]))
        raw = self.architecture(x)
        raw = raw[..., :original[0], :original[1], :original[2]]
        if self.name == "doubleblock":
            # Authors: [ET, TC, WT].  Project: [WT, TC, ET].
            return raw[:, [2, 1, 0]]

        # Authors train independent sigmoid logits for labels 1, 2 and 4;
        # their background channel is not in the training objective.  OR of
        # independent Bernoulli probabilities has logit log(e^a+e^b+e^(a+b)).
        def union_logit(a, b):
            return torch.logsumexp(torch.stack((a, b, a + b), dim=0), dim=0)

        et = raw[:, 3]
        tc = union_logit(raw[:, 1], et)
        wt = union_logit(tc, raw[:, 2])
        return torch.stack((wt, tc, et), dim=1)
