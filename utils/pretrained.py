"""Require real ImageNet / torchvision checkpoints — never silent random init."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch


_TORCH_CKPT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

# Expected filenames for Weights.DEFAULT (torchvision)
REQUIRED_FILES = {
    "alexnet": "alexnet-owt-7be5be79.pth",
    "resnet50": "resnet50-11ad3fa6.pth",
    "resnet101": "resnet101-cd907fc2.pth",
    "resnet152": "resnet152-f82ba261.pth",
    "densenet169": "densenet169-b2777c0a.pth",
    "convnext_tiny": "convnext_tiny-983f1562.pth",
    "vit_b_16": "vit_b_16-c867db91.pth",
}


def assert_no_partial_downloads(cache_dir: Path | None = None) -> None:
    d = cache_dir or _TORCH_CKPT
    if not d.is_dir():
        return
    partials = list(d.glob("*.partial"))
    if partials:
        names = ", ".join(p.name for p in partials[:5])
        raise RuntimeError(
            f"Incomplete weight downloads in {d}: {names}. "
            "Re-run scripts/download_pretrained_weights.py and wait until finished."
        )


def require_checkpoint_file(backbone: str, cache_dir: Path | None = None) -> Path:
    assert_no_partial_downloads(cache_dir)
    d = cache_dir or _TORCH_CKPT
    name = REQUIRED_FILES.get(backbone)
    if not name:
        raise KeyError(f"Unknown backbone key for checkpoint check: {backbone}")
    path = d / name
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            f"Missing ImageNet weights for {backbone}: {path}\n"
            "Run: python -u scripts/download_pretrained_weights.py"
        )
    return path


def load_torchvision_with_weights(builder: Callable, weights, backbone_key: str):
    """Call torchvision builder(weights=...) after verifying the file is on disk."""
    require_checkpoint_file(backbone_key)
    if weights is None:
        raise ValueError(f"Refusing random init for {backbone_key}: weights must be DEFAULT")
    model = builder(weights=weights)
    return model
