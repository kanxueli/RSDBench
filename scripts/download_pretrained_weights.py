#!/usr/bin/env python
"""Force-download ImageNet / torchvision weights used by RSDBench encoders.

Hard rule: every backbone must resolve a real weight file before any experiment.
No random-init shortcuts for feature extraction or model training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.models import (
    AlexNet_Weights,
    ConvNeXt_Tiny_Weights,
    DenseNet169_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    alexnet,
    convnext_tiny,
    densenet169,
    resnet101,
    resnet152,
    resnet50,
    vit_b_16,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.pretrained import REQUIRED_FILES, assert_no_partial_downloads, require_checkpoint_file


def main():
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    # Drop incomplete downloads so torchvision retries cleanly
    for p in cache.glob("*.partial"):
        print(f"[weights] removing partial {p.name}", flush=True)
        p.unlink()

    print(f"[weights] torch cache → {cache}")
    specs = [
        ("alexnet", lambda: alexnet(weights=AlexNet_Weights.DEFAULT)),
        ("resnet50", lambda: resnet50(weights=ResNet50_Weights.DEFAULT)),
        ("resnet101", lambda: resnet101(weights=ResNet101_Weights.DEFAULT)),
        ("resnet152", lambda: resnet152(weights=ResNet152_Weights.DEFAULT)),
        ("densenet169", lambda: densenet169(weights=DenseNet169_Weights.DEFAULT)),
        ("convnext_tiny", lambda: convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)),
        ("vit_b_16", lambda: vit_b_16(weights=ViT_B_16_Weights.DEFAULT)),
    ]
    for name, fn in specs:
        print(f"[weights] downloading/loading {name} ...", flush=True)
        m = fn()
        n = sum(p.numel() for p in m.parameters())
        print(f"  ok {name}: {n/1e6:.1f}M params")
        del m
        require_checkpoint_file(name, cache)

    assert_no_partial_downloads(cache)
    for key, fname in REQUIRED_FILES.items():
        p = cache / fname
        print(f"  verified {key}: {p} ({p.stat().st_size/1e6:.1f} MB)")

    src = Path("pretrained/torchvision_vit_b16_in1k.pth")
    dst = ROOT / ".cache" / "pretrained" / "torchvision_vit_b16_in1k.pth"
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            import shutil

            shutil.copy2(src, dst)
            print(f"[weights] copied SWAG ViT → {dst}")
    print("[weights] all required ImageNet checkpoints available — safe to train")


if __name__ == "__main__":
    main()
