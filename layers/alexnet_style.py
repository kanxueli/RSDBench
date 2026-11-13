"""AlexNet-style visual encoder for Rivoir 2019 / 2020."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torchvision.models import alexnet


_CONV_SLICES = {
    "conv1": (0, 3),
    "conv2": (3, 6),
    "conv3": (6, 8),
    "conv4": (8, 10),
    "conv5": (10, 13),
}


class AlexNetEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = False,
        imagenet_norm: bool = True,
    ):
        super().__init__()
        if pretrained:
            from torchvision.models import AlexNet_Weights

            net = alexnet(weights=AlexNet_Weights.DEFAULT)
        else:
            net = alexnet(weights=None)
        self.features = net.features
        self.avgpool = net.avgpool
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim
        self.imagenet_norm = bool(imagenet_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] → (B, embed_dim)."""
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if self.imagenet_norm:
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
        h = self.avgpool(self.features(x))
        h = torch.flatten(h, 1)
        return self.fc(self.dropout(h))

    def named_blocks(self) -> Dict[str, List[nn.Parameter]]:
        blocks: Dict[str, List[nn.Parameter]] = {}
        for name, (a, b) in _CONV_SLICES.items():
            params: List[nn.Parameter] = []
            for i in range(a, b):
                params.extend(list(self.features[i].parameters()))
            blocks[name] = params
        blocks["fc1"] = list(self.fc.parameters())
        return blocks

    def set_requires_grad(self, trainable: Iterable[str], freeze_rest: bool = True) -> None:
        train_set = set(trainable)
        if freeze_rest:
            for p in self.parameters():
                p.requires_grad = False
        for name, params in self.named_blocks().items():
            if name in train_set:
                for p in params:
                    p.requires_grad = True
