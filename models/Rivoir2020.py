"""Rivoir 2020."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class _LockedDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or (not self.training and not getattr(self, "_mc", False)):
            return x
        if x.dim() == 2:
            mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - self.p) / (1 - self.p)
            return x * mask
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask

class _ResNetEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Linear(512, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)

class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.embed_dim = int(m.get("embed_dim", 256))
        self.hidden = int(m.get("hidden_size", 512))
        self.dropout_p = float(m.get("dropout", 0.2))
        self.mc_samples = int(m.get("mc_samples", 10))
        self.n_phases = int(m.get("n_phases", 7))
        self.cnn_batch = int(m.get("cnn_batch", 192))
        self.max_len = float(m.get("max_len", 100.0))
        self.cnn = _ResNetEncoder(
            embed_dim=self.embed_dim,
            pretrained=bool(m.get("pretrained_cnn", True)),
        )
        self.lock_drop = _LockedDropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + 2, self.hidden, num_layers=1, batch_first=True)
        self.head_rsd = nn.Linear(self.hidden, 1)
        self.head_cls = nn.Linear(self.hidden, self.n_phases)
        self.head_prog = nn.Linear(self.hidden, 1)
        self.last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        outs = []
        bs = max(1, self.cnn_batch)
        for i in range(0, flat.size(0), bs):
            outs.append(self.cnn(flat[i : i + bs]))
        feat = torch.cat(outs, 0).view(b, t, -1)
        return feat.squeeze(0) if squeeze else feat

    def _elapsed_channels(self, elapsed: torch.Tensor) -> torch.Tensor:
        raw = elapsed.unsqueeze(-1)
        norm = (elapsed / max(self.max_len, 1e-6)).clamp(0.0, 2.0).unsqueeze(-1)
        return torch.cat([raw, norm], dim=-1)

    def _forward_once(self, feat: torch.Tensor, elapsed: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        feat = self.lock_drop(feat)
        x = torch.cat([feat, self._elapsed_channels(elapsed)], dim=-1)
        out, _ = self.lstm(x)
        out = self.lock_drop(out)
        rsd = F.softplus(self.head_rsd(out).squeeze(-1))
        logits = self.head_cls(out)
        progress = torch.sigmoid(self.head_prog(out).squeeze(-1))
        if squeeze:
            rsd, logits, progress = rsd.squeeze(0), logits.squeeze(0), progress.squeeze(0)
        return {"rsd": rsd, "phase_logits": logits, "progress": progress}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        single = frames.dim() == 4
        feat = self._encode(frames)
        if single and feat.dim() == 3:
            feat = feat.squeeze(0)

        do_mc = (not self.training) and self.mc_samples > 1
        if not do_mc:
            self.lock_drop._mc = self.training
            return self._forward_once(feat, elapsed)

        self.lock_drop._mc = True
        preds, progs = [], []
        logits_acc = None
        for _ in range(self.mc_samples):
            out = self._forward_once(feat, elapsed)
            preds.append(out["rsd"])
            progs.append(out["progress"])
            logits_acc = out["phase_logits"] if logits_acc is None else logits_acc + out["phase_logits"]
        self.lock_drop._mc = False
        stack = torch.stack(preds, 0)
        return {
            "rsd": stack.mean(0),
            "rsd_std": stack.std(0),
            "progress": torch.stack(progs, 0).mean(0),
            "phase_logits": logits_acc / float(self.mc_samples),
        }
