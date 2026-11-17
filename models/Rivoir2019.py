"""Rivoir 2019."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.alexnet_style import AlexNetEncoder
_SEG_CNN_SCHEDULE: List[List[str]] = [
    [],  # iter0: LSTM+FC4 only
    [],  # iter1: +FC2 (still no CNN)
    ["fc1"],
    ["fc1", "conv5"],
    ["fc1", "conv5", "conv4"],
    ["fc1", "conv5", "conv4", "conv3"],
    ["fc1", "conv5", "conv4", "conv3", "conv2"],
    ["fc1", "conv5", "conv4", "conv3", "conv2", "conv1"],
]

def corridor_smooth_l1(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    elapsed: torch.Tensor,
    t_median: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    pred = pred.reshape(-1)
    tgt = tgt.reshape(-1)
    elapsed = elapsed.reshape(-1)
    n = torch.clamp(t_median - elapsed, min=0.0)
    prog = elapsed / (tgt + elapsed).clamp(min=1e-6)
    alpha = 1.0 - 2.0 / (1.0 + torch.exp(5.0 * prog))
    c = alpha * tgt + (1.0 - alpha) * n
    lo = torch.minimum(tgt, c)
    hi = torch.maximum(tgt, c)
    inside = (pred >= lo) & (pred <= hi)
    denom = (c - tgt).abs().clamp(min=1e-6)
    pi = torch.ones_like(pred)
    pi = torch.where(inside, ((pred - tgt).abs() / denom) ** 2, pi)
    loss = pi * F.smooth_l1_loss(pred, tgt, reduction="none")
    if weights is not None:
        loss = loss * weights.reshape(-1)
    return loss.mean()

class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.embed_dim = int(m.get("embed_dim", 256))
        self.hidden = int(m.get("hidden_size", 512))
        self.n_sub = int(m.get("n_subactivities", 10))
        self.rsd_scale = float(m.get("rsd_scale", 0.05))
        self.dropout_p = float(m.get("dropout", 0.2))
        self.pipeline = str(m.get("pipeline", "regularization"))
        self.cnn_batch = int(m.get("cnn_batch", 192))
        self.cnn = AlexNetEncoder(
            embed_dim=self.embed_dim,
            dropout=self.dropout_p,
            pretrained=bool(m.get("pretrained_cnn", False)),
            imagenet_norm=bool(m.get("imagenet_norm", False)),
        )
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim + 1, self.hidden, num_layers=1, batch_first=True)
        self.fc3 = nn.Linear(self.hidden, 1)
        self.fc4 = nn.Linear(self.hidden, self.n_sub)
        self.last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(T,3,H,W) or (B,T,3,H,W) → (T,D) / (B,T,D)."""
        squeeze = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = []
        bs = max(1, self.cnn_batch)
        for i in range(0, flat.size(0), bs):
            feats.append(self.cnn(flat[i : i + bs]))
        feat = torch.cat(feats, 0).view(b, t, -1)
        feat = F.relu(self.fc2(feat))
        return feat.squeeze(0) if squeeze else feat

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "features" in batch:
            feat = batch["features"]
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
        else:
            frames = batch["frames"]
            if frames.dim() == 4:
                frames = frames.unsqueeze(0)
            feat = self.encode_frames(frames)
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
        elapsed = batch["elapsed"]
        if elapsed.dim() == 1:
            elapsed = elapsed.unsqueeze(0)
        x = torch.cat([feat, elapsed.unsqueeze(-1)], dim=-1)
        init = self.last_state if bool(batch.get("stateful", False)) else None
        out, st = self.lstm(x, init)
        self.last_state = (st[0].detach(), st[1].detach())
        rsd = self.fc3(out).squeeze(-1) / self.rsd_scale
        sub_logits = self.fc4(out)
        if rsd.dim() == 2 and rsd.size(0) == 1 and batch["elapsed"].dim() == 1:
            rsd = rsd.squeeze(0)
            sub_logits = sub_logits.squeeze(0)
        return {"rsd": rsd, "sub_logits": sub_logits}

    def configure_seg_trainable(self, seg_iter: int) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = True
        for p in self.fc4.parameters():
            p.requires_grad = True
        if seg_iter >= 1:
            for p in self.fc2.parameters():
                p.requires_grad = True
        idx = min(max(int(seg_iter), 0), len(_SEG_CNN_SCHEDULE) - 1)
        self.cnn.set_requires_grad(_SEG_CNN_SCHEDULE[idx], freeze_rest=True)
        for p in self.fc3.parameters():
            p.requires_grad = False

    def configure_rsd_trainable(self, pipeline: Optional[str] = None) -> None:
        pipe = (pipeline or self.pipeline).lower().replace("-", "_")
        self.pipeline = pipe
        for p in self.parameters():
            p.requires_grad = False

        if pipe in ("feature_extraction", "fe"):
            for mod in (self.fc2, self.lstm, self.fc3, self.fc4):
                for p in mod.parameters():
                    p.requires_grad = True
        elif pipe in ("pretraining", "pt"):
            self.cnn.set_requires_grad(["conv5", "fc1"], freeze_rest=True)
            for mod in (self.fc2, self.lstm, self.fc3, self.fc4):
                for p in mod.parameters():
                    p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def rsd_param_groups(self, lr: float, pretrained_lr_mult: float = 0.1):
        """Adam/SGD groups; PT pipeline uses smaller lr on Conv5+FC1."""
        pipe = self.pipeline.lower().replace("-", "_")
        if pipe not in ("pretraining", "pt"):
            return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]
        slow, fast = [], []
        slow_ids = {id(p) for p in self.cnn.named_blocks()["conv5"] + self.cnn.named_blocks()["fc1"]}
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (slow if id(p) in slow_ids else fast).append(p)
        groups = []
        if fast:
            groups.append({"params": fast, "lr": lr})
        if slow:
            groups.append({"params": slow, "lr": lr * float(pretrained_lr_mult)})
        return groups
