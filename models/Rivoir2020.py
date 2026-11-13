"""Rivoir 2020 (Rivoir et al., MICCAI 2020)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.alexnet_style import AlexNetEncoder


class _LockedDropout(nn.Module):
    """Apply one Bernoulli mask across the time axis (Gal et al.)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) or (B,D)
        if self.p <= 0 or (not self.training and not getattr(self, "_mc", False)):
            return x
        if x.dim() == 2:
            mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - self.p) / (1 - self.p)
            return x * mask
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.embed_dim = int(m.get("embed_dim", 256))
        self.hidden = int(m.get("hidden_size", 512))
        self.dropout_p = float(m.get("dropout", 0.2))
        self.mc_samples = int(m.get("mc_samples", 10))
        self.n_phases = int(m.get("n_phases", 7))  # 3-class style regularizer → use phase CE
        self.cnn_batch = int(m.get("cnn_batch", 192))
        self.cnn = AlexNetEncoder(
            embed_dim=self.embed_dim,
            dropout=self.dropout_p,
            pretrained=bool(m.get("pretrained_cnn", False)),
        )
        self.lock_drop = _LockedDropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + 1, self.hidden, num_layers=1, batch_first=True)
        self.head_rsd = nn.Linear(self.hidden, 1)
        self.head_cls = nn.Linear(self.hidden, self.n_phases)
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

    def _forward_once(self, feat: torch.Tensor, elapsed: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        feat = self.lock_drop(feat)
        x = torch.cat([feat, elapsed.unsqueeze(-1)], dim=-1)
        out, _ = self.lstm(x)
        out = self.lock_drop(out)
        rsd = self.head_rsd(out).squeeze(-1)
        logits = self.head_cls(out)
        if squeeze:
            rsd, logits = rsd.squeeze(0), logits.squeeze(0)
        return {"rsd": rsd, "phase_logits": logits}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        if frames.dim() == 4:
            single = True
        else:
            single = False
        feat = self._encode(frames if not single else frames)
        if single and feat.dim() == 3:
            feat = feat.squeeze(0)

        # Training: one stochastic forward. Eval: MC mean unless mc_samples==1.
        do_mc = (not self.training) and self.mc_samples > 1
        if not do_mc:
            self.lock_drop._mc = self.training
            return self._forward_once(feat, elapsed)

        was_training = self.training
        self.train(True)  # enable dropout
        self.lock_drop._mc = True
        preds = []
        logits_acc = None
        for _ in range(self.mc_samples):
            out = self._forward_once(feat, elapsed)
            preds.append(out["rsd"])
            logits_acc = out["phase_logits"] if logits_acc is None else logits_acc + out["phase_logits"]
        self.train(was_training)
        self.lock_drop._mc = False
        stack = torch.stack(preds, 0)
        return {
            "rsd": stack.mean(0),
            "rsd_std": stack.std(0),
            "phase_logits": logits_acc / float(self.mc_samples),
        }
