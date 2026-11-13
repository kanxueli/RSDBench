"""SUPR-GAN (Ban et al., IEEE RA-L 2022)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class _CNN(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # (B,512,1,1)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.backbone(x).flatten(1)


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.n_phases = int(m.get("n_phases", 7))
        self.hidden = int(m.get("hidden_size", 32))
        self.tp = int(m.get("past_horizon", 15))
        self.tf = int(m.get("future_horizon", 15))
        self.ns = int(m.get("n_samples", 10))
        self.noise_dim = int(m.get("noise_dim", 16))
        self.gumbel_tau = float(m.get("gumbel_tau", 1.0))
        self.cnn_batch = int(m.get("cnn_batch", 128))
        self.cnn = _CNN(pretrained=bool(m.get("pretrained_cnn", True)))
        self.enc_lstm = nn.LSTM(self.cnn.out_dim, self.hidden, batch_first=True)
        self.phase_head = nn.Linear(self.hidden, self.n_phases)
        self.noise_fc = nn.Linear(self.hidden + self.noise_dim, self.hidden)
        self.dec_lstm = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.dec_head = nn.Linear(self.hidden, self.n_phases)
        # Discriminator: past / future phase encoders
        self.d_past = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.d_future = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.d_head = nn.Linear(self.hidden, 1)
        # Phase-duration prior for RSD (filled by Exp after fit)
        self.register_buffer("phase_refs", torch.zeros(self.n_phases), persistent=True)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(T,3,H,W) → (T,512)."""
        outs = []
        bs = max(1, self.cnn_batch)
        for i in range(0, frames.size(0), bs):
            outs.append(self.cnn(frames[i : i + bs]))
        return torch.cat(outs, 0)

    def encode_video(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encode_frames(frames)  # (T,512)
        h, _ = self.enc_lstm(feat.unsqueeze(0))  # (1,T,H)
        h = h.squeeze(0)
        logits = self.phase_head(h)
        return {"feat": feat, "h": h, "phase_logits": logits}

    def decode_future(self, h_last: torch.Tensor, n_samples: Optional[int] = None):
        """Sample future phases via Gumbel-Softmax.

        Returns:
          hard: (Ns, Tf, P) one-hots for the discriminator
          logits: (Ns, Tf, P) pre-softmax scores for reconstruction CE
        """
        ns = int(n_samples or self.ns)
        z = torch.randn(ns, self.noise_dim, device=h_last.device)
        h0 = self.noise_fc(torch.cat([h_last.unsqueeze(0).expand(ns, -1), z], dim=-1))
        h0 = h0.unsqueeze(0)  # (1,Ns,H)
        c0 = torch.zeros_like(h0)
        y = torch.zeros(ns, 1, self.n_phases, device=h_last.device)
        hards, logits_list = [], []
        state = (h0, c0)
        for _ in range(self.tf):
            out, state = self.dec_lstm(y, state)
            logits = self.dec_head(out)  # (Ns,1,P)
            logits_list.append(logits)
            y = F.gumbel_softmax(logits.squeeze(1), tau=self.gumbel_tau, hard=True, dim=-1).unsqueeze(1)
            hards.append(y)
        # logits_list items are (Ns,1,P) → (Ns,Tf,P)
        return torch.cat(hards, dim=1), torch.cat(logits_list, dim=1)

    def discriminate(self, past_oh: torch.Tensor, future_oh: torch.Tensor) -> torch.Tensor:
        """past_oh/future_oh: (B,T,P) → logits (B,)."""
        _, (hp, _) = self.d_past(past_oh)
        _, (hf, _) = self.d_future(future_oh)
        return self.d_head(hf.squeeze(0)).squeeze(-1)

    def set_phase_refs(self, refs: np.ndarray) -> None:
        t = torch.as_tensor(refs, dtype=torch.float32)
        if t.numel() != self.n_phases:
            t2 = torch.zeros(self.n_phases)
            n = min(self.n_phases, t.numel())
            t2[:n] = t[:n]
            t = t2
        self.phase_refs.copy_(t.to(self.phase_refs.device))

    def rsd_from_phases(self, phases: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        """PhaseInferred-style RSD from hard/soft phase ids."""
        if phases.dim() == 2:  # soft (T,P)
            phases = phases.argmax(-1)
        phases = phases.long().cpu().numpy()
        elapsed_np = elapsed.detach().float().cpu().numpy().reshape(-1)
        refs = self.phase_refs.detach().cpu().numpy()
        out = np.zeros_like(elapsed_np)
        phase_elapsed = 0.0
        phase_start = float(elapsed_np[0]) if len(elapsed_np) else 0.0
        suffix = np.cumsum(refs[::-1])[::-1]
        for i, k in enumerate(phases):
            k = int(np.clip(k, 0, len(refs) - 1))
            if i == 0 or k != int(phases[i - 1]):
                phase_start = float(elapsed_np[i])
            pe = max(0.0, float(elapsed_np[i]) - phase_start)
            rem_cur = max(0.0, float(refs[k]) - pe)
            rem_fut = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
            out[i] = rem_cur + rem_fut
        return torch.as_tensor(out, device=elapsed.device, dtype=elapsed.dtype)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        if frames.dim() == 5:
            frames = frames[0]
        elapsed = batch["elapsed"]
        if elapsed.dim() == 2:
            elapsed = elapsed[0]
        enc = self.encode_video(frames)
        logits = enc["phase_logits"]
        rsd = self.rsd_from_phases(logits.softmax(-1), elapsed)
        return {"rsd": rsd, "phase_logits": logits, "h": enc["h"]}
