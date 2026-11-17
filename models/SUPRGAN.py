"""SUPR-GAN."""

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
        self.hidden = int(m.get("hidden_size", 128))
        self.tp = int(m.get("past_horizon", 15))
        self.tf = int(m.get("future_horizon", 15))
        self.ns = int(m.get("n_samples", 10))
        self.noise_dim = int(m.get("noise_dim", 16))
        self.gumbel_tau = float(m.get("gumbel_tau", 1.0))
        self.cnn_batch = int(m.get("cnn_batch", 128))
        self.rsd_leave_weight = float(m.get("rsd_leave_weight", 0.0))
        self.rsd_mode = str(m.get("rsd_mode", "learned")).lower()
        self.rsd_blend = float(m.get("rsd_blend", 0.7))
        self.max_len = float(m.get("max_len", 20.0))
        self.fps = float(m.get("fps", 1.0))
        self.cnn = _CNN(pretrained=bool(m.get("pretrained_cnn", True)))
        self.enc_lstm = nn.LSTM(self.cnn.out_dim + 2, self.hidden, batch_first=True)
        self.phase_head = nn.Linear(self.hidden, self.n_phases)
        self.prog_head = nn.Linear(self.hidden, 1)
        self.noise_fc = nn.Linear(self.hidden + self.noise_dim, self.hidden)
        self.dec_lstm = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.dec_head = nn.Linear(self.hidden, self.n_phases)
        self.d_past = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.d_future = nn.LSTM(self.n_phases, self.hidden, batch_first=True)
        self.d_head = nn.Linear(2 * self.hidden, 1)
        self.rsd_head = nn.Sequential(
            nn.Linear(self.hidden + 2 + self.n_phases, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(m.get("rsd_dropout", 0.1))),
            nn.Linear(self.hidden, 1),
        )
        self.register_buffer("phase_refs", torch.zeros(self.n_phases), persistent=True)

    def _elapsed_channels(self, elapsed: torch.Tensor) -> torch.Tensor:
        raw = elapsed.unsqueeze(-1)
        norm = (elapsed / max(self.max_len, 1e-6)).clamp(0.0, 2.0).unsqueeze(-1)
        return torch.cat([raw, norm], dim=-1)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        outs = []
        bs = max(1, self.cnn_batch)
        for i in range(0, frames.size(0), bs):
            outs.append(self.cnn(frames[i : i + bs]))
        return torch.cat(outs, 0)

    def encode_video(self, frames: torch.Tensor, elapsed: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encode_frames(frames)
        if elapsed.dim() == 2:
            elapsed = elapsed[0]
        x = torch.cat([feat, self._elapsed_channels(elapsed)], dim=-1)
        h, _ = self.enc_lstm(x.unsqueeze(0))
        h = h.squeeze(0)
        logits = self.phase_head(h)
        progress = torch.sigmoid(self.prog_head(h).squeeze(-1))
        return {"feat": feat, "h": h, "phase_logits": logits, "progress": progress}

    def predict_rsd(self, h: torch.Tensor, logits: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        """Differentiable RSD in minutes (softplus)."""
        e = self._elapsed_channels(elapsed)
        prob = F.softmax(logits, dim=-1)
        x = torch.cat([h, e, prob], dim=-1)
        return F.softplus(self.rsd_head(x).squeeze(-1))

    def decode_future(
        self,
        h_last: torch.Tensor,
        n_samples: Optional[int] = None,
        greedy: bool = False,
    ):
        ns = 1 if greedy else int(n_samples or self.ns)
        z = torch.zeros(ns, self.noise_dim, device=h_last.device) if greedy else torch.randn(
            ns, self.noise_dim, device=h_last.device
        )
        h0 = self.noise_fc(torch.cat([h_last.unsqueeze(0).expand(ns, -1), z], dim=-1))
        h0 = h0.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        y = torch.zeros(ns, 1, self.n_phases, device=h_last.device)
        hards, logits_list = [], []
        state = (h0, c0)
        for _ in range(self.tf):
            out, state = self.dec_lstm(y, state)
            logits = self.dec_head(out)
            logits_list.append(logits)
            if greedy:
                idx = logits.squeeze(1).argmax(-1)
                y = F.one_hot(idx, self.n_phases).float().unsqueeze(1)
            else:
                y = F.gumbel_softmax(logits.squeeze(1), tau=self.gumbel_tau, hard=True, dim=-1).unsqueeze(1)
            hards.append(y)
        return torch.cat(hards, dim=1), torch.cat(logits_list, dim=1)

    def discriminate(self, past_oh: torch.Tensor, future_oh: torch.Tensor) -> torch.Tensor:
        _, (hp, _) = self.d_past(past_oh)
        _, (hf, _) = self.d_future(future_oh)
        h = torch.cat([hp.squeeze(0), hf.squeeze(0)], dim=-1)
        return self.d_head(h).squeeze(-1)

    def set_phase_refs(self, refs: np.ndarray) -> None:
        t = torch.as_tensor(refs, dtype=torch.float32)
        if t.numel() != self.n_phases:
            t2 = torch.zeros(self.n_phases)
            n = min(self.n_phases, t.numel())
            t2[:n] = t[:n]
            t = t2
        self.phase_refs.copy_(t.to(self.phase_refs.device))

    def _phase_elapsed(self, phases: np.ndarray, elapsed_np: np.ndarray) -> np.ndarray:
        pe = np.zeros_like(elapsed_np)
        phase_start = float(elapsed_np[0]) if len(elapsed_np) else 0.0
        for i, k in enumerate(phases):
            if i == 0 or k != int(phases[i - 1]):
                phase_start = float(elapsed_np[i])
            pe[i] = max(0.0, float(elapsed_np[i]) - phase_start)
        return pe

    def rsd_from_phases(self, phases: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        if phases.dim() == 2:
            phases = phases.argmax(-1)
        hard = phases.long().cpu().numpy().reshape(-1)
        elapsed_np = elapsed.detach().float().cpu().numpy().reshape(-1)
        refs = self.phase_refs.detach().cpu().numpy()
        suffix = np.cumsum(refs[::-1])[::-1]
        pe = self._phase_elapsed(hard, elapsed_np)
        out = np.zeros(len(elapsed_np), dtype=np.float64)
        for i, k in enumerate(hard):
            k = int(np.clip(k, 0, len(refs) - 1))
            rem_cur = max(0.0, float(refs[k]) - pe[i])
            rem_fut = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
            out[i] = rem_cur + rem_fut
        return torch.as_tensor(out, device=elapsed.device, dtype=elapsed.dtype)

    @torch.no_grad()
    def rsd_with_leave_refine(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        elapsed: torch.Tensor,
        n_samples: Optional[int] = None,
        stride: int = 8,
    ) -> torch.Tensor:
        hard_t = logits.argmax(-1)
        base = self.rsd_from_phases(hard_t, elapsed)
        w = float(self.rsd_leave_weight)
        if w <= 0 or self.tf <= 0:
            return base
        elapsed_np = elapsed.detach().float().cpu().numpy().reshape(-1)
        refs = self.phase_refs.detach().cpu().numpy()
        suffix = np.cumsum(refs[::-1])[::-1]
        hard = hard_t.detach().cpu().numpy().reshape(-1)
        pe = self._phase_elapsed(hard, elapsed_np)
        dt_min = 1.0 / max(self.fps, 1e-6)
        ns = int(n_samples or self.ns)
        T = h.size(0)
        out = base.detach().cpu().numpy().astype(np.float64).copy()
        idxs = list(range(0, T, max(1, int(stride))))
        if idxs[-1] != T - 1:
            idxs.append(T - 1)
        for i in idxs:
            k = int(np.clip(hard[i], 0, len(refs) - 1))
            rem_cur = max(0.0, float(refs[k]) - float(pe[i]))
            rem_fut = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
            fakes_g, _ = self.decode_future(h[i], greedy=True)
            fakes_s, _ = self.decode_future(h[i], n_samples=max(1, ns // 2))
            fut = torch.cat([fakes_g, fakes_s], 0).argmax(-1).detach().cpu().numpy()
            leaves = []
            for j in range(fut.shape[0]):
                leave = next((t for t, p in enumerate(fut[j]) if int(p) != k), len(fut[j]))
                leaves.append(leave * dt_min)
            rem_cur_new = min(rem_cur, float(np.median(leaves)))
            refined = rem_cur_new + rem_fut
            out[i] = (1.0 - w) * out[i] + w * refined
        if len(idxs) >= 2:
            for a, b in zip(idxs[:-1], idxs[1:]):
                if b <= a + 1:
                    continue
                for t in range(a + 1, b):
                    alpha = (t - a) / float(b - a)
                    out[t] = (1.0 - alpha) * out[a] + alpha * out[b]
        return torch.as_tensor(out, device=elapsed.device, dtype=elapsed.dtype)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        if frames.dim() == 5:
            frames = frames[0]
        elapsed = batch["elapsed"]
        if elapsed.dim() == 2:
            elapsed = elapsed[0]
        enc = self.encode_video(frames, elapsed)
        logits = enc["phase_logits"]
        h = enc["h"]
        learned = self.predict_rsd(h, logits, elapsed)
        mode = self.rsd_mode
        if mode == "learned":
            rsd = learned
        elif mode == "phase":
            if (not self.training) and self.rsd_leave_weight > 0:
                rsd = self.rsd_with_leave_refine(h, logits, elapsed)
            else:
                rsd = self.rsd_from_phases(logits.argmax(-1), elapsed)
        else:  # blend
            if (not self.training) and self.rsd_leave_weight > 0:
                prior = self.rsd_with_leave_refine(h, logits, elapsed)
            else:
                prior = self.rsd_from_phases(logits.argmax(-1), elapsed)
            a = float(self.rsd_blend)
            rsd = a * learned + (1.0 - a) * prior.to(dtype=learned.dtype)
        return {
            "rsd": rsd,
            "phase_logits": logits,
            "h": h,
            "rsd_learned": learned,
            "progress": enc["progress"],
        }
