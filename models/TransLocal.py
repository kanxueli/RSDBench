"""TransLocal (Loukas et al., Int J Med Robot 2024)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101

from layers.aws_saliency import aws_crop_tensor_batch


class _LocalMHAEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, local_window: int):
        super().__init__()
        self.local_window = local_window
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _local_attn_mask(self, t: int, device) -> torch.Tensor:
        mask = torch.ones(t, t, dtype=torch.bool, device=device)
        for i in range(t):
            lo = max(0, i - self.local_window + 1)
            mask[i, lo : i + 1] = False
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        attn_mask = self._local_attn_mask(t, x.device)
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.drop(y))
        y = self.ffn(x)
        x = self.norm2(x + self.drop(y))
        return x


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs.get("model", configs)
        self.max_len = float(mcfg.get("max_len", 100.0))
        self.n_phases = int(mcfg.get("n_phases", 7))
        self.local_window = int(mcfg.get("local_window", 60))
        self.chunk = int(mcfg.get("cnn_chunk", 32))
        self.clip_len = int(mcfg.get("clip_len", 20))
        self.use_aws = bool(mcfg.get("use_aws", True))
        self.aws_out = int(mcfg.get("aws_crop", 224))
        d_model = self.n_phases + 1
        nhead = int(mcfg.get("nhead", 1))
        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1

        from utils.pretrained import load_torchvision_with_weights

        cnn = load_torchvision_with_weights(resnet101, ResNet101_Weights.DEFAULT, "resnet101")
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.feat_dim = 2048
        # Normalize CNN features before concat with elapsed∈[0,1]
        self.feat_norm = nn.LayerNorm(self.feat_dim)
        self.phase_lstm = nn.LSTM(self.feat_dim + 1, 32, num_layers=1, batch_first=True)
        self.phase_head = nn.Linear(32, self.n_phases)

        self.tb1 = _LocalMHAEncoderLayer(d_model, nhead, dropout=0.1, local_window=self.local_window)
        self.tb2 = _LocalMHAEncoderLayer(d_model, nhead, dropout=0.1, local_window=self.local_window)
        self.rsd_head = nn.Linear(d_model, 1)

    def _cnn(self, frames: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, frames.size(0), self.chunk):
            outs.append(self.cnn(frames[i : i + self.chunk]))
        return torch.cat(outs, dim=0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        single = frames.dim() == 4
        if single:
            frames = frames.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
        b, t, c, h, w = frames.shape
        assert b == 1
        flat = frames.reshape(b * t, c, h, w)
        if self.use_aws:
            flat = aws_crop_tensor_batch(flat, out_size=self.aws_out)
        # ImageNet normalize
        mean = flat.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = flat.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        flat = (flat - mean) / std
        feat = self.feat_norm(self._cnn(flat).reshape(t, -1))
        el = elapsed[0].float().to(feat.device)
        el_n = (el / max(self.max_len, 1e-6)).unsqueeze(-1)
        phase_logits = []
        for i in range(t):
            lo = max(0, i - self.clip_len + 1)
            clip_f = feat[lo : i + 1].unsqueeze(0)
            clip_e = el_n[lo : i + 1].unsqueeze(0)
            seq = torch.cat([clip_f, clip_e], dim=-1)
            out, _ = self.phase_lstm(seq)
            phase_logits.append(self.phase_head(out[0, -1]))
        phase_logits_t = torch.stack(phase_logits, dim=0)
        phase_prob = F.softmax(phase_logits_t, dim=-1)
        p_tau = torch.cat([phase_prob, el_n], dim=-1).unsqueeze(0)
        enc = self.tb2(self.tb1(p_tau))
        # Bounded minutes: avoids soft-collapse of unbounded ReLU to a narrow band.
        rsd = torch.sigmoid(self.rsd_head(enc)).squeeze(-1) * self.max_len
        return {
            "rsd": rsd.squeeze(0),
            "phase_logits": phase_logits_t,
        }
