"""SWAG building blocks (Boels et al.)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_spstar_prior(
    prior_json: str | Path,
    n_future: int,
    horizon: int,
) -> torch.Tensor:
    """
    Load naive2_*_class_probs_at{H}.json → tensor (n_curr, horizon, n_future) probabilities.
    JSON: {curr_class: [ {future_class: count_or_prob}, ... horizon entries ]}.
    """
    path = Path(prior_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Support both count dicts and already-normalized probs
    table = torch.full((n_future, horizon, n_future), 1.0 / max(n_future, 1))
    for ck, freq_list in raw.items():
        c = int(ck)
        if c >= n_future:
            continue
        for h, freq_dict in enumerate(freq_list[:horizon]):
            tot = sum(float(v) for v in freq_dict.values()) + 1e-8
            row = torch.zeros(n_future)
            for fk, v in freq_dict.items():
                fi = int(fk)
                if fi < n_future:
                    row[fi] = float(v) / tot
            # leftover mass → uniform over unseen if empty
            if row.sum() <= 0:
                row[:] = 1.0 / n_future
            else:
                row = row / row.sum()
            table[c, h] = row
    return table


class KeyRecorder(nn.Module):
    """Global key-pooling of past + running max over local present (SWAG)."""

    def __init__(self, dim: int = 256, reduc_dim: int = 64, sampling_rate: int = 10, local_size: int = 20):
        super().__init__()
        self.local_size = int(local_size)
        self.sampling_rate = int(sampling_rate)
        self.linear_reduction = nn.Sequential(
            nn.Linear(dim, reduc_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(reduc_dim),
        )
        self.linear_expand = nn.Sequential(
            nn.Linear(reduc_dim, dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, D)
        b, t, _ = feats.shape
        loc = min(self.local_size, t)
        compressed = self.linear_reduction(feats)
        past = compressed[:, : max(t - loc, 1) : self.sampling_rate]
        if past.numel() == 0:
            past_pool = compressed[:, :1].amax(dim=1)
        else:
            past_pool = past.amax(dim=1)
        present = compressed[:, -loc:]
        running = past_pool
        outs = []
        for i in range(present.size(1)):
            running = torch.maximum(running, present[:, i])
            outs.append(running)
        global_reduc = torch.stack(outs, dim=1)  # (B, loc, reduc)
        return self.linear_expand(global_reduc)


class FusionHead(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.LayerNorm(dim))

    def forward(self, key_feats: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = self.norm1(key_feats + feats)
        return self.norm2(self.fc(x) + x)


class SinglePassFutureDecoder(nn.Module):
    """SWAG-SP*: anticipation queries + optional dataset JSON class priors."""

    def __init__(
        self,
        d_model: int,
        n_future: int,
        horizon: int,
        nhead: int = 4,
        dropout: float = 0.1,
        prior_table: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_future = n_future
        self.horizon = horizon
        self.queries = nn.Parameter(torch.empty(horizon, d_model))
        nn.init.xavier_uniform_(self.queries)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_future)
        # Fallback learnable prior if JSON absent
        self.prior_emb = nn.Embedding(n_future * horizon, n_future)
        if prior_table is not None:
            # (n_curr, H, C) → register as buffer log-prob bias
            self.register_buffer("prior_logprob", torch.log(prior_table.clamp_min(1e-8)))
        else:
            self.prior_logprob = None

    def forward(
        self,
        memory: torch.Tensor,
        curr_phase_id: Optional[torch.Tensor] = None,
        use_prior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = memory.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.attn(q, memory, memory, need_weights=False)
        x = self.norm(q + attn_out)
        x = self.norm2(x + self.ff(x))
        logits = self.cls(x)
        if use_prior and curr_phase_id is not None:
            h = self.horizon
            c = self.n_future
            cid = curr_phase_id.clamp(0, c - 1)
            if self.prior_logprob is not None:
                # (B, H, C)
                bias = self.prior_logprob[cid]  # index first dim
                logits = logits + bias
            else:
                idx = cid.unsqueeze(1) * h + torch.arange(h, device=memory.device).unsqueeze(0)
                logits = logits + self.prior_emb(idx.clamp(0, self.prior_emb.num_embeddings - 1))
        return logits, x


def soft_rsd_from_future_phases(future_logits: torch.Tensor, eos_id: int, minutes_per_step: float = 1.0) -> torch.Tensor:
    """Expected minutes until EOS from minute-level future phase logits (B, H, C) → (B, 1)."""
    probs = F.softmax(future_logits, dim=-1)
    p_eos = probs[..., eos_id]  # (B, H)
    # survival: not yet EOS at step k; first-EOS mass
    surv = torch.cumprod(1.0 - p_eos + 1e-8, dim=-1)
    # P(first EOS at k) ≈ surv[k-1] * p_eos[k]
    prev = torch.ones_like(p_eos[:, :1])
    surv_prev = torch.cat([prev, surv[:, :-1]], dim=-1)
    first = surv_prev * p_eos
    steps = torch.arange(1, future_logits.size(1) + 1, device=future_logits.device, dtype=future_logits.dtype)
    # if never EOS in horizon, charge full horizon
    mass = first.sum(dim=-1, keepdim=True).clamp(max=1.0)
    expected = (first * steps.unsqueeze(0)).sum(dim=-1, keepdim=True)
    expected = expected + (1.0 - mass) * float(future_logits.size(1))
    return expected * minutes_per_step


def build_future_phase_targets(
    phase: torch.Tensor,
    horizon: int,
    eos_id: int,
    step: int = 60,
) -> torch.Tensor:
    """phase (T,) → future targets (T, H) at +step,+2step,... frames; pad with EOS."""
    t = phase.numel()
    fut = torch.full((t, horizon), eos_id, dtype=torch.long, device=phase.device)
    for h in range(horizon):
        idx = torch.arange(t, device=phase.device) + step * (h + 1)
        valid = idx < t
        fut[valid, h] = phase[idx[valid]]
    return fut
