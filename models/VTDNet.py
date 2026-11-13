"""VTD-Net (Bodenstedt et al., IJCARS 2019)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


class SigmoidGRU(nn.Module):
    """GRU with sigmoid candidate activation."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, 3 * hidden_size)
        self.w_hh = nn.Linear(hidden_size, 3 * hidden_size)
        bound = 1.0 / math.sqrt(hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, D) → hidden states (T, H)."""
        gi = self.w_ih(x)
        h = x.new_zeros(self.hidden_size)
        outs = []
        for t in range(x.size(0)):
            gh = self.w_hh(h)
            i_r, i_z, i_n = gi[t].chunk(3)
            h_r, h_z, h_n = gh.chunk(3)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.sigmoid(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.stack(outs)


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs.get("model", configs)
        self.mode = str(m.get("mode", "V")).upper()
        if self.mode not in ("V", "VT"):
            raise ValueError(f"VTDNet mode={self.mode!r}: only V / VT are supported")
        self.n_tools = int(m.get("n_tools", 7))
        self.fps = float(m.get("fps", 1.0))
        feat_dim = int(m.get("feature_size", 4096))
        gru_in = feat_dim + (self.n_tools if self.mode == "VT" else 0)
        self.gru = SigmoidGRU(gru_in, int(m.get("gru_hidden", 512)))
        self.head = nn.Sequential(
            nn.Linear(int(m.get("gru_hidden", 512)), 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "features" not in batch:
            raise ValueError("VTDNet expects frozen AlexNet features (batch['features']); see module docstring")
        x = batch["features"].float()
        if x.dim() == 3:
            x = x[0]
        if self.mode == "VT":
            tools = batch["tools"].float()
            tools = tools[0] if tools.dim() == 3 else tools
            fixed = x.new_zeros(x.size(0), self.n_tools)
            k = min(self.n_tools, tools.size(-1))
            fixed[:, :k] = tools[:, :k]
            x = torch.cat([x, fixed], dim=-1)
        progress = self.head(self.gru(x)).squeeze(-1)
        i = torch.arange(1, x.size(0) + 1, device=x.device, dtype=progress.dtype)
        duration = i / progress.clamp(min=1e-4)
        rsd = (duration - i).clamp(min=0) / (self.fps * 60.0)
        return {"rsd": rsd, "progress": progress}
