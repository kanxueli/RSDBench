"""Naive RSD baseline: max(0, T_ref - elapsed)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fit on train durations (median or mean), predict online with elapsed only.
    No visual input.
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.stat = str(mcfg.get("stat", "median"))
        self.t_ref = float(mcfg.get("t_ref", 0.0))
        # nn.Parameter so state_dict works; not trained by grad
        self.register_buffer("t_ref_buf", torch.tensor([self.t_ref], dtype=torch.float32))

    def fit(self, durations_min: np.ndarray) -> None:
        durations_min = np.asarray(durations_min, dtype=np.float64)
        if self.stat == "mean":
            self.t_ref = float(np.mean(durations_min))
        else:
            self.t_ref = float(np.median(durations_min))
        self.t_ref_buf.fill_(self.t_ref)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        elapsed = batch["elapsed"]
        if not torch.is_tensor(elapsed):
            elapsed = torch.as_tensor(elapsed, dtype=torch.float32)
        t_ref = self.t_ref_buf.to(elapsed.device).reshape(())
        rsd = torch.clamp(t_ref - elapsed, min=0.0)
        return {"rsd": rsd}
