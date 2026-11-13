"""Conditional / adaptive median baseline (Dexter-style)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Given historical total durations {T_i}, at elapsed t:
      R_hat = median({T_i | T_i > t}) - t
    Falls back to max(0, median(T) - t) if no survivors.
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.durations = np.asarray([], dtype=np.float64)
        self.stat = str(mcfg.get("stat", "median"))

    def fit(self, durations_min: np.ndarray) -> None:
        self.durations = np.sort(np.asarray(durations_min, dtype=np.float64))

    def _predict_numpy(self, elapsed: np.ndarray) -> np.ndarray:
        out = np.zeros_like(elapsed, dtype=np.float64)
        d = self.durations
        if len(d) == 0:
            return out
        global_ref = float(np.median(d) if self.stat == "median" else np.mean(d))
        for i, t in enumerate(elapsed):
            survivors = d[d > t]
            if len(survivors) == 0:
                out[i] = 0.0
            else:
                ref = float(np.median(survivors) if self.stat == "median" else np.mean(survivors))
                out[i] = max(0.0, ref - t)
            # numerical guard
            if not np.isfinite(out[i]):
                out[i] = max(0.0, global_ref - t)
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        elapsed = batch["elapsed"]
        if torch.is_tensor(elapsed):
            device = elapsed.device
            e = elapsed.detach().cpu().numpy().reshape(-1)
            pred = self._predict_numpy(e)
            return {"rsd": torch.from_numpy(pred.astype(np.float32)).to(device).reshape_as(elapsed)}
        pred = self._predict_numpy(np.asarray(elapsed, dtype=np.float64).reshape(-1))
        return {"rsd": torch.from_numpy(pred.astype(np.float32))}
