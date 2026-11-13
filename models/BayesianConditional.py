"""Bayesian / Dexter conditional duration: truncate prior given T > elapsed."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Slide: p(T | data, T > t) ∝ p(T | data) * 1[T > t], then take median (or mean).

    p(T|data) is the empirical train duration list, optionally restricted to the
    same surgeon / experience when available (Cataract-101).  Without covariates
    this matches ConditionalMedian; with covariates it is a stratified Dexter update.
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.stat = str(mcfg.get("stat", "median"))
        self.durations = np.asarray([], dtype=np.float64)
        self.by_group: Dict[Any, np.ndarray] = {}
        self.by_exp: Dict[Any, np.ndarray] = {}

    def fit(self, durations_min: np.ndarray) -> None:
        self.durations = np.sort(np.asarray(durations_min, dtype=np.float64))

    def fit_from_videos(self, phases_list, elapsed_list, durations, groups=None, experiences=None) -> None:
        d = np.asarray(durations, dtype=np.float64)
        self.fit(d)
        groups = list(groups) if groups is not None else [None] * len(d)
        experiences = list(experiences) if experiences is not None else [None] * len(d)
        by_g: Dict[Any, List[float]] = {}
        by_e: Dict[Any, List[float]] = {}
        for T, g, e in zip(d, groups, experiences):
            if g is not None:
                by_g.setdefault(g, []).append(float(T))
            if e is not None:
                by_e.setdefault(e, []).append(float(T))
        self.by_group = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in by_g.items()}
        self.by_exp = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in by_e.items()}

    def _pool(self, group: Any = None, experience: Any = None) -> np.ndarray:
        if group is not None and group in self.by_group and len(self.by_group[group]) >= 3:
            return self.by_group[group]
        if experience is not None and experience in self.by_exp and len(self.by_exp[experience]) >= 3:
            return self.by_exp[experience]
        return self.durations

    def _predict_numpy(
        self,
        elapsed: np.ndarray,
        group: Any = None,
        experience: Any = None,
    ) -> np.ndarray:
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        d = self._pool(group, experience)
        out = np.zeros_like(elapsed)
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
            if not np.isfinite(out[i]):
                out[i] = max(0.0, global_ref - t)
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        elapsed = batch["elapsed"]
        group = batch.get("surgeon", batch.get("group"))
        experience = batch.get("surgeon_exp", batch.get("experience"))
        if torch.is_tensor(elapsed):
            device = elapsed.device
            pred = self._predict_numpy(elapsed.detach().cpu().numpy(), group, experience)
            return {"rsd": torch.from_numpy(pred.astype(np.float32)).to(device).reshape_as(elapsed)}
        pred = self._predict_numpy(np.asarray(elapsed), group, experience)
        return {"rsd": torch.from_numpy(pred.astype(np.float32))}
