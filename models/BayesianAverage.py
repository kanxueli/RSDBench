"""Dexter-style Bayesian average of prior + historical evidence (no vision)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Dexter 2009 pooling (adapted when scheduled OR time is unavailable):

      μ* = (κ * x_s + n * x̄) / (κ + n)

    - x_s: population prior (train global mean) — surrogate for scheduled duration
    - x̄, n: mean & count of same-surgeon (else same-experience, else global) train cases
    - κ: prior strength (pseudo-counts)

    Online: R_hat = max(0, μ* - t_el).  Does NOT truncate on T>t
    (that is BayesianConditional / ConditionalMedian).
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.kappa = float(mcfg.get("kappa", 3.0))
        self.x_s = 0.0
        self.global_mean = 0.0
        self.group_stats: Dict[Any, tuple[float, int]] = {}  # key -> (mean, n)
        self.exp_stats: Dict[Any, tuple[float, int]] = {}
        self._default_mu = 0.0

    def fit(self, durations_min: np.ndarray) -> None:
        d = np.asarray(durations_min, dtype=np.float64)
        self.global_mean = float(np.mean(d)) if len(d) else 0.0
        self.x_s = self.global_mean
        self._default_mu = self.global_mean
        self.group_stats = {"__all__": (self.global_mean, len(d))}

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
        self.group_stats = {k: (float(np.mean(v)), len(v)) for k, v in by_g.items()}
        self.group_stats["__all__"] = (self.global_mean, len(d))
        self.exp_stats = {k: (float(np.mean(v)), len(v)) for k, v in by_e.items()}

    def _mu_star(self, group: Any = None, experience: Any = None) -> float:
        x_s, kappa = self.x_s, self.kappa
        if group is not None and group in self.group_stats:
            xbar, n = self.group_stats[group]
        elif experience is not None and experience in self.exp_stats:
            xbar, n = self.exp_stats[experience]
        else:
            xbar, n = self.global_mean, int(self.group_stats.get("__all__", (0.0, 0))[1] or 0)
            if n <= 0:
                # fall back: treat as pure prior
                return x_s
        return float((kappa * x_s + n * xbar) / (kappa + n))

    def _predict_numpy(
        self,
        elapsed: np.ndarray,
        group: Any = None,
        experience: Any = None,
    ) -> np.ndarray:
        mu = self._mu_star(group, experience)
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        return np.maximum(0.0, mu - elapsed)

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
