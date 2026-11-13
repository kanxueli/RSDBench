"""Similar-case retrieval baseline (edit distance on completed phase steps)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _compress_phases(phases: np.ndarray) -> Tuple[int, ...]:
    """Collapse consecutive identical phase ids into a step sequence."""
    phases = np.asarray(phases).reshape(-1).astype(np.int64)
    if len(phases) == 0:
        return tuple()
    steps = [int(phases[0])]
    for p in phases[1:]:
        if int(p) != steps[-1]:
            steps.append(int(p))
    return tuple(steps)


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    # two-row DP
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


class Model(nn.Module):
    """
    PitRSDNet-style similar case retrieval:
      fit stores train step sequences + total durations;
      at elapsed t with observed phase prefix, find k nearest by edit distance,
      R_hat = max(0, mean(T_j) - t).
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.k = int(mcfg.get("k", 3))
        self.compare = str(mcfg.get("compare", "prefix"))  # prefix | full
        self.train_steps: List[Tuple[int, ...]] = []
        self.train_durations: np.ndarray = np.asarray([], dtype=np.float64)

    def fit(self, durations_min: np.ndarray) -> None:
        # durations alone insufficient; prefer fit_from_videos
        self.train_durations = np.asarray(durations_min, dtype=np.float64)

    def fit_from_videos(self, phases_list, elapsed_list, durations) -> None:
        steps, durs = [], []
        for phases, T in zip(phases_list, durations):
            steps.append(_compress_phases(phases))
            durs.append(float(T))
        self.train_steps = steps
        self.train_durations = np.asarray(durs, dtype=np.float64)

    def _neighbor_mean_T(self, query: Tuple[int, ...]) -> float:
        if len(self.train_durations) == 0:
            return 0.0
        dists = []
        for s, T in zip(self.train_steps, self.train_durations):
            if self.compare == "prefix" and len(s) >= len(query) and len(query) > 0:
                ref = s[: len(query)]
            else:
                ref = s
            dists.append((_levenshtein(query, ref), float(T)))
        dists.sort(key=lambda x: (x[0], x[1]))
        k = min(self.k, len(dists))
        return float(np.mean([T for _, T in dists[:k]]))

    def _predict_numpy(self, phases: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        phases = np.asarray(phases).reshape(-1).astype(np.int64)
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        out = np.zeros_like(elapsed)
        # Cache predictions per unique step-prefix length change to save compute
        cache: Dict[Tuple[int, ...], float] = {}
        for i in range(len(elapsed)):
            q = _compress_phases(phases[: i + 1])
            if q not in cache:
                cache[q] = self._neighbor_mean_T(q)
            out[i] = max(0.0, cache[q] - float(elapsed[i]))
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        phases = batch["phase"]
        elapsed = batch["elapsed"]
        if torch.is_tensor(elapsed):
            device = elapsed.device
            pred = self._predict_numpy(
                phases.detach().cpu().numpy() if torch.is_tensor(phases) else phases,
                elapsed.detach().cpu().numpy(),
            )
            return {"rsd": torch.from_numpy(pred.astype(np.float32)).to(device).reshape_as(elapsed)}
        pred = self._predict_numpy(np.asarray(phases), np.asarray(elapsed))
        return {"rsd": torch.from_numpy(pred.astype(np.float32))}
