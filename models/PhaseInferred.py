"""Phase-inferred RSD baseline (reference phase durations)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fit per-phase reference durations on train videos (median by default).
    At time t with phase k:
      R = max(0, t_ref[k] - elapsed_in_phase) + sum_{m>k} t_ref[m]
    Assumes ordered phases 0..N-1 without optional skips (Cholec80/Cataract-ish).
    """

    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.stat = str(mcfg.get("stat", "median"))
        self.n_phases = int(mcfg.get("n_phases", 7))
        self.phase_refs = np.zeros(self.n_phases, dtype=np.float64)

    def fit_from_videos(self, phases_list, elapsed_list, durations) -> None:
        """
        phases_list[i]: (T_i,) phase ids
        elapsed_list[i]: (T_i,) elapsed minutes
        """
        per_phase = [[] for _ in range(self.n_phases)]
        for phases, elapsed, T in zip(phases_list, elapsed_list, durations):
            phases = np.asarray(phases)
            elapsed = np.asarray(elapsed, dtype=np.float64)
            # approximate phase duration by span of contiguous segments
            if len(phases) == 0:
                continue
            # sample dt
            dt = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else (T / max(len(phases), 1))
            for p in range(self.n_phases):
                per_phase[p].append(float((phases == p).sum() * dt))
        refs = []
        for vals in per_phase:
            if not vals:
                refs.append(0.0)
            elif self.stat == "mean":
                refs.append(float(np.mean(vals)))
            else:
                refs.append(float(np.median(vals)))
        self.phase_refs = np.asarray(refs, dtype=np.float64)

    def _predict_numpy(self, phases: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        phases = np.asarray(phases).reshape(-1).astype(np.int64)
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        out = np.zeros_like(elapsed)
        refs = self.phase_refs
        # elapsed within current phase: time since phase start
        phase_elapsed = np.zeros_like(elapsed)
        phase_start_t = float(elapsed[0]) if len(elapsed) else 0.0
        for i in range(len(phases)):
            if i == 0 or phases[i] != phases[i - 1]:
                phase_start_t = float(elapsed[i])
            phase_elapsed[i] = max(0.0, elapsed[i] - phase_start_t)
        suffix = np.cumsum(refs[::-1])[::-1]  # sum from k..end
        for i, k in enumerate(phases):
            k = int(np.clip(k, 0, len(refs) - 1))
            rem_cur = max(0.0, refs[k] - phase_elapsed[i])
            rem_future = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
            out[i] = rem_cur + rem_future
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        phases = batch["phase"]
        elapsed = batch["elapsed"]
        if torch.is_tensor(elapsed):
            device = elapsed.device
            pred = self._predict_numpy(
                phases.detach().cpu().numpy(),
                elapsed.detach().cpu().numpy(),
            )
            return {"rsd": torch.from_numpy(pred.astype(np.float32)).to(device).reshape_as(elapsed)}
        pred = self._predict_numpy(np.asarray(phases), np.asarray(elapsed))
        return {"rsd": torch.from_numpy(pred.astype(np.float32))}
