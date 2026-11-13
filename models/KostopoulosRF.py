"""Kostopoulos random-forest baseline (Kostopoulos et al.)."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor


def _as_tools(tools, T: int, n_tools: int) -> np.ndarray:
    if tools is None:
        return np.zeros((T, n_tools), dtype=np.float64)
    arr = np.asarray(tools, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, n_tools)
    out = np.zeros((T, n_tools), dtype=np.float64)
    m = min(T, arr.shape[0])
    n = min(n_tools, arr.shape[1])
    out[:m, :n] = arr[:m, :n]
    return out


def _phase_tool_rates(phases: np.ndarray, tools: np.ndarray, n_tools: int) -> np.ndarray:
    T = len(phases)
    rates = np.zeros((T, n_tools), dtype=np.float64)
    if T == 0:
        return rates
    start = 0
    for i in range(T):
        if i > 0 and phases[i] != phases[i - 1]:
            start = i
        seg = tools[start : i + 1]
        rates[i] = seg.mean(axis=0) if len(seg) else 0.0
    return rates


def _combo_code(tools_row: np.ndarray) -> float:
    bits = (tools_row > 0.5).astype(np.int64)
    code = 0
    for b in bits:
        code = (code << 1) | int(b)
    return float(code)


def build_features(
    phases: np.ndarray,
    elapsed: np.ndarray,
    tools: Optional[np.ndarray],
    n_phases: int,
    n_tools: int,
    mode: str = "rich",
) -> np.ndarray:
    phases = np.asarray(phases).reshape(-1).astype(np.int64)
    elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
    T = len(phases)
    tools_a = _as_tools(tools, T, n_tools)
    rates = _phase_tool_rates(phases, tools_a, n_tools)

    phase_elapsed = np.zeros(T, dtype=np.float64)
    phase_start = float(elapsed[0]) if T else 0.0
    for i in range(T):
        if i == 0 or phases[i] != phases[i - 1]:
            phase_start = float(elapsed[i])
        phase_elapsed[i] = max(0.0, elapsed[i] - phase_start)

    if mode == "paper":
        X = np.zeros((T, 10), dtype=np.float64)
        X[:, :n_tools] = rates
        for i in range(T):
            X[i, 7] = _combo_code(tools_a[i])
            X[i, 8] = elapsed[i]
            X[i, 9] = float(np.clip(phases[i], 0, n_phases - 1))
        return X

    # rich: 7 rates + 7 instant + combo + elapsed + phase + phase_elapsed + n_active
    # (+ optional phase-prior column appended by caller)
    X = np.zeros((T, 7 + 7 + 5), dtype=np.float64)
    X[:, :n_tools] = rates
    X[:, 7 : 7 + n_tools] = (tools_a > 0.5).astype(np.float64)
    for i in range(T):
        X[i, 14] = _combo_code(tools_a[i])
        X[i, 15] = elapsed[i]
        X[i, 16] = float(np.clip(phases[i], 0, n_phases - 1))
        X[i, 17] = phase_elapsed[i]
        X[i, 18] = float((tools_a[i] > 0.5).sum())
    return X


def _phase_inferred_rsd(phases: np.ndarray, elapsed: np.ndarray, refs: np.ndarray) -> np.ndarray:
    phases = np.asarray(phases).reshape(-1).astype(np.int64)
    elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
    out = np.zeros_like(elapsed)
    if len(elapsed) == 0:
        return out
    phase_elapsed = np.zeros_like(elapsed)
    phase_start = float(elapsed[0])
    for i in range(len(phases)):
        if i == 0 or phases[i] != phases[i - 1]:
            phase_start = float(elapsed[i])
        phase_elapsed[i] = max(0.0, elapsed[i] - phase_start)
    suffix = np.cumsum(refs[::-1])[::-1]
    for i, k in enumerate(phases):
        k = int(np.clip(k, 0, len(refs) - 1))
        rem_cur = max(0.0, refs[k] - phase_elapsed[i])
        rem_future = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
        out[i] = rem_cur + rem_future
    return out


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.n_phases = int(mcfg.get("n_phases", 7))
        self.n_tools = int(mcfg.get("n_tools", 7))
        self.n_estimators = int(mcfg.get("n_estimators", 400))
        self.max_depth = mcfg.get("max_depth", 16)
        if self.max_depth is not None:
            self.max_depth = int(self.max_depth)
        self.min_samples_leaf = int(mcfg.get("min_samples_leaf", 1))
        self.subsample_stride = int(mcfg.get("subsample_stride", 2))
        self.feature_mode = str(mcfg.get("feature_mode", "rich"))
        self.regressor = str(mcfg.get("regressor", "rf"))  # rf | hgb | gbr
        self.dual = bool(mcfg.get("dual", True))
        self.max_iter = int(mcfg.get("max_iter", 400))
        self.learning_rate = float(mcfg.get("learning_rate", 0.05))
        self.use_phase_prior = bool(mcfg.get("use_phase_prior", True))
        self.blend_prior = float(mcfg.get("blend_prior", 0.35))  # final = (1-b)*rf + b*prior
        self.rf_short = None
        self.rf_long = None
        self.threshold_c = float(mcfg.get("threshold_c", 0.0))
        self.phase_refs = np.zeros(self.n_phases, dtype=np.float64)
        self._fitted = False

    def _new_reg(self):
        if self.regressor == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth if self.max_depth is not None else 12,
                min_samples_leaf=self.min_samples_leaf,
                random_state=0,
            )
        if self.regressor == "gbr":
            return GradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 300),
                learning_rate=self.learning_rate,
                max_depth=self.max_depth if self.max_depth is not None else 5,
                min_samples_leaf=self.min_samples_leaf,
                random_state=0,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )

    def fit_from_videos(
        self,
        phases_list,
        elapsed_list,
        durations,
        groups=None,
        experiences=None,
        tools_list: Optional[Sequence] = None,
    ) -> None:
        tools_list = tools_list or [None] * len(phases_list)
        durs = np.asarray(durations, dtype=np.float64)
        # phase duration refs for prior feature / blend
        per_phase = [[] for _ in range(self.n_phases)]
        for phases, elapsed, T in zip(phases_list, elapsed_list, durs):
            phases = np.asarray(phases)
            elapsed = np.asarray(elapsed, dtype=np.float64)
            if len(phases) == 0:
                continue
            dt = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else (T / max(len(phases), 1))
            for p in range(self.n_phases):
                per_phase[p].append(float((phases == p).sum() * dt))
        self.phase_refs = np.asarray(
            [float(np.median(v)) if v else 0.0 for v in per_phase], dtype=np.float64
        )

        if self.dual and len(durs) >= 2:
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = km.fit_predict(durs.reshape(-1, 1))
            centers = km.cluster_centers_.reshape(-1)
            order = np.argsort(centers)
            self.threshold_c = float(0.5 * (centers[order[0]] + centers[order[1]]))
            short_label = int(order[0])
        else:
            self.threshold_c = float(np.median(durs)) if len(durs) else 0.0
            labels = np.zeros(len(durs), dtype=np.int64)
            short_label = 0
            self.dual = False

        Xs, ys, Xl, yl = [], [], [], []
        stride = max(1, self.subsample_stride)
        for phases, elapsed, tools, lab, T in zip(
            phases_list, elapsed_list, tools_list, labels, durs
        ):
            rsd = np.maximum(0.0, float(T) - np.asarray(elapsed, dtype=np.float64))
            X = build_features(
                phases, elapsed, tools, self.n_phases, self.n_tools, mode=self.feature_mode
            )
            if self.use_phase_prior:
                prior = _phase_inferred_rsd(phases, elapsed, self.phase_refs)
                X = np.concatenate([X, prior.reshape(-1, 1)], axis=1)
            idx = np.arange(0, len(X), stride)
            # late-frame emphasis via duplicate last third
            late = idx[idx >= int(0.66 * len(X))]
            idx = np.unique(np.concatenate([idx, late])) if len(late) else idx
            if (not self.dual) or int(lab) == short_label:
                Xs.append(X[idx])
                ys.append(rsd[idx])
            else:
                Xl.append(X[idx])
                yl.append(rsd[idx])

        def _stack(xs, ys_):
            if not xs:
                return np.zeros((0, 1)), np.zeros((0,))
            return np.concatenate(xs, axis=0), np.concatenate(ys_, axis=0)

        X_s, y_s = _stack(Xs, ys)
        X_l, y_l = _stack(Xl, yl)
        if self.dual:
            if len(y_s) == 0 and len(y_l) > 0:
                X_s, y_s = X_l, y_l
            if len(y_l) == 0 and len(y_s) > 0:
                X_l, y_l = X_s, y_s
        if len(y_s) == 0:
            raise RuntimeError("KostopoulosRF: no training frames")

        self.rf_short = self._new_reg()
        self.rf_short.fit(X_s, y_s)
        if self.dual:
            self.rf_long = self._new_reg()
            self.rf_long.fit(X_l, y_l)
        else:
            self.rf_long = self.rf_short
        self._fitted = True
        print(
            f"[KostopoulosRF] C={self.threshold_c:.2f} dual={self.dual} "
            f"reg={self.regressor} feat={self.feature_mode} "
            f"short_n={len(y_s)} long_n={len(y_l) if self.dual else len(y_s)}",
            flush=True,
        )

    def _predict_numpy(self, phases, elapsed, tools) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("KostopoulosRF not fitted")
        X = build_features(
            phases, elapsed, tools, self.n_phases, self.n_tools, mode=self.feature_mode
        )
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        prior = _phase_inferred_rsd(phases, elapsed, self.phase_refs)
        if self.use_phase_prior:
            X = np.concatenate([X, prior.reshape(-1, 1)], axis=1)
        pred = np.zeros(len(elapsed), dtype=np.float64)
        if not self.dual:
            pred = self.rf_short.predict(X)
        else:
            short_m = elapsed < self.threshold_c
            if short_m.any():
                pred[short_m] = self.rf_short.predict(X[short_m])
            if (~short_m).any():
                pred[~short_m] = self.rf_long.predict(X[~short_m])
        b = float(np.clip(self.blend_prior, 0.0, 1.0))
        if b > 0:
            pred = (1.0 - b) * pred + b * prior
        return np.maximum(pred, 0.0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        phases = batch["phase"]
        elapsed = batch["elapsed"]
        tools = batch.get("tools")
        if torch.is_tensor(elapsed):
            device = elapsed.device
            tools_np = None if tools is None else tools.detach().cpu().numpy()
            pred = self._predict_numpy(
                phases.detach().cpu().numpy(),
                elapsed.detach().cpu().numpy(),
                tools_np,
            )
            return {"rsd": torch.from_numpy(pred.astype(np.float32)).to(device).reshape_as(elapsed)}
        tools_np = None if tools is None else np.asarray(tools)
        pred = self._predict_numpy(np.asarray(phases), np.asarray(elapsed), tools_np)
        return {"rsd": torch.from_numpy(pred.astype(np.float32))}
