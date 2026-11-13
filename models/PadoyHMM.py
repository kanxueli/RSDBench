"""Simplified Padoy 2012 — tool-signal HMM → phase → reference RSD.

Improvements over the first cut:
  - topology: seq (stay/next only) or lr (upper-triangular)
  - optional causal tool smoothing
  - decode: offline Viterbi / causal forward-MAP / posterior-weighted RSD
  - stay_boost: strengthen self-loops (duration modeling)
Cataract (no tools): GT-phase PhaseInferred fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def _tools_bin(tools, T: int, n_tools: int, smooth: int = 0) -> np.ndarray:
    if tools is None:
        return np.zeros((T, n_tools), dtype=np.float64)
    arr = np.asarray(tools, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, n_tools)
    out = np.zeros((T, n_tools), dtype=np.float64)
    m = min(T, arr.shape[0])
    n = min(n_tools, arr.shape[1])
    out[:m, :n] = (arr[:m, :n] > 0.5).astype(np.float64)
    if smooth and smooth > 1 and T > 0:
        k = int(smooth)
        csum = np.cumsum(out, axis=0)
        sm = np.zeros_like(out)
        for i in range(T):
            lo = max(0, i - k + 1)
            sm[i] = (csum[i] - (csum[lo - 1] if lo > 0 else 0.0)) / float(i - lo + 1)
        out = (sm >= 0.5).astype(np.float64)
    return out


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.n_phases = int(mcfg.get("n_phases", 7))
        self.n_tools = int(mcfg.get("n_tools", 7))
        self.stat = str(mcfg.get("stat", "median"))
        self.eps = float(mcfg.get("emit_eps", 1e-3))
        self.topology = str(mcfg.get("topology", "seq"))  # seq | lr
        self.tool_smooth = int(mcfg.get("tool_smooth", 5))
        self.decode = str(mcfg.get("decode", "posterior"))  # viterbi | forward | posterior
        self.stay_boost = float(mcfg.get("stay_boost", 0.8))
        self.phase_refs = np.zeros(self.n_phases, dtype=np.float64)
        self.start_prob = np.zeros(self.n_phases, dtype=np.float64)
        self.trans = np.zeros((self.n_phases, self.n_phases), dtype=np.float64)
        self.emit = np.full((self.n_phases, self.n_tools), 0.5, dtype=np.float64)
        self.use_hmm = True
        self._fitted = False

    def _fit_phase_refs(self, phases_list, elapsed_list, durations) -> None:
        per_phase: List[List[float]] = [[] for _ in range(self.n_phases)]
        for phases, elapsed, T in zip(phases_list, elapsed_list, durations):
            phases = np.asarray(phases)
            elapsed = np.asarray(elapsed, dtype=np.float64)
            if len(phases) == 0:
                continue
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
        self._fit_phase_refs(phases_list, elapsed_list, durations)

        has_tools = any(t is not None for t in tools_list)
        self.use_hmm = bool(has_tools)
        if not self.use_hmm:
            print(
                "[PadoyHMM] no tool channels — fallback to GT-phase PhaseInferred RSD",
                flush=True,
            )
            self._fitted = True
            return

        N, D = self.n_phases, self.n_tools
        start_c = np.ones(N, dtype=np.float64) * self.eps
        trans_c = np.ones((N, N), dtype=np.float64) * self.eps
        emit_c = np.ones((N, D), dtype=np.float64) * self.eps
        emit_n = np.ones((N, D), dtype=np.float64) * (2 * self.eps)

        for phases, tools in zip(phases_list, tools_list):
            phases = np.asarray(phases).reshape(-1).astype(np.int64)
            T = len(phases)
            if T == 0:
                continue
            obs = _tools_bin(tools, T, D, smooth=self.tool_smooth)
            p0 = int(np.clip(phases[0], 0, N - 1))
            start_c[p0] += 1.0
            for t in range(T):
                p = int(np.clip(phases[t], 0, N - 1))
                emit_c[p] += obs[t]
                emit_n[p] += 1.0
                if t + 1 < T:
                    q = int(np.clip(phases[t + 1], 0, N - 1))
                    if self.topology == "seq":
                        if q == p or q == p + 1:
                            trans_c[p, q] += 1.0
                        elif q > p + 1:
                            # skip → count as next
                            if p + 1 < N:
                                trans_c[p, p + 1] += 1.0
                            else:
                                trans_c[p, p] += 1.0
                        else:
                            trans_c[p, p] += 1.0
                    else:
                        if q >= p:
                            trans_c[p, q] += 1.0
                        else:
                            trans_c[p, p] += 1.0

        self.start_prob = start_c / start_c.sum()
        self.trans = trans_c / np.maximum(trans_c.sum(axis=1, keepdims=True), self.eps)
        self.emit = np.clip(emit_c / np.maximum(emit_n, self.eps), self.eps, 1.0 - self.eps)
        if self.topology == "seq":
            mask = np.eye(N) + np.eye(N, k=1)
        else:
            mask = np.triu(np.ones((N, N)))
        self.trans = self.trans * mask
        # strengthen self-loops
        if self.stay_boost > 0:
            for i in range(N):
                self.trans[i, i] += self.stay_boost
        self.trans = self.trans / np.maximum(self.trans.sum(axis=1, keepdims=True), self.eps)
        self._fitted = True
        print(
            f"[PadoyHMM] topo={self.topology} decode={self.decode} smooth={self.tool_smooth} "
            f"stay={self.stay_boost} refs={np.round(self.phase_refs, 2)}",
            flush=True,
        )

    def _emit_logprob(self, obs_t: np.ndarray) -> np.ndarray:
        p = self.emit
        lp = obs_t * np.log(p) + (1.0 - obs_t) * np.log(1.0 - p)
        return lp.sum(axis=1)

    def _viterbi(self, obs: np.ndarray) -> np.ndarray:
        T = obs.shape[0]
        N = self.n_phases
        log_start = np.log(np.maximum(self.start_prob, self.eps))
        log_trans = np.log(np.maximum(self.trans, self.eps))
        delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=np.int64)
        delta[0] = log_start + self._emit_logprob(obs[0])
        for t in range(1, T):
            emit = self._emit_logprob(obs[t])
            for j in range(N):
                scores = delta[t - 1] + log_trans[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + emit[j]
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def _forward_posteriors(self, obs: np.ndarray) -> np.ndarray:
        """Causal filtered posteriors α_t(j) normalized."""
        T = obs.shape[0]
        N = self.n_phases
        post = np.zeros((T, N), dtype=np.float64)
        log_trans = np.log(np.maximum(self.trans, self.eps))
        log_alpha = np.log(np.maximum(self.start_prob, self.eps)) + self._emit_logprob(obs[0])
        # stabilize
        log_alpha -= np.max(log_alpha)
        a = np.exp(log_alpha)
        a /= a.sum()
        post[0] = a
        for t in range(1, T):
            # predict
            pred = a @ self.trans
            emit = np.exp(self._emit_logprob(obs[t]))
            a = pred * emit
            s = a.sum()
            a = a / s if s > 0 else np.ones(N) / N
            post[t] = a
        return post

    def _rsd_from_phases(self, phases: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        phases = np.asarray(phases).reshape(-1).astype(np.int64)
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        out = np.zeros_like(elapsed)
        refs = self.phase_refs
        phase_elapsed = np.zeros_like(elapsed)
        phase_start_t = float(elapsed[0]) if len(elapsed) else 0.0
        for i in range(len(phases)):
            if i == 0 or phases[i] != phases[i - 1]:
                phase_start_t = float(elapsed[i])
            phase_elapsed[i] = max(0.0, elapsed[i] - phase_start_t)
        suffix = np.cumsum(refs[::-1])[::-1]
        for i, k in enumerate(phases):
            k = int(np.clip(k, 0, len(refs) - 1))
            rem_cur = max(0.0, refs[k] - phase_elapsed[i])
            rem_future = float(suffix[k + 1]) if k + 1 < len(refs) else 0.0
            out[i] = rem_cur + rem_future
        return out

    def _rsd_from_posterior(self, post: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        """Mixture of PhaseInferred RSD under each phase, using soft phase occupancy.
        Phase-elapsed approx: elapsed * (posterior mass so far is weak) → use hard MAP
        phase-elapsed for simplicity, mix only the remaining-ref part.
        """
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        T, N = post.shape
        # MAP phases for phase-elapsed clock
        map_ph = post.argmax(axis=1).astype(np.int64)
        phase_elapsed = np.zeros(T, dtype=np.float64)
        phase_start = float(elapsed[0]) if T else 0.0
        for i in range(T):
            if i == 0 or map_ph[i] != map_ph[i - 1]:
                phase_start = float(elapsed[i])
            phase_elapsed[i] = max(0.0, elapsed[i] - phase_start)
        refs = self.phase_refs
        suffix = np.cumsum(refs[::-1])[::-1]
        out = np.zeros(T, dtype=np.float64)
        for i in range(T):
            acc = 0.0
            for k in range(N):
                rem_cur = max(0.0, refs[k] - phase_elapsed[i])
                rem_future = float(suffix[k + 1]) if k + 1 < N else 0.0
                acc += post[i, k] * (rem_cur + rem_future)
            out[i] = acc
        return out

    def _predict_numpy(self, phases, elapsed, tools) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PadoyHMM not fitted")
        elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
        phases = np.asarray(phases).reshape(-1).astype(np.int64)
        if not self.use_hmm or tools is None:
            return self._rsd_from_phases(phases, elapsed)
        obs = _tools_bin(tools, len(elapsed), self.n_tools, smooth=self.tool_smooth)
        if self.decode == "viterbi":
            pred_phases = self._viterbi(obs)
            return self._rsd_from_phases(pred_phases, elapsed)
        post = self._forward_posteriors(obs)
        if self.decode == "forward":
            return self._rsd_from_phases(post.argmax(axis=1), elapsed)
        return self._rsd_from_posterior(post, elapsed)

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
