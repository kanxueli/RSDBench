"""Unified RSD metrics (case-equal / video-macro aggregation)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _as_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mae(pred: Sequence[float], target: Sequence[float]) -> float:
    p, t = _as_1d(pred), _as_1d(target)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def relative_error(
    pred: Sequence[float],
    target: Sequence[float],
    eps: float = 1e-6,
    min_true: float = 1.0,
) -> float:
    """MAPE-style relative error: mean |p-t|/|t| on frames with |t|>=min_true (default 1 min).

    Near-zero RSD frames are excluded so end-of-surgery zeros do not explode the mean.
    Use SMAPE when a score over *all* frames is needed.
    """
    p, t = _as_1d(pred), _as_1d(target)
    if len(p) == 0:
        return float("nan")
    m = np.abs(t) >= float(min_true)
    if not m.any():
        return float("nan")
    p, t = p[m], t[m]
    denom = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs(p - t) / denom))


def smape(
    pred: Sequence[float],
    target: Sequence[float],
    eps: float = 1e-6,
) -> float:
    """Symmetric MAPE: mean 2|p-t| / (|p|+|t|+eps), in [0, 2] (report as fraction, not %)."""
    p, t = _as_1d(pred), _as_1d(target)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)))


def window_mask(rsd_true: np.ndarray, window_min: float) -> np.ndarray:
    """Frames whose ground-truth remaining time is within the last `window_min` minutes."""
    return rsd_true <= float(window_min)


def video_mae(
    pred: Sequence[float],
    target: Sequence[float],
    mask: Optional[Sequence[bool]] = None,
) -> float:
    p, t = _as_1d(pred), _as_1d(target)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        p, t = p[m], t[m]
    return mae(p, t)


def _masked_metric(fn, pred, target, mask: Optional[np.ndarray]) -> float:
    p, t = _as_1d(pred), _as_1d(target)
    if mask is not None:
        p, t = p[mask], t[mask]
    if len(p) == 0:
        return float("nan")
    return fn(p, t)


def aggregate_video_maes(video_maes: Mapping[str, float]) -> float:
    vals = [v for v in video_maes.values() if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_rsd_metrics_for_video(
    pred: Sequence[float],
    target: Sequence[float],
    windows: Sequence[float] = (2, 5, 10, 30),
) -> Dict[str, float]:
    p, t = _as_1d(pred), _as_1d(target)
    out = {
        "mae_all": mae(p, t),
        "rel_err_all": relative_error(p, t),
        "smape_all": smape(p, t),
    }
    for w in windows:
        m = window_mask(t, w)
        tag = f"{int(w) if float(w).is_integer() else w}"
        if m.any():
            out[f"mae_{tag}"] = video_mae(p, t, m)
            out[f"rel_err_{tag}"] = _masked_metric(relative_error, p, t, m)
            out[f"smape_{tag}"] = _masked_metric(smape, p, t, m)
        else:
            out[f"mae_{tag}"] = float("nan")
            out[f"rel_err_{tag}"] = float("nan")
            out[f"smape_{tag}"] = float("nan")
    return out


def compute_dataset_metrics(
    per_video: Mapping[str, Mapping[str, Sequence[float]]],
    windows: Sequence[float] = (2, 5, 10, 30),
) -> Dict[str, float]:
    """
    per_video[video_id] = {"pred": [...], "target": [...]}
    Returns video-macro means of each metric.
    """
    bucket: Dict[str, List[float]] = {}
    for _, seqs in per_video.items():
        m = compute_rsd_metrics_for_video(seqs["pred"], seqs["target"], windows=windows)
        for k, v in m.items():
            bucket.setdefault(k, []).append(v)
    return {k: float(np.nanmean(vs)) for k, vs in bucket.items()}


def metrics_from_prediction_rows(
    rows: Sequence[Mapping],
    windows: Sequence[float] = (2, 5, 10, 30),
    pred_col: str = "rsd_pred",
    true_col: str = "rsd_true",
    video_col: str = "video_id",
) -> Dict[str, float]:
    """Recompute metrics from saved predictions.csv rows (video-macro)."""
    per_video: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        vid = str(r[video_col])
        bucket = per_video.setdefault(vid, {"pred": [], "target": []})
        bucket["pred"].append(float(r[pred_col]))
        bucket["target"].append(float(r[true_col]))
    return compute_dataset_metrics(per_video, windows=windows)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def format_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def protocol_windows(configs: Mapping) -> List[float]:
    """Read metrics.windows from protocol, else infer from mae_* primary keys."""
    mcfg = configs.get("metrics", {}) if isinstance(configs, Mapping) else {}
    if isinstance(mcfg, Mapping) and mcfg.get("windows"):
        return [float(w) for w in mcfg["windows"]]
    primary = (mcfg.get("primary", []) if isinstance(mcfg, Mapping) else []) or []
    win_vals: List[float] = []
    for w in primary:
        if isinstance(w, str) and w.startswith("mae_") and w != "mae_all":
            try:
                win_vals.append(float(w.split("_", 1)[1]))
            except ValueError:
                pass
    return win_vals or [2, 5, 10, 30]


def information_gain(baseline_mae: float, model_mae: float) -> float:
    """G = MAE_base - MAE_model (minutes). Positive ⇒ model better than baseline."""
    if not (np.isfinite(baseline_mae) and np.isfinite(model_mae)):
        return float("nan")
    return float(baseline_mae - model_mae)
