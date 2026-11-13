"""Shared 1fps frame loading for Rivoir / SUPR-GAN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from data_provider.data_factory import _data_root
from data_provider.datasets.base_rsd import elapsed_and_rsd_minutes
from preprocessing.finetune_cnn import build_video_plans
from preprocessing.prepare_frames_1fps import DEFAULT_OUT as NPY_OUT

SAMPLE_FPS = 1.0
ROOT = Path(__file__).resolve().parents[1]
JPG_FPS1 = ROOT / ".cache" / "frames"

# NPY mmap handles are cheap; JPEG float tensors are heavy — separate LRUs.
_NPY_CACHE: Dict[str, Any] = {}
_NPY_ORDER: List[str] = []
_NPY_CACHE_MAX = 64  # mmap refs only (same idea as PitRSDNet)
_JPG_CACHE: Dict[str, Any] = {}
_JPG_ORDER: List[str] = []
_JPG_CACHE_MAX = 2  # full float videos — keep tiny to avoid RAM thrash


def _jpg_dir(dataset: str, vid: Any) -> Path:
    # Cataract JPEG cache uses case_<id>/; Cholec uses videoXX or bare id.
    base = JPG_FPS1 / dataset / "fps1"
    cand = base / str(vid)
    if cand.is_dir():
        return cand
    alt = base / f"case_{vid}"
    if alt.is_dir():
        return alt
    return cand


def _npy_path(dataset: str, vid: Any, frame_dir: Optional[Path] = None) -> Path:
    base = frame_dir or (NPY_OUT / dataset)
    return Path(base) / f"{vid}.npy"


class _NpyFrames:
    """PitRSDNet-style NPY mmap: keep handle, convert only on slice/index."""

    __slots__ = ("_arr",)

    def __init__(self, path: str):
        self._arr = np.load(path, mmap_mode="r")  # THWC uint8

    def size(self, dim: int = 0) -> int:
        if dim != 0:
            raise IndexError(dim)
        return int(self._arr.shape[0])

    def _to_float_tchw(self, u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(u8, dtype=np.float32) * (1.0 / 255.0))
        return x.permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, idx) -> torch.Tensor:
        return self._to_float_tchw(self._arr[idx])

    def to(self, device, non_blocking: bool = False) -> torch.Tensor:
        return self._to_float_tchw(self._arr).to(device, non_blocking=non_blocking)


def resolve_frame_source(dataset: str, vid: Any, frame_dir: Optional[Path] = None) -> Dict[str, Any]:
    # Incremental: prefer the same NPY cache PitRSDNet already uses; JPEG remains fallback.
    np_path = _npy_path(dataset, vid, frame_dir)
    if np_path.exists():
        arr_n = int(np.load(np_path, mmap_mode="r").shape[0])
        return {"kind": "npy", "path": str(np_path), "n": arr_n}
    jd = _jpg_dir(dataset, vid)
    if jd.is_dir() and any(jd.glob("*.jpg")):
        files = sorted(jd.glob("*.jpg"))
        return {"kind": "jpg", "path": str(jd), "n": len(files), "files": [str(p) for p in files]}
    raise FileNotFoundError(
        f"No frame cache for {dataset}/{vid}. Expected {np_path} or {jd}/*.jpg"
    )


def load_videos(configs, ids) -> Dict[Any, Dict[str, Any]]:
    ds = configs["dataset"]
    frame_dir = Path(configs["frame_cache_1fps"]) / ds if configs.get("frame_cache_1fps") else None
    plans = build_video_plans(ds, _data_root(configs, ds), ids, sample_fps=SAMPLE_FPS)
    out = {}
    for vid, (_, idx, phase, _) in plans.items():
        src = resolve_frame_source(ds, vid, frame_dir)
        n = min(len(idx), src["n"], len(phase))
        elapsed, rsd, _ = elapsed_and_rsd_minutes(n, SAMPLE_FPS)
        out[vid] = {
            "frames": src,
            "phase": np.asarray(phase[:n], dtype=np.int64),
            "elapsed": elapsed.astype(np.float32),
            "rsd": rsd.astype(np.float32),
        }
    return out


def _lru_put(cache: Dict[str, Any], order: List[str], key: str, x: Any, max_items: int) -> Any:
    if key in cache:
        return cache[key]
    cache[key] = x
    order.append(key)
    while len(order) > max_items:
        old = order.pop(0)
        cache.pop(old, None)
    return x


def read_frames(src: Dict[str, Any] | str, device: torch.device | None = None):
    """CPU frames. NPY → mmap handle (PitRSDNet-style); JPEG → float tensor LRU."""
    del device  # frames must not sit on GPU when packing multiple seeds
    if isinstance(src, str):
        src = {"kind": "npy", "path": src}
    key = f"{src.get('kind')}:{src.get('path')}"
    if src["kind"] == "npy":
        hit = _NPY_CACHE.get(key)
        if hit is not None:
            return hit
        return _lru_put(_NPY_CACHE, _NPY_ORDER, key, _NpyFrames(src["path"]), _NPY_CACHE_MAX)
    hit = _JPG_CACHE.get(key)
    if hit is not None:
        return hit
    files = src.get("files")
    if not files:
        files = [str(p) for p in sorted(Path(src["path"]).glob("*.jpg"))]
    frames = [np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0 for fp in files]
    x = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()
    return _lru_put(_JPG_CACHE, _JPG_ORDER, key, x, _JPG_CACHE_MAX)


def gpu_window(frames_cpu, start: int, end: int, device: torch.device) -> torch.Tensor:
    """Copy only [start:end] to GPU (mmap slice for NPY)."""
    return frames_cpu[start:end].to(device, non_blocking=True)


def phase_refs_from_videos(videos: Dict[Any, Dict], ids: List, n_phases: int, stat: str = "median") -> np.ndarray:
    per = [[] for _ in range(n_phases)]
    for vid in ids:
        v = videos[vid]
        phases, elapsed = v["phase"], v["elapsed"]
        dt = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else 1.0 / 60.0
        for p in range(n_phases):
            per[p].append(float((phases == p).sum() * dt))
    refs = []
    for vals in per:
        if not vals:
            refs.append(0.0)
        elif stat == "mean":
            refs.append(float(np.mean(vals)))
        else:
            refs.append(float(np.median(vals)))
    return np.asarray(refs, dtype=np.float64)
