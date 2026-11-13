"""Shared RSD sample contract and label helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class VideoMeta:
    video_id: Any
    n_frames_src: int
    source_fps: float
    sample_fps: float
    duration_min: float
    phase: Optional[np.ndarray] = None          # per sampled frame
    tools: Optional[np.ndarray] = None
    surgeon_exp: Optional[int] = None
    # absolute frame indices in source video for each sample
    frame_indices: Optional[np.ndarray] = None


def build_time_axis(n_src: int, source_fps: float, sample_fps: float) -> np.ndarray:
    """Return source frame indices sampled at sample_fps."""
    duration_s = n_src / float(source_fps)
    n_out = max(1, int(np.floor(duration_s * sample_fps)))
    # center of each sample bin
    t_s = (np.arange(n_out) + 0.5) / float(sample_fps)
    idx = np.clip((t_s * source_fps).astype(np.int64), 0, n_src - 1)
    return idx


def elapsed_and_rsd_minutes(
    n_samples: int,
    sample_fps: float,
    duration_min: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Elapsed time at sample i is i / sample_fps / 60 minutes (start at 0).
    If duration_min is None, use last sample time as T.
    """
    elapsed = np.arange(n_samples, dtype=np.float64) / float(sample_fps) / 60.0
    if duration_min is None:
        duration_min = float(elapsed[-1]) if n_samples else 0.0
    rsd = np.maximum(0.0, duration_min - elapsed)
    return elapsed, rsd, float(duration_min)


def expand_phase_boundaries(
    boundaries: Sequence[tuple[int, int]],
    n_src: int,
) -> np.ndarray:
    """
    boundaries: list of (start_frame, phase_id) sorted by start_frame.
    Returns per-source-frame phase labels.
    """
    labs = np.zeros(n_src, dtype=np.int64)
    if not boundaries:
        return labs
    b = sorted(boundaries, key=lambda x: x[0])
    for i, (start, ph) in enumerate(b):
        end = b[i + 1][0] if i + 1 < len(b) else n_src
        start = max(0, min(int(start), n_src))
        end = max(start, min(int(end), n_src))
        labs[start:end] = int(ph)
    return labs


class RSDVideoDataset(Dataset):
    """
    Minimal interface: __getitem__ returns a dict with the RSDBench contract.

    Concrete datasets may return either:
      - full-video tensors (baselines / RNN), or
      - clip/window samples (large video models).
    """

    def __init__(self, video_ids: Sequence[Any], split: str = "train"):
        self.video_ids = list(video_ids)
        self.split = split

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError


def collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate for batch_size=1 video dicts; stacks when shapes match."""
    if len(batch) == 1:
        return batch[0]
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                out[k] = vals
        else:
            out[k] = vals
    return out
