"""Elapsed-time as 4th image channel (CataNet / PitRSDNet / BD-Net style)."""

from __future__ import annotations

import torch


def append_elapsed_channel(
    frames: torch.Tensor,
    elapsed_min: torch.Tensor,
    max_len_min: float,
) -> torch.Tensor:
    """
    frames: (B,T,3,H,W) or (T,3,H,W)
    elapsed_min: (B,T) or (T,)
    returns frames with channel dim 4, values in [0,1]
    """
    squeeze = False
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
        elapsed_min = elapsed_min.unsqueeze(0)
        squeeze = True
    rel = (elapsed_min / float(max_len_min)).clamp(0, 1)
    rel = rel.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    rel = rel.expand(-1, -1, 1, frames.shape[-2], frames.shape[-1])
    out = torch.cat([frames, rel.to(frames.dtype)], dim=2)
    return out.squeeze(0) if squeeze else out
