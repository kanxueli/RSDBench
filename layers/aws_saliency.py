"""Adaptive whitening saliency (AWS) crop."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gabor_kernel_torch(ksize: int, sigma: float, theta: float, lambd: float, gamma: float = 0.5) -> torch.Tensor:
    half = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    gb = torch.exp(-(x_t**2 + (gamma * y_t) ** 2) / (2 * sigma**2)) * torch.cos(2 * np.pi * x_t / lambd)
    return gb


@lru_cache(maxsize=4)
def _bank_kernels(n_orient: int, scales: Tuple[float, ...], device: str) -> List[torch.Tensor]:
    """List of (1,1,K,K) kernels for depthwise-style filtering."""
    kernels = []
    thetas = [i * np.pi / n_orient for i in range(n_orient)]
    for sigma in scales:
        ksize = int(max(7, 2 * int(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        lambd = max(2.0, sigma * 1.5)
        for th in thetas:
            ker = _gabor_kernel_torch(ksize, sigma, th, lambd).to(device)
            kernels.append(ker.view(1, 1, ksize, ksize))
    return kernels


def _rgb_to_opponent_t(rgb: torch.Tensor) -> torch.Tensor:
    # rgb (N,3,H,W) in [0,1]
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    i = (r + g + b) / 3.0
    rg = (r - g) / 2.0 + 0.5
    by = (b - (r + g) / 2.0) / 2.0 + 0.5
    return torch.cat([i, rg, by], dim=1)


def _zca_whiten_t(feats: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # feats (N, D)
    x = feats - feats.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.size(0) - 1, 1)
    w, v = torch.linalg.eigh(cov)
    w = torch.clamp(w, min=0.0)
    inv_sqrt = v @ torch.diag(1.0 / torch.sqrt(w + eps)) @ v.T
    return x @ inv_sqrt.T


def aws_saliency_maps_torch(
    frames: torch.Tensor,
    n_orient: int = 6,
    scales: Sequence[float] = (3.0, 6.0),
    map_scale: float = 0.5,
) -> torch.Tensor:
    """
    frames: (T,3,H,W) float [0,1] on any device → saliency (T,H,W).
    Computes AWS on a downscaled map then upsamples (benchmark resize=0.5).
    """
    assert frames.dim() == 4 and frames.size(1) == 3
    t, _, h0, w0 = frames.shape
    device = frames.device
    if frames.max() > 1.5:
        frames = frames / 255.0
    nh, nw = max(16, int(h0 * map_scale)), max(16, int(w0 * map_scale))
    small = F.interpolate(frames, size=(nh, nw), mode="bilinear", align_corners=False)
    opp = _rgb_to_opponent_t(small)  # T,3,h,w
    sal = torch.zeros(t, nh, nw, device=device, dtype=frames.dtype)
    scales_t = tuple(float(s) for s in scales)
    # group kernels by scale for ZCA-per-scale
    thetas = [i * np.pi / n_orient for i in range(n_orient)]
    for sigma in scales_t:
        ksize = int(max(7, 2 * int(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        lambd = max(2.0, sigma * 1.5)
        bank = []
        for c in range(3):
            ch = opp[:, c : c + 1]  # T,1,h,w
            for th in thetas:
                ker = _gabor_kernel_torch(ksize, sigma, th, lambd).to(device=device, dtype=frames.dtype)
                ker = ker.view(1, 1, ksize, ksize)
                pad = ksize // 2
                resp = F.conv2d(ch, ker, padding=pad)
                bank.append(resp)
        vol = torch.cat(bank, dim=1)  # T, D, h, w
        d = vol.size(1)
        flat = vol.permute(0, 2, 3, 1).reshape(-1, d)
        # Whitening on pooled sample for speed: deterministic subsample
        # (randperm made AWS crop jitter across forwards and destabilized training).
        n = flat.size(0)
        if n > 8000:
            idx = torch.linspace(0, n - 1, 8000, device=device).long()
            sample = flat[idx]
        else:
            sample = flat
        # Fit ZCA on sample, apply to all
        mu = sample.mean(dim=0, keepdim=True)
        x = sample - mu
        cov = (x.T @ x) / max(x.size(0) - 1, 1)
        w, v = torch.linalg.eigh(cov)
        w = torch.clamp(w, min=0.0)
        inv_sqrt = v @ torch.diag(1.0 / torch.sqrt(w + 1e-5)) @ v.T
        white = (flat - mu) @ inv_sqrt.T
        energy = (white**2).sum(dim=1).view(t, nh, nw)
        sal = sal + energy
    # blur
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=device, dtype=frames.dtype) / 16.0
    k = k.view(1, 1, 3, 3)
    sal = F.conv2d(sal.unsqueeze(1), k, padding=1).squeeze(1)
    sal = sal - sal.amin(dim=(1, 2), keepdim=True)
    sal = sal / (sal.amax(dim=(1, 2), keepdim=True) + 1e-8)
    if (nh, nw) != (h0, w0):
        sal = F.interpolate(sal.unsqueeze(1), size=(h0, w0), mode="bilinear", align_corners=False).squeeze(1)
    return sal


def aws_crop_tensor_batch(frames: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """
    frames: (T,3,H,W) or (1,T,3,H,W) → AWS-cropped (same leading dims) out_size×out_size.
    """
    single_bt = False
    if frames.dim() == 5:
        assert frames.size(0) == 1
        frames = frames[0]
        single_bt = True
    assert frames.dim() == 4
    t, c, h, w = frames.shape
    device = frames.device
    sal = aws_saliency_maps_torch(frames)
    ch = cw = int(out_size)
    # ensure large enough
    if h < ch or w < cw:
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="replicate")
        sal = F.pad(sal, (0, pad_w, 0, pad_h), mode="replicate")
        h, w = frames.shape[-2:]
    # peak per frame
    flat = sal.view(t, -1)
    idx = flat.argmax(dim=1)
    cy = idx // w
    cx = idx % w
    y0 = torch.clamp(cy - ch // 2, 0, h - ch)
    x0 = torch.clamp(cx - cw // 2, 0, w - cw)
    outs = []
    for i in range(t):
        yy, xx = int(y0[i]), int(x0[i])
        outs.append(frames[i, :, yy : yy + ch, xx : xx + cw])
    out = torch.stack(outs, dim=0)
    if single_bt:
        out = out.unsqueeze(0)
    return out.to(device=device)


# numpy helpers kept for offline scripts
def aws_crop_image(rgb: np.ndarray, out_size: int = 224, target_height: int = 224) -> np.ndarray:
    if rgb.max() > 1.5:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    if h != target_height:
        nh = target_height
        nw = max(1, int(round(w * (nh / float(h)))))
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        rgb = t[0].permute(1, 2, 0).numpy()
    ten = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    crop = aws_crop_tensor_batch(ten, out_size=out_size)[0]
    return crop.permute(1, 2, 0).numpy()
