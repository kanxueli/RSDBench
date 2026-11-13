#!/usr/bin/env python
"""Offline frame extraction to JPEG cache (training reads cache, not raw mp4 each epoch).

Cataract-101: sample axis matches Cataract101RSD (incision → end, not full file from 0).
Uses sequential decode (no CAP_PROP_POS_FRAMES seek).

Modes:
  - square (default): resize to HxW JPEG under fps{tag}/
  - aws: aspect-preserve height → AWS crop → square JPEG under fps{tag}_aws{out}/
    (TransLocal offline precompute; training sets use_aws=false)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.datasets.base_rsd import build_time_axis


def _cataract_sample_indices(
    videos_csv: Path,
    ann_csv: Path,
    vid: int,
    sample_fps: float,
    source_fps_default: float = 25.0,
) -> Tuple[np.ndarray, float]:
    """Same span logic as Cataract101RSD._load_meta (no corrected labels)."""
    vdf = pd.read_csv(videos_csv, sep=";")
    row = vdf[vdf["VideoID"] == vid].iloc[0]
    n_src = int(row.Frames)
    source_fps = float(row.FPS) if "FPS" in row.index else float(source_fps_default)
    ann = pd.read_csv(ann_csv, sep=";")
    sub = ann[ann["VideoID"] == vid].sort_values("FrameNo")
    start_f, end_f = 0, n_src - 1
    if len(sub):
        start_f = int(sub.iloc[0].FrameNo)
        end_f = n_src - 1
    span = max(1, end_f - start_f + 1)
    frame_idx = build_time_axis(span, source_fps, sample_fps) + start_f
    frame_idx = np.clip(frame_idx, 0, n_src - 1)
    return frame_idx.astype(np.int64), source_fps


def extract_one(
    video_path: Path,
    out_dir: Path,
    frame_indices: Sequence[int],
    size: tuple[int, int],
    jpeg_quality: int = 95,
) -> int:
    """Sequential decode; write sample i → {i:06d}.jpg (i = sample order, not source frame)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.size == 0:
        return 0
    # Skip if already complete
    if all((out_dir / f"{i:06d}.jpg").exists() for i in range(len(idx))):
        return len(idx)

    from collections import defaultdict

    src_to_samples = defaultdict(list)
    for i, f in enumerate(idx.tolist()):
        src_to_samples[int(f)].append(i)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    w, h = size
    max_need = int(idx.max())
    fidx = 0
    while fidx <= max_need:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in src_to_samples:
            if ok and frame is not None:
                rgb_bgr = frame  # OpenCV BGR; keep BGR for imwrite
                resized = cv2.resize(rgb_bgr, (w, h))
            else:
                resized = np.zeros((h, w, 3), dtype=np.uint8)
            for i in src_to_samples[fidx]:
                out = out_dir / f"{i:06d}.jpg"
                if not out.exists():
                    cv2.imwrite(str(out), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        fidx += 1
    cap.release()
    # Fill any missing with black
    for i in range(len(idx)):
        out = out_dir / f"{i:06d}.jpg"
        if not out.exists():
            cv2.imwrite(str(out), np.zeros((h, w, 3), dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return len(idx)


def extract_one_aws(
    video_path: Path,
    out_dir: Path,
    frame_indices: Sequence[int],
    aspect_height: int = 224,
    aws_crop: int = 224,
    jpeg_quality: int = 95,
    aws_batch: int = 8,
) -> int:
    """Sequential decode → aspect-preserve → AWS crop → {i:06d}.jpg (sample order)."""
    import torch

    from layers.aws_saliency import aws_crop_tensor_batch

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.size == 0:
        return 0
    if all((out_dir / f"{i:06d}.jpg").exists() for i in range(len(idx))):
        return len(idx)

    from collections import defaultdict

    src_to_samples = defaultdict(list)
    for i, f in enumerate(idx.tolist()):
        src_to_samples[int(f)].append(i)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    # sample_i → RGB uint8 aspect-resized (variable width)
    buf: dict[int, np.ndarray] = {}
    max_need = int(idx.max())
    fidx = 0
    nh = int(aspect_height)
    while fidx <= max_need:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in src_to_samples:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            nw = max(1, int(round(rgb.shape[1] * (nh / float(rgb.shape[0])))))
            rgb = cv2.resize(rgb, (nw, nh))
            for i in src_to_samples[fidx]:
                buf[i] = rgb
        fidx += 1
    cap.release()

    # Pad missing
    default_w = int(round(nh * 16 / 9))
    for i in range(len(idx)):
        if i not in buf:
            buf[i] = np.zeros((nh, default_w, 3), dtype=np.uint8)

    # Pad all to max_w (same as Cataract101RSD aspect path), then AWS in batches
    max_w = max(im.shape[1] for im in buf.values())
    order = list(range(len(idx)))
    crops_bgr: dict[int, np.ndarray] = {}
    for start in range(0, len(order), int(aws_batch)):
        chunk = order[start : start + int(aws_batch)]
        tensors = []
        for i in chunk:
            im = buf[i]
            if im.shape[1] < max_w:
                pad = np.zeros((nh, max_w, 3), dtype=np.uint8)
                pad[:, : im.shape[1]] = im
                im = pad
            tensors.append(torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1))
        batch = torch.stack(tensors, dim=0)  # (B,3,H,W)
        with torch.no_grad():
            cropped = aws_crop_tensor_batch(batch, out_size=int(aws_crop))
        for j, i in enumerate(chunk):
            rgb = (cropped[j].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            crops_bgr[i] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for i in order:
        out = out_dir / f"{i:06d}.jpg"
        if not out.exists():
            cv2.imwrite(str(out), crops_bgr[i], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return len(idx)


def _job_cataract(args: tuple) -> str:
    vp, out_dir, videos_csv, ann_csv, sample_fps, size, jpeg_quality = args
    vid = int(vp.stem.split("_")[-1])
    frame_idx, _ = _cataract_sample_indices(Path(videos_csv), Path(ann_csv), vid, sample_fps)
    n = extract_one(vp, Path(out_dir), frame_idx, size=size, jpeg_quality=jpeg_quality)
    return f"{vp.stem}: {n} frames -> {out_dir}"


def _job_cataract_aws(args: tuple) -> str:
    vp, out_dir, videos_csv, ann_csv, sample_fps, aspect_height, aws_crop, jpeg_quality, aws_batch = args
    vid = int(vp.stem.split("_")[-1])
    frame_idx, _ = _cataract_sample_indices(Path(videos_csv), Path(ann_csv), vid, sample_fps)
    n = extract_one_aws(
        vp,
        Path(out_dir),
        frame_idx,
        aspect_height=int(aspect_height),
        aws_crop=int(aws_crop),
        jpeg_quality=int(jpeg_quality),
        aws_batch=int(aws_batch),
    )
    return f"{vp.stem}: {n} aws-frames -> {out_dir}"


def _job_cholec(args: tuple) -> str:
    vp, out_dir, sample_fps, source_fps, size, jpeg_quality = args
    cap = cv2.VideoCapture(str(vp))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_idx = build_time_axis(n_src, source_fps, sample_fps)
    n = extract_one(vp, Path(out_dir), frame_idx, size=size, jpeg_quality=jpeg_quality)
    return f"{vp.stem}: {n} frames -> {out_dir}"


def _job_cholec_aws(args: tuple) -> str:
    vp, out_dir, sample_fps, source_fps, aspect_height, aws_crop, jpeg_quality, aws_batch = args
    cap = cv2.VideoCapture(str(vp))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_idx = build_time_axis(n_src, source_fps, sample_fps)
    n = extract_one_aws(
        vp,
        Path(out_dir),
        frame_idx,
        aspect_height=int(aspect_height),
        aws_crop=int(aws_crop),
        jpeg_quality=int(jpeg_quality),
        aws_batch=int(aws_batch),
    )
    return f"{vp.stem}: {n} aws-frames -> {out_dir}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cholec80", "cataract101"], required=True)
    p.add_argument("--data_root", type=str, default=str(ROOT.parent / "DATASET"))
    p.add_argument("--cache_root", type=str, default=str(ROOT / ".cache" / "frames"))
    p.add_argument("--sample_fps", type=float, default=1.0)
    p.add_argument("--source_fps", type=float, default=25.0)
    p.add_argument("--size", type=int, nargs=2, default=[224, 224], help="train frame_size; avoid double resize")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="optional max videos for dry-run")
    p.add_argument("--mode", choices=["square", "aws"], default="square")
    p.add_argument("--aspect_height", type=int, default=224, help="AWS mode: aspect-preserve height")
    p.add_argument("--aws_crop", type=int, default=224, help="AWS mode: output crop size")
    p.add_argument("--aws_batch", type=int, default=8, help="AWS mode: frames per AWS forward")
    args = p.parse_args()

    size = (int(args.size[0]), int(args.size[1]))
    fps_tag = f"{args.sample_fps:g}"  # 1 -> fps1, 2.5 -> fps2.5
    aws_mode = args.mode == "aws"

    if args.dataset == "cholec80":
        vdir = Path(args.data_root) / "cholec80" / "videos"
        videos = sorted(vdir.glob("video*.mp4"))
        if aws_mode:
            out_root = Path(args.cache_root) / "cholec80" / f"fps{fps_tag}_aws{int(args.aws_crop)}"
            jobs = [
                (
                    vp,
                    str(out_root / vp.stem),
                    args.sample_fps,
                    args.source_fps,
                    args.aspect_height,
                    args.aws_crop,
                    args.jpeg_quality,
                    args.aws_batch,
                )
                for vp in videos
            ]
            worker = _job_cholec_aws
        else:
            out_root = Path(args.cache_root) / "cholec80" / f"fps{fps_tag}"
            jobs = [
                (vp, str(out_root / vp.stem), args.sample_fps, args.source_fps, size, args.jpeg_quality)
                for vp in videos
            ]
            worker = _job_cholec
    else:
        root = Path(args.data_root) / "cataract-101"
        vdir = root / "videos"
        videos = sorted(vdir.glob("case_*.mp4"))
        videos_csv = str(root / "videos.csv")
        ann_csv = str(root / "annotations.csv")
        if aws_mode:
            out_root = Path(args.cache_root) / "cataract101" / f"fps{fps_tag}_aws{int(args.aws_crop)}"
            jobs = [
                (
                    vp,
                    str(out_root / vp.stem),
                    videos_csv,
                    ann_csv,
                    args.sample_fps,
                    args.aspect_height,
                    args.aws_crop,
                    args.jpeg_quality,
                    args.aws_batch,
                )
                for vp in videos
            ]
            worker = _job_cataract_aws
        else:
            out_root = Path(args.cache_root) / "cataract101" / f"fps{fps_tag}"
            jobs = [
                (vp, str(out_root / vp.stem), videos_csv, ann_csv, args.sample_fps, size, args.jpeg_quality)
                for vp in videos
            ]
            worker = _job_cataract

    if args.limit > 0:
        jobs = jobs[: args.limit]

    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[extract] dataset={args.dataset} mode={args.mode} n={len(jobs)} "
        f"fps={args.sample_fps} -> {out_root}",
        flush=True,
    )

    if args.workers <= 1:
        for job in tqdm(jobs, desc="extract"):
            print(worker(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, job) for job in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
                print(fut.result(), flush=True)
    ready = out_root / ".READY"
    ready.write_text(f"mode={args.mode}\nn={len(jobs)}\n", encoding="utf-8")
    print(f"[extract] done -> {ready}", flush=True)


if __name__ == "__main__":
    main()
