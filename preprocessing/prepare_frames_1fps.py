#!/usr/bin/env python
"""Cache 1fps square-cropped frames (uint8 NPY) for image-encoder methods."""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.finetune_cnn import DATASET_DIRS, build_video_plans
from utils.config import default_data_root

DEFAULT_OUT = ROOT / ".cache" / "frames_1fps_sq256"


def centre_square(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def cache_video(job) -> str:
    vid, vpath, idx, out_path, size = job
    if out_path.exists():
        return f"skip {vid}"
    want = {int(f): k for k, f in enumerate(idx.tolist())}
    frames = np.zeros((len(idx), size, size, 3), dtype=np.uint8)
    got = np.zeros(len(idx), dtype=bool)
    cap = cv2.VideoCapture(str(vpath))
    fidx, last = 0, int(idx.max())
    while fidx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in want:
            k = want[fidx]
            frames[k] = centre_square(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
            got[k] = True
        fidx += 1
    cap.release()
    # np.clip at the span end can repeat the last index; copy the decoded frame forward.
    for k in range(1, len(idx)):
        if not got[k] and idx[k] == idx[k - 1]:
            frames[k], got[k] = frames[k - 1], got[k - 1]
    if not got.all():
        raise RuntimeError(f"{vid}: decoded {int(got.sum())}/{len(idx)} frames from {vpath}")
    tmp = out_path.with_name(out_path.stem + ".partial.npy")
    np.save(tmp, frames)
    tmp.rename(out_path)
    return f"done {vid} T={len(idx)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True)
    ap.add_argument("--data_root", default=default_data_root())
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.data_root) / DATASET_DIRS[args.dataset]
    if args.dataset == "cataract101":
        ids = pd.read_csv(root / "videos.csv", sep=";")["VideoID"].astype(int).tolist()
    else:
        ids = [f"video{i:02d}" for i in range(1, 81)]
    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = build_video_plans(args.dataset, root, ids, sample_fps=1.0)
    jobs = [(vid, vpath, idx, out_dir / f"{vid}.npy", args.size) for vid, (vpath, idx, _, _) in plans.items()]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(cache_video, jobs):
            print(msg, flush=True)
    print(f"[frames] {len(jobs)} videos → {out_dir}")


if __name__ == "__main__":
    main()
