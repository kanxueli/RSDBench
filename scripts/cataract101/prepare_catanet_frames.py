#!/usr/bin/env python3
"""Prepare Cataract-101 frames/labels for CataNet (2.5 fps)."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import default_data_root

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def expand_phase(boundaries: list[tuple[int, int]], n: int) -> np.ndarray:
    """boundaries: (frame, phase1to10) sorted; return per-frame phase (0 if before first)."""
    phase = np.zeros(n, dtype=np.int64)
    if not boundaries:
        return phase
    for i, (f, p) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else n
        phase[f:end] = int(p)
    return phase


def extract_frames_cv2(
    v_path: Path,
    out_dir: Path,
    start_f: int,
    end_f: int,
    source_fps: float,
    sample_fps: float = 2.5,
    dim: int = 256,
) -> int:
    """Decode with OpenCV (no ffmpeg required). Returns number of PNGs written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vname = out_dir.name
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {v_path}")
    step = max(1, int(round(source_fps / sample_fps)))
    idxs = list(range(start_f, end_f + 1, step))
    n_out = 0
    for k, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fidx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)
        # Dataset expects 1-based %04d like ffmpeg
        out_path = out_dir / f"{vname}_{k + 1:04d}.png"
        cv2.imwrite(str(out_path), frame)
        n_out += 1
    cap.release()
    return n_out


def build_one(
    vid: int,
    video_path: Path,
    ann: pd.DataFrame,
    n_frames: int,
    source_fps: float,
    experience: int,
    out_root: Path,
    sample_fps: float = 2.5,
    force: bool = False,
) -> None:
    out_dir = out_root / str(vid)
    label_path = out_dir / f"{vid}.csv"
    if out_dir.exists() and label_path.exists() and any(out_dir.glob("*.png")) and not force:
        return

    sub = ann[ann.VideoID == vid].sort_values("FrameNo")
    boundaries = [(int(r.FrameNo), int(r.Phase)) for _, r in sub.iterrows()]
    if not boundaries:
        raise RuntimeError(f"no annotations for video {vid}")

    start_f = boundaries[0][0]
    end_f = max(boundaries[-1][0], start_f + 1)
    # include a short tail after last phase boundary (until video end, capped)
    end_f = min(n_frames - 1, max(end_f, boundaries[-1][0]))

    phase_src = expand_phase(boundaries, n_frames)
    # remaining source frames → minutes via /60/25 in train_rnn
    rsd_src = np.arange(n_frames - 1, -1, -1, dtype=np.int64)

    sample_factor = max(1, int(round(source_fps / sample_fps)))
    idxs = np.arange(start_f, end_f + 1, sample_factor, dtype=np.int64)
    steps = np.clip(phase_src[idxs], 0, 10)
    exp = np.full(len(idxs), int(experience), dtype=np.int64)
    rsd_frames = rsd_src[idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()

    n_png = extract_frames_cv2(
        video_path,
        out_dir,
        start_f=start_f,
        end_f=end_f,
        source_fps=source_fps,
        sample_fps=sample_fps,
    )
    if n_png == 0:
        raise RuntimeError(f"no frames extracted for {vid}")

    m = min(n_png, len(idxs))
    df = pd.DataFrame(
        {
            "step": steps[:m],
            "u1": np.zeros(m, dtype=np.int64),
            "experience": exp[:m],
            "u2": np.zeros(m, dtype=np.int64),
            "u3": np.zeros(m, dtype=np.int64),
            "rsd_frames": rsd_frames[:m],
        }
    )
    # DatasetCataract101: genfromtxt skip_header=1, [:,1:] drops index column
    df.to_csv(label_path, index=True)
    for p in sorted(out_dir.glob("*.png"))[m:]:
        p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, default=Path(default_data_root()) / "cataract-101")
    ap.add_argument("--out", type=Path, default=Path(".cache/catanet_cataract101/pool"))
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N videos")
    args = ap.parse_args()

    videos_csv = pd.read_csv(args.dataset_root / "videos.csv", sep=";")
    ann = pd.read_csv(args.dataset_root / "annotations.csv", sep=";")
    vdir = args.dataset_root / "videos"
    args.out.mkdir(parents=True, exist_ok=True)

    rows = list(videos_csv.itertuples(index=False))
    if args.limit > 0:
        rows = rows[: args.limit]

    for row in tqdm(rows, desc="prepare catanet"):
        vid = int(row.VideoID)
        # files named case_{id}.mp4
        cand = [vdir / f"case_{vid}.mp4", vdir / f"{vid}.mp4"]
        vpath = next((p for p in cand if p.exists()), None)
        if vpath is None:
            raise FileNotFoundError(f"missing video for {vid}")
        build_one(
            vid=vid,
            video_path=vpath,
            ann=ann,
            n_frames=int(row.Frames),
            source_fps=float(row.FPS),
            experience=int(row.Experience),
            out_root=args.out,
            sample_fps=float(args.fps),
            force=args.force,
        )
    print(f"done. pool={args.out}")


if __name__ == "__main__":
    main()
