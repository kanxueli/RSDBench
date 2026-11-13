"""Cataract-101 RSD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from data_provider.datasets.base_rsd import (
    RSDVideoDataset,
    build_time_axis,
    elapsed_and_rsd_minutes,
    expand_phase_boundaries,
)


class Cataract101RSD(RSDVideoDataset):
    def __init__(
        self,
        root: str | Path,
        video_ids: Sequence[int],
        split: str = "train",
        sample_fps: float = 1.0,
        source_fps: float = 25.0,
        return_frames: bool = False,
        frame_size: Sequence[int] = (224, 224),
        corrected_labels_dir: Optional[str | Path] = None,
        aspect_height: Optional[int] = None,
        frame_cache_dir: Optional[str | Path] = None,
    ):
        super().__init__(list(video_ids), split=split)
        self.root = Path(root)
        self.sample_fps = float(sample_fps)
        self.source_fps = float(source_fps)
        self.return_frames = return_frames
        self.frame_size = tuple(frame_size)
        self.aspect_height = int(aspect_height) if aspect_height else None
        # JPEG cache (square / AWS-precomputed). Skipped when aspect_height is set
        # (online TransLocal path). data_factory clears aspect_height when aws_precomputed.
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.corrected_labels_dir = Path(corrected_labels_dir) if corrected_labels_dir else None
        self.videos_dir = self.root / "videos"
        self.videos_csv = pd.read_csv(self.root / "videos.csv", sep=";")
        self.ann = pd.read_csv(self.root / "annotations.csv", sep=";")
        self._meta_map = {int(r.VideoID): r for _, r in self.videos_csv.iterrows()}
        self._index = [self._load_meta(int(vid)) for vid in self.video_ids]

    def _load_meta(self, vid: int) -> Dict[str, Any]:
        row = self._meta_map[vid]
        n_src = int(row.Frames)
        source_fps = float(row.FPS) if "FPS" in row else self.source_fps
        surgeon_exp = int(row.Experience) if "Experience" in row else None
        surgeon_id = int(row.Surgeon) if "Surgeon" in row else None

        # Phase boundaries from annotations.csv (start frames)
        sub = self.ann[self.ann["VideoID"] == vid].sort_values("FrameNo")
        # annotations.csv Phase is 1..10; store 0-indexed for model heads / PhaseInferred
        boundaries = [(int(r.FrameNo), int(r.Phase) - 1) for _, r in sub.iterrows()]
        phase_src = expand_phase_boundaries(boundaries, n_src)

        # Endpoint: first annotated phase start → last frame (fallback phase_span).
        # When corrected labels exist, prefer their start/end.
        start_f, end_f = 0, n_src - 1
        if boundaries:
            start_f = boundaries[0][0]
            end_f = n_src - 1
        if self.corrected_labels_dir is not None:
            # Placeholder: corrected CSV per video if provided later
            corr = self.corrected_labels_dir / f"{vid}.csv"
            if corr.exists():
                cdf = pd.read_csv(corr)
                if "start" in cdf.columns and "end" in cdf.columns:
                    start_f = int(cdf["start"].iloc[0])
                    end_f = int(cdf["end"].iloc[0])

        # Sample only inside surgical span
        span = max(1, end_f - start_f + 1)
        # Build indices relative to full video then clip to span
        frame_idx = build_time_axis(span, source_fps, self.sample_fps) + start_f
        frame_idx = np.clip(frame_idx, 0, n_src - 1)
        phase_s = phase_src[frame_idx]
        elapsed, rsd, duration = elapsed_and_rsd_minutes(len(frame_idx), self.sample_fps)

        return {
            "video_id": vid,
            "frame_indices": frame_idx,
            "phase": phase_s,
            "elapsed": elapsed.astype(np.float32),
            "rsd": rsd.astype(np.float32),
            "progress": (elapsed / max(duration, 1e-6)).astype(np.float32),
            "duration_min": float(duration),
            "surgeon_exp": surgeon_exp,
            "surgeon": surgeon_id,
            "n_src": n_src,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta = self._index[index]
        out = {
            "video_id": meta["video_id"],
            "elapsed": torch.from_numpy(meta["elapsed"]),
            "rsd": torch.from_numpy(meta["rsd"]),
            "progress": torch.from_numpy(meta["progress"]),
            "phase": torch.from_numpy(meta["phase"]),
            "duration_min": meta["duration_min"],
            "frame_indices": torch.from_numpy(meta["frame_indices"].astype(np.int64)),
        }
        if meta["surgeon_exp"] is not None:
            out["surgeon_exp"] = int(meta["surgeon_exp"])
        if meta.get("surgeon") is not None:
            out["surgeon"] = int(meta["surgeon"])
        if self.return_frames:
            out["frames"] = self._read_frames(meta["video_id"], meta["frame_indices"])
        return out

    def _cache_ready(self, vid: int, n: int) -> bool:
        if self.frame_cache_dir is None or self.aspect_height:
            return False
        d = self.frame_cache_dir / f"case_{vid}"
        if not d.is_dir():
            return False
        # require first/last + midpoint to exist (full check is expensive)
        for i in (0, max(0, n // 2), max(0, n - 1)):
            if not (d / f"{i:06d}.jpg").is_file():
                return False
        return True

    def _read_frames_from_cache(self, vid: int, n: int) -> torch.Tensor:
        import cv2

        h, w = self.frame_size
        d = self.frame_cache_dir / f"case_{vid}"
        frames = []
        for i in range(n):
            path = d / f"{i:06d}.jpg"
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            frames.append(img)
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def _read_frames(self, vid: int, frame_indices: np.ndarray) -> torch.Tensor:
        """Prefer JPEG cache; else sequential mp4 decode. aspect_height → always mp4."""
        import cv2

        n = int(len(frame_indices))
        if self._cache_ready(vid, n):
            return self._read_frames_from_cache(vid, n)

        path = self.videos_dir / f"case_{vid}.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        h, w = self.frame_size
        want = {int(i): k for k, i in enumerate(frame_indices.tolist())}
        max_need = int(frame_indices.max()) if len(frame_indices) else -1
        buf = {}
        fidx = 0
        while fidx <= max_need:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.aspect_height:
                    nh = int(self.aspect_height)
                    nw = max(1, int(round(frame.shape[1] * (nh / float(frame.shape[0])))))
                    frame = cv2.resize(frame, (nw, nh))
                else:
                    frame = cv2.resize(frame, (w, h))
                buf[fidx] = frame
            fidx += 1
        cap.release()
        frames = []
        for i in frame_indices.tolist():
            if int(i) in buf:
                frames.append(buf[int(i)])
            elif self.aspect_height:
                frames.append(
                    np.zeros((self.aspect_height, int(round(self.aspect_height * 16 / 9)), 3), dtype=np.uint8)
                )
            else:
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        if self.aspect_height and frames:
            max_w = max(f.shape[1] for f in frames)
            nh = int(self.aspect_height)
            padded = []
            for f in frames:
                if f.shape[1] < max_w:
                    pad = np.zeros((nh, max_w, 3), dtype=f.dtype)
                    pad[:, : f.shape[1]] = f
                    padded.append(pad)
                else:
                    padded.append(f)
            frames = padded
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def iter_meta(self) -> List[Dict[str, Any]]:
        return list(self._index)
