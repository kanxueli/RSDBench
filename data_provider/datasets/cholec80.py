"""Cholec80 RSD dataset (phase/tool annotations + derived RSD labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence  # Optional used for aspect_height

import numpy as np
import pandas as pd
import torch

from data_provider.datasets.base_rsd import (
    RSDVideoDataset,
    build_time_axis,
    elapsed_and_rsd_minutes,
)


PHASE_MAP = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}


class Cholec80RSD(RSDVideoDataset):
    def __init__(
        self,
        root: str | Path,
        video_ids: Sequence[str],
        split: str = "train",
        sample_fps: float = 1.0,
        source_fps: float = 25.0,
        return_frames: bool = False,
        frame_size: Sequence[int] = (224, 224),
        aspect_height: Optional[int] = None,
        frame_cache_dir: Optional[str | Path] = None,
    ):
        super().__init__(video_ids, split=split)
        self.root = Path(root)
        self.sample_fps = float(sample_fps)
        self.source_fps = float(source_fps)
        self.return_frames = return_frames
        self.frame_size = tuple(frame_size)
        # If set (e.g. 224), resize keeping aspect to this height (TransLocal ~398×224).
        self.aspect_height = int(aspect_height) if aspect_height else None
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.phase_dir = self.root / "phase_annotations"
        self.tool_dir = self.root / "tool_annotations"
        self.videos_dir = self.root / "videos"
        self._index = [self._load_meta(vid) for vid in self.video_ids]

    def _phase_path(self, vid: str) -> Path:
        return self.phase_dir / f"{vid}-phase.txt"

    def _tool_path(self, vid: str) -> Path:
        return self.tool_dir / f"{vid}-tool.txt"

    def _load_phase(self, vid: str) -> np.ndarray:
        path = self._phase_path(vid)
        df = pd.read_csv(path, sep="\t")
        # columns: Frame, Phase
        phases = []
        for p in df.iloc[:, 1].astype(str).tolist():
            if p in PHASE_MAP:
                phases.append(PHASE_MAP[p])
            else:
                # already numeric?
                try:
                    phases.append(int(p))
                except ValueError:
                    phases.append(PHASE_MAP.get(p.replace(" ", ""), 0))
        return np.asarray(phases, dtype=np.int64)

    def _load_tools(self, vid: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        path = self._tool_path(vid)
        if not path.exists():
            return None
        df = pd.read_csv(path, sep="\t")
        # Frame + 7 tools; annotations typically every 25 frames (1 fps)
        tool_frames = df.iloc[:, 0].to_numpy(dtype=np.int64)
        tools = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        return tool_frames, tools

    def _load_meta(self, vid: str) -> Dict[str, Any]:
        phase = self._load_phase(vid)
        n_src = len(phase)
        frame_idx = build_time_axis(n_src, self.source_fps, self.sample_fps)
        phase_s = phase[frame_idx]
        elapsed, rsd, duration = elapsed_and_rsd_minutes(len(frame_idx), self.sample_fps)
        tools_pack = self._load_tools(vid)
        tools_s = None
        if tools_pack is not None:
            tool_frames, tools = tools_pack
            # map each sampled source frame to nearest preceding tool annotation
            t_idx = np.searchsorted(tool_frames, frame_idx, side="right") - 1
            t_idx = np.clip(t_idx, 0, len(tools) - 1)
            tools_s = tools[t_idx]
        return {
            "video_id": vid,
            "frame_indices": frame_idx,
            "phase": phase_s,
            "tools": tools_s,
            "elapsed": elapsed.astype(np.float32),
            "rsd": rsd.astype(np.float32),
            "progress": (elapsed / max(duration, 1e-6)).astype(np.float32),
            "duration_min": float(duration),
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
        if meta["tools"] is not None:
            out["tools"] = torch.from_numpy(meta["tools"])
        if self.return_frames:
            out["frames"] = self._read_frames(meta["video_id"], meta["frame_indices"])
        return out

    def _cache_dir_for(self, vid: str) -> Optional[Path]:
        if self.frame_cache_dir is None or self.aspect_height:
            # aspect-preserving path is not stored in the square JPEG cache
            return None
        d = self.frame_cache_dir / vid
        return d if d.is_dir() else None

    def _read_frames_from_cache(self, cache_dir: Path, n: int) -> torch.Tensor:
        import cv2

        h, w = self.frame_size
        frames = []
        for k in range(n):
            path = cache_dir / f"{k:06d}.jpg"
            if not path.exists():
                raise FileNotFoundError(path)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                rgb = cv2.resize(rgb, (w, h))
            frames.append(rgb)
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def _read_frames(self, vid: str, frame_indices: np.ndarray) -> torch.Tensor:
        """Prefer JPEG frame cache; else sequential mp4 decode (seeks are pathologically slow)."""
        import cv2

        cache = self._cache_dir_for(vid)
        if cache is not None:
            n = int(len(frame_indices))
            marker = cache / f"{n-1:06d}.jpg"
            if marker.exists():
                return self._read_frames_from_cache(cache, n)

        path = self.videos_dir / f"{vid}.mp4"
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
            else:
                if self.aspect_height:
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
