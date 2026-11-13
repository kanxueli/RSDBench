"""Load pre-extracted per-video feature .pt files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset


class FeatureRSD(Dataset):
    def __init__(self, feature_dir: str | Path, video_ids: Sequence[Any]):
        self.feature_dir = Path(feature_dir)
        self.video_ids = list(video_ids)
        missing = [vid for vid in self.video_ids if not (self.feature_dir / f"{vid}.pt").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing feature files under {self.feature_dir}: {missing[:5]}..."
                f" ({len(missing)} missing). Run preprocessing/extract_cnn_features.py first."
            )

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        vid = self.video_ids[index]
        obj = torch.load(self.feature_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        out = {
            "video_id": obj["video_id"],
            "features": obj["features"].float(),
            "elapsed": obj["elapsed"].float(),
            "rsd": obj["rsd"].float(),
            "progress": obj["progress"].float(),
            "phase": obj["phase"].long(),
            "duration_min": float(obj["duration_min"]),
        }
        if "tools" in obj:
            out["tools"] = obj["tools"].float()
        if "surgeon_exp" in obj:
            out["surgeon_exp"] = int(obj["surgeon_exp"])
        return out

    def iter_meta(self):
        for i in range(len(self)):
            yield self[i]
