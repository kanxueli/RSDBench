"""Dataset / DataLoader factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from torch.utils.data import DataLoader

from data_provider.datasets.base_rsd import collate_video_batch
from data_provider.datasets.cataract101 import Cataract101RSD
from data_provider.datasets.cholec80 import Cholec80RSD


def _data_root(cfg: Mapping[str, Any], dataset: str) -> Path:
    block = cfg.get(dataset, {}) if isinstance(cfg.get(dataset), Mapping) else {}
    root = block.get("root") or cfg.get("data_root")
    if not root:
        raise KeyError(f"Missing data root for dataset={dataset}")
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Dataset root not found: {path}")
    return path


def build_dataset(
    cfg: Mapping[str, Any],
    video_ids: Sequence[Any],
    split: str,
    return_frames: bool = False,
):
    dataset = cfg["dataset"]
    sample_fps = float(cfg.get("sampling", {}).get("fps", 1.0))
    source_fps = float(cfg.get("sampling", {}).get("source_fps", 25.0))
    root = _data_root(cfg, dataset)

    mcfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    aspect_height = mcfg.get("aspect_height") or cfg.get("aspect_height")
    frame_size = tuple(mcfg.get("frame_size") or cfg.get("frame_size") or (224, 224))

    # Optional JPEG frame cache:
    #   square: {cache_root}/frames/{dataset}/fps{sample_fps:g}/
    #   AWS precompute (TransLocal): .../fps{tag}_aws{out}/  (clears aspect_height)
    fc = cfg.get("frame_cache") if isinstance(cfg.get("frame_cache"), Mapping) else {}
    frame_cache_dir = None
    aws_pre = bool(fc.get("aws_precomputed", False))
    if bool(fc.get("enabled", False)):
        cache_root = Path(str(cfg.get("cache_root") or (Path(__file__).resolve().parents[1] / ".cache")))
        if aws_pre:
            aws_out = int(fc.get("aws_crop") or mcfg.get("aws_crop") or 224)
            frame_cache_dir = cache_root / "frames" / dataset / f"fps{sample_fps:g}_aws{aws_out}"
            # Cache already holds aspect→AWS 224² crops; do not re-aspect in dataset.
            aspect_height = None
        else:
            frame_cache_dir = cache_root / "frames" / dataset / f"fps{sample_fps:g}"

    if dataset == "cholec80":
        kwargs = dict(
            root=root,
            video_ids=list(video_ids),
            split=split,
            sample_fps=sample_fps,
            source_fps=source_fps,
            return_frames=return_frames,
            frame_size=frame_size,
            aspect_height=aspect_height,
        )
        import inspect

        sig = inspect.signature(Cholec80RSD.__init__)
        if "frame_cache_dir" in sig.parameters and frame_cache_dir is not None:
            kwargs["frame_cache_dir"] = frame_cache_dir
        return Cholec80RSD(**kwargs)
    if dataset == "cataract101":
        corr = cfg.get("cataract101", {}).get("corrected_labels_dir")
        kwargs = dict(
            root=root,
            video_ids=[int(v) for v in video_ids],
            split=split,
            sample_fps=sample_fps,
            source_fps=source_fps,
            return_frames=return_frames,
            corrected_labels_dir=corr,
        )
        import inspect

        from data_provider.datasets.cataract101 import Cataract101RSD as _C

        sig = inspect.signature(_C.__init__)
        if "aspect_height" in sig.parameters:
            kwargs["aspect_height"] = aspect_height
        if "frame_size" in sig.parameters:
            kwargs["frame_size"] = frame_size
        if "frame_cache_dir" in sig.parameters and frame_cache_dir is not None:
            kwargs["frame_cache_dir"] = frame_cache_dir
        return _C(**kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


def build_dataloader(
    dataset,
    cfg: Mapping[str, Any],
    shuffle: bool = False,
) -> DataLoader:
    dl_cfg = cfg.get("dataloader", {})
    train_cfg = cfg  # batch_size may sit at top-level from train yaml
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(dl_cfg.get("num_workers", train_cfg.get("num_workers", 0))),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        collate_fn=collate_video_batch,
    )
