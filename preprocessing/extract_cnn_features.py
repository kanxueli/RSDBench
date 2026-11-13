#!/usr/bin/env python
"""Extract frozen CNN features at protocol fps (GPU). Sequential decode (fast)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.datasets.base_rsd import build_time_axis, elapsed_and_rsd_minutes, expand_phase_boundaries
from data_provider.datasets.cholec80 import PHASE_MAP
from utils.config import default_cache_root, default_data_root


def build_backbone(name: str, checkpoint: str | None = None):
    """Build feature extractor. Prefer method-specific fine-tuned checkpoint over raw ImageNet.

    Never constructs a randomly initialized encoder.
    """
    from utils.pretrained import load_torchvision_with_weights, require_checkpoint_file

    if name == "resnet50":
        require_checkpoint_file("resnet50")
        m = load_torchvision_with_weights(models.resnet50, models.ResNet50_Weights.DEFAULT, "resnet50")
    elif name == "resnet152":
        require_checkpoint_file("resnet152")
        m = load_torchvision_with_weights(models.resnet152, models.ResNet152_Weights.DEFAULT, "resnet152")
    elif name == "alexnet_siamese":
        # VTD-Net V-Net trunk: AlexNet up to fc7 (4096-d), weights from preprocessing/siamese_pretrain_vtd.py.
        if not checkpoint:
            raise ValueError("alexnet_siamese needs --checkpoint .../siamese_alexnet.pth")
        require_checkpoint_file("alexnet")
        m = load_torchvision_with_weights(models.alexnet, models.AlexNet_Weights.DEFAULT, "alexnet")
        m.classifier = torch.nn.Sequential(*list(m.classifier.children())[:-1])
        pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing = m.load_state_dict(pack["trunk_state_dict"], strict=True)
        print(f"[extract] AlexNet Siamese trunk from {checkpoint} ({missing})")
        return m
    elif name == "densenet169":
        require_checkpoint_file("densenet169")
        m = load_torchvision_with_weights(models.densenet169, models.DenseNet169_Weights.DEFAULT, "densenet169")
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            m.load_state_dict(state, strict=False)
        m.classifier = torch.nn.Identity()
        return m
    else:
        raise ValueError(name)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # Fine-tuned heads (phase/progress) are dropped; keep convolutional trunk.
        filtered = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing = m.load_state_dict(filtered, strict=False)
        print(f"[extract] loaded trunk from {checkpoint} missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    m.fc = torch.nn.Identity()
    return m


def load_phase_cholec(path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    phases = []
    for p in df.iloc[:, 1].astype(str).tolist():
        phases.append(PHASE_MAP.get(p, PHASE_MAP.get(p.replace(" ", ""), 0)))
    return np.asarray(phases, dtype=np.int64)


def load_tools_cholec(path: Path):
    import pandas as pd

    if not path.exists():
        return None, None
    df = pd.read_csv(path, sep="\t")
    return df.iloc[:, 0].to_numpy(np.int64), df.iloc[:, 1:].to_numpy(np.float32)


@torch.no_grad()
def encode_batch(model, frames_np, device, mean, std):
    # frames_np: list of RGB uint8 HxWxC
    x = np.stack(frames_np, axis=0).astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    t = (t - mean) / std
    return model(t).cpu()


def extract_from_frame_cache(
    cache_vid: Path,
    n: int,
    model,
    device,
    size=(224, 224),
    chunk=32,
):
    """Read sample-indexed JPEGs 000000..(n-1).jpg (same as extract_frames.py)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    w, h = size
    ordered = []
    for k in range(n):
        path = cache_vid / f"{k:06d}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            ordered.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h))
        ordered.append(rgb)
    feats = []
    for s in range(0, len(ordered), chunk):
        feats.append(encode_batch(model, ordered[s : s + chunk], device, mean, std))
    return torch.cat(feats, dim=0)


def extract_video_sequential(
    video_path: Path,
    want_idx: np.ndarray,
    model,
    device,
    size=(224, 224),
    chunk=32,
    frame_cache_dir: Path | None = None,
    vid: str | None = None,
):
    n = len(want_idx)
    if frame_cache_dir is not None and vid is not None and n > 0:
        cache_vid = Path(frame_cache_dir) / vid
        if (cache_vid / f"{n - 1:06d}.jpg").exists() and (cache_vid / "000000.jpg").exists():
            return extract_from_frame_cache(cache_vid, n, model, device, size=size, chunk=chunk)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    want_set = set(int(i) for i in want_idx.tolist())
    max_need = int(want_idx.max()) if len(want_idx) else -1
    buf = {}
    fidx = 0
    while fidx <= max_need:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in want_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            buf[fidx] = frame
        fidx += 1
    cap.release()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ordered = []
    for i in want_idx.tolist():
        if int(i) in buf:
            ordered.append(buf[int(i)])
        else:
            ordered.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))

    feats = []
    for s in range(0, len(ordered), chunk):
        feats.append(encode_batch(model, ordered[s : s + chunk], device, mean, std))
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cholec80", "cataract101"], required=True)
    ap.add_argument(
        "--data_root",
        default=default_data_root(),
        help="dataset root (env RSDBENCH_DATA_ROOT, else <repo_parent>/DATASET)",
    )
    ap.add_argument(
        "--out_root",
        default=str(Path(default_cache_root()) / "features"),
        help="feature output root (under env RSDBENCH_CACHE_ROOT when set)",
    )
    ap.add_argument("--backbone", default="resnet152")
    ap.add_argument("--checkpoint", default=None, help="method-specific fine-tuned CNN (.pth)")
    ap.add_argument("--tag", default=None, help="output subdir tag, e.g. TimeLSTM_phaseFT")
    ap.add_argument("--sample_fps", type=float, default=1.0)
    ap.add_argument("--source_fps", type=float, default=25.0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing .pt feature files",
    )
    ap.add_argument(
        "--frame_cache",
        type=str,
        default="",
        help="JPEG frame cache dir (empty = auto <cache_root>/frames/{dataset}/fps{sample_fps})",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device(args.device)
    model = build_backbone(args.backbone, checkpoint=args.checkpoint).to(device).eval()

    tag = args.tag or (f"{args.backbone}_fps{args.sample_fps}" if not args.checkpoint else f"{args.backbone}_FT_fps{args.sample_fps}")
    out_dir = Path(args.out_root) / args.dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_cache.strip():
        frame_cache_dir = Path(args.frame_cache.strip())
    else:
        frame_cache_dir = (
            Path(default_cache_root()) / "frames" / args.dataset / f"fps{args.sample_fps:g}"
        )
    if not frame_cache_dir.is_dir():
        print(f"[warn] frame_cache missing ({frame_cache_dir}); falling back to mp4", flush=True)
        frame_cache_dir = None
    print(f"[extract] writing features → {out_dir}")
    print(f"[extract] frame_cache={'on:' + str(frame_cache_dir) if frame_cache_dir else 'off(mp4)'}", flush=True)

    if args.dataset == "cholec80":
        root = Path(args.data_root) / "cholec80"
        ids = [f"video{i:02d}" for i in range(1, 81)]
        if args.limit > 0:
            ids = ids[: args.limit]
        for vid in tqdm(ids, desc="extract"):
            path = out_dir / f"{vid}.pt"
            if path.exists() and not args.force:
                continue
            vpath = root / "videos" / f"{vid}.mp4"
            phase = load_phase_cholec(root / "phase_annotations" / f"{vid}-phase.txt")
            n_src = len(phase)
            frame_idx = build_time_axis(n_src, args.source_fps, args.sample_fps)
            feat = extract_video_sequential(
                vpath,
                frame_idx,
                model,
                device,
                size=(args.width, args.height),
                chunk=args.chunk,
                frame_cache_dir=frame_cache_dir,
                vid=vid,
            )
            elapsed, rsd, duration = elapsed_and_rsd_minutes(len(frame_idx), args.sample_fps)
            phase_s = phase[frame_idx]
            tool_frames, tools = load_tools_cholec(root / "tool_annotations" / f"{vid}-tool.txt")
            tools_s = None
            if tools is not None:
                t_idx = np.clip(np.searchsorted(tool_frames, frame_idx, side="right") - 1, 0, len(tools) - 1)
                tools_s = tools[t_idx]
            payload = {
                "video_id": vid,
                "features": feat,
                "elapsed": torch.from_numpy(elapsed.astype(np.float32)),
                "rsd": torch.from_numpy(rsd.astype(np.float32)),
                "progress": torch.from_numpy((elapsed / max(duration, 1e-6)).astype(np.float32)),
                "phase": torch.from_numpy(phase_s.astype(np.int64)),
                "duration_min": float(duration),
            }
            if tools_s is not None:
                payload["tools"] = torch.from_numpy(tools_s)
            torch.save(payload, path)
    else:
        import pandas as pd

        root = Path(args.data_root) / "cataract-101"
        meta = pd.read_csv(root / "videos.csv", sep=";")
        ann = pd.read_csv(root / "annotations.csv", sep=";")
        ids = meta["VideoID"].astype(int).tolist()
        if args.limit > 0:
            ids = ids[: args.limit]
        for vid in tqdm(ids, desc="extract"):
            path = out_dir / f"{vid}.pt"
            if path.exists():
                continue
            row = meta[meta.VideoID == vid].iloc[0]
            n_src = int(row.Frames)
            source_fps = float(row.FPS)
            sub = ann[ann.VideoID == vid].sort_values("FrameNo")
            boundaries = [(int(r.FrameNo), int(r.Phase) - 1) for _, r in sub.iterrows()]
            phase_src = expand_phase_boundaries(boundaries, n_src)
            start_f = boundaries[0][0] if boundaries else 0
            span = max(1, n_src - start_f)
            frame_idx = build_time_axis(span, source_fps, args.sample_fps) + start_f
            frame_idx = np.clip(frame_idx, 0, n_src - 1)
            vpath = root / "videos" / f"case_{vid}.mp4"
            feat = extract_video_sequential(
                vpath, frame_idx, model, device, size=(args.width, args.height), chunk=args.chunk
            )
            elapsed, rsd, duration = elapsed_and_rsd_minutes(len(frame_idx), args.sample_fps)
            payload = {
                "video_id": vid,
                "features": feat,
                "elapsed": torch.from_numpy(elapsed.astype(np.float32)),
                "rsd": torch.from_numpy(rsd.astype(np.float32)),
                "progress": torch.from_numpy((elapsed / max(duration, 1e-6)).astype(np.float32)),
                "phase": torch.from_numpy(phase_src[frame_idx].astype(np.int64)),
                "duration_min": float(duration),
                "surgeon_exp": int(row.Experience),
            }
            torch.save(payload, path)


if __name__ == "__main__":
    main()
