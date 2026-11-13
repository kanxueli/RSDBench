#!/usr/bin/env python
"""SWAG stage-1 ViT feature extractor (Boels et al.; LoViT)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from utils.config import default_data_root

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import timm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.datasets.base_rsd import elapsed_and_rsd_minutes
from preprocessing.finetune_cnn import DATASET_DIRS, build_video_plans
from preprocessing.prepare_frames_1fps import DEFAULT_OUT as FRAME_CACHE
from utils.seed import set_seed
from utils.splits import assert_disjoint_split, build_protocol_splits, get_protocol_repeat

ALPHA = 30
JX_VIT = ROOT / ".cache" / "pretrained" / "jx_vit_base_p16_224-80ecf9dd.pth"


def phase_starts(phase: np.ndarray) -> np.ndarray:
    """Index where the contiguous segment containing each frame begins."""
    starts = np.zeros(len(phase), dtype=np.int64)
    for i in range(1, len(phase)):
        starts[i] = starts[i - 1] if phase[i] == phase[i - 1] else i
    return starts


def clip_indices(t: int, b: int, alpha: int = ALPHA) -> np.ndarray:
    w = math.ceil((t - b) / alpha)
    return np.clip(t - w * np.arange(alpha - 1, -1, -1), 0, None)


def pixel_stats(videos: dict, ids) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std (0-1 scale) over all cached 1 fps frames of the given videos."""
    n, s1, s2 = 0, np.zeros(3), np.zeros(3)
    for vid in ids:
        f = np.load(videos[vid]["frames"], mmap_mode="r")
        for k in range(0, len(f), 64):
            x = np.asarray(f[k : k + 64], dtype=np.float64).reshape(-1, 3) / 255.0
            n += len(x)
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
    mean = s1 / n
    std = np.sqrt(s2 / n - mean**2)
    return (torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))


def spatial(frames: torch.Tensor, train: bool, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 (T,H,W,3) → normalised float (T,3,224,224); one transform per clip."""
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    short = random.randint(248, 280) if train else 248
    x = F.interpolate(x, size=(short, short), mode="bilinear", align_corners=False, antialias=True)
    if train:
        y0, x0 = random.randint(0, short - 224), random.randint(0, short - 224)
    else:
        y0 = x0 = (short - 224) // 2
    x = x[:, :, y0 : y0 + 224, x0 : x0 + 224]
    return (x - mean) / std


class LoViTClips(Dataset):
    def __init__(self, videos: dict, ids, target_frac: int, train: bool, rng: random.Random, norm):
        self.videos = videos
        self.train = train
        self.mean, self.std = norm
        self.items = []
        for vid in ids:
            n = len(videos[vid]["phase"])
            ts = list(range(n))
            if target_frac > 1:
                ts = sorted(rng.sample(ts, max(1, n // target_frac)))
            self.items += [(vid, t) for t in ts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, t = self.items[i]
        v = self.videos[vid]
        idx = clip_indices(t, int(v["starts"][t]))
        frames = torch.from_numpy(np.asarray(np.load(v["frames"], mmap_mode="r")[idx]))
        return spatial(frames, self.train, self.mean, self.std), int(v["phase"][t])


class LoViTExtractor(nn.Module):
    def __init__(self, n_classes: int, inter_dim=2048, n_layer=6, n_head=4, dropout=0.2):
        super().__init__()
        if not JX_VIT.is_file():
            raise FileNotFoundError(f"{JX_VIT} missing (timm release v0.1-vitjx)")
        self.vit = timm.create_model(
            "vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True,
            pretrained_cfg_overlay=dict(file=str(JX_VIT)), num_classes=0,
        )
        self.encoder = nn.Linear(768, inter_dim, bias=False)
        self.gpt = GPT2Model(
            GPT2Config(n_embd=inter_dim, n_layer=n_layer, n_head=n_head, vocab_size=768, use_cache=False,
                       bos_token_id=None, eos_token_id=None)
        )
        del self.gpt.wte
        self.decoder = nn.Linear(inter_dim, 768, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(768, n_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t = clips.shape[:2]
        feats = self.vit(clips.flatten(0, 1)).view(b, t, -1)
        h = self.gpt(inputs_embeds=self.encoder(feats)).last_hidden_state
        return self.cls(self.dropout(self.decoder(h[:, -1])))


def lr_at(epoch_float: float, base_lr: float, warmup: int, total: int) -> float:
    if epoch_float < warmup:
        return base_lr * epoch_float / warmup
    p = (epoch_float - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, p)))


def load_videos(dataset: str, data_root: Path, ids) -> dict:
    plans = build_video_plans(dataset, data_root / DATASET_DIRS[dataset], ids, sample_fps=1.0)
    out = {}
    for vid, (_, idx, phase, _) in plans.items():
        path = FRAME_CACHE / dataset / f"{vid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run preprocessing/prepare_frames_1fps.py --dataset {dataset}")
        phase = np.asarray(phase, dtype=np.int64)
        out[vid] = {"frames": str(path), "phase": phase, "starts": phase_starts(phase), "n": len(idx)}
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x.to(device, non_blocking=True)).argmax(-1)
        correct += int((pred.cpu() == y).sum())
        total += len(y)
    return correct / max(total, 1)


@torch.no_grad()
def extract(model, videos: dict, out_dir: Path, device, norm, chunk=128):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid, v in tqdm(videos.items(), desc="[lovit] extract"):
        path = out_dir / f"{vid}.pt"
        if path.exists():
            continue
        frames = torch.from_numpy(np.load(v["frames"]))
        feats = []
        for s in range(0, len(frames), chunk):
            x = spatial(frames[s : s + chunk], False, *norm).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(model.vit(x).float().cpu())
        elapsed, rsd, duration = elapsed_and_rsd_minutes(v["n"], 1.0)
        torch.save(
            {
                "video_id": vid,
                "features": torch.cat(feats),
                "elapsed": torch.from_numpy(elapsed.astype(np.float32)),
                "rsd": torch.from_numpy(rsd.astype(np.float32)),
                "progress": torch.from_numpy((elapsed / max(duration, 1e-6)).astype(np.float32)),
                "phase": torch.from_numpy(v["phase"]),
                "duration_min": float(duration),
            },
            path,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/cataract101_main.yaml")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--data_root", default=default_data_root())
    ap.add_argument("--n_classes", type=int, default=None, help="default: 10 Cataract-101 / 7 Cholec80")
    ap.add_argument("--target_frac", type=int, default=5, help="use 1/target_frac of target frames per epoch")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=4, help="gradient accumulation micro-batch (memory)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=15, help="early stopping on val accuracy (epochs)")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    protocol = yaml.safe_load((ROOT / args.protocol).read_text())
    dataset = protocol["dataset"]
    rep = get_protocol_repeat(build_protocol_splits(protocol, ROOT), args.repeat)
    train_ids, val_ids, test_ids = list(rep["train"]), list(rep.get("val") or []), list(rep["test"])
    assert_disjoint_split(train_ids, test_ids, val_ids)
    seed = int(rep["seed"])
    set_seed(seed)
    n_classes = args.n_classes or {"cataract101": 10, "cholec80": 7}[dataset]
    device = torch.device(args.device)
    out_dir = ROOT / ".cache" / "backbones" / "SWAG" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "vit_lovit.pth"
    feat_dir = ROOT / ".cache" / "features" / dataset / f"SWAG_vitLoViT_seed{seed}"
    root = Path(args.data_root)

    model = LoViTExtractor(n_classes).to(device)
    # Skip stage-1 when best ckpt exists (pause/resume → extract + stage 2 without retraining).
    if ckpt.exists():
        print(f"[lovit] resume extract-only: found {ckpt} (skip stage-1 train)", flush=True)
    else:
        videos = load_videos(dataset, root, train_ids + val_ids)
        norm = pixel_stats(videos, train_ids)
        print(f"[lovit] train-set pixel mean={norm[0].flatten().tolist()} std={norm[1].flatten().tolist()}", flush=True)
        rng = random.Random(seed)
        val_loader = DataLoader(
            LoViTClips(videos, val_ids, args.target_frac, train=False, rng=random.Random(seed + 1), norm=norm),
            batch_size=args.micro_batch * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
        accum = args.batch_size // args.micro_batch
        best, best_ep, history = -1.0, 0, []
        print(f"[lovit] {dataset} seed={seed} train={len(train_ids)} val={len(val_ids)} "
              f"test_heldout={len(test_ids)} target_frac=1/{args.target_frac} → {out_dir}", flush=True)
        for ep in range(args.epochs):
            model.train()
            ds = LoViTClips(videos, train_ids, args.target_frac, train=True, rng=rng, norm=norm)
            loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True, drop_last=True)
            steps = len(loader) // accum
            losses = []
            opt.zero_grad(set_to_none=True)
            for i, (x, y) in enumerate(tqdm(loader, desc=f"[lovit] ep{ep+1}/{args.epochs}", leave=False)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x.to(device, non_blocking=True)).float(), y.to(device))
                (loss / accum).backward()
                losses.append(float(loss.item()))
                if (i + 1) % accum == 0:
                    lr = lr_at(ep + (i + 1) / (steps * accum), args.lr, args.warmup, args.epochs)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            acc = evaluate(model, val_loader, device)
            history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_acc": acc, "lr": lr})
            mark = ""
            if acc > best:
                best, best_ep = acc, ep
                torch.save({"vit": model.vit.state_dict(), "norm_mean": norm[0], "norm_std": norm[1],
                            "init": JX_VIT.name, "epoch": ep + 1, "val_acc": acc, "seed": seed,
                            "train_ids": train_ids, "val_ids": val_ids, "test_ids_heldout": test_ids}, ckpt)
                mark = "  ** saved"
            print(f"[lovit] epoch={ep+1} lr={lr:.4g} train_loss={np.mean(losses):.4f} val_acc={acc:.4f}{mark}", flush=True)
            if ep - best_ep >= args.patience:
                print(f"[lovit] early stop: val_acc not improved for {args.patience} epochs (best epoch {best_ep+1})",
                      flush=True)
                break
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    state = torch.load(ckpt, map_location="cpu")
    model.vit.load_state_dict(state["vit"])
    norm = (state["norm_mean"], state["norm_std"])
    extract(model, load_videos(dataset, root, train_ids + val_ids + test_ids), feat_dir, device, norm)
    print(f"[lovit] features → {feat_dir}", flush=True)


if __name__ == "__main__":
    main()
