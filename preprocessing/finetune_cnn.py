#!/usr/bin/env python
"""Stage-1 CNN fine-tuning for TimeLSTM / RSDNet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.datasets.base_rsd import build_time_axis, elapsed_and_rsd_minutes
from data_provider.datasets.cataract101 import Cataract101RSD
from data_provider.datasets.cholec80 import PHASE_MAP, Cholec80RSD
from utils.config import default_cache_root, default_data_root
from utils.seed import set_seed
from utils.splits import assert_disjoint_split, build_protocol_splits, get_protocol_repeat


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DATASET_DIRS = {"cholec80": "cholec80", "cataract101": "cataract-101"}
DATASET_N_PHASES = {"cholec80": 7, "cataract101": 10}


def load_phase_cholec(path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    phases = []
    for p in df.iloc[:, 1].astype(str).tolist():
        phases.append(PHASE_MAP.get(p, PHASE_MAP.get(p.replace(" ", ""), 0)))
    return np.asarray(phases, dtype=np.int64)


def build_video_plans(dataset: str, root: Path, video_ids, sample_fps=1.0, source_fps=25.0) -> dict:
    """vid → (video_path, sampled source-frame indices, phase per sample, progress per sample).

    Cataract-101 reuses Cataract101RSD so sampling matches run.py (surgical span only).
    """
    root = Path(root)
    plans = {}
    if dataset == "cataract101":
        ds = Cataract101RSD(root, [int(v) for v in video_ids], sample_fps=sample_fps, source_fps=source_fps)
        for m in ds.iter_meta():
            vid = m["video_id"]
            plans[vid] = (root / "videos" / f"case_{vid}.mp4", m["frame_indices"], m["phase"], m["progress"])
        return plans
    for vid in video_ids:
        phase = load_phase_cholec(root / "phase_annotations" / f"{vid}-phase.txt")
        idx = build_time_axis(len(phase), source_fps, sample_fps)
        elapsed, _, duration = elapsed_and_rsd_minutes(len(idx), sample_fps)
        plans[vid] = (root / "videos" / f"{vid}.mp4", idx, phase[idx], elapsed / max(duration, 1e-6))
    return plans


def _cache_key(dataset: str, vid) -> str:
    """Folder name under extract_frames cache: video01 / case_12."""
    if dataset == "cataract101":
        return f"case_{int(vid)}"
    return str(vid)


def _cache_ready(cache_dir: Path | None, dataset: str, vid, n: int) -> Path | None:
    """Return per-video JPEG cache dir if sample files 000000..(n-1) exist."""
    if cache_dir is None or n <= 0:
        return None
    d = Path(cache_dir) / _cache_key(dataset, vid)
    if (d / f"{n - 1:06d}.jpg").exists() and (d / "000000.jpg").exists():
        return d
    return None


def _load_cached_rgb(cache_vid: Path, k: int, size: int) -> np.ndarray:
    path = cache_vid / f"{k:06d}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size))
    return rgb


class FrameIndex(Dataset):
    """Index of sampled frames with method-specific labels (phase or progress)."""

    def __init__(
        self,
        root: Path,
        video_ids,
        sample_fps=1.0,
        source_fps=25.0,
        task="phase",
        size=224,
        dataset="cholec80",
    ):
        self.root = Path(root)
        self.size = int(size)
        self.task = task
        self.items = []  # (video_path, src_frame_idx, label)
        plans = build_video_plans(dataset, self.root, video_ids, sample_fps, source_fps)
        for vid, (vpath, frame_idx, phase_s, progress) in plans.items():
            for i, fidx in enumerate(frame_idx.tolist()):
                y = float(phase_s[i]) if task == "phase" else float(progress[i])
                self.items.append((str(vpath), int(fidx), y, vid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Random-access decode (OK for CNN FT with moderate batch); prefer sequential loader below for speed.
        vpath, fidx, y, vid = self.items[index]
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size))
        x = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = (x - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)
        return x, torch.tensor(y), vid


class SequentialVideoEpoch:
    """Iterate train videos and yield frame batches.

    Prefers offline JPEG frame cache (extract_frames.py layout); falls back to
    sequential mp4 decode (still faster than random seek).
    """

    def __init__(
        self,
        root,
        video_ids,
        task,
        sample_fps=1.0,
        source_fps=25.0,
        size=224,
        batch_size=32,
        dataset="cholec80",
        frame_cache_dir=None,
    ):
        self.root = Path(root)
        self.video_ids = list(video_ids)
        self.task = task
        self.size = size
        self.batch_size = batch_size
        self.dataset = dataset
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.plans = build_video_plans(dataset, self.root, self.video_ids, sample_fps, source_fps)

    def __iter__(self):
        for vid in self.video_ids:
            vpath, want, phase_s, progress = self.plans[vid]
            buf_x, buf_y = [], []
            cache_vid = _cache_ready(self.frame_cache_dir, self.dataset, vid, len(want))
            if cache_vid is not None:
                for k in range(len(want)):
                    rgb = _load_cached_rgb(cache_vid, k, self.size)
                    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
                    x = (x - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)
                    if self.task == "phase":
                        y = torch.tensor(int(phase_s[k]), dtype=torch.long)
                    else:
                        y = torch.tensor(float(progress[k]), dtype=torch.float32)
                    buf_x.append(x)
                    buf_y.append(y)
                    if len(buf_x) >= self.batch_size:
                        yield torch.stack(buf_x), torch.stack(buf_y)
                        buf_x, buf_y = [], []
            else:
                want_set = {int(i): k for k, i in enumerate(want.tolist())}
                cap = cv2.VideoCapture(str(vpath))
                fidx = 0
                max_need = int(want.max()) if len(want) else -1
                while fidx <= max_need:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if fidx in want_set:
                        k = want_set[fidx]
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (self.size, self.size))
                        x = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
                        x = (x - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)
                        if self.task == "phase":
                            y = torch.tensor(int(phase_s[k]), dtype=torch.long)
                        else:
                            y = torch.tensor(float(progress[k]), dtype=torch.float32)
                        buf_x.append(x)
                        buf_y.append(y)
                        if len(buf_x) >= self.batch_size:
                            yield torch.stack(buf_x), torch.stack(buf_y)
                            buf_x, buf_y = [], []
                    fidx += 1
                cap.release()
            if buf_x:
                yield torch.stack(buf_x), torch.stack(buf_y)


def _to_cpu(obj):
    """Recursively copy tensors to CPU (GB10: avoid device→host copies inside torch.save)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").contiguous()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def build_resnet152(task: str, n_phases: int) -> nn.Module:
    m = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    if task == "phase":
        m.fc = nn.Linear(2048, n_phases)
    else:
        m.fc = nn.Linear(2048, 1)
    return m


@torch.no_grad()
def eval_phase(model, root, ids, device, batch_size=48, dataset="cholec80", frame_cache_dir=None):
    model.eval()
    correct, total = 0, 0
    for x, y in SequentialVideoEpoch(
        root, ids, task="phase", batch_size=batch_size, dataset=dataset, frame_cache_dir=frame_cache_dir
    ):
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_progress(model, root, ids, device, batch_size=48, dataset="cholec80", frame_cache_dir=None):
    model.eval()
    err, total = 0.0, 0
    for x, y in SequentialVideoEpoch(
        root, ids, task="progress", batch_size=batch_size, dataset=dataset, frame_cache_dir=frame_cache_dir
    ):
        x = x.to(device)
        y = y.to(device)
        pred = torch.sigmoid(model(x).reshape(-1))
        err += float((pred - y).abs().sum().item())
        total += y.numel()
    return err / max(total, 1)


def _resolve_split_ids(args) -> tuple[list, list, list, dict]:
    """Return train_ids, val_ids, test_ids, meta. Test is never used for FT."""
    if args.protocol:
        proto_path = Path(args.protocol)
        if not proto_path.is_absolute():
            proto_path = ROOT / proto_path
        with proto_path.open("r", encoding="utf-8") as f:
            protocol = yaml.safe_load(f)
        pack = build_protocol_splits(protocol, ROOT)
        rep = get_protocol_repeat(pack, args.repeat)
        train_ids = list(rep["train"])
        val_ids = list(rep.get("val") or [])
        test_ids = list(rep["test"])
        if not val_ids:
            # Carve a small val from train only (never test).
            n_val = max(1, len(train_ids) // 5)
            val_ids = train_ids[-n_val:]
            train_ids = train_ids[:-n_val]
        meta = {
            "protocol": str(proto_path),
            "dataset": protocol.get("dataset"),
            "scheme": pack.get("scheme"),
            "repeat": args.repeat,
            "seed": rep.get("seed"),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        }
        return train_ids, val_ids, test_ids, meta

    # Explicit lists (legacy). Empty defaults kept only with --allow_legacy_40_40.
    if args.train_ids:
        train_ids = [x for x in args.train_ids.split(",") if x]
        val_ids = [x for x in args.val_ids.split(",") if x] if args.val_ids else []
        test_ids = [x for x in args.test_ids.split(",") if x] if args.test_ids else []
        meta = {"protocol": None, "legacy_explicit": True}
        return train_ids, val_ids, test_ids, meta

    if not args.allow_legacy_40_40:
        raise SystemExit(
            "Refuse default video01-40 FT: leaks relative to cholec80_main test folds.\n"
            "Use: --protocol configs/protocol/cholec80_main.yaml --repeat {0..4}\n"
            "Or pass --train_ids/--val_ids that exclude that repeat's test set.\n"
            "Emergency only: --allow_legacy_40_40"
        )
    train_ids = [f"video{i:02d}" for i in range(1, 41)]
    val_ids = [f"video{i:02d}" for i in range(41, 49)]
    test_ids = [f"video{i:02d}" for i in range(49, 81)]
    meta = {"protocol": None, "legacy_40_40": True, "warning": "not comparable to cholec80_main"}
    return train_ids, val_ids, test_ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["TimeLSTM", "RSDNet"], required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS), default=None, help="default: protocol dataset")
    ap.add_argument(
        "--data_root",
        default=default_data_root(),
        help="dataset root (env RSDBENCH_DATA_ROOT, else <repo_parent>/DATASET)",
    )
    ap.add_argument("--out_dir", default=None)
    ap.add_argument(
        "--protocol",
        default="configs/protocol/cholec80_main.yaml",
        help="RSDBench protocol yaml (same splits as run.py). Empty string to disable.",
    )
    ap.add_argument("--repeat", type=int, default=0, help="protocol repeat index 0..n_repeats-1")
    ap.add_argument("--train_ids", type=str, default="", help="comma-separated; overrides protocol if set")
    ap.add_argument("--val_ids", type=str, default="", help="comma-separated; must not include test")
    ap.add_argument("--test_ids", type=str, default="", help="optional; only for leakage assert, never trained on")
    ap.add_argument(
        "--allow_legacy_40_40",
        action="store_true",
        help="Allow old video01-40 FT (NOT fair vs cholec80_main). Default off.",
    )
    ap.add_argument("--epochs", type=int, default=20, help="max epochs (early-stop may finish earlier)")
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early-stop patience on val (0 disables; default 10)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=None, help="default: protocol repeat seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--frame_cache",
        type=str,
        default="",
        help="JPEG frame cache dir (empty = auto .cache/frames/{dataset}/fps1, or mp4 if missing)",
    )
    args = ap.parse_args()
    if args.protocol == "":
        args.protocol = None
    if args.train_ids:
        args.protocol = None  # explicit lists win

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for CNN fine-tuning")

    train_ids, val_ids, test_ids, split_meta = _resolve_split_ids(args)
    assert_disjoint_split(train_ids, test_ids or [], val_ids)
    proto_dataset = split_meta.get("dataset")
    if args.dataset and proto_dataset and args.dataset != proto_dataset:
        raise SystemExit(f"--dataset {args.dataset} != protocol dataset {proto_dataset}")
    dataset = args.dataset or proto_dataset or "cholec80"
    if dataset not in DATASET_DIRS:
        raise SystemExit(f"Unknown dataset={dataset}; expected one of {sorted(DATASET_DIRS)}")
    root = Path(args.data_root) / DATASET_DIRS[dataset]
    seed = int(args.seed if args.seed is not None else split_meta.get("seed") or 42)
    set_seed(seed)
    device = torch.device(args.device)

    task = "phase" if args.method == "TimeLSTM" else "progress"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = f"seed{seed}" if split_meta.get("seed") is not None else "legacy"
        out_dir = ROOT / ".cache" / "backbones" / args.method / dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_phases = DATASET_N_PHASES[dataset]
    model = build_resnet152(task, n_phases).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.epochs // 3, 1), gamma=0.1)

    if args.frame_cache.strip():
        frame_cache_dir = Path(args.frame_cache.strip())
    else:
        frame_cache_dir = Path(default_cache_root()) / "frames" / dataset / "fps1"
    if not frame_cache_dir.is_dir():
        print(f"[warn] frame_cache missing ({frame_cache_dir}); falling back to mp4", flush=True)
        frame_cache_dir = None

    best = float("inf") if task == "progress" else 0.0
    bad = 0
    patience = int(args.patience)
    history = []
    start_ep = 0
    # Resume: last.pt is written after every epoch; history.json only once training is finished.
    last_p = out_dir / "last.pt"
    if last_p.exists() and not (out_dir / "history.json").exists():
        st = torch.load(last_p, map_location="cpu")
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        start_ep, best, bad, history = int(st["epoch"]), float(st["best"]), int(st["bad"]), list(st["history"])
        print(f"[resume] {last_p} epoch={start_ep} best={best:.4f} bad={bad}", flush=True)
    print(
        f"[finetune] method={args.method} dataset={dataset} n_phases={n_phases} "
        f"task={task} train={len(train_ids)} val={len(val_ids)} test_heldout={len(test_ids)}"
    )
    print(f"[finetune] split_meta={split_meta}")
    print(f"[finetune] ImageNet ResNet-152 → {out_dir}")
    print(
        f"[finetune] max_epochs={args.epochs} patience={patience} "
        f"frame_cache={'on:' + str(frame_cache_dir) if frame_cache_dir else 'off(mp4)'}",
        flush=True,
    )

    for ep in range(start_ep, args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        # shuffle video order each epoch
        order = list(train_ids)
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(order)
        for x, y in tqdm(
            SequentialVideoEpoch(
                root,
                order,
                task=task,
                batch_size=args.batch_size,
                dataset=dataset,
                frame_cache_dir=frame_cache_dir,
            ),
            desc=f"epoch {ep+1}/{args.epochs}",
        ):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if task == "phase":
                loss = F.cross_entropy(logits, y.long())
            else:
                pred = torch.sigmoid(logits.reshape(-1))
                loss = F.smooth_l1_loss(pred, y.float())
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
        sched.step()
        train_loss = total_loss / max(n, 1)

        if task == "phase":
            metric = eval_phase(
                model,
                root,
                val_ids,
                device,
                batch_size=args.batch_size,
                dataset=dataset,
                frame_cache_dir=frame_cache_dir,
            )
            print(f"[ep {ep+1}] loss={train_loss:.4f} val_acc={metric:.4f} best={best:.4f} bad={bad}")
            improved = metric > best + 1e-4
            if improved:
                best = metric
        else:
            metric = eval_progress(
                model,
                root,
                val_ids,
                device,
                batch_size=args.batch_size,
                dataset=dataset,
                frame_cache_dir=frame_cache_dir,
            )
            print(f"[ep {ep+1}] loss={train_loss:.4f} val_mae_progress={metric:.4f} best={best:.4f} bad={bad}")
            improved = metric < best - 1e-4
            if improved:
                best = metric

        history.append({"epoch": ep + 1, "train_loss": train_loss, "val_metric": float(metric)})
        if improved:
            bad = 0
            # Save CPU copy only. Under multi-job GPU contention (GB10 UVM),
            # torch.save(model.state_dict()) can hit Xid 31 / CUDA launch failure
            # on cuMemcpyDtoHAsync — training itself had already finished the epoch.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            cpu_sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "method": args.method,
                    "dataset": dataset,
                    "task": task,
                    "n_phases": n_phases,
                    "state_dict": cpu_sd,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids_heldout": test_ids,
                    "split_meta": split_meta,
                    "val_metric": float(metric),
                    "epoch": ep + 1,
                    "seed": seed,
                    "patience": patience,
                    "max_epochs": args.epochs,
                },
                out_dir / "best_cnn.pth",
            )
            del cpu_sd
            print(f"  saved best → {out_dir / 'best_cnn.pth'}")
        else:
            bad += 1
        torch.save(
            {
                "state_dict": _to_cpu(model.state_dict()),
                "optim": _to_cpu(opt.state_dict()),
                "sched": sched.state_dict(),
                "epoch": ep + 1,
                "best": best,
                "bad": bad,
                "history": history,
            },
            last_p,
        )
        if not improved and patience > 0 and bad >= patience:
            print(
                f"[early-stop] epoch={ep+1} best_val={best:.4f} patience={patience}",
                flush=True,
            )
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(
        json.dumps(
            {
                **split_meta,
                "dataset": dataset,
                "n_phases": n_phases,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids_heldout": test_ids,
                "seed": seed,
                "patience": patience,
                "max_epochs": args.epochs,
                "stopped_epoch": history[-1]["epoch"] if history else 0,
                "frame_cache": str(frame_cache_dir) if frame_cache_dir else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] best_metric={best:.4f}")


if __name__ == "__main__":
    main()
