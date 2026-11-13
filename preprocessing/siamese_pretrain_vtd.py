#!/usr/bin/env python
"""Siamese temporal-order pretraining for VTD-Net (Bodenstedt et al., IJCARS 2019)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.config import default_data_root

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import AlexNet_Weights, alexnet
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.finetune_cnn import DATASET_DIRS, build_video_plans
from utils.seed import set_seed
from utils.splits import assert_disjoint_split, build_protocol_splits, get_protocol_repeat

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TemporalOrderPairs(Dataset):
    """Sample (frame_a, frame_b, label) with label=1 if a is earlier than b."""

    def __init__(self, root, video_ids, sample_fps=1.0, source_fps=25.0, size=224, pairs_per_video=200, seed=0, dataset="cholec80"):
        self.root = Path(root)
        self.size = int(size)
        self.items = []  # (video_path, fa, fb, label)
        rng = np.random.default_rng(seed)
        plans = build_video_plans(dataset, self.root, video_ids, sample_fps, source_fps)
        for vid in video_ids:
            vpath, idx, _, _ = plans[vid]
            if len(idx) < 4:
                continue
            for _ in range(pairs_per_video):
                i, j = rng.choice(len(idx), size=2, replace=False)
                if i == j:
                    continue
                if i < j:
                    self.items.append((str(vpath), int(idx[i]), int(idx[j]), 1))
                else:
                    self.items.append((str(vpath), int(idx[i]), int(idx[j]), 0))

    def __len__(self):
        return len(self.items)

    def _read(self, vpath, fidx):
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
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, i):
        vpath, fa, fb, y = self.items[i]
        return self._read(vpath, fa), self._read(vpath, fb), torch.tensor(y, dtype=torch.float32)


class SiameseAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Must download ImageNet AlexNet — no random trunk.
        net = alexnet(weights=AlexNet_Weights.DEFAULT)
        net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])  # 4096
        self.trunk = net
        self.head = nn.Sequential(
            nn.Linear(4096 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def encode(self, x):
        return self.trunk(x)

    def forward(self, xa, xb):
        fa = self.encode(xa)
        fb = self.encode(xb)
        logits = self.head(torch.cat([fa, fb], dim=-1)).squeeze(-1)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/cholec80_main.yaml")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--data_root", default=default_data_root())
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pairs_per_video", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for VTD Siamese pretrain")

    proto_path = Path(args.protocol)
    if not proto_path.is_absolute():
        proto_path = ROOT / proto_path
    with proto_path.open() as f:
        protocol = yaml.safe_load(f)
    pack = build_protocol_splits(protocol, ROOT)
    rep = get_protocol_repeat(pack, args.repeat)
    train_ids = list(rep["train"])
    val_ids = list(rep.get("val") or [])
    test_ids = list(rep["test"])
    if not val_ids:
        n_val = max(1, len(train_ids) // 5)
        val_ids = train_ids[-n_val:]
        train_ids = train_ids[:-n_val]
    assert_disjoint_split(train_ids, test_ids, val_ids)
    seed = int(rep.get("seed") or 42)
    set_seed(seed)

    dataset = protocol.get("dataset", "cholec80")
    root = Path(args.data_root) / DATASET_DIRS[dataset]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / ".cache" / "backbones" / "VTDNet" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TemporalOrderPairs(root, train_ids, pairs_per_video=args.pairs_per_video, seed=seed, dataset=dataset)
    val_ds = TemporalOrderPairs(
        root, val_ids, pairs_per_video=max(32, args.pairs_per_video // 4), seed=seed + 1, dataset=dataset
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(args.device)
    model = SiameseAlexNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    history = []
    print(f"[siamese] train={len(train_ids)} val={len(val_ids)} test_heldout={len(test_ids)} → {out_dir}")
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for xa, xb, y in tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}"):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xa, xb)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        # val acc
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for xa, xb, y in val_loader:
                xa, xb, y = xa.to(device), xb.to(device), y.to(device)
                pred = (torch.sigmoid(model(xa, xb)) > 0.5).float()
                correct += int((pred == y).sum().item())
                tot += y.numel()
        acc = correct / max(tot, 1)
        history.append({"epoch": ep + 1, "train_loss": total / max(n, 1), "val_acc": acc})
        print(f"[ep {ep+1}] loss={total/max(n,1):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            cpu_sd = {k: v.detach().cpu() for k, v in model.trunk.state_dict().items()}
            torch.save(
                {
                    "task": "siamese_temporal_order",
                    "trunk_state_dict": cpu_sd,
                    "val_acc": acc,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids_heldout": test_ids,
                    "seed": seed,
                    "epoch": ep + 1,
                },
                out_dir / "siamese_alexnet.pth",
            )
            print(f"  saved → {out_dir / 'siamese_alexnet.pth'}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best_val_acc={best:.4f}")


if __name__ == "__main__":
    main()
