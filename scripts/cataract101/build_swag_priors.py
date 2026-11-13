#!/usr/bin/env python
"""Build SWAG class-frequency priors for Cataract-101."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

from utils.config import default_data_root

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.datasets.cataract101 import Cataract101RSD
from utils.splits import build_protocol_splits, get_protocol_repeat

N_PHASES = 10


def class_freq_positions(phase_seqs, eos_class: int, lookahead: int = 18, anticip_time: int = 60):
    lookahead_1fps = lookahead * 60
    n_tokens = lookahead_1fps // anticip_time
    freq = defaultdict(lambda: [defaultdict(int) for _ in range(n_tokens)])
    for seq in phase_seqs:
        data = np.concatenate([np.asarray(seq, dtype=np.int64), np.full(lookahead_1fps, eos_class)])
        for i in range(len(data) - lookahead_1fps):
            cur = int(data[i])
            for j in range(n_tokens):
                freq[cur][j][int(data[i + j * anticip_time])] += 1
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/cataract101_main.yaml")
    ap.add_argument("--data_root", default=default_data_root())
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--lookahead", type=int, default=18)
    ap.add_argument("--out_root", default=str(ROOT / "data" / "swag_priors" / "cataract101"))
    args = ap.parse_args()

    protocol = yaml.safe_load((ROOT / args.protocol).read_text())
    root = Path(args.data_root) / "cataract-101"
    for seed in [int(s) for s in args.seeds.split(",")]:
        proto = copy.deepcopy(protocol)
        proto["split"]["n_repeats"] = 1
        proto["split"]["seed_start"] = seed
        rep = get_protocol_repeat(build_protocol_splits(proto, ROOT), 0)
        train_ids = [int(v) for v in rep["train"]]
        ds = Cataract101RSD(root, train_ids, sample_fps=1.0)
        seqs = [m["phase"] for m in ds.iter_meta()]
        freq = class_freq_positions(seqs, eos_class=N_PHASES, lookahead=args.lookahead)
        out = {
            str(c): [{str(k): float(v) for k, v in sorted(d.items())} for d in freq[c]]
            for c in sorted(freq)
            if c < N_PHASES
        }
        out_dir = Path(args.out_root) / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"naive2_cataract101_class_probs_at{args.lookahead}.json"
        path.write_text(json.dumps(out), encoding="utf-8")
        (out_dir / "train_ids.json").write_text(json.dumps(train_ids), encoding="utf-8")
        print(f"[prior] seed{seed} train={len(train_ids)} classes={sorted(out)} → {path}")


if __name__ == "__main__":
    main()
