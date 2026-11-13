#!/usr/bin/env python3
"""Convert CataNet inference CSVs + pool labels → metrics/predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.metrics import compute_dataset_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=float, nargs="+", default=[2.0, 5.0])
    args = ap.parse_args()

    rows = []
    per_video = {}
    for csv_path in sorted(args.pred_dir.glob("*.csv")):
        vid = csv_path.stem
        df = pd.read_csv(csv_path)
        lab = pd.read_csv(args.pool / vid / f"{vid}.csv")
        rsd_true = lab["rsd_frames"].to_numpy(dtype=np.float64) / 60.0 / 25.0
        n = min(len(df), len(rsd_true))
        pred = df["predicted_rsd"].to_numpy(dtype=np.float64)[:n]
        tgt = rsd_true[:n]
        elapsed = df["elapsed"].to_numpy(dtype=np.float64)[:n]
        per_video[vid] = {"pred": pred, "target": tgt}
        for j in range(n):
            rows.append(
                {
                    "video_id": vid,
                    "t_idx": j,
                    "elapsed_min": float(elapsed[j]),
                    "rsd_true": float(tgt[j]),
                    "rsd_pred": float(pred[j]),
                }
            )

    metrics = compute_dataset_metrics(per_video, windows=args.windows)
    metrics["seed"] = int(args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(json.dumps({k: metrics[k] for k in ("mae_all", "mae_2", "mae_5")}, indent=2))


if __name__ == "__main__":
    main()
