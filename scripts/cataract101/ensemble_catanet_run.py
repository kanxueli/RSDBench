#!/usr/bin/env python3
"""Ensemble 6 fold CataNet predictions on fixed test; write metrics for one run_seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.metrics import compute_dataset_metrics  # type: ignore


def load_fold_pred(path: Path) -> pd.DataFrame:
    # elapsed,progress,predicted_rsd,predicted_step,predicted_exp,...
    df = pd.read_csv(path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--windows", type=float, nargs="+", default=[2.0, 5.0])
    args = ap.parse_args()

    # buckets[(vid, t_idx)] -> list of rsd_pred; also keep true from DatasetNoLabel via elapsed
    # Inference CSV has elapsed + predicted_rsd but NOT rsd_true.
    # Recompute rsd_true from elapsed: for each video, duration = max(elapsed)+rsd_at_end approx
    # Better: from pool labels. For now derive: at each row, if we have elapsed and know duration.
    # DatasetNoLabel returns rsd = duration - elapsed; inference saves elapsed and rsd_pred only.
    # Reconstruct duration per video as max(elapsed) + last predicted? NO - use pool csv rsd_frames.

    pool = Path("./.cache/catanet_cataract101/pool")
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    meta: dict[tuple[str, int], tuple[float, float]] = {}

    for fold in range(1, 7):
        pred_dir = args.run_dir / f"fold{fold}" / "test_pred"
        if not pred_dir.exists():
            raise FileNotFoundError(pred_dir)
        for csv_path in sorted(pred_dir.glob("*.csv")):
            vid = csv_path.stem
            df = load_fold_pred(csv_path)
            # true RSD from prepared labels
            lab = pd.read_csv(pool / vid / f"{vid}.csv")
            # after index col: step,u1,experience,u2,u3,rsd_frames
            rsd_true_min = lab["rsd_frames"].to_numpy(dtype=np.float64) / 60.0 / 25.0
            n = min(len(df), len(rsd_true_min))
            for j in range(n):
                key = (vid, j)
                buckets[key].append(float(df["predicted_rsd"].iloc[j]))
                meta[key] = (float(df["elapsed"].iloc[j]), float(rsd_true_min[j]))

    rows = []
    for (vid, j), preds in sorted(buckets.items()):
        elapsed, rsd_true = meta[(vid, j)]
        rows.append(
            {
                "video_id": vid,
                "t_idx": j,
                "elapsed_min": elapsed,
                "rsd_true": rsd_true,
                "rsd_pred": float(np.mean(preds)),
            }
        )
    out_pred = args.run_dir / "predictions_ensemble.csv"
    pd.DataFrame(rows).to_csv(out_pred, index=False)

    # video-macro metrics
    per_video = {}
    for r in rows:
        per_video.setdefault(r["video_id"], {"pred": [], "target": []})
        per_video[r["video_id"]]["pred"].append(r["rsd_pred"])
        per_video[r["video_id"]]["target"].append(r["rsd_true"])
    for vid in per_video:
        per_video[vid]["pred"] = np.asarray(per_video[vid]["pred"], dtype=np.float64)
        per_video[vid]["target"] = np.asarray(per_video[vid]["target"], dtype=np.float64)

    metrics = compute_dataset_metrics(per_video, windows=args.windows)
    metrics["seed"] = int(args.seed)
    metrics["n_folds"] = 6
    with open(args.run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in list(metrics)[:8]}, indent=2))


if __name__ == "__main__":
    main()
