#!/usr/bin/env python
"""Recompute MAE / relative error / SMAPE from saved predictions.csv (no retrain).

Example:
  python scripts/recompute_metrics_from_predictions.py \\
    --pred_glob 'results/cholec80_PhaseInferred/seed*/predictions.csv'
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import save_json
from utils.metrics import format_mean_std, mean_std, metrics_from_prediction_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_glob", required=True, help="glob of predictions.csv (relative to RSDBench or abs)")
    ap.add_argument("--windows", default="2,5,10,30")
    ap.add_argument("--out", default="", help="aggregated summary json")
    args = ap.parse_args()
    windows = [float(x) for x in args.windows.split(",") if x.strip()]

    pattern = args.pred_glob
    if not pattern.startswith("/"):
        pattern = str(ROOT / pattern)
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit(f"No files match {args.pred_glob}")

    per_run = {}
    for path in paths:
        df = pd.read_csv(path)
        m = metrics_from_prediction_rows(df.to_dict(orient="records"), windows=windows)
        per_run[str(path.parent.name)] = m
        save_json(m, path.parent / "metrics_recomputed.json")
        print(
            f"{path.parent.name}: mae_all={m['mae_all']:.4f} "
            f"rel_err_all={m['rel_err_all']:.4f} smape_all={m['smape_all']:.4f}"
        )

    keys = sorted({k for m in per_run.values() for k in m})
    summary = {}
    for k in keys:
        mu, sd = mean_std([per_run[r][k] for r in per_run if k in per_run[r]])
        summary[k] = {"mean": mu, "std": sd, "display": format_mean_std(mu, sd)}
    print("--- summary ---")
    for k in ("mae_all", "rel_err_all", "smape_all", "mae_5", "rel_err_5", "smape_5", "mae_30", "smape_30"):
        if k in summary:
            print(f"{k}: {summary[k]['display']}")
    out = Path(args.out) if args.out else paths[0].parents[1] / "summary_recomputed.json"
    if not out.is_absolute():
        out = ROOT / out
    save_json({"per_run": per_run, "summary": summary}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
