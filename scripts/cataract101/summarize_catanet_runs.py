#!/usr/bin/env python3
"""Aggregate CataNet run42..46 metrics into summary.json (mean±std)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="seed*/metrics.json")
    args = ap.parse_args()

    runs = sorted(args.root.glob(args.pattern))
    if not runs:
        # backward compat
        runs = sorted(args.root.glob("run*/metrics.json"))
    if not runs:
        raise SystemExit(f"no metrics under {args.root} pattern={args.pattern}")

    keys = None
    series: dict[str, list[float]] = {}
    for p in runs:
        m = json.loads(p.read_text())
        flat = {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and k != "seed"}
        if keys is None:
            keys = sorted(flat)
        for k in keys:
            series.setdefault(k, []).append(float(flat[k]))

    summary = {
        "n_runs": len(runs),
        "runs": [p.parent.name for p in runs],
        "summary": {
            k: {"mean": float(np.mean(vs)), "std": float(np.std(vs, ddof=0)), "values": vs}
            for k, vs in series.items()
        },
    }
    out = args.root / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out} n_runs={len(runs)}")
    for k in ("mae_all", "mae_2", "mae_5"):
        if k in summary["summary"]:
            s = summary["summary"][k]
            print(f"  {k}: {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
