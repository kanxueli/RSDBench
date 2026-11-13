#!/usr/bin/env python
"""Build academic-style leaderboard: datasets as column groups, methods as rows.

Fill baselines now; deep methods appear as — until summary.json exists.
Metrics: mae_2, mae_5, mae_10, mae_30, mae_all, rel_err_all, smape_all
(Cataract board columns omit mae_10/mae_30; Cholec keeps them.)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import mean_std

METRICS = ["mae_2", "mae_5", "mae_10", "mae_30", "mae_all", "rel_err_all", "smape_all"]
METRIC_LABELS = {
    "mae_2": "MAE₂↓",
    "mae_5": "MAE₅↓",
    "mae_10": "MAE₁₀↓",
    "mae_30": "MAE₃₀↓",
    "mae_all": "MAE↓",
    "rel_err_all": "RelErr↓",
    "smape_all": "SMAPE↓",
}
LABEL_TO_METRIC = {v: k for k, v in METRIC_LABELS.items()}
DATASETS = [("cataract101", "Cataract-101"), ("cholec80", "Cholec80")]
# Cataract protocol windows are only [2, 5] — omit MAE₁₀/₃₀ columns on the board.
DATASET_METRICS = {
    "cataract101": ["mae_2", "mae_5", "mae_all", "rel_err_all", "smape_all"],
    "cholec80": ["mae_2", "mae_5", "mae_10", "mae_30", "mae_all", "rel_err_all", "smape_all"],
}
BASELINE_ORDER = [
    "NaiveMean",
    "NaiveMedian",
    "ConditionalMedian",
    "BayesianAverage",
    "BayesianConditional",
    "PhaseInferred",
    "SequenceMatch",
    "KostopoulosRF",
    "PadoyHMM",
]
# Board rows for deep methods (placeholders until finished)
DEEP_ORDER = [
    "CataNet",
    "BDNet",
    "TransLocal",
    "PitRSDNet",
    "SWAG",
    "TimeLSTM",
    "RSDNet",
    "VTDNet_V",
    "VTDNet_VT",
    "Rivoir2019",
    "Rivoir2020",
    "SUPRGAN",
]

# Map experiment directory suffixes → board method name
EXP_ALIASES = {
    "TransLocal_AWS": "TransLocal",
    "SWAG_SP": "SWAG",
    "VTDNet_V": "VTDNet_V",
    "VTDNet_VT": "VTDNet_VT",
    # legacy experiment dir aliases
    "TimeLSTM_paperloss": "TimeLSTM",
    "RSDNet_paperloss": "RSDNet",
}
# Experiment dirs that are not board entries:
# *Feat runs on shared ImageNet ResNet-50 features
EXCLUDED_EXP = re.compile(r"^(TimeLSTM|RSDNet)(_protocol|Feat)$")


FULL_SEEDS = 5  # protocol n_repeats (seeds 42..46)
SEED_N: dict = {}  # (method, dataset) -> number of finished seeds


def fmt(mean, std, digits=2) -> str:
    if mean is None or not np.isfinite(mean):
        return "—"
    if std is None or not np.isfinite(std) or abs(std) < 1e-12:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def normalize_method(rest: str) -> str:
    if rest in EXP_ALIASES:
        return EXP_ALIASES[rest]
    if rest in BASELINE_ORDER or rest in DEEP_ORDER:
        return rest
    if rest.startswith("TimeLSTM"):
        return "TimeLSTM"
    if rest.startswith("RSDNet"):
        return "RSDNet"
    if rest.startswith("VTDNet"):
        if rest.endswith("_VT") or "_VT" in rest:
            return "VTDNet_VT"
        return "VTDNet_V"
    if rest.startswith("TransLocal"):
        return "TransLocal"
    if rest.startswith("SWAG"):
        return "SWAG"
    if rest.startswith("CataNet"):
        return "CataNet"
    if rest.startswith("BDNet"):
        return "BDNet"
    if rest.startswith("PitRSDNet"):
        return "PitRSDNet"
    return rest


def parse_exp_name(name: str):
    name = re.sub(r"_seed\d+$", "", name)
    for ds, _ in DATASETS:
        if not name.startswith(f"{ds}_"):
            continue
        rest = name[len(ds) + 1 :]
        if rest.startswith("main_"):
            rest = rest[5:]
        if EXCLUDED_EXP.match(rest):
            return None
        method = normalize_method(rest)
        if method in BASELINE_ORDER:
            setting = "Baselines"
        else:
            setting = "Deep methods"
        return ds, method, setting
    return None


def load_cells(results_dir: Path):
    """Pool per-seed repeats across experiment dirs (e.g. *_seed42..46) per (method, dataset)."""
    cells = {}
    settings = {}
    pooled = {}  # (method, ds) → {seed_key: {metric: value}}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        parsed = parse_exp_name(exp_dir.name)
        if not parsed:
            continue
        ds, method, setting = parsed
        sj = exp_dir / "summary.json"
        if not sj.exists():
            continue
        obj = json.loads(sj.read_text())
        repeats = obj.get("repeats") or {}
        if not repeats:
            pack = obj.get("summary", {})
            summ = {m: (pack[m]["mean"], pack[m]["std"]) for m in METRICS if m in pack}
            if summ:
                cells.setdefault(method, {})[ds] = summ
                settings[method] = setting
            continue
        pooled.setdefault((method, ds), {}).update(repeats)
        settings[method] = setting
    for (method, ds), reps in pooled.items():
        summ = {}
        for m in METRICS:
            vals = [r[m] for r in reps.values() if m in r]
            if vals:
                summ[m] = mean_std(vals)
        if summ:
            cells.setdefault(method, {})[ds] = summ
            SEED_N[(method, ds)] = len(reps)
    for method in list(settings):
        if method in BASELINE_ORDER:
            settings[method] = "Baselines"
    # Ensure all board methods exist as rows
    for m in BASELINE_ORDER:
        cells.setdefault(m, {})
        settings[m] = "Baselines"
    for m in DEEP_ORDER:
        cells.setdefault(m, {})
        settings.setdefault(m, "Deep methods")
    return cells, settings


def order_methods(cells):
    ordered = []
    for m in BASELINE_ORDER + DEEP_ORDER:
        if m not in ordered:
            ordered.append(m)
    for m in sorted(cells):
        if m not in ordered:
            ordered.append(m)
    return ordered


def highlight_marks(cells, ordered):
    marks = {}
    for ds, _ in DATASETS:
        for metric in DATASET_METRICS[ds]:
            pairs = []
            for m in ordered:
                if SEED_N.get((m, ds), FULL_SEEDS) < FULL_SEEDS:
                    continue  # incomplete runs are not ranked
                if ds in cells.get(m, {}) and metric in cells[m][ds]:
                    pairs.append((m, cells[m][ds][metric][0]))
            finite = [(m, v) for m, v in pairs if v is not None and np.isfinite(v)]
            if not finite:
                continue
            finite.sort(key=lambda x: x[1])
            marks[(finite[0][0], ds, metric)] = "best"
            if len(finite) > 1:
                marks[(finite[1][0], ds, metric)] = "second"
    return marks


def cell_md(cells, marks, method, ds, metric):
    if ds not in cells.get(method, {}) or metric not in cells[method][ds]:
        return "—"
    s = fmt(*cells[method][ds][metric])
    n = SEED_N.get((method, ds), FULL_SEEDS)
    if n < FULL_SEEDS:
        s = f"{s} (n={n})"
    tag = marks.get((method, ds, metric))
    if tag == "best":
        return f"**{s}**"
    if tag == "second":
        return f"<u>{s}</u>"
    return s


def build_markdown(cells, settings, ordered) -> str:
    marks = highlight_marks(cells, ordered)
    n_cols = sum(len(DATASET_METRICS[ds]) for ds, _ in DATASETS)
    lines = [
        "# RSDBench leaderboard",
        "",
        "Layout: **datasets → columns**, **methods → rows**.",
        "Baselines filled; deep methods update automatically when `results/<exp>/summary.json` appears.",
        "Metrics: MAE (minutes) ↓, RelErr ↓ (|t|≥1 min), SMAPE ↓. "
        "Cataract protocol windows are [2, 5] only (no MAE₁₀/₃₀ columns); Cholec also reports MAE₁₀/₃₀.",
        "**Bold** = best in column; <u>underline</u> = second best (only rows with all 5 seeds are ranked).",
        f"`(n=k)` = only k of {FULL_SEEDS} seeds finished so far; shown for progress, not ranked.",
        "Metrics: MAE (minutes) ↓ on the test set; see README for protocols.",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Refresh: `python -u scripts/collect_results.py`",
        "",
    ]

    def join(xs):
        return "| " + " | ".join(xs) + " |"

    ds_row = ["Setting", "Method"]
    for ds, dlabel in DATASETS:
        n_m = len(DATASET_METRICS[ds])
        ds_row.append(f"**{dlabel}**")
        ds_row.extend([""] * (n_m - 1))
    met_row = ["", ""]
    for ds, _ in DATASETS:
        met_row.extend([METRIC_LABELS[m] for m in DATASET_METRICS[ds]])
    sep = "| " + " | ".join(["---"] * (2 + n_cols)) + " |"

    lines.append(join(ds_row))
    lines.append(sep)
    lines.append(join(met_row))

    groups = OrderedDict()
    for m in ordered:
        groups.setdefault(settings.get(m, "Other"), []).append(m)
    for gname, methods in groups.items():
        for i, m in enumerate(methods):
            r = [f"**{gname}**" if i == 0 else "", m]
            for ds, _ in DATASETS:
                for metric in DATASET_METRICS[ds]:
                    r.append(cell_md(cells, marks, m, ds, metric))
            lines.append(join(r))
    return "\n".join(lines) + "\n"


_CELL = re.compile(r"^([-\d.]+)(?: ± ([-\d.]+))?(?: \(n=(\d+)\))?$")


def keep_existing(cells, settings, md_path: Path) -> int:
    """Fill (method, dataset) cells with no local results from an existing leaderboard.md.

    Results are produced on several servers and only the leaderboard is tracked in git, so a
    refresh must not erase rows computed elsewhere. Local results always take precedence.
    Column layout is inferred from the metric-label header so older boards (e.g. without MAE₁₀)
    still parse correctly.
    """
    if not md_path.exists():
        return 0
    kept, group = 0, None
    col_metrics: list[str | None] = []  # index into data cols (after Setting/Method)
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3:
            continue
        # Metric header row: "", "", MAE₂↓, ...
        if cols[1] in ("", "Method") and any(c in LABEL_TO_METRIC for c in cols[2:]):
            col_metrics = [LABEL_TO_METRIC.get(c) for c in cols[2:]]
            continue
        if not col_metrics or cols[1] in ("Method", "") or cols[1] == "---":
            continue
        if cols[0]:
            group = cols[0].strip("*")
        method = cols[1]
        if len(cols) < 2 + len(col_metrics):
            continue
        # Split flat metric header into per-dataset slices. Prefer current
        # DATASET_METRICS widths when the header length matches; else equal split
        # (older boards used the same columns for every dataset).
        expected = [len(DATASET_METRICS[ds]) for ds, _ in DATASETS]
        if len(col_metrics) == sum(expected):
            widths = expected
        elif len(col_metrics) % len(DATASETS) == 0:
            widths = [len(col_metrics) // len(DATASETS)] * len(DATASETS)
        else:
            continue
        offset = 0
        for d, (ds, _) in enumerate(DATASETS):
            width = widths[d]
            if ds in cells.get(method, {}):
                offset += width
                continue
            summ, n_seen = {}, None
            for k in range(width):
                metric = col_metrics[offset + k]
                if not metric:
                    continue
                raw = re.sub(r"\*\*|</?u>", "", cols[2 + offset + k]).strip()
                m = _CELL.match(raw)
                if not m:
                    continue
                summ[metric] = (float(m.group(1)), float(m.group(2) or 0.0))
                n_seen = int(m.group(3)) if m.group(3) else FULL_SEEDS
            if summ:
                cells.setdefault(method, {})[ds] = summ
                SEED_N[(method, ds)] = n_seen
                settings.setdefault(method, group or "Deep methods")
                kept += 1
            offset += width
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/leaderboard.md")
    ap.add_argument("--no_keep", action="store_true",
                    help="do not keep rows of the existing leaderboard that have no local results")
    args = ap.parse_args()
    root = Path(args.results_dir)
    cells, settings = load_cells(root)
    if not args.no_keep:
        n = keep_existing(cells, settings, Path(args.out))
        if n:
            print(f"[keep] {n} method×dataset cells kept from existing {args.out} (no local results)")
    ordered = order_methods(cells)
    md = build_markdown(cells, settings, ordered)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    flat = []
    for m in ordered:
        for ds, dlabel in DATASETS:
            row = {"setting": settings[m], "method": m, "dataset": dlabel}
            if ds in cells[m]:
                for metric in METRICS:
                    if metric in cells[m][ds]:
                        mu, sd = cells[m][ds][metric]
                        row[f"{metric}_mean"] = mu
                        row[f"{metric}_std"] = sd
                        row[metric] = fmt(mu, sd)
            flat.append(row)
    pd.DataFrame(flat).to_csv(out.with_suffix(".csv"), index=False)
    print(md)


if __name__ == "__main__":
    main()
