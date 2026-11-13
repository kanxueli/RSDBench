from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions_csv(
    rows: list,
    path: Union[str, Path],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_repeat_metrics(
    repeat_metrics: Mapping[str, Mapping[str, float]],
) -> dict:
    """repeat_metrics[repeat_id][metric] -> {metric: {mean, std, values}}."""
    from utils.metrics import mean_std

    keys = set()
    for m in repeat_metrics.values():
        keys.update(m.keys())
    out = {}
    for k in sorted(keys):
        vals = [repeat_metrics[r][k] for r in sorted(repeat_metrics) if k in repeat_metrics[r]]
        mu, sd = mean_std(vals)
        out[k] = {"mean": mu, "std": sd, "values": vals}
    return out


def save_summary(output_dir: Union[str, Path], repeat_metrics: Mapping[str, Mapping[str, float]]) -> dict:
    """Write <output_dir>/summary.json over every repeat in the experiment dir.

    Seeds are often run one process at a time into the same experiment dir
    (split.n_repeats=1), so the summary pools all <output_dir>/{seed*,fold*}/metrics.json
    with this run's repeat_metrics (this run wins on duplicate ids).
    """
    output_dir = Path(output_dir)
    merged = {}
    for p in sorted(output_dir.glob("seed*/metrics.json")) + sorted(output_dir.glob("fold*/metrics.json")):
        merged[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    merged.update(repeat_metrics)
    summary = summarize_repeat_metrics(merged)
    save_json({"repeats": merged, "summary": summary}, output_dir / "summary.json")
    return {"summary": summary, "repeats": merged}
