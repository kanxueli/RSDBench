#!/usr/bin/env python
"""Lightweight sanity checks (no GPU / no video decode required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_splits():
    from utils.splits import build_protocol_splits, make_repeated_holdout, cholec80_video_ids, load_cataract_catanet_folds

    reps = make_repeated_holdout(cholec80_video_ids(), n_repeats=5, seed_start=42)
    assert len(reps) == 5
    assert [r["seed"] for r in reps] == [42, 43, 44, 45, 46]
    for r in reps:
        assert len(r["test"]) == 20, r
        assert len(r["train"]) + len(r["val"]) == 60, r
        assert len(r["train"]) == 48 and len(r["val"]) == 12, r
        ids = set(r["train"]) | set(r["val"]) | set(r["test"])
        assert len(ids) == 80
        assert not (set(r["train"]) & set(r["test"]))
        assert not (set(r["val"]) & set(r["test"]))
        assert not (set(r["train"]) & set(r["val"]))
    # different seeds => different splits (extremely likely)
    assert reps[0]["test"] != reps[1]["test"]

    cat = load_cataract_catanet_folds(ROOT / "data_provider/splits/cataract101_catanet.yaml")
    assert len(cat["test"]) == 20
    assert len(cat["train_pool"]) == 81
    assert len(cat["folds"]) == 6
    for f in cat["folds"]:
        assert set(f["test"]) == set(cat["test"])
        assert len(f["train"]) + len(f["val"]) == 81
        assert not (set(f["train"]) & set(f["val"]))
        assert not (set(f["val"]) & set(f["test"]))
    print("[ok] splits")


def check_config_merge():
    from utils.config import load_experiment_config

    cfg = load_experiment_config(
        model_cfg=ROOT / "configs/model/ConditionalMedian.yaml",
        protocol_cfg=ROOT / "configs/protocol/cholec80_main.yaml",
        data_cfg=ROOT / "configs/data/default.yaml",
        train_cfg=ROOT / "configs/train/default.yaml",
        overrides={"split": {"n_repeats": 3}, "model_name": "ConditionalMedian"},
    )
    assert cfg["dataset"] == "cholec80"
    assert cfg["split"]["n_repeats"] == 3
    assert cfg["split"]["seed_start"] == 42
    assert cfg["sampling"]["fps"] == 1.0
    assert str(cfg["cholec80"]["root"]).endswith("cholec80")
    assert Path(cfg["cholec80"]["root"]).is_dir(), cfg["cholec80"]["root"]
    assert cfg["device"] == "cuda"
    assert cfg["allow_cpu"] is False
    print("[ok] config merge + data root exists")


def check_metrics():
    import numpy as np
    from utils.metrics import compute_dataset_metrics, mean_std, information_gain

    per = {
        "v1": {"pred": np.array([1.0, 2.0, 3.0]), "target": np.array([1.0, 2.5, 4.0])},
        "v2": {"pred": np.array([0.0, 0.0]), "target": np.array([1.0, 1.0])},
    }
    m = compute_dataset_metrics(per, windows=[2, 5])
    assert "mae_all" in m and np.isfinite(m["mae_all"])
    assert "rel_err_all" in m and "smape_all" in m
    assert np.isfinite(m["rel_err_all"]) and np.isfinite(m["smape_all"])
    mu, sd = mean_std([1.0, 2.0, 3.0])
    assert abs(mu - 2.0) < 1e-6 and sd > 0
    assert abs(information_gain(10.0, 7.0) - 3.0) < 1e-6
    print("[ok] metrics")


def check_protocol_build():
    from utils.config import load_yaml
    from utils.splits import build_protocol_splits

    proto = load_yaml(ROOT / "configs/protocol/cholec80_main.yaml")
    pack = build_protocol_splits(proto, ROOT)
    assert pack["seeds"] == [42, 43, 44, 45, 46]
    assert len(pack["repeats"]) == 5
    print("[ok] protocol split build")


def check_dataset_labels_smoke():
    """Load one Cholec80 / Cataract meta without decoding frames."""
    from data_provider.datasets.cholec80 import Cholec80RSD
    from data_provider.datasets.cataract101 import Cataract101RSD

    c80 = Cholec80RSD(
        root="${RSDBENCH_DATA_ROOT}/cholec80",
        video_ids=["video01"],
        sample_fps=1.0,
        return_frames=False,
    )
    m = c80.iter_meta()[0]
    assert m["elapsed"].ndim == 1 and len(m["elapsed"]) > 10
    assert m["rsd"][0] >= m["rsd"][-1] - 1e-4
    assert m["phase"].min() >= 0 and m["phase"].max() <= 6
    assert m["tools"] is not None and m["tools"].shape[1] == 7
    print("[ok] cholec80 meta", "T=", len(m["elapsed"]), "dur=", round(m["duration_min"], 2))

    cat = Cataract101RSD(
        root="${RSDBENCH_DATA_ROOT}/cataract-101",
        video_ids=[269],
        sample_fps=1.0,
        return_frames=False,
    )
    m2 = cat.iter_meta()[0]
    assert m2["phase"].min() >= 0 and m2["phase"].max() <= 9
    print("[ok] cataract101 meta", "T=", len(m2["elapsed"]), "dur=", round(m2["duration_min"], 2))


def main():
    check_splits()
    check_config_merge()
    check_metrics()
    check_protocol_build()
    check_dataset_labels_smoke()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
