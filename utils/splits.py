"""Split generation for RSDBench protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

from utils.seed import repeat_seeds


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def make_repeated_holdout(
    video_ids: Sequence[Any],
    n_repeats: int = 5,
    seed_start: int = 42,
    train_ratio: float = 0.75,
    test_ratio: float = 0.25,
    val_ratio_in_train: float = 0.2,
    use_val: bool = True,
) -> List[Dict[str, Any]]:
    """
    N x train(/val)/test random splits.
    Seeds = seed_start .. seed_start+n_repeats-1 (default 42,43,...).
    """
    ids = list(video_ids)
    n = len(ids)
    n_test = int(round(n * test_ratio))
    n_train_pool = n - n_test
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "train_ratio + test_ratio must be 1"

    splits = []
    for seed in repeat_seeds(seed_start, n_repeats):
        order = _rng(seed).permutation(n)
        shuffled = [ids[i] for i in order]
        test_ids = sorted(shuffled[:n_test], key=lambda x: (str(type(x)), x))
        train_pool = shuffled[n_test:]
        if use_val and val_ratio_in_train > 0:
            n_val = max(1, int(round(n_train_pool * val_ratio_in_train)))
            val_ids = sorted(train_pool[:n_val], key=lambda x: (str(type(x)), x))
            train_ids = sorted(train_pool[n_val:], key=lambda x: (str(type(x)), x))
        else:
            val_ids = []
            train_ids = sorted(train_pool, key=lambda x: (str(type(x)), x))
        splits.append(
            {
                "repeat": len(splits),
                "seed": seed,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
        )
    return splits


def load_cataract_catanet_folds(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Legacy CataNet fixed-test + 6-fold pack (kept for appendix / ablation only)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    folds = raw["train"]
    test_ids = list(raw["test"])
    fold_splits = []
    fold_names = sorted(folds.keys(), key=lambda x: int(x.replace("fold", "")))
    all_train_pool = []
    for name in fold_names:
        all_train_pool.extend(folds[name])
    all_train_pool = sorted(set(all_train_pool))

    for name in fold_names:
        val_ids = list(folds[name])
        train_ids = [i for i in all_train_pool if i not in set(val_ids)]
        fold_splits.append(
            {
                "fold": name,
                "fold_index": int(name.replace("fold", "")),
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
        )
    return {
        "scheme": "fixed_holdout_inner_kfold",
        "test": test_ids,
        "train_pool": all_train_pool,
        "folds": fold_splits,
    }


def cholec80_video_ids(n: int = 80) -> List[str]:
    return [f"video{i:02d}" for i in range(1, n + 1)]


def load_cataract_surgeon_map(videos_csv: Union[str, Path]) -> Dict[int, int]:
    """VideoID -> Surgeon id from cataract-101 videos.csv (semicolon)."""
    import pandas as pd

    df = pd.read_csv(videos_csv, sep=";")
    return {int(r.VideoID): int(r.Surgeon) for _, r in df.iterrows()}


def make_cataract_stratified_repeated_holdout(
    surgeon_map: Mapping[int, int],
    n_repeats: int = 5,
    seed_start: int = 42,
    n_test_per_surgeon: int = 5,
    val_ratio_in_train: float = 0.2,
    use_val: bool = True,
) -> List[Dict[str, Any]]:
    """
    Cataract main protocol (aligned with Cholec reporting style):
    - Each seed re-samples train/val/test (IDs change).
    - Test is surgeon-stratified: n_test_per_surgeon videos per surgeon
      (default 5 × 4 surgeons = 20), not a pure global shuffle.
    """
    by_surg: Dict[int, List[int]] = {}
    for vid, surg in surgeon_map.items():
        by_surg.setdefault(int(surg), []).append(int(vid))
    for surg in by_surg:
        by_surg[surg] = sorted(by_surg[surg])

    splits: List[Dict[str, Any]] = []
    for seed in repeat_seeds(seed_start, n_repeats):
        rng = _rng(seed)
        test_ids: List[int] = []
        train_pool: List[int] = []
        per_surg_test: Dict[int, List[int]] = {}
        for surg, vids in sorted(by_surg.items()):
            if len(vids) < n_test_per_surgeon:
                raise ValueError(
                    f"surgeon {surg} has {len(vids)} videos < n_test_per_surgeon={n_test_per_surgeon}"
                )
            order = list(rng.permutation(len(vids)))
            chosen = [vids[i] for i in order[:n_test_per_surgeon]]
            rest = [vids[i] for i in order[n_test_per_surgeon:]]
            per_surg_test[surg] = sorted(chosen)
            test_ids.extend(chosen)
            train_pool.extend(rest)
        test_ids = sorted(test_ids)
        rng.shuffle(train_pool)
        if use_val and val_ratio_in_train > 0:
            n_val = max(1, int(round(len(train_pool) * val_ratio_in_train)))
            val_ids = sorted(train_pool[:n_val])
            train_ids = sorted(train_pool[n_val:])
        else:
            val_ids = []
            train_ids = sorted(train_pool)
        assert_disjoint_split(train_ids, test_ids, val_ids)
        splits.append(
            {
                "repeat": len(splits),
                "seed": seed,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "test_per_surgeon": {str(k): v for k, v in per_surg_test.items()},
            }
        )
    return splits


def build_protocol_splits(protocol: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    """Build or load splits described by a protocol config."""
    dataset = protocol["dataset"]
    split_cfg = protocol["split"]
    scheme = split_cfg["scheme"]

    if dataset == "cholec80" and scheme == "repeated_holdout":
        splits = make_repeated_holdout(
            video_ids=cholec80_video_ids(80),
            n_repeats=int(split_cfg.get("n_repeats", 5)),
            seed_start=int(split_cfg.get("seed_start", 42)),
            train_ratio=float(split_cfg.get("train_ratio", 0.75)),
            test_ratio=float(split_cfg.get("test_ratio", 0.25)),
            val_ratio_in_train=float(split_cfg.get("val_ratio_in_train", 0.2)),
            use_val=bool(split_cfg.get("use_val", True)),
        )
        return {
            "dataset": dataset,
            "scheme": scheme,
            "n_repeats": split_cfg.get("n_repeats", 5),
            "seed_start": split_cfg.get("seed_start", 42),
            "seeds": repeat_seeds(
                int(split_cfg.get("seed_start", 42)),
                int(split_cfg.get("n_repeats", 5)),
            ),
            "repeats": splits,
        }

    if dataset == "cataract101" and scheme in (
        "stratified_repeated_holdout",
        "repeated_holdout",
    ):
        # Main Cataract protocol: re-split every seed; stratify test by surgeon.
        data_root = protocol.get("data_root")  # optional
        videos_csv = (
            Path(split_cfg["videos_csv"])
            if split_cfg.get("videos_csv")
            else repo_root.parent / "DATASET" / "cataract-101" / "videos.csv"
        )
        # Prefer DATASET path from configs/data if present via env-less default
        if not videos_csv.exists():
            alt = Path("${RSDBENCH_DATA_ROOT}/cataract-101/videos.csv")
            if alt.exists():
                videos_csv = alt
        surgeon_map = load_cataract_surgeon_map(videos_csv)
        splits = make_cataract_stratified_repeated_holdout(
            surgeon_map=surgeon_map,
            n_repeats=int(split_cfg.get("n_repeats", 5)),
            seed_start=int(split_cfg.get("seed_start", 42)),
            n_test_per_surgeon=int(split_cfg.get("n_test_per_surgeon", 5)),
            val_ratio_in_train=float(split_cfg.get("val_ratio_in_train", 0.2)),
            use_val=bool(split_cfg.get("use_val", True)),
        )
        return {
            "dataset": dataset,
            "scheme": "stratified_repeated_holdout",
            "n_repeats": int(split_cfg.get("n_repeats", 5)),
            "seed_start": int(split_cfg.get("seed_start", 42)),
            "seeds": repeat_seeds(
                int(split_cfg.get("seed_start", 42)),
                int(split_cfg.get("n_repeats", 5)),
            ),
            "n_test_per_surgeon": int(split_cfg.get("n_test_per_surgeon", 5)),
            "split_note": (
                "each seed re-splits; test stratified ~5 videos/surgeon; "
                "not CataNet fixed-test+6fold"
            ),
            "repeats": splits,
        }

    if dataset == "cataract101" and scheme == "fixed_holdout_inner_kfold":
        # Appendix only: original CataNet fixed test + 6-fold
        path = repo_root / "data_provider" / "splits" / "cataract101_catanet.yaml"
        pack = load_cataract_catanet_folds(path)
        pack["dataset"] = dataset
        pack["n_repeats"] = int(split_cfg.get("n_repeats", 1))
        pack["seed_start"] = int(split_cfg.get("seed_start", 42))
        pack["seeds"] = repeat_seeds(
            int(split_cfg.get("seed_start", 42)),
            int(split_cfg.get("n_repeats", 1)),
        )
        pack["ensemble"] = bool(split_cfg.get("ensemble", True))
        pack["split_note"] = "LEGACY appendix: CataNet fixed test + 6fold"
        return pack

    raise ValueError(f"Unsupported protocol split: dataset={dataset}, scheme={scheme}")


def save_splits(splits: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Write splits; repeated-holdout repeats are merged by seed with an existing file.

    Seeds are often run one process at a time into the same experiment dir
    (split.n_repeats=1), so overwriting would keep only the last seed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(splits)
    if path.exists() and "repeats" in out:
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        if prev.get("scheme") == out.get("scheme") and prev.get("dataset") == out.get("dataset"):
            by_seed = {int(r["seed"]): r for r in prev.get("repeats", [])}
            by_seed.update({int(r["seed"]): r for r in out["repeats"]})
            out["repeats"] = [by_seed[k] for k in sorted(by_seed)]
            out["seeds"] = sorted(by_seed)
            out["n_repeats"] = len(by_seed)
            out["seed_start"] = min(by_seed)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_splits(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_disjoint_split(train_ids: Sequence[Any], test_ids: Sequence[Any], val_ids: Optional[Sequence[Any]] = None) -> None:
    """Test IDs must never appear in train or val (fair comparison invariant)."""
    train_set, test_set = set(train_ids), set(test_ids)
    leak = train_set & test_set
    if leak:
        raise ValueError(f"Train/test leakage ({len(leak)} ids), e.g. {sorted(leak, key=str)[:5]}")
    if val_ids:
        val_set = set(val_ids)
        leak_v = val_set & test_set
        if leak_v:
            raise ValueError(f"Val/test leakage ({len(leak_v)} ids), e.g. {sorted(leak_v, key=str)[:5]}")
        leak_tv = train_set & val_set
        if leak_tv:
            raise ValueError(f"Train/val overlap ({len(leak_tv)} ids)")


def video_id_to_cholec_int(vid: Any) -> int:
    """video01 / 1 → 1 for SWAG-style integer indices."""
    if isinstance(vid, int):
        return int(vid)
    s = str(vid)
    if s.startswith("video"):
        return int(s.replace("video", ""))
    return int(s)


def get_protocol_repeat(splits_pack: Mapping[str, Any], repeat_index: int = 0) -> Dict[str, Any]:
    """Return one repeated_holdout / stratified_repeated_holdout entry by index."""
    if "repeats" not in splits_pack:
        raise ValueError("Expected pack with 'repeats'")
    reps = splits_pack["repeats"]
    if repeat_index < 0 or repeat_index >= len(reps):
        raise IndexError(f"repeat_index={repeat_index} out of range [0,{len(reps)})")
    rep = dict(reps[repeat_index])
    assert_disjoint_split(rep["train"], rep["test"], rep.get("val") or [])
    return rep
