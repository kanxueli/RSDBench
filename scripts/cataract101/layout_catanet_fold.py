#!/usr/bin/env python3
"""Symlink CataNet pool videos into train/val/test for one fold (no file move)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def link_ids(ids, pool: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # clear previous links/dirs
    for child in dest.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    for vid in ids:
        src = pool / str(vid)
        if not src.exists():
            raise FileNotFoundError(src)
        target = dest / str(vid)
        target.symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="layout root with train/val/test")
    ap.add_argument("--fold", type=int, required=True, choices=range(1, 7))
    ap.add_argument(
        "--data_config",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    with open(args.data_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_ids = []
    for name, ids in cfg["train"].items():
        if name != f"fold{args.fold}":
            train_ids.extend(ids)
    val_ids = list(cfg["train"][f"fold{args.fold}"])
    test_ids = list(cfg["test"])

    link_ids(train_ids, args.pool, args.out / "train")
    link_ids(val_ids, args.pool, args.out / "val")
    link_ids(test_ids, args.pool, args.out / "test")
    print(
        f"fold{args.fold}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
