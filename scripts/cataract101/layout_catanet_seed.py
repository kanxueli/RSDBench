#!/usr/bin/env python3
"""Symlink CataNet pool into train/val/test for one stratified seed split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def link_ids(ids, pool: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in list(dest.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    for vid in ids:
        src = pool / str(vid)
        if not src.exists():
            raise FileNotFoundError(src)
        (dest / str(vid)).symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits_json", type=Path, required=True, help="protocol splits.json")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    pack = json.loads(args.splits_json.read_text(encoding="utf-8"))
    rep = next(r for r in pack["repeats"] if int(r["seed"]) == int(args.seed))
    link_ids(rep["train"], args.pool, args.out / "train")
    link_ids(rep.get("val") or [], args.pool, args.out / "val")
    link_ids(rep["test"], args.pool, args.out / "test")
    print(
        f"seed{args.seed}: train={len(rep['train'])} val={len(rep.get('val') or [])} "
        f"test={len(rep['test'])} -> {args.out}"
    )


if __name__ == "__main__":
    main()
