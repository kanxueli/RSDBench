#!/usr/bin/env python
"""Export cholec80_main train/val/test per seed for FT / SWAG / external tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.splits import assert_disjoint_split, build_protocol_splits, video_id_to_cholec_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/cholec80_main.yaml")
    ap.add_argument("--out_dir", default="data_provider/splits/cholec80_main_exports")
    args = ap.parse_args()

    proto_path = Path(args.protocol)
    if not proto_path.is_absolute():
        proto_path = ROOT / proto_path
    protocol = yaml.safe_load(proto_path.read_text(encoding="utf-8"))
    pack = build_protocol_splits(protocol, ROOT)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for rep in pack["repeats"]:
        assert_disjoint_split(rep["train"], rep["test"], rep.get("val") or [])
        seed = int(rep["seed"])
        payload = {
            "seed": seed,
            "repeat": rep["repeat"],
            "train": list(rep["train"]),
            "val": list(rep.get("val") or []),
            "test": list(rep["test"]),
            "train_int": [video_id_to_cholec_int(v) for v in rep["train"]],
            "val_int": [video_id_to_cholec_int(v) for v in (rep.get("val") or [])],
            "test_int": [video_id_to_cholec_int(v) for v in rep["test"]],
        }
        path = out / f"seed{seed}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index.append({"seed": seed, "path": str(path.relative_to(ROOT)), **{k: len(payload[k]) for k in ("train", "val", "test")}})
        print(f"wrote {path} train={len(payload['train'])} val={len(payload['val'])} test={len(payload['test'])}")

    (out / "index.json").write_text(json.dumps({"protocol": str(proto_path), "repeats": index}, indent=2), encoding="utf-8")
    print(f"index → {out / 'index.json'}")


if __name__ == "__main__":
    main()
