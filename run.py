#!/usr/bin/env python
"""RSDBench unified entry (activate conda env `rsdbench` before running)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import ConfigDict, dump_resolved_config, load_experiment_config
from utils.seed import set_seed


BASELINE_NAMES = {
    "Naive",
    "NaiveMedian",
    "NaiveMean",
    "ConditionalMedian",
    "PhaseInferred",
    "SequenceMatch",
    "BayesianAverage",
    "BayesianConditional",
    "KostopoulosRF",
    "PadoyHMM",
}


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Override must be key=value, got: {p}")
        k, v = p.split("=", 1)
        # nested keys: split.n_repeats=3
        cur = out
        parts = k.split(".")
        for name in parts[:-1]:
            cur = cur.setdefault(name, {})
        # cast ints/floats/bools
        if v.lower() in {"true", "false"}:
            val = v.lower() == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        cur[parts[-1]] = val
    return out


def main():
    parser = argparse.ArgumentParser(description="RSDBench runner")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. ConditionalMedian")
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Protocol yaml, e.g. configs/protocol/cholec80_main.yaml",
    )
    parser.add_argument("--model_cfg", type=str, default=None, help="Optional model yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data/default.yaml")
    parser.add_argument("--train_cfg", type=str, default="configs/train/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dotlist overrides, e.g. split.n_repeats=3 split.seed_start=42",
    )
    args = parser.parse_args()

    model_cfg = args.model_cfg or f"configs/model/{args.model}.yaml"
    if not Path(model_cfg).exists():
        # minimal inline model cfg
        model_cfg_path = ROOT / "configs" / "model" / "_minimal.yaml"
        model_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_cfg_path.exists():
            model_cfg_path.write_text("model: {}\n", encoding="utf-8")
        model_cfg = str(model_cfg_path)

    overrides = parse_overrides(args.override)
    overrides["model_name"] = args.model
    overrides.setdefault("model", {})
    if isinstance(overrides["model"], dict):
        overrides["model"]["name"] = args.model
    overrides["output_dir"] = args.output_dir
    overrides["exp_name"] = args.exp_name or f"{Path(args.protocol).stem}_{args.model}"

    cfg = load_experiment_config(
        model_cfg=model_cfg,
        protocol_cfg=args.protocol,
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        overrides=overrides,
    )
    cfg = ConfigDict(cfg)

    import torch

    is_baseline = args.model in BASELINE_NAMES
    # Deep models must use GPU. Statistical baselines may run on CPU (no CNN).
    require_gpu = (not is_baseline) and (not bool(cfg.get("allow_cpu", False)))
    if require_gpu:
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA unavailable. Activate `conda activate rsdbench` with CUDA PyTorch. "
                "Deep models cannot run on CPU."
            )
        print(f"[RSDBench] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    elif torch.cuda.is_available():
        print(f"[RSDBench] GPU available: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    else:
        print("[RSDBench] CUDA not available — running statistical baseline on CPU")
        cfg["device"] = "cpu"
        cfg["allow_cpu"] = True

    set_seed(int(cfg.get("seed", 42)))

    out = Path(cfg["output_dir"]) / cfg["exp_name"]
    out.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(cfg, out / "resolved_config.yaml")

    if args.model in BASELINE_NAMES:
        from exp.exp_baseline import ExpBaseline

        exp = ExpBaseline(cfg, ROOT)
    elif args.model == "CataNet":
        from exp.exp_catanet import ExpCataNet

        exp = ExpCataNet(cfg, ROOT)
    elif args.model == "BDNet":
        from exp.exp_bdnet import ExpBDNet

        exp = ExpBDNet(cfg, ROOT)
    elif args.model == "PitRSDNet":
        from exp.exp_pitrsdnet import ExpPitRSDNet

        exp = ExpPitRSDNet(cfg, ROOT)
    elif args.model == "VTDNet":
        from exp.exp_vtdnet import ExpVTDNet

        exp = ExpVTDNet(cfg, ROOT)
    elif args.model == "SWAG":
        from exp.exp_swag import ExpSWAG

        exp = ExpSWAG(cfg, ROOT)
    elif args.model == "Rivoir2019":
        from exp.exp_rivoir2019 import ExpRivoir2019

        exp = ExpRivoir2019(cfg, ROOT)
    elif args.model == "Rivoir2020":
        from exp.exp_rivoir2020 import ExpRivoir2020

        exp = ExpRivoir2020(cfg, ROOT)
    elif args.model == "SUPRGAN":
        from exp.exp_suprgan import ExpSUPRGAN

        exp = ExpSUPRGAN(cfg, ROOT)
    elif args.model in ("TimeLSTM", "RSDNet") or bool(cfg.get("use_features", False)):
        from exp.exp_feature_rsd import ExpFeatureRSD

        exp = ExpFeatureRSD(cfg, ROOT)
    else:
        from exp.exp_rsd import ExpRSD

        exp = ExpRSD(cfg, ROOT)

    result = exp.run()
    print("==== summary (mean ± std) ====")
    for metric, pack in result["summary"].items():
        print(f"{metric}: {pack['mean']:.4f} ± {pack['std']:.4f}")


if __name__ == "__main__":
    main()
