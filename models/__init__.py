"""Model zoo registry (TSlib-style: one file per method)."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any, Dict

MODEL_DICT = {
    "Naive": "models.Naive",
    "NaiveMedian": "models.Naive",
    "NaiveMean": "models.Naive",
    "ConditionalMedian": "models.ConditionalMedian",
    "PhaseInferred": "models.PhaseInferred",
    "SequenceMatch": "models.SequenceMatch",
    "BayesianAverage": "models.BayesianAverage",
    "BayesianConditional": "models.BayesianConditional",
    "KostopoulosRF": "models.KostopoulosRF",
    "PadoyHMM": "models.PadoyHMM",
    "TimeLSTM": "models.TimeLSTM",
    "RSDNet": "models.RSDNet",
    "CataNet": "models.CataNet",
    "BDNet": "models.BDNet",
    "PitRSDNet": "models.PitRSDNet",
    "TransLocal": "models.TransLocal",
    "VTDNet": "models.VTDNet",
    "SWAG": "models.SWAG",
    "Rivoir2019": "models.Rivoir2019",
    "Rivoir2020": "models.Rivoir2020",
    "SUPRGAN": "models.SUPRGAN",
}


def build_model(name: str, configs: Dict[str, Any]):
    if name not in MODEL_DICT:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_DICT)}")
    cfg = deepcopy(configs)
    model_block = dict(cfg.get("model") or {})
    model_block["name"] = name
    if name == "NaiveMean":
        model_block["stat"] = "mean"
    elif name == "NaiveMedian":
        model_block.setdefault("stat", "median")
    cfg["model"] = model_block
    cfg["model_name"] = name
    mod = import_module(MODEL_DICT[name])
    return mod.Model(cfg)
