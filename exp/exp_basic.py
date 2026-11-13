from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from models import build_model
from utils.config import ConfigDict


class ExpBasic:
    def __init__(self, configs: Mapping[str, Any]):
        self.configs = ConfigDict(configs) if not isinstance(configs, ConfigDict) else configs
        self.device = self._acquire_device()
        self.model = None

    def _acquire_device(self) -> torch.device:
        allow_cpu = bool(self.configs.get("allow_cpu", False))
        want = str(self.configs.get("device", "cuda"))
        if not want.startswith("cuda"):
            if allow_cpu:
                return torch.device(want)
            raise RuntimeError(
                f"device='{want}' rejected: RSDBench experiments must run on GPU. "
                "Use device=cuda (or set allow_cpu=true only for debugging)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this Python environment. "
                "Activate conda env `rsdbench` with a CUDA build of PyTorch "
                "(do NOT install the CPU-only wheel)."
            )
        # optional device index: cuda / cuda:0
        return torch.device(want)

    def _build_model(self):
        name = self.configs.get("model_name") or self.configs.get("model", {}).get("name")
        if not name:
            raise ValueError("configs must define model_name or model.name")
        model = build_model(name, dict(self.configs))
        model = model.to(self.device)
        if self.device.type == "cuda" and bool(self.configs.get("cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True
        return model

    @property
    def output_dir(self) -> Path:
        root = Path(self.configs.get("output_dir", "results"))
        exp = self.configs.get("exp_name", "default")
        return root / exp
