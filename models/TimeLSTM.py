"""TimeLSTM (Aksamentov et al., MICCAI 2017)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        feature_size = int(mcfg.get("feature_size", 2048))
        if feature_size != 2048:
            raise ValueError(
                f"TimeLSTM expects ResNet-152 features (2048), got feature_size={feature_size}"
            )
        self.max_length = float(mcfg.get("max_len", 20))
        self.lstm = nn.LSTM(feature_size, hidden_size=512, num_layers=1, batch_first=True)
        self.linear_rsd = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_state = None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feat = batch["features"]
        squeeze = False
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            squeeze = True
        init_state = self.last_state if bool(batch.get("stateful", False)) else None
        lstm_output, last_state = self.lstm(feat, init_state)
        self.last_state = (last_state[0].detach(), last_state[1].detach())
        rsd = self.sigmoid(self.linear_rsd(lstm_output)) * self.max_length
        if squeeze:
            rsd = rsd.squeeze(0)
        return {"rsd": rsd.squeeze(-1) if rsd.shape[-1] == 1 else rsd}
