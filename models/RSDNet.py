"""RSDNet (Twinanda et al., TMI 2019)."""
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
                f"RSDNet expects ResNet-152 features (2048), got feature_size={feature_size}"
            )
        self.rsd_norm = float(mcfg.get("rsd_norm", mcfg.get("max_len", 5.0)))
        self.lstm = nn.LSTM(feature_size, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear_progress = nn.Linear(513, 1)
        self.linear_rsd = nn.Linear(513, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_state = None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feat = batch["features"]
        elapsed = batch["elapsed"]
        squeeze = False
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
            squeeze = True
        init_state = self.last_state if bool(batch.get("stateful", False)) else None
        lstm_output, last_state = self.lstm(self.dropout(feat), init_state)
        self.last_state = (last_state[0].detach(), last_state[1].detach())
        lstm_output = self.dropout(lstm_output)
        cat = torch.cat((lstm_output, elapsed.unsqueeze(2)), 2)
        progress = self.sigmoid(self.linear_progress(cat))
        rsd = self.linear_rsd(cat) * self.rsd_norm
        if squeeze:
            progress, rsd = progress.squeeze(0), rsd.squeeze(0)
        return {"rsd": rsd.squeeze(-1), "progress": progress.squeeze(-1)}
