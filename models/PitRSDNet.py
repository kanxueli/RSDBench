"""PitRSDNet (Wijekoon et al., HTL 2024)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from layers.time_channel import append_elapsed_channel

def _head(d_in: int, hidden, d_out: int) -> nn.Module:
    """Linear when hidden is empty, else MLP d_in → hidden[0] → … → d_out with ReLU."""
    layers: List[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(d_in, int(h)), nn.ReLU()]
        d_in = int(h)
    layers.append(nn.Linear(d_in, d_out))
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs.get("model", configs)
        self.rsd_norm = float(mcfg.get("rsd_norm", 10.0))
        # elapsed 4th channel = elapsed / max_len, clamped to [0, 1]; must cover the longest surgery
        self.max_len = float(mcfg.get("max_len", 20.0))
        self.n_phases = int(mcfg.get("n_phases", 7))
        self.chunk = int(mcfg.get("cnn_chunk", 32))
        self.context_hat_t = int(mcfg.get("context_hat_t", 5))
        self.hidden = int(mcfg.get("lstm_hidden", 512))
        head_hidden = list(mcfg.get("head_hidden", [512, 256]))

        from utils.pretrained import load_torchvision_with_weights

        cnn = load_torchvision_with_weights(convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, "convnext_tiny")
        old = cnn.features[0][0]
        new = nn.Conv2d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3] = old.weight.mean(dim=1)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        cnn.features[0][0] = new
        cnn.classifier[2] = nn.Identity()
        self.cnn = cnn
        self.feature_size = 768

        # f ⊕ last_step_probs ⊕ mean_step_probs
        lstm_in = self.feature_size + 2 * self.n_phases
        self.lstm = nn.LSTM(lstm_in, self.hidden, num_layers=2, batch_first=True)
        self.fc_step = _head(self.hidden, head_hidden, self.n_phases)
        self.fc_rsd = _head(self.hidden, head_hidden, 1)

    def _cnn_feats(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.size(0), self.chunk):
            outs.append(self.cnn(x[i : i + self.chunk]))
        return torch.cat(outs, dim=0)

    def encode(self, frames: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        """RGB frames (T,3,H,W) + elapsed minutes (T,) → ConvNeXt features (T,F)."""
        x = append_elapsed_channel(frames, elapsed, self.max_len)
        return self._cnn_feats(x)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        single = frames.dim() == 4
        if single:
            frames = frames.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
        x = append_elapsed_channel(frames, elapsed, self.max_len)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        b, t, c, h, w = x.shape
        assert b == 1, "PitRSDNet online context assumes batch_size=1 sequences"
        feat = self._cnn_feats(x.reshape(b * t, c, h, w)).reshape(b, t, -1)[0]  # (T,F)
        return self.forward_features(feat)

    def forward_features(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage-2 LSTM over (frozen) per-frame encoder features feat (T,F)."""
        t = feat.size(0)
        step_logits_list: List[torch.Tensor] = []
        rsd_list: List[torch.Tensor] = []
        step_prob_hist: List[torch.Tensor] = []
        h_state: Optional[tuple] = None
        prev_step = feat.new_zeros(self.n_phases)
        for i in range(t):
            if step_prob_hist:
                mean_ctx = torch.stack(step_prob_hist[-self.context_hat_t :], dim=0).mean(dim=0)
            else:
                mean_ctx = feat.new_zeros(self.n_phases)
            lt = torch.cat([feat[i], prev_step, mean_ctx], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,D)
            out, h_state = self.lstm(lt, h_state)
            out = out.squeeze(0).squeeze(0)
            logits = self.fc_step(out)
            rsd_n = self.fc_rsd(out)
            probs = F.softmax(logits, dim=-1)
            step_logits_list.append(logits)
            rsd_list.append(rsd_n.squeeze(-1) * self.rsd_norm)
            step_prob_hist.append(probs)
            prev_step = probs


        phase_logits = torch.stack(step_logits_list, dim=0)
        rsd = torch.stack(rsd_list, dim=0)
        return {"rsd": rsd, "phase_logits": phase_logits}
