"""BD-Net (Wu et al., MICCAI 2022)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, TypeVar

import numpy as np
import torch
from torch import nn
from torchvision.models import DenseNet169_Weights, densenet169

T = TypeVar("T", bound="nn.Module")


class MtFc(nn.Module):
    def __init__(self, n_in, n_step, n_experience, n_rsd=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_step)
        self.fc2 = nn.Linear(n_in, n_experience)
        self.fc3 = nn.Linear(n_in, n_rsd)
        self.fc4 = nn.Linear(n_in, 1)

    def forward(self, x):
        step = self.fc1(x.clone())
        experience = self.fc2(x.clone())
        rsd = self.fc3(x.clone())
        s = self.fc4(x.clone())
        return step, experience, rsd, s


def _zeros_like(batch: int, hidden: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros(batch, hidden, device=ref.device, dtype=ref.dtype)


class Rnn_Model1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=128)
        hidden_size = self.rnn_cell.hidden_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.fc = MtFc(128, 11, 2)
        self.last_state = None
        self.batch_size = 128
        self.drop_p = 0.0

    def forward(self, X, stateful=False, ret_feature=False):
        if stateful:
            h, c = self.last_state
        else:
            h = _zeros_like(1, self.hidden_size, X)
            c = _zeros_like(1, self.hidden_size, X)
        mask_x = nn.Dropout(p=self.drop_p)(torch.ones(1, self.input_size, device=X.device))
        mask_h = nn.Dropout(p=self.drop_p)(torch.ones(1, self.hidden_size, device=X.device))
        output = []
        for x in X:
            h, c = self.rnn_cell(x * mask_x, (h * mask_h, c))
            output.append(h.view(1, h.size(0), -1))
        output = torch.cat(output, 0)
        self.last_state = (h.detach(), c.detach())
        output = output.squeeze()
        y = output
        step, experience, rsd, s = self.fc(output)
        if ret_feature:
            return step, experience, rsd, s, y
        return step, experience, rsd, s


class Rnn_Model2(Rnn_Model1):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.drop_p = 0.2


class Rnn_Model3(Rnn_Model1):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.drop_p = 0.4


class Rnn_Model4(Rnn_Model1):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.drop_p = 0.6


class Rnn_Model5(Rnn_Model1):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.drop_p = 0.4


class Parallel_fc(nn.Module):
    def __init__(self, n_in, n_1, n_2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_1)
        self.fc2 = nn.Linear(n_in, n_2)

    def forward(self, X):
        y1 = self.fc1(X.clone())
        y2 = self.fc2(X)
        return y1, y2


class CatRSDNet1(nn.Module):
    def __init__(self, n_step_classes=11, n_expertise_classes=2, max_len=20):
        super().__init__()
        feature_size = 1664
        self.max_len = max_len
        self.cnn = self.initCNN(feature_size, n_classes_1=n_step_classes, n_classes_2=n_expertise_classes)
        self.rnn1 = Rnn_Model1(input_size=feature_size, output_size=n_step_classes)
        self.rnn2 = Rnn_Model2(input_size=feature_size, output_size=n_step_classes)
        self.rnn3 = Rnn_Model3(input_size=feature_size, output_size=n_step_classes)
        self.rnn4 = Rnn_Model4(input_size=feature_size, output_size=n_step_classes)
        self.rnn5 = Rnn_Model5(input_size=feature_size, output_size=n_step_classes)

    def train(self: T, mode: bool = True) -> T:
        return super().train(mode)

    def freeze_cnn(self, freeze=True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze=True):
        for rnn in (self.rnn1, self.rnn2, self.rnn3, self.rnn4, self.rnn5):
            for param in rnn.parameters():
                param.requires_grad = not freeze

    def initCNN(self, feature_size, n_classes_1, n_classes_2):
        from utils.pretrained import load_torchvision_with_weights

        cnn = load_torchvision_with_weights(densenet169, DenseNet169_Weights.DEFAULT, "densenet169")
        tmp_conv_weights = cnn.features.conv0.weight.data.clone()
        cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
        cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights
        cnn.classifier = Parallel_fc(n_in=feature_size, n_1=n_classes_1, n_2=n_classes_2)
        return cnn

    def set_cnn_as_feature_extractor(self):
        self.cnn.classifier = torch.nn.Identity()

    def forwardCNN(self, images, elapsed_time):
        rel_time = elapsed_time / self.max_len
        rel_time = (
            torch.ones_like(images[:, 0]).unsqueeze(1) * rel_time[:, None, None, None]
        ).to(images.device)
        images_and_elapsed_time = torch.cat((images, rel_time), 1).float()
        return self.cnn(images_and_elapsed_time)

    def _forward_member(self, rnn, X, stateful=False, skip_features=False):
        if skip_features:
            features = X
        else:
            self.set_cnn_as_feature_extractor()
            features = self.cnn(X)
            features = features.unsqueeze(1)
        return rnn(features, stateful=stateful)

    def forwardRNN1(self, X, stateful=False, skip_features=False):
        return self._forward_member(self.rnn1, X, stateful, skip_features)

    def forwardRNN2(self, X, stateful=False, skip_features=False):
        return self._forward_member(self.rnn2, X, stateful, skip_features)

    def forwardRNN3(self, X, stateful=False, skip_features=False):
        return self._forward_member(self.rnn3, X, stateful, skip_features)

    def forwardRNN4(self, X, stateful=False, skip_features=False):
        return self._forward_member(self.rnn4, X, stateful, skip_features)

    def forwardRNN5(self, X, stateful=False, skip_features=False):
        return self._forward_member(self.rnn5, X, stateful, skip_features)

    def forward(self, X, elapsed_time=None, stateful=False):
        self.set_cnn_as_feature_extractor()
        if elapsed_time is not None:
            assert X.shape[1] == 3
            features = self.forwardCNN(X, elapsed_time.view(-1))
        else:
            assert X.shape[1] == 4
            features = self.cnn(X)
        features = features.unsqueeze(1)
        return self.rnn1(features, stateful=stateful)


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs.get("model", configs)
        self.net = CatRSDNet1(
            n_step_classes=int(mcfg.get("n_phases", 11)),
            n_expertise_classes=int(mcfg.get("n_experience", 2)),
            max_len=float(mcfg.get("max_len", 20)),
        )
        self.max_len = float(mcfg.get("max_len", 20))
        # Chunk DenseNet over time — full-length eval at fps=2.5 otherwise OOMs on V100.
        self.cnn_chunk = int(mcfg.get("cnn_chunk", 24) or 24)

    def _encode_cnn(self, x4: torch.Tensor) -> torch.Tensor:
        """x4: [T,4,H,W] → feats [T,1,C], chunked along T."""
        self.net.set_cnn_as_feature_extractor()
        chunk = max(1, int(self.cnn_chunk))
        parts = []
        for s in range(0, x4.size(0), chunk):
            parts.append(self.net.cnn(x4[s : s + chunk]))
        return torch.cat(parts, dim=0).unsqueeze(1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        single = frames.dim() == 4
        if single:
            frames = frames.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
        assert frames.size(0) == 1
        t, _, h, w = frames.shape[1], frames.shape[2], frames.shape[3], frames.shape[4]
        rel = (elapsed[0] / self.max_len).clamp(0, 1).view(t, 1, 1, 1).expand(t, 1, h, w)
        x4 = torch.cat([frames[0], rel.to(frames.dtype)], dim=1)
        feats = self._encode_cnn(x4)  # [T,1,C] — iterate time in each RNN
        outs = []
        for rnn in (self.net.rnn1, self.net.rnn2, self.net.rnn3, self.net.rnn4, self.net.rnn5):
            step, exp, rsd, s = rnn(feats, stateful=False)
            if rsd.dim() == 1:
                rsd = rsd.unsqueeze(-1)
            outs.append(rsd)
        stack = torch.stack(outs, dim=0)
        rsd_mean = stack.mean(dim=0).squeeze(-1)
        unc = stack.std(dim=0).squeeze(-1)
        step0, _, _, _ = self.net.rnn1(feats, stateful=False)
        return {"rsd": rsd_mean, "uncertainty": unc, "phase_logits": step0}
