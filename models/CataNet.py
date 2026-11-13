"""CataNet (Marafioti et al., MICCAI 2021)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TypeVar

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

    def forward(self, x):
        step = self.fc1(x.clone())
        experience = self.fc2(x.clone())
        rsd = self.fc3(x.clone())
        return step, experience, rsd


class Rnn_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rnn_cell = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.0,
            batch_first=True,
        )
        if isinstance(self.rnn_cell.hidden_size, list):
            hidden_size = self.rnn_cell.hidden_size[-1]
        else:
            hidden_size = self.rnn_cell.hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.last_state = None

    def forward(self, X, stateful=False, ret_feature=False):
        if stateful:
            init_state = self.last_state
        else:
            init_state = None

        y, last_state = self.rnn_cell(X, init_state)
        self.last_state = (last_state[0].detach(), last_state[1].detach())
        y = y.squeeze(0)
        y_out = self.fc(y)
        if ret_feature:
            return y_out, y
        return y_out


class Parallel_fc(nn.Module):
    def __init__(self, n_in, n_1, n_2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_1)
        self.fc2 = nn.Linear(n_in, n_2)

    def forward(self, X):
        y1 = self.fc1(X.clone())
        y2 = self.fc2(X)
        return y1, y2


class CatRSDNet(nn.Module):
    def __init__(self, n_step_classes=11, n_expertise_classes=2, max_len=20):
        super().__init__()
        feature_size = 1664
        self.max_len = max_len
        self.cnn = self.initCNN(feature_size, n_classes_1=n_step_classes, n_classes_2=n_expertise_classes)
        self.rnn = Rnn_Model(input_size=feature_size, output_size=n_step_classes)
        self.rnn.fc = MtFc(128, n_step_classes, n_expertise_classes)

    def train(self: T, mode: bool = True) -> T:
        return super().train(mode)

    def freeze_cnn(self, freeze=True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze=True):
        for param in self.rnn.parameters():
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

    def forwardRNN(self, X, stateful=False, skip_features=False):
        if skip_features:
            features = X
        else:
            self.set_cnn_as_feature_extractor()
            features = self.cnn(X)
            features = features.unsqueeze(0)
        step_prediction, exp_prediction, rsd_predictions = self.rnn(features, stateful=stateful)
        return step_prediction, exp_prediction, rsd_predictions

    def forward(self, X, elapsed_time=None, stateful=False):
        self.set_cnn_as_feature_extractor()
        if elapsed_time is not None:
            assert X.shape[1] == 3, "provide images with RGB channels only if elapsed time is set."
            features = self.forwardCNN(X, elapsed_time.view(-1))
        else:
            assert X.shape[1] == 4, "provide images with RGB+Timestamp if elapsed time is not set."
            features = self.cnn(X)
        features = features.unsqueeze(0)
        step_prediction, exp_prediction, rsd_predictions = self.rnn(features, stateful=stateful)
        return step_prediction, exp_prediction, rsd_predictions


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        mcfg = configs.get("model", configs)
        n_step = int(mcfg.get("n_phases", mcfg.get("n_step_classes", 11)))
        n_exp = int(mcfg.get("n_experience", 2))
        max_len = float(mcfg.get("max_len", 20.0))
        self.net = CatRSDNet(
            n_step_classes=n_step,
            n_expertise_classes=n_exp,
            max_len=max_len,
        )
        self.max_len = max_len
        self.use_experience = bool(mcfg.get("use_experience", True))

    def freeze_cnn(self, freeze=True):
        self.net.freeze_cnn(freeze)

    def freeze_rnn(self, freeze=True):
        self.net.freeze_rnn(freeze)

    def set_cnn_as_feature_extractor(self):
        self.net.set_cnn_as_feature_extractor()

    @property
    def cnn(self):
        return self.net.cnn

    @property
    def rnn(self):
        return self.net.rnn

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        elapsed = batch["elapsed"]
        single = frames.dim() == 4
        if single:
            frames = frames.unsqueeze(0)
            elapsed = elapsed.unsqueeze(0)
        b, t, c, h, w = frames.shape
        assert b == 1, "batch_size=1 required"
        rel = (elapsed[0] / self.max_len).clamp(0, 1).view(t, 1, 1, 1).expand(t, 1, h, w)
        x4 = torch.cat([frames[0], rel.to(frames.dtype)], dim=1)  # (T,4,H,W)
        self.net.set_cnn_as_feature_extractor()
        step, exp, rsd = self.net.forwardRNN(x4, stateful=False, skip_features=False)
        rsd = rsd.squeeze(-1) if rsd.dim() > 1 else rsd
        out: Dict[str, torch.Tensor] = {
            "rsd": rsd,
            "phase_logits": step,
        }
        if self.use_experience:
            out["experience_logits"] = exp
        return out
