"""SWAG stage 2 (Boels et al.)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from layers.informer.models.model import Informer


class KeyRecorder(nn.Module):
    def __init__(self, num_ctx_tokens: int = 12, dim: int = 512, reduc_dim: int = 64, sampling_rate: int = 10):
        super().__init__()
        self.linear_reduction = nn.Sequential(nn.Linear(dim, reduc_dim), nn.ReLU(), nn.LayerNorm(reduc_dim))
        self._linear_expand = nn.Sequential(nn.Linear(reduc_dim, dim), nn.ReLU(), nn.LayerNorm(dim))
        self._sampling_rate = sampling_rate
        self._local_size = num_ctx_tokens

    def forward(self, obs_frames: torch.Tensor) -> torch.Tensor:
        compressed = self.linear_reduction(obs_frames)
        past_pooled = torch.max(compressed[:, : -self._local_size : self._sampling_rate, :], dim=1)[0]
        present = compressed[:, -self._local_size :, :]
        global_max = 0 + present
        for i in range(present.size(1)):
            past_pooled = torch.max(past_pooled, present[:, i, :])
            global_max[:, i, :] = past_pooled
        return self._linear_expand(global_max)


class FusionHead(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim))

    def forward(self, key_feats: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        feats = self.norm1(key_feats + feats)
        return self.norm2(self.fc_layer(feats) + feats)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        pe = torch.zeros(length, self.d_model, device=x.device)
        position = torch.arange(0, length, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class QueryDecoder(nn.Module):
    """transformers.py ClassConditionedTransformerDecoder with conditional_probs_embeddings=False."""

    def __init__(self, input_dim=512, hidden_dim=512, n_heads=8, n_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        queries = self.positional_encoding(queries)
        memory = self.embedding(memory) + self.positional_encoding(memory)
        out = self.transformer_decoder(queries.transpose(0, 1), memory.transpose(0, 1))
        return self.output_layer(out.transpose(0, 1))


class Model(nn.Module):
    def __init__(self, configs: Mapping[str, Any]):
        super().__init__()
        m = configs["model"] if isinstance(configs.get("model"), Mapping) else configs
        self.n_phases = int(m.get("n_phases", 7))
        self.input_dim = int(m.get("input_dim", 768))
        self.d_model = int(m.get("d_model", 512))
        self.present_length = int(m.get("present_length", 20))
        self.ctx_length = int(m.get("ctx_length", 1440))
        self.num_ctx_tokens = int(m.get("num_ctx_tokens", 12))
        self.num_ant_queries = int(m.get("num_ant_queries", 1))
        dec_dim = int(m.get("n_embd", 512))
        inf = dict(m.get("informer") or {})

        self.informer = Informer(
            enc_in=self.input_dim, dec_in=self.input_dim, c_out=self.d_model,
            seq_len=int(inf.get("seq_len", 96)), label_len=int(inf.get("label_len", 48)), out_len=1,
            factor=int(inf.get("factor", 5)), d_model=self.d_model, n_heads=int(inf.get("n_heads", 8)),
            e_layers=int(inf.get("e_layers", 2)), d_layers=int(inf.get("d_layers", 2)),
            d_ff=int(inf.get("d_ff", 1024)), dropout=float(inf.get("dropout", 0.1)), attn="full",
            embed="fixed", freq="h", activation="gelu", output_attention=False, distil=False, mix=True,
            use_decoder=True,
        )
        self.tokens_pooler = KeyRecorder(self.num_ctx_tokens, self.d_model, int(m.get("pooling_dim", 64)), 10)
        self.fusion_head1 = FusionHead(self.d_model)
        self.frame_decoder = QueryDecoder(
            input_dim=self.d_model, hidden_dim=dec_dim, n_heads=int(m.get("n_head", 8)),
            n_layers=int(m.get("n_layer", 2)),
        )
        self.curr_frames_classifier = nn.Linear(self.d_model, self.n_phases)
        self.regression_head = nn.Linear(dec_dim, self.n_phases + 1)
        self.input_queries = nn.Parameter(torch.randn(self.num_ant_queries, self.d_model))

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0, 0.01)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
        nn.init.xavier_uniform_(self.input_queries)

    def encoder(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.size(0)
        past = obs[:, : -self.present_length].contiguous().view(-1, self.present_length, self.input_dim)
        past = self.informer(past, past).contiguous().view(b, -1, self.d_model)
        present = obs[:, -self.present_length :]
        present = self.informer(present, present)
        return torch.cat((past, present), dim=1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """obs: (B, ctx_length, input_dim) features of frames [t-ctx_length, t), zero-padded at the front."""
        enc = self.encoder(obs)
        pooled = self.tokens_pooler(enc)[:, -self.num_ctx_tokens :]
        local = enc[:, -self.num_ctx_tokens :]
        mem = self.fusion_head1(pooled, local)
        queries = self.input_queries.expand(mem.size(0), -1, -1)
        next_action = self.frame_decoder(queries, mem)
        return {
            "curr_frames": self.curr_frames_classifier(mem),
            "remaining_time": self.regression_head(next_action),
        }
