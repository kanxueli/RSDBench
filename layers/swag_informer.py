"""Informer window encoder (Boels et al., SWAG)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from layers.informer.models.model import Informer


class InformerWindowEncoder(nn.Module):
    def __init__(
        self,
        enc_in: int = 768,
        d_model: int = 512,
        window: int = 20,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_decoder: bool = True,
    ):
        super().__init__()
        self.window = int(window)
        self.d_model = int(d_model)
        self.enc_in = int(enc_in)
        self.informer = Informer(
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=d_model,
            seq_len=window,
            label_len=window // 2,
            out_len=1,
            factor=5,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn="full",
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=False,
            mix=True,
            use_decoder=use_decoder,
        )
        # When use_decoder, Informer returns dec features of length = x_dec length,
        # then projects via self.informer.projection to c_out. That projection sits
        # on decoder path; encoder-only path returns d_model from encoder.
        # SKIT uses informer(past, past) with use_decoder=true → dec_out before
        # projection is d_model; wait — forward returns dec_out without projection!
        # Looking at model.py: returns dec_out (d_model), NOT projection(dec_out).
        # So output dim = d_model. Good.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, f = x.shape
        assert f == self.enc_in, f"Informer enc_in={self.enc_in}, got {f}"
        w = self.window
        pad = (w - (t % w)) % w
        if pad:
            x = torch.cat([x, x[:, -1:].expand(b, pad, f)], dim=1)
        n = x.size(1) // w
        chunks = x.reshape(b * n, w, f)
        # Informer expects (x_enc, x_dec); SWAG passes the same tensor twice.
        out = self.informer(chunks, chunks)  # (B*n, w, d_model)
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(b, n * w, -1)
        return out[:, :t]
