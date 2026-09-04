"""RNN-family models for the privacy × deep-learning time-series benchmark.

Track: `experiment/timeseries` (design contract: `research/privacy-dl-ts/README.md`).
Model pick (user-directed, 2026-09-04): the benchmark models are recurrent — LSTM for
classification, GRU for seq2seq forecasting; PatchTST ships as the OPTIONAL attention ablation
and is NOT part of the default matrix.

Length policy (recorded because it is a benchmark decision, not an implementation detail): the
mapper emits 4096-point z-normed traces (data/clinics_ecg_seq/). Classification arms consume them
at `downsample=4` → 1,024 timesteps: an anti-alias decimation that keeps the AF band (≤12 Hz
physiology versus a 75 Hz effective Nyquist) intact while cutting recurrent computation 4×. The
full 4096 view stays available (`downsample=1`) for the fidelity check row in the matrix.

All modules implement the flattening contract the privacy machinery needs: parameters are named
tensors accessible via `state_dict()`, and `param_vector()` / `load_param_vector()` present them
as ONE contiguous float32 vector, mirroring how `models/logreg.py` exposes `coef_`/`intercept_`
to the Flower strategies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ParamVectorMixin:
    """Expose parameters as a single dense float32 vector (the privacy-machinery contract)."""

    @staticmethod
    def _items(module: nn.Module):
        for name, tensor in module.state_dict().items():
            if tensor.dtype.is_floating_point:
                yield name, tensor.detach()

    def param_vector(self) -> torch.Tensor:
        parts = [t.reshape(-1) for _, t in self._items(self)]
        return torch.cat(parts) if parts else torch.zeros(0)

    def load_param_vector(self, vec: torch.Tensor) -> None:
        ptr = 0
        with torch.no_grad():
            for _, t in self._items(self):
                n = t.numel()
                setattr_vector = vec[ptr : ptr + n].view_as(t)
                t.copy_(setattr_vector.type_as(t))
                ptr += n
        assert ptr == vec.numel(), f"vector has {vec.numel()} scalars, model consumed {ptr}"

    def num_params(self) -> int:
        return sum(t.numel() for t in self.parameters())


class LSTMClassifier(ParamVectorMixin, nn.Module):
    """THE classification arm: LSTM hidden 64 × 2 layers, mean-pool over time → logit.

    forward(x): x is (B, L, C) — seq clinics are single-lead (C=1).
    """

    def __init__(self, channels: int = 1, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True,
                           dropout=0.0 if layers == 1 else 0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # (B, L, H)
        return self.head(out.mean(dim=1)).squeeze(-1)  # mean-pool over time -> (B,)


class GRUForecaster(ParamVectorMixin, nn.Module):
    """THE forecasting arm: GRU encoder → horizon decoder.

    forward(x): x is (B, in_len, C) → (B, horizon, out_channels). Teacher-forced decoding of a
    learned first-step state; the decoder receives the zero input each step (plain next-step
    regression), which is the honest no-covariate-future baseline the doc's matrix wants.
    """

    def __init__(self, channels: int, horizon: int, out_channels: int = 1,
                 hidden: int = 64, layers: int = 1):
        super().__init__()
        self.horizon, self.out_channels = horizon, out_channels
        self.encoder = nn.GRU(channels, hidden, num_layers=layers, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(x)                # h: (layers, B, H)
        state = h[-1]                         # (B, H)
        outs = []
        for _ in range(self.horizon):
            state = self.cell(state, state)   # zero-input decoder step
            outs.append(self.head(state))     # (B, out_c)
        return torch.stack(outs, dim=1)       # (B, horizon, out_c)


class PatchTST(ParamVectorMixin, nn.Module):
    """OPTIONAL attention ablation (Nie et al. 2023 small cfg): patch 16 / stride 8, channel-
    independent, d_model 128, 3 layers. Not a benchmark arm — kept to answer "would attention
    change this?" without leaving the repo."""

    def __init__(self, seq_len: int, channels: int = 1, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, n_heads: int = 8, depth: int = 3):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = 1 + (seq_len - patch_len) // stride
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(self.n_patches * d_model, 1)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C) -> (B*C, n_patches, patch): channel-independent
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, -1)
        return x.unfold(-1, self.patch_len, self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._patches(x)                     # (B*C, P, patch)
        z = self.encoder(self.embed(p) + self.pos)
        z = z.reshape(-1, self.n_patches * z.shape[-1])
        return self.head(z).squeeze(-1)


def decimate(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Anti-alias decimation: average non-overlapping blocks of `factor` samples (×→ L/f)."""
    if factor == 1:
        return x
    B, L, C = x.shape
    trimmed = L // factor * factor
    return x[:, :trimmed].reshape(B, trimmed // factor, factor, C).mean(dim=2)


def temperature_check() -> dict:
    """Param counts + forward-shape unit smoke (the carried-over check from the `dl` extra task)."""
    torch.manual_seed(42)
    cls = LSTMClassifier()
    fc = GRUForecaster(channels=7, horizon=192, out_channels=1)
    pt = PatchTST(seq_len=1024)
    xl = torch.randn(4, 1024, 1)
    xf = torch.randn(4, 96, 7)
    out = {"LSTMClassifier": {"params": cls.num_params(), "out": tuple(cls(xl).shape)},
           "GRUForecaster": {"params": fc.num_params(), "out": tuple(fc(xf).shape)},
           "PatchTST-ablation": {"params": pt.num_params(), "out": tuple(pt(xl).shape)}}
    # flattening round-trip
    v = cls.param_vector()
    clone = LSTMClassifier(); clone.load_param_vector(v + 1.0)
    assert torch.allclose(clone.param_vector(), v + 1.0), "param vector round-trip broken"
    out["vector_roundtrip"] = True
    return out


if __name__ == "__main__":
    import pprint

    pprint.pprint(temperature_check())
