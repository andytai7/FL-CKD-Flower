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
    """Expose TRAINABLE parameters as a single dense float32 vector (the privacy-machinery
    contract). Buffers (BatchNorm running stats) stay clinic-local: they are deterministic
    functions of local data, are not covered by the DP noise budget, and so never travel."""

    @staticmethod
    def _items(module: nn.Module):
        for name, tensor in module.named_parameters():
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
    """THE classification arm: optional 1-D conv stem → LSTM hidden 64 × 2 → mean+max-pool → logit.

    forward(x): x is (B, L, C) — seq clinics are single-lead (C=1).

    Gate history (2026-09-04, both measured on the seq clinics):
    1. v1 mean-only pooling failed the sanity gate (AUROC ≈0.53 through round 7) — burst-
       localised AF evidence was diluted by averaging over 1,024 steps.
    2. mean+max pooling + 3 local epochs still failed (≈0.51 through round 3): 1,024-step BPTT
       under federated averaging washes out per-round learning before ranking forms.
    3. So the arm ships with `frontend=True`: Conv1d(1→16, k15, s4)+BN → Conv1d(16→32, k9, s2)
       shrinks the recurrent part to L/8 ≈ 128 steps (the standard raw-single-lead-ECG stem).
       The RNN remains the sequence core — RNN pick stands; `frontend=False` reproduces v2 as
       the documented no-stem ablation row.
    """

    def __init__(self, channels: int = 1, hidden: int = 64, layers: int = 2, frontend: bool = True):
        super().__init__()
        self.frontend = (
            nn.Sequential(nn.Conv1d(channels, 16, 15, stride=4, padding=7),
                          nn.BatchNorm1d(16), nn.ReLU(),
                          nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU())
            if frontend else None
        )
        in_c = 32 if frontend else channels
        self.rnn = nn.LSTM(in_c, hidden, num_layers=layers, batch_first=True,
                           dropout=0.0 if layers == 1 else 0.1)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frontend is not None:
            x = self.frontend(x.transpose(1, 2)).transpose(1, 2)  # (B, L/8, 32)
        out, _ = self.rnn(x)          # (B, L', H)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)  # (B, 2H)
        return self.head(pooled).squeeze(-1)


class LSTMForecaster(ParamVectorMixin, nn.Module):
    """GRU-Fcst's LSTM cross-check twin: encoder LSTM over the window → LSTMCell zero-input
    decoder (identical interface + decoder contract; forward: (B, in_len, C) →
    (B, horizon, out_channels)). Parameter count ≈ GRU × (4/3)."""

    def __init__(self, channels: int, horizon: int, out_channels: int = 1,
                 hidden: int = 64, layers: int = 1):
        super().__init__()
        self.horizon, self.out_channels = horizon, out_channels
        self.encoder = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True)
        self.cell = nn.LSTMCell(hidden, hidden)
        self.head = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)            # h, c: (layers, B, H)
        hs, cs = h[-1], c[-1]                  # (B, H)
        outs = []
        for _ in range(self.horizon):
            hs, cs = self.cell(hs, (hs, cs))   # zero-input decoder step (hs doubles as input)
            outs.append(self.head(hs))         # (B, out_c)
        return torch.stack(outs, dim=1)        # (B, horizon, out_c)


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


class LinearForecaster(ParamVectorMixin, nn.Module):
    """Parameter-efficient linear reference (DLinear-lite, channel-shared): a single linear
    map from the flattened lookback window to the horizon, weights SHARED across channels
    (individual=False). d = in_len*horizon + horizon (~9.4k at in96/h96 vs the GRU's ~39.4k)
    — the arm that tests whether user-level (trajectory) DP becomes feasible on forecasting:
    whole-update noise energy scales as σ·√d, so halving √d is the entire game.

    Channel-shared map: x is (B, L, C); the reference target convention (forecast.py y
    windows) is channel 0, so the model applies the SAME map per channel and returns
    channel-0's horizon — matches the (B, horizon, out_channels=1) contract."""

    def __init__(self, in_len: int, horizon: int, out_channels: int = 1):
        super().__init__()
        self.in_len, self.horizon, self.out_channels = in_len, horizon, out_channels
        self.proj = nn.Linear(in_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x.permute(0, 2, 1))          # (B, C, horizon)
        return y[:, : self.out_channels, :].permute(0, 2, 1)  # (B, horizon, out_c)


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
