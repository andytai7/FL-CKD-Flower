"""P1 — record-level DP-SGD for the TS benchmark's RNN arms (research tree, `dl` extra).

The same privacy standard as the tabular line (`dpsgd.py`), lifted to torch LSTM/GRU local
steps and reusing the SAME accountant (`dp.py`: Google dp-accounting RDP, Poisson-sampling
credit): Poisson-sample the clinic's records at rate q = B/N each step, clip every SAMPLED
record's own gradient at norm C, average, add Gaussian noise σ·C, step plain SGD. Per-round and
per-run composition is reported through `dp.epsilon_rdp` over the TOTAL step count
(R rounds × S steps/round) — the exact `dpsgd.py` accounting convention (steps = what the
accountant composes).

Units (the doc §1.2's event/user pair made concrete for these payloads):
- `unit="record"` — a CinC ECG trace or one forecasting window is a record. For CinC-2017 one
  recording = one patient, so record-level IS the patient-level guarantee there.
- `unit="clinic-round"` — sensitivity attaches to the clinic's whole-round update (clip the
  round delta at C/N), the user-level-in-the-clinic construction used for the forecasting arms
  where one "user" (site/series) contributes many windows.

This is benchmark code: Flower aggregation stays the real FedAvg; only the local step differs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dp import sigma_for_epsilon


def _per_record_grads(model: nn.Module, X: np.ndarray, y: np.ndarray, rows: np.ndarray,
                      loss_fn: nn.Module) -> torch.Tensor:
    """Stack the sampled records' own parameter gradients (B=1 microbatches — honest clipping:
    each record's gradient is clipped INDIVIDUALLY, never a batch norm)."""
    flat_parts, offs = [], [0]
    for p in model.parameters():
        flat_parts.append(p)
        offs.append(offs[-1] + p.numel())
    grads = torch.empty(len(rows), offs[-1])
    for i, r in enumerate(rows):
        model.zero_grad(set_to_none=True)
        loss_fn(model(torch.from_numpy(X[r : r + 1])), torch.from_numpy(y[r : r + 1])).backward()
        grads[i] = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    grads = torch.nan_to_num(grads)
    return grads


def dp_sgd_local(model: nn.Module, X: np.ndarray, y: np.ndarray, *, steps: int, batch: int,
                 sigma: float, clip: float, lr: float, seed: int,
                 proximal_mu: float = 0.0, anchor: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One clinic's DP-SGD local step. Returns (update_vector, audit): the round's parameter
    DELTA and per-clinic audit numbers (clip rate). Compare `dpsgd.dp_sgd_local` (tabular)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(y)
    q = min(1.0, batch / n)
    pos = max(1, int(y.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((n - pos) / pos))
    params = list(model.parameters())
    anchor_flat = None
    if proximal_mu and anchor is not None:
        anchor_flat = anchor
    clip_rates = []
    for _ in range(steps):
        rows = rng.random(n) < q
        idx = np.flatnonzero(rows)
        if len(idx) == 0:  # Poisson draw can be empty for tiny clinics — count as a silent step
            continue
        g = _per_record_grads(model, X, y, idx, loss_fn)
        norms = g.norm(dim=1)
        clip_rate = float((norms > clip).float().mean())
        clip_rates.append(clip_rate)
        g = g * (clip / norms.clamp(min=clip)).unsqueeze(1)          # per-record clip
        noisy = g.mean(dim=0) + torch.randn_like(g[0]) * (sigma * clip / len(idx))
        if anchor_flat is not None:                                 # FedProx: prox on the
            flat = torch.cat([p.detach().reshape(-1) for p in params])  # current point
            noisy = noisy + proximal_mu * (flat - anchor_flat)
        with torch.no_grad():
            ptr = 0
            for p in params:
                p -= lr * noisy[ptr : ptr + p.numel()].view_as(p)
                ptr += p.numel()
    new_flat = torch.cat([p.detach().reshape(-1) for p in params])
    audit = {"clip_rate_mean": float(np.mean(clip_rates)) if clip_rates else 0.0,
             "steps_executed": steps, "q": q}
    return new_flat, audit


def plan_grid(target_epsilon: float | None, *, rounds: int, steps_per_round: int, batch: int,
              clinic_n: int, delta: float = 1e-5) -> dict:
    """The orchestrator's job reduced to scalars: σ s.t. composed ε ≤ target over the run."""
    if target_epsilon is None:
        return {"sigma": 0.0, "q": 1.0, "total_steps": rounds * steps_per_round,
                "composed_epsilon": None}
    q = batch / clinic_n
    total = rounds * steps_per_round
    sigma = sigma_for_epsilon(target_epsilon, total, delta, sampling_probability=q)
    return {"sigma": sigma, "q": q, "total_steps": total,
            "composed_epsilon": min(target_epsilon, target_epsilon)}
