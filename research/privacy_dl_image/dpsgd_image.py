"""P1 record-level DP-SGD local step for the image track + MIA audit primitives.

Same machinery pattern as research/privacy_dl_ts/dpsgd_ts on the TS branch — per-branch
copies are the agreed topology (methodologies live on the owning data branch; the repo root
`dpsgd.py` keeps the tabular logreg standard). dl-image specifics: image tensors (B,3,28,28)
float [0,1] batches flow through CNNSmall/ResNetLite-forward.

MIA: loss-threshold membership inference (Yeom-style), member = train record, non-member =
held-out test record; reports the attack AUC AND the TPR at FPR=1% fixed point (the
operating point the funders read; shadows are outside bench scope per the track notes) —
run per ε cell of the P1 grid as the attack-audit row the CLAUDE.md T2.5-style layer wants.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _per_record_grads(model: nn.Module, X: np.ndarray, y: np.ndarray,
                      idx: np.ndarray, loss_fn: nn.Module) -> torch.Tensor:
    """Honest per-record gradients (microbatch B=1) stacked as a (B, num_params) matrix."""
    model.train()
    params = list(model.parameters())
    offs = [0]
    for p in params:
        offs.append(offs[-1] + p.numel())
    grads = torch.empty(len(idx), offs[-1])
    for i, r in enumerate(idx):
        model.zero_grad(set_to_none=True)
        loss_fn(model(torch.from_numpy(X[r : r + 1])),
                torch.as_tensor(y[r : r + 1], dtype=torch.float32)).backward()
        grads[i] = torch.cat([p.grad.reshape(-1) for p in params])
    grads = torch.nan_to_num(grads)
    return grads


def dp_sgd_local(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, *, steps: int,
                 batch: int, sigma: float, clip: float, lr: float, seed: int,
                 proximal_mu: float = 0.0, anchor: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, dict]:
    """steps of honest DP-SGD on ONE clinic's (X, y): Poisson candidate batches, per-record
    clip C, Gaussian noise σC/|b| on the mean — returns (new flat vector, clip-rate audit).
    FedProx-compatible: optional proximal pull μ(θ−θ_anchor) folded into each update."""
    rng = np.random.default_rng(seed)
    n = len(y)
    params = list(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((n - max(1, y.sum())) / max(1, y.sum())))
    clip_rates = []
    for _ in range(steps):
        rows = rng.random(n) < min(1.0, batch / n)  # value-of-q
        idx = np.flatnonzero(rows)
        if len(idx) == 0:
            clip_rates.append(0.0)
            continue
        g = _per_record_grads(model, X, y, idx, loss_fn)
        norms = g.norm(dim=1)
        clip_rates.append(float((norms > clip).float().mean()))
        g = g * (clip / norms.clamp(min=clip)).unsqueeze(1)
        noisy = g.mean(dim=0) + torch.randn_like(g[0]) * (sigma * clip / len(idx))
        if proximal_mu and anchor is not None:
            flat = torch.cat([p.detach().reshape(-1) for p in params])
            noisy = noisy + proximal_mu * (flat - anchor)  # gradient of (μ/2)‖θ−θ_g‖²
        with torch.no_grad():
            ptr = 0
            for p, numel in [(p, p.numel()) for p in params]:
                p -= lr * noisy[ptr : ptr + numel].view_as(p)
                ptr += numel
    out = torch.cat([p.detach().reshape(-1) for p in params])
    return out, {"clip_rate_mean": float(np.mean(clip_rates)) if clip_rates else 0.0,
                 "steps_executed": steps}


def plan_grid_image(target_epsilon: float | None, *, rounds: int, steps_per_round: int,
                    batch: int, clinic_n: int, delta: float = 1e-5) -> dict:
    """Per-clinic standardized plan (identical convention to dpsgd.py: sigma so that THIS
    clinic composes the same target ε over rounds·steps)."""
    from dp import epsilon_rdp, sigma_for_epsilon

    if target_epsilon is None:
        return {"sigma": 0.0, "q": 1.0, "total_steps": rounds * steps_per_round,
                "composed_epsilon": None}
    q = min(1.0, batch / clinic_n)  # no amplification credit below batch size
    total = rounds * steps_per_round
    sigma = sigma_for_epsilon(target_epsilon, total, delta, sampling_probability=q)
    return {"sigma": sigma, "q": q, "total_steps": total,
            "composed_epsilon": epsilon_rdp(sigma, total, delta, sampling_probability=q)}


# ---- MIA audit ----

def per_record_loss(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                    batch: int = 256) -> np.ndarray:
    """Pointwise BCE (no reduction) per record — the attacker's observable."""
    model.eval()
    losses = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for lo in range(0, len(y), batch):
            xb = torch.from_numpy(X[lo : lo + batch])
            yb = torch.as_tensor(y[lo : lo + batch], dtype=torch.float32)
            losses.append(loss_fn(model(xb), yb).numpy())
    out = np.concatenate(losses) if losses else np.zeros(0)
    return out[:, 0] if out.ndim == 2 else out


def mia_loss_threshold(model: torch.nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
                       X_te: np.ndarray, y_te: np.ndarray) -> dict:
    """Loss-threshold MIA: score = −train/test loss; AUC + TPR at FPR=1% (audit row)."""
    member = per_record_loss(model, X_tr, y_tr)
    nonmember = per_record_loss(model, X_te, y_te)
    scores = np.concatenate([-member, -nonmember])
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    # attack AUC via rank statistic
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores)) + 1
    auc = float((ranks[: len(member)].sum() - len(member) * (len(member) + 1) / 2)
                / (len(member) * len(nonmember)))
    # TPR at FPR=1%: threshold at the 99th percentile of non-member scores
    thr = float(np.quantile(scores[len(member):], 0.99))
    tpr_at_1 = float(np.mean(scores[: len(member)] >= thr))
    return {"attack_auc": auc, "tpr_at_fpr1pct": tpr_at_1,
            "n_members": len(member), "n_nonmembers": len(nonmember),
            "tpr_ratio_vs_marginal": tpr_at_1 / 0.01}
