"""In-process federated runner for the dermoscopy benchmark — honest Flower aggregation.

Default transport: **FedProx μ=0.1** (the measured track default; FedAvg available as the
comparator arm). Aggregation is Flower's real `FedAvg.aggregate_train` over param vectors —
FedProx differs only in the local objective, so the same Flower strategy object serves both.
Per-clinic evaluation flows through the real `server_app.weighted_and_worst` (rule 5).
Local step: Adam on prevalence-weighted BCE + optional proximal pull toward the round's global
weights ((μ/2)·‖w−w_global‖²). The DP variants (P1 record-level DP-SGD) add clip/noise inside
the local step without touching this contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.serverapp.strategy import FedAvg

from data import load_clinic_frames, to_xy
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics

from .models import CNNSmall, to_images

CLINICS_DIR = Path(__file__).resolve().parents[2] / "data" / "clinics_dermamnist"


def load_image_clinics(clinics_dir: Path = CLINICS_DIR, *, size: int = 28) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-clinic (X, y) with X as (n, 3, S, S) in [0,1] — NOT z-standardised (image convention;
    the logreg track's per-client StandardScaler stays on the logreg path)."""
    out = []
    for df in load_clinic_frames(clinics_dir):
        Xf, y = to_xy(df)  # generic dispatch: label `melanoma`
        X = to_images(torch.from_numpy(Xf.astype("float32")), size).numpy()
        out.append((X, y))
    return out


def _split(X: np.ndarray, y: np.ndarray, seed: int, partition_id: int) -> dict:
    rng = np.random.default_rng(seed + partition_id)
    perm = rng.permutation(len(X))
    cut = max(1, min(len(X) - 1, int(round(0.8 * len(X)))))
    tr, te = perm[:cut], perm[cut:]
    return {"Xtr": X[tr], "ytr": y[tr], "Xte": X[te], "yte": y[te]}


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int = 1,
                lr: float = 1e-3, batch: int = 128, seed: int = 0,
                proximal_mu: float = 0.0, anchor: torch.Tensor | None = None) -> torch.Tensor:
    """One clinic's local step: Adam on prevalence-weighted BCE (+ FedProx proximal term)."""
    torch.manual_seed(seed)
    model.train()
    pos = max(1, int(y.sum()))
    pos_weight = torch.tensor((len(y) - pos) / pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())
    anchor_parts = None
    if proximal_mu and anchor is not None:
        flat, offs, shapes = [], [0], []
        for p in params:
            shapes.append(p.shape)
            offs.append(offs[-1] + p.numel())
        anchor_parts = [anchor[offs[i] : offs[i + 1]].view_as(p) for i, p in enumerate(params)]
    idx = np.arange(len(y))
    for _ in range(epochs):
        np.random.default_rng(seed).shuffle(idx)
        for lo in range(0, len(idx), batch):
            rows = idx[lo : lo + batch]
            xb = torch.from_numpy(X[rows])
            yb = torch.from_numpy(y[rows]).float()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if anchor_parts is not None:
                loss = loss + 0.5 * proximal_mu * sum(
                    ((p - a) ** 2).sum() for p, a in zip(params, anchor_parts)
                )
            loss.backward()
            opt.step()
    return model.param_vector()


@torch.no_grad()
def _eval_probs(model: nn.Module, X: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat(
        [torch.sigmoid(model(torch.from_numpy(X[lo : lo + batch]))) for lo in range(0, len(X), batch)]
    ).numpy()


def run_fedprox_smoke(*, rounds: int, seeds: tuple[int, ...] = (42,), epochs: int = 1,
                      proximal_mu: float = 0.1, quiet: bool = False,
                      model_factory=CNNSmall) -> list[dict]:
    """Weight-sharing run over the dermamnist clinics; returns per-round dual-level metrics."""
    clinics = load_image_clinics()
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      evaluate_metrics_aggr_fn=weighted_and_worst)
    history = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        model = model_factory()
        torch.manual_seed(seed)
        vec = model.param_vector()
        for rnd in range(1, rounds + 1):
            replies = []
            for k, s in enumerate(splits):
                m = model_factory(); m.load_param_vector(vec.clone())
                out_vec = local_train(m, s["Xtr"], s["ytr"], epochs=epochs,
                                      proximal_mu=proximal_mu, anchor=vec,
                                      seed=seed * 1_000_003 + k * 9_973 + rnd * 91_193)
                replies.append(train_reply([out_vec.numpy()], len(s["ytr"])))
            with hushed():
                arrays, _ = strategy.aggregate_train(rnd, replies)
            vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])

            eval_replies = []
            for k, s in enumerate(splits):
                m = model_factory(); m.load_param_vector(vec.clone())
                metrics = compute_metrics(s["yte"], _eval_probs(m, s["Xte"]))
                eval_replies.append(evaluate_reply(metrics, len(s["yte"]), k))
            with hushed():
                agg = strategy.aggregate_evaluate(rnd, eval_replies)
            row = dict(agg) if agg else {}
            row.update({"seed": seed, "round": rnd, "mu": proximal_mu})
            history.append(row)
            if not quiet:
                print(f"seed {seed} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                      f"(worst {row.get('auc_worst', float('nan')):.3f}) "
                      f"sens={row.get('sensitivity', float('nan')):.3f}")
    return history


if __name__ == "__main__":
    run_fedprox_smoke(rounds=10, seeds=(42,))
