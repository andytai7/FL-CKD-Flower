"""In-process federated runner for the TS benchmark research tree — honest Flower aggregation.

Mirrors dpsgd.py's discipline: parameter vectors leave the clinic as numpy arrays through the
real `flwr.serverapp.strategy.FedAvg.aggregate_train` on real `Message` objects; per-clinic
evaluation flows through the real `server_app.weighted_and_worst` (rule 5 dual-level metrics).
The local step is plain Adam on weighted BCE — the *sanity* path; DP-SGD variants (P1) add the
clip/noise inside the local step without touching this runner's aggregation contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.serverapp.strategy import FedAvg

from data.external.ecg_cinc2017_to_clinics import SEQ_LEN
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics

from .models import LSTMClassifier, decimate

SEQ_DIR = Path(__file__).resolve().parents[2] / "data" / "clinics_ecg_seq"


def load_seq_clinics(seq_dir: Path = SEQ_DIR, *, downsample: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-clinic (X, y) arrays; X float32 (n, L/downsample) after anti-alias decimation."""
    out = []
    for f in sorted(seq_dir.glob("clinic_*.npz")):
        z = np.load(f)
        X = torch.from_numpy(z["X"]).unsqueeze(-1)          # (n, L, 1)
        X = decimate(X, downsample).numpy()
        out.append((X, z["y"].astype("int64")))
    return out


def _split(X: np.ndarray, y: np.ndarray, seed: int, partition_id: int) -> dict:
    """Same reproducible 80/20 local split as client_app._local_split."""
    rng = np.random.default_rng(seed + partition_id)
    perm = rng.permutation(len(X))
    cut = max(1, min(len(X) - 1, int(round(0.8 * len(X)))))
    tr, te = perm[:cut], perm[cut:]
    return {"Xtr": X[tr], "ytr": y[tr], "Xte": X[te], "yte": y[te]}


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int = 1,
                lr: float = 1e-3, batch: int = 64, seed: int = 0, class_weight: bool = True,
                proximal_mu: float = 0.0, anchor: torch.Tensor | None = None) -> torch.Tensor:
    """One clinic's local step (sanity config): Adam on BCE — prevalence-weighted by default
    (the bench convention); distillation passes class_weight=False since consensus labels
    must be FIT, not rebalanced (+ optional FedProx proximal pull (μ/2)·‖θ−θ_global‖²)."""
    torch.manual_seed(seed)
    model.train()
    pos = max(1, int(y.sum()))
    pos_weight = torch.tensor((len(y) - pos) / pos, dtype=torch.float32) if class_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())
    anchor_parts = None
    if proximal_mu and anchor is not None:
        offs = [0]
        for p in params:
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
def _eval_probs(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    return torch.cat(
        [torch.sigmoid(model(torch.from_numpy(X[lo : lo + batch]))) for lo in range(0, len(X), batch)]
    ).numpy()


def run_fedavg_smoke(*, rounds: int, seeds: tuple[int, ...] = (42,), epochs: int = 1,
                     downsample: int = 4, proximal_mu: float = 0.0,
                     transport: str = "fedavg", momentum_beta: float = 0.6,
                     quiet: bool = False) -> list[dict]:
    """Weight-sharing LSTM over the ECG seq clinics. transport ∈ {fedavg, fedavgm (β),
    fedprox (proximal_mu)}; FedAvg/FedAvgM aggregate through Flower's strategy machinery."""
    clinics = load_seq_clinics(downsample=downsample)
    n_features = SEQ_LEN // downsample
    if transport == "fedavgm":
        from flwr.serverapp.strategy import FedAvgM  # flwr built-in (server_lr=1, β below)

        strategy = FedAvgM(
            server_momentum=momentum_beta, fraction_train=1.0, fraction_evaluate=1.0,
            evaluate_metrics_aggr_fn=weighted_and_worst)
    else:
        strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                          evaluate_metrics_aggr_fn=weighted_and_worst)
    history = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        model = LSTMClassifier()
        torch.manual_seed(seed)
        vec = model.param_vector()
        if transport == "fedavgm":  # built-in needs server weights primed before round 1
            from flwr.common import ArrayRecord

            strategy.current_arrays = ArrayRecord([vec.numpy()])
        for rnd in range(1, rounds + 1):
            replies = []
            for k, s in enumerate(splits):
                m = LSTMClassifier(); m.load_param_vector(vec.clone())
                out_vec = local_train(m, s["Xtr"], s["ytr"], epochs=epochs,
                                      proximal_mu=proximal_mu, anchor=vec,
                                      seed=seed * 1_000_003 + k * 9_973 + rnd * 91_193)
                replies.append(train_reply([out_vec.numpy()], len(s["ytr"])))
            with hushed():
                arrays, _ = strategy.aggregate_train(rnd, replies)
            vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])

            eval_replies = []
            for k, s in enumerate(splits):
                m = LSTMClassifier(); m.load_param_vector(vec.clone())
                metrics = compute_metrics(s["yte"], _eval_probs(m, s["Xte"]))
                eval_replies.append(evaluate_reply(metrics, len(s["yte"]), k))
            with hushed():
                agg = strategy.aggregate_evaluate(rnd, eval_replies)
            row = dict(agg) if agg else {}
            row.update({"seed": seed, "round": rnd, "seq_len": n_features})
            history.append(row)
            if not quiet:
                print(f"seed {seed} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                      f"(worst {row.get('auc_worst', float('nan')):.3f}) "
                      f"sens={row.get('sensitivity', float('nan')):.3f}")
    return history


if __name__ == "__main__":
    run_fedavg_smoke(rounds=10, seeds=(42,))
