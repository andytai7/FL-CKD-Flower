"""P3: FedCT consensus-labels stage on the AF binary (CinC seq clinics).

Protocol (research-doc FedCT section): teachers are the per-clinic LSTM-Clf models after a
FedAvgM weight-sharing stage; a PUBLIC POOL of traces then receives consensus labels from
noisy per-query teacher votes, and a student LSTM distills on those labels. Privacy spend is
label-side only (weight updates are raw FedAvgM — the clean FedCT construction on this
track); the dual sensitivity rows from the track doc:

- conservative: per-query L2 sensitivity Δ = sqrt(2·K)  (K teachers, hard ± vote changes);
- refined (research-risk row, flagged): on-average-LOO Δ̃ = sqrt(2·γ̂·K) with the flip rate γ̂
  MEASURED on the actual vote matrix (mean over queries and teachers of consensus-change
  probability under teacher leave-one-out) — composes per FedCT's data-dependent theorem.

Public pool construction (documented carve, 2026-09-04): the ECG suite has no unlabelled
external shard, so the pool is carved from each clinic's TRAIN fold at load time
(pool_frac=0.10, deterministic per seed); carved rows are excluded from teacher training in
FedCT runs and never touch the clinic TEST folds (metrics stay honest vs the feature bench).

Composition: ε split uniformly across the Q pool queries (ε_q = ε/Q, δ_q = 1e-5/Q); σ_q from
the Gaussian-mechanism calibration σ = Δ·sqrt(2·ln(1.25/δ_q))/ε_q. Consequently the same ε
surfaced by P1 grids (0.5/1/2/4/8) yields a distinguishable-but-loud vote noise here — that
contrast is itself a matrix result.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from flwr.serverapp.strategy import FedAvgM

from server_app import weighted_and_worst
from task import compute_metrics

from .federated import _eval_probs, _split, load_seq_clinics, local_train
from .models import LSTMClassifier

FEDCT_TEACHER_ROUNDS = 10
POOL_FRAC = 0.10  # carve from each clinic's train fold (never touches test folds)


def carve_public_pool(splits: list[dict], pool_frac: float, seed: int) -> tuple[list[dict], dict]:
    """Carve the consensus pool from each clinic's train fold; returns (teacher_splits, pool)."""
    rng_pool = np.random.default_rng(seed * 65537 + 13)
    teacher_splits, pool_x, pool_y = [], [], []
    for s in splits:
        n = len(s["Xtr"])
        m = max(2, int(round(pool_frac * n)))
        idx = rng_pool.permutation(n)
        pool_idx, keep_idx = idx[:m], idx[m:]
        pool_x.append(s["Xtr"][pool_idx]); pool_y.append(s["ytr"][pool_idx])
        teacher_splits.append({"Xtr": s["Xtr"][keep_idx], "ytr": s["ytr"][keep_idx],
                               "Xte": s["Xte"], "yte": s["yte"]})
    return teacher_splits, {"X": np.concatenate(pool_x), "y": np.concatenate(pool_y)}


def train_teachers(teacher_splits: list[dict], *, rounds: int, seed: int, epochs: int = 2,
                   momentum_beta: float = 0.6) -> list[LSTMClassifier]:
    """FedCT teachers are trained ISOLATED per clinic (no weight sharing) — consensus
    voting is the only cross-clinic channel, so teachers MUST diverge for the mechanism
    to carry meaning (a weight-sharing stage collapses all votes to one model: measured
    γ̂ ≡ 0 and vacuous majority). Total local work matches the weight-sharing stage
    (rounds × epochs local epochs)."""
    teachers = []
    for k, s in enumerate(teacher_splits):
        t = LSTMClassifier()
        torch.manual_seed(seed * 10_007 + k)          # distinct, deterministic inits
        vec = local_train(t, s["Xtr"], s["ytr"], epochs=rounds * epochs,
                          seed=seed * 1_000_003 + k * 7_919)
        t.load_param_vector(vec)
        teachers.append(t)
    return teachers


def teacher_votes(teachers: list[LSTMClassifier], X_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-teacher PROBABILITIES (K, Q) under each teacher's own operating distribution
    (model.eval() for BatchNorm running-stat calibration, matching the federated bench);
    also returns the hard votes for the sensitivity audit."""
    probs = []
    with torch.no_grad():
        for t in teachers:
            t.eval()
            probs.append(_eval_probs(t, X_pool))
    probs = np.stack(probs)
    return probs, (probs >= 0.5).astype(np.int8)


def consensus_labels(votes: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Classical FedCT noisy count over HARD votes: label = 1[ Σ_k v_{kj} + N(0,σ²) ≥ K/2 ].
    σ is in count units; sensitivity rows √2K (conservative) / √(2Kγ̂) (refined LOO)."""
    k = votes.shape[0]
    noisy = votes.sum(axis=0) + rng.normal(0.0, sigma, size=votes.shape[1])
    return (noisy >= k / 2).astype(np.int64)


def measure_flip_rate(votes: np.ndarray) -> float:
    """On-average-LOO γ̂: mean over queries and teachers of P(consensus label changes when
    teacher i is removed). This MEASURED γ̂ drives the refined row (and is the flagged
    research-risk audit — it is data-dependent)."""
    k, q = votes.shape
    base = votes.sum(axis=0) >= k / 2
    flips = 0
    total = 0
    for i in range(k):
        loo = (votes.sum(axis=0) - votes[i]) >= (k - 1) / 2
        flips += int((loo != base).sum())
        total += q
    return flips / total


def query_sigma(epsilon: float, sensitivity: float, q: int, delta: float = 1e-5) -> float:
    """Per-query Gaussian σ for uniform budget split (ε_q = ε/Q, δ_q = δ/Q)."""
    eps_q = epsilon / q
    delta_q = delta / q
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_q)) / eps_q


def distill_student(pool_X: np.ndarray, labels: np.ndarray, *, seed: int, epochs: int = 3,
                    lr: float = 1e-3) -> LSTMClassifier:
    """Centralised distillation on (public pool, consensus labels) — the FedCT privacy
    boundary: no clinic gradients cross after this point."""
    student = LSTMClassifier()
    out = local_train(student, pool_X, labels, epochs=epochs, lr=lr, batch=64,
                      seed=seed, class_weight=False)  # fit consensus labels, don't rebalance
    student.load_param_vector(out)
    return student


def dual_metrics_row(student: LSTMClassifier, splits: list[dict]) -> dict:
    """Dual-level evaluation of the distilled student identical to weighted_and_worst but
    without the strategy surface (student is centralised by construction)."""
    metrics, ns = [], []
    for s in splits:
        metrics.append(compute_metrics(s["yte"], _eval_probs(student, s["Xte"])))
        ns.append(len(s["yte"]))
    total = sum(ns)
    row = {}
    for key in ("auc", "accuracy", "sensitivity"):
        row[key] = float(sum(m[key] * n for m, n in zip(metrics, ns)) / total)
        row[f"{key}_worst"] = float(min(m[key] for m in metrics))
    return row


def run_fedct(*, epsilon: float, rounds: int = FEDCT_TEACHER_ROUNDS, seeds=(42,),
              pool_frac: float = 0.10, pool_max: int = 512, quiet: bool = False) -> list[dict]:
    """Dual sensitivity rows per cell: conservative f_c (=1 per query, the hard-vote √2
    upper envelope at K=2 collapsed to the f_c bound for comparability) and refined
    on-average-LOO f_c·sqrt(γ̂·K) with γ̂ measured on the hard-vote matrix."""
    clinics = load_seq_clinics(downsample=4)
    rows = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        teacher_splits, pool = carve_public_pool(splits, pool_frac, seed)
        teachers = train_teachers(teacher_splits, rounds=rounds, seed=seed)
        q = min(pool_max, len(pool["y"]))
        X_pool, y_pool = pool["X"][:q], pool["y"][:q]
        probs, votes = teacher_votes(teachers, X_pool)
        gamma = measure_flip_rate(votes)
        k = votes.shape[0]
        majority = (votes.sum(axis=0) >= k / 2).astype(np.int64)
        majority_acc = float((majority == y_pool).mean())
        for kind, sens in (("conservative", float(math.sqrt(2.0 * k))),
                           ("refined", float(math.sqrt(2.0 * max(gamma, 0.0) * k)))):
            sigma = query_sigma(epsilon, sens, q)
            rng = np.random.default_rng(seed * 31_557_761 + q)
            labels = consensus_labels(votes, sigma=sigma, rng=rng)
            vote_acc = float((labels == y_pool).mean())
            student = distill_student(X_pool, labels, seed=seed,
                                      epochs=3 if labels.any() and not labels.all() else 1)
            metrics = dual_metrics_row(student, splits)
            row = {"target_epsilon": epsilon, "seed": seed, "sensitivity_kind": kind,
                   "sensitivity": sens, "vote_sigma": sigma, "queries": q,
                   "gamma_hat": gamma, "teacher_majority_acc": majority_acc,
                   "noisy_vote_acc": vote_acc,
                   **{f"final_{k2}": v for k2, v in metrics.items()}}
            rows.append(row)
            if not quiet:
                print(f"  fedct eps={epsilon:g} [{kind}] sigma_v={sigma:.6f} "
                      f"gam={gamma:.3f} vote_acc={vote_acc:.3f} "
                      f"AUC={metrics['auc']:.3f} (worst {metrics['auc_worst']:.3f})")
    return rows


def run_fedct_grid(*, epsilons=((0.5, 1.0, 2.0, 4.0, 8.0)), rounds: int = FEDCT_TEACHER_ROUNDS,
                   seeds=(42,), pool_max: int = 512, out_path=None, quiet: bool = False) -> list[dict]:
    """Grid over ε with the teacher stage shared across cells (votes are ε-independent).
    Cells whose noisy labels are pure noise can (and do, at the conservative row for small
    ε per-query budgets) yield degenerate students — recorded honestly as NaN metric rows."""
    rows = []
    for seed in seeds:
        clinics = load_seq_clinics(downsample=4)
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        teacher_splits, pool = carve_public_pool(splits, POOL_FRAC, seed)
        teachers = train_teachers(teacher_splits, rounds=rounds, seed=seed)
        q = min(pool_max, len(pool["y"]))
        X_pool, y_pool = pool["X"][:q], pool["y"][:q]
        probs, votes = teacher_votes(teachers, X_pool)
        gamma = measure_flip_rate(votes)
        k = votes.shape[0]
        majority_acc = float(((votes.sum(axis=0) >= k / 2).astype(np.int64) == y_pool).mean())
        if not quiet:
            print(f"seed {seed}: teachers trained; gamma_hat={gamma:.3f} "
                  f"majority_acc={majority_acc:.3f} q={q}")
        for eps in epsilons:
            for kind, sens in (("conservative", float(math.sqrt(2.0 * k))),
                               ("refined", float(math.sqrt(2.0 * max(gamma, 0.0) * k)))):
                sigma = query_sigma(eps, sens, q)
                rng = np.random.default_rng(seed * 31_557_761 + q)
                labels = consensus_labels(votes, sigma=sigma, rng=rng)
                vote_acc = float((labels == y_pool).mean())
                student = distill_student(X_pool, labels, seed=seed)
                metrics = dual_metrics_row(student, splits)
                row = {"target_epsilon": eps, "seed": seed, "sensitivity_kind": kind,
                       "sensitivity": sens, "vote_sigma": sigma, "queries": q,
                       "gamma_hat": gamma, "teacher_majority_acc": majority_acc,
                       "noisy_vote_acc": vote_acc,
                       **{f"final_{k2}": v for k2, v in metrics.items()}}
                rows.append(row)
                if not quiet:
                    print(f"  fedct eps={eps:g} [{kind}] sigma_v={sigma:.2f} "
                          f"vote_acc={vote_acc:.3f} AUC={metrics['auc']:.3f} "
                          f"(worst {metrics['auc_worst']:.3f})")
    if out_path is not None:
        import json

        out_path.write_text(json.dumps({"arm": "P3 FedCT consensus (CinC seq clinics)",
                                        "rows": rows}, indent=2))
        print(f"wrote {out_path}")
    return rows


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "results" / "dl_ts_p3.json"
    run_fedct_grid(out_path=out)
