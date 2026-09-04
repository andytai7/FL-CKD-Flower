"""P3 for the image track: FedCT consensus on CNN-Small (DermaMNIST melanoma).

Mirror of research/privacy_dl_ts/fedct_ts.py with image-suite loaders and the CNNSmall
architecture. Public pool carves from clinic TRAIN folds (pool_frac default 0.10 capped
at q=512 queries — the official MedMNIST test split is pooled INSIDE the clinic folds by
the partition, so the only leak-free pool source is train-side donor data; the docstring
in fedct_ts records the same convention). Teachers train ISOLATED per clinic; consensus
voting is the only cross-clinic channel. Probe with `__main__`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from task import compute_metrics

from .federated import _eval_probs, _split, load_image_clinics, local_train
from .models import CNNSmall

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p3.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)
POOL_FRAC = 0.10
Q_CAP = 512


def train_teachers(teacher_splits: list[dict], *, seed: int, epochs: int = 6) -> list[CNNSmall]:
    """Isolated per-clinic teachers (weight-sharing teachers collapse votes to one model —
    the TS γ̂≡0 trap). Total local work = epochs; isolated is the convention."""
    teachers = []
    for k, s in enumerate(teacher_splits):
        t = CNNSmall()
        torch.manual_seed(seed * 10_003 + k)
        vec = local_train(t, s["Xtr"], s["ytr"], epochs=epochs,
                              seed=seed * 1_000_003 + k * 9_931)
        t.load_param_vector(vec)
        teachers.append(t)
    return teachers


def teacher_votes(teachers: list[CNNSmall], X_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    with torch.no_grad():
        for t in teachers:
            t.eval()
            probs.append(_eval_probs(t, X_pool))
    probs = np.stack(probs)
    return probs, (probs >= 0.5).astype(np.int8)


def consensus_labels(votes: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    k = votes.shape[0]
    noisy = votes.sum(axis=0) + rng.normal(0.0, sigma, size=votes.shape[1])
    return (noisy >= k / 2).astype(np.int64)


def measure_flip_rate(votes: np.ndarray) -> float:
    k, q = votes.shape
    base = votes.sum(axis=0) >= k / 2
    flips, total = 0, 0
    for i in range(k):
        loo = (votes.sum(axis=0) - votes[i]) >= (k - 1) / 2
        flips += int((loo != base).sum())
        total += q
    return flips / total


def query_sigma(epsilon: float, sensitivity: float, q: int, delta: float = 1e-5) -> float:
    eps_q = epsilon / q
    delta_q = delta / q
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_q)) / eps_q


def distill_student(pool_X: np.ndarray, labels: np.ndarray, *, seed: int,
                    epochs: int = 4, lr: float = 1e-3) -> CNNSmall:
    student = CNNSmall()
    out = local_train(student, pool_X, labels, epochs=epochs, lr=lr, batch=64,
                      seed=seed)  # consensus labels carry their own prevalence weighting
    student.load_param_vector(out)
    return student


def dual_metrics_row(student: CNNSmall, splits: list[dict]) -> dict:
    """Mirror of server_app.weighted_and_worst: NaN rows (single-class held-out fold,
    e.g. clinic 5's 8 all-negative rows) skip the weighting AND the worst and are counted."""
    metrics, ns, skipped = [], [], 0
    for s in splits:
        m = compute_metrics(s["yte"], _eval_probs(student, s["Xte"]))
        if np.isnan(m["auc"]) or np.isnan(m["sensitivity"]):
            skipped += 1
            continue
        metrics.append(m)
        ns.append(len(s["yte"]))
    total = sum(ns)
    row = {"eval_clinics_defined": len(metrics), "eval_clinics_skipped": skipped}
    for key in ("auc", "accuracy", "sensitivity"):
        row[key] = float(sum(m[key] * n for m, n in zip(metrics, ns)) / total)
        row[f"{key}_worst"] = float(min(m[key] for m in metrics))
    return row


def run_fedct(*, epsilon: float | None, q: int, seed: int, pool_frac: float = POOL_FRAC,
              quiet: bool = False) -> dict:
    clinics = load_image_clinics()
    teacher_splits, pool_frames = [], []
    for k, (X, y) in enumerate(clinics):
        s = _split(X, y, seed, k)
        cut = int(len(s["Xtr"]) * (1.0 - pool_frac))
        teacher_splits.append({"Xtr": s["Xtr"][:cut], "ytr": s["ytr"][:cut],
                               "Xte": s["Xte"], "yte": s["yte"]})
        pool_frames.append({"X": s["Xtr"][cut:], "y": s["ytr"][cut:]})
    pool_X = np.concatenate([p["X"] for p in pool_frames])
    pool_y = np.concatenate([p["y"] for p in pool_frames])
    rng = np.random.default_rng(seed * 77 + 3)
    take = rng.permutation(len(pool_y))[: min(q, len(pool_y))]
    pool_X, pool_y = pool_X[take], pool_y[take]

    teachers = train_teachers(teacher_splits, seed=seed)
    _, votes = teacher_votes(teachers, pool_X)
    gamma = measure_flip_rate(votes)
    clean = consensus_labels(votes, sigma=0.0, rng=rng)
    pool_rate = float(pool_y.mean())

    # Row builder (conservative √(2K) vs refined √(2Kγ̂) sensitivity at the requested ε)
    cells = []
    row_defs = [("conservative", math.sqrt(2.0 * len(teachers))),
                ("refined", math.sqrt(2.0 * len(teachers) * max(gamma, 1e-9)))]
    for sens_name, sens in row_defs:
        sig = query_sigma(epsilon, sens, len(pool_y)) if epsilon is not None else 0.0
        labels = clean if epsilon is None else consensus_labels(
            votes, sigma=sig, rng=np.random.default_rng(seed * 55 + 1))
        student = distill_student(pool_X, labels, seed=seed)
        row = dual_metrics_row(student, teacher_splits)
        row.update({"sensitivity": sens_name, "sens_value": sens, "sigma_votes": sig})
        cells.append(row)
    clean_auc = dual_metrics_row(distill_student(pool_X, clean, seed=seed), teacher_splits)
    tacc = [float((t_v == pool_y).mean()) for t_v in votes]
    out = {"epsilon": epsilon, "q": int(len(pool_y)), "pool_frac": pool_frac,
           "pool_prevalence": pool_rate, "gamma_hat": gamma,
           "teacher_acc_mean": float(np.mean(tacc)), "teacher_acc_min": float(np.min(tacc)),
           "majority_acc": float(((votes.sum(axis=0) >= len(teachers) / 2) == pool_y).mean()),
           "clean_student": clean_auc, "cells": cells, "seed": seed, "arm": "P3-fedct"}
    for c in cells:
        eps_txt = "off" if epsilon is None else f"{epsilon:g}"
        print(f"  imgP3 q={len(pool_y)} eps={eps_txt:<4} [{c['sensitivity']:>11}] "
              f"σ_v={c['sigma_votes']:>8.1f} -> AUROC {c['auc']:.3f} "
              f"(maj-clin acc {out['majority_acc']:.3f}, γ̂={gamma:.3f})")
    return out


def run_fedct_grid(*, epsilons=(0.5, 1.0, 2.0, 4.0, 8.0), q: int = Q_CAP,
                   seeds=(42,), quiet: bool = False) -> list[dict]:
    rows = [run_fedct(epsilon=None, q=q, seed=s, quiet=quiet) for s in seeds]
    rows += [run_fedct(epsilon=e, q=q, seed=s, quiet=quiet) for s in seeds for e in epsilons]
    RESULTS.write_text(json.dumps({"arm": "P3 FedCT consensus (CNN-Small, DermaMNIST)",
                                   "pool": "clinic-train carve 10% (official test is INSIDE folds on this dataset)",
                                   "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_fedct_grid()
    print(f"wrote {RESULTS}")
