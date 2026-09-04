"""P1 spine for the image track: record-level DP-SGD ε-grid on CNN-Small (DermaMNIST
melanoma) + per-cell loss-threshold MIA audit (TPR@FPR=1%).

Conventions: per-clinic standardised plans (σ_k so EVERY clinic composes the same target ε
over rounds·steps — the tabular-honest rule), clip-rate audit per round, FedProx μ=0.1
transport (the image track's measured default; explicit transport field on every row).
The ε=off cell measures the honest non-private control under this exact trainer (plain-SGD
steps with clipping at C — not the Adam sanity runner).

MIA audit: pooled train (members) vs pooled test (non-members) at up to 2000 records each,
loss-threshold attack via task BCE per-record losses; AUC + TPR@FPR=1% reported. Shadow-
model MIA is outside bench scope per the track notes.

Output: results/dl_image_p1.json rows with plans, audits, and dual-level round metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from flwr.serverapp.strategy import FedAvg

from dp import epsilon_rdp, sigma_for_epsilon
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics

from .dpsgd_image import dp_sgd_local, mia_loss_threshold
from .federated import _eval_probs, _split, load_image_clinics
from .models import CNNSmall

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p1.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def run_cell(*, epsilon: float | None, rounds: int, splits: list[dict], clinic_ns: list[int],
             seed: int, batch: int = 64, steps_epochs: float = 1.0, clip: float = 1.0,
             lr: float = 0.1, proximal_mu: float = 0.1, quiet: bool = False) -> dict:
    plans = []
    for n in clinic_ns:
        steps = max(1, math.ceil(steps_epochs * n / batch))
        q = min(1.0, batch / n)
        sigma = (sigma_for_epsilon(epsilon, rounds * steps, 1e-5, sampling_probability=q)
                 if epsilon is not None else 0.0)
        plans.append({"n": n, "steps": steps, "sigma": sigma, "q": q,
                      "composed_epsilon": epsilon_rdp(sigma, rounds * steps, 1e-5,
                                                      sampling_probability=q)
                      if epsilon is not None else None})
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      evaluate_metrics_aggr_fn=weighted_and_worst)
    model = CNNSmall()
    torch.manual_seed(seed)
    vec = model.param_vector()
    history = []
    for rnd in range(1, rounds + 1):
        replies = []
        clips = []
        for k, (s, plan) in enumerate(zip(splits, plans)):
            m = CNNSmall(); m.load_param_vector(vec.clone())
            out, audit = dp_sgd_local(m, s["Xtr"], s["ytr"], steps=plan["steps"],
                                      batch=batch, sigma=plan["sigma"], clip=clip, lr=lr,
                                      proximal_mu=proximal_mu, anchor=vec,
                                      seed=seed * 1_000_003 + k * 9_973 + rnd * 91_193)
            clips.append(audit["clip_rate_mean"])
            replies.append(train_reply([out.numpy()], len(s["ytr"])))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])
        eval_replies = []
        for k2, s in enumerate(splits):
            m = CNNSmall(); m.load_param_vector(vec.clone())
            eval_replies.append(evaluate_reply(compute_metrics(s["yte"], _eval_probs(m, s["Xte"])),
                                               len(s["yte"]), k2))
        with hushed():
            agg = strategy.aggregate_evaluate(rnd, eval_replies)
        row = dict(agg) if agg else {}
        row.update({"round": rnd, "clip_rate": float(np.mean(clips))})
        history.append(row)
        if not quiet:
            eps_txt = "off" if epsilon is None else f"{epsilon:g}"
            print(f"  imgP1 eps={eps_txt:<4} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                  f"(worst {row.get('auc_worst', float('nan')):.3f}) clip={row['clip_rate']:.2f}")
    # MIA audit on the final cell model (pooled, staples ≤2000 each)
    m = CNNSmall(); m.load_param_vector(vec.clone())
    rng = np.random.default_rng(seed * 733 + 17)
    X_m = np.concatenate([s["Xtr"] for s in splits]); y_m = np.concatenate([s["ytr"] for s in splits])
    X_n = np.concatenate([s["Xte"] for s in splits]); y_n = np.concatenate([s["yte"] for s in splits])
    cap_m = rng.permutation(len(y_m))[:2000]; cap_n = rng.permutation(len(y_n))[:2000]
    mia = mia_loss_threshold(m, X_m[cap_m], y_m[cap_m], X_n[cap_n], y_n[cap_n])
    final = history[-1]
    return {"target_epsilon": epsilon, "seed": seed, "transport": "fedprox",
            "proximal_mu": proximal_mu, "plans": plans, "mia": mia,
            "final_auc": float(final.get("auc", np.nan)),
            "final_auc_worst": float(final.get("auc_worst", np.nan)),
            "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
            "history": [{k: (round(float(v), 4) if isinstance(v, float) else v)
                         for k, v in h.items()} for h in history]}


def run_grid(*, epsilons=((None, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)), rounds: int = 10,
             seeds=(42,), quiet: bool = False) -> list[dict]:
    clinics = load_image_clinics()
    rows = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        ns = [len(s["ytr"]) for s in splits]
        for eps in epsilons:
            rows.append(run_cell(epsilon=eps, rounds=rounds, splits=splits, clinic_ns=ns,
                                 seed=seed, quiet=quiet))
            # checkpoint after each cell
            RESULTS.write_text(json.dumps({"arm": "P1 record-level DP-SGD (CNN-Small, DermaMNIST melanoma)",
                                           "mia": "loss-threshold attack, pooled members/non-members cap 2000",
                                           "rows": rows}, indent=2))
    return rows


def main() -> None:
    rows = run_grid(rounds=10, seeds=(42,))
    RESULTS.write_text(json.dumps({"arm": "P1 record-level DP-SGD (CNN-Small, DermaMNIST melanoma)",
                                   "mia": "loss-threshold attack, pooled members/non-members cap 2000",
                                   "rows": rows}, indent=2))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
