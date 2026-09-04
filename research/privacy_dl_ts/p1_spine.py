"""P1 spine: record-level DP-SGD ε-grid on LSTM-CinC — the benchmark's reference arm.

Per-clinic standardised plans mirror dpsgd.py's orchestrator semantics: clinic k with N_k train
records runs ceil(N_k / B) Poisson steps per round at q_k = B / N_k; σ_k = sigma_for_epsilon(ε,
T=R·ceil(N_k/B), q_k) so EVERY clinic composes to the SAME target ε over the run (census-size-
aware — the rule that made the tabular standard honest). Grid cells differ only in (seed, ε):
locals and splits are per-(seed) fixed.

Output: results/dl_ts_p1.json rows carrying target ε, realised σ per clinic, dual-level metrics
per round (weighted + worst — rule 5), and the clip-rate audit, under seeds 42–46.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from flwr.serverapp.strategy import FedAvg

from dp import epsilon_rdp
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics

from .dpsgd_ts import dp_sgd_local
from .federated import _eval_probs, _split, load_seq_clinics
from .models import LSTMClassifier

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_p1.json"


def run_cell(*, epsilon: float | None, rounds: int, splits: list[dict], clinic_ns: list[int],
             seed: int, batch: int = 64, steps_epochs: float = 1.0, clip: float = 1.0,
             lr: float = 0.5, proximal_mu: float = 0.0, transport: str = "fedavgm",
             momentum_beta: float = 0.6, quiet: bool = False) -> dict:
    """One (ε, seed, transport) cell: R rounds of DP-SGD local steps aggregated by a real
    Flower strategy (FedAvgM default — the track's measured transport; fedavg and fedprox
    comparator arms for matrix Q3)."""
    plans = []
    for n in clinic_ns:
        steps = max(1, math.ceil(steps_epochs * n / batch))
        q = min(1.0, batch / n)  # small clinics train full-batch: no subsampling credit
        if epsilon is None:
            sigma = 0.0
        else:
            from dp import sigma_for_epsilon

            sigma = sigma_for_epsilon(epsilon, rounds * steps, 1e-5,
                                      sampling_probability=q)
        plans.append({"n": n, "steps": steps, "sigma": sigma,
                      "q": q,
                      "composed_epsilon": epsilon_rdp(sigma, rounds * steps, 1e-5,
                                                      sampling_probability=q)
                      if epsilon is not None else None})
    if transport == "fedavgm":
        from flwr.serverapp.strategy import FedAvgM  # flwr built-in (server_lr=1, β below)

        strategy = FedAvgM(server_momentum=momentum_beta, fraction_train=1.0,
                           fraction_evaluate=1.0,
                           evaluate_metrics_aggr_fn=weighted_and_worst)
    else:
        strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                          evaluate_metrics_aggr_fn=weighted_and_worst)
    model = LSTMClassifier()
    torch.manual_seed(seed)
    vec = model.param_vector()
    if transport == "fedavgm":  # built-in needs server weights primed before round 1
        from flwr.common import ArrayRecord

        strategy.current_arrays = ArrayRecord([vec.numpy()])
    history = []
    for rnd in range(1, rounds + 1):
        replies = []
        clip_rates = []
        for k, (s, plan) in enumerate(zip(splits, plans)):
            m = LSTMClassifier(); m.load_param_vector(vec.clone())
            new_flat, audit = dp_sgd_local(
                m, s["Xtr"], s["ytr"], steps=plan["steps"], batch=batch, sigma=plan["sigma"],
                clip=clip, lr=lr, seed=seed * 1_000_003 + k * 9_973 + rnd * 91_193,
                proximal_mu=proximal_mu, anchor=vec)
            clip_rates.append(audit["clip_rate_mean"])
            replies.append(train_reply([new_flat.numpy()], len(s["ytr"])))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])

        eval_replies = []
        for k, s in enumerate(splits):
            m = LSTMClassifier(); m.load_param_vector(vec.clone())
            eval_replies.append(evaluate_reply(compute_metrics(s["yte"], _eval_probs(m, s["Xte"])),
                                               len(s["yte"]), k))
        with hushed():
            agg = strategy.aggregate_evaluate(rnd, eval_replies)
        row = dict(agg) if agg else {}
        row.update({"round": rnd, "clip_rate": float(np.mean(clip_rates))})
        history.append(row)
        if not quiet:
            eps_txt = "off" if epsilon is None else f"{epsilon:g}"
            print(f"  eps={eps_txt:<5} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                  f"(worst {row.get('auc_worst', float('nan')):.3f}) clip={row['clip_rate']:.2f}")
    final = history[-1]
    return {"target_epsilon": epsilon, "seed": seed, "transport": transport,
            "proximal_mu": proximal_mu, "plans": plans,
            "final_auc": float(final.get("auc", np.nan)),
            "final_auc_worst": float(final.get("auc_worst", np.nan)),
            "final_sensitivity": float(final.get("sensitivity", np.nan)),
            "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
            "history": [{k: (round(float(v), 4) if isinstance(v, float) else v)
                         for k, v in h.items()} for h in history]}


def run_grid(*, epsilons=((None, 0.5, 1.0, 2.0, 4.0, 8.0)), rounds: int = 10, seeds=(42,),
             proximal_mu: float = 0.0, quiet: bool = False) -> list[dict]:
    clinics = load_seq_clinics(downsample=4)
    rows = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        ns = [len(s["ytr"]) for s in splits]
        for eps in epsilons:
            rows.append(run_cell(epsilon=eps, rounds=rounds, splits=splits, clinic_ns=ns,
                                 seed=seed, proximal_mu=proximal_mu, quiet=quiet))
    return rows


def main() -> None:
    rows = run_grid(rounds=10, seeds=(42,))
    RESULTS.write_text(json.dumps({"arm": "P1 record-level DP-SGD (LSTM, CinC seq clinics)",
                                   "grid": "eps x seed; per-clinic standardised (B=64) plans",
                                   "rows": rows}, indent=2))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
