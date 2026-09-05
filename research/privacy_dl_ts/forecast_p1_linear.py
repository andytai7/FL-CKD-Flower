"""User-level (trajectory) DP feasibility on the LINEAR forecaster — the parameter-efficient
counterpart to the GRU user-level collapse.

The GRU arm's user-level failure was noise-to-signal geometry: σ·√d with σ≈2.0 at ε=1 over
d≈39.0k → the noised update norm exceeds the clip bound by ~400× (whole-update Gaussian
energy scales as σ²d while the clipped signal is bounded by ‖Δ‖≤C=1). `LinearForecaster`
(channel-shared in96→horizon map) cuts d to ~9.4k, halving √d. This module measures whether
that halving reopens trajectory-level DP at the band standard ε∈[1,4].

Same user-level protocol as forecast_p1.user_level_cell: 1 local epoch per clinic per round,
whole-update clip at C=1.0, per-coord noise σ·C, UNIFORM client weighting, accountant
sigma_for_epsilon(ε, rounds, 1e-5, q=1). Every row carries sigma_sqrt_d so the scaling law is
readable directly. Output: results/dl_ts_forecast_p1_linear.json (idempotent resume).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from flwr.serverapp.strategy import FedAvg

from dp import epsilon_rdp, sigma_for_epsilon
from messages import hushed, train_reply

from .forecast import IN_LEN, forecast_local, load_suite, split_clinic
from .forecast_p1 import _eval_row
from .models import LinearForecaster

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_forecast_p1_linear.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def user_level_linear_cell(*, epsilon: float | None, rounds: int, suite: str, horizon: int,
                           seed: int, clip: float = 1.0, quiet: bool = False) -> dict:
    clinics = load_suite(suite, horizon)
    splits = [split_clinic(c, horizon, suite) for c in clinics]
    splits = [s for s in splits if len(s["ytr"]) and len(s["yte"])]
    sigma = (sigma_for_epsilon(epsilon, rounds, 1e-5, sampling_probability=1.0)
             if epsilon is not None else 0.0)
    strategy = FedAvg(fraction_train=1.0)
    model = LinearForecaster(IN_LEN, horizon)
    torch.manual_seed(seed)
    vec = model.param_vector()
    d = vec.numel()
    history = []
    for rnd in range(1, rounds + 1):
        replies = []
        for k2, s in enumerate(splits):
            m = LinearForecaster(IN_LEN, horizon)
            m.load_param_vector(vec.clone())
            out = forecast_local(m, s["Xtr"], s["ytr"], epochs=1, lr=1e-3, batch=256,
                                 seed=seed * 17 + k2 * 13 + rnd)
            delta = out - vec
            norm = float(delta.norm())
            if norm > clip:
                delta = delta * (clip / norm)
            noise = torch.randn(delta.numel(), generator=torch.Generator().manual_seed(
                seed * 998_244_353 + k2 * 3_313 + rnd)) * (sigma * clip)
            replies.append(train_reply([(vec + delta + noise).numpy()], 1))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])
        m = LinearForecaster(IN_LEN, horizon)
        m.load_param_vector(vec.clone())
        row = _eval_row(m, splits, horizon, rnd, suite, "user", seed, epsilon)
        history.append(row)
        if not quiet:
            eps_txt = "off" if epsilon is None else f"{epsilon:g}"
            print(f"  usP1-LIN {suite}/h{horizon} eps={eps_txt} "
                  f"r{rnd}: MSE={row['mse']:.4f} (worst {row['mse_worst']:.4f})")
    return {"level": "user", "model": "linear", "suite": suite, "horizon": horizon,
            "target_epsilon": epsilon, "seed": seed, "client_sigma": sigma,
            "clip_norm": clip, "d_params": d, "sigma_sqrt_d": sigma * math.sqrt(d),
            "composed_epsilon": (epsilon_rdp(sigma, rounds, 1e-5) if epsilon is not None else None),
            "final_mse": history[-1]["mse"], "history": history}


def run_grid(*, epsilons=(None, 1.0, 4.0), rounds: int = 3, seeds=(42,),
             suites=("etth1", "weather"), horizon: int = 96, quiet: bool = False) -> list[dict]:
    rows = json.loads(RESULTS.read_text())["rows"] if RESULTS.exists() else []
    done = {(r["seed"], r["target_epsilon"], r["suite"], r["horizon"]) for r in rows}
    for suite in suites:
        for eps in epsilons:
            for seed in seeds:
                if (seed, eps, suite, horizon) in done:
                    continue
                r = user_level_linear_cell(epsilon=eps, rounds=rounds, suite=suite,
                                           horizon=horizon, seed=seed, quiet=quiet)
                rows.append(r)
                RESULTS.write_text(json.dumps(
                    {"arm": "user-level DP feasibility on LinearForecaster (d~9.3k vs GRU 39.0k)",
                     "protocol": "whole-update clip C=1, uniform client weighting, q=1, 3 rounds",
                     "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_grid()
    print(f"wrote {RESULTS}")
