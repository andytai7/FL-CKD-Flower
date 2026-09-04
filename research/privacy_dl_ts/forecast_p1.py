"""P1 variants on the forecasting arms: event-level AND user-level DP grids.

Scopes (pilot scale, exact conventions recorded per row):
- event-level: per-WINDOW record DP-SGD on GRU-Fcst (MSE loss): Poisson windows, per-window
  clip C, Gaussian σC — handled by the same honest per-record machinery as classification,
  with steps scaled down (steps_epochs=0.25) and 5 rounds because decoder rollouts are heavy
  on CPU. Suite: weather/h96 (richest channels); eps {off,0.5,1,2,4,8}.
- user-level (conservative per-clinic trajectory DP): local Adam epoch on the clinic's OWN
  data, update Δ clipped to ‖Δ‖≤C_u=1 (parameter space), Gaussian σ_u·C_u added per round
  per clinic, server FedAvg of the clipped noised updates; accounting = RDP of the client-
  level mechanism over R rounds at q=1 (all clinics participate — the trajectory unit is
  the clinic's whole month-slice, which matches the user-level trajectory definition in
  the track doc). Suites: all three at h=96; eps {off,0.5,1,2,4,8}.

Metrics per cell: dual-level MSE/MAE + ACF fidelity (the temporal-artifact row).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.serverapp.strategy import FedAvg

from dp import epsilon_rdp, sigma_for_epsilon
from messages import hushed, train_reply

from .forecast import (acf_fidelity, forecast_local, load_suite, predict, split_clinic,
                       _mse_mae)
from .models import GRUForecaster

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_forecast_p1.json"


def per_window_grads(model: nn.Module, X: np.ndarray, y: np.ndarray,
                     idx: np.ndarray) -> torch.Tensor:
    """Honest per-window MSE gradients stacked (B, num_params)."""
    model.train()
    params = list(model.parameters())
    offs = [0]
    for p in params:
        offs.append(offs[-1] + p.numel())
    grads = torch.empty(len(idx), offs[-1])
    lf = nn.MSELoss()
    for i, r in enumerate(idx):
        model.zero_grad(set_to_none=True)
        lf(model(torch.from_numpy(X[r : r + 1])), torch.from_numpy(y[r : r + 1])).backward()
        grads[i] = torch.cat([p.grad.reshape(-1) for p in params])
    return torch.nan_to_num(grads)


def event_level_cell(*, epsilon: float | None, rounds: int, splits: list[dict],
                     clinic_ns: list[int], suite: str, horizon: int, seed: int,
                     batch: int = 32, steps_epochs: float = 0.25, clip: float = 1.0,
                     lr: float = 0.1, quiet: bool = False) -> dict:
    n_ch = splits[0]["Xtr"].shape[2]
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
    strategy = FedAvg(fraction_train=1.0)
    model = GRUForecaster(n_ch, horizon)
    torch.manual_seed(seed)
    vec = model.param_vector()
    history = []
    for rnd in range(1, rounds + 1):
        replies = []
        for k, (s, plan) in enumerate(zip(splits, plans)):
            m = GRUForecaster(n_ch, horizon); m.load_param_vector(vec.clone())
            rng = np.random.default_rng(seed * 1_000_003 + k * 9_973 + rnd * 91_193)
            n = len(s["ytr"])
            for _ in range(plan["steps"]):
                idx = np.flatnonzero(rng.random(n) < plan["q"])
                if len(idx) == 0:
                    continue
                g = per_window_grads(m, s["Xtr"], s["ytr"], idx)
                norms = g.norm(dim=1)
                g = g * (clip / norms.clamp(min=clip)).unsqueeze(1)
                noisy = g.mean(dim=0) + torch.randn_like(g[0]) * (plan["sigma"] * clip / len(idx))
                with torch.no_grad():
                    ptr = 0
                    for p in m.parameters():
                        p -= lr * noisy[ptr : ptr + p.numel()].view_as(p)
                        ptr += p.numel()
            replies.append(train_reply([m.param_vector().numpy()], n))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])
        m = GRUForecaster(n_ch, horizon); m.load_param_vector(vec.clone())
        row = _eval_row(m, splits, horizon, rnd, suite, f"event", seed, epsilon)
        history.append(row)
        if not quiet:
            print(f"  evP1 {suite}/h{horizon} eps={'off' if epsilon is None else epsilon:g} "
                  f"r{rnd}: MSE={row['mse']:.4f} (worst {row['mse_worst']:.4f}) "
                  f"acfcorr={row['acf_corr']:.3f}")
    return {"level": "event", "suite": suite, "horizon": horizon, "target_epsilon": epsilon,
            "seed": seed, "steps_epochs": steps_epochs, "plans": plans,
            "final_mse": history[-1]["mse"], "history": history}


def user_level_cell(*, epsilon: float | None, rounds: int, splits: list[dict],
                    clinic_ns: list[int], suite: str, horizon: int, seed: int,
                    clip: float = 1.0, quiet: bool = False) -> dict:
    n_ch = splits[0]["Xtr"].shape[2]
    k = len(splits)
    sigma = (sigma_for_epsilon(epsilon, rounds, 1e-5, sampling_probability=1.0)
             if epsilon is not None else 0.0)
    strategy = FedAvg(fraction_train=1.0)
    model = GRUForecaster(n_ch, horizon); torch.manual_seed(seed)
    vec = model.param_vector()
    history = []
    for rnd in range(1, rounds + 1):
        replies = []
        for k2, s in enumerate(splits):
            m = GRUForecaster(n_ch, horizon); m.load_param_vector(vec.clone())
            out = forecast_local(m, s["Xtr"], s["ytr"], epochs=1, lr=1e-3, batch=64,
                                 seed=seed * 17 + k2 * 13 + rnd)
            delta = out - vec
            norm = float(delta.norm())
            if norm > clip:
                delta = delta * (clip / norm)
            noise = torch.randn(delta.numel(), generator=torch.Generator().manual_seed(
                seed * 998_244_353 + k2 * 3_313 + rnd)) * (sigma * clip)
            # user-level DP: UNIFORM client weighting (weight 1) — n_k weighting would dilute
            # small clinics' noise below their claimed budget
            replies.append(train_reply([(vec + delta + noise).numpy()], 1))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])
        m = GRUForecaster(n_ch, horizon); m.load_param_vector(vec.clone())
        row = _eval_row(m, splits, horizon, rnd, suite, "user", seed, epsilon)
        history.append(row)
        if not quiet:
            print(f"  usP1 {suite}/h{horizon} eps={'off' if epsilon is None else epsilon:g} "
                  f"r{rnd}: MSE={row['mse']:.4f} (worst {row['mse_worst']:.4f})")
    return {"level": "user", "suite": suite, "horizon": horizon, "target_epsilon": epsilon,
            "seed": seed, "client_sigma": sigma, "clip_norm": clip,
            "composed_epsilon": (epsilon_rdp(sigma, rounds, 1e-5) if epsilon is not None else None),
            "final_mse": history[-1]["mse"], "history": history}


def _eval_row(model: nn.Module, splits: list[dict], horizon: int, rnd: int, suite: str,
              level: str, seed: int, epsilon: float | None) -> dict:
    mses, maes, ns_ = [], [], []
    for s in splits:
        pred = predict(model, s["Xte"])
        mm = _mse_mae(pred, s["yte"])
        mses.append(mm["mse"]); maes.append(mm["mae"]); ns_.append(len(s["yte"]))
        s["pred_series"] = pred[:, :, 0].reshape(-1)
    tot = sum(ns_)
    row = {"round": rnd, "level": level, "seed": seed, "suite": suite, "horizon": horizon,
           "target_epsilon": epsilon,
           "mse": float(sum(a * n for a, n in zip(mses, ns_)) / tot),
           "mae": float(sum(a * n for a, n in zip(maes, ns_)) / tot),
           "mse_worst": float(max(mses))}
    lev_true = np.concatenate([s["yte"][:, :, 0].mean(axis=1) for s in splits])
    lev_pred = np.concatenate([s["pred_series"].reshape(len(s["yte"]), horizon).mean(axis=1)
                               for s in splits])
    row.update(acf_fidelity(lev_pred.astype(np.float64), lev_true.astype(np.float64)))
    return row


def run_forecasting_p1(*, seeds=(42,), quiet: bool = False) -> list[dict]:
    rows = []
    # event-level pilot: weather/h96 only
    clinics = load_suite("weather", 96)
    for seed in seeds:
        splits = [split_clinic(c, 96, "weather") for c in clinics]
        splits = [s for s in splits if len(s["ytr"]) and len(s["yte"])]
        ns = [len(s["ytr"]) for s in splits]
        for eps in (None, 0.5, 1.0, 2.0, 4.0, 8.0):
            rows.append(event_level_cell(epsilon=eps, rounds=5, splits=splits, clinic_ns=ns,
                                         suite="weather", horizon=96, seed=seed, quiet=quiet))
            RESULTS.write_text(json.dumps({"arm": "P1 forecasting: event-level (weather/h96 pilot) + user-level (3 suites)",
                                           "rows": rows}, indent=2))
    # user-level: all three suites at h=96
    for suite in ("etth1", "ettm1", "weather"):
        clinics = load_suite(suite, 96)
        for seed in seeds:
            splits = [split_clinic(c, 96, suite) for c in clinics]
            splits = [s for s in splits if len(s["ytr"]) and len(s["yte"])]
            ns = [len(s["ytr"]) for s in splits]
            for eps in (None, 0.5, 1.0, 2.0, 4.0, 8.0):
                rows.append(user_level_cell(epsilon=eps, rounds=5, splits=splits, clinic_ns=ns,
                                            suite=suite, horizon=96, seed=seed, quiet=quiet))
                RESULTS.write_text(json.dumps({"arm": "P1 forecasting: event-level (weather/h96 pilot) + user-level (3 suites)",
                                               "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_forecasting_p1()
    print(f"wrote {RESULTS}")
