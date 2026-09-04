"""Forecasting arms of the TS track: GRU vs LSTM on the 9 ETT/Weather seq-clinic suites.

Suites: data/clinics_forecast/<name>_in96_h{96,192,336}/clinic_*.npz carrying the FULL series
+ per-clinic window starts (month-Dirichlet alpha=0.5 partition, exact coverage). Windows:
(x = 96-step history) -> (y = h-step target on channel 0 = the OT/target convention).
Splits are CHRONOLOGICAL per clinic (70/15/15) with a purge gap of `horizon` between the
train and test segments so no window straddles the boundary. Per-suite z-normalisation uses
TRAIN-only statistics (all clinics pooled) — the no-leakage convention.

Metrics: MSE/MAE, dual-level (window-count weighted + worst clinic) per task.py's metrics
rule; plus the temporal-ACF fidelity row the P1 transport noise must survive:
correlation between the ACF(lags 1..24) of the test targets and of the forecasts, plus
|Δ lag-1 ACF| as the headline artifact indicator. ACF computed on the concatenated per-clinic
test horizon means (level series), not per-window.

Transport: FedAvg default for regression arms (no measured drift on the forecasting
partition); FedAvgM arm selectable via transport="fedavgm".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.serverapp.strategy import FedAvg, FedAvgM

from messages import evaluate_reply, hushed, train_reply

from .models import GRUForecaster, LSTMForecaster

FORECAST_DIR = Path(__file__).resolve().parents[2] / "data" / "clinics_forecast"
RESULTS_P2STUB = Path(__file__).resolve().parents[2] / "results"
IN_LEN = 96
HORIZONS = (96, 192, 336)
SUITES = ("etth1", "ettm1", "weather")


def suite_dir(suite: str, horizon: int) -> Path:
    return FORECAST_DIR / f"{suite}_in{IN_LEN}_h{horizon}"


def load_suite(suite: str, horizon: int) -> list[dict]:
    """Clinic payloads for one suite: raw full series + starts; windows materialised per
    split by `windows_for`."""
    d = suite_dir(suite, horizon)
    clinics = []
    for npz in sorted(d.glob("clinic_*.npz")):
        data = np.load(npz)
        series = data["series"].astype(np.float32)          # (T, C) full series copy
        starts = data["starts"].astype(np.int64)            # this clinic's window starts
        clinics.append({"series": series, "starts": starts})
    return clinics


TARGET_CHANNEL = {"etth1": -1, "ettm1": -1, "weather": 1}  # OT (last) for ETT, T for weather


def windows_for(clinic: dict, start: int, end: int, horizon: int,
                mu: np.ndarray, sd: np.ndarray, target_ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose start lies in [start, end); x=(96,) history, y=(h,) target channel.
    Normalisation applied to the full series first (own-train stats)."""
    series = (clinic["series"] - mu) / sd
    ch = target_ch if target_ch >= 0 else series.shape[1] + target_ch
    mask = (clinic["starts"] >= start) & (clinic["starts"] + horizon <= end + IN_LEN)
    xs, ys = [], []
    for s in clinic["starts"][mask]:
        x = series[s : s + IN_LEN]                          # (96, C)
        y = series[s + IN_LEN : s + IN_LEN + horizon, ch : ch + 1]
        if y.shape[0] == horizon and x.shape[0] == IN_LEN:
            xs.append(x); ys.append(y)
    if not xs:
        return (np.zeros((0, IN_LEN, series.shape[1]), np.float32),
                np.zeros((0, horizon, 1), np.float32))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def split_clinic(clinic: dict, horizon: int, suite: str = "etth1",
                 train_frac: float = 0.70, test_frac: float = 0.15) -> dict:
    """Chronological 70/15/15 with a `horizon` purge before the test segment's FIRST window;
    z-normalisation on THE CLINIC'S OWN train segment (the partition's documented leak-free
    convention — never pooled across clinics, never touching the test span)."""
    t = clinic["series"].shape[0]
    cut_tr = int(train_frac * t)
    test_start = int((train_frac + test_frac) * t)          # val segment lands in the middle
    tr = clinic["series"][:cut_tr]
    mu = tr.mean(axis=0, keepdims=True)
    sd = tr.std(axis=0, keepdims=True) + 1e-8
    ch = TARGET_CHANNEL[suite]
    Xtr, ytr = windows_for(clinic, 0, cut_tr, horizon, mu, sd, ch)
    Xte, yte = windows_for(clinic, test_start, t, horizon, mu, sd, ch)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}


def _nf(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def _mse_mae(pred: np.ndarray, true: np.ndarray) -> dict:
    e = pred - true
    return {"mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e)))}


def acf_fidelity(y_forecast_mean: np.ndarray, y_true_mean: np.ndarray,
                 lags: int = 24) -> dict:
    """ACF fidelity between forecast and truth mean level series (per-clinic aggregated
    across the test segment): Pearson correlation of ACF vectors + |Δ lag-1|."""
    def acf(x: np.ndarray, lags: int) -> np.ndarray:
        x = x - x.mean()
        d = float((x * x).mean())
        if d <= 1e-12:
            return np.zeros(lags + 1)
        return np.array([(x[: len(x) - l] * x[l:]).mean() / d for l in range(lags + 1)])

    a_f, a_t = acf(y_forecast_mean, lags), acf(y_true_mean, lags)
    corr = float(np.corrcoef(a_f[1:], a_t[1:])[0, 1]) if a_f[1:].std() > 1e-12 else float("nan")
    return {"acf_corr": corr, "acf_lag1_delta": float(abs(a_f[1] - a_t[1]))}


def forecast_local(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int, lr: float,
                   batch: int, seed: int) -> torch.Tensor:
    """Local MSE training for one registry run."""
    torch.manual_seed(seed)
    model.train()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = np.arange(len(y))
    for _ in range(epochs):
        np.random.default_rng(seed).shuffle(idx)
        for lo in range(0, len(idx), batch):
            rows = idx[lo : lo + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(_nf(X[rows])), _nf(y[rows]))
            loss.backward()
            opt.step()
    return model.param_vector()


def predict(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for lo in range(0, len(X), batch):
            outs.append(model(_nf(X[lo : lo + batch])).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, model.horizon, model.out_channels))


def run_forecast_smoke(*, suite: str = "etth1", horizon: int = 96, rounds: int = 5,
                       seeds: tuple[int, ...] = (42,), epochs: int = 1, model_kind: str = "gru",
                       transport: str = "fedavg", quiet: bool = False) -> list[dict]:
    """GRU/LSTM over one suite; dual-level MSE/MAE + ACF fidelity per round."""
    clinics = load_suite(suite, horizon)
    n_ch = clinics[0]["series"].shape[1]
    make = (lambda: GRUForecaster(n_ch, horizon)) if model_kind == "gru" else \
        (lambda: LSTMForecaster(n_ch, horizon))
    if transport == "fedavgm":
        from flwr.common import ArrayRecord
    strategy = (FedAvgM(server_momentum=0.6) if transport == "fedavgm" else FedAvg)(
        fraction_train=1.0, fraction_evaluate=1.0)
    history: list[dict] = []
    for seed in seeds:
        splits = [split_clinic(c, horizon, suite) for c in clinics]
        splits = [s for s in splits if len(s["ytr"]) and len(s["yte"])]
        model = make(); torch.manual_seed(seed)
        vec = model.param_vector()
        if transport == "fedavgm":
            strategy.current_arrays = ArrayRecord([vec.numpy()])
        for rnd in range(1, rounds + 1):
            replies = []
            for k, s in enumerate(splits):
                m = make(); m.load_param_vector(vec.clone())
                out = forecast_local(m, s["Xtr"], s["ytr"], epochs=epochs, lr=1e-3,
                                     batch=64, seed=seed * 1_000_033 + k * 9_967 + rnd * 90_977)
                replies.append(train_reply([out.numpy()], len(s["ytr"])))
            with hushed():
                arrays, _ = strategy.aggregate_train(rnd, replies)
            vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])
            m = make(); m.load_param_vector(vec.clone())
            row = {"round": rnd, "seed": seed, "suite": suite, "horizon": horizon,
                   "model": model_kind, "transport": transport}
            mses, maes, ns_ = [], [], []
            fc_means = np.zeros(len(splits), dtype=np.float64)
            for k, s in enumerate(splits):
                pred = predict(m, s["Xte"])
                mm = _mse_mae(pred, s["yte"])
                mses.append(mm["mse"]); maes.append(mm["mae"]); ns_.append(len(s["yte"]))
                s["pred_series"] = pred[:, :, 0].reshape(-1)
            tot = sum(ns_)
            row["mse"] = float(sum(a * n for a, n in zip(mses, ns_)) / tot)
            row["mae"] = float(sum(a * n for a, n in zip(maes, ns_)) / tot)
            row["mse_worst"] = float(max(mses))
            # ACF fidelity: pooled over clinics by concatenating mean level series
            lev_true = np.concatenate([s["yte"][:, :, 0].mean(axis=1) for s in splits])
            lev_pred = np.concatenate([s["pred_series"].reshape(len(s["yte"]), horizon).mean(axis=1)
                                       for s in splits])
            row.update(acf_fidelity(lev_pred.astype(np.float64), lev_true.astype(np.float64)))
            history.append(row)
            if not quiet:
                print(f"  fcst {suite}/h{horizon} {model_kind} s{seed} r{rnd}: "
                      f"MSE={row['mse']:.4f} (worst {row['mse_worst']:.4f}) "
                      f"MAE={row['mae']:.4f} acfcorr={row['acf_corr']:.3f} "
                      f"dACF1={row['acf_lag1_delta']:.3f}")
    return history


def main() -> None:
    """Cross-check grid: GRU vs LSTM × 3 suites × 3 horizons, 5 rounds, seeds 42-43;
    incremental JSON per (suite, model) column so partial state is recoverable."""
    out = RESULTS_P2STUB / "dl_ts_forecast_grid.json"
    rows: list[dict] = []
    for suite in SUITES:
        for model_kind in ("gru", "lstm"):
            for horizon in HORIZONS:
                col = run_forecast_smoke(suite=suite, horizon=horizon, rounds=5,
                                         seeds=(42,), epochs=1, model_kind=model_kind)
                rows += col
                out.write_text(json.dumps({"grid": "GRU-vs-LSTM x suites x horizons, 5 rounds",
                                           "rows": rows}, indent=2))
                print(f"[grid] {suite}/{model_kind}/h{horizon} -> MSE {col[-1]['mse']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
