"""P4 spine for the image track: DDG-hybrid arm on CNN-Small (DermaMNIST melanoma).

Mirrors research/privacy_dl_ts/p4_spine (TS branch) — per-branch copies are the agreed
topology. Per round, per clinic: per-record grads → clip C → Poisson batch SUM → quantize
at SCALE=1e-3 → norm proof → per-clinic DDG mask (integer units) mod 2^32 → FedAvg wire
(no momentum; the P4 row isolates the encoding). Accounting route identical to P1-image
(epsilon_rdp over rounds·steps at the clinic's q with σ_float; the integer σ_int = ⌈σ·C/s⌉
is what actually travels on the wire — KLS feasibility σ_int ≥ 2 recorded per plan).
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
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics

from .dpsgd_image import _per_record_grads
from .federated import _eval_probs, _split, load_image_clinics
from .models import CNNSmall

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p4.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)
MODULUS = 2 ** 32
SCALE = 1e-3


def quantize_params(vec: np.ndarray, scale: float) -> np.ndarray:
    return np.round(vec / scale).astype(np.int64)


def run_cell(*, epsilon: float | None, rounds: int, splits: list[dict], clinic_ns: list[int],
             seed: int, batch: int = 64, clip: float = 1.0, lr: float = 0.1,
             quiet: bool = False) -> dict:
    from research.privacy_dl_image.p4_ddg_image import (protected_upload, prove_norm,
                                                        verify_norm, sample_dgauss)
    plans = []
    for n in clinic_ns:
        steps = max(1, math.ceil(n / batch))
        q = min(1.0, batch / n)
        sigma_float = (sigma_for_epsilon(epsilon, rounds * steps, 1e-5,
                                         sampling_probability=q)
                       if epsilon is not None else 0.0)
        sigma_int = math.ceil(sigma_float * clip / SCALE)
        plans.append({"n": n, "steps": steps, "q": q, "sigma_float": sigma_float,
                      "sigma_int": sigma_int, "feasible": sigma_int >= 2,
                      "composed_epsilon": epsilon_rdp(sigma_int * SCALE / clip, rounds * steps,
                                                      1e-5, sampling_probability=q)
                      if epsilon is not None else None})
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      evaluate_metrics_aggr_fn=weighted_and_worst)
    model = CNNSmall()
    torch.manual_seed(seed)
    vec = model.param_vector()
    history = []
    d = vec.numel()
    for rnd in range(1, rounds + 1):
        replies = []
        all_ok = []
        clip_rates = []
        for k, (s, plan) in enumerate(zip(splits, plans)):
            n = len(s["ytr"])
            rng = np.random.default_rng(seed * 1_000_003 + k * 9_973 + rnd * 91_193)
            rng2 = np.random.default_rng(seed * 61_176_011 + k * 101 + rnd * 71)
            m = CNNSmall(); m.load_param_vector(vec.clone())
            m.train()
            pos = max(1, int(s["ytr"].sum()))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((n - pos) / pos))
            for _step in range(plan["steps"]):
                idx = np.flatnonzero(rng.random(n) < plan["q"])
                if len(idx) == 0:
                    clip_rates.append(0.0)
                    continue
                g = _per_record_grads(m, s["Xtr"], s["ytr"], idx, loss_fn)
                norms = g.norm(dim=1)
                clip_rates.append(float((norms > clip).float().mean()))
                g = g * (clip / norms.clamp(min=clip)).unsqueeze(1)
                summed = g.sum(dim=0).numpy()
                q_int = quantize_params(summed, SCALE)
                bound = len(idx) * clip / SCALE + math.sqrt(d) / 2.0
                proof = prove_norm(q_int, clip_bound=bound, scale=1.0)
                all_ok.append(verify_norm(proof))
                if epsilon is None:
                    wire = q_int
                else:
                    up = protected_upload(q_int, plan["sigma_int"], MODULUS, rng2)
                    wire = up["payload_mod"].astype(np.int64)
                    wire = np.where(wire >= MODULUS // 2, wire - MODULUS, wire)
                mean = torch.from_numpy(wire.astype(np.float64) * SCALE / batch).float()
                with torch.no_grad():                        # descent step
                    m.load_param_vector(m.param_vector() - lr * mean)
            replies.append(train_reply([m.param_vector().numpy()], n))
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
        row.update({"round": rnd, "clip_rate": float(np.mean(clip_rates)),
                    "norm_proofs_ok": all(all_ok)})
        history.append(row)
        if not quiet:
            eps_txt = "off" if epsilon is None else f"{epsilon:g}"
            print(f"  imgP4 eps={eps_txt:<4} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                  f"(worst {row.get('auc_worst', float('nan')):.3f}) clip={row['clip_rate']:.2f} "
                  f"proofs={'OK' if row['norm_proofs_ok'] else 'FAIL'}")
    final = history[-1]
    return {"target_epsilon": epsilon, "seed": seed, "arm": "P4-ddg", "transport": "fedavg",
            "modulus": MODULUS, "scale": SCALE, "plans": plans,
            "feasible_all": all(p["feasible"] for p in plans),
            "norm_proofs_ok_all": all(h["norm_proofs_ok"] for h in history),
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
            RESULTS.write_text(json.dumps({"arm": "P4 verified-hybrid DDG (CNN-Small, DermaMNIST melanoma)",
                                           "encoding": f"mod-{MODULUS} fixed-point at scale {SCALE}",
                                           "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_grid()
    print(f"wrote {RESULTS}")
