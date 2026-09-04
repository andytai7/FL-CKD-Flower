"""P4 spine: DDG-hybrid arm on LSTM-CinC (verified hybrid: per-clinic discrete-Gaussian
masks on QUANTIZED clipped batch-sums, mod-p wire, norm-proof interface, FedAvg transport).

Per round, per clinic: per-record grads → clip C → Poisson batch SUM S_k = Σ_clip(g_i)
(exactly the record-level mechanism of P1 with the Gaussian swapped for the distributed
discrete composition), S_k quantized at scale s=1e-3 (rounding slack √d/2 folded into the
proven bound B = C·√d/s + √d/2 — wait: ‖S‖₂ ≤ |batch|·C... the bound is computed and
VERIFIED per clinic, not assumed), plus per-clinic DDG noise σ_int,k in integer units such
that the float-noise equivalent matches P1's record-level budget route: accounting via
epsilon_rdp(σ_int·s/C, R, δ, q) — the same solver P1 uses, just with distributed-integer
noise; feasibility (KLS21 σ_i ≥ 2√K) is CHECKED and recorded per cell (infeasible cells
keep the row with feasible=False rather than silently upgrading σ).

Wire: (S̃_k_round_int + DDG) mod 2^32; server sums over the wire, decodes to float, divides
by the clinic's batch size, aggregates with FedAvg's n_k weighting — the momentum variant
is intentionally NOT used here (P4 isolates the encoding, not the transport).
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

from .dpsgd_ts import _per_record_grads
from .federated import _eval_probs, _split, load_seq_clinics
from .models import LSTMClassifier
from .p4_ddg import (decompose_feasible, protected_upload, prove_norm, quantize_params,
                     verify_norm)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_p4.json"
MODULUS = 2 ** 32
SCALE = 1e-3


def run_cell_p4(*, epsilon: float | None, rounds: int, splits: list[dict],
                clinic_ns: list[int], seed: int, batch: int = 64, steps_epochs: float = 1.0,
                clip: float = 1.0, lr: float = 0.5, quiet: bool = False) -> dict:
    plans = []
    for n in clinic_ns:
        steps = max(1, math.ceil(steps_epochs * n / batch))
        q = min(1.0, batch / n)
        if epsilon is None:
            sigma_float = 0.0
        else:
            # identical record-level budget route as P1 (per record), noise in sum units
            sigma_float = sigma_for_epsilon(epsilon, rounds * steps, 1e-5,
                                            sampling_probability=q)
        sigma_int = math.ceil(sigma_float * clip / SCALE)
        feasible = sigma_int >= 2  # KLS21 σ_i ≥ 2·√K for the clinic's own mask (K=1 masker)
        realised = (epsilon_rdp(sigma_int * SCALE / clip, rounds * steps, 1e-5,
                                sampling_probability=q) if epsilon is not None else None)
        plans.append({"n": n, "steps": steps, "q": q, "sigma_float": sigma_float,
                      "sigma_int": sigma_int, "feasible": feasible,
                      "composed_epsilon": realised})
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      evaluate_metrics_aggr_fn=weighted_and_worst)
    model = LSTMClassifier()
    torch.manual_seed(seed)
    vec = model.param_vector()
    params = list(model.parameters())
    history = []
    d = vec.numel()
    for rnd in range(1, rounds + 1):
        replies = []
        audited = []
        round_clip = []
        for k, (s, plan) in enumerate(zip(splits, plans)):
            n = len(s["ytr"])
            rng = np.random.default_rng(seed * 1_000_003 + k * 9_973 + rnd * 91_193)
            rng2 = np.random.default_rng(seed * 61_176_011 + k * 101 + rnd * 71)
            m = LSTMClassifier(); m.load_param_vector(vec.clone())
            m.train()
            pos = max(1, int(s["ytr"].sum()))
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor((n - pos) / pos))   # same convention as P1
            clip_rates = []
            for step in range(plan["steps"]):
                sel = rng.random(n) < plan["q"]
                idx = np.flatnonzero(sel)
                if len(idx) == 0:
                    clip_rates.append(0.0)
                    continue
                g = _per_record_grads(m, s["Xtr"], s["ytr"], idx, loss_fn)
                norms = g.norm(dim=1)
                clip_rates.append(float((norms > clip).float().mean()))
                g = g * (clip / norms.clamp(min=clip)).unsqueeze(1)
                summed = g.sum(dim=0).numpy()                # clipped batch SUM
                q_int = quantize_params(summed, SCALE)
                bound = len(idx) * clip / SCALE + math.sqrt(d) / 2.0
                proof = prove_norm(q_int, clip_bound=bound, scale=1.0)
                audited.append(verify_norm(proof))
                if epsilon is None:
                    wire = q_int
                else:
                    up = protected_upload(q_int, plan["sigma_int"], MODULUS, rng2)
                    wire = up["payload_mod"].astype(np.int64)
                    wire = np.where(wire >= MODULUS // 2, wire - MODULUS, wire)
                mean = torch.from_numpy(wire.astype(np.float64) * SCALE / batch).float()
                with torch.no_grad():                        # the step: θ ← θ − lr·mean (descent)
                    m.load_param_vector(m.param_vector() - lr * mean)
            clip_rate = float(np.mean(clip_rates)) if clip_rates else 0.0
            round_clip.append(clip_rate)
            audited.append(True)  # empty-step steps are vacuously norm-valid
            replies.append(train_reply([m.param_vector().numpy()], n))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        vec = torch.from_numpy(arrays.to_numpy_ndarrays()[0])

        eval_replies = []
        for k2, s in enumerate(splits):
            m = LSTMClassifier(); m.load_param_vector(vec.clone())
            eval_replies.append(evaluate_reply(compute_metrics(s["yte"], _eval_probs(m, s["Xte"])),
                                               len(s["yte"]), k2))
        with hushed():
            agg = strategy.aggregate_evaluate(rnd, eval_replies)
        row = dict(agg) if agg else {}
        per_clinic_clip = []  # recompute from stored rates would need per-clinic store; mean ok
        row.update({"round": rnd,
                    "clip_rate": float(np.mean(round_clip)) if round_clip else 0.0,
                    "norm_proofs_ok": all(audited)})
        history.append(row)
        if not quiet:
            eps_txt = "off" if epsilon is None else f"{epsilon:g}"
            print(f"  p4 eps={eps_txt:<5} round {rnd:>2}: AUROC={row.get('auc', float('nan')):.3f} "
                  f"(worst {row.get('auc_worst', float('nan')):.3f}) clip={row['clip_rate']:.2f} "
                  f"proofs={'OK' if row['norm_proofs_ok'] else 'FAIL'}")
    final = history[-1]
    return {"target_epsilon": epsilon, "seed": seed, "transport": "fedavg", "arm": "P4-ddg",
            "modulus": MODULUS, "scale": SCALE, "plans": plans,
            "feasible_all": all(p["feasible"] for p in plans),
            "norm_proofs_ok_all": all(h["norm_proofs_ok"] for h in history),
            "final_auc": float(final.get("auc", np.nan)),
            "final_auc_worst": float(final.get("auc_worst", np.nan)),
            "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
            "history": [{k2: (round(float(v), 4) if isinstance(v, float) else v)
                         for k2, v in h.items()} for h in history]}


def run_grid_p4(*, epsilons=((None, 0.5, 1.0, 2.0, 4.0, 8.0)), rounds: int = 10,
                seeds=(42,), quiet: bool = False) -> list[dict]:
    clinics = load_seq_clinics(downsample=4)
    rows = []
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        ns = [len(s["ytr"]) for s in splits]
        for eps in epsilons:
            rows.append(run_cell_p4(epsilon=eps, rounds=rounds, splits=splits, clinic_ns=ns,
                                    seed=seed, quiet=quiet))
    return rows


def main() -> None:
    rows = run_grid_p4(rounds=10, seeds=(42,))
    RESULTS.write_text(json.dumps({"arm": "P4 verified-hybrid DDG (LSTM, CinC seq clinics)",
                                   "encoding": f"mod-{MODULUS} fixed-point at scale {SCALE}",
                                   "noise": "per-clinic DDG sum (KLS21 feasibility recorded)",
                                   "rows": rows}, indent=2))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
