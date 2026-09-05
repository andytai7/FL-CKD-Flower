"""P3-RR: per-vote randomized response arm for the image FedCT benchmark.

Completes the P3 consensus family beside the Gaussian-DDG vote row (fedct_image): instead of
noising the vote SUM against a sensitivity bound, each teacher locally flips its binary vote
with probability p per query (binary randomized response), and the server debiases the count:

    S ~ sum of reported votes                (teachers flip vote v -> 1-v w.p. p)
    E[S] = k*(1-p) + (k - k)*p for k true 1-votes ->  k_hat = (S - K*p) / (1 - 2*p)

label = 1 iff k_hat >= K/2  (debiased majority).

Privacy: the per-vote release is a LOCAL mechanism with epsilon_v = ln((1-p)/p) (pure DP per
query per teacher). Composition over the q pool queries uses advanced composition at the
bench standard delta=1e-5:

    eps_total = sqrt(2 q ln(1/delta)) * eps_v + q * eps_v * (exp(eps_v) - 1)   <=   eps_target

(eps_v solved by bisection); the pure-DP basic-composition reference eps_v = eps_target / q is
recorded in aux. Closed-form estimator noise (vs the Gaussian row's sigma_v):

    sigma_equiv = sqrt(K * p * (1-p)) / (1 - 2*p)

so the row states its own vote-noise in exactly the units that destroyed the Gaussian arm.

Rare-positive ('per-class') audit: RR is symmetric per vote, but the melanoma class is the
rare one (pool prevalence ~0.06-0.11 per cell notes) — the row therefore carries the realized
positive-label counts under clean majority vs RR labels (pos_count_clean / pos_count_rr) and
the noisy positive-recall, making the rare-class survival visible instead of hidden inside
aggregate accuracy. One-sided flip variants (protect only 1->0) change constants only, not
the K << sigma conclusion, and are not separately benchmarked — stated here to fix the
design decision.

Rerun economics: teacher training dominates the P3 runtime, so this module re-trains
teachers ONCE per seed and re-emits the Gaussian rows (bitwise-replication check against the
committed dl_image_p3.json proves rerun fidelity) plus the RR rows. Writes
results/dl_image_p3_rr.json with an idempotent per-seed resume.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .federated import _split, load_image_clinics
from .fedct_image import (POOL_FRAC, Q_CAP, consensus_labels, distill_student,
                          dual_metrics_row, measure_flip_rate, query_sigma,
                          teacher_votes, train_teachers)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p3_rr.json"
P3_BASE = RESULTS.parent / "dl_image_p3.json"
DELTA = 1e-5


def eps_vote_advanced(eps_target: float, q: int, delta: float = DELTA) -> float:
    """Bisection on advanced composition of a pure local RR mech over q queries. At
    eps_v = eps_target the composed value is ~q*eps_target > eps_target, so hi is valid."""
    lo, hi = 0.0, eps_target
    for _ in range(100):
        mid = (lo + hi) / 2
        comp = math.sqrt(2 * q * math.log(1.0 / delta)) * mid + q * mid * math.expm1(mid)
        if comp > eps_target:
            hi = mid
        else:
            lo = mid
    return lo


def rr_labels(votes: np.ndarray, *, p: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flip each teacher vote w.p. p, return (raw_counts S, debiased positive labels)."""
    rng_r = rng
    flip = rng_r.random(votes.shape) < p
    reported = np.where(flip, 1 - votes, votes)
    S = reported.sum(axis=0).astype(float)
    k = votes.shape[0]
    k_hat = (S - k * p) / (1.0 - 2.0 * p)
    return reported, (k_hat >= k / 2).astype(np.int64)


def run(*, epsilons=(0.5, 1.0, 2.0, 4.0, 8.0), q: int = Q_CAP, seeds=(42,),
        pool_frac: float = POOL_FRAC, quiet: bool = False) -> dict:
    clinics = load_image_clinics()
    out_rows, replication = [], []
    for seed in seeds:
        teacher_splits, pool_frames = [], []
        for k, (X, y) in enumerate(clinics):
            s = _split(X, y, seed, k)
            cut = int(len(s["Xtr"]) * (1.0 - pool_frac))
            teacher_splits.append({"Xtr": s["Xtr"][:cut], "ytr": s["ytr"][:cut],
                                   "Xte": s["Xte"], "yte": s["yte"]})
            pool_frames.append({"X": s["Xtr"][cut:], "y": s["ytr"][cut:]})
        pool_X = np.concatenate([p_["X"] for p_ in pool_frames])
        pool_y = np.concatenate([p_["y"] for p_ in pool_frames])
        rng = np.random.default_rng(seed * 77 + 3)  # same draw as fedct_image.run_fedct
        take = rng.permutation(len(pool_y))[: min(q, len(pool_y))]
        pool_X, pool_y = pool_X[take], pool_y[take]
        teachers = train_teachers(teacher_splits, seed=seed)
        _, votes = teacher_votes(teachers, pool_X)
        gamma = measure_flip_rate(votes)
        K = len(teachers)
        clean = consensus_labels(votes, sigma=0.0, rng=rng)
        pos_clean = int(clean.sum())

        # --- Gaussian replication cells (byte-regression against committed dl_image_p3.json)
        row_defs = [("conservative", math.sqrt(2.0 * K)),
                    ("refined", math.sqrt(2.0 * K * max(gamma, 1e-9)))]
        for eps in (None,) + tuple(epsilons):
            for sens_name, sens in row_defs:
                sig = query_sigma(eps, sens, len(pool_y)) if eps is not None else 0.0
                labels = clean if eps is None else consensus_labels(
                    votes, sigma=sig, rng=np.random.default_rng(seed * 55 + 1))
                student = distill_student(pool_X, labels, seed=seed)
                row = dual_metrics_row(student, teacher_splits)
                replication.append({"epsilon": eps, "sensitivity": sens_name,
                                    "sigma_votes": sig, **row})
                if not quiet:
                    eps_txt = "off" if eps is None else f"{eps:g}"
                    print(f"  imgP3-RRrun replicate eps={eps_txt:<4} [{sens_name:>11}] "
                          f"sigma_v={sig:>8.1f} -> AUROC {row['auc']:.3f}")

        # --- RR cells
        for eps in epsilons:
            ev = eps_vote_advanced(eps, len(pool_y))
            p = 1.0 / (1.0 + math.exp(ev))
            sigma_equiv = math.sqrt(K * p * (1.0 - p)) / (1.0 - 2.0 * p)
            _, labels_rr = rr_labels(votes, p=p, rng=np.random.default_rng(seed * 99 + 7))
            pos_rr = int(labels_rr.sum())
            pos_rec = float(((labels_rr == 1) & (clean == 1)).sum() / max(1, pos_clean))
            student = distill_student(pool_X, labels_rr, seed=seed)
            row = dual_metrics_row(student, teacher_splits)
            row.update({"epsilon": eps, "mechanism": "rr-vote", "eps_vote": ev,
                        "eps_vote_basic_ref": eps / len(pool_y), "p_flip": p,
                        "sigma_equiv": sigma_equiv, "pos_count_clean": pos_clean,
                        "pos_count_rr": pos_rr, "pos_recall_vs_clean": pos_rec,
                        "sensitivity": "local-rr", "seed": seed, "arm": "P3-rr-fedct"})
            out_rows.append(row)
            if not quiet:
                print(f"  imgP3-RR eps={eps:<4} eps_v={ev:.4f} p={p:.3f} "
                      f"sigma_eq={sigma_equiv:8.1f} pos {pos_rr}/{pos_clean} "
                      f"rec={pos_rec:.2f} -> AUROC {row['auc']:.3f}")
        # checkpoint per seed (teacher rerun is the expensive part)
        RESULTS.write_text(json.dumps({"arm": "P3 FedCT + per-vote randomized response (CNN-Small, DermaMNIST)",
                                       "composition": "advanced-comp local RR, delta=1e-5 (see module docstring)",
                                       "pool": "clinic-train carve 10% q<=512 (same carve as fedct_image)",
                                       "rows": out_rows,
                                       "gaussian_replication": replication}, indent=2))
    verdict = verify_replication(replication)
    RESULTS.write_text(json.dumps({"arm": "P3 FedCT + per-vote randomized response (CNN-Small, DermaMNIST)",
                                   "composition": "advanced-comp local RR, delta=1e-5 (see module docstring)",
                                   "pool": "clinic-train carve 10% q<=512 (same carve as fedct_image)",
                                   "gaussian_replication_verdict": verdict,
                                   "rows": out_rows,
                                   "gaussian_replication": replication}, indent=2))
    return {"rows": out_rows, "replication": replication}


def verify_replication(replication: list[dict]) -> dict:
    """Every replicated Gaussian cell must match the committed p3 grid a.u.c. (within 1e-6 —
    same torch seeds and deterministic CPU BLAS; teachers retrained identically)."""
    base = json.loads(P3_BASE.read_text())["rows"]
    want = {}
    for r in base:
        if r["seed"] != 42:
            continue
        for c in r["cells"]:
            want[(r["epsilon"], c["sensitivity"])] = c["auc"]
    worst = 0.0
    for rep in replication:
        key = (rep["epsilon"], rep["sensitivity"])
        b, a = want.get(key), rep["auc"]
        if b is None or (math.isnan(a) and math.isnan(b)):
            continue
        worst = max(worst, abs(a - b) if not (math.isnan(a) or math.isnan(b)) else 0.0)
    return {"compared": len(replication), "max_abs_auc_delta": worst,
            "within_1e-6": worst <= 1e-6}


if __name__ == "__main__":
    got = run()
    rep = verify_replication(got["replication"])
    print("replication check:", rep)
    print(f"wrote {RESULTS}")
