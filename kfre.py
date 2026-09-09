"""Tangri 8-variable Kidney Failure Risk Equation (KFRE) — the rule-based clinical baseline.

The published, fixed-coefficient clinical score (Tangri et al., JAMA 2011;301(15):1553-1559,
8-variable model), exactly as programmed by MDCalc's "Kidney Failure Risk Calculator"
(https://www.mdcalc.com/calc/10045/kidney-failure-risk-calculator):

    P(kidney failure within 5 years) = 1 - S0^x
    x = exp( -0.1992 * (age/10 - 7.036) + 0.1602 * (male - 0.5642)
             -0.4919 * (eGFR/5 - 7.222) + 0.3364 * (ln ACR - 5.137)
             -0.3441 * (albumin - 3.997) + 0.2604 * (phosphorus - 3.916)
             -0.07354 * (bicarbonate - 25.57) - 0.2228 * (calcium - 9.355) )
    S0 = 0.9096 (North America) / 0.9245 (non-North America)

**Role in this repo: evaluation reference only** — one level stricter than `centralized.py`'s
pooled-data ceiling (which at least trains the logreg). Nothing is learned; the coefficients are
published constants, and per-practice evaluation runs locally with only scalar metrics leaving
the practice (rule 3 intact). It is NOT a federated model and NOT an entry in the deployable
model set — that stays logreg-only (rule 2). The federated logreg's job on a KFRE-capable
dataset is to *beat* this rule.

Inputs (US conventional units, as the equation was developed): `age_years`, `male` (0/1),
`egfr` (mL/min/1.73 m²), `acr` (urine albumin-creatinine ratio, mg/g — natural-logged inside),
`albumin` (g/dL), `phosphorus` (mg/dL), `bicarbonate` (mEq/L = mmol/L), `calcium` (mg/dL).
The published score has no imputation rule: rows missing any input are excluded per practice
(complete-case) and the scored share is reported as `coverage`.

⚠️ Task caveat on the NHANES-derived federation (`data/clinics_nhanes_kfre/`): its label is
*prevalent* CKD stage ≥3 (eGFR<60), so the rule is evaluated as a discriminator of prevalent
CKD — a proxy. The equation's true target (5-year progression to kidney failure among CKD
patients) needs longitudinal data; the same code evaluates the real task unchanged once
`extract_features.sql` carries these labs.

    uv run ckd-kfre                         # rule metrics on data/clinics_nhanes_kfre/
    uv run ckd-kfre --region non_north_america
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import KFRE_FEATURE_COLS, LABEL_COL, has_kfre_features, load_clinic_frames
from task import compute_metrics

# Default dataset: the KFRE-capable NHANES federation written by `nhanes_to_clinics`.
DEFAULT_CLINICS_DIR = Path(__file__).resolve().parent / "data" / "clinics_nhanes_kfre"

# (coefficient, centering mean, transform) per input, in KFRE_FEATURE_COLS order — Tangri 2011
# 8-variable model, verified against the MDCalc formula text (see module docstring).
_TERMS: list[tuple[str, float, float, object]] = [
    ("age_years", -0.1992, 7.036, lambda v: v / 10.0),
    ("male", 0.1602, 0.5642, lambda v: v),
    ("egfr", -0.4919, 7.222, lambda v: v / 5.0),
    ("acr", 0.3364, 5.137, np.log),
    ("albumin", -0.3441, 3.997, lambda v: v),
    ("phosphorus", 0.2604, 3.916, lambda v: v),
    ("bicarbonate", -0.07354, 25.57, lambda v: v),
    ("calcium", -0.2228, 9.355, lambda v: v),
]

# 5-year baseline survival by calibration region (MDCalc/KFRE).
S0_5Y = {"north_america": 0.9096, "non_north_america": 0.9245}


def predict_proba(df: pd.DataFrame, region: str = "north_america") -> np.ndarray:
    """P(kidney failure within 5 years) per row; NaN where any KFRE input is missing/invalid."""
    linear = np.zeros(len(df))
    valid = np.ones(len(df), dtype=bool)
    for col, coef, mean, transform in _TERMS:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        if transform is np.log:  # ln ACR undefined at <= 0
            v = np.where(v > 0, v, np.nan)
            v_t = np.where(np.isnan(v), np.nan, np.log(np.where(np.isnan(v), 1.0, v)))
        else:
            v_t = transform(v)
        valid &= np.isfinite(v_t)
        linear += coef * (np.nan_to_num(v_t, nan=0.0) - mean)
    risk = 1.0 - np.power(S0_5Y[region], np.exp(linear))
    return np.where(valid, risk, np.nan)


def evaluate_frames(
    frames: list[pd.DataFrame], *, seed: int = 42, region: str = "north_america"
) -> dict:
    """Score the rule on each practice's local held-out split; aggregate dual-level.

    Row selection exactly mirrors the federated clients: `_local_split(seed, partition_id)` is
    the same permutation every client applies before training, so these are the identical test
    rows a federated model is evaluated on. The rule runs on complete cases only (published
    score: no imputation), unlike the median-imputed features the logreg trains on — `coverage`
    reports how much of each split the rule could score.
    """
    from client_app import _local_split  # the single split definition the clients run

    bad = [pid for pid, df in enumerate(frames) if not has_kfre_features(df)]
    if bad:
        raise ValueError(
            f"frame(s) {bad} lack the KFRE inputs ({KFRE_FEATURE_COLS}) — the rule baseline is "
            f"only defined on a KFRE-capable dataset (e.g. {DEFAULT_CLINICS_DIR.name}; build: "
            f"uv run python -m data.external.nhanes_to_clinics)."
        )

    per_practice, weights, n_total = [], [], 0
    for pid, df in enumerate(frames):
        y = df[LABEL_COL].to_numpy(dtype="int64")
        X_raw = df[KFRE_FEATURE_COLS].to_numpy(dtype="float64")  # unstandardized; rule units
        _, _, X_test, y_test = _local_split(X_raw, y, seed, pid)
        n_total += len(y_test)
        score = predict_proba(pd.DataFrame(X_test, columns=KFRE_FEATURE_COLS), region=region)
        complete = np.isfinite(score)
        n_complete = int(complete.sum())
        m = (
            compute_metrics(y_test[complete], score[complete]) if n_complete
            else {"accuracy": float("nan"), "sensitivity": float("nan"), "auc": float("nan")}
        )
        # A 0.5 cutoff is meaningless for a 5-year failure risk (scores run 0-25% in CKD
        # cohorts) — also report sensitivity at the 10% high-risk threshold from KFRE triage
        # guidance.
        m["sensitivity_at_10pct"] = (
            compute_metrics(y_test[complete], score[complete], threshold=0.10)["sensitivity"]
            if n_complete else float("nan")
        )
        m.update({
            "n_eval": n_complete,
            "coverage": n_complete / len(y_test) if len(y_test) else float("nan"),
        })
        per_practice.append(m)
        weights.append(n_complete)

    out: dict = {"region": region, "horizon_years": 5, "practices": len(frames)}
    for key in ("accuracy", "sensitivity", "sensitivity_at_10pct", "auc", "coverage"):
        pairs = [(n, float(m[key])) for n, m in zip(weights, per_practice)
                 if n and not np.isnan(float(m[key]))]
        if pairs:
            total = sum(n for n, _ in pairs)
            out[key] = sum(n * v for n, v in pairs) / total
            out[f"{key}_worst"] = min(v for _, v in pairs)
    out["n_eval"] = int(sum(weights))
    out["n_eval_total"] = n_total
    out["per_practice"] = per_practice
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="KFRE rule-based clinical baseline (Tangri 8-var)")
    parser.add_argument("--clinics-dir", default=str(DEFAULT_CLINICS_DIR),
                        help="KFRE-capable clinics dir (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--region", choices=list(S0_5Y), default="north_america",
                        help="calibration region; NHANES is a US cohort -> north_america")
    args = parser.parse_args()

    frames = load_clinic_frames(args.clinics_dir)
    r = evaluate_frames(frames, seed=args.seed, region=args.region)
    print(
        f"KFRE 8-variable rule (Tangri 2011 / MDCalc) — 5-year risk, {r['region']}\n"
        f"dataset={Path(args.clinics_dir).name}  practices={r['practices']}  "
        f"held-out rows scored={r['n_eval']}/{r['n_eval_total']}\n" + "-" * 64
    )
    for pid, m in enumerate(r["per_practice"]):
        print(
            f"  practice {pid:>2}: n={m['n_eval']:<5} (coverage {m['coverage']:.0%})  "
            f"AUROC={m['auc']:.3f}  sens={m['sensitivity']:.3f}  "
            f"sens@10%={m['sensitivity_at_10pct']:.3f}"
        )
    print(
        f"  aggregated:  AUROC={r['auc']:.3f} (worst practice {r['auc_worst']:.3f})  "
        f"sensitivity={r['sensitivity']:.3f}  sens@10%={r['sensitivity_at_10pct']:.3f}  "
        f"accuracy={r['accuracy']:.3f}  coverage={r['coverage']:.0%}"
    )


if __name__ == "__main__":
    main()
