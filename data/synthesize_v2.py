"""Harder per-clinic synthetic CKD generator (V2) — ADDITIVE to the frozen V1 pipeline.

V1 (`data/synthesize.py`, frozen per `data/VERSIONS.md`) draws the label from a *linear* logit in
the observed features — the exact model class the pipeline trains. Monte-Carlo on 200k patients:
Bayes AUROC = pooled-logreg AUROC = 0.8513. Zero misspecification headroom, independent
comorbidities (max |corr| 0.057 among dx_*), 31% prevalence. Result: AUROC is flat across
composed ε 0.5–8 (notebook 03) and the central-DP knee only appears at ε ≈ 12
(`results/privacy.json`). A privacy-utility curve with no knee can't calibrate an ε-policy.

V2 hardens the *data-generating process* along five orthogonal axes while keeping the V1 schema
(exactly `FEATURE_COLS` + `LABEL_COL`, one CSV per practice), so every existing runner consumes
the output unchanged via `uv run ckd-simulate --clinics-dir <dir>` / `load_clinic_frames(<dir>)`.
No new loader, no edits to any V1 file.

Knobs (a `Hardness` value; defaults reproduce V1-equivalent behaviour):

  interactions      truth has terms logreg cannot express: diabetes×age interaction and
                    accelerating old-age risk (relu(age−65)). Puts the Bayes ceiling ABOVE the
                    logreg ceiling — misspecification headroom for DP noise to erode.
  frailty           latent per-patient frailty g~N(0,1) shifts every comorbidity prevalence AND
                    carries unexplained risk. Correlates the flags (real EHRs have this; V1
                    doesn't) and adds Bayes error via an unobserved confounder.
  label_noise       documentation delay: a share of true CKD never gets coded (fn), rare false
                    codes (fp). Makes rule-5's primary sensitivity metric noise-sensitive.
  risk_noise        independent unobserved risk u~N(0,σ) in the logit: irreducible Bayes error
                    no feature can explain. The knob that makes the ε-knee appear — with
                    everything else, signal often just moves.
  concept_shift     per-clinic truth jitter β_j = β + δ_j — no single global truth (the
                    archetypes only shifted P(X) in V1; V2 can shift P(y|X)).
  undercoding       per-clinic documentation rate ρ_j ∈ U[0.6,1]: true dx flags are dropped with
                    1−ρ_j (years_since follows its flag to 0 per §3a). Label is sampled from the
                    TRUE biology, clinics display the under-coded view. Informative structural
                    zeros, heterogeneously severe per practice.
  imbalance         per-clinic base-rate jitter σ_int (prevalence spread 2%→45%) and, when
                    `sizes` is given, the quantity ladder (compound with the V1 ladder).
  years_lognormal   years_since draws become lognormal (heavy right tail) instead of V1's
                    uniform(0,19). Realism knob; on in `main`, off in ablations.

Additive suites written per preset to `data/clinics_v2_<name>/` (V1 paths never touched).
`main` = all knobs on, calibrated to prevalence ≈ 0.13; ablation suites flip exactly one knob
against the V1-equivalent baseline, isolating what moves the ε knee in notebook 05.

Determinism: one suite base seed → derived per-clinic seeds (base+1000+cid, V1's scheme) and
separate streams for sizes (base), β-jitter (base+600_000+cid) and calibration MC
(base+500_000). Default base = 1_000_000 + 42 is disjoint from every V1 stream
(V1: ≤1051 clinic, 90_042 public). `bayes_diagnosis()` re-derives the truth in closed form to
report (Bayes AUROC, pooled-logreg ceiling) for a suite without touching disk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import FEATURE_COLS, LABEL_COL
from .synthesize import _BETA, ARCHETYPES, Archetype  # V1 biology, single source of truth

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Hardness:
    """V2 generation knobs. All-off == V1-equivalent process (for a *different* RNG stream)."""

    interaction_dm_age: float = 0.0   # γ1: dx_diabetes × clip((age−60)/15, −2, 3)
    interaction_age_relu: float = 0.0  # γ2: relu(age−65)/20 accelerating old-age risk
    frailty_sd: float = 0.0           # latent frailty sd (0 = V1-independent flags)
    frailty_feature_load: float = 1.3  # how strongly frailty shifts comorbidity prevalences
    frailty_risk_beta: float = 0.55    # unexplained risk carried by frailty in the logit
    risk_noise_sd: float = 0.0        # irreducible risk u~N(0,σ), INDEPENDENT of every feature —
                                      # pure Bayes error (frailty is partially recoverable through
                                      # the flags it correlates; u is not)
    label_fn: float = 0.0             # P(true CKD uncoded)  — documentation delay
    label_fp: float = 0.0             # P(false N18 code)
    beta_jitter: float = 0.0          # per-clinic concept shift, sd ∝ (0.3+|β|)
    undercode_lo: float = 1.0         # doc rate ρ_j ~ U[undercode_lo, 1] (1 = V1 full coding)
    intercept_jitter: float = 0.0     # per-clinic base-rate shift sd (prevalence spread)
    years_lognormal: bool = False     # heavy-tailed years_since instead of uniform(0,19)
    intercept_shift: float = 0.0      # calibration: shifts pooled prevalence to target


# Suite registry. Ablations isolate one axis against the otherwise-V1 process; `main` combines.
# Tuned 2026-09-03 against the acceptance band (probe=calibrate_prevalence+bayes_diagnosis):
# ceiling BELOW V1's 0.861, misspecification gap ≥0.02, pooled prevalence ≈0.13. `main`:
# Bayes 0.868 / logreg 0.820 / gap 0.048 / flag corr 0.18. Calibration sharpened the tradeoff:
# lowering prevalence to realistic CKD rates RAISES measured AUROC, so risk_noise does the work
# of keeping the ceiling reachable-erodible.
_HARD_MAIN = dict(
    interaction_dm_age=0.25, interaction_age_relu=0.5,
    frailty_sd=1.0, frailty_risk_beta=0.2, risk_noise_sd=1.4,
    label_fn=0.18, label_fp=0.02, beta_jitter=0.15,
    undercode_lo=0.6, years_lognormal=True,
)
HARDNESS_PRESETS: dict[str, Hardness] = {
    "main": Hardness(**_HARD_MAIN),
    "interactions": Hardness(interaction_dm_age=0.45, interaction_age_relu=0.9),
    "frailty": Hardness(frailty_sd=1.0),
    "risk_noise": Hardness(risk_noise_sd=1.4),
    "label_noise": Hardness(label_fn=0.18, label_fp=0.02),
    "concept_shift": Hardness(beta_jitter=0.15),
    "undercoding": Hardness(undercode_lo=0.6),
    "imbalance": Hardness(intercept_jitter=0.9),
}

# Quantity ladder for the imbalance suite (Era-15-style explicit sizes, wider than V1's band).
IMBALANCE_SIZES = [40, 60, 90, 120, 150, 200, 280, 380, 500, 620]

TARGET_PREVALENCE = 0.13  # primary-care CKD stage ≥3 ballpark; V1 pooled was ≈ 0.31


def _years(rng: np.random.Generator, present: np.ndarray, lognormal: bool) -> np.ndarray:
    """years_since paired with its flag; 0 encodes absent (§3a). V1 uniform or heavy-tail."""
    if not lognormal:
        return np.where(present == 1, rng.uniform(0, 19, len(present)), 0.0)
    yrs = np.clip(np.exp(rng.normal(1.6, 0.8, len(present))), 0.0, 40.0)
    return np.where(present == 1, yrs, 0.0)


def _truth_logit(
    df: pd.DataFrame, frailty: np.ndarray, h: Hardness, beta_delta: dict[str, float],
) -> np.ndarray:
    """Per-patient logit of the TRUE label: V1 linear biology + V2 hardness terms + clinic δ."""
    logit = np.full(len(df), _BETA["intercept"] + h.intercept_shift, dtype=float)
    for col, beta in _BETA.items():
        if col == "intercept":
            continue
        logit += (beta + beta_delta.get(col, 0.0)) * df[col].to_numpy(dtype=float)
    age = df["age_years"].to_numpy(dtype=float)
    dm = df["dx_diabetes"].to_numpy(dtype=float)
    logit += h.interaction_dm_age * dm * np.clip((age - 60.0) / 15.0, -2.0, 3.0)
    logit += h.interaction_age_relu * np.maximum(age - 65.0, 0.0) / 20.0
    logit += (h.frailty_risk_beta if h.frailty_sd > 0 else 0.0) * frailty
    return logit


def _risk_noise(rng: np.random.Generator, h: Hardness, n: int) -> np.ndarray:
    """Independent unobserved risk — drawn on its own call so ablation RNG streams stay aligned."""
    return rng.normal(0.0, h.risk_noise_sd, n) if h.risk_noise_sd > 0 else np.zeros(n)


def _draw_patients_v2(
    arch: Archetype, n: int, rng: np.random.Generator, h: Hardness,
    beta_delta: dict[str, float], doc_rate: float, base_rate_shift: float,
    *, return_truth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """One clinic: TRUE biology → label (from true state) → under-coded OBSERVED features."""
    g = rng.normal(0.0, h.frailty_sd, n) if h.frailty_sd > 0 else np.zeros(n)
    age = np.clip(rng.normal(arch.age_mean + (5.0 * g if h.frailty_sd > 0 else 0.0),
                             arch.age_sd, n), 18, 95)

    def flag(p: float) -> np.ndarray:
        if h.frailty_sd > 0:  # correlated via latent frailty
            prob = 1.0 / (1.0 + np.exp(-(math.log(p / (1 - p)) + h.frailty_feature_load * g)))
            return (rng.random(n) < prob).astype(int)
        return (rng.random(n) < p).astype(int)  # V1 behaviour exactly

    true_flags = {
        "dx_hypertonie": flag(arch.p_hypertonie),
        "dx_diabetes": flag(arch.p_diabetes),
        "dx_khk": flag(arch.p_khk),
        "dx_adipositas": flag(arch.p_adipositas),
        "dx_herzinsuffizienz": flag(arch.p_herzinsuffizienz),
        "dx_hyperurikaemie": flag(arch.p_hyperurikaemie),
    }
    years_cols = {
        "hypertonie": "years_since_hypertonie_dx",
        "diabetes": "years_since_diabetes_dx",
        "khk": "years_since_khk_dx",
    }

    df = pd.DataFrame({"age_years": age, **true_flags})
    for key, col in years_cols.items():
        df[col] = _years(rng, true_flags[f"dx_{key}"], h.years_lognormal)

    # Label samples from TRUE biology + per-clinic base-rate shift (imbalance knob) +
    # independent unobserved risk (nothing in X can predict u).
    logit = _truth_logit(df, g, h, beta_delta) + base_rate_shift + _risk_noise(rng, h, n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y_true = (rng.random(n) < prob).astype(int)
    # Documentation delay: true CKD uncoded with label_fn; false N18 codes with label_fp.
    flip_to0 = (y_true == 1) & (rng.random(n) < h.label_fn)
    flip_to1 = (y_true == 0) & (rng.random(n) < h.label_fp)
    df[LABEL_COL] = np.where(flip_to0, 0, np.where(flip_to1, 1, y_true))

    # Under-coding of the OBSERVED record: drop true flags with 1−ρ (paired years → 0).
    keep = (rng.random((n, 6)) < doc_rate)
    for i, col in enumerate(true_flags):
        df[col] = df[col] * keep[:, i]
    for key, col in years_cols.items():
        df.loc[df[f"dx_{key}"] == 0, col] = 0.0

    out = df[[*FEATURE_COLS, LABEL_COL]]
    return (out, prob) if return_truth else out


def _beta_delta(rng: np.random.Generator, h: Hardness) -> dict[str, float]:
    """Per-clinic concept shift: δc ~ N(0, jitter²) · (0.3+|βc|) — proportional wiggle."""
    if h.beta_jitter <= 0:
        return {}
    return {
        col: float(rng.normal(0.0, h.beta_jitter) * (0.3 + abs(b)))
        for col, b in _BETA.items() if col != "intercept"
    }


def generate_clinics_v2(
    num_clinics: int = 10, *, base_seed: int = 1_000_042, hardness: Hardness = HARDNESS_PRESETS["main"],
    sizes: list[int] | None = None, min_n: int = 120, max_n: int = 600,
) -> list[tuple[int, str, pd.DataFrame]]:
    """V2 analogue of `synthesize.generate_clinics` (same tuple contract)."""
    if sizes is not None and len(sizes) != num_clinics:
        raise ValueError(f"sizes has {len(sizes)} entries for {num_clinics} clinics")
    rng_sizes = np.random.default_rng(base_seed)
    clinics = []
    h = hardness
    for cid in range(num_clinics):
        arch = ARCHETYPES[cid % len(ARCHETYPES)]
        n = sizes[cid] if sizes is not None else int(rng_sizes.integers(min_n, max_n + 1))
        rng_jitter = np.random.default_rng(base_seed + 600_000 + cid)
        clinic_rng = np.random.default_rng(base_seed + 1000 + cid)
        delta = _beta_delta(rng_jitter, h)
        doc = float(rng_jitter.uniform(h.undercode_lo, 1.0))
        shift = float(rng_jitter.normal(0.0, h.intercept_jitter))
        clinics.append((cid, arch.name, _draw_patients_v2(arch, n, clinic_rng, h, delta, doc, shift)))
    return clinics


def bayes_diagnosis(
    hardness: Hardness, *, base_seed: int = 1_000_042, per_archetype: int = 30_000,
) -> dict:
    """Closed-form diagnostics for a suite: Bayes AUROC vs pooled-logreg ceiling (big MC sample).

    Bayes = score the observed labels with theTRUE probabilities the generator used
    (label noise attenuates both). Logreg ceiling = honest 80/20 pooled fit on observed features.
    The pair (bayes, logreg) is the suite's headroom statement: V1's was (0.851, 0.851).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # Under-coding enters at the band midpoint — the written suites draw ρ_j per clinic from
    # U[lo,1], whose expectation is this value (labels are unaffected; only observed flags drop).
    doc = 0.5 * (hardness.undercode_lo + 1.0)
    frames, probs = [], []
    for i, arch in enumerate(ARCHETYPES):
        rng = np.random.default_rng(base_seed + 500_000 + i)
        df, p = _draw_patients_v2(arch, per_archetype, rng, hardness, {}, doc, 0.0,
                                  return_truth=True)
        frames.append(df)
        probs.append(p)
    big = pd.concat(frames, ignore_index=True)
    p_true = np.concatenate(probs)
    X = big[FEATURE_COLS].to_numpy(float)
    y = big[LABEL_COL].to_numpy()
    auc_bayes = float(roc_auc_score(y, p_true))

    rng_split = np.random.default_rng(base_seed + 500_099)
    cut = rng_split.random(len(big)) < 0.8
    sc = StandardScaler().fit(X[cut])
    lr = LogisticRegression(max_iter=2000, C=1e3).fit(sc.transform(X[cut]), y[cut])
    auc_lr = float(roc_auc_score(y[~cut], lr.decision_function(sc.transform(X[~cut]))))
    C = np.corrcoef(X[cut], rowvar=False)
    np.fill_diagonal(C, 0.0)
    flag_idx = [FEATURE_COLS.index(c) for c in FEATURE_COLS if c.startswith("dx_")]
    Cf = C[np.ix_(flag_idx, flag_idx)]
    return {
        "bayes_auroc": round(auc_bayes, 4),
        "logreg_ceiling_auroc": round(auc_lr, 4),
        "prevalence_observed": round(float(y.mean()), 4),
        "max_abs_feature_corr": round(float(np.abs(C).max()), 4),
        "max_abs_flag_corr": round(float(np.abs(Cf).max()), 4),
    }


def calibrate_prevalence(
    hardness: Hardness, *, base_seed: int = 1_000_042, target: float = TARGET_PREVALENCE,
) -> Hardness:
    """Bisection on intercept_shift until the MC pooled observed prevalence hits `target`."""
    def prev(shift: float) -> float:
        lo, hi = [], []
        for i, arch in enumerate(ARCHETYPES):
            rng = np.random.default_rng(base_seed + 500_100 + i)
            df = _draw_patients_v2(arch, 20_000, rng, dataclass_replace(hardness, intercept_shift=shift),
                                   {}, 1.0, 0.0)
            lo.append(df[LABEL_COL].mean())
        return float(np.mean(lo))

    lo_s, hi_s = -3.0, 3.0
    for _ in range(22):
        mid = 0.5 * (lo_s + hi_s)
        if prev(mid) > target:
            hi_s = mid  # prevalence too high -> lower intercept -> search below mid
        else:
            lo_s = mid
    return dataclass_replace(hardness, intercept_shift=0.5 * (lo_s + hi_s))


def dataclass_replace(h: Hardness, **kw) -> Hardness:  # local, avoids dataclasses import ambiguity
    d = asdict(h)
    d.update(kw)
    return Hardness(**d)


def write_suite(
    name: str, out_dir: Path | None = None, *, num_clinics: int = 10, base_seed: int = 1_000_042,
) -> Path:
    """Generate one suite, calibrate its intercept, write per-clinic CSVs + provenance JSON."""
    out = Path(out_dir) if out_dir else _HERE / f"clinics_v2_{name}"
    out.mkdir(parents=True, exist_ok=True)
    hardness = HARDNESS_PRESETS[name]
    sizes = IMBALANCE_SIZES if name == "imbalance" else None
    hardness = calibrate_prevalence(hardness, base_seed=base_seed)
    clinics = generate_clinics_v2(num_clinics, base_seed=base_seed, hardness=hardness, sizes=sizes)

    combined = []
    for cid, arch, df in clinics:
        df.to_csv(out / f"clinic_{cid:02d}_{arch}.csv", index=False)
        combined.append(df.assign(clinic_id=cid, clinic_archetype=arch))
    pd.concat(combined, ignore_index=True).to_csv(out / "all_clinics.csv", index=False)

    meta = {
        "suite": name, "num_clinics": num_clinics, "base_seed": base_seed,
        "hardness": asdict(hardness), "sizes": sizes,
        "diagnostics": bayes_diagnosis(hardness, base_seed=base_seed),
        "clinics": [
            {"clinic_id": cid, "archetype": arch, "n": len(df),
             "ckd_rate": round(float(df[LABEL_COL].mean()), 4)}
            for cid, arch, df in clinics
        ],
        "v1_reference": {"bayes_auroc": 0.8513, "logreg_ceiling_auroc": 0.8513,
                          "prevalence_pooled": 0.309, "max_abs_flag_corr": 0.057,
                          "note": "200k Monte-Carlo, 2026-09-03"},
    }
    (out / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate harder V2 clinic suites (V1 untouched)")
    parser.add_argument("--suites", nargs="+", default=["main"],
                        choices=[*HARDNESS_PRESETS, "all"])
    parser.add_argument("--clinics", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1_000_042)
    args = parser.parse_args()

    names = list(HARDNESS_PRESETS) if "all" in args.suites else args.suites
    for name in names:
        out = write_suite(name, num_clinics=args.clinics, base_seed=args.seed)
        meta = json.loads((out / "suite_meta.json").read_text())
        d = meta["diagnostics"]
        print(f"{name:<14} -> {out.name}/  Bayes={d['bayes_auroc']:.3f} "
              f"logreg={d['logreg_ceiling_auroc']:.3f} prev={d['prevalence_observed']:.3f} "
              f"max|flag_corr|={d['max_abs_flag_corr']:.3f}")


if __name__ == "__main__":
    main()
