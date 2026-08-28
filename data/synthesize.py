"""Generate signal-bearing, per-clinic synthetic CKD data to simulate the FLIP-IT federation.

Each *clinic* is a separate GP practice with its own patient population — **non-IID**: different
age and comorbidity profiles, different CKD prevalence, and different panel sizes. But every clinic
shares the **same underlying CKD biology**: one fixed logistic risk model maps features → P(CKD).

That shared truth is the whole point. Because each clinic only sees its slice of the population, a
model trained at a single clinic generalizes poorly to the others; federating across clinics with
Flower recovers a global model that works everywhere — **without any clinic sharing patient rows**.

Each clinic's CSV simulates the output of running `extract_features.sql` against that practice's
Tomedo/PostgreSQL database (the real per-practice extract). Schema = `data.loader.FEATURE_COLS` +
`LABEL_COL`; the combined file additionally carries `clinic_id` / `clinic_archetype` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import FEATURE_COLS, LABEL_COL

# ── The shared "true" CKD risk model (log-odds weights) ──────────────────────
# This is the biology a *global* model should learn; it is identical at every clinic. Age, diabetes
# and hypertension dominate (as in real CKD epidemiology). The intercept sets the base rate.
_BETA: dict[str, float] = {
    "intercept": -6.0,
    "age_years": 0.06,
    "dx_diabetes": 1.3,
    "dx_hypertonie": 0.9,
    "dx_khk": 0.5,
    "dx_herzinsuffizienz": 0.7,
    "dx_adipositas": 0.3,
    "dx_hyperurikaemie": 0.2,
    "years_since_diabetes_dx": 0.08,
    "years_since_hypertonie_dx": 0.05,
    "years_since_khk_dx": 0.04,
}


@dataclass(frozen=True)
class Archetype:
    """A clinic population profile — drives covariate & label shift (the non-IID part)."""

    name: str
    age_mean: float
    age_sd: float
    p_hypertonie: float
    p_diabetes: float
    p_khk: float
    p_adipositas: float
    p_herzinsuffizienz: float
    p_hyperurikaemie: float


# Clinically interpretable archetypes (CLAUDE.md §4), not a monotonic ramp.
ARCHETYPES: list[Archetype] = [
    Archetype("urban-young",   42, 14, 0.20, 0.08, 0.04, 0.18, 0.03, 0.10),
    Archetype("rural-elderly", 68, 12, 0.55, 0.28, 0.20, 0.25, 0.18, 0.22),
    Archetype("metabolic",     58, 13, 0.45, 0.40, 0.12, 0.40, 0.10, 0.30),
    Archetype("high-cvd",      64, 12, 0.50, 0.22, 0.30, 0.24, 0.22, 0.18),
    Archetype("mixed",         54, 18, 0.33, 0.16, 0.10, 0.22, 0.08, 0.16),
]


def _generate_patients(arch: Archetype, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One clinic's patients: features drawn from the archetype, label from the SHARED risk model."""
    age = np.clip(rng.normal(arch.age_mean, arch.age_sd, n), 18, 95)

    def flag(p: float) -> np.ndarray:
        return (rng.random(n) < p).astype(int)

    dx_hyp = flag(arch.p_hypertonie)
    dx_dia = flag(arch.p_diabetes)
    dx_khk = flag(arch.p_khk)
    dx_adi = flag(arch.p_adipositas)
    dx_hi = flag(arch.p_herzinsuffizienz)
    dx_hu = flag(arch.p_hyperurikaemie)

    def years(present: np.ndarray) -> np.ndarray:
        # years_since_* is 0 when the diagnosis is absent (paired with its dx_ flag, per §3a).
        return np.where(present == 1, rng.uniform(0, 19, n), 0.0)

    df = pd.DataFrame({
        "age_years": age,
        "dx_hypertonie": dx_hyp,
        "dx_diabetes": dx_dia,
        "dx_khk": dx_khk,
        "dx_adipositas": dx_adi,
        "dx_herzinsuffizienz": dx_hi,
        "dx_hyperurikaemie": dx_hu,
        "years_since_hypertonie_dx": years(dx_hyp),
        "years_since_diabetes_dx": years(dx_dia),
        "years_since_khk_dx": years(dx_khk),
    })

    # Shared true risk -> Bernoulli label. Same _BETA at every clinic = one global truth to learn.
    logit = np.full(n, _BETA["intercept"], dtype=float)
    for col, beta in _BETA.items():
        if col == "intercept":
            continue
        logit += beta * df[col].to_numpy(dtype=float)
    prob = 1.0 / (1.0 + np.exp(-logit))
    df[LABEL_COL] = (rng.random(n) < prob).astype(int)

    return df[[*FEATURE_COLS, LABEL_COL]]


def generate_clinics(
    num_clinics: int = 10, *, base_seed: int = 42, min_n: int = 120, max_n: int = 600,
    sizes: list[int] | None = None,
) -> list[tuple[int, str, pd.DataFrame]]:
    """Generate `num_clinics` practices as (clinic_id, archetype_name, DataFrame).

    Non-IID on all three axes from CLAUDE.md §4: archetype covariate shift, prevalence/label shift,
    and quantity shift (unequal panel sizes). One global seed → derived per-clinic seeds.
    `sizes` (Era 15 ladder, docs/ERAS.md erratum 5-a): an optional exact per-clinic size
    override — required because uniform band draws compress realized max/min to < 10× over
    ten clinics. When given it MUST have length `num_clinics`; it replaces only the size draw
    (per-clinic content seeds and the archetype cycle are untouched), and the default call is
    bit-identical to before.
    """
    rng = np.random.default_rng(base_seed)
    clinics = []
    for cid in range(num_clinics):
        arch = ARCHETYPES[cid % len(ARCHETYPES)]
        n = sizes[cid] if sizes is not None else int(rng.integers(min_n, max_n + 1))
        if sizes is not None and len(sizes) != num_clinics:
            raise ValueError(f"sizes has {len(sizes)} entries for {num_clinics} clinics")
        clinic_rng = np.random.default_rng(base_seed + 1000 + cid)  # derived per-clinic seed
        clinics.append((cid, arch.name, _generate_patients(arch, n, clinic_rng)))
    return clinics


# Seed offset for the public cohort. Distinct from every clinic's derived seed (base+1000+cid), so
# the public patients are guaranteed disjoint from every practice's panel.
PUBLIC_SEED_OFFSET = 90_000


def generate_public_cohort(
    n: int = 400, *, base_seed: int = 42, class_balanced: bool = True
) -> pd.DataFrame:
    """The shared **unlabelled** public cohort `U` that FedMosaic co-trains over.

    Built to the paper's specification (docs/2507.00259v3.pdf, "Experimental Setup"): a small,
    class-balanced sample drawn IID from the *global* training distribution and disjoint from every
    client dataset. "IID from the global distribution" here means drawn across all archetypes
    rather than from any single clinic's population, so no practice's patients are over-represented.

    The label column is returned so the harness can report an oracle diagnostic, but the protocol
    itself never reads it — clients only ever see `U`'s features (see `MosaicPractice`).
    """
    rng = np.random.default_rng(base_seed + PUBLIC_SEED_OFFSET)
    # Oversample, then trim: class-balancing discards rows, and CKD is the minority class.
    per_archetype = max(1, int(np.ceil(n * 4 / len(ARCHETYPES))))
    frames = [
        _generate_patients(arch, per_archetype, np.random.default_rng(
            base_seed + PUBLIC_SEED_OFFSET + i
        ))
        for i, arch in enumerate(ARCHETYPES)
    ]
    pool = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=base_seed)

    if not class_balanced:
        return pool.head(n).reset_index(drop=True)

    per_class = n // 2
    positives = pool[pool[LABEL_COL] == 1].head(per_class)
    negatives = pool[pool[LABEL_COL] == 0].head(n - per_class)
    balanced = pd.concat([positives, negatives], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(drop=True)


def combined_frame(clinics: list[tuple[int, str, pd.DataFrame]]) -> pd.DataFrame:
    """Stack all clinics into one frame with `clinic_id` / `clinic_archetype` columns."""
    frames = []
    for cid, name, df in clinics:
        tagged = df.copy()
        tagged.insert(0, "clinic_id", cid)
        tagged.insert(1, "clinic_archetype", name)
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True)


def write_clinic_csvs(
    clinics: list[tuple[int, str, pd.DataFrame]], out_dir: str | Path,
) -> Path:
    """Write one CSV per clinic (the per-practice `extract_features.sql` output) + a combined file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cid, name, df in clinics:
        df.to_csv(out / f"clinic_{cid:02d}_{name}.csv", index=False)
    combined_frame(clinics).to_csv(out / "all_clinics.csv", index=False)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic per-clinic CKD datasets")
    parser.add_argument("--clinics", type=int, default=10, help="number of clinics to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "clinics"),
        help="output directory for the per-clinic CSVs",
    )
    args = parser.parse_args()

    clinics = generate_clinics(args.clinics, base_seed=args.seed)
    out = write_clinic_csvs(clinics, args.out)
    print(f"Wrote {len(clinics)} clinics + all_clinics.csv to {out}")
    for cid, name, df in clinics:
        print(f"  clinic {cid:>2} [{name:<13}] n={len(df):<4} CKD rate={df[LABEL_COL].mean():.0%}")


if __name__ == "__main__":
    main()
