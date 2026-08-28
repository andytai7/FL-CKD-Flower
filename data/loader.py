"""Data loading + preprocessing for the synthetic CKD dataset.

Source of truth for the *real* data is `extract_features.sql`; this loader handles the
synthetic stand-in `synthetic_ckd_data.csv` (10-feature schema) and applies the missingness
rules documented in CLAUDE.md §3:

- Binary diagnostic flags (`dx_*`)  -> structural zero (absent == not documented). No imputation.
- `years_since_*`                   -> 0 encodes "diagnosis absent" (paired with its dx_ flag).
- Continuous labs (eGFR/HbA1c, real data only) -> median impute + a binary missing-indicator.

When the canonical SQL schema replaces the synthetic one, extend CONTINUOUS_LAB_COLS and the
missing-indicator logic below; the public API (`load_xy`) stays the same.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Synthetic schema (matches synthetic_ckd_data.csv). This is the 10-feature stand-in schema, NOT
# the canonical real-data contract in extract_features.sql — see CLAUDE.md §3 for the divergence.
BINARY_FLAG_COLS = [
    "dx_hypertonie",
    "dx_diabetes",
    "dx_khk",
    "dx_adipositas",
    "dx_herzinsuffizienz",
    "dx_hyperurikaemie",
]
YEARS_SINCE_COLS = [
    "years_since_hypertonie_dx",
    "years_since_diabetes_dx",
    "years_since_khk_dx",
]
CONTINUOUS_COLS = ["age_years", *YEARS_SINCE_COLS]

# Continuous lab columns exist only in the real (extract_features.sql) schema; empty for synthetic.
CONTINUOUS_LAB_COLS: list[str] = []

FEATURE_COLS = ["age_years", *BINARY_FLAG_COLS, *YEARS_SINCE_COLS]
LABEL_COL = "ckd_stage3plus"

NUM_FEATURES = len(FEATURE_COLS)  # 10 for the synthetic schema

# ── Canonical real-data schema (extract_features.sql §9) ─────────────────────
# The production FHIR preprocessor (data/fhir_loader.py) and the Tomedo SQL export both emit
# this contract: German column names, labs, `geschlecht`, and a CKD *incidence* label. It is a
# different prediction task from the synthetic schema above (CLAUDE.md §3b) — the two coexist
# deliberately, and `to_xy` dispatches on which label column a frame carries.
CANONICAL_META_COLS = ["t0"]  # constant Stichtag per extraction; never a feature
CANONICAL_BINARY_FLAG_COLS = ["dm", "aht", "cvd"]
CANONICAL_TAGE_COLS = ["tage_seit_dm_diagnose", "tage_seit_aht_diagnose", "tage_seit_cvd_diagnose"]
CANONICAL_LAB_COLS = ["egfr_letzter", "egfr_mittelwert_3", "hba1c_letzter", "hba1c_mittelwert_3"]

# X column order: 8 base features, then for each lab the median-imputed value followed by its
# missing-indicator (mirrors the synthetic lab hook below).
CANONICAL_FEATURE_COLS = [
    "alter_jahre",
    "geschlecht",
    *CANONICAL_BINARY_FLAG_COLS,
    *CANONICAL_TAGE_COLS,
]
CANONICAL_LABEL_COL = "ckd_incident"
CANONICAL_NUM_FEATURES = len(CANONICAL_FEATURE_COLS) + 2 * len(CANONICAL_LAB_COLS)  # 16

_HERE = Path(__file__).resolve().parent
DEFAULT_CSV = _HERE / "synthetic_ckd_data.csv"

# Per-clinic datasets written by `ckd-clinics` (data/synthesize.py) — the natural, non-simulated
# federation: one CSV per practice, each a stand-in for that clinic's extract_features.sql output.
CLINICS_DIR = _HERE / "clinics"


def load_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the CKD CSV and validate the expected columns are present."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df


def list_clinic_files(clinics_dir: str | Path | None = None) -> list[Path]:
    """Sorted per-clinic CSVs in `clinics_dir` (default data/clinics/); excludes all_clinics.csv."""
    directory = Path(clinics_dir) if clinics_dir is not None else CLINICS_DIR
    return sorted(directory.glob("clinic_*.csv"))


def load_clinic_frames(clinics_dir: str | Path | None = None) -> list[pd.DataFrame]:
    """Load every on-disk clinic as its own DataFrame — one natural practice per clinic."""
    files = list_clinic_files(clinics_dir)
    if not files:
        where = clinics_dir if clinics_dir is not None else CLINICS_DIR
        raise FileNotFoundError(
            f"No clinic CSVs found in {where}. Generate them first:  uv run ckd-clinics"
        )
    return [load_dataframe(path) for path in files]


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (possibly per-practice) dataframe to model-ready (X, y) arrays.

    The single preprocessing entry point for both schemas, dispatched on the label column:
    - canonical (`ckd_incident`, extract_features.sql §9) — `_canonical_to_xy`
    - synthetic (`ckd_stage3plus`) — the rules below

    Synthetic missingness: structural zero for flags / years_since; median + indicator for any
    continuous lab columns (none in the synthetic schema — canonical labs are handled in
    `_canonical_to_xy`).
    """
    if CANONICAL_LABEL_COL in df.columns:
        return _canonical_to_xy(df)
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Frame carries neither label {LABEL_COL!r} (synthetic) nor "
            f"{CANONICAL_LABEL_COL!r} (canonical) — cannot dispatch preprocessing."
        )

    df = df.copy()

    # Structural zeros: an absent flag or years_since means "not documented" -> 0.
    df[BINARY_FLAG_COLS] = df[BINARY_FLAG_COLS].fillna(0)
    df[YEARS_SINCE_COLS] = df[YEARS_SINCE_COLS].fillna(0)

    feature_frames = [df[FEATURE_COLS].astype("float32")]

    # Continuous labs (real data): median impute + binary missing-indicator (informative NaN).
    for col in CONTINUOUS_LAB_COLS:
        indicator = df[col].isna().astype("float32")
        imputed = df[col].fillna(df[col].median()).astype("float32")
        feature_frames.append(imputed.rename(col).to_frame())
        feature_frames.append(indicator.rename(f"{col}__missing").to_frame())

    X = pd.concat(feature_frames, axis=1).to_numpy(dtype="float32")
    y = df[LABEL_COL].to_numpy(dtype="int64")
    return X, y


def _canonical_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """extract_features.sql contract -> model-ready (X, y), per its §8/§9 preprocessing notes.

    - `dm` / `aht` / `cvd` flags — structural zeros, no imputation (same rule as synthetic).
    - `tage_seit_*` — SQL NULL encodes "diagnosis absent" and the paired flag already carries
      presence; §9 sanctions 0-imputation.
    - labs — median imputation **plus** a binary missing-indicator (CLAUDE.md §3a / SQL §8:
      a missing lab is itself informative). A lab column that is entirely missing at one
      practice imputes to 0.0 — a constant — with its indicator set on every row.
    - `t0` and any `patient_pseudonym` are metadata, never features.
    """
    df = df.copy()

    df[CANONICAL_BINARY_FLAG_COLS] = df[CANONICAL_BINARY_FLAG_COLS].fillna(0)
    df[CANONICAL_TAGE_COLS] = df[CANONICAL_TAGE_COLS].fillna(0)

    feature_frames = [df[CANONICAL_FEATURE_COLS].astype("float32")]
    for col in CANONICAL_LAB_COLS:
        indicator = df[col].isna().astype("float32")
        imputed = df[col].fillna(df[col].median()).fillna(0.0).astype("float32")
        feature_frames.append(imputed.rename(col).to_frame())
        feature_frames.append(indicator.rename(f"{col}__missing").to_frame())

    X = pd.concat(feature_frames, axis=1).to_numpy(dtype="float32")
    y = df[CANONICAL_LABEL_COL].to_numpy(dtype="int64")
    return X, y
