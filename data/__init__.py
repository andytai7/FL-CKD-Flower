"""Data loading, preprocessing, and non-IID partitioning for the CKD dataset.

Patient-level schemas share one preprocessing entry point (`to_xy` dispatches on the columns):
- synthetic (`FEATURE_COLS` + `LABEL_COL`)        — the baseline/evaluation task
- canonical (`CANONICAL_*`, extract_features.sql)  — the real-data task (CLAUDE.md §3b)
- KFRE (`KFRE_FEATURE_COLS` + `LABEL_COL`, `data/clinics_nhanes_kfre/`) — the Tangri rule-based
  baseline track (`kfre.py`); the same 8 inputs feed the logreg comparators

Three sources feed the synthetic schema:
- `load_dataframe` / `load_partition` — the flat synthetic CSV, split into simulated practices
- `load_clinic_frames`               — the per-clinic CSVs from `ckd-clinics`

The production path feeds the canonical schema:
- `load_practice_frame`              — one practice's live FHIR server, de-identified (L0)
"""

from .fhir_loader import load_practice_frame
from .loader import (
    CANONICAL_BINARY_FLAG_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_LABEL_COL,
    CANONICAL_LAB_COLS,
    CANONICAL_META_COLS,
    CANONICAL_NUM_FEATURES,
    CANONICAL_TAGE_COLS,
    CLINICS_DIR,
    FEATURE_COLS,
    KFRE_FEATURE_COLS,
    KFRE_LAB_COLS,
    KFRE_NUM_FEATURES,
    LABEL_COL,
    NUM_FEATURES,
    has_kfre_features,
    list_clinic_files,
    load_clinic_frames,
    load_dataframe,
    to_xy,
)
from .partition import load_partition

__all__ = [
    "FEATURE_COLS",
    "LABEL_COL",
    "NUM_FEATURES",
    "CANONICAL_BINARY_FLAG_COLS",
    "CANONICAL_FEATURE_COLS",
    "CANONICAL_LABEL_COL",
    "CANONICAL_LAB_COLS",
    "CANONICAL_META_COLS",
    "CANONICAL_NUM_FEATURES",
    "CANONICAL_TAGE_COLS",
    "CLINICS_DIR",
    "KFRE_FEATURE_COLS",
    "KFRE_LAB_COLS",
    "KFRE_NUM_FEATURES",
    "has_kfre_features",
    "load_dataframe",
    "to_xy",
    "load_partition",
    "list_clinic_files",
    "load_clinic_frames",
    "load_practice_frame",
]
