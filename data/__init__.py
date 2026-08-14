"""Data loading, preprocessing, and non-IID partitioning for the CKD dataset.

Three sources, one schema (`FEATURE_COLS` + `LABEL_COL`), so everything downstream is unchanged:
- `load_dataframe` / `load_partition` — the flat synthetic CSV, split into simulated practices
- `load_clinic_frames`               — the per-clinic CSVs from `ckd-clinics`
- `load_practice_frame`              — one practice's live FHIR server (the production path)
"""

from .fhir_loader import load_practice_frame
from .loader import (
    CLINICS_DIR,
    FEATURE_COLS,
    LABEL_COL,
    NUM_FEATURES,
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
    "CLINICS_DIR",
    "load_dataframe",
    "to_xy",
    "load_partition",
    "list_clinic_files",
    "load_clinic_frames",
    "load_practice_frame",
]
 