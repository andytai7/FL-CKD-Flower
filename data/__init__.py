"""Data loading, preprocessing, and non-IID partitioning for the synthetic CKD dataset."""

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
]
 