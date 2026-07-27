"""Non-IID partitioning of the flat synthetic CSV into N simulated practices.

`synthetic_ckd_data.csv` has no practice column, so the federation is simulated by partitioning.
A naive even random split is IID and unrealistic. We use **Dirichlet label partitioning** — the
standard mechanism for controllable non-IID splits — which produces both *label shift* (different
CKD prevalence per practice) and *quantity shift* (unequal panel sizes), tuned by a single
concentration knob `alpha`:

    alpha -> 0    strongly non-IID (each practice skewed toward few classes / sizes)
    alpha -> inf  IID (even, balanced split)

This realizes the CLAUDE.md §4 "label/prior shift" and "quantity shift" knobs. The richer
archetype / covariate-shift scheme (urban-young, rural-elderly, ...) is a documented future
refinement; Dirichlet label partitioning is the robust, well-understood baseline.

Determinism: a single `seed` plus the practice's `partition_id` fully determines its rows, so the
server and every client agree on the split without sharing data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loader import LABEL_COL, load_dataframe


def dirichlet_partition(
    df: pd.DataFrame,
    num_partitions: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Return a list of row-index arrays, one per partition, via Dirichlet label partitioning."""
    rng = np.random.default_rng(seed)
    labels = df[LABEL_COL].to_numpy()
    part_indices: list[list[int]] = [[] for _ in range(num_partitions)]

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        # Proportion of this class assigned to each practice.
        proportions = rng.dirichlet(alpha * np.ones(num_partitions))
        cut_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for p, chunk in enumerate(np.split(cls_idx, cut_points)):
            part_indices[p].extend(chunk.tolist())

    return [np.array(sorted(idx), dtype=int) for idx in part_indices]


def iid_partition(df: pd.DataFrame, num_partitions: int, seed: int) -> list[np.ndarray]:
    """Even, shuffled split (IID baseline)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    return [np.array(sorted(c), dtype=int) for c in np.array_split(idx, num_partitions)]


def load_partition(
    partition_id: int,
    num_partitions: int,
    *,
    alpha: float = 0.5,
    iid: bool = False,
    seed: int = 42,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the rows belonging to one practice (`partition_id`) out of `num_partitions`."""
    df = load_dataframe(csv_path)
    parts = (
        iid_partition(df, num_partitions, seed)
        if iid
        else dirichlet_partition(df, num_partitions, alpha, seed)
    )
    return df.iloc[parts[partition_id]].reset_index(drop=True)
