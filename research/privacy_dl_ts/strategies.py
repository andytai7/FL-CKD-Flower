"""Rule-8 strategy subclasses for the TS benchmark research tree.

FedAvgM: server momentum on the aggregated update — the standard instrument for the
non-convex drift measured on this track (FedAvg learns early, drifts mid-run as client
weights stop aligning; FedProx under this track's gradient magnitudes over-suppresses
learning entirely — see the gate evidence in the track commit history). Aggregation stays
Flower's real `FedAvg.aggregate_train`; the subclass only post-processes the returned
aggregate with the momentum recurrence — pure server momentum, Hsu et al. 2019.
"""

from __future__ import annotations

import numpy as np
from flwr.serverapp.strategy import FedAvg


class FedAvgM(FedAvg):
    """FedAvg + server momentum: v ← βv + (avg − θ_prev); θ ← θ_prev + v (β=0.6 typical)."""

    def __init__(self, *args, beta: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self._prev: list[np.ndarray] | None = None
        self._velocity: list[np.ndarray] | None = None

    def aggregate_train(self, server_round: int, replies):  # noqa: D401 (flwr signature)
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None:
            return None, metrics
        base = arrays.to_numpy_ndarrays()
        if self._prev is None:
            self._prev = [a.copy() for a in base]
            self._velocity = [np.zeros_like(a) for a in base]
            return arrays, metrics
        updates = []
        for b, p, v in zip(base, self._prev, self._velocity, strict=True):
            v_new = self.beta * v + (b - p)
            updates.append(p + v_new)
            v[:] = v_new
            p[:] = p + v_new
        from flwr.common import ArrayRecord  # local import: keeps musl import cost scoped

        return ArrayRecord(updates), metrics
