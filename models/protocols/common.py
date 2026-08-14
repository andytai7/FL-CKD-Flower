"""Shared logistic-regression machinery for the protocol benchmark (CLAUDE.md §4).

Two of the three protocols need explicit control of the local objective, which sklearn's
`SGDClassifier` does not expose:

- **FedProx** adds a proximal term `(μ/2)‖w − w_global‖²` to the local loss. Flower's `FedProx`
  strategy only ships `proximal-mu` in the config — implementing the term is the *client's* job.
- **FedMosaic** minimises `L(h, D_i) + α_i^t · L(h, P)` over local data plus a pseudo-labelled
  public set, with `α` recomputed every round from the two losses.

So the protocols share one small, explicit logistic regression here. To be precise about immutable
rule 1: the *optimiser math* below is ordinary logistic regression, not a Flower primitive. Every
piece of *federation* — client selection, messaging, aggregation — remains Flower's.

Running all protocols on this same local learner is what makes the benchmark fair (immutable rule
6): the data, splits, seeds and local optimiser are identical, and only the protocol varies.

Parameter layout is `w = [coef_ (d), intercept (1)]`, matching what `models/logreg.py` exchanges.
"""

from __future__ import annotations

import numpy as np

L2 = 1e-3  # ridge penalty; matches the `alpha` of models/logreg.py's SGDClassifier
_EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample balanced class weights (mirrors models/logreg.py)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.ones(len(y), dtype=np.float64)
    n = len(y)
    weight_map = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y], dtype=np.float64)


def init_weights(n_features: int) -> np.ndarray:
    """Zero-initialised `[coef_, intercept]`. Deterministic — no seed needed."""
    return np.zeros(n_features + 1, dtype=np.float64)


def to_flower_arrays(w: np.ndarray) -> list[np.ndarray]:
    """Split the packed vector into the `[coef_, intercept]` pair Flower exchanges."""
    return [w[:-1].astype(np.float32), w[-1:].astype(np.float32)]


def from_flower_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    coef, intercept = arrays
    return np.concatenate([
        np.asarray(coef, dtype=np.float64).ravel(),
        np.asarray(intercept, dtype=np.float64).ravel(),
    ])


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=np.float64) @ w[:-1] + w[-1])


def _grad(X: np.ndarray, y: np.ndarray, sw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """∇ of the sample-weighted mean log-loss + L2. Shape (d+1,)."""
    residual = (predict_proba(X, w) - y) * sw
    denom = max(float(sw.sum()), _EPS)
    grad = np.empty_like(w)
    grad[:-1] = X.T @ residual / denom + L2 * w[:-1]
    grad[-1] = residual.sum() / denom
    return grad


def _loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Unweighted mean cross-entropy — the quantity FedMosaic's α compares across datasets."""
    p = np.clip(predict_proba(X, w), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogRegLocal:
    """One practice's local logistic-regression objective over its own rows.

    Nothing here is ever transmitted. Only what a protocol explicitly returns leaves the practice:
    weights for FedAvg/FedProx, or public-set predictions for FedMosaic.
    """

    def __init__(self, X_train, y_train, X_test, y_test, *, class_weight_balanced: bool = True):
        self.X = np.asarray(X_train, dtype=np.float64)
        self.y = np.asarray(y_train, dtype=np.float64)
        self.X_test = np.asarray(X_test, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.sw = balanced_weights(self.y) if class_weight_balanced else np.ones(len(self.y))
        self.n_features = self.X.shape[1]

    @property
    def num_examples(self) -> int:
        return len(self.y)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return _grad(self.X, self.y, self.sw, w)

    def loss(self, w: np.ndarray) -> float:
        return _loss(self.X, self.y, w)

    def test_scores(self, w: np.ndarray) -> np.ndarray:
        return predict_proba(self.X_test, w)

    def local_sgd(
        self, w: np.ndarray, *, epochs: int, lr: float,
        proximal_mu: float = 0.0, w_global: np.ndarray | None = None,
    ) -> np.ndarray:
        """`epochs` full-batch steps, optionally with FedProx's proximal pull toward `w_global`.

        The proximal term `(μ/2)‖w − w_global‖²` contributes `μ(w − w_global)` to the gradient — it
        keeps a practice from wandering too far toward its own population's optimum during local
        training, which is FedProx's whole mechanism for surviving non-IID data.
        """
        w = w.copy()
        anchor = w.copy() if w_global is None else np.asarray(w_global, dtype=np.float64)
        for _ in range(epochs):
            step = self.gradient(w)
            if proximal_mu:
                step = step + proximal_mu * (w - anchor)
            w -= lr * step
        return w

    def train_mosaic(
        self, w: np.ndarray, *, epochs: int, lr: float,
        X_public: np.ndarray | None, y_pseudo: np.ndarray | None, alpha: float,
    ) -> np.ndarray:
        """Minimise `L(h, D_i) + α·L(h, P)` — FedMosaic's dynamically weighted local objective."""
        w = w.copy()
        if X_public is None or y_pseudo is None or alpha == 0.0:
            for _ in range(epochs):
                w -= lr * self.gradient(w)
            return w

        pub_sw = np.ones(len(y_pseudo), dtype=np.float64)
        for _ in range(epochs):
            step = self.gradient(w) + alpha * _grad(X_public, y_pseudo, pub_sw, w)
            w -= lr * step
        return w
