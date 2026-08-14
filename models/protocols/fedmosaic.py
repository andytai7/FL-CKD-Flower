"""FedMosaic — federated co-training with adaptivity and specialization.

Implements Algorithm 1 of the paper in `docs/2507.00259v3.pdf` ("Preprint. Under review."), adapted
to binary CKD screening (C = 2) and logistic regression.

**The base mechanism (federated co-training).** Nothing about a practice's model leaves the
building — no coefficients, no gradients. Every practice predicts on a *shared unlabelled public
cohort* `U`; the server turns those predictions into consensus pseudo-labels `L_t`; each practice
then trains on its own data plus the pseudo-labelled public set `P = (U, L_t)`. The paper motivates
this precisely because parameter sharing carries privacy risk (it cites Zhu et al. 2019 on gradient
leakage).

**The two things FedMosaic adds**, which are the reason it beats plain co-training:

1. *When to trust the global signal* — **dynamic loss weighting**. Each client minimises

       ℓ = ℓ_priv + α_i^t · ℓ_pseudo ,   α_i^t = exp( −(ℓ_pseudo − ℓ_priv) / ℓ_priv )

   so a practice whose local data conflicts with the consensus (ℓ_pseudo ≫ ℓ_priv) drives α → 0 and
   falls back on itself, while a practice for which the consensus is *cleaner* than its own noisy
   data gets α > 1 and leans in. For a small, skewed practice this is the difference between
   collaboration helping and actively hurting.

2. *Whose predictions to trust* — **confidence-based aggregation**. Alongside its one-hot
   predictions `L_i^t`, each client sends an **expertise vector** `E_i^t`, and the server forms

       S_t = Σ_i diag(E_i^t) · L_i^t ,   L_t[j] = argmax_c S_t[j, c]

   so a practice that is genuinely competent on a given patient profile outweighs one that is
   guessing. The paper gives two instantiations of `E`, both implemented here: a
   class-frequency heuristic and an uncertainty score from predictive entropy.

**Why it suits FLIP-IT.** It is a *personalized* FL method — every practice keeps its own model,
which is how a CKD risk score would actually be deployed in a practice. The paper's own headline
finding is that local training is a brutally strong baseline that most personalized-FL methods fail
to beat, so `--protocol local` is included in this benchmark as the honest comparator.

**Communication.** Per client per round the paper bounds the uplink at
`|U| · (⌈log₂ C⌉ + b_E)` bits. For binary CKD that is `|U| · (1 + b_E)` — with `b_E = 8`, nine bits
per public patient, versus 32 × 11 = 352 bits for the coefficient vector of our logistic regression.
Whether that wins depends entirely on `|U|`, and the report measures it rather than assuming it.
"""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from .common import LogRegLocal, predict_proba

NUM_CLASSES = 2  # binary CKD screening
EXPERTISE_KINDS = ("entropy", "class-frequency")


class FedMosaic(FedAvg):
    """Server side of FedMosaic: confidence-weighted consensus over client predictions.

    Subclasses `FedAvg` so Flower's own node sampling, message construction and reply validation
    are reused; only the aggregation rule is replaced (Algorithm 1, lines 14-18).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.consensus: np.ndarray | None = None
        self.last_agreement: float = float("nan")

    def summary(self) -> None:
        print("      FedMosaic | confidence-weighted consensus over a shared public cohort")

    def aggregate_train(
        self, server_round: int, replies
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """S_t = Σ_i diag(E_i) L_i ;  L_t[j] = argmax_c S_t[j, c]."""
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None

        score = None
        num_examples = 0.0
        for msg in valid:
            arrays = msg.content["arrays"].to_numpy_ndarrays()
            labels = np.asarray(arrays[0], dtype=np.float64).ravel()          # predicted class idx
            expertise = np.asarray(arrays[1], dtype=np.float64).ravel()       # E_i^t
            onehot = np.zeros((len(labels), NUM_CLASSES), dtype=np.float64)
            onehot[np.arange(len(labels)), labels.astype(int)] = 1.0

            contribution = expertise[:, None] * onehot
            score = contribution if score is None else score + contribution

            metrics = next(iter(msg.content.metric_records.values()))
            num_examples += float(metrics["num-examples"])

        self.consensus = np.argmax(score, axis=1).astype(np.float64)

        # Share of total expertise mass behind the winning label: 1.0 = unanimous federation.
        total = score.sum(axis=1)
        winning = score.max(axis=1)
        self.last_agreement = float(np.mean(winning / np.maximum(total, 1e-12)))

        return (
            ArrayRecord([self.consensus.astype(np.float32)]),
            MetricRecord({
                "num-examples": num_examples,
                "consensus-agreement": self.last_agreement,
                "consensus-positive-rate": float(self.consensus.mean()),
            }),
        )


def expertise_vector(
    probs: np.ndarray, kind: str, *, local_class_freq: np.ndarray | None = None
) -> np.ndarray:
    """The paper's two instantiations of `E_i^t` (§ "Confidence-Based Aggregation").

    - ``entropy``:         confidence = log C − H(p), in [0, log C]  (DP sensitivity c = log C)
    - ``class-frequency``: how well represented the predicted class is locally, in [0, 1] (c = 1)
    """
    if kind == "entropy":
        p = np.clip(probs, 1e-12, 1 - 1e-12)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return np.log(NUM_CLASSES) - entropy
    if kind == "class-frequency":
        if local_class_freq is None:
            raise ValueError("class-frequency expertise needs the local class frequencies")
        predicted = (probs >= 0.5).astype(int)
        return local_class_freq[predicted]
    raise ValueError(f"Unknown expertise kind {kind!r}; choose from {EXPERTISE_KINDS}")


class MosaicPractice:
    """One practice's FedMosaic client (Algorithm 1, lines 2-13). Keeps its OWN model, always."""

    def __init__(
        self, local: LogRegLocal, X_public: np.ndarray, *,
        lr: float, epochs: int, expertise: str = "entropy",
    ):
        self.local = local
        self.X_public = np.asarray(X_public, dtype=np.float64)
        self.lr = lr
        self.epochs = epochs
        self.expertise_kind = expertise
        self.w = np.zeros(local.n_features + 1, dtype=np.float64)
        self.pseudo: np.ndarray | None = None   # P = (U, L_t) once the first consensus arrives
        self.alpha: float = 0.0

        freq = np.array([1.0 - local.y.mean(), local.y.mean()], dtype=np.float64)
        self.local_class_freq = np.clip(freq, 1e-6, 1.0)

    @property
    def num_examples(self) -> int:
        return self.local.num_examples

    def local_step(self) -> float:
        """Lines 3-7: compute α from the two losses, then train on the combined objective."""
        loss_priv = max(self.local.loss(self.w), 1e-12)
        if self.pseudo is None:
            self.alpha = 0.0
        else:
            from .common import _loss
            loss_pseudo = _loss(self.X_public, self.pseudo, self.w)
            # α = exp( −(ℓ_pseudo − ℓ_priv) / ℓ_priv ), clipped so one pathological round cannot
            # blow the local objective up; α ≤ e is already "trust the consensus more than myself".
            self.alpha = float(np.clip(np.exp(-(loss_pseudo - loss_priv) / loss_priv), 0.0, np.e))

        self.w = self.local.train_mosaic(
            self.w, epochs=self.epochs, lr=self.lr,
            X_public=self.X_public, y_pseudo=self.pseudo, alpha=self.alpha,
        )
        return self.alpha

    def share(self) -> tuple[np.ndarray, np.ndarray]:
        """Lines 9-10: the one-hot prediction matrix and the expertise vector this client sends."""
        probs = predict_proba(self.X_public, self.w)
        labels = (probs >= 0.5).astype(np.float64)
        expertise = expertise_vector(
            probs, self.expertise_kind, local_class_freq=self.local_class_freq
        )
        return labels, expertise

    def adopt(self, consensus: np.ndarray) -> None:
        """Line 12: P ← (U, L_t)."""
        self.pseudo = np.asarray(consensus, dtype=np.float64)

    def test_scores(self) -> np.ndarray:
        return self.local.test_scores(self.w)


def build_reply(labels: np.ndarray, expertise: np.ndarray, num_examples: int) -> Message:
    """A FedMosaic train reply: (L_i^t, E_i^t) — never a parameter vector."""
    content = RecordDict({
        "arrays": ArrayRecord([labels.astype(np.float32), expertise.astype(np.float32)]),
        "metrics": MetricRecord({"num-examples": num_examples}),
    })
    return Message(content=content, message_type=MessageType.TRAIN, dst_node_id=1)


def uplink_bits(num_public: int, expertise_bits: int = 8) -> int:
    """The paper's per-client, per-round uplink budget: |U| · (⌈log₂ C⌉ + b_E)."""
    return num_public * (int(np.ceil(np.log2(NUM_CLASSES))) + expertise_bits)
