"""Export and load the federated global CKD model as a single JSON artifact.

This is the "model shared via federated learning" made portable: the ServerApp-side global
``[coef_, intercept_]`` after N rounds of Flower FedAvg over the per-clinic SuperNodes, plus a
reference ``StandardScaler`` and the feature contract — everything a local scoring surface (see
``webapp.py``) needs, without any Flower machinery at inference time.

Training uses Flower's **real** FedAvg (``simulate.fedavg`` → ``FedAvg.aggregate_train``), per
project rule 1. Only weights are ever aggregated; each practice's rows stay in
``build_client_from_frame`` (rule 3).

⚠️ Sandbox status: the model is trained on the *synthetic* per-clinic data (CLAUDE.md §3a) and the
reference scaler is fit on the pooled synthetic cohort as a stand-in — a real deployment serves the
model together with the practice-local scaler, on the practice box. This artifact is a demo object,
not a medical device.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from client_app import build_client_from_frame
from data.loader import FEATURE_COLS, NUM_FEATURES, load_clinic_frames, to_xy
from models import LogRegModel
from simulate import fedavg
from task import fit_scaler

ARTIFACT_PATH = Path(__file__).resolve().parent / "models" / "global_model.json"

# Mirror of [tool.flwr.app.config] training defaults (see simulate.DEFAULT_CONFIG).
TRAIN_CONFIG = {"seed": 42, "class-weight-balanced": True, "local-epochs": 2}

# Schema semantics (CLAUDE.md §3): a years_since_* is meaningful only when its dx_ flag is set.
_FLAG_FOR_YEARS = {
    "years_since_hypertonie_dx": "dx_hypertonie",
    "years_since_diabetes_dx": "dx_diabetes",
    "years_since_khk_dx": "dx_khk",
}


def train_global_model(rounds: int = 10, seed: int = 42) -> tuple[list[np.ndarray], dict]:
    """Train the federated global logreg over the on-disk clinics; return (weights, metrics).

    Weights are the FedAvg-aggregated ``[coef_, intercept_]``; metrics are the dual-level numbers
    project rule 5 requires on the final round: global sample-weighted **and** worst-practice.
    """
    config = {**TRAIN_CONFIG, "seed": seed}
    frames = load_clinic_frames()
    clients = [build_client_from_frame(df, pid, config) for pid, df in enumerate(frames)]

    model = LogRegModel(
        class_weight_balanced=bool(config["class-weight-balanced"]), seed=seed
    )
    model.initialize(NUM_FEATURES)
    ndarrays = model.get_parameters()

    for _ in range(rounds):
        updates = [client.fit(ndarrays) for client in clients]  # (params, num-examples)
        ndarrays = fedavg(updates)  # Flower's real FedAvg strategy

    per_client = []
    for client in clients:
        metrics, n = client.evaluate(ndarrays)
        per_client.append((metrics, n))

    total = sum(n for _, n in per_client)
    global_auc = sum(m["auc"] * n for m, n in per_client) / total
    global_sens = sum(m["sensitivity"] * n for m, n in per_client) / total
    metrics = {
        "global_auc": float(global_auc),
        "worst_practice_auc": float(min(m["auc"] for m, _ in per_client)),
        "global_sensitivity": float(global_sens),
        "worst_practice_sensitivity": float(min(m["sensitivity"] for m, _ in per_client)),
        "practices": len(clients),
        "patients_evaluated": int(total),
    }
    return ndarrays, metrics


def export_artifact(
    rounds: int = 10, seed: int = 42, path: Path = ARTIFACT_PATH
) -> dict:
    """Train, attach the schema + reference scaler + metrics, and write the JSON artifact."""
    ndarrays, metrics = train_global_model(rounds=rounds, seed=seed)
    coef, intercept = ndarrays

    # Reference scaler for the demo surface: fit on the pooled synthetic cohort (the labelled
    # centralized-reference role this sandbox already uses for ceilings — synthetic data only).
    frames = load_clinic_frames()
    X_all = np.vstack([to_xy(df)[0] for df in frames]).astype("float64")
    scaler = fit_scaler(X_all)

    artifact = {
        "format": "ckd-fl global model artifact v1",
        "exported": date.today().isoformat(),
        "model": "logreg",
        "label": "ckd_stage3plus",
        "task": "P(CKD stage >= 3) — synthetic prevalence task (CLAUDE.md §3a)",
        "feature_cols": FEATURE_COLS,
        "n_features": NUM_FEATURES,
        "coef": [float(v) for v in coef],
        "intercept": [float(v) for v in intercept],
        "scaler": {
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        },
        "training": {
            "strategy": "Flower FedAvg (flwr 1.33)",
            "rounds": rounds,
            "seed": seed,
            "data": "data/clinics/*.csv — synthetic per-clinic stand-in data",
        },
        "metrics": metrics,
        "warning": (
            "Trained on synthetic data for the FLIP-IT pre-kickoff sandbox. "
            "Not a medical device; not for clinical decisions."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    return artifact


class Scorer:
    """One-patient-at-a-time scorer loaded from an exported artifact (no Flower needed)."""

    def __init__(self, artifact: dict):
        self.feature_cols: list[str] = artifact["feature_cols"]
        self._mean = np.asarray(artifact["scaler"]["mean"], dtype="float64")
        self._scale = np.asarray(artifact["scaler"]["scale"], dtype="float64")
        self._model = LogRegModel()
        self._model.initialize(int(artifact["n_features"]))
        self._model.set_parameters(
            [
                np.asarray(artifact["coef"], dtype="float32"),
                np.asarray(artifact["intercept"], dtype="float32"),
            ]
        )

    @classmethod
    def from_file(cls, path: Path = ARTIFACT_PATH) -> "Scorer":
        if not path.exists():
            raise FileNotFoundError(
                f"No exported model at {path}. Train and export first:  uv run ckd-export-model"
            )
        return cls(json.loads(path.read_text()))

    def score_one(self, raw: dict[str, float]) -> float:
        """Score one patient from raw form values keyed by feature name.

        Missing features default to 0 (structural-zero rules, CLAUDE.md §3); years_since_* is
        forced to 0 when its paired dx_ flag is 0.
        """
        row = {name: float(raw.get(name, 0.0) or 0.0) for name in self.feature_cols}
        for years_col, flag_col in _FLAG_FOR_YEARS.items():
            if not row.get(flag_col):
                row[years_col] = 0.0
        x = np.asarray([row[name] for name in self.feature_cols], dtype="float64")
        x_scaled = (x - self._mean) / self._scale
        return float(self._model.predict_proba(x_scaled[None, :])[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the federated global model and export the webapp artifact."
    )
    parser.add_argument("--rounds", type=int, default=10, help="federation rounds (converged by ~10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()

    artifact = export_artifact(rounds=args.rounds, seed=args.seed, path=args.output)
    m = artifact["metrics"]
    print(f"Exported {args.output}")
    print(
        f"  global AUROC {m['global_auc']:.3f} | worst practice {m['worst_practice_auc']:.3f} "
        f"| sensitivity {m['global_sensitivity']:.3f} "
        f"({m['practices']} practices, {m['patients_evaluated']} held-out patients)"
    )


if __name__ == "__main__":
    main()
