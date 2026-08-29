# Physician demo webapp (`ckd-web`)

The smallest end-to-end cut of the "SuperLink ships the model, the practice serves it" deployment
story: the federated global model is exported once, and a tiny stdlib-only server on the practice
machine lets a physician enter patient features and read back the model's **P(CKD stage ≥ 3)**.

> ⚠️ **Research demo on synthetic data. Not a medical device; not for clinical decisions.**
> The form collects no identifiers and stores nothing.

## Quick start

```bash
uv run ckd-export-model     # 10 rounds of Flower FedAvg over data/clinics -> models/global_model.json
uv run ckd-web --port 8765  # http://127.0.0.1:8765
```

Flags: `--host` (default 127.0.0.1), `--port` (default 8080), `--model <path>` (default
`models/global_model.json`). If the artifact is missing the server exits with the export command
printed.

## How it fits the federation

```
 training run:   flwr run .  →  Flower FedAvg aggregates [coef_, intercept_] across practices
 export:         ckd-export-model  →  models/global_model.json
                                         (weights + feature schema + reference scaler + metrics)
 serve locally:  ckd-web  →  Scorer.from_file(...)  →  form → sigmoid risk score
```

- Only **weights** ever cross the federation boundary during training (project rule 3); the webapp
  evaluates the exported global model **locally** — the practice's patient features never leave the
  box.
- The artifact's reference scaler is fit on the pooled *synthetic* cohort as a sandbox stand-in. A
  real deployment pairs the model with the practice-local scaler instead (see
  `model_artifact.py` docstring).
- In the pod, the app is reached through code-server's port proxy:
  `https://kubeflow.kite.ume.de/notebook/man-tai/workstation/proxy/<port>/`
  (trailing slash matters — all app links are relative so they survive the prefix).

## Interface

- `GET /` — patient-details form: `age_years`, six diagnosis checkboxes
  (I10, E10/E11/E13, I25, E66, I50/I11.0, M10), three `years_since_*` inputs.
- `POST /score` — form-encoded body of the same field names → rendered result page.
  Unchecked diagnoses are `0`; a `years_since_*` is forced to `0` when its diagnosis is unticked
  (structural-zero semantics, CLAUDE.md §3). Missing/empty years fields default to `0`;
  non-numeric or out-of-range (0–120) values → `400` with a readable message.
- The result shows the risk percentage, an **illustrative** band
  (<0.20 lower · <0.50 intermediate · ≥0.50 elevated) and an echo of the inputs.

## Screenshots

`diagrams/webapp-demo/01_patient_form.png` (the form) and
`diagrams/webapp-demo/02_risk_score_result.png` (a 68-year-old with 15 years of hypertension and
12 of diabetes → 99.3%, band: elevated).

## Files

| Path | Role |
|---|---|
| `webapp.py` | The server (`ckd-web`): form rendering, input validation, scoring |
| `model_artifact.py` | `ckd-export-model` training/export + the `Scorer` used by the webapp |
| `models/global_model.json` | The exported federated model artifact (generate with `ckd-export-model`) |
