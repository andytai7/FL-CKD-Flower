# CKD Federated Learning — Pre-Kickoff Baseline

Federated **chronic-kidney-disease (CKD) risk models** across simulated GP practices, trained with
[Flower.ai](https://flower.ai). The model travels to the data — only weights are exchanged, patient
rows never leave a practice. This repo is the **pre-kickoff baseline sandbox** for the FLIP-IT
project: it establishes reference metrics and a working federated pipeline on a **synthetic**
stand-in dataset before the real HL7-FHIR practice data is available. See [CLAUDE.md](CLAUDE.md)
for the full project background.

**Every model is Flower-federatable** — the project deploys federated, so a model that can't run with
Flower isn't a candidate (see [CLAUDE.md](CLAUDE.md) §0). That's exactly three architectures, each
driven by a real `flwr` strategy:

| Architecture | Flower strategy | Role |
|---|---|---|
| **Logistic regression** (SGD, warm-start) | `FedAvg` | reference model (grant task T2.2) |
| **Small MLP** (sklearn, torch-free) | `FedAvg` | linear-vs-non-linear comparison |
| **XGBoost** | `FedXgbBagging` | gradient-boosted trees, federated by bagging |

`logreg`/`mlp` exchange weight vectors (FedAvg averages them); XGBoost can't be averaged, so Flower
federates it by **bagging** — each practice grows a few local trees and the server appends them into
one global ensemble (`flwr.server.strategy.FedXgbBagging`). Non-federatable models (random forest,
plain GBT, LightGBM) are intentionally **not** included. On macOS, XGBoost needs OpenMP at runtime:
`brew install libomp`.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). No manual venv activation needed.

```bash
make setup          # create/repair the venv + install all dependencies
make baseline       # centralized pooled-data ceiling (logreg + mlp + xgboost)
make simulate       # federated simulation: logistic regression, 12 practices, non-IID
make notebook       # run the exploration notebook headless
make help           # list every target
```

Don't have `make`? The same commands live as shell scripts in [`scripts/`](scripts/) (run from
anywhere — each `cd`s to the repo root first):

```bash
./scripts/setup.sh
./scripts/baseline.sh
./scripts/simulate.sh
./scripts/notebook.sh
```

And under the hood every script/target is just a `uv` command, if you'd rather type them directly:

```bash
uv sync --extra dev --extra notebook        # = make setup / scripts/setup.sh

uv run ckd-baseline --model all             # centralized ceiling: logreg | mlp | xgboost | all
uv run ckd-baseline --model xgboost

uv run ckd-simulate                         # logreg via FedAvg, 12 practices, 20 rounds, non-IID
uv run ckd-simulate --model mlp --rounds 10
uv run ckd-simulate --model xgboost         # federated trees via FedXgbBagging
uv run ckd-simulate --iid                   # IID split for comparison
uv run ckd-simulate --practices 25 --alpha 0.1   # 25 practices, strongly non-IID
uv run ckd-simulate --help                  # all flags

uv run ckd-clinics --clinics 10             # generate synthetic per-clinic datasets -> data/clinics/
uv run ckd-simulate --clinics               # federate over the on-disk clinics (one practice per CSV)
uv run ckd-simulate --clinics --model xgboost
```

> The scripts pass extra flags straight through, e.g. `./scripts/simulate.sh --model mlp --rounds 10`.

### Notebook

An interactive walkthrough that **simulates the FLIP-IT federation** — many synthetic clinics, a
single-clinic model that doesn't generalize, then Flower (FedAvg + FedXgbBagging) recovering ~the
centralized ceiling without sharing data — lives in
[`notebooks/01_explore_and_baselines.ipynb`](notebooks/01_explore_and_baselines.ipynb).

```bash
make lab            # open it in JupyterLab
# …or open the .ipynb in VS Code and pick the .venv interpreter as the kernel.
```

---

## How the federation runs locally

> ⚠️ **Use `uv run ckd-simulate`, not `flwr run .`, on this machine.** Flower's `flwr run` uses the
> **Ray** engine, which cannot handle a project path that contains a **space**
> (`.../flipit/FL CKD Flower/…`) — Ray's worker launcher splits the path at the space and crashes.
> `ckd-simulate` is still **genuine Flower**: it drives Flower's own strategies in-process —
> `flwr.server.strategy.FedAvg` for `logreg`/`mlp` (with the real `flwr.client.NumPyClient`) and
> `flwr.server.strategy.FedXgbBagging` for `xgboost` — calling their `aggregate_fit` /
> `aggregate_evaluate`, the exact aggregation `flwr run` would perform, just without the Ray
> transport layer. To use the full Ray engine and the real `ServerApp` / `ClientApp` end to end,
> point the venv at a space-free path (`export UV_PROJECT_ENVIRONMENT=/tmp/ckd_venv && uv sync &&
> uv run flwr run .`) — verified working (see CLAUDE.md §6).

```text
[Practice 1]  ─┐
[Practice 2]  ─┤   local train (warm-start)        ┌─► global AUROC + sensitivity
   …           ├──────────────────────────────────►├
[Practice 12] ─┘   weights only ─► FedAvg ─► global └─► worst-practice AUROC + sensitivity
```

Each round logs **dual-level metrics** (CLAUDE.md §5): the sample-weighted **global** mean *and*
the **worst single practice**, so a model that collapses on one outlier practice can't hide behind
a strong aggregate.

---

## Repository layout

| Path | Purpose |
|---|---|
| `data/` | Loading + §3 missingness rules, non-IID **Dirichlet partitioning**, and `synthesize.py` — the per-clinic generator (`ckd-clinics` → `data/clinics/`). |
| `models/` | `base.py` (FedAvg interface) + `logreg.py`, `mlp.py` (FedAvg) + `fedxgb.py` (federated XGBoost via FedXgbBagging). |
| `task.py` | Local `StandardScaler` + the imbalanced-data metrics (AUROC, sensitivity, accuracy). |
| `client_app.py` / `server_app.py` | Flower `ClientApp` / `ServerApp` (FedAvg + dual-level metrics) for `flwr run`. |
| `simulate.py` | **Ray-free** in-process federated runner → `uv run ckd-simulate`. |
| `centralized.py` | Pooled-data ceiling baselines → `uv run ckd-baseline`. |
| `notebooks/` | The FLIP-IT federation simulation notebook (clinics → single-clinic → federated). |
| `data/synthetic_ckd_data.csv` | 2000 synthetic patients, 10 features + label (signal-less placeholder). |
| `data/clinics/` | Generated per-clinic datasets (one CSV per practice = a simulated `extract_features.sql` extract). |
| `extract_features.sql` | Canonical feature contract for the **real** Tomedo→PostgreSQL export (run once per practice). |
| `pyproject.toml` | uv project: dependencies, `ckd-simulate`/`ckd-baseline`/`ckd-clinics` scripts, Flower config. |

---

## Features (synthetic baseline schema)

| Feature | Type | Description |
|---|---|---|
| `age_years` | continuous | age at the index date |
| `dx_hypertonie` | binary | ICD I10 present |
| `dx_diabetes` | binary | ICD E10/E11/E13 |
| `dx_khk` | binary | ICD I25 (coronary heart disease) |
| `dx_adipositas` | binary | ICD E66 |
| `dx_herzinsuffizienz` | binary | ICD I50/I11.0 |
| `dx_hyperurikaemie` | binary | ICD M10 |
| `years_since_hypertonie_dx` | continuous | years since first diagnosis (0 if absent) |
| `years_since_diabetes_dx` | continuous | years since first diagnosis (0 if absent) |
| `years_since_khk_dx` | continuous | years since first diagnosis (0 if absent) |

**Label** `ckd_stage3plus = 1` for CKD stage ≥ 3 (ICD N18.3–N18.6, or ≥ 2 eGFR < 60 more than 90
days apart). **Missingness:** `dx_*` flags are structural zeros (absent = not documented, no
imputation); `years_since_*` already encode absence as 0. Continuous labs (eGFR/HbA1c, real data
only) will be median-imputed **+** carry a binary missing-indicator.

---

## ⚠️ About the numbers you'll see

The current synthetic CSV is a **wiring placeholder with no learnable signal** — every feature's
correlation with the label is < 0.03 and the classes are ~52/48 (not a realistic low-prevalence
CKD cohort). So **all models score AUROC ≈ 0.5 (chance)**, and that is *correct* behaviour, not a
bug — the notebook's step 2 shows exactly why. The pipeline, partitioning, FedAvg, and metrics are
all verified end-to-end; the expected performance **lift comes with the real data**, whose biggest
addition is **eGFR — the single strongest CKD predictor**, currently missing from this schema.

Once real data lands, migrate the schema to the `extract_features.sql` contract (adds
eGFR/HbA1c/sex). The README's old target of *sensitivity ~0.6–0.75* applies to that real-data run,
**not** to today's synthetic noise.

---

## Configuration

Federated defaults live in `pyproject.toml` under `[tool.flwr.app.config]` (rounds, model,
`num-practices`, Dirichlet `alpha`, `class-weight-balanced`, `seed`) and are overridable per-run
via the `ckd-simulate` flags above. Metrics prioritize **AUROC** and **sensitivity/recall** over
raw accuracy, because a model that never predicts CKD can still look "accurate" on an imbalanced
cohort.
