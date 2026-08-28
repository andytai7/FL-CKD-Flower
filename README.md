# CKD Federated Learning — Pre-Kickoff Baseline

Federated **chronic-kidney-disease (CKD) risk models** across simulated GP practices, trained with
[Flower.ai](https://flower.ai). The model travels to the data — only weights (or, for FedMosaic,
predictions on a public cohort) are exchanged; patient rows never leave a practice.

This repo is the **pre-kickoff baseline sandbox** for the **FLIP-IT** project (NEXT.IN.NRW /
EFRE-JTF NRW): it establishes reference metrics and a working federated pipeline on **synthetic**
data before the real HL7-FHIR practice data is available. See [CLAUDE.md](CLAUDE.md) for the full
project background and the immutable project rules.

**Every model is Flower-federatable** — the project deploys federated, so a model that can't run
with Flower isn't a candidate ([CLAUDE.md §0](CLAUDE.md) rule 2).

| Architecture | Flower strategy | Role |
|---|---|---|
| **Logistic regression** (SGD, warm-start) | `FedAvg` / `FedProx` | reference model (grant task T2.2) |
| **Small MLP** (sklearn, torch-free) | `FedAvg` | linear-vs-non-linear comparison |
| **XGBoost** | `FedXgbBagging` | gradient-boosted trees, federated by bagging |

---

## Results at a glance

Full write-up: **[docs/REPORT.md](docs/REPORT.md)** · privacy architecture:
**[docs/PRIVACY.md](docs/PRIVACY.md)** · deployment: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**

**The answer to counsel's questions is the Technical Expert Report in [`paper/`](paper/)** — one
section per question, built from a single pinned run (`./scripts/paper.sh`). It supersedes
`docs/LEGAL-TECHNICAL-ANSWER.md`, which is retained but marked superseded.

The reproducible experiment, with charts, is
[`notebooks/01_explore_and_baselines.ipynb`](notebooks/01_explore_and_baselines.ipynb).

10 non-IID practices, 3,276 patients, 20 rounds, **5 seeds** (mean ± s.d.).

| | AUROC | Worst practice | Uplink bits/round |
|---|---|---|---|
| *centralized ceiling* | *0.861* | *n/a* | *n/a* |
| **FedProx** | **0.808 ± 0.009** | **0.675 ± 0.060** | 352 |
| **FedMosaic** | 0.800 ± 0.011 | 0.661 ± 0.076 | 3,600 |
| **FedXgbBagging** | 0.729 ± 0.021 | 0.555 ± 0.050 | 20,902 |
| *fedavg* (baseline) | *0.808 ± 0.009* | *0.675 ± 0.061* | *352* |
| *local* (baseline) | *0.789 ± 0.009* | *0.495 ± 0.141* | *0* |

Four things worth knowing before you read further:

- **Federation lands within 0.054 of the pooled ceiling** without any practice sharing a patient row.
- **The win is equity, not average accuracy.** Global AUROC separates training-alone from the best
  protocol by 0.019; the *worst practice* separates them by **+0.180**.
- **More rounds do not help.** The logistic protocols converge by round 10; XGBoost actively
  *degrades* with more rounds (0.780 at round 5 → 0.686 at round 50).
- **Differential privacy has a deployment standard**: patient-level DP-SGD inside each clinic,
  SecAgg+ masking on the wire, and a rule-based server agent ([`dpsgd.py`](dpsgd.py) +
  [`orchestrator.py`](orchestrator.py)) — deterministic census → ε-policy → per-clinic accountant
  inversion → ConfigRecord dispatch; no LLM — that standardises the composed ε across clinic sizes.
  Measured in [`notebooks/03_dpsgd_secagg_standard.ipynb`](notebooks/03_dpsgd_secagg_standard.ipynb):
  AUROC 0.797–0.799 holds at composed ε 0.5–8 over 5 seeds. Central and local DP remain as measured
  comparison baselines in `privacy.py` / notebook 02.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/). No manual venv activation needed.

```bash
make setup          # create/repair the venv + install all dependencies
make clinics        # generate the per-clinic datasets (needed by everything below)
make baseline       # centralized pooled-data ceiling
make simulate       # federated simulation: logistic regression, non-IID
make help           # list every target
```

No `make`? The same commands are shell scripts in [`scripts/`](scripts/) — run them from anywhere,
each `cd`s to the repo root first:

```bash
./scripts/setup.sh && ./scripts/clinics.sh && ./scripts/simulate.sh --clinics
```

Under the hood every target is a `uv` command:

```bash
uv sync --extra dev --extra notebook

uv run ckd-clinics --clinics 10             # per-clinic datasets -> data/clinics/

uv run ckd-simulate --clinics               # logreg via FedAvg, non-IID
uv run ckd-simulate --clinics --model xgboost
uv run ckd-simulate --clinics --protocol fedmosaic
uv run ckd-simulate --help                  # all flags

uv run ckd-baseline --model all --clinics   # pooled ceiling on the SAME data
uv run ckd-benchmark --rounds 20            # full protocol benchmark -> results/
uv run ckd-privacy --seeds 42 43 44 45 46   # central + local DP comparison baselines, SecAgg probe -> results/
uv run ckd-audit   --seeds 42 43 44 45 46   # membership-inference leakage audit -> results/

uv run flwr run . --run-config "secure-aggregation=true"   # SecAgg+: server sees only the sum
uv run flwr run . --run-config "local-dp-epsilon=5.0"      # noise inside the SuperNode
uv run flwr run . --run-config "min-train-examples=50"     # exclude practices smaller than 50 patients
uv run flwr run . --run-config "central-dp-epsilon=8"       # server derives the noise to meet the budget

./scripts/paper.sh --recompute              # rebuild the expert report's figures/tables/numbers

uv run flwr run .                           # full Flower stack (Ray engine, real ServerApp/ClientApp)
```

> ⚠️ **Match the data source when comparing.** `ckd-baseline` defaults to the flat placeholder CSV.
> If the federated side used `--clinics`, the ceiling must too — otherwise the "price of privacy"
> you compute is just a dataset difference.

---

## The two datasets

| Dataset | What it is | Expected result |
|---|---|---|
| `data/synthetic_ckd_data.csv` | 2000-row **wiring placeholder**; max \|feature-label correlation\| **0.031** | **AUROC ≈ 0.5 is CORRECT** — use as a negative control |
| `data/clinics/*.csv` | Per-clinic, non-IID, sharing one true risk model (`uv run ckd-clinics`) | ≈ 0.80 federated, 0.86 pooled |

The placeholder has no learnable signal by design. Every protocol scoring ≈ 0.5 on it is the
evidence that the ≈ 0.80 numbers on the clinics data are real, not an artefact of the harness.

---

## How the federation runs

```text
[Practice 1]  ─┐
[Practice 2]  ─┤   local train (warm-start)        ┌─► global AUROC + sensitivity
   …           ├──────────────────────────────────►┤
[Practice 10] ─┘   weights only ─► strategy ─► global └─► worst-practice AUROC + sensitivity
```

Each round logs **dual-level metrics** ([CLAUDE.md §5](CLAUDE.md)): the sample-weighted **global**
mean *and* the **worst single practice**, so a model that collapses on one outlier practice can't
hide behind a strong aggregate.

Two execution modes:

- **`uv run flwr run .`** — the full Ray simulation engine with the real `ServerApp`/`ClientApp`.
  Use it for end-to-end checks and as the rehearsal for deployment.
- **`uv run ckd-simulate`** — an in-process runner that drives Flower's **real strategies**
  (`aggregate_train` / `aggregate_evaluate` on real `Message` objects) without the Ray transport.
  Faster, and returns per-round history for benchmarking.

---

## Repository layout

| Path | Purpose |
|---|---|
| `client_app.py` / `server_app.py` | Flower `ClientApp` / `ServerApp` on the 1.33 Message API |
| `simulate.py` | In-process runner over Flower's real strategies → `ckd-simulate` |
| `benchmark.py` | Protocol benchmark → `ckd-benchmark` → `results/benchmark.json` |
| `privacy.py` | Central + local DP sweeps (comparison baselines to the DP-SGD standard), RDP accountant, SecAgg+ probe → `ckd-privacy` |
| `audit.py` | Membership-inference leakage audit (T2.5 / layer L6) → `ckd-audit` |
| `dpsgd.py` | Patient-level DP-SGD (Poisson sampling, per-sample clipping, Gaussian noise) + in-process runner emulating SecAgg masked-sum over Flower's real FedAvg |
| `orchestrator.py` | Rule-based server agent (no LLM): deterministic census → ε-policy → accountant inversion → ConfigRecord dispatch; per-clinic (batch, σ) plans so every clinic composes to the same target ε; `DpsgdOrchestrator(FedAvg)` |
| `paper/` | The Technical Expert Report (LaTeX). `make_paper.py` generates every figure, table and inline number from one run |
| `centralized.py` | Pooled-data ceilings → `ckd-baseline` |
| `messages.py` | The Flower `Message` shapes the in-process runners exchange |
| `task.py` | Local scaler, imbalanced-data metrics, T2.5 fairness metrics |
| `data/` | Loading + missingness rules, Dirichlet partitioning, clinic generator, FHIR loader — plus the datasets |
| `models/` | `logreg`, `mlp` (FedAvg) and `fedxgb` (FedXgbBagging) |
| `models/protocols/` | The protocol benchmark, incl. `fedmosaic.py` (a Flower `Strategy` subclass) |
| `extract_features.sql` | Canonical feature contract for the **real** Tomedo→PostgreSQL export |
| `docs/` | `REPORT.md`, `PRIVACY.md`, `DEPLOYMENT.md`, `LEGAL-TECHNICAL-ANSWER.md` (superseded) + source PDFs |
| `notebooks/` | Interactive federation walkthrough; `03_dpsgd_secagg_standard.ipynb` measures the DP-SGD + SecAgg + orchestrator standard live |
| `diagrams/` | Architecture draw.io diagram of the DP-SGD + SecAgg + orchestrator standard |

---

## Features (synthetic baseline schema)

| Feature | Type | Description |
|---|---|---|
| `age_years` | continuous | age at the index date |
| `dx_hypertonie` | binary | ICD I10 |
| `dx_diabetes` | binary | ICD E10/E11/E13 |
| `dx_khk` | binary | ICD I25 |
| `dx_adipositas` | binary | ICD E66 |
| `dx_herzinsuffizienz` | binary | ICD I50/I11.0 |
| `dx_hyperurikaemie` | binary | ICD M10 |
| `years_since_{hypertonie,diabetes,khk}_dx` | continuous | years since first diagnosis (0 if absent) |

**Label** `ckd_stage3plus = 1` for CKD stage ≥ 3. **Missingness:** `dx_*` flags are structural zeros
(absent = not documented, no imputation); `years_since_*` encode absence as 0. Continuous labs
(eGFR/HbA1c, real data only) get median imputation **plus** a binary missing-indicator.

> ⚠️ **This is not the real schema.** [`extract_features.sql`](extract_features.sql) defines the
> canonical contract, and it differs in both columns and *task*: German column names, plus eGFR,
> HbA1c and sex, and an **incidence** label (`ckd_incident` within `[t0, t0+365d)` among patients
> with no prior CKD) rather than prevalence. Migrating is a re-specification, not a rename —
> see [CLAUDE.md §3b](CLAUDE.md).

---

## Deploying

SuperLink on central infrastructure, one SuperNode per practice, TLS and key-based node
authentication throughout. Full runbook — certificates, key registration, `~/.flwr/config.toml`,
`flwr run . flipit-prod`, and a two-node local rehearsal — in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

Each SuperNode can read its cohort from the practice's own **FHIR server** (`data-source=fhir`)
instead of a CSV; the mapping lives in `data/fhir_loader.py` and emits the **canonical
`extract_features.sql` contract** — labs, `geschlecht`, and the incidence label — de-identified
at the source: no identifier is ever read, rows are keyed positionally in a hash order, and an
optional per-practice `pseudonym-salt` (`--node-config`) enables the salted-HMAC join key named in
[PRIVACY.md](docs/PRIVACY.md) §2. `to_xy` dispatches on the schema, so models and the client
builder consume either path unchanged (CLAUDE.md §3b: the two schemas are different prediction
tasks). Rehearse an extraction on the practice box with
`uv run ckd-fhir-extract http://localhost:8080/fhir`.
