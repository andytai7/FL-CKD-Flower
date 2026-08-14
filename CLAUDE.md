# CLAUDE.md — FLIP-IT CKD Federated Learning (pre-kickoff baseline sandbox)

> Project memory + onboarding guide + baseline modelling strategy.
> Read this first. It tells you what this repo is, what the data means, how the federated pipeline
> is built, and how the work maps onto the funded project plan.
>
> Operational counterpart: [`.claude/skills/flip-it-ckd/SKILL.md`](.claude/skills/flip-it-ckd/SKILL.md)
> — *what to do*, in what order, and what to refuse. This file is *what is true*.

---

## 0. Immutable constraints — DO NOT VIOLATE (read first)

> 🔒 **These are non-negotiable project invariants.** They override convenience, performance, and
> any default behaviour. Do not weaken, remove, or "temporarily" work around them, and preserve them
> verbatim across any edit to this file. If a task appears to require breaking one, STOP and surface
> the conflict to the user instead of proceeding.

1. **🔒 Flower.ai is MANDATORY — this is a partnership requirement.** All federated training and
   aggregation MUST use the real `flwr` package: the `ServerApp` / `ClientApp` components and
   `flwr.serverapp.strategy` (`FedAvg`, `FedProx`, `FedXgbBagging`). **NEVER hand-roll, reimplement,
   copy, or substitute any Flower primitive** — not FedAvg averaging, not the client/server
   protocol, not the strategies. Where a runner drives strategies in-process rather than over the
   wire, it still calls **Flower's own strategy objects** (`aggregate_train` / `aggregate_evaluate`)
   — it does not write a custom aggregator.

2. **🔒 ONLY models that can run with Flower are allowed — no exceptions.** The system deploys
   federated, so a model with no `flwr` federation path is never a candidate, not even as a "quick
   baseline" or a centralized-only extra. The permitted set is exactly **`logreg` + `mlp`
   (FedAvg/FedProx)** and **`xgboost` (FedXgbBagging)**. Do **NOT** add random forest, plain
   GBT/HistGradientBoosting, LightGBM, CatBoost, k-NN, or SVM. A new model may be introduced only if
   it has a genuine Flower federation path. The pooled-data runs in `centralized.py` are the
   non-federated *ceiling* and may use **only these same Flower-compatible models**.

3. **🔒 Patient data never leaves the practice.** Only model parameters, or the explicitly designed
   protocol payloads, cross the federation boundary — never raw patient rows. Do not centralize,
   pool, upload, or transmit practice data. The *only* permitted pooling is the explicitly-labelled
   centralized "ceiling" baselines (`centralized.py`), which are non-federated references and must
   stay clearly marked as such.

4. **🔒 `uv` is the only environment & dependency manager.** No `pip` / `conda` / `poetry` /
   `virtualenv`, and no hand-editing `.venv`. Add dependencies via `uv add` / `pyproject.toml` +
   `uv sync`; run everything via `uv run`. The lockfile (`uv.lock`) stays authoritative.

5. **🔒 Imbalanced-data metrics are primary and dual-level.** Always report **AUROC** and
   **sensitivity** (recall for CKD positives), and always log **both** the global sample-weighted
   mean **and** the worst/min client per round. Never report accuracy alone, and never drop the
   per-/worst-client logging — a model can look "accurate" while collapsing on an outlier practice.
   Every strategy must be constructed with `evaluate_metrics_aggr_fn=weighted_and_worst`.

6. **🔒 Reproducibility is fixed.** One global seed → derived per-practice seeds; identical seeds,
   splits, and feature set across every model in a comparison. Vary only the thing under test.
   Never compare architectures or protocols on different splits/seeds.

7. **🔒 All project code lives inside `FL-CKD-Flower/`.** No project file is created, moved, or
   imported outside this directory. `/home/jovyan/Fed_Agent/` is **not** part of this project — it
   is an unrelated stub repo and must never be treated as project code.

8. **🔒 Custom federation logic is a Flower `Strategy` subclass.** New protocols subclass
   `flwr.serverapp.strategy.Strategy` (or `FedAvg`, which is one) and override its methods. That is
   Flower's own sanctioned extension point and does **not** violate rule 1 — reimplementing FedAvg's
   averaging or the client/server transport still does. Prefer a built-in strategy whenever one
   exists.

---

## 1. Project overview

**FLIP-IT** is a NEXT.IN.NRW / EFRE-JTF NRW innovation project run by a consortium of **docport
GmbH** (coordinator, project lead Dr. med. Nicolas Conze), the **Institut für Künstliche Intelligenz
in der Medizin (IKIM), Universitätsmedizin Essen** (HL7 FHIR harmonisation via the SHIP platform;
KITE GPU/Kubernetes infrastructure), the **AG Trustworthy Machine Learning at Ruhr-Universität
Bochum** (**Prof. Dr. Michael Kamp** — federated learning on non-IID data, FedBN), and the
medical-law firm **Jorzig & Partner** (legal / data-protection review).

The goal is a **federated-learning infrastructure spanning 25 regional GP (Hausarzt) practices**
that trains a **chronic kidney disease (CKD) risk model** on **HL7-FHIR–harmonised** routine care
data — *the model travels to the data, the patient data never leaves the practice* — coordinated
with **Flower.ai** (the Antrag records a Letter of Intent from FlowerAI), and hardened with
**Differential Privacy and Secure Aggregation**.

Ground truth for all of the above: [`docs/01 Projektantrag Innovationswettbewerb NEXT.IN.NRW.pdf`](docs/).

> ⚠️ **This repository is the pre-kickoff baseline *sandbox*, not the funded production system.**
> Its purpose is to de-risk the project before kickoff: establish reference metrics, a working
> federated pipeline, and a documented feature contract on **synthetic** data standing in for real
> practice data.

`docs/2025_11_Guetersloh_KI im klinischen Alltag.pdf` is an image-based slide deck (one extractable
text slide, about CellViT). **It contains nothing citable for this project** — do not source claims
from it.

---

## 2. Repository map

| Path | Purpose |
|---|---|
| `client_app.py` | Flower `ClientApp` — one practice. `@app.train()` / `@app.evaluate()` on the Message API. `build_client_from_frame()` is the **only** client constructor. |
| `server_app.py` | Flower `ServerApp` — `@app.main()`, `strategy.start()`, and `weighted_and_worst` (dual-level metrics + the L5 privacy policy). |
| `simulate.py` | In-process runner over Flower's real strategies → `uv run ckd-simulate`. Fast per-round benchmarking without Ray. |
| `benchmark.py` | The protocol benchmark → `uv run ckd-benchmark` → `results/benchmark.json`. |
| `privacy.py` | DP sweep + SecAgg+ feasibility probe → `uv run ckd-privacy` → `results/privacy.json`. |
| `centralized.py` | Pooled-data ceiling baselines → `uv run ckd-baseline`. **Use `--clinics` when comparing against `--clinics` runs.** |
| `messages.py` | Single definition of the Flower `Message` shapes the in-process runners exchange. |
| `task.py` | Local `StandardScaler`, the imbalanced-data metrics, and the T2.5 `fairness_metrics`. |
| `data/` | `loader.py` (+ §3 missingness rules), `partition.py` (Dirichlet non-IID), `synthesize.py` (per-clinic generator + the FedMosaic public cohort), `fhir_loader.py` (the production FHIR path). Also holds the datasets. |
| `models/` | `base.py`, `logreg.py`, `mlp.py` (FedAvg-compatible), `fedxgb.py` (FedXgbBagging). |
| `models/protocols/` | The protocol benchmark: `common.py` (explicit logistic regression), `fedmosaic.py` (the `Strategy` subclass). |
| `extract_features.sql` | Canonical feature contract for the **real** Tomedo→PostgreSQL export. |
| `docs/` | `PRIVACY.md`, `DEPLOYMENT.md`, `REPORT.md`, `LEGAL-TECHNICAL-ANSWER.md`, and the source PDFs. |
| `notebooks/` | The federation walkthrough. |
| `results/` | Generated benchmark/privacy JSON (gitignored). |

---

## 3. Data

### 3a. Synthetic working schema — 10 features + label

`age_years`; the binary flags `dx_hypertonie` (I10), `dx_diabetes` (E10/E11/E13), `dx_khk` (I25),
`dx_adipositas` (E66), `dx_herzinsuffizienz` (I50/I11.0), `dx_hyperurikaemie` (M10); and
`years_since_{hypertonie,diabetes,khk}_dx`. Label `ckd_stage3plus`.

**Missingness rules (apply in every model's preprocessing):**
- **`dx_*` flags** — an absent value is a **structural zero** (not documented → assumed absent). **Do not median-impute.**
- **`years_since_*`** — 0 already encodes "diagnosis absent", paired with its `dx_` flag.
- **Continuous labs (real data only)** — median-impute **plus** a binary missing-indicator; a missing lab is itself informative.

Two datasets exist and they behave very differently:

| Dataset | What it is | Expected result |
|---|---|---|
| `data/synthetic_ckd_data.csv` | 2000-row **wiring placeholder**, max \|feature-label correlation\| 0.031 | **AUROC ≈ 0.5 is CORRECT.** Use it as a negative control. |
| `data/clinics/*.csv` (`uv run ckd-clinics`) | Per-clinic, non-IID, sharing one true logistic risk model | AUROC ≈ 0.80 federated, 0.86 pooled ceiling |

### 3b. Canonical contract for real data — `extract_features.sql`

When real HL7-FHIR / Tomedo data arrives, **the SQL defines the truth**.

- **Landmark (`Stichtag`):** rolling, `CURRENT_DATE − 365 days`. **Outcome window** `[t0, t0+365d)`.
- **Inclusion:** living patients, 18–95, sex documented; ≥1 contact ≥52 weeks before `t0`; **no known CKD (`N18.*`) before `t0`**; not a test patient.
- **Exclusion TODOs still open in the SQL:** dialysis (`OPS 8-854.*` / `Z99.2`), transplant (`Z94.0`), Vertretungsscheine.
- **Label `ckd_incident`:** confirmed `N18.*` (typ='G') **or** eGFR ≤ 60 with a predecessor ≥ 90 days earlier also ≤ 60.
- **CVD coverage TODO:** has I20–I25, I50, I63–I66; missing I47–I49, I60–I62, I70, I73.9.

> ⚠️ **The two schemas are not the same prediction task.** The synthetic label is CKD stage ≥ 3
> **prevalence**; the SQL label is **incidence in the following year among patients with no prior
> CKD**. The columns differ too (`alter_jahre`, `geschlecht`, `dm`/`aht`/`cvd`, `tage_seit_*`,
> `egfr_*`, `hba1c_*`). Migrating is a re-specification, not a rename.
>
> The real contract adds **eGFR — the single strongest CKD predictor** — and `geschlecht`, without
> which the T2.5 sex-based bias analysis cannot be run at all.

---

## 4. Models and protocols

### Models (all Flower-federatable, rule 2)

| Model | Flower strategy | Role |
|---|---|---|
| Logistic regression (SGD, warm-start) | `FedAvg` / `FedProx` | reference model, grant task **T2.2** |
| Small MLP (sklearn, torch-free) | `FedAvg` | linear-vs-non-linear comparison |
| XGBoost | `FedXgbBagging` | gradient-boosted trees, federated by bagging |

### The three protocols under test

| Protocol | Strategy | Model | Wire payload |
|---|---|---|---|
| **FedProx** | `flwr.serverapp.strategy.FedProx` (built-in) | logreg | coefficients |
| **FedXgbBagging** | `flwr.serverapp.strategy.FedXgbBagging` (built-in) | xgboost | serialized trees |
| **FedMosaic** | `models/protocols/fedmosaic.py` (`Strategy` subclass) | logreg | predictions + expertise on a public cohort |

Plus two **reference baselines** that are not protocols but are required to interpret them: `local`
(no collaboration — the FedMosaic paper's key finding is that this is a strong baseline) and
`fedavg` (isolates what FedProx's μ term actually bought).

**FedMosaic** implements Algorithm 1 of `docs/2507.00259v3.pdf`: dynamic loss weighting
`α = exp(−(ℓ_pseudo − ℓ_priv)/ℓ_priv)` decides *when* to trust the consensus, and confidence-based
aggregation `S = Σ diag(E_i)·L_i` decides *whose* predictions to trust. It is a **personalized** FL
method — every practice keeps its own model.

**SCAFFOLD is not available.** It is not in the `flwr` package (verified by inspection of 1.33.0 —
neither `flwr.serverapp.strategy` nor `flwr.server.strategy` has it); it exists only in Flower
Baselines as a standalone PyTorch reproduction project. FedProx is the shipped alternative for
client-drift correction.

### Non-IID design

`data/synthesize.py` builds clinics from five clinically interpretable archetypes (urban-young,
rural-elderly, metabolic, high-CVD, mixed) that differ in age, comorbidity prevalence, CKD rate and
panel size — non-IID on covariate, label and quantity — while sharing **one true risk model**, so
there is a global truth to learn. `data/partition.py` provides Dirichlet label partitioning
(`alpha→0` skewed, `alpha→∞` IID) for the flat CSV.

---

## 5. Federated setup

- **Client contract:** `@app.train()` / `@app.evaluate()` receive a `Message`; weights arrive as
  `msg.content["arrays"]` and reply as `RecordDict({"arrays": ArrayRecord, "metrics": MetricRecord})`
  with `num-examples` — the key Flower weights the average by.
- **Two execution modes:** `flwr run .` (full Ray simulation engine, real ServerApp/ClientApp) for
  end-to-end checks; `uv run ckd-simulate` (in-process, Flower's real strategies) for fast
  reproducible benchmarking with per-round history.
- **Dual-level metric logging is required** (rule 5) and is *pre-staging the T2.5 bias analysis*.
- **Fairness (T2.5):** the Antrag names **demographic parity, equal opportunity, equalized odds, and
  calibration by group**. `task.fairness_metrics()` implements all four as gaps. The Antrag
  specifies them **by sex**, which the synthetic schema cannot support (§3b).

---

## 6. Conventions & commands

`uv`-managed (rule 4). `flwr` is pinned to the **1.33** line — the Message API and the
`flwr.serverapp.strategy` namespace.

```bash
uv sync --extra dev --extra notebook      # build/repair the env  (= make setup)

uv run ckd-clinics --clinics 10           # generate the per-clinic datasets -> data/clinics/
uv run ckd-simulate --clinics             # logreg via FedAvg, non-IID
uv run ckd-simulate --model xgboost --clinics
uv run ckd-simulate --protocol fedmosaic --clinics
uv run ckd-baseline --model all --clinics # pooled ceiling on the SAME data
uv run ckd-benchmark --rounds 20          # full protocol benchmark -> results/
uv run ckd-privacy --seeds 42 43 44 45 46 # DP sweep + SecAgg probe  -> results/

uv run flwr run .                         # full Ray engine, real ServerApp/ClientApp
```

**Deployment:** SuperLink connection config lives in `~/.flwr/config.toml` from flwr 1.30+, not in
`pyproject.toml`. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## 7. Roadmap alignment (Projektantrag)

| Activity here | Maps to | When |
|---|---|---|
| Feature contract from `extract_features.sql` | **T2.1** Identifikation Prädiktoren | Q4 2025 – Q1 2026 |
| Logistic-regression federated baseline | **T2.2** Logistisches Regressionsmodell | Q1 – Q3 2026 |
| DP sweep, SecAgg probe, protocol payload analysis | **T2.3** Privacy Enhancing Features | from Q2 2026 |
| Centralized-vs-federated gap, protocol benchmark | **T2.4** Modellvalidierung | Q1 – Q3 2027 |
| Dual-level metrics, `fairness_metrics` | **T2.5** Datenschutztests & Bias-Analyse | Q1 2027 onward |

Milestones: **MS1** (m9) harmonised data + functional platform; **MS2** (m12) training workshop;
**MS3** (m13) first collection at `t0`; **MS4** (m18) **validated CKD model incl. DP + Secure
Aggregation**; **MS5** (m24) pilot in **25 practices**; **MS6** (m34) transfer strategy.

---

## 8. Live issues

1. **`extract_features.sql:342` exports `ep.patientid`** — a direct identifier in every practice
   CSV. Drop or hash before any real extraction ([PRIVACY.md §2](docs/PRIVACY.md)).
2. **DP ε is far too large to be meaningful** at the cohort size (48–969 over 20 rounds). The
   biggest MS4 risk; needs more practices, fewer rounds, and a real accountant
   ([PRIVACY.md §3](docs/PRIVACY.md)).
3. **SecAgg+ is legacy-path only** in flwr 1.33 and cannot compose with `strategy.start()`
   ([PRIVACY.md §4](docs/PRIVACY.md)).
4. **No membership-inference testbed** — the L6 leakage audit is not yet built.
5. **No test suite.** `pytest` and the `test` target were removed rather than left broken; real
   tests still need writing.
6. **Fairness is audited by age band, not sex** — blocked on the real schema's `geschlecht`.
7. **XGBoost federates poorly here** (0.712 vs a 0.860 pooled ceiling) and cannot be protected by
   SecAgg+. Logistic regression is the recommended deployment model.
