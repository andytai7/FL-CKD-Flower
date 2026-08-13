# CLAUDE.md — FLIP-IT CKD Federated Learning (Pre-Kickoff Baseline Sandbox)

> Project memory + onboarding guide + pre-kickoff baseline modeling strategy.
> Read this first. It tells you what this repo is, what the data means, how the
> federated baseline pipeline should be built, and how the baseline work maps onto the
> funded project plan.

---

## 0. Immutable constraints — DO NOT VIOLATE (read first)

> 🔒 **These are non-negotiable project invariants.** They override convenience, performance, and
> any default behavior. Do not weaken, remove, or "temporarily" work around them, and preserve them
> verbatim across any edit to this file. If a task appears to require breaking one, STOP and surface
> the conflict to the user instead of proceeding.

1. **🔒 Flower.ai is MANDATORY — this is a partnership requirement.** All federated training and
   aggregation MUST use the real `flwr` package: `flwr.client.NumPyClient`, the `ServerApp` /
   `ClientApp` components, and `flwr.server.strategy` (`FedAvg`, and `FedXgbBagging` for federated
   trees). **NEVER hand-roll, reimplement, copy, or substitute any Flower primitive** — not FedAvg
   averaging, not the client/server protocol, not the strategies. If Flower's Ray runtime can't run
   locally (e.g. the space-in-path issue), you still drive **Flower's own strategy objects**
   in-process (`FedAvg.aggregate_fit` / `aggregate_evaluate`) — you do **not** write a custom
   aggregator. *Current state: `simulate.py` drives the real `flwr` `FedAvg`; `flwr run .` runs the
   full stack from a space-free venv via `UV_PROJECT_ENVIRONMENT` (see §6).*

2. **🔒 ONLY models that can run with Flower are allowed — no exceptions.** The system deploys
   federated, so a model with no `flwr` federation path is never a candidate, not even as a "quick
   baseline" or a centralized-only extra. The permitted set is exactly **`logreg` + `mlp` (FedAvg)**
   and **`xgboost` (FedXgbBagging)**. Do **NOT** add random forest, plain GBT/HistGradientBoosting,
   LightGBM, CatBoost, k-NN, SVM, or any estimator that can't be aggregated by a Flower strategy. A
   new model may be introduced only if it has a genuine Flower federation path (weight-averageable
   for `FedAvg`, or a real Flower tree/strategy). The pooled-data runs in `centralized.py` are the
   non-federated *ceiling* and may use **only these same Flower-compatible models** — they are never
   an excuse to bring in a non-federatable one. If a model can't federate with Flower, it does not
   belong in this repo.

3. **🔒 Patient data never leaves the practice.** Only model parameters/weights cross the federation
   boundary — never raw patient rows. Do not centralize, pool, upload, or transmit practice data.
   The *only* permitted pooling is the explicitly-labeled centralized "ceiling" baselines
   (`centralized.py`), which are non-federated references and must stay clearly marked as such.

4. **🔒 `uv` is the only environment & dependency manager.** No `pip` / `conda` / `poetry` /
   `virtualenv`, and no hand-editing `.venv`. Add dependencies via `uv add` / `pyproject.toml` +
   `uv sync`; run everything via `uv run`. The lockfile (`uv.lock`) stays authoritative.

5. **🔒 Imbalanced-data metrics are primary and dual-level.** Always report **AUROC** and
   **sensitivity** (recall for CKD positives), and always log **both** the global sample-weighted
   mean **and** the worst/min client per round. Never report accuracy alone, and never drop the
   per-/worst-client logging — a model can look "accurate" while collapsing on an outlier practice.

6. **🔒 Reproducibility is fixed.** One global seed → derived per-practice seeds; identical seeds,
   splits, and feature set across every model in a comparison. Vary only the model. Never compare
   architectures on different splits/seeds.

---

## 1. Project overview

**FLIP-IT** (*Federated Learning im Praxisnetzwerk – Infrastruktur für dezentrales
Training medizinischer KI-Modelle*) is a NEXT.IN.NRW / EFRE-JTF NRW innovation project run
by a consortium of **docport GmbH** (coordinator, Essen), the **Institut für Künstliche
Intelligenz in der Medizin (IKIM), Universitätsmedizin Essen** (HL7 FHIR harmonization via
the SHIP platform), the **AG Trustworthy Machine Learning at Ruhr-Universität Bochum**
(**Prof. Dr. Michael Kamp** — the PI for this repository; federated learning on non-IID data,
FedBN), and the medical-law firm **Jorzig & Partner** (legal/data-protection review). The goal
is a **federated-learning infrastructure spanning 25 regional general-practitioner (Hausarzt)
practices** that trains a **chronic kidney disease (CKD) risk model** on **HL7-FHIR–harmonized**
routine care data — *the model travels to the data, the patient data never leaves the
practice* — coordinated with **Flower.ai**, and ultimately hardened with **Differential Privacy
(DP) and Secure Aggregation (SecAgg)**.

> ⚠️ **This repository is the pre-kickoff baseline *sandbox*, not the funded production
> system.** Its purpose is to *de-risk* the project before official kickoff: stand up
> preliminary baseline models now — on the **synthetic CKD dataset** that stands in for the
> real practice data — so that when funded work begins the team already has (a) reference
> metrics, (b) a working federated pipeline, and (c) a documented feature contract. Privacy
> mechanisms (DP/SecAgg) are intentionally **deferred** here; the baseline runs **plain
> FedAvg** to establish clean reference numbers.

---

## 2. Repository map

The repo centers on the **root scikit-learn baseline** driven by the **synthetic CKD dataset**.

| File | Purpose |
|---|---|
| `synthetic_ckd_data.csv` | **Primary working dataset.** 2000 synthetic patients, 10 features + `ckd_stage3plus` label. Stand-in for real practice data until the HL7 FHIR export is available. Detailed in §3. |
| `client.py` | Flower `NumPyClient` running per practice; logistic regression via sklearn `SGDClassifier(loss="log_loss")` with warm-start across rounds. **Has known bugs — see §8.** |
| `server.py` | Flower server, **FedAvg** strategy, 20 rounds, min 8/12 clients. |
| `extract_features.sql` | **Canonical feature contract for the *real* data.** Tomedo→PostgreSQL CKD "landmark" extraction: inclusion/exclusion, dx flags, eGFR/HbA1c (last + mean-of-3), composite KDIGO-style incident label. Run once per practice to produce a local CSV once real data arrives. Detailed in §3. |
| `requirements.txt` | Lightweight env: `flwr==1.8.0`, scikit-learn, pandas, numpy. |
| `README.md` | German-language baseline description, deployment steps, feature table, expected metrics. |

> 🗑️ **`flower_demo/` is temporary practice scaffolding and will be deleted.** It was Andy's
> initial Flower/PyTorch exploration before the real data was available. **Do not build on it
> or treat it as part of the project.** Concepts worth carrying forward (a small MLP, a non-IID
> data partitioner) are re-specified below as **net-new code in the root repo**, not as reuse
> of `flower_demo/`.

---

## 3. Data

### 3a. Synthetic working dataset — `synthetic_ckd_data.csv` (use this now)

A flat file of **2000 synthetic patients**, one row each, **no practice column** (federation is
simulated by partitioning — see §4). Schema matches `client.py`'s `FEATURE_COLS`:

| Column | Type | Notes |
|---|---|---|
| `age_years` | continuous | age at landmark |
| `dx_hypertonie` | binary | ICD I10 present |
| `dx_diabetes` | binary | ICD E10/E11/E13 |
| `dx_khk` | binary | ICD I25 (coronary heart disease) |
| `dx_adipositas` | binary | ICD E66 |
| `dx_herzinsuffizienz` | binary | ICD I50/I11.0 |
| `dx_hyperurikaemie` | binary | ICD M10 |
| `years_since_hypertonie_dx` | continuous | 0 when diagnosis absent |
| `years_since_diabetes_dx` | continuous | 0 when diagnosis absent |
| `years_since_khk_dx` | continuous | 0 when diagnosis absent |
| `ckd_stage3plus` | binary | **label** — CKD stage ≥ 3 |

**Missingness rule (apply in every model's preprocessing):**

- **Binary diagnostic flags (`dx_*`):** an absent value is a **structural zero** (condition not
  documented → assumed absent). **Do not median-impute these.**
- **`years_since_*`:** 0 already encodes "diagnosis absent" and is paired with the matching
  `dx_*` flag, so keep the flag as the real signal and do not double-count.
- **Continuous labs (when added — see §3b):** median-impute **+ carry a binary
  missing-indicator**; a missing lab is itself informative.

> ⚠️ **This synthetic schema is the *old* 10-feature set and differs from the canonical
> `extract_features.sql` contract** (§3b) — notably it has **no eGFR / HbA1c / sex** and uses a
> different comorbidity granularity. It is a usable stand-in for wiring up the federated
> pipeline today, but plan to reconcile onto the SQL contract when real data lands (§8).

### 3b. Canonical feature contract for real data — `extract_features.sql`

When the real HL7 FHIR / Tomedo data arrives, **the SQL defines the truth** and the synthetic
schema is migrated to match it.

- **Landmark / Stichtag:** rolling, `CURRENT_DATE − 365 days` (fix to a constant date for
  reproducible runs). **Outcome window** is `[t0, t0 + 365 days)` — incidence in the following
  year.
- **Inclusion:** living patients, **18–95 years**, sex documented; ≥ 1 contact ≥ 52 weeks before
  `t0`; **no known CKD (`N18.*`) before `t0`**; not a test patient.
- **Exclusion TODOs (open in SQL):** dialysis (`OPS 8-854.*` / `ICD Z99.2`), kidney transplant
  (`Z94.0`), substitute-billing scheins. Flag these as not-yet-implemented in cohort counts.
- **Label `ckd_incident` (composite, KDIGO-aligned):** 1 if within the window either **(A)** a
  confirmed (`typ='G'`, non-anamnestic) `N18.*` first diagnosis, **or (B)** eGFR ≤ 60 with a
  predecessor ≥ 90 days earlier also ≤ 60 and every value in the rolling 90-day window ≤ 60
  (approximates KDIGO ≥ 3-month persistence).
- **Features:** `alter_jahre`, `geschlecht` (1=M/0=W), `dm`/`aht`/`cvd` flags, `tage_seit_*`
  (NULL when absent), `egfr_letzter`, `egfr_mittelwert_3`, `hba1c_letzter`, `hba1c_mittelwert_3`.
  Same missingness rule as §3a (structural-zero flags; median-impute + indicator for labs).
- **CVD coverage TODO:** currently I20–I25, I50, I63–I66; still missing I47–I49, I60–I62, I70,
  I73.9 (PAVK).

**Why this matters for the baseline:** the canonical contract adds **eGFR/HbA1c**, and **eGFR is
the single strongest CKD predictor**. Its absence is the main limitation of today's synthetic
baseline and the main expected lift once real data arrives.

---

## 4. Pre-kickoff baseline strategy (core)

Build **one model-agnostic federated pipeline** with **three swappable estimators**, all
evaluated on the **same** feature set (synthetic now, canonical SQL contract later). The point is
comparability: hold data, splits, seeds, and metrics fixed; vary only the model.

### Shared pipeline

1. **Partition `synthetic_ckd_data.csv` across N simulated practices** (the file has no practice
   column — see the non-IID partitioner below).
2. **Per-practice preprocessing** (fit locally, never shared): `StandardScaler` for linear/NN;
   §3 missingness handling; class imbalance via `class_weight='balanced'` (sklearn) / weighted
   `BCEWithLogitsLoss` (NN).
3. **Local train/test split** — 80/20, seeded, reproducible permutation before split.
4. **Flower `NumPyClient`** — `get_parameters` / `set_parameters` / `fit` / `evaluate`.
5. **FedAvg server** — aggregate weights; report **AUROC + sensitivity** per round, **both global
   and per-client** (see §5).

### Architecture A — Logistic Regression (linear; FLIP-IT **T2.2** anchor)

- Keep the `SGDClassifier(loss="log_loss")` **warm-start** pattern from `client.py` (parameter
  exchange = `[coef_.flatten(), intercept_]`); `partial_fit` enables cross-round warm-starting,
  which is why SGD is used instead of plain `LogisticRegression`.
- This is the **reference model** and the one explicitly named in the grant (T2.2).
- ⚠️ Fix the two `client.py` bugs (§8) when unifying.

### Architecture B — Small MLP (neural network) — **build fresh in root**

- Implement a small feed-forward net (a few hidden layers, dropout). **Two options:**
  - **sklearn `MLPClassifier`** — keeps the lightweight env (no torch), but exposing/setting
    `coefs_`/`intercepts_` for FedAvg weight exchange is fiddly.
  - **A minimal PyTorch MLP** — cleaner `state_dict` weight exchange for Flower; adds a `torch`
    dependency, so gate it behind an **optional** extra in `requirements.txt` rather than making
    it mandatory for the logistic baseline.
- **BatchNorm caveat:** if you use BatchNorm, per-practice batches can be tiny / class-skewed,
  which destabilizes running stats under non-IID. Prefer `LayerNorm`/`GroupNorm`, or adopt
  **FedBN** (keep BN params local) — directly aligned with Prof. Kamp's cited work.
- **Input-dim:** set to the actual feature count after adding any missing-indicator columns;
  don't hard-code an old fixed width.

### Architecture C — Federated XGBoost via `FedXgbBagging`

- XGBoost is the tree model the project keeps, because it is the **only tree model Flower can
  federate** (immutable rule 2: every model must run with Flower). It can't FedAvg — there are no
  weight vectors to average — so Flower federates it by **bagging**: each practice grows a few local
  boosting rounds and the server appends those trees into one global ensemble
  (`flwr.server.strategy.FedXgbBagging`). Implemented in `models/fedxgb.py`; run with
  `uv run ckd-simulate --model xgboost`.
- A **centralized, pooled-data** XGBoost (`ckd-baseline --model xgboost`) is kept only as the
  non-federated **"best case if data could be centralized"** ceiling — the *same* Flower-compatible
  model, trained on pooled data.
- **Removed:** random forest, plain HistGradientBoosting, and LightGBM — none can run with Flower,
  so they are not deployment candidates and have no place here.
- **Purpose:** quantify (i) the **federated-vs-centralized** gap and (ii) the
  **linear-vs-non-linear** gap on tabular EHR features.

### Experimental protocol

- **Fixed:** identical seeds, splits, and feature set across all three architectures.
- **Report:** per-round **distributed AUROC + sensitivity** (§5).
- **Compare:** (i) centralized-pooled vs federated per architecture; (ii) the three architectures
  against one another; (iii) **IID vs non-IID** practice partitions.
- **Expected baseline** (from `README.md`): sensitivity ~**0.6–0.75**, accuracy ~**0.70–0.80**.

### Non-IID partitioner for 25 practices — **build fresh in root**

`synthetic_ckd_data.csv` is a flat 2000-row file with **no practice id**, so the federation must
be **simulated by partitioning** (or by regenerating data per practice). A naive equal random
split is IID and unrealistic. Build a partitioner using a **hierarchical / archetype scheme**:

- **Practice archetypes** (clinically interpretable, not a monotonic ramp): *urban-young*,
  *rural-elderly*, *metabolic / high-diabetes*, *high-CVD*, *mixed*. Distribute the 25 practices
  across these archetypes.
- **Hierarchical priors (random effects):** each practice draws its own age / comorbidity-
  prevalence profile from population-level priors anchored to German primary-care epidemiology
  (hypertension ~25–30 %, diabetes ~9–15 %, plausible CKD base rate) — used either to **sample
  rows** from the pooled CSV by skewed acceptance or to **regenerate** per-practice synthetic
  rows.
- **Three independently tunable non-IID knobs:** (1) **label/prior shift** — per-practice CKD
  rate from a Beta prior; (2) **covariate shift** — per-practice feature means/variances;
  (3) **quantity shift** — unequal panel sizes (small rural vs large urban). A Dirichlet
  concentration `α` sweeps IID (`α→∞`) → strongly non-IID (`α→0`).
- **Reproducibility:** one global seed → per-practice **derived** seeds.
- **Rationale:** mirrors the **FedBN** / Prof. Kamp "non-IID features via local batch
  normalization" motivation in the Projektantrag and stress-tests the MLP BatchNorm caveat.

---

## 5. Federated setup

- **Client/server contract (Flower):** each client implements `get_parameters`,
  `set_parameters`, `fit`, `evaluate`; the server runs **FedAvg** (sample-weighted average of
  client updates). The parameter serializer is architecture-specific: `[coef_, intercept_]` for
  the sklearn logistic client, `state_dict` arrays for a PyTorch MLP.
- **Two execution modes:**
  - **Simulation** — many virtual clients in one process; fast IID/non-IID sweeps and
    architecture comparison. (Build a small root simulation entry point to replace the
    `flower_demo` one being deleted.)
  - **Real deployment** — `server.py` + one `client.py` per practice machine over the network;
    the shape the funded pilot will take.
- **Metrics that matter (imbalanced data):** prioritize **AUROC** and **sensitivity/recall** for
  CKD-positive cases over raw accuracy — a model that never predicts CKD can still look
  "accurate" on a low-prevalence cohort.
- **Dual-level metric logging (required).** In a non-IID network a model can show strong
  **global aggregated** AUROC/sensitivity while **collapsing on outlier archetypes**
  (rural-elderly vs urban-young). Log **both** the global sample-weighted mean **and**
  **per-client local validation** metrics every round (including the **worst/min client**). This
  quantifies performance drops from local feature variance and **pre-stages the T2.5 bias
  analysis**.

---

## 6. Conventions & commands

`uv`-managed project (see immutable rule 4). Dependencies + scripts live in `pyproject.toml`; the
federated stack is `flwr` (real Flower, immutable rule 1) + scikit-learn / pandas / numpy. Optional
extras: `dev`, `notebook`. (XGBoost is a **core** dependency — it's a first-class federated model
via `FedXgbBagging`, not an optional extra.) The `scripts/` wrappers and `Makefile` targets all call
these.

```bash
uv sync --extra dev --extra notebook      # build/repair the env  (= make setup)

# Federated baseline — drives Flower's REAL strategies in-process (works on this space-in-path):
uv run ckd-simulate                       # logreg via FedAvg, 12 practices, 20 rounds, non-IID
uv run ckd-simulate --model mlp --iid
uv run ckd-simulate --model xgboost       # federated trees via FedXgbBagging
uv run ckd-simulate --clinics             # one practice per on-disk data/clinics/ CSV (run ckd-clinics first)

# Generate synthetic per-clinic datasets (the FLIP-IT federation: many non-IID clinics, one shared
# CKD truth) -> data/clinics/. Each CSV simulates one practice's extract_features.sql output:
uv run ckd-clinics --clinics 10           # data/synthesize.py; see notebooks/ for the full demo

# Centralized "ceiling" references (non-federated, pooled-data):
uv run ckd-baseline --model all           # centralized ceiling: logreg | mlp | xgboost | all
```

**Full Flower stack via `flwr run` (real `ServerApp`/`ClientApp` + Ray engine).** `flwr run .`
crashes here because Ray cannot handle the **space** in this folder's path. The fix is a venv at a
space-free path (does *not* require moving the project) — verified working:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/ckd_venv     # any space-free path
uv sync --extra dev
uv run flwr run .                               # full Flower simulation engine, real ServerApp/ClientApp
```

For real multi-node deployment, run the `ServerApp` on the central infrastructure and one
`ClientApp` (SuperNode) per practice; per practice, export local data once with
`psql -d tomedo -c "\COPY ($(cat extract_features.sql)) TO '/tmp/praxis_data.csv' CSV HEADER"`.

> **MLP / torch:** the federated MLP uses sklearn (torch-free). A torch MLP is the documented
> upgrade path and must stay behind an optional extra so the core stack stays lightweight.

---

## 7. Roadmap alignment (baseline → AP2 milestones)

| Baseline activity (now) | Maps to | When (per Projektantrag) |
|---|---|---|
| Identify predictors from `extract_features.sql`; document feature contract | **T2.1** Identifikation Prädiktoren | Q4 2025 – Q1 2026 |
| Logistic-regression federated baseline (Architecture A) | **T2.2** Logistisches Regressionsmodell | Q1 – Q3 2026 |
| (deferred) DP-SGD / Secure Aggregation experiments | **T2.3** Privacy Enhancing Features | from Q2 2026 |
| Centralized-vs-federated + architecture comparison; AUROC/sensitivity validation | **T2.4** Modellvalidierung | Q1 – Q3 2027 |
| Per-client / per-archetype performance, bias audit (dual-level metrics) | **T2.5** Datenschutztests & Bias-Analyse | Q1 2027 onward |

Milestones for orientation: **MS1** (month 9) harmonized data + functional platform; **MS3**
(month 13) first data collection at `t0`; **MS4** (month 18) **validated CKD model incl. DP +
Secure Aggregation** (`t1`); **MS5** (month 24) pilot in **25 practices**. DP/SecAgg belong to
MS4 — hence the baseline defers them.

---

## 8. Known issues / TODO

### Bugs (documented, **not** fixed in this doc — fix during the baseline build)

- **`client.py:148`** — uses `roc_auc_score` but **never imports it**
  (`from sklearn.metrics import roc_auc_score` missing → `NameError` at evaluation).
- **`client.py:83`** — reads the bare name `class_weight_balanced` inside `__init__`, but it is
  only defined as a `main()` argument; `CKDClient.__init__` has no such parameter → constructor
  **`NameError`**. (Add it as a constructor parameter or pass it through.)

### Build / migration work (future)

1. Fix the two `client.py` bugs above.
2. **Delete `flower_demo/`** once its useful ideas (MLP, non-IID partitioner) are reimplemented in
   the root repo.
3. Build a **non-IID partitioner** for `synthetic_ckd_data.csv` (§4 archetype scheme) and a small
   **root simulation runner**.
4. Add a **model-selector flag** (`--model {logreg,mlp,xgboost}`) so all three architectures share one
   pipeline, one preprocessing path, and one metrics harness.
5. Implement **dual-level metric logging** (global + per-client / worst-client) in the server
   metrics aggregation (§5).
6. When real data arrives, **migrate the synthetic schema to the `extract_features.sql` contract**
   (adds eGFR/HbA1c/sex — the expected performance lift).
7. Leave DP/SecAgg hooks inactive until T2.3/MS4.
