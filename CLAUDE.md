# CLAUDE.md — FLIP-IT CKD Federated Learning (pre-kickoff baseline sandbox)

> Project memory + onboarding guide + baseline modelling strategy.
> Read this first. It tells you what this repo is, what the data means, how the federated pipeline
> is built, and how the work maps onto the funded project plan.
>
> Operational counterpart: [`.claude/skills/flip-it-ckd/SKILL.md`](.claude/skills/flip-it-ckd/SKILL.md)
> — *what to do*, in what order, and what to refuse. This file is *what is true*.

---

## 0a. Branch charter — main

The frozen-infrastructure branch: the Flower app, the runners, the V1 data contract, the
published baselines, and the L0–L6 privacy stack as documented in `docs/PRIVACY.md`. Changes
arrive only as reviewed merges from `dev` (README "Branches" table).

**Topology rule (user-directed, 2026-09-04): branches name data kinds — never methodologies.**
`experiment/tabular` (CKD clinics + mapped external cohorts), `experiment/image` (DermaMNIST),
`experiment/timeseries` (CinC 2017 ECG). The privacy/FL methodologies (P1–P4 of the deep
time-series benchmark: gradient-space DP, SecAgg at scale, FedCT consensus, verified hybrid) are
**shared code in the toolkit on `dev`** under `research/privacy-dl-ts/` — every data track runs
the same methodology matrix over its own data. `experiment/privacy-protocol` stays stale by
policy.

This CLAUDE.md is the canonical text; sibling branches carry their own §0a charters, and merges
keep the target branch's charter.

---

## 0. Immutable constraints — DO NOT VIOLATE (read first)

> 🔒 **These are non-negotiable project invariants.** They override convenience, performance, and
> any default behaviour. Do not weaken, remove, or "temporarily" work around them, and preserve them
> verbatim across any edit to this file. If a task appears to require breaking one, STOP and surface
> the conflict to the user instead of proceeding.

1. **🔒 Flower.ai is MANDATORY — this is a partnership requirement.** All federated training and
   aggregation MUST use the real `flwr` package: the `ServerApp` / `ClientApp` components and
   `flwr.serverapp.strategy` (`FedAvg`, `FedProx`). **NEVER hand-roll, reimplement,
   copy, or substitute any Flower primitive** — not FedAvg averaging, not the client/server
   protocol, not the strategies. Where a runner drives strategies in-process rather than over the
   wire, it still calls **Flower's own strategy objects** (`aggregate_train` / `aggregate_evaluate`)
   — it does not write a custom aggregator.

2. **🔒 ONLY models that can run with Flower are allowed — no exceptions.** The system deploys
   federated, so a model with no `flwr` federation path is never a candidate, not even as a "quick
   baseline" or a centralized-only extra. The permitted set is exactly **`logreg`**
   (FedAvg/FedProx; FedMosaic is a protocol over the same logistic regression). Do **NOT** add
   gradient-boosted trees, random forest, neural networks, LightGBM, CatBoost, k-NN, or SVM.
   A new model may be introduced only if it has a genuine Flower federation path. The pooled-data
   runs in `centralized.py` are the non-federated *ceiling* and may use **only this same
   Flower-compatible model**.

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
| `privacy.py` | Central + local DP sweeps, SecAgg+ probe → `uv run ckd-privacy` → `results/privacy.json`. |
| `dp.py` | The (ε, δ) accounting: RDP accountant + `sigma_for_epsilon` budget→noise inversion. The single definition shared by the sweeps, the audit, and the live server (`central-dp-epsilon`). |
| `audit.py` | Membership-inference leakage audit (T2.5 / layer L6) → `uv run ckd-audit` → `results/audit.json`. |
| `dpsgd.py` | Patient-level DP-SGD (Poisson sampling, per-sample clipping, Gaussian noise) + the in-process runner that emulates SecAgg masked-sum semantics over Flower's real FedAvg. |
| `equity.py` | Era 14 sweep — census-shaped per-clinic ε at a frozen clinic-mean budget → `uv run ckd-equity` → `results/equity.json`. Registration + verdict: `docs/ERAS.md` §3–§4. |
| `orchestrator.py` | Rule-based server agent: the Level-3 ε orchestrator (deterministic if-then logic + one accountant inversion; no LLM). `plan()` standardises per-clinic (batch, σ) so every clinic composes to the same target ε; `DpsgdOrchestrator` is the server-brain `FedAvg` subclass; `uniform_settings_audit` shows why uniform DP-SGD configs are incoherent across heterogeneous N. |
| `centralized.py` | Pooled-data ceiling baselines → `uv run ckd-baseline`. **Use `--clinics` when comparing against `--clinics` runs.** |
| `messages.py` | Single definition of the Flower `Message` shapes the in-process runners exchange. |
| `task.py` | Local `StandardScaler`, the imbalanced-data metrics, and the T2.5 `fairness_metrics`. |
| `data/` | `loader.py` (+ §3 missingness rules), `partition.py` (Dirichlet non-IID), `synthesize.py` (per-clinic generator + the FedMosaic public cohort), `fhir_loader.py` (the production FHIR path: canonical-contract preprocessor, de-identified, §3b), `synthesize_v2.py` (hardness-calibrated V2 generator, seven knobs + prevalence bisection), `external/` (raw-payload provenance `SOURCES.md` + V1-schema mappers for NHANES / UCI CKD / Synthea; raw downloads are gitignored), `VERSIONS.md` + `manifest_v1.sha256` (V1 frozen: `sha256sum -c` audits 48 entries). Also holds the datasets: V1 (`synthetic_ckd_data.csv`, `clinics/`, `clinics_ladder/` — frozen), `clinics_v2_*/` (eight hardness suites), `clinics_{nhanes,nhanes_s,uci,synthea}/` (real data in the V1 schema). |
| `models/` | `base.py`, `logreg.py` — the only model class in the codebase. |
| `models/protocols/` | The protocol benchmark: `common.py` (explicit logistic regression), `fedmosaic.py` (the `Strategy` subclass). |
| `model_artifact.py` | Train the federated global logreg (Flower FedAvg via `simulate.fedavg`) and export it as JSON → `ckd-export-model` → `models/global_model.json`; holds the `Scorer` for one-patient inference. |
| `webapp.py` | Local physician demo surface (stdlib-only, no framework dep): patient-features form → risk score from the exported artifact → `ckd-web`. Collects no identifiers, stores nothing. Synthetic-data demo, not a medical device. |
| `extract_features.sql` | Canonical feature contract for the **real** Tomedo→PostgreSQL export. |
| `docs/` | `PRIVACY.md`, `DEPLOYMENT.md`, `REPORT.md`, `WEBAPP.md` (the `ckd-web` physician-demo doc), `ERAS.md` (the era registry: pruned lane list, Era-14 registration, verdicts), `Law_Questions.docx`, and the source PDFs. |
| `notebooks/` | The federation walkthrough: `01` explorations + baselines, `02` update-level central/local DP sweeps, `03` the DP-SGD + SecAgg + orchestrator deployment standard, `04` the real external datasets through the identical pipeline (+ the first fairness-by-sex run), `05` V1-vs-V2 ε-response with hard acceptance gates. |
| `diagrams/` | The camera-ready diagrams: `Flipit-privacy.xml` + `.png` (the DP-SGD + SecAgg + rule-based-agent deployment standard) and `Flipit-process.xml` + `.png` (the end-to-end pipeline from practice onboarding to the consultation) — the process pair is regenerated by `make_process_diagram.py`. |
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
>
> **`data/fhir_loader.py` implements this contract over FHIR R4** — inclusion/exclusion, labs
> (eGFR/HbA1c), the KDIGO-approximating incidence label, and L0 de-identification — and
> `data/loader.py`'s `to_xy` dispatches on the label column, so the two tasks coexist without
> touching the synthetic baseline. The synthetic schema remains the evaluation task until the
> T2.1/T2.2 migration re-specifies models and benchmarks against the canonical one.

---

## 4. Models and protocols

### The model (Flower-federatable, rule 2)

| Model | Flower strategy | Role |
|---|---|---|
| Logistic regression (SGD, warm-start) | `FedAvg` / `FedProx` | reference model, grant task **T2.2**; the **only** model class in the codebase since 2026-08-28 |

### The two protocols under test

| Protocol | Strategy | Wire payload |
|---|---|---|
| **FedProx** | `flwr.serverapp.strategy.FedProx` (built-in) | coefficients |
| **FedMosaic** | `models/protocols/fedmosaic.py` (`Strategy` subclass) | predictions + expertise on a public cohort |

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
uv run ckd-simulate --protocol fedmosaic --clinics
uv run ckd-baseline --clinics             # pooled ceiling on the SAME data
uv run ckd-benchmark --rounds 20          # full protocol benchmark -> results/
uv run ckd-privacy --seeds 42 43 44 45 46 # central + local DP sweeps, SecAgg probe -> results/
uv run ckd-audit   --seeds 42 43 44 45 46 # membership-inference audit (L6) -> results/

uv run flwr run .                         # full Ray engine, real ServerApp/ClientApp
uv run flwr run . --run-config "secure-aggregation=true"   # the SecAgg+ path (L3)
uv run flwr run . --run-config "local-dp-epsilon=5.0"      # noise inside the SuperNode (L4-local)
```

**Privacy switches are run config, not environment variables.** `ClientApp` takes its mods at
construction time, so `secure-aggregation` and `local-dp-epsilon` are installed as dispatching mods
that read `ctx.run_config` when called — which is also what makes them survive the Ray worker
process boundary.

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

Closed in the privacy-hardening pass (see [PRIVACY.md](docs/PRIVACY.md)): the `patientid` export,
the SecAgg+ gap, the missing leakage audit, the loose ε accounting, and a reproducibility bug that
had invalidated every published DP figure.

1. **Local DP is unusable at a defensible ε on this cohort.** At composed ε ≈ 4.3 the worst practice
   falls to 0.397 — below the 0.495 it gets by not collaborating at all. Central DP is far cheaper
   but does **not** stop the server seeing individual updates. ~~This is the live MS4 risk, and the lever is the pilot's 25 practices ([PRIVACY.md §3](docs/PRIVACY.md)).~~ **Resolved:** closed by the record-level DP-SGD + SecAgg + orchestration standard (`dpsgd.py` + `orchestrator.py`), measured in [`notebooks/03_dpsgd_secagg_standard.ipynb`](notebooks/03_dpsgd_secagg_standard.ipynb) — AUROC 0.797–0.799 held at composed ε 0.5–8 over 5 seeds. The 0.397 collapse above stays as the evidence trail for update-level local DP (PRIVACY.md §3.3).
2. **The leakage audit covers 1 of the 3 attack families EDPB Opinion 28/2024 names.** Membership
   inference is built and passes at every setting including no-DP; **model inversion and
   reconstruction are not built** ([PRIVACY.md §5](docs/PRIVACY.md)).
3. **The audit has only ever run on synthetic data** — ten features, no eGFR. A pass is evidence
   about the method, not about the pilot model. Re-run before any anonymity claim.
4. **No test suite.** `pytest` and the `test` target were removed rather than left broken; real
   tests still need writing. The determinism gate in PRIVACY.md §7 is the closest thing to one.
5. **Fairness is audited by age band, not sex** — blocked on real *data*: the FHIR preprocessor now emits `geschlecht` (§3b), so the blocker is no longer the schema.
6. **XGBoost federated poorly here** (0.729, 5-seed mean (0.712 was the seed-42 only figure) vs a 0.860 pooled ceiling) and could not be protected by
   SecAgg+. **Scope reduction 2026-08-28:** both the MLP and XGBoost/FedXgbBagging model paths were evaluated, then removed from scope — the codebase is
   now logreg-only, and logistic regression is the deployment model.
7. **The federated analytics path is not implemented here.** The legal assessment distinguishes it
   from the AI training path; this repo only implements the latter. "Analytics" is not a Flower
   concept — it appears nowhere in flwr 1.33 — so the *structural* answers (who sees what, whether
   SecAgg+/DP apply) carry over unchanged, but the disclosure-control questions (cell suppression,
   k-anonymity, differencing across repeated queries) need the other team's indicator spec.
8. **Row order in the SQL export still follows `patientid`** even though the column is dropped — a
   weak ordering channel, documented in the SQL rather than removed.
