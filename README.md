# FLIP-IT — federated CKD risk models on Flower

FLIP-IT trains chronic-kidney-disease risk models across GP practices without patient data leaving
any practice. The model travels to the data: a practice fits the global model to its own patients,
sends back coefficients, the server averages them, and the next global model goes out. No patient
row, and nothing per-patient, ever crosses that boundary.

This repository is the project's pre-kickoff baseline: the federated pipeline, the privacy stack,
and the numbers they produce, run on synthetic cohorts while real practice data is onboarded.
Every model here is a logistic regression, aggregated by Flower's own strategies — FedAvg,
FedProx and FedMosaic. That is not a starting point: logistic regression is the deployment model,
and it is the only model class in the tree.

What is published here is **the code, the notebooks, and the raw results**. Internal docs,
presentations, the paper and the practice extracts stay with the maintainers — see
*What is not in this repository* at the end.

## What the pipeline does

1. **Give every practice its own cohort.** `uv run ckd-clinics` generates ten non-IID practices
   from five clinical archetypes that differ in size, age and comorbidity mix but share one true
   risk model — so there is a global truth to learn. For a real practice, the cohort comes from
   its own Helios FHIR server (`data/fhir_loader.py`, or the SQL-on-FHIR views in
   `data/helios.py`), or from the bundled two-stage CheCKD SQL run against its PostgreSQL export
   (`data/sql_loader.py`). Both reproduce the same contract as `extract_features.sql`.

2. **Train locally, on the practice's own rows.** The SuperNode starts from the global weights the
   server sent and fits the logistic regression on its own patients.

3. **Send model state, not data.** The reply carries coefficients and the example count that
   weights them; the evaluating pass adds aggregate quality numbers. Nothing per-patient is ever
   serialised.

4. **Average and iterate.** `server_app.py` combines the practices' updates with Flower's FedAvg
   and broadcasts the next global model. `simulate.py` drives the same strategies in-process when
   you do not need the Ray transport.

5. **Report both levels, every round.** `weighted_and_worst` logs the sample-weighted **global**
   AUROC and sensitivity *and* the **worst single practice** — a model that collapses on one
   outlier practice cannot hide behind a strong average.

6. **Protect the run when real patients are in it.** Record-level DP-SGD inside each practice,
   SecAgg+ masking on the wire, and a rule-based server agent that gives every clinic its own
   (batch, σ) so heterogeneous clinics compose to the same ε. All of it is off by default;
   the baselines are meant to be clean reference numbers.

7. **Keep the model, drop everything else.** `uv run ckd-export-model` writes the finished
   federated model — feature order, scaler, coefficients — to `models/global_model.json`, which
   `uv run ckd-web` serves as a local physician form. On the `sql` path the practice's temporary
   tables are dropped when the run ends.

## What we measured

Ten practices, 3,276 patients, 20 rounds, five seeds (mean ± s.d.). Raw JSON in [`results/`](results/).

| | AUROC | Worst practice | Uplink per practice per round |
|---|---|---|---|
| *pooled ceiling* (non-federated) | 0.853 ± 0.011 | *n/a* | *n/a* |
| **FedProx** | 0.808 ± 0.009 | 0.675 ± 0.061 | 352 bits |
| *FedAvg* (baseline) | 0.808 ± 0.009 | 0.675 ± 0.061 | 352 bits |
| **FedMosaic** | 0.800 ± 0.011 | 0.661 ± 0.076 | 3,600 bits |
| *local* (no collaboration) | 0.789 ± 0.009 | 0.495 ± 0.141 | 0 |

- Federation lands within **0.045** of the pooled ceiling, with no practice sharing a patient row.
- **The win is equity, not average accuracy.** Training alone and the best protocol differ by 0.019
  in global AUROC, and by **0.180** on the worst practice.
- **More rounds do not help.** The logistic protocols converge by round 10 and are flat after, so
  extra rounds only spend privacy budget.
- **The privacy standard costs nothing measurable here.** DP-SGD + SecAgg + orchestration holds
  AUROC 0.797–0.798 at composed ε 0.5–8 over five seeds ([notebook 03](notebooks/03_dpsgd_secagg_standard.ipynb)).
  Update-level local DP at a comparable budget does *not*: the worst practice falls to 0.397, below
  the 0.495 it gets by not collaborating at all ([notebook 02](notebooks/02_differential_privacy_epsilon.ipynb)).
- **The leakage audit passes.** Membership-inference attack AUC 0.501 ± 0.006 in the no-DP runs,
  with a positive control at 0.655 proving the attack works. Model inversion and reconstruction are
  not implemented — the gap is named in `results/audit.json`.

## Install and run

Requires [`uv`](https://docs.astral.sh/uv/); no manual venv activation.

```bash
make setup      # create/repair the venv and install everything
make clinics    # generate the per-clinic cohorts (everything below needs them)
make baseline   # pooled-data ceiling on the same cohorts
make simulate   # federated logistic regression, non-IID
make help       # every target
```

The same commands as `uv` directly, plus the rest of the surface:

```bash
uv run ckd-simulate --clinics                        # FedAvg, in-process
uv run ckd-simulate --clinics --protocol fedmosaic   # the personalized protocol
uv run ckd-baseline --clinics                        # ceiling — match the data source!
uv run ckd-benchmark --rounds 20                     # full protocol benchmark -> results/
uv run ckd-privacy --seeds 42 43 44 45 46            # DP sweeps + SecAgg probe
uv run ckd-audit   --seeds 42 43 44 45 46            # membership-inference audit

uv run flwr run .                                    # full Flower stack (Ray, real ServerApp/ClientApp)
uv run flwr run . --run-config "dpsgd-epsilon=8.0 alpha=5.0 secure-aggregation=true"
```

> Comparing a federated run against the ceiling? Pass `--clinics` to both. `ckd-baseline` defaults
> to the flat placeholder CSV, whose ceiling is AUROC ≈ 0.5 by construction — the difference is the
> dataset, not the privacy.

## Privacy

DP and Secure Aggregation are **off by default** and enabled per run. The mechanism is chosen by
run-config key, never by an environment variable, so it survives the SuperNode process boundary:

| Key | Effect |
|---|---|
| `dpsgd-epsilon` | The standard: per-record DP-SGD in each practice + a server-side census and ε-orchestrator that plans each clinic's (batch, σ). Composes with `secure-aggregation` |
| `secure-aggregation` | SecAgg+: the server opens only the sum, never one practice's update |
| `central-dp-epsilon` | Noise during aggregation; protects the released model, not the wire |
| `local-dp-epsilon` | Noise inside the SuperNode before transmission; expensive at this cohort size |
| `min-cohort-size`, `metric-privacy` | Suppress small cohorts and identify no practice in the logs |

ε is never derived by hand: `dp.py` holds the single RDP accountant (`epsilon_rdp`,
`sigma_for_epsilon`), and the sweeps, the audit and the live server all invert the same one. Only a
**composed** budget over the configured rounds is ever quoted.

## Notebooks

Each one runs top to bottom and is the audit trail for its numbers.

| Notebook | What it shows |
|---|---|
| [01](notebooks/01_explore_and_baselines.ipynb) | The federation itself: protocols, convergence, the centralized gap |
| [02](notebooks/02_differential_privacy_epsilon.ipynb) | DP accounting in detail, and what update-level local DP costs |
| [03](notebooks/03_dpsgd_secagg_standard.ipynb) | The deployment standard, built and measured: DP-SGD + SecAgg + orchestrator |
| [04](notebooks/04_external_ckd_data.ipynb) | NHANES, UCI CKD and Synthea pushed through the identical pipeline |
| [04-b](notebooks/04_helios_sql_federated_fedprox.ipynb) | Helios SQL-on-FHIR as the cohort source, then FedProx over plain weights |
| [05](notebooks/05_harder_synthetic_epsilon.ipynb) | Harder synthetic data, where the ε-response finally becomes visible |
| [06](notebooks/06_real_practice_ckd_progression.ipynb) | Real practice d001: CKD progression on 1/2/5-year horizons |
| [07](notebooks/07_landmark_time_progression_d001.ipynb) | Same practice, six yearly landmarks: does AUPRC improve over time? |

Notebooks 06 and 07 read the practice extracts under `practice_1/`, which are not published here —
they run only on a machine that holds them. Everything else regenerates locally.

## Repository layout

```text
.
├── client_app.py, server_app.py     Flower ClientApp / ServerApp (1.33 Message API)
├── simulate.py                      in-process runner over Flower's real strategies
├── benchmark.py                     protocol benchmark -> results/
├── privacy.py                       central / local DP sweeps + SecAgg probe
├── dpsgd.py                         record-level DP-SGD + the SecAgg-masked runner
├── orchestrator.py                  the rule-based server agent + DpsgdOrchestrator(FedAvg)
├── dp.py                            the single (ε, δ) accountant
├── audit.py                         membership-inference audit
├── equity.py, phasemap.py           ε-pricing and census-spread sweeps
├── centralized.py                   pooled-data ceiling (non-federated reference)
├── model_artifact.py                export the federated model -> models/global_model.json
├── webapp.py                        local physician demo (stdlib only, stores nothing)
├── task.py, messages.py             metrics + fairness gaps; the Message shapes
├── extract_features.sql             canonical feature contract for a real export
├── models/                          logreg — the only model — plus protocols/ (FedMosaic)
├── data/                            loaders, generators, the Helios host, the CheCKD SQL
├── notebooks/                       the walkthrough, 01–07
├── results/                         raw JSON from the runs quoted above
└── scripts/, Makefile               thin wrappers around the uv commands
```

## Deployment

SuperLink on central infrastructure, one SuperNode per practice, TLS and key-based node
authentication throughout. The SuperLink address and CA live in `~/.flwr/config.toml`; the
practice-local settings (its SQL DSN, its model path, its Helios FHIR base URL) live in the
SuperNode's own secrets file, and the private key never leaves the machine.

```bash
flwr supernode register praxis_01.pub flipit-prod     # admin, once per practice
flwr run . flipit-prod --stream \
  --run-config "num-practices=25 num-server-rounds=10 data-source='sql' dpsgd-epsilon=3.0 secure-aggregation=true"
```

`num-practices` must equal the number of connected SuperNodes or the run never starts.

## Development

`uv` only — `uv add` for dependencies, `uv run` for everything else. `make lint` runs ruff.

There is no test suite: it was removed rather than left broken, and the closest thing to one is the
determinism gate (one global seed, derived per-practice seeds, identical splits and features across
every model in a comparison — only the thing under test varies). Negative results are reported as
findings, not hidden: FedProx ≈ FedAvg here, and local training is nearly as good on average.

Branch topology: `main` is frozen infrastructure, `dev` integrates, and every `experiment/` branch
names a data kind — tabular, image, time-series — never a methodology.

## What is not in this repository

- **Internal docs** — the project memory, the privacy architecture, the deployment runbook, the
  report. Some module docstrings still cite their sections (`CLAUDE.md §…`, `docs/PRIVACY.md`).
- **Presentations, decks, the paper** and their build products.
- **Datasets.** The per-practice cohorts are generated by the committed generators and never
  committed; the external cohorts are rebuilt from their public sources by the mappers in
  `data/external/`. The one exception is the 46 KB flat synthetic fixture,
  `data/synthetic_ckd_data.csv` — a signal-less wiring placeholder (AUROC ≈ 0.5 is the correct
  answer on it) that the loaders and the Helios seeder need; it carries no real patient data.
- **Practice extracts.** Real practice data lives with the practice. Notebooks 06 and 07 are the
  only artefacts that assume its presence.
