---
name: flip-it-ckd
description: Use when working anywhere in FL-CKD-Flower — the FLIP-IT federated CKD risk model (logistic regression only). Covers Flower.ai federation (FedAvg, FedProx, FedMosaic, ServerApp/ClientApp, SuperLink/SuperNode), the practice/clinic datasets, HL7 FHIR extraction, differential privacy and secure aggregation, dual-level AUROC/sensitivity metrics, and bias analysis. Trigger on any request to add or change a model, run or compare federation protocols, touch privacy layers, deploy to a SuperNode, or interpret CKD results.
---

# FLIP-IT CKD — working rules

`CLAUDE.md` says what is true about this project. This says what to do, in what order, and what to
refuse. Read it before editing; it takes thirty seconds and prevents the drift that caused the last
cleanup.

## 1. Stop-first gate

Answer all four **before** you edit. Any ✋ means stop and surface the conflict — do not proceed and
explain afterwards.

| # | Question | ✋ if |
|---|---|---|
| 1 | Is the file inside `FL-CKD-Flower/`? | No. Nothing about this project lives elsewhere. `~/Fed_Agent/` is not ours. |
| 2 | Can this model federate with a real `flwr` strategy? | No — see §2. |
| 3 | Does this change move patient rows across a boundary? | Yes, unless it is the explicitly-labelled `centralized.py` ceiling. |
| 4 | Does it change seeds, splits, or features inside a live comparison? | Yes. Vary only the thing under test. |

Never hand-roll a Flower primitive. Need custom federation? Subclass
`flwr.serverapp.strategy.Strategy` (or `FedAvg`) — that is Flower's own extension point. Reimplementing
FedAvg's averaging or the transport is the violation.

## 2. Model admissibility test

Name the `flwr.serverapp.strategy` class that will aggregate it. No answer → it does not go in this
repo, **not even as a "quick centralized baseline"**.

| Model shape | Strategy | Verdict |
|---|---|---|
| Logistic regression (the repo's only model) | `FedAvg`, `FedProx` | ✅ |
| Random forest, GBT/HistGB, LightGBM, CatBoost, k-NN, SVM, NN | none in scope | ❌ reject |

Before writing a new strategy, check it isn't already shipped:
`python -c "import flwr.serverapp.strategy as s; print(dir(s))"`. FedAvg, FedProx, FedAdam/Adagrad/Yogi,
FedAvgM, FedMedian, FedTrimmedAvg, Krum, MultiKrum, Bulyan, QFedAvg and the
four DP wrappers all exist. **SCAFFOLD does not** — it is only in Flower Baselines as a standalone
PyTorch project.

## 3. Command surface

`uv run` only. Everything else is wrong.

| Task | Command |
|---|---|
| Build/repair env | `uv sync --extra dev --extra notebook` |
| Regenerate clinics | `uv run ckd-clinics --clinics 10` |
| Federated run | `uv run ckd-simulate --clinics` |
| Protocol run | `uv run ckd-simulate --protocol fedmosaic --clinics` |
| Pooled ceiling | `uv run ckd-baseline --clinics` |
| Full benchmark | `uv run ckd-benchmark --rounds 20` |
| Privacy sweep | `uv run ckd-privacy --seeds 42 43 44 45 46` |
| Real Flower stack | `uv run flwr run .` |
| DP-SGD + SecAgg standard, live | `uv run flwr run . --run-config "dpsgd-epsilon=8.0 alpha=5.0 secure-aggregation=true"` (α=5.0 keeps sim partitions above the census floor) |
| Lint | `uv run ruff check .` |

Wrong → right:
- `pip install …` → `uv add …` (rule 4)
- `python simulate.py` → `uv run ckd-simulate`
- `pytest` → there is no test suite; don't invoke one
- `uv run ckd-baseline` alone when comparing to `--clinics` → **always match the data source**, or the "price of privacy" you compute is just a dataset difference

## 4. Where things live

- `to_xy()` in `data/loader.py` — the **only** preprocessing entry point. Applies the §3 missingness rules. Don't write another.
- `build_client_from_frame()` in `client_app.py` — the **only** client constructor. Handles local split + local scaler + model build.
- `messages.py` — the only definition of the Flower `Message` shapes. Don't rebuild them inline.
- `weighted_and_worst` in `server_app.py` — the dual-level aggregator. Every strategy takes it.
- `dp.py` — the **only** DP accountant definition (`epsilon_rdp`, `sigma_for_epsilon` = RDP via
  Google `dp_accounting`), shared by sweeps, audit, and the live server. Never compose by hand.
- `dpsgd.py` — the record-level DP-SGD trainer + in-process runner (`run_standard`, `run_epsilon_sweep`).
- `orchestrator.py` — the **rule-based server agent** (no LLM): `AGENT_RULES` is the cited rule table;
  `plan()`/`DpsgdOrchestrator` decide per-clinic (batch, σ). Extend planning here, never beside it.
- `diagrams/` — the standard's diagrams; the PNG embeds its drawio source — edit/update the XML,
  and re-export the raster in draw.io (no renderer in this repo).
- `models/protocols/common.py` — explicit logistic regression for protocols needing gradient control (FedProx's proximal term, FedMosaic's weighted objective).
- Datasets live in `data/`; generated `data/clinics/` and `results/` are gitignored.

## 5. Metrics contract

- **AUROC + sensitivity, never accuracy alone.** A model that never predicts CKD still looks accurate on a low-prevalence cohort.
- **Global *and* worst-practice, every round.** Construct every strategy with
  `evaluate_metrics_aggr_fn=weighted_and_worst`. Flower's default aggregator silently drops the
  `_worst` keys — if you see `worst practice nan`, that's the bug.
- **Anything headed for a report** also gets `task.fairness_metrics()` — the four measures the
  Projektantrag names for T2.5 (demographic parity, equal opportunity, equalized odds, calibration
  by group).
- **Sanity rule:** on `data/synthetic_ckd_data.csv` the correct answer is **AUROC ≈ 0.5**. It is a
  signal-less wiring placeholder. A higher number there is a leak or a bug, never a win. Use it as
  the negative control; use `data/clinics/` for real comparisons.

## 6. Privacy defaults

DP and SecAgg stay **off by default** — baselines must be clean reference numbers — and are enabled
explicitly. Full architecture in [`docs/PRIVACY.md`](../../../docs/PRIVACY.md).

Standing prohibitions:
- Never add `patientid` (or any identifier) to a feature frame.
- Never log a **named** practice's metrics below the cohort floor — use `--metric-privacy` / `--min-cohort-size`.
- Never pool practice data outside `centralized.py`.
- Never type a DP ε derived from hand arithmetic into docs or reports — the accountant is `dp.py` (`epsilon_rdp`/`sigma_for_epsilon`), and only a **composed** budget over the configured rounds may be quoted (δ = 1e-5).

One constraint to state rather than paper over: **SecAgg+ is legacy-path
only in flwr 1.33** — it needs `LegacyContext` and cannot compose with `strategy.start()`.
(The repo is logreg-only, so every payload is a maskable numeric vector; the tree path that SecAgg+
could not protect was removed on 2026-08-28.)

## 7. Failure modes already hit here

| Symptom | Cause | Fix |
|---|---|---|
| `worst practice nan` | Strategy built without `evaluate_metrics_aggr_fn` | Pass `weighted_and_worst` |
| `ImportError: cannot import name 'ClientApp' from 'flwr.app'` | 1.33 moved them | `ClientApp` ← `flwr.clientapp`; `ServerApp`/`Grid` ← `flwr.serverapp`; records ← `flwr.app` |
| `SGDClassifier` dtype error | float32/float64 mismatch | Keep `coef_`/`X` float64; exchange float32 |
| Centralized ceiling ≈ 0.5 while federated ≈ 0.8 | `ckd-baseline` defaulted to the flat CSV | Add `--clinics` |
| All protocols look identical | Run too short, or the flat dataset | Use `--clinics`, ≥20 rounds |
| `ValueError: ... below the smallest candidate batch 8` | Simulation partition under the DP-SGD census floor | The orchestrator correctly refuses unsafe plans — raise `alpha` (5.0) or shrink `num-practices` |

## 8. Reporting standards

Every number traces to a committed command someone else can rerun. Report negative and null results
plainly — "FedProx ≈ FedAvg here" and "local training is nearly as good" are findings, not failures.
When something can't be verified, say so and name the blocker rather than implying coverage.
