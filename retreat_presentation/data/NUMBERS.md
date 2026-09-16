# FLIP-IT retreat deck — measured numbers digest

Every number the deck may use, with its provenance. **Do not invent, round differently, or
extrapolate.** Provenance tags:

- `[live …]` — measured in this checkout on 2026-09-16; machine-readable source in
  `retreat_presentation/data/*.json` (identical files in `results/`).
- `[docs …]` — published in-repo docs; measured earlier, not re-run here.

## Cohort (task setting)

`[live data/benchmark.json → datasets.clinics.cohort]`

- 10 synthetic GP practices, **3,276 patients**, panel sizes **161–532**
  (sizes: 162, 492, 434, 331, 328, 532, 161, 455, 216, 165)
- CKD prevalence per practice: **8.3 %–46.0 %**
- 10 features: `age_years`; six `dx_*` flags (hypertonie, diabetes, khk, adipositas,
  herzinsuffizienz, hyperurikaemie); three `years_since_*` durations. Label: `ckd_stage3plus`.
- Non-IID on covariates (5 clinical archetypes), labels (prevalence), and quantity (panel size);
  one shared true logistic risk model behind all practices.

## Problem 1 — data modality → model

**Protocol benchmark, 5 seeds (42–46), 20 federation rounds, per-clinic federation.
`[live data/benchmark_5seed_summary.json; per-seed files benchmark_seed*.json]`**
Identical to the published table (docs/REPORT.md §2) at three decimals.

| | Global AUROC | Worst practice | Sensitivity | Uplink bits/client/round |
|---|---|---|---|---|
| **Pooled ceiling** (not deployable) | **0.861** (seed 42; 5-seed mean 0.853 ± 0.010) | n/a | 0.808 (seed 42; 5-seed 0.778) | n/a |
| **FedProx** (μ = 0.1) | **0.808 ± 0.008** | **0.675 ± 0.054** | 0.703 | 352 |
| **FedMosaic** | 0.800 ± 0.010 | 0.661 ± 0.068 | 0.709 | 3,600 |
| FedAvg (baseline) | 0.808 ± 0.008 | 0.675 ± 0.054 | 0.702 | 352 |
| local (baseline, no federation) | 0.789 ± 0.008 | 0.495 ± 0.126 | 0.699 | 0 |

Headline gaps: federated vs ceiling −0.053; federated vs training alone on the **worst
practice +0.180** (0.495 → 0.675). FedProx ≡ FedAvg to three decimals.

**Negative control** (signal-less flat CSV, max |feature–label correlation| 0.031):
`[live benchmark_5seed_summary.json → datasets.flat]` — no protocol exceeds ≈ 0.53 AUROC
(final means 0.485–0.524; ceiling 0.503). Learning is real on the clinics cohort.

**Round count.** `[docs REPORT.md §2, runs extended to 50 rounds]` — the logistic protocols
converge by round ~**10** and stay flat to round 50; `local` drifts down (overfitting). Cutting
50 → 10 rounds costs no accuracy and reduces the composed DP budget ~2.7× (ε 22.0 → 8.1 at
σ = 2.0). `num-server-rounds` defaults to 10 for exactly this reason.

**Model-class decision (historical, removed 2026-08-28).** `[docs REPORT.md §2]` — XGBoost
pooled ceiling 0.860 ≈ logreg's 0.861, but federated bagging reached only **0.729** at **61× the
bandwidth**, degraded monotonically with more rounds (0.780 @ r5 → 0.686 @ r50), and its
serialized split thresholds could not be protected by SecAgg at all. MLP + XGBoost paths deleted;
the repo is now logreg-only by measurement, not by taste.

## Problem 2 — protocol ↔ model + privacy

`[docs REPORT.md §3 / PRIVACY.md §6]`, wire payloads confirmed in live benchmark:

| Protocol | On the wire | Bits/client/round | Disclosure surface |
|---|---|---:|---|
| FedAvg / FedProx | model coefficients | 352 | parameter vector fit to this practice's patients — the gradient-inversion target |
| FedMosaic | predictions + expertise on a **public** cohort | 3,600 | no parameter vector leaves the practice — nothing to invert |

**Convergence economics:** fewer rounds = less composed ε (see round-count point above).

## Problem 3 — privacy ↔ law

**Central DP (server-side noise, Flower `DifferentialPrivacyServerSideFixedClipping`, clip 1.0,
20 rounds, 5 seeds).** `[live data/privacy.json → dp_sweep]` — reproduces docs/PRIVACY.md §3.1
exactly:

| σ | Global AUROC | Worst practice | ε (RDP, δ=1e-5) |
|---:|---|---|---:|
| 0.00 | 0.808 ± 0.008 | 0.675 ± 0.054 | — |
| 0.10 | 0.808 ± 0.007 | 0.676 ± 0.056 | 1211.8 |
| 0.25 | 0.807 ± 0.008 | 0.669 ± 0.054 | 244.0 |
| 0.50 | 0.803 ± 0.011 | 0.681 ± 0.033 | 81.1 |
| 1.00 | 0.794 ± 0.014 | 0.646 ± 0.061 | 30.1 |
| 2.00 | 0.773 ± 0.017 | 0.608 ± 0.097 | **12.3** |

**Local DP (noise inside the practice's SuperNode, Flower `LocalDpMod`, same seeds/rounds).**
`[live data/privacy.json → local_dp_sweep]` — reproduces §3.2 exactly:

| ε/round | ε composed | Global AUROC | Worst practice |
|---:|---:|---|---|
| off | — | 0.808 ± 0.008 | 0.675 ± 0.054 |
| 50 | 1283.4 | 0.806 ± 0.008 | 0.677 ± 0.044 |
| 20 | 257.6 | 0.791 ± 0.017 | 0.650 ± 0.052 |
| 10 | 85.0 | 0.768 ± 0.041 | 0.597 ± 0.061 |
| 5 | 31.4 | 0.754 ± 0.009 | 0.574 ± 0.053 |
| 1 | **4.3** | 0.627 ± 0.057 | **0.397 ± 0.130** |

**The matched-ε comparison and the collapse.** At composed ε ≈ 30/31: central 0.794/0.646 vs
local 0.754/0.574 — ≈ 0.04 global and 0.07 worst-practice is the measured price of the server
never holding an un-noised update. At ε ≈ 4.3 the worst practice (0.397) falls **below the 0.495
it achieves by not collaborating at all** — update-level local DP is not usable at a defensible ε
on this cohort.

**The legal reading.** `[docs PRIVACY.md §3.3]` — central DP does not answer the question the
legal assessment asks: the server receives every practice's un-noised update *before* noising
during aggregation. Only local DP (or SecAgg, or DP-SGD inside the clinic) changes what the
aggregating party sees. Note flwr's `DifferentialPrivacyClientSideFixedClipping` is, per its own
docstring, "central DP with client-side clipping" — the noise is still server-side.

**Deployment standard (record-level).** `[docs PRIVACY.md §3.5; measured in
notebooks/03_dpsgd_secagg_standard.ipynb, 5 seeds]` — patient-level DP-SGD inside each clinic
(Poisson sampling, per-sample clipping, Gaussian noise) **+ SecAgg+ on the wire + a rule-based
server agent** that standardises ε across clinic sizes:

| target ε* per clinic | Global AUROC | Worst practice |
|---:|---|---|
| σ = 0 reference (same runner) | 0.796 ± 0.010 | 0.659 ± 0.050 |
| 0.5 | 0.798 ± 0.008 | 0.660 ± 0.051 |
| 2 | 0.797 ± 0.011 | 0.649 ± 0.060 |
| 8 | 0.798 ± 0.011 | 0.656 ± 0.050 |

Orchestrator detail `[docs PRIVACY.md §3.5]`: identical settings are incoherent across census
sizes (batch 64, σ 1.5 → ε = 1.10 for a 50 k-patient clinic vs ε = 16.04 for 500); the agent
inverts the RDP accountant per clinic — σ spans 2.57–3.47 at ε* = 8, 8.32–11.68 at ε* = 2,
largest σ on the smallest census. Per-clinic plans are certified `achieved ε ≤ ε*` before issue.

**Orchestration verdicts** `[docs ERAS.md §4, §6]`: census-shaped ε allocation is **null** at
this cohort (vulnerable-trio Δ(γ=+0.5−γ=0) = −0.0011, 95 % CI [−0.0040, +0.0018]) and flat
through a 100× census-spread ladder — *the equity lever is census scale, not allocation shape*;
the uniform-ε* standard carries no measurable allocation equity tax at pilot-realistic scale.

**SecAgg+.** `[live data/privacy.json → secagg]` — flwr 1.33 ships SecAgg+ only in legacy
namespaces; reachable by driving `DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))` with a
`LegacyContext` from a modern ServerApp — **implemented in this repo** (`secure-aggregation=true`,
12 practices, fit+evaluate, 0 failures). Semi-honest threat model; participation visible; the
aggregate is still model parameters.

**Leakage audit (EDPB Opinion 28/2024, family 1 of 3).**
`[live data/audit.json]` — reproduces docs/PRIVACY.md §5: loss-threshold and shadow-model
membership-inference attacks find no usable signal at **any** privacy setting including no DP
(no-DP: 0.501 ± 0.006 / 0.502 ± 0.015; central σ=0.1–2: 0.502–0.504 / 0.502–0.509; local ε=1–50:
0.492–0.503 / 0.494–0.506; pass bar: 95 % CI upper bound ≤ 0.520). Positive control (deliberately
overfit, n=40): **0.655 / 0.659** — the testbed fires when leakage exists. Explanation: model
capacity (11 parameters over 3,276 patients), **not** privacy engineering — and it is not an
anonymity proof: 1 of 3 attack families, synthetic data, no eGFR; inversion and reconstruction
remain open.

## Project frame facts (docs/CLAUDE.md)

- **FLIP-IT**: NEXT.IN.NRW / EFRE-JTF NRW innovation project; consortium docport GmbH
  (coordinator, Dr. med. Nicolas Conze), IKIM Universitätsmedizin Essen, AG Trustworthy ML at
  RUB (Prof. Dr. Michael Kamp), Jorzig & Partner (legal).
- Goal: federated-learning infrastructure across **25 GP practices** training a CKD risk model on
  HL7-FHIR–harmonised routine-care data; **the model travels, patient data never leaves the
  practice**; Flower.ai coordination; hardened with DP + Secure Aggregation.
- Consortium FHIR server: **Helios FHIR** (SQL-on-FHIR); canonical feature contract
  `extract_features.sql` (adds eGFR — the strongest CKD predictor — and sex, needed for the T2.5
  fairness analysis).
- Model = logistic regression (grant task T2.2); T2.3 privacy features; T2.4 validation; T2.5
  privacy/bias audits. Milestones: **MS4 (month 18) = validated CKD model incl. DP + SecAgg**;
  MS5 (m24) pilot in 25 practices.
- Stage: pre-kickoff **baseline sandbox on synthetic data** — de-risking before kickoff.
