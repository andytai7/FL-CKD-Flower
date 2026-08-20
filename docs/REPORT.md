# Federated CKD risk modelling — accuracy and privacy

Pre-kickoff baseline results for **FLIP-IT**. Three federation protocols, one logistic-regression
model class, measured on synthetic GP-practice data, with the cost of each privacy layer measured
rather than asserted.

**Reproduce everything:**

```bash
uv sync --extra dev --extra notebook
uv run ckd-clinics --clinics 10
uv run ckd-benchmark --rounds 20                      # -> results/benchmark.json
uv run ckd-privacy --rounds 20 --seeds 42 43 44 45 46 # -> results/privacy.json
```

Or run the whole experiment interactively, with charts, in
[`notebooks/01_explore_and_baselines.ipynb`](../notebooks/01_explore_and_baselines.ipynb) —
that notebook is the reproducible source for every number below.

---

## 1. What was tested

| | |
|---|---|
| **Cohort** | 10 synthetic GP practices, 3,276 patients, panel sizes 161–532, CKD prevalence 8.3 %–46 % |
| **Non-IID on** | covariate shift (5 clinical archetypes), label shift (prevalence), quantity shift (panel size) |
| **Model** | logistic regression, 10 features + intercept (`fedxgb` uses XGBoost — a different model class) |
| **Held fixed** | seeds, local 80/20 splits, local `StandardScaler`, local optimiser. Only the protocol varies |
| **Seeds** | 5 (42–46) — every figure is a mean ± s.d. across them |
| **Rounds** | 20 for the headline table; runs extended to 50 to test convergence |
| **Negative control** | `synthetic_ckd_data.csv`, max \|feature-label correlation\| 0.031 |

### The three protocols

| Protocol | Strategy | Wire payload |
|---|---|---|
| **FedProx** | `flwr.serverapp.strategy.FedProx` (built-in), μ = 0.1 | model coefficients |
| **FedXgbBagging** | `flwr.serverapp.strategy.FedXgbBagging` (built-in) | serialized decision trees |
| **FedMosaic** | `models/protocols/fedmosaic.py`, a `Strategy` subclass implementing Algorithm 1 of `docs/2507.00259v3.pdf` | binary predictions + expertise on a public cohort |

Two **reference baselines** make the protocols interpretable: `local` (no collaboration at all) and
`fedavg` (plain averaging — isolates what FedProx's proximal term actually bought).

---

## 2. Accuracy

**5 seeds, 20 rounds, per-clinic federation. Mean ± s.d.** Pooled-data ceiling — one model trained
on every practice's data together, which reality does not permit — is **0.861**.

| | AUROC | Worst practice | Sensitivity | Uplink bits/round |
|---|---|---|---|---|
| *centralized ceiling* | *0.861* | *n/a* | *0.808* | *n/a* |
| **FedProx** | **0.808 ± 0.009** | **0.675 ± 0.060** | 0.703 | 352 |
| **FedMosaic** | 0.800 ± 0.011 | 0.661 ± 0.076 | 0.709 | 3,600 |
| **FedXgbBagging** | 0.729 ± 0.021 | 0.555 ± 0.050 | 0.599 | 20,902 |
| *fedavg* (baseline) | *0.808 ± 0.009* | *0.675 ± 0.061* | *0.702* | *352* |
| *local* (baseline) | *0.789 ± 0.009* | *0.495 ± 0.141* | *0.699* | *0* |

The ceiling has no worst-practice figure by construction: pooling destroys the practice boundary the
measure is defined over. It is the accuracy target, not a fairness comparator.

### The five findings

**1. Federation lands within 0.054 AUROC of the pooled ceiling** — 0.808 against 0.861 — without any
practice sharing a patient row. That is the core FLIP-IT claim and it holds.

**2. The benefit is equity, not average accuracy.** Global AUROC separates training-alone from the
best protocol by 0.019. The worst practice separates them by **+0.180** — from 0.495 to 0.675. A
small or skewed practice cannot learn a usable CKD model alone, and federation is what fixes that.
For a pilot whose premise is exactly this, worst-practice AUROC is the number to report, and it is
also the T2.5 bias measure.

**3. FedProx and FedAvg are indistinguishable**, to three decimals on both measures. The proximal
term has almost no client drift to correct at two local epochs per round. A null result, reported as
one — retest if local epochs increase.

**4. FedMosaic does not win on accuracy.** It sits ~0.008 below the weight-sharing protocols on
global AUROC and ~0.014 below on worst-practice. Its case is its **disclosure profile**, not its
score: it never transmits a model. A single-seed run (seed 42) showed FedMosaic ahead; across five
seeds that reverses, and worst-practice variance is wide enough (± 0.06–0.14) that single-seed
protocol rankings should not be trusted.

**5. XGBoost federates badly, and the failure is federation-specific.** Its pooled ceiling is 0.860 —
indistinguishable from logistic regression's 0.861 — but bagging reaches only 0.729, at 59× the
bandwidth.

### More rounds do not help — and hurt the tree path

Read from the same runs, extended to 50 rounds:

| rounds | fedavg | fedprox | fedmosaic | local | fedxgb |
|---:|---|---|---|---|---|
| 5 | 0.805 | 0.805 | 0.800 | 0.791 | **0.780** |
| 10 | 0.807 | 0.807 | 0.800 | 0.790 | 0.757 |
| 20 | 0.808 | 0.808 | 0.800 | 0.789 | 0.729 |
| 35 | 0.808 | 0.808 | 0.800 | 0.786 | 0.703 |
| 50 | 0.808 | 0.808 | 0.799 | 0.785 | **0.686** |

The logistic protocols are converged by round ~10 and flat thereafter. `local` drifts *down* as each
practice overfits its own panel. **`fedxgb` peaks at round 5 and degrades monotonically** —
FedXgbBagging appends every practice's new trees to one ensemble each round, and under non-IID data
those trees encode contradictory local rules, so more rounds means more contradiction.

This has a direct privacy consequence: **cutting the round count from 50 to 10 costs no accuracy and
cuts the DP budget five-fold.** It is the cheapest ε improvement available (§3).

### Negative control

On the signal-less flat CSV the highest score any protocol reaches is **0.528**, against a pooled
ceiling of 0.513. Nothing learned signal that is not there — which is what licenses reading the
clinics numbers as real.

## 3. Privacy

### What each protocol discloses

| Protocol | Payload | Disclosure surface |
|---|---|---|
| FedProx / FedAvg | model coefficients | A parameter vector fitted to this practice's patients — the object gradient-inversion attacks target |
| FedXgbBagging | serialized trees | **Split thresholds are literal patient feature values.** The highest-disclosure payload of the three |
| FedMosaic | predictions + expertise on a *public* cohort | Opinions about patients who are already public. No parameter vector exists for the server to invert |

FedMosaic's advantage is qualitative, not incremental: there is no model to steal. It costs ~10×
FedProx's bandwidth (3,600 vs 352 bits per practice per round) — but still **6× less** than
FedXgbBagging, which has the worst privacy profile *and* the worst accuracy.

### Cost of Differential Privacy

Flower's real `DifferentialPrivacyServerSideFixedClipping` over FedAvg, clipping norm 1.0, 20
rounds, **5 seeds** (mean ± std). ε from an **RDP accountant** (`dp-accounting`), δ = 1e-5:

| Noise σ | AUROC | Worst practice | ε (RDP) | AUROC cost |
|---:|---|---|---:|---:|
| 0.00 | **0.808 ± 0.008** | 0.675 ± 0.054 | — | — |
| 0.10 | 0.808 ± 0.007 | 0.676 ± 0.056 | 1211.8 | −0.000 |
| 0.25 | 0.807 ± 0.008 | 0.669 ± 0.054 | 244.0 | −0.001 |
| 0.50 | 0.803 ± 0.011 | 0.681 ± 0.033 | 81.1 | −0.005 |
| 1.00 | 0.794 ± 0.014 | 0.646 ± 0.061 | 30.1 | −0.014 |
| 2.00 | 0.773 ± 0.017 | 0.608 ± 0.097 | **12.3** | −0.035 |

> ### ⚠️ The finding that matters for MS4
>
> The tempting read is "DP is nearly free at σ ≤ 0.25". **The ε column says otherwise.** Meaningful
> guarantees are single-digit; getting there means far more noise, and by σ = 2.0 the worst practice
> has already fallen to 0.608.
>
> The cause is cohort size: 3,276 patients across 10 practices is very little to hide in. Levers:
> the funded pilot's 25 practices, fewer rounds (each composition spends budget), subsampling
> amplification, and accepting a larger ε with SecAgg+ carrying more of the load.

**Local DP costs far more, and it is the mechanism that answers the legal question.** Central DP
noises *during aggregation*, so the server still receives every un-noised update; only local DP
(`LocalDpMod`, inside the SuperNode) changes that. At a comparable composed ε ≈ 31 it reaches AUROC
0.754 / worst 0.574 against central DP's 0.794 / 0.646 — and at ε ≈ 4.3, the first defensible
budget, the worst practice falls to **0.397**, below the 0.495 it gets by not collaborating at all.
Full table: [PRIVACY.md §3](PRIVACY.md).

**On the earlier ε figures.** Previous versions of this table used a hand-rolled basic-composition
bound. Where both are meaningful the accountant is 2–4× tighter, and at low σ the old expression was
**not a valid bound at all** (it holds only for per-round ε ≤ 1, which every row violated). Earlier
figures were also irreproducible — see [PRIVACY.md §7](PRIVACY.md). **Do not quote them.**

### Secure Aggregation

Probed against the installed flwr 1.33.0:

| Probe | Result |
|---|---|
| `secaggplus_mod` in `flwr.clientapp.mod` (Message API) | ❌ absent |
| `secaggplus_mod` in `flwr.client.mod` (legacy) | ✅ present |
| `SecAggPlusWorkflow.__call__` requires `LegacyContext` | ✅ yes |
| DP mods in `flwr.clientapp.mod` | ✅ all three present |

SecAgg+ has not been ported to the Message API in 1.33 and cannot compose with `strategy.start()`.
It **is** reachable by driving `DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))` with a
`LegacyContext` from inside a modern `ServerApp` — and **that is now implemented**:

```bash
uv run flwr run . --run-config "secure-aggregation=true"
```

Verified end-to-end: 12 practices, fit and evaluate aggregating with 0 failures. Construction and
the three limits that must travel with any claim about it — semi-honest threat model, visible
participation, and an aggregate that is still model parameters — in
[PRIVACY.md §4](PRIVACY.md).

**SecAgg+ cannot protect FedXgbBagging at all** — trees are not a vector to mask. Any
SecAgg-protected deployment is a logistic-regression deployment, which the accuracy results make an
easy trade.

### Leakage audit

Membership inference against the released model (`uv run ckd-audit`): loss-threshold and
shadow-model attacks find no usable signal at any privacy setting, **including with no DP at all**
(attack AUROC 0.501 / 0.502 against a chance baseline of 0.500), while a positive control on a
deliberately overfit model reaches 0.659 — so the attack fires when leakage is present. The
explanation is model capacity, not privacy engineering: eleven parameters over 3,276 patients.
This is one of the three attack families EDPB Opinion 28/2024 names, on synthetic data. See
[PRIVACY.md §5](PRIVACY.md).

---

## 4. Recommendation

**Deploy logistic regression, not XGBoost.** Equal pooled ceiling (0.861 against 0.860), far better
federated accuracy (0.808 against 0.729), 59× less bandwidth, it degrades rather than improves with
more rounds, and it is the only one of the two secure aggregation can protect. It is also the model
named in grant task T2.2.

**Use FedProx or FedAvg as the default.** They are indistinguishable from each other and are the
most accurate option on both global and worst-practice AUROC, at the smallest payload. Keep FedProx
as the nominal choice — it costs nothing over FedAvg and its drift correction may start earning its
keep once local epochs increase in the real deployment.

**Reach for FedMosaic when disclosure is the binding constraint, not accuracy.** It gives up ~0.008
global AUROC and ~0.014 worst-practice, and costs 10× the bandwidth — in exchange for never putting
a model on the wire at all. That is a real option to have if the legal review (Jorzig & Partner)
takes a hard line on parameter sharing, but it is not the accuracy choice.

**Cut the round count to ~10.** The logistic protocols are converged there, and every additional
round spends differential-privacy budget for no accuracy.

**Treat the DP budget as an open risk**, not a solved requirement.

## 5. Limitations

| Limitation | Consequence |
|---|---|
| Synthetic data throughout | Absolute AUROC is not a clinical claim. The *relative* protocol ordering is the transferable result |
| No eGFR | The strongest CKD predictor is absent from the synthetic schema; real data should lift every number |
| Synthetic label is prevalence, real label is incidence | `ckd_stage3plus` vs `ckd_incident` are different prediction tasks (CLAUDE.md §3b) |
| 10 practices, not 25 | Understates what the funded pilot can achieve, especially for the DP budget |
| Fairness audited by age band, not sex | The Antrag specifies sex; `geschlecht` exists only in the real schema |
| No membership-inference testbed | DP's benefit is argued from ε, not demonstrated against an attack |
| FedMosaic's own DP mechanisms unimplemented | Its privacy claim currently rests on payload shape alone |
| DP measured on FedAvg only | FedProx and FedMosaic DP costs unmeasured |
| Worst-practice AUROC has high seed variance | ± 0.06–0.14 across seeds. Single-seed protocol rankings are unreliable — an earlier single-seed run put FedMosaic first, which five seeds reverse |
