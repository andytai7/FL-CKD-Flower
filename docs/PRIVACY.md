# Privacy architecture

Evidence for **Milestone 4** (month 18), whose acceptance criterion in the Projektantrag is
*"Implementierung von Secure Aggregation und Differential Privacy"*, and for **T2.3** (Privacy
Enhancing Features) and **T2.5** (Datenschutztests und Bias-Analyse).

Reproduce everything here with:

```bash
uv run ckd-privacy --rounds 20 --seeds 42 43 44 45 46    # -> results/privacy.json
uv run ckd-audit   --rounds 20 --seeds 42 43 44 45 46    # -> results/audit.json
```

Both are **reproducible**: rerunning either reproduces the tables below exactly. That was not true
before — see §7.

---

## 1. Defence in depth

Seven layers, each independently verifiable. "Status" is what is true in this repo **today**, not
what is planned.

| Layer | Mechanism | Protects against | Status |
|---|---|---|---|
| **L0** Minimisation | Drop direct identifiers at extraction; no patient id ever enters a feature frame | Direct re-identification | ✅ **Closed — `patientid` no longer exported (§2)** |
| **L1** Transport | TLS on SuperLink↔SuperNode and SuperLink↔`flwr` CLI | Network interception | ✅ Documented in [DEPLOYMENT.md](DEPLOYMENT.md) |
| **L2** Identity | SuperNode public-key authentication; only registered practice keys admitted | A rogue node joining the federation | ✅ Documented in [DEPLOYMENT.md](DEPLOYMENT.md) |
| **L3** Confidential aggregation | SecAgg+ — the server sees only the sum, never one practice's update | An honest-but-curious SuperLink operator | ✅ **Implemented and verified end-to-end (§4)** |
| **L4** Formal guarantee | **Deployment standard: patient-level DP-SGD** (record-level Poisson sampling + per-sample clipping + Gaussian noise, inside the clinic) composed with L3 SecAgg+ on the wire; ε standardised across heterogeneous clinic sizes by the rule-based server agent (`orchestrator.py`). Central DP (server-side) and local DP (in-SuperNode) remain as the measured comparison baselines the standard was chosen against (§3.1–§3.3) | Reconstruction / membership inference from the released model | ✅ **Standard measured live (§3.5): `notebooks/03_dpsgd_secagg_standard.ipynb` + the in-process runner `dpsgd.py`** |
| **L5** Metric hygiene (server-side log) | Suppress small cohorts; anonymised lines are shuffled per round so line position carries no identity | Re-identification through the dual-level logs | ✅ **Implemented and ON by default** (log layer; the wire itself is still attributed — see §8) |
| **L6** Audit | Fairness gaps by group; membership-inference testbed | Undetected bias / leakage (T2.5) | 🟡 **MIA built (§5)**; inversion + reconstruction still open |

---

## 2. L0 — the identifier gap, closed

[`extract_features.sql`](../extract_features.sql) §9 previously selected `ep.patientid` into the
per-practice export. Nothing downstream used it — `data/loader.py` selects only `FEATURE_COLS` — but
the **CSV on the practice's disk carried it**, which made that file personal data under DSGVO rather
than a pseudonymous extract, and dragged it into the scope of every access-control and retention
obligation.

**The column is now dropped.** If T4.2 later needs a stable join key across extractions, the
replacement is a per-practice salted hash, decided deliberately — not the raw identifier restored.

One residual channel is documented in the SQL rather than silently left: the export is still
`ORDER BY ep.patientid`, so while the column is gone, **row order still follows it**. If patient ids
are assigned chronologically, row position is a weak signal for relative enrolment order. Downstream
code must not treat row position as information; `ORDER BY random()` removes the channel at the cost
of reproducible extracts.

The FHIR path enforces L0 structurally: `data/fhir_loader.py` reads only the contract fields
(`birthDate`, `gender`, condition codes/dates, lab values/dates, encounter dates), keeps the FHIR
patient `id` in memory solely to join resources, and emits rows **ordered by SHA-256(id)** — a
deterministic order that carries no enrolment information (the channel the SQL export still has).
An optional per-practice `pseudonym-salt` node-config adds a `patient_pseudonym` column
(HMAC-SHA256 over the id): exactly the salted-hash replacement named above, off by default, and
never usable as a cross-practice link because each practice salts its own.

---

## 3. L4 — what Differential Privacy actually costs

**Read §3.3 first if you read nothing else: central and local DP protect different things.**

### 3.1 Central DP — noise applied by the server during aggregation

Flower's real `DifferentialPrivacyServerSideFixedClipping` wrapped around `FedAvg`, clipping norm
1.0, 10 non-IID practices, 20 rounds, **5 seeds**. Mean ± s.d. of the final round:

| Noise σ | AUROC | Worst practice | ε (RDP) | naive bound | AUROC cost |
|---:|---|---|---:|---:|---:|
| 0.00 | **0.808 ± 0.008** | 0.675 ± 0.054 | — | — | — |
| 0.10 | 0.808 ± 0.007 | 0.676 ± 0.056 | 1211.8 | 969 | −0.000 |
| 0.25 | 0.807 ± 0.008 | 0.669 ± 0.054 | 244.0 | 388 | −0.001 |
| 0.50 | 0.803 ± 0.011 | 0.681 ± 0.033 | 81.1 | 194 | −0.004 |
| 1.00 | 0.794 ± 0.014 | 0.646 ± 0.061 | 30.1 | 97 | −0.014 |
| 2.00 | 0.773 ± 0.017 | 0.608 ± 0.097 | **12.3** | 48 | −0.035 |
> **Per-practice reading of the ε columns.** Flower spreads the noise σ·C evenly over the sampled
> clients while FedAvg weights each update by cohort size, so practice *k* effectively receives
> σ/(K·ρ_k) with ρ_k its update share (K·ρ_max = 1.63 on this census). The ε columns quote the
> uniform-weight figure; the largest practice's effective ε reads top-to-bottom
> 3017 / 562 / 173 / 59.7 / **22.8** (≈1.9–2.5× above the uniform quote at low σ). These sweeps
> predate the correction and are kept as the *utility* evidence — utility is unchanged; only the
> per-practice ε reading moves. The LIVE `central-dp-epsilon` path IS corrected: server_app
> inverts σ at the largest census share (σ scaled by K·ρ_max, requires fraction-fit = 1.0), so
> the configured bound holds for every practice, largest included.

### 3.2 Local DP — noise applied inside the SuperNode, before transmission

Flower's real `LocalDpMod`, same clipping norm, same seeds and rounds. Local DP is parameterised by
(ε, δ) **per round**; the composed budget is reported by the same accountant so the two tables read
on one scale.

| ε per round | ε composed | AUROC | Worst practice | AUROC cost |
|---:|---:|---|---|---:|
| off | — | **0.808 ± 0.008** | 0.675 ± 0.054 | — |
| 50 | 1283.4 | 0.806 ± 0.008 | 0.677 ± 0.044 | −0.002 |
| 20 | 257.6 | 0.791 ± 0.017 | 0.650 ± 0.052 | −0.017 |
| 10 | 85.0 | 0.768 ± 0.041 | 0.597 ± 0.061 | −0.040 |
| 5 | 31.4 | 0.754 ± 0.009 | 0.574 ± 0.053 | −0.054 |
| 1 | **4.3** | 0.627 ± 0.057 | **0.397 ± 0.130** | −0.181 |

### 3.3 The comparison that matters

> **Central DP does not answer the question the legal assessment asks.** Its noise is applied by
> the server *as it aggregates*, so the server necessarily receives every practice's un-noised
> update first. It constrains what can be inferred from the *published model*; it does not
> constrain what the aggregating party sees. Only **local DP** changes that.
>
> ⚠️ Note also that flwr's `DifferentialPrivacyClientSideFixedClipping` does **not** close this gap
> despite its name — its own docstring describes it as *"central DP with client-side clipping"*.
> Only the clipping moves to the client; the noise is still added by the server.

**And that protection is expensive.** At comparable composed budgets:

| composed ε | mechanism | AUROC | Worst practice |
|---:|---|---|---|
| ≈30 | central DP (σ=1.0) | 0.794 | 0.646 |
| ≈31 | local DP (ε=5/round) | 0.754 | 0.574 |

Roughly **0.04 AUROC and 0.07 worst-practice** is the measured price of the server never holding an
individual update — at matched ε, on this cohort. The reason is structural: under local DP each of
the ten practices adds independent noise, so the noise entering the aggregate grows with the number
of practices instead of being added once.

At ε ≈ 4.3 — the first genuinely meaningful budget in either table — the worst practice falls to
**0.397**, well below the 0.495 it achieves by not collaborating at all. At this cohort size, local
DP at a defensible ε is not a usable configuration. More practices is the lever that changes this.

### 3.4 About the ε numbers

The reported ε comes from an **RDP accountant** (Google's `dp_accounting`), at δ = 1e-5. The
previous figures used the plain Gaussian mechanism composed by basic composition; they are retained
in `results/privacy.json` as `epsilon_basic_upper_bound` **for comparison only**.

Two things changed, and only one of them is "tighter accounting":

- Where both are meaningful (larger σ), RDP is **2–4× tighter** at identical noise — σ=2.0 over 20
  rounds is ε = 12.3, not 48.
- At small σ the old expression was **not a valid bound at all**. The classical Gaussian analysis
  ε₁ = √(2 ln(1.25/δ))/σ holds only for ε₁ ≤ 1, and every row violated that badly (at σ=0.1 the
  per-round ε₁ alone is ~48). The old figures were not conservative; they were outside the regime
  where the formula says anything. This is why σ=0.1 now reports a *larger* ε than before.

The mechanism is unchanged throughout — same clipping, same noise. Only the accounting is honest now.

**The cheapest remaining improvement is fewer rounds.** Every round composes additional loss, and
the logistic protocols are converged by round ~10 ([REPORT.md](REPORT.md)). At σ=2.0, halving 20
rounds to 10 takes ε from 12.3 to **8.1** at no measured accuracy cost. `num-server-rounds` is now
10 by default for exactly this reason. Subsampling amplification (`fraction-fit` < 1.0) is credited
by the accountant and is the next lever.

The live run is configured by the budget, not the noise: `central-dp-epsilon` in the run config
makes the ServerApp derive σ by inverting the accountant (`dp.sigma_for_epsilon`) for the run's
own round count, so the number a deployment documents stays fixed while the schedule changes.
The accounting itself — `epsilon_rdp`, the retained naive bound, and the inversion — lives in
`dp.py`, the single definition shared by this sweep, the audit, and the server.

### 3.5 The deployment standard: record-level DP-SGD + SecAgg + rule-based epsilon orchestration

§3.1–§3.3 are the baselines the standard was chosen against. The deployment route is
**patient-level DP-SGD inside each clinic** — record-level Poisson sampling, per-sample gradient
clipping, calibrated Gaussian noise — composed with L3 SecAgg+ masking on the wire, with ε
**standardised across heterogeneous clinic sizes by the rule-based server agent**
(`orchestrator.py`) — deterministic if-then logic plus one accountant inversion, no LLM anywhere
in the loop. Measured live in `notebooks/03_dpsgd_secagg_standard.ipynb` through the in-process
runner `dpsgd.py`.

**The agent is a rule table, not a model.** Every decision is deterministic if-then logic,
reproducible from the code, auditable line by line, and unchanged run to run:

1. **Inventory.** At the start of the run, count the connected SuperNodes and read each
   clinic's reported census N. N is the only datum exchanged.
2. **Epsilon target.** One fixed global policy for the whole federation — the run-config
   `dpsgd-epsilon` (ε*) plus δ = 1e-5. Same target for every clinic.
3. **Parameter calculation.** For each clinic, invert the shared RDP accountant
   (`dp.sigma_for_epsilon`): the smallest σ_k with ε(σ_k; q = b/N_k, T = ⌈E·N_k/b⌉ per round,
   R rounds) ≤ ε*, picking b from the candidate grid to minimise per-epoch injected noise.
   This subsumes the hand-tuned sketch "if N ≥ 1000 then σ=1.1, b=64; if N < 1000 then raise σ
   or b" — a fixed σ per size band does **not** equalise ε (`uniform_settings_audit` below
   measures uniform settings at ε 1.10 vs 16.04 for 50k vs 500 patients), so the rule is an
   inversion, not a lookup.
4. **Dispatch.** `ClinicPlan.to_config()` is bundled into the ConfigRecord stamped per
   outgoing instruction; clinics are generic executors. Changing ε* is one server config
   edit — zero phone calls.

**Mechanism.** Each clinic trains on its own records with sampling rate q = b/N and
T = ⌈epochs·N/b⌉ steps per round; R rounds × T steps compose through the shared RDP accountant
(`dp.epsilon_rdp`, δ = 1e-5). The server collects exactly one number per clinic — the census N —
and inverts the accountant (`dp.sigma_for_epsilon`) per clinic to pick a (batch b, σ) pair such
that every clinic composes to the same target ε*. Among the batch candidates that fit inside N
it keeps the one minimising expected per-epoch gradient-noise variance. The plan is stamped onto
the clinic's outgoing train Message as five standard-name keys (`ClinicPlan.to_config()`) and the
runner executes it verbatim, so the accounted T is the executed T. `DpsgdOrchestrator(FedAvg)`
wires this into the live strategy.

**Why uniform settings fail.** The same instruction delivers different privacy at different
census sizes. `orchestrator.uniform_settings_audit` with batch=64, σ=1.5, E=10, R=10 gives a
50,000-patient clinic ε = 1.10 and a 500-patient clinic ε = 16.04 — a ~15× spread from identical
settings. On this wiring cohort under the deployment schedule (E=2, R=10) the same instruction
already spans ε = 7.25 … 15.26: the large clinics are over-protected (utility spent needlessly)
and the small ones under-protected (far more budget burned than anyone signed for).

**The rule-based fix.** Every plan row is certified `achieved_ε ≤ ε*` **before it is issued**
— the check is a `raise` in `orchestrator.plan`, not a convention (and not a bare assert — survives `python -O`). The census heterogeneity is
absorbed by σ: at ε* = 8 the per-clinic σ spans 2.57 … 3.47 across the ten practices; at ε* = 2
it spans 8.32 … 11.68 — in both cases the smallest census carries the largest σ.

**Measured utility** — the full route through `dpsgd.py`, 10 practices, R = 10 rounds, δ = 1e-5,
**5 seeds**, mean ± s.d. of the final round:

| Target ε* (every clinic, at or below) | AUROC | Worst practice |
|---:|---|---|
| none (σ = 0, same runner) | 0.796 ± 0.010 | 0.659 ± 0.050 |
| 0.5 | 0.798 ± 0.008 | 0.660 ± 0.051 |
| 2 | 0.797 ± 0.011 | 0.649 ± 0.060 |
| 8 | 0.798 ± 0.011 | 0.656 ± 0.050 |

Contrast with the §3.2 update-level local-DP collapse: at **half** of the composed ε ≈ 4.3 budget
where local DP fell to 0.627 ± 0.057 / worst 0.397 ± 0.130 — below the 0.495 a practice achieves
by not collaborating at all — the standard holds 0.797 / 0.649 at ε* = 2, and the server
additionally sees only the masked aggregate of record-noised updates.

⚠️ The no-DP **same-runner** reference is 0.796, not the 0.8077 full-batch figure published in
§3.1: DP-SGD applies per-sample clipping even at σ = 0, so the ~0.01 gap is the clipping bias of
this training path, not data drift (notebook 02 jointly verified identical data handling across
paths). Utility comparisons against the standard must use the same-runner row.

---

## 4. L3 — SecAgg+ is implemented

Probed against the installed `flwr` 1.33.0:

| Probe | Result |
|---|---|
| `secaggplus_mod` in `flwr.clientapp.mod` (Message API) | ❌ absent |
| `secaggplus_mod` in `flwr.client.mod` (legacy) | ✅ present |
| `SecAggPlusWorkflow` importable | ✅ present |
| `SecAggPlusWorkflow.__call__` requires `LegacyContext` | ✅ yes |
| DP mods in `flwr.clientapp.mod` | ✅ `fixedclipping_mod`, `adaptiveclipping_mod`, `LocalDpMod` |

SecAgg+ has not been ported to the Message API in 1.33, so it cannot compose with
`strategy.start()`. It **is** reachable by driving the legacy workflow from inside a modern
`ServerApp`, and **that is now built** — `server_app._run_secure_aggregation`:

```bash
uv run flwr run . --run-config "secure-aggregation=true"
```

Verified end-to-end: 12 practices, fit **and** evaluate aggregating with 0 failures.

Two implementation notes worth keeping:

- **One ClientApp serves both paths.** The legacy workflow speaks the legacy record shape
  (`fitins.parameters` / `evaluateins.parameters`) rather than the Message API's `arrays`, so
  `client_app.py`'s handlers accept either and translate through Flower's own compat bridge.
- **The switch is run config, not an environment variable.** Mods are attached at *construction*
  time, before any run config exists, so SecAgg+ and local DP are each installed as a small
  dispatching mod that reads `ctx.run_config` when called. This also survives the process boundary —
  the simulation engine runs ClientApps in Ray workers that do not inherit the launching shell's
  environment.

### What SecAgg+ does not give you

1. **The threat model is semi-honest.** The guarantee holds against a server that follows the
   protocol but inspects what it receives. It is *not* a guarantee against a server that deviates —
   for example by running a round with a single practice, whose "aggregate" is that practice. The
   control is `min_fit_clients`, set to the full practice count so such a round fails rather than
   succeeds.
2. **Participation stays visible.** The server always learns which practices took part, and the
   aggregate. Only the individual contribution is hidden.
3. **The aggregate is still model parameters.** SecAgg+ answers *who may see one practice's update*.
   It does not by itself make the released model anonymous — that is what L4 and L6 are for.

**SecAgg+ cannot protect the XGBoost path at all.** `FedXgbBagging` transmits serialized decision
trees whose split thresholds are derived from patient values; there is no vector to mask. Any
SecAgg-protected deployment is a logistic-regression (or MLP) deployment. The benchmark's finding
that trees federate poorly here ([REPORT.md](REPORT.md)) makes that an easy trade.

---

## 5. L6 — the membership-inference audit

EDPB Opinion 28/2024 does not accept that a model is anonymous because DP is configured; it asks for
a case-by-case assessment and names **membership inference**, **model inversion** and
**reconstruction** as the relevant tests. The first is now built (`uv run ckd-audit`): a
loss-threshold attack (Yeom et al. 2018) and a shadow-model attack (Shokri et al. 2017, 16 shadow
models), both against the released global model, scored as attack AUROC where 0.5 is a coin flip.

Members are every practice's training rows; non-members are their held-out rows — the
attacker-favourable framing, since both come from the same practices and the same distribution, so
any separation is memorisation rather than population shift.

| Mechanism | Threshold attack AUC | Shadow attack AUC | Verdict |
|---|---|---|---|
| **positive control** (n=40, deliberately overfit) | **0.655** | **0.659** | attack fires |
| none (no DP at all) | 0.501 ± 0.006 | 0.502 ± 0.015 | pass |
| central DP σ=0.1 … 2.0 | 0.502 – 0.504 | 0.502 – 0.509 | pass |
| local DP ε=1 … 50 /round | 0.492 – 0.503 | 0.494 – 0.506 | pass |

Pass criterion, one-sided and on effect size: the 95% CI upper bound of attack AUC stays within
0.5 + 0.02. (Significance alone is the wrong test — with small seed variance it flags AUC = 0.495,
an attack performing *worse* than chance, as a failure.)

> ### How to read this
>
> **Neither attack finds a usable membership signal — including with no DP at all.** The positive
> control shows this is not a broken testbed: the same attacks reach 0.66 against a model trained to
> memorise.
>
> The explanation is capacity, not privacy engineering. The released model is logistic regression
> with **eleven parameters** fitted over 3,276 patients; there is almost nowhere for an individual
> to be memorised. That is a genuine and useful argument about this model class — and it means DP is
> **not** what is providing the protection here, which matters when choosing how much utility to
> spend on it.
>
> ⚠️ It is **not** an anonymity proof. It is one of three attack families, on synthetic data, with
> ten features and no eGFR. Real practice data is richer and the result may not survive it. The
> audit must be re-run on real data before any anonymity claim is made.

---

## 6. What each protocol puts on the wire

| Protocol | Payload | Disclosure surface | Bits/client/round |
|---|---|---|---:|
| `fedprox` / `fedavg` | model coefficients | A parameter vector fit to this practice's patients — the object gradient-inversion attacks target | 352 |
| `fedxgb` | serialized decision trees | **Split thresholds are literal patient feature values.** The highest-disclosure payload here | ~20,900 |
| `fedmosaic` | binary predictions + expertise on a *public* cohort | Opinions about patients who are already public. No parameter vector exists for the server to invert | 3,600 |

FedMosaic's disclosure profile is qualitatively different, and it is why the paper motivates the
approach on privacy grounds. It also carries its own per-round DP construction (XOR mechanism on the
label matrix, Gaussian on the expertise vector) which this repo does **not** yet implement — a clear
next step for T2.3.

---

## 7. Reproducibility — a bug that invalidated the earlier tables

The DP figures published before this revision could not be reproduced from their stated seeds, and
three sources in this repo disagreed about them.

**Cause.** Flower draws its DP noise from NumPy's *global* legacy RNG
(`flwr/supercore/differential_privacy.py:46`, `np.random.normal(...)`). `run_dp_sweep`'s `seed`
argument only reached the local train/test split, never that generator — so the DP noise was the one
unseeded step in an otherwise fully seeded pipeline. The signature is unmistakable: σ = 0.0 draws no
noise and was bit-identical everywhere, while divergence grew with σ.

**Fix.** `privacy._seed_dp_noise` seeds the global RNG per **(seed, σ) cell** before each run — per
cell rather than once per run, because the noise stream is consumed sequentially, so a single
per-run seed would make every row depend on which σ values precede it in `NOISE_MULTIPLIERS`. Each
of the five seeds still draws different noise, so the error bars keep their meaning.

This was a violation of CLAUDE.md §0 rule 6. **Any DP figure quoted from a document dated before
this revision should be re-checked against `results/privacy.json`.**

---

## 8. Honest gaps

| Gap | Consequence | Owner |
|---|---|---|
| Model inversion and reconstruction attacks not built | L6 covers 1 of the 3 attack families EDPB names | AG Kamp (T2.5) |
| Audit run on synthetic data only | A pass here is evidence about the method, not about the pilot model | pending real data |
| No penetration / disclosure testing, no k-anonymity analysis | The analytics path's disclosure controls are unassessed | docport + Jorzig & Partner |
| DP standard's numbers measured on the synthetic wiring cohort only | ~~Local DP unusable at a defensible ε~~ — **closed by the §3.5 standard** (record-level noise: no per-practice-per-round noise penalty; the standard holds 0.797 AUROC at ε* = 2 where local DP collapsed at ε ≈ 4.3). The residual risk: every ε/utility figure in §3.5 is wiring-cohort evidence — re-run on real data before any claim | pending real data |
| FedMosaic's own DP mechanisms unimplemented | Its privacy claim rests on payload shape alone | this repo (T2.3) |
| Fairness audited by age band, not sex | Antrag specifies sex; `geschlecht` exists only in the real schema | pending real data |
| DP measured on FedAvg only | FedProx/FedMosaic DP cost unmeasured | this repo |
| Row order in the SQL export still follows `patientid` | Weak ordering channel; documented, not removed | docport |
| Per-round per-practice metric payloads (partition-id, cohort size, AUROC) cross the wire attributed | L5 anonymises the server LOG only; the semi-honest SuperLink still sees named small-cohort metrics — inherent to dual-level aggregation (weighted AND worst need practice grouping). Mitigation is contractual, not cryptographic | federation operator + DPA |
