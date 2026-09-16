# FLIP-IT retreat talk — slide-by-slide content + speaker notes

- **Talk:** *FLIP-IT — federated learning for kidney-disease risk, without moving a single patient record.*
- **Length:** 10 minutes · **Audience:** FL/ML researchers (retreat) · **Language:** English
- **Frame:** *Federated learning has three problems:* **(1) data modality chooses the model**,
  **(2) the model and the privacy goal choose the protocol**, **(3) the privacy protocol must
  answer to the law**. Each problem gets a measured answer from the FLIP-IT baseline sandbox.
- **Number policy:** every figure below is fixed in `data/NUMBERS.md` (live re-run on
  2026-09-16 or clearly-marked docs measurements). Use them verbatim; never paraphrase numbers.
- Timings sum to ≈ 10:00. Slides B1–B3 are backup (not in the 10 minutes).

---

## Slide 1 — Title (0:30)

**Title:** FLIP-IT — federated learning for kidney-disease risk, without moving a single patient record

**Subtitle:** Pre-kickoff baseline: synthetic GP practices, real Flower.ai federation, measured privacy.
NEXT.IN.NRW / EFRE-JTF NRW · docport · IKIM (Universitätsmedizin Essen) · RUB Trustworthy ML · Jorzig & Partner

**Visual:** no image; text-forward title, dark blue field.

> **Speaker notes (~75 words):** Good morning. FLIP-IT is a funded NRW project: a federated
> learning infrastructure across 25 GP practices to predict chronic kidney disease from routine
> care data. The premise in one sentence — the model travels to the data; the patient record
> never leaves the practice. This talk is ten minutes and one argument: federated learning has
> three problems, and for each one we built the sandbox, ran the measurement, and got an answer.
> Synthetic data only — this is the pre-kickoff baseline, not a clinical claim.

---

## Slide 2 — The three problems (1:00)

**Title:** One promise, three problems

**Body (three boxes, numbered):**

1. **Data modality → model.** Ten features of diagnosis flags and durations from heterogeneous
   practice systems. What can be modelled — and federated — on *that*?
2. **Protocol ↔ model + privacy.** Strategies, wire payloads, rounds. What runs between the
   practices, and what does it cost?
3. **Privacy protocol ↔ law.** DP and secure aggregation are maths; GDPR and EDPB Opinion
   28/2024 are the acceptance test. What does the aggregating party legally see?

**Footer:** 10 synthetic practices · 3,276 patients · one shared ground-truth risk model · Flower 1.33

> **Speaker notes (~150 words):** Frame first. "Data stays home" is the marketing sentence; the
> engineering content is three problems that constrain each other. Problem one is data modality:
> German GP routine data, FHIR-harmonised, is essentially diagnosis flags, age, and time since
> diagnosis — sparse, binary, class-imbalanced, and wildly non-IID across practices. That shape
> picks the model for you. Problem two: the model and what you're willing to disclose pick the
> federation protocol — coefficients, or in one protocol nothing but predictions on public data.
> Problem three is the one this room usually underestimates: the privacy protocol is not done when
> the accountant returns a number — it has to answer a legal question, which in Europe is EDPB
> Opinion 28/2024. Three problems, and I'll spend roughly equal time on measured answers to each.
> Everything you'll see is reproducible from the repo.

---

## Slide 3 — Problem 1: the data chooses the model (1:15)

**Title:** Problem 1 — data modality: what GP data actually looks like

**Figure:** `assets/fig01_cohort.png` (full-width below the title)

**Body (right or below, compact):**

- 10 practices, panels **161–532 patients**; CKD prevalence **8.3–46 %** — non-IID in covariates,
  labels, *and* quantity; five clinical archetypes share one true logistic risk model.
- Feature space: age + six binary diagnosis flags + three durations. Class imbalance up to ~17:1.
- The canonical real-data contract (`extract_features.sql`) adds **eGFR** (strongest CKD
  predictor) and **sex** — it is an *incidence* task; the synthetic cohort is a *prevalence* task.

> **Speaker notes (~190 words):** Problem one. The data modality is not "an EHR" in the deep
> learning sense — it's ten columns: age, six ICD-coded diagnosis flags, three durations. And the
> federation is not cosmetic. Panel sizes span 161 to 532, CKD prevalence from 8 to 46 percent;
> an urban-young practice and a rural-elderly practice are different distributions sharing one
> underlying risk equation. Two consequences. First, the model must be *federatable by Flower* —
> that's a project invariant, the model travels as a parameter vector. Second, on ten tabular
> features with class imbalance this extreme, logistic regression is not the compliance-friendly
> fallback, it is the right estimator — and the grant's task T2.2 names it. We did test the obvious
> challenger: XGBoost pooled matched logreg's ceiling, 0.860 against 0.861, but federated bagging
> collapsed to 0.729 at sixty-one times the bandwidth, got *worse* with more rounds, and its trees —
> serialized split thresholds, literal patient values — could never be secure-aggregated. Both tree
> and MLP paths are deleted now; the repo is logreg-only by measurement. So problem one answers the
> model question before a single strategy runs.

---

## Slide 4 — Problem 1 evidence: what federation buys (1:25)

**Title:** Federation recovers the ceiling — and, more importantly, fixes the worst practice

**Figure:** `assets/fig02_protocols.png` (dominant)

**Body (bullets beside/below):**

- FedProx/FedAvg: **0.808 ± 0.008** global vs **0.861** pooled ceiling — a −0.053 price of
  federation (5 seeds, 20 rounds).
- The gain is **equity, not average**: worst practice 0.495 → **0.675 ± 0.054** (+0.180).
- FedProx ≡ FedAvg to three decimals — the μ term has nothing to correct at 2 local epochs.
- FedMosaic: 0.800, −0.008 — it pays accuracy for its disclosure profile (see Problem 2).
- Negative control (signal-less data): nothing exceeds ≈ 0.53 → the learning is real.

> **Speaker notes (~210 words):** The measurement. Ten practices, five seeds, twenty rounds; only
> the protocol varies — same splits, seeds, preprocessing. Blue bars are the global sample-weighted
> AUROC, red bars the worst single practice. Read it as two findings. Finding one: federation
> works. FedProx lands at 0.808 against a pooled ceiling of 0.861 — ninety-four percent of the
> ceiling, zero patient rows moved. Finding two is the one that matters for the project: the
> benefit is not the mean, it's the floor. A practice training alone on its own panel gets 0.495
> on its worst member — that's a coin flip. Federation lifts that floor by eighteen points to
> 0.675, while the global mean moves less than two. For a pilot whose entire premise is that small
> and skewed practices can't learn a CKD model alone, worst-practice AUROC is *the* number — and
> it doubles as our T2.5 bias measure, so we log both levels every round. Two honest nulls:
> FedProx equals FedAvg to three decimals, so the proximal term earns nothing at two local epochs;
> and on a signal-less control CSV nothing beats about 0.53, which is what lets us read 0.808 as
> signal and not wiring.

---

## Slide 5 — Problem 2: the protocol IS the disclosure surface (1:15)

**Title:** Problem 2 — what actually crosses the wire

**Body (two-column wire table):**

| | FedAvg / FedProx | FedMosaic |
|---|---|---|
| Payload | model coefficients | predictions + expertise on a **public** cohort |
| Bits/client/round | **352** | **3,600** |
| Server can invert? | yes — the vector is the attack object | no — no model leaves the practice |

**Figure:** `assets/fig03_convergence.png` (right/lower half)

**Body (one line under figure):** Converged by round ~10 → cutting 50→10 rounds costs no accuracy
and cuts composed DP budget ~2.7× (ε 22.0 → 8.1 at σ=2.0).

> **Speaker notes (~190 words):** Problem two connects the model to privacy through the protocol.
> Same task, two qualitatively different wires. FedAvg and FedProx ship the coefficients — three
> hundred fifty-two bits per practice per round — and that vector is exactly the object gradient
> inversion attacks chew on. FedMosaic ships binary predictions and an expertise vector computed on
> a *public* reference cohort: there is no model of private patients for the server to steal. That
> costs ten times the bandwidth and about eight-thousandths of global AUROC — a real option if the
> consortium's legal review ever hardens against parameter sharing. Second point, and it feeds
> problem three directly: rounds are privacy currency. The logistic protocols are converged by
> round ten — you can see FedProx flatten and training-alone drift *down* from overfitting — but
> every extra round composes more differential-privacy loss. Halving the schedule cuts composed
> epsilon about 2.7-fold for free. And one infrastructure note: secure aggregation in Flower 1.33
> lives only in the legacy API; we drove the legacy workflow from inside a modern ServerApp and
> verified it end-to-end — twelve practices, zero failures. So all three protocol levers work here.

---

## Slide 6 — Problem 3: privacy is a legal acceptance test (1:00)

**Title:** Problem 3 — the privacy protocol must answer to the law

**Figure:** `assets/Flipit-privacy.png` (dominant; the deployment-standard diagram)

**Body (short, beside figure or as 3 lines):**

- Defence in depth L0–L6: identifier drop at extraction → TLS & node identity → SecAgg+ → DP →
  metric hygiene → leakage audit. Each layer independently verifiable.
- **The legal question:** what does the *aggregating party* see? Central DP doesn't answer it —
  the server receives un-noised updates first. Only in-clinic noise + secure aggregation do.
- EDPB Opinion 28/2024: the released *model* must pass membership-inference, inversion, and
  reconstruction tests — configured DP alone is not anonymity.

> **Speaker notes (~150 words):** Problem three, and here's the mis-set expectation to fix:
> privacy is not a flag on the training job. The diagram is our deployment standard: record-level
> DP-SGD computed *inside* each practice, secure aggregation on the wire, and a rule-based server
> agent that sets per-practice noise so every practice lands the same epsilon. Around it, layers
> zero through six — no patient identifier ever enters a feature frame, TLS and public-key node
> identity, log hygiene, and a leakage audit as the final check. The legal framing matters: GDPR
> asks what the aggregating party sees, and EDPB Opinion 28/2024 says a trained model is anonymous
> only if it survives attack families — membership inference, inversion, reconstruction. So the
> acceptance test for our privacy protocol is not an epsilon; it's an evidence pack. Which means
> every layer needs a number. Next slide: the numbers that surprised us.

---

## Slide 7 — Problem 3 evidence: DP is cheap only until ε means something (1:15)

**Title:** Record-level DP inside the clinic survives ε = 0.5–8; update-level local DP does not

**Figure:** `assets/fig05_central_vs_local.png` (dominant)

**Body (3 tight bullets):**

- Central DP: accuracy holds to σ=2.0 — but ε is still **12.3**; the tempting "DP is nearly free"
  reading dies in the ε column.
- Update-level local DP at composed ε ≈ 4.3: worst practice **0.397 < 0.495** — worse than not
  collaborating at all.
- Deployment standard (record-level DP-SGD + SecAgg + ε-orchestrator): **0.797–0.798 global,
  0.649–0.660 worst at ε* 0.5–8** — at *half* the budget where local DP collapsed.

> **Speaker notes (~190 words):** The cost curves, x-axis is composed epsilon on log scale —
> left is more private. Blue is central DP: accuracy looks almost free, and that's the trap —
> at σ two-point-oh you've lost three points globally but the RDP accountant still says ε 12.3,
> and defensible budgets are single-digit. Note the honest-accounting detail: earlier versions of
> this table used a basic-composition bound that was outside its own validity regime — we fixed
> that, it's in the docs. Red is update-level local DP, the only mechanism in that pair that
> changes what the server sees: at ε about 4.3 the worst practice falls to 0.397 — below the
> 0.495 it could score by refusing to federate entirely. That killed local-DP-as-configured. The
> green stars are the fix: move the noise *inside* the record, not onto the update — patient-level
> DP-SGD with Poisson sampling, secure aggregation so the server only opens the masked sum, and a
> rule-based agent inverting the accountant per clinic so a 532-patient practice and a 161-patient
> practice spend the *same* epsilon by paying different noise. At ε two it holds 0.797 — inside
> seed noise of its own no-DP baseline. That is the MS4 architecture.

---

## Slide 8 — The audit: clean — for the right reason, with the right caveat (0:55)

**Title:** Membership inference finds nothing — because there's nowhere to hide it

**Figure:** `assets/fig06_audit.png` (dominant)

**Body (3 bullets):**

- Loss-threshold and shadow-model attacks: **0.49–0.51 at every setting, including no DP**
  (pass bar 0.520); positive control fires at **0.655/0.659** — the testbed works.
- Why: **11 parameters over 3,276 patients** — capacity, not privacy engineering, does the work.
- Caveat: 1 of the 3 EDPB attack families; synthetic data, no eGFR. **Not an anonymity proof** —
  inversion and reconstruction audits are open (T2.5).

> **Speaker notes (~140 words):** Last measurement, the leakage audit — this is the evidence-pack
> piece. Two canonical membership-inference attacks against the released global model, members
> versus same-distribution held-out non-members, five seeds. Result: attack AUROC sits on the
> chance line at every privacy setting — including with no DP at all. And before you ask: the
> testbed is not broken; the positive control, a deliberately overfit model on forty rows, gets
> 0.66. The protection here is model capacity — eleven parameters on three thousand patients,
> there is nowhere for a person to be memorised. That's a genuinely useful property of choosing
> logistic regression, and it means DP is not what's earning this result — which matters when
> deciding how much utility to spend on it. What it is NOT is an anonymity proof: EDPB names three
> attack families, we've built one, on synthetic data. Re-run on real data before any claim.

---

## Slide 9 — What the pilot changes (0:55)

**Title:** From 10 synthetic practices to 25 real ones

**Figure:** `assets/Flipit-process.png` (left; portrait pipeline) + optionally the two webapp
demo screenshots small on the right (`assets/01_patient_form.png`, `assets/02_risk_score_result.png`)

**Body (bullets on the right):**

- **25 practices = more patients = cheaper privacy**: subsampling amplification + census are the
  levers that move the DP curves; orchestration experiments say the ε-standard scales fine.
- Real data adds **eGFR** and **sex** → full T2.5 fairness analysis (sex-based, per the Antrag).
- Each practice hosts its **Helios FHIR server**; feature building is SQL-on-FHIR, one canonical
  contract (`extract_features.sql`).
- Milestone MS4 (month 18): validated CKD model incl. DP + SecAgg. Physician demo webapp exists.

> **Speaker notes (~140 words):** So, what does the funded pilot change? Scale, and scale is the
> privacy lever. Twenty-five practices with real panels instead of ten synthetic ones — more
> patients to hide in, subsampling amplification kicks in, and every privacy curve I showed shifts
> in our favour. We stress-tested the orchestration specifically: shaping per-clinic epsilon by
> census buys nothing measurable — flat through a hundred-fold census spread — so the simple,
> auditable uniform-epsilon standard is what ships, and bigger censuses are what help. Real data
> also adds eGFR, the single strongest CKD predictor, and sex — which unlocks the fairness
> analysis the Antrag actually specifies, by sex, for T2.5. Every practice runs its own Helios
> FHIR server; feature extraction is SQL-on-FHIR against one canonical contract, so onboarding
> practice twenty-five is a config change, not a schema negotiation. Milestone MS4 at month
> eighteen is exactly this stack: validated model, DP, secure aggregation.

---

## Slide 10 — Summary (0:40)

**Title:** Three problems, three measured answers

**Body (three numbered lines, big):**

1. **Modality → model.** Sparse, non-IID tabular flags ⇒ federatable logistic regression.
   Federation: **0.808 vs 0.861 ceiling; worst practice +0.180**.
2. **Model + privacy → protocol.** Coefficients (352 bits) or predictions-only (3,600 bits);
   converged by round 10; SecAgg+ verified end-to-end.
3. **Protocol → law.** Central ε isn't the legal answer; update-level local DP collapses
   (**0.397 at ε≈4.3**); the record-level standard holds **≈0.798 at ε\* 0.5–8**; audit clean on
   1 of 3 EDPB families.

**Closer line:** *The model travels. The data stays. The guarantee is measured, not asserted.*

> **Speaker notes (~100 words):** To close. Problem one, the data's shape picked the model, and
> federation recovers nearly all of the pooled ceiling — while lifting the worst practice by
> eighteen points. Problem two, the protocol is the disclosure surface — coefficients or
> predictions-on-public-data, and we cut the privacy budget by converging early. Problem three,
> privacy has to survive a lawyer: the standard that survives is record-level DP-SGD inside the
> clinic plus secure aggregation, holding point-seven-nine-eight at single-digit epsilon, with an
> audit that finds no membership signal and two attack families still on the roadmap. Everything
> you've seen reproduces from the repo with three commands. Thanks — questions.

---

## Backup B1 — Central DP detail (not timed)

**Title:** Backup — the central DP sweep, row by row

**Figure:** `assets/fig04_dp_central.png`

**Body:** table or bullets from `data/NUMBERS.md` central sweep; per-practice ε correction note
(largest practice's effective ε ≈ 1.9–2.5× the uniform quote pre-correction; the live
`central-dp-epsilon` path scales σ by K·ρ_max).

## Backup B2 — Orchestration experiments (not timed)

**Title:** Backup — is uniform ε* fair? (registered eras 14–15)

**Body:**

- Identical DP-SGD settings are incoherent across census sizes: batch 64/σ 1.5 → ε **1.10**
  (50 k patients) vs **16.04** (500). The server agent inverts the accountant per clinic instead
  (σ spans 2.57–3.47 at ε* = 8; 8.32–11.68 at ε* = 2 — largest σ on the smallest census).
- Census-shaped (equity) allocation: **null** at this cohort (Δ −0.0011, 95 % CI
  [−0.0040, +0.0018]) and **flat through 100× census spread** → uniform ε* standard ships.
- Reproduce: `ckd-equity` → `results/equity.json`, `ckd-phasemap` → `results/phasemap.json`.

## Backup B3 — Reproduce & references (not timed)

**Title:** Backup — reproduce everything

**Body (monospace block):**

```
uv run ckd-clinics --clinics 10
uv run ckd-benchmark --rounds 20            # per-seed; seeds 42-46 for the 5-seed table
uv run ckd-privacy --rounds 20 --seeds 42 43 44 45 46
uv run ckd-audit   --rounds 20 --seeds 42 43 44 45 46
```

**References:** FedMosaic: Algorithm 1 of `docs/2507.00259v3.pdf` · EDPB Opinion 28/2024 ·
Flower 1.33 (`flwr.serverapp.strategy`) · docs/REPORT.md, docs/PRIVACY.md, docs/ERAS.md,
docs/DEPLOYMENT.md · Project docs: `docs/01 Projektantrag Innovationswettbewerb NEXT.IN.NRW.pdf`.
