# Era registry — FLIP-IT CKD (FL-CKD-Flower)

> The project's experiment-lane registry, in the era discipline (probe → registered design with
> kill criteria → measured verdict; thresholds never move after results exist).
> Written 2026-08-27 as the first such registry for this repository: the era numbering is shared
> with the sister project that introduced it (15 registered lanes exist there, E1–E13 + B-lines),
> so this registry reviews the full inherited list, prunes it to what belongs to THIS repository,
> and continues the numbering. Nothing in the sister project's own records is deleted — its
> charter treats negative results as findings and its archive as read-only; the pruning below
> scopes the *FL-CKD-Flower planning view only*.
>
> **Scope note (2026-08-28):** the codebase was reduced to logistic regression only — the MLP and
> XGBoost/`FedXgbBagging` model paths were evaluated and **removed**. Future lanes below that
> touch the protocol benchmark (e.g. the C5→Era-16 FedMosaic null-twin suite) exercise the logreg
> protocols (FedAvg / FedProx / FedMosaic) only; closed era records above are left untouched as
> history.

## 1. Review of the inherited era list

Deletion rule (owner instruction 2026-08-27): delete an era iff it is **both**
(a) **non-executed** — no registration, gate, probe, or run artifact exists — **and**
(b) **does not relate to the FL-CKD-Flower architecture** (Flower strategies, the
DP-SGD + SecAgg + orchestrator privacy standard, the protocol benchmark, the dual-level
metrics, the clinic-cohort data layer).

| Era | Subject | Executed? | FL-CKD architecture? | Decision |
|---|---|---|---|---|
| E1 | mechanism-aware absence toolbox | ✅ falsified | no | keep (record) |
| E2 | federated imputation anchoring | ✅ void | no | keep (record) |
| E3 | per-client data curation | ✅ falsified at probe | no | keep (record) |
| E4 | server-side plan reasoner | ✅ falsified at probe | no | keep (record) |
| E5–E8 | compositional-narration arc | ✅ falsified/feasibility | no | keep (record) |
| E9 | episode-priced narration | ✅ **win** (+3.328) | no | keep (record) |
| E10a | mechanism transfer, family 2 | ✅ **win** (+3.068) | no | keep (record) |
| E10b | priced evidence routing | ✅ routing null, witness stands | no | keep (record) |
| E11 | bounded-acquisition tool agents | ✅ closed at Stage-0 G3 | no | keep (record) |
| E12 | the price of federation | ✅ measured (+0.810) | no | keep (record) |
| E13 | stateless evidence cards | ✅ registered, gates consumed dev spend | no | keep (live, sister project) |
| B3/B5 | text-arm drift remediation | ✅ won | no (LLM sentinel) | keep (record) |
| B4 | RAASi dip adjudication | ✅ falsified at probe | no | keep (record) |
| — | B1 interactive case review | ❌ | no | **DELETE** |
| — | real-data transfer (eICU/MIMIC) | ❌ | no (this repo's real-data path is the FHIR/Tomedo contract, a separate roadmap) | **DELETE** |
| — | D3 PRISMA desk pass | ❌ | no | **DELETE** |
| — | C1 priced-privacy routing in agent worlds | ❌ | no (experiment bed is the sister repo) | **DELETE** |
| — | C2 canary audit of card payloads | ❌ | no (instruments the sister repo's E13) | **DELETE** |
| — | C3 DP equity tax under per-clinic ε standardisation | ❌ | **yes** — `orchestrator.py`, `dpsgd.py`, dual-level metrics | **PROMOTE → Era 14** (§3) |
| — | C4 rule-table vs learned planner falsification | ❌ | **yes** — the orchestrator's enumerable space | keep as future lane (§2) |
| — | C5 FedMosaic null-twin suite | ❌ | **yes** — `models/protocols/fedmosaic.py` benchmark | keep as future lane (§2) |

**Updated registry:** E1–E13 + B3/B4/B5 stand as closed records (untouched); the five ❌/no
lanes are removed from this project's planning view. Highest occupied era number = **13**, and
numbers are never recycled (E13's artifacts exist in the sister project), so the next lane here
is **Era 14**.

## 2. New additions proposed for this architecture (journal-aimed)

Brainstorm constrained to THIS repo's architecture, each with the venue that pays for it:

1. **Era 14 — Equity-priced privacy orchestration** *(selected, §3).* The deployment standard
   standardises ε\* per clinic; measured consequence (`docs/PRIVACY.md` §3.5): σ spans
   2.57–3.47 at ε\*=8 and 8.32–11.68 at ε\*=2, **largest on the smallest census** — uniform
   privacy buys non-uniform noise. Proposed addition: shape per-clinic ε by census under a
   frozen clinic-mean budget and measure worst-cohort utility vs the standard. Journal fit:
   *JAIR* / *Artificial Intelligence* (method: budget allocation as a first-class server-side
   orchestration problem); *npj Digital Medicine* / *Lancet Digital Health* (the deployment
   reading: the rural practice's model is the one DP hurts most).
2. **Era 15 candidate — completing the EDPB audit triangle.** `audit.py` covers membership
   inference only; EDPB Opinion 28/2024 also names model inversion and reconstruction
   (PRIVACY.md live issue #2). Addition: inversion + reconstruction attacks against the
   released global model under the DP-SGD + SecAgg standard, same five-seed discipline, with
   audit results as committed artifacts. Fit: *IEEE TIFS* / *npj Digital Medicine*.
3. **Era 16 candidate — FedMosaic null twins.** The protocol benchmark's dynamic weighting
   `α = exp(−(ℓ_pseudo − ℓ_priv)/ℓ_priv)` and expertise aggregation have never faced a
   deranged-expertise or permuted-consensus control. Addition: a null-twin suite over
   `models/protocols/fedmosaic.py`. Fit: *JAIR* (short), benchmark venues.
4. **Era 17 candidate — planner ceiling falsification.** `AGENT_RULES` is four rules + one
   accountant inversion over an enumerable space; test whether ANY learned policy beats the
   rule table on (batch, σ) planning at matched budget — a registered falsification, cheap,
   CPU-only. Fit: methodology note, *TMLR*/*JAIR*.

## 3. Era 14 — registration (frozen 2026-08-27, before any run)

**Name:** *Equity-Priced Privacy Orchestration in Cross-Silo Federated Learning.*

**Question.** At a frozen clinic-mean privacy budget, does census-shaped per-clinic ε
allocation beat the uniform-ε\* standard on the **pre-specified vulnerable-cohort** AUROC —
the three smallest practices — without paying it back on global AUROC?

**Single dial (nothing else moves).** `γ`, the allocation shape in
`orchestrator.equity_targets`: ε_k ∝ N_k^(−γ), renormalised so the K-clinic mean equals the
budget ε\*. γ = 0 is **exactly** the deployed standard (`orchestrator.plan`); γ > 0 gives
larger budgets (less noise) to small practices; **γ < 0 is the anti-equity twin** — the
control that must NOT buy vulnerable-cohort utility. All else inherited from
`notebooks/03_dpsgd_secagg_standard.ipynb`: seeds 42–46, R = 10 rounds, E = 2 local epochs,
lr = 0.5 (`privacy.LEARNING_RATE`), δ = 1e-5, clip = `dp.CLIPPING_NORM`, the same
`prepare_practices` splits, the same `run_standard` FedAvg+SecAgg-semantics runner, the same
`weighted_and_worst` dual-level logging. One σ = 0 clipped reference row.

**Grid.** γ ∈ {−1.0, −0.5, 0.0, +0.5, +1.0} × ε\* ∈ {0.5, 2.0, 8.0}; letters evaluated at the
frozen primary pair (ε\* = 2.0, γ ∈ {0, +0.5}); the rest of the surface is descriptive.

**Privacy semantics disclosed in every table.** Uniform ε\* is *privacy-equitable*;
census-shaped allocation is *utility-targeted*: per-clinic ε_k **spread widens** (min/max ε_k
quoted per arm). The budget identity `mean_k ε_k = ε*` is certified by construction (raise,
not assert), and every plan row carries `achieved_epsilon ≤ target_k` from
`orchestrator.plan`'s existing certification.

**Estimator.** Paired per-seed difference, 95% t-CI (t(4) = 2.776), tested in both
directions; primary endpoint = mean AUROC over the pre-specified vulnerable trio (the three
smallest train censuses — the min-over-clients estimator is a named order-statistic trap and
is **never** the primary; worst-client is still logged every round per the metrics contract).
Secondary: global weighted AUROC; sensitivity reported throughout.

**Kill criteria / letters (frozen; thresholds never move after results exist):**

- **K0 replication guard:** the γ = 0 row must sit within ±2σ of the published standard row
  (`PRIVACY.md §3.5`: 0.797 ± 0.011 / worst 0.649 ± 0.060 at ε\*=2). Failure ⇒ instrument
  drift: the run is VOID, findings reported as none.
- **K1 primary:** vulnerable-trio Δ(γ=+0.5 − γ=0) at ε\*=2 — 95% CI containing 0 ⇒ the
  allocation claim is NULL at this census; reported plainly; no dial re-tuning.
- **K2 anti-equity twin:** Δ(γ=−0.5 − γ=0) 95% CI entirely **above** 0 ⇒ misallocation
  outperforms the standard on the vulnerable cohort: the equity mechanism reads backwards —
  claim dead with mechanism reported.
- **K3 price letter:** global-AUROC Δ(γ=+0.5 − γ=0) CI entirely below −0.01 ⇒ the equity
  purchase costs more global utility than the letter allows: reported as equity-at-a-price,
  the win claim is narrowed accordingly.
- **K4 budget integrity:** any plan row violating `achieved ≤ target` or the mean-budget
  identity ⇒ VOID by construction (enforced by raises in `orchestrator.plan_equity`).

**Reproduce:** `uv run ckd-equity` → `results/equity.json`; `uv run ckd-equity --quick` for
the 3-arm single-seed smoke.


## 4. Verdict (measured 2026-08-27 — `results/equity.json`, full registered grid, 5 seeds)

Pre-specified vulnerable trio: clinics [6, 0, 9], train N = [128, 129, 132].
5-seed means; per-clinic ε_k spread disclosed per arm (the allocation's price is a wider
guarantee spread, quoted openly):

| arm | trio | worst | global | sens | ε_k spread |
|---|---|---|---|---|---|
| σ=0 ref | 0.751 | 0.659 | 0.796 | 0.491 | — |
| γ=−1.0 (twin) ε*=0.5 | 0.755 | 0.651 | 0.794 | 0.484 | [0.24, 0.81] |
| γ=0 (standard) ε*=0.5 | 0.753 | 0.660 | 0.798 | 0.512 | [0.50, 0.50] |
| γ=+1.0 ε*=0.5 | 0.755 | 0.658 | 0.798 | 0.535 | [0.25, 0.83] |
| γ=−1.0 (twin) ε*=2.0 | 0.744 | 0.649 | 0.797 | 0.474 | [0.98, 3.25] |
| γ=0 (standard) ε*=2.0 | 0.746 | 0.649 | 0.797 | 0.477 | [2.00, 2.00] |
| γ=+0.5 ε*=2.0 (**primary pair**) | 0.744 | 0.649 | 0.797 | 0.476 | [1.45, 2.65] |
| γ=−1.0 (twin) ε*=8.0 | 0.750 | 0.656 | 0.798 | 0.472 | [3.91, 13.00] |
| γ=0 (standard) ε*=8.0 | 0.749 | 0.656 | 0.798 | 0.471 | [8.00, 8.00] |
| γ=+1.0 ε*=8.0 | 0.749 | 0.656 | 0.798 | 0.472 | [4.00, 13.27] |

(Full 16-row surface incl. γ=±0.5 in `results/equity.json`; γ=±0.5 interpolate γ=0 and γ=±1
throughout — no hidden corner.)

**Letter outcomes, against §3's frozen criteria:**

- **K0 PASS.** The γ=0 allocator is bit-identical to `orchestrator.plan` (certified before
  compute), and the γ=0 rows recompute the published standard rows at the quoted precision
  (0.798/0.660 at ε*=0.5; 0.797/0.649 at ε*=2; 0.798/0.656 at ε*=8; σ=0 ref 0.796/0.659 —
  `PRIVACY.md` §3.5's exact figures).
- **K1 NULL — the claim is dead at this census.** Vulnerable-trio Δ(γ=+0.5 − γ=0) at ε*=2
  = **−0.0011, 95% CI [−0.0040, +0.0018]**: contains 0 with high precision. Reported plainly
  per rule; no dial re-tuning, no seed additions.
- **K2 quiet.** Anti-equity twin Δ(γ=−0.5 − γ=0) = +0.0004 [−0.0018, +0.0026] — misallocation
  does not beat the standard either; the estimator sees the dial, not noise.
- **K3 quiet.** Global-AUROC Δ(γ=+0.5 − γ=0) = +0.0001 [−0.0009, +0.0012] — the allocation
  is global-utility-free but also gain-free.
- **K4 PASS by construction** (mean-budget identity and achieved ≤ target raises never fired;
  every row's audit trail in the JSON).

**Reading.** The executed plans genuinely moved (per-clinic σ spans 5.4–22.3 across the
surface, vs 8.3–11.7 at the standard), but trio AUROC stays flat to ±0.004: at this census
(N 128–425) the DP-SGD noise regime is saturated — σ changes of ±30% are below the
detectable utility granularity of ~30-130-row test AUROC. **The equity lever is not
allocation shape; it is census scale.** This strengthens the standing sentence in
`PRIVACY.md` §3.3 (*more practices is the lever*): uniform-ε\* standardisation is NOT
established here as the thing that taxes small practices — at pilot-relevant heterogeneity
(hundreds vs tens of thousands of patients per practice, ε_k spread then 1.10 vs 16.04
under uniform *settings*; `uniform_settings_audit`) the experiment should be re-registered
with a powered estimator before that sentence is extended.

**Re-open condition (registered):** new world/census only — a practice-size spread of ≥ 2
orders of magnitude with the same 5-seed paired estimator; never a threshold edit here.

## 5. Era 15 — registration (frozen 2026-08-27, before any run)

**Name:** *The Census-Spread Phase Map for Privacy Orchestration — where does allocation
shape begin to matter?*

**Why this experiment is publishable in every outcome cell (design intent, inherited from
the sister project's doctrine):** it is a *characterization*, not a contest. Era 14 measured
the flat point (census spread 3.3×, NULL with ±0.004 precision); `uniform_settings_audit`
measures accounting incoherence at 100× hypothetical spread. The transition between those
anchors is unmeasured. Every registered letter below is **two-sided**: slope-positive,
slope-null-through-100×, and slope-negative are each pre-committed sentences, and the phase
map itself is the artifact. A null at 100× spread is the stronger deployment finding — it
says the pilot can standardise ε\* without an equity tax; an onset says allocation shape is
load-bearing and prices where it starts to pay. Both are quotable; neither requires a win.

**Question.** As the census band's max/min ratio R stretches through {5, 20, 100} at a frozen
geometric mean (~2000 patients/clinic), does census-shaped allocation (γ=+1) begin to pay the
pre-specified vulnerable trio vs the uniform-ε\* standard (γ=0) — and what does it cost
globally per level?

**Frozen ladder and cohorts** (`data/clinics_ladder/r{5,20,100}/`, generated by
`data.synthesize.generate_clinics` — the repo's only generator; archetype cycle and the
shared true risk model `_BETA` are untouched; seed rule one-global→derived-per-clinic holds):

| level | band (min_n, max_n) | base_seed | clinics |
|---|---|---|---|
| R5 | (894, 4472) | 11000 | 10 |
| R20 | (447, 8944) | 12000 | 10 |
| R100 | (200, 20000) | 13000 | 10 |
> **Erratum 5-a (pre-evidence dev-launch, disclosed per the era's culture; toward the
> registered design, no letter or floor moved).** The smoke launch surfaced a generator
> property: size draws are uniform-in-band, and `E[max/min]` over 10 uniform draws is
> `(10R+1)/(R+10)`, which **never exceeds 10** — the ladder as drawn realized
> {4.4×, 4.7×, 7.8×}, i.e. no ladder. The registered object is the SPREAD; the bands above
> therefore fix the per-clinic size SETS exactly, log-spaced with geometric mean ≈ 2000:
> `sizes_k = round(exp(ln(min) + k/9·(ln(max)−ln(min))))` for k = 0..9, passed through
> `generate_clinics`'s new optional `sizes` override (additive; the default draw is
> bit-unchanged). Realized max/min is then exactly {5, 20, 100}. Base seeds and archetype
> cycle stand; cohorts regenerate once and deterministically.

Realized censuses are recorded in `results/phasemap.json`; the vulnerable trio is the three
smallest TRAIN censuses per level, specified from the generated census before any training.

**Arms per level** (everything else identical to the Era-14 registered constants: seeds
42–46, R = 10 rounds, E = 2 epochs, lr = 0.5, δ = 1e-5, `run_standard` FedAvg + SecAgg
semantics, `weighted_and_worst` dual-level logging):

- `naive` — un-orchestrated uniform settings (batch 64, σ 1.5, the canonical audit row):
  the no-orchestrator control whose failure mode is **accounting incoherence** — its per-clinic
  achieved ε varies by N (reported as the L3 descriptor), not a utility letter.
- γ ∈ {0 (the deployed standard), +1 (equity direction), −1 (anti-equity twin)} at
  ε\* ∈ {2.0, 8.0} — `orchestrator.plan_equity` / `plan`, the Era-14 machinery unchanged.
- one σ = 0 clipped reference row per level (the level's own no-DP twin — controls for the
  fact that small-cohort AUROC estimation noise grows with spread even without DP).

**Estimators.** Paired per-seed contrasts (95% t-CI, both directions). The onset slope: OLS
over per-(seed, level) differences of trio AUROC (γ=+1 − γ=0) against log10(R) at ε\*=2,
15 points; the independence caveat (levels share seeds but not cohorts) is stated beside the
fit, per-level paired CIs reported alongside. Worst-client is logged every round per the
metrics contract and never used as an estimator (the order-statistic trap).

**Letters (frozen; thresholds never move after results exist):**

- **G1 allocator identity (structural guard):** γ=0 plans bit-identical to
  `orchestrator.plan` at every level (raise otherwise; the run is VOID on failure).
- **G2 coherence audit (structural guard):** the naive arm's per-clinic achieved ε must equal
  `orchestrator.uniform_settings_audit`'s arithmetic on the same census to machine precision
  (raise otherwise; VOID).
- **L1 onset slope (primary, three pre-quoted sentences):** at ε\*=2 — CI entirely > 0 ⇒
  "an onset exists; allocation pays increasingly with heterogeneity" (the crossing level is
  estimated from the band); CI contains 0 through R=100 ⇒ "flat through 100× spread: the
  deployed standard carries no measurable allocation equity tax at pilot-realistic scale"
  (the strong-null deployment sentence); CI entirely < 0 ⇒ reverse-onset, "allocation to the
  small actively harms them at scale" — mechanism investigation is the next era, registered
  separately.
- **L2 price per level (secondary, two-sided):** global-AUROC Δ(γ=+1 − γ=0) per level —
  any level's CI entirely < −0.01 ⇒ equity at scale is priced, quoted per level; all quiet ⇒
  "shape reallocation is global-utility-free at every measured scale".
- **L3 disclosure-coherence descriptor (not a letter):** achieved per-clinic ε [min, max] per
  arm per level — the naive arm's widening incoherence IS its characterization; orchestrated
  arms quote their registered spreads.

**Reproduce:** `uv run ckd-phasemap` → `results/phasemap.json`;
`uv run ckd-phasemap --quick` = R5 level, seed 42, {naive, γ0@2, γ+1@2} + σ=0 reference.
> **Erratum 5-b (post-evidence code defect, disclosed; no letter, floor, seed, arm, or world
> moved).** The first full launch completed every registered cell and lost the run at the
> letters evaluator — a label defect (`g+0` vs the registered `g0` lookup), discovered from
> the traceback after 54 min of otherwise-valid compute. The run writes results only after
> letters, so the evidence was re-derived by deterministic relaunch: all cells are seeded
> per (seed, budget) and reproduce bit-for-bit. The repair touches the label only.
Target venues given the outcome cells: JAIR / Artificial Intelligence (the map + onset law);
npj Digital Medicine / Lancet Digital Health (the flat-through-100× or onset sentence as a
deployment rule for the 25-practice pilot).



## 6. Era 15 — verdict (measured 2026-08-27 — `results/phasemap.json`, 3 levels × 8 arms × 5 seeds)

Realized censuses (train): R5 [715–3577], R20 [357–7155], R100 [160–16000]; ratios 5.0 / 20.0 / 100.0
exact by the erratum-5-a size sets. 5-seed means, trio = three smallest train censuses per level:

| level | arm | trio | worst | global | sens | ε_k spread |
|---|---|---|---|---|---|---|
| R5 | standard γ=0 @2 | 0.826 | 0.772 | 0.816 | 0.505 | [2.00, 2.00] |
| R5 | γ=+1 @2 | 0.826 | 0.772 | 0.816 | 0.504 | [0.79, 3.93] |
| R5 | naive | 0.827 | 0.770 | 0.816 | 0.503 | [2.03, 5.28] |
| R5 | σ=0 ref | 0.816 | 0.761 | 0.806 | 0.513 | — |
| R20 | standard γ=0 @2 | 0.819 | 0.769 | 0.816 | 0.477 | [2.00, 2.00] |
| R20 | γ=+1 @2 | 0.819 | 0.768 | 0.816 | 0.479 | [0.29, 5.88] |
| R20 | naive | 0.820 | 0.769 | 0.816 | 0.486 | [1.36, 8.10] |
| R100 | standard γ=0 @2 | 0.781 | 0.675 | 0.818 | 0.495 | [2.00, 2.00] |
| R100 | γ=+1 @2 | **0.775** | 0.684 | 0.813 | 0.471 | **[0.08, 8.06]** |
| R100 | γ=−1 @2 (twin) | 0.782 | 0.678 | 0.818 | 0.495 | [0.08, 8.06] |
| R100 | naive | **0.785** | 0.685 | 0.818 | 0.497 | **[0.87, 12.10]** |
| R100 | σ=0 ref | 0.772 | 0.669 | 0.809 | 0.509 | — |

(ε*=8 rows mirror ε*=2 throughout; full 24-cell surface in `results/phasemap.json`.)

**Letter outcomes, against §5's frozen criteria:**

- **G1/G2 PASS** (allocator identity at every level; naive accounting matches
  `uniform_settings_audit` to machine precision; both raises never fired).
- **L1 — FLAT: no onset through 100× spread.** Slope = −0.0037 per decade, 95% CI
  [−0.009, +0.0015] contains 0 (per-level paired: R5 −0.0003 [−0.0009, +0.0003]; R20 −0.0004
  [−0.0013, +0.0005]; R100 −0.0051 [−0.0159, +0.0058]). The registered strong-null sentence
  stands: **allocation shape carries no measurable equity tax from 5× to 100× census
  heterogeneity — the deployed uniform-ε\* standard may keep its one-line policy** at
  pilot-realistic and far-beyond-pilot spread.
- **L2 — FREE at the registered 0.01 materiality floor**, every level and budget. Disclosed
  in full: two sub-floor CIs exclude 0 (global Δ −0.0004 [−0.0007, −0.0001] at R20@2 and
  −0.0055 [−0.0083, −0.0027] at R100@2): a real but five-to-twenty-times-sub-material cost —
  the map reports it; the letter does not fire.
- **L3 descriptor (the motivation's own evidence):** at R100 the naive un-orchestrated row
  delivers ε ∈ [0.87, 12.10] for one shared instruction — a 14× accounting incoherence — yet
  buys **nothing** utility-wise over the certified standard (trio 0.785 vs 0.781, worst
  0.685 vs 0.675, both within seed noise). Coherence is not a utility sacrifice; it is free.

**The two-era sentence (Era 14 + Era 15, both on this architecture):** in cross-silo DP-SGD
under Rényi-composed per-clinic budgets, *per-clinic ε allocation is a solved non-problem* —
uniform-target standardisation cost nothing (L2) and nothing is recoverable by shaping (Era
14 NULL at 3.3×; Era 15 FLAT through 100×). What heterogeneity actually taxes is **scale
itself**: the vulnerable trio falls 0.826 → 0.781 from R5 to R100 on the *standard* arm and
0.816 → 0.772 even at σ=0 — small-cohort estimation noise, untouched by any privacy lever
measured here. The publication-grade claims, all artifact-backed: (i) an orchestration
standard whose per-clinic plans certify achieved ε ≤ target holds utility-flat across two
orders of census magnitude; (ii) the privacy-vs-equity conflict hypothesised from
centralised-DP disparate-impact work does NOT materialise as an *allocation* problem in the
cross-silo client-level regime; (iii) the naive uniform-settings shortcut's only measured
casualty is accountability (14× ε incoherence), utility untouched — which is precisely why
the orchestrator's certificate, not its accuracy, is the contribution.

**Next lane (per §2's ordering, unchanged):** Era 16 — FedMosaic null twins; Era 17 —
planner ceiling. The phase map is closed; thresholds stood unmodified end-to-end.
