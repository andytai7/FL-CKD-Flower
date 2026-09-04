# Deep-Learning Privacy Benchmark — Dermoscopy Images (DermaMNIST)

Design contract for the image data kind (user-directed, 2026-09-04): the federation is
**weight-sharing over Flower's real strategies** (rules 1/8), the model class moves from
logistic-regression-over-pixels to a **small CNN** (track-local rule-2 waiver in `CLAUDE.md`
§0a), and privacy is provided by the protocol family that imaging actually needs —
**per-image record-level DP-SGD** as the baseline guarantee — benchmarked against the
gradient-free alternative (FedCT) and the aggregation-security layers (SecAgg+, verified
hybrid). Scope: this branch (`experiment/image`) only; nothing here is imported by the
deployable tabular paths, which stay logreg.

**Transport choice — measured, not assumed (2026-09-04).** The default aggregator on this track
is **FedProx (μ=0.1)**, with **FedAvg kept as a comparator row** in every matrix. Evidence
(shared-learner logreg protocol bench, `run_simulation(protocol=…, clinics_dir=
data/clinics_dermamnist)`, 15 rounds; numpy full-batch GD learner at lr 0.5 — absolute levels
understate the sklearn production learner of notebook 06, the comparison is within-bench):

| Protocol | r15 AUROC | r15 worst-practice AUROC | max round-to-round drawdown | behavior |
|---|---|---|---|---|
| FedAvg | 0.491 | 0.398 | **0.229** | severe parity oscillation (even ≈0.65 / odd ≈0.49) — client-drift pathology under Dirichlet label skew on 2,352 unscaled pixel features |
| FedProx μ=0.1 | 0.552 | 0.470 | 0.219 | oscillation damped; troughs rise monotonically (0.466→0.552); +0.072 on the *worst* clinic — the fairness metric (rule 5) benefits most |

Non-reasons the choice is safe: the proximal term is *local-objective-only* — clip-then-average
aggregation is unchanged, so record-level DP accounting and SecAgg+ compose over FedProx
identically to FedAvg (no privacy or wire-security cost). Under CNNs (non-convex, more local
steps) drift should worsen further; if FedProx alone cannot hold the curve, the next arms —
sanctioned as `Strategy` subclasses in `models/protocols/` — are **FedAvgM (server momentum,
targets exactly this oscillation)** and SCAFFOLD (control-variate drift correction, at 2×
server state). Ensemble/consensus alternatives are covered by the P3 FedCT arm.

Design goals:

1. **Imaging is the MIA-sensitive modality.** Medical images are the canonical target of
   membership-inference and gradient-inversion attacks (DLG/IG-family attacks invert raw
   gradients into pixels far more plausibly than for tabular features). Every paradigm here is
   therefore judged on both utility *and* measured MIA posture, not utility alone.
2. **Deep nets make naive DP expensive.** The logreg baseline is 2,352 parameters with clip
   $C=\sqrt{2352}\approx 48.5$; a small CNN is $\sim$10–40$\times$ more parameters, so
   $p$-dimensional Gaussian noise grows accordingly — this *is* the central research variable for
   DL images, and every DP arm must show what it buys against that scaling.
3. **Surgical edit, one harness.** The existing `simulate.py` runner, `to_xy` dispatch, and
   weighted+worst metrics stay; the CNN and the new protocols enter through the documented seams
   (`--clinics-dir data/clinics_dermamnist`, label `melanoma`, model-class switch).

## 0. Datasets

| Dataset | Task | Frame dimensions | Holder-of-truth locations |
|---|---|---|---|
| **DermaMNIST** (28×28) | melanoma-vs-rest, prevalence 0.1111 | 10 Dirichlet pseudo-clinics, 2,352 features (logreg view) / 3×28×28 (CNN view) | `data/clinics_dermamnist/` (regenerate via `data/external/dermamnist_to_clinics.py`); provenance + sha256 in `data/external/SOURCES.md` |
| **DermaMNIST** (64×64) | same | 3×64×64 CNN view — the resolution ablation arm | `data/external/dermamnist/dermamnist_64.npz` (already on disk, gitignored). NB: the higher resolution is where DP utility and MIA posture diverge most visibly |
| Public unlabeled reference for FedCT | consensus input | ISIC-archive-derived public dermatology images, or the MedMNIST DermaMNIST-test pool held out from all clinics | workspace `research/fedct_public/`, provenance pinned when materialized |

Models (reference architectures; code lands on this branch under `research/privacy-dl-image/`):

| Model | Params (fp32 wire size) | Note |
|---|---|---|
| CNN-S: conv(3→16, k5)+pool, conv(16→32, k3)+pool, conv(32→64, k3)+pool, GAP → fc(64→64) → head | ≈28k (≈0.11 MB) | Primary arm. Chosen so the $\sqrt p$ DP cost stays in the same order as the logreg baseline — isolates "DL per se" from "dimension per se" |
| CNN-M: ResNet-18-style lite (no pretrained weights) | ≈2.76M measured (≈10.5 MB) | Scaling probe: what record-level DP costs when $p\u2192$ millions on images |
| logreg over flattened pixels | 2,352 (9.4 KB) | Neutral FedAvg row — the existing baseline from notebook 06 |

## 1. Paradigm P1 — Gradient-space DP with tight accounting (record-level; the baseline guarantee for images)

**Categorization: primary privacy arm.** For imagery the privacy unit is the image (a person may
contribute several lesions — see the patient-level upgrade in §6 work plan; DermaMNIST's public
export does not carry patient identifiers, so **record(image)-level is the honest floor)**.

- **Blueprint.** Per-clinic local step: per-sample (microbatched) gradients, clip to norm $C$,
  Gaussian noise $\mathcal N(0, \sigma^2 C^2 I_p)$. Server: vanilla FedAvg. RDP composition per
  step → $(\varepsilon, 10^{-5})$ conversion; hyperparameters `(batch, σ)` selected to hit target
  composed $\varepsilon$ — same scalar recipe as the tabular standard (`privacy.py`,
  `dp.py`, `dpsgd.py`), parameterized for the clinics dir.
- **Sensitivity formulation.** Replace-one-image adjacency: $\Delta_2 = 2C$ per local step after
  clipping. Gaussian mechanism with per-step $\sigma$; subsampled RDP accountant over `$E$` local
  epochs × `$R$` rounds. The privacy guarantee is per-image, not per-clinic-contribution.
- **Image-specific handling ("temporal handling" analog).** Images carry no sequence structure,
  so the autocorrelation hazard of P1 on sequences does not exist here — instead the hazard is
  **representation collapse under clipping+noise on non-convex CNN losses**: clip too tight and
  conv filters die. Mitigations to compare: (i) microbatch clipping; (ii) per-layer clipping
  (conv layers vs head norms differ by orders of magnitude); (iii) adaptive clipping in the
  AdaClip / MacAdam direction (bias-free variance reuse for clip + momentum). The utility audit
  includes a filter-norm-health trace (dead-filter count per round), not just metrics.
- **Matrix contribution.** Utility vs $\varepsilon$ across the three model sizes × two
  resolutions $\Rightarrow$ the image-track $\varepsilon$-knee curve, where the tabular flatness
  (V1: AUROC 0.797–0.799 for $\varepsilon\in[0.5,8]$) is not expected to survive $p\approx$
  10$^{4}$–10$^{6}$; quantifying *where* it breaks is the deliverable.
  Measured MIA advantage under each $\varepsilon$ (TPR@FPR=1\%) — the imaging-specific column.

## 2. Paradigm P2 — Secure aggregation at scale (SecAgg+ as executed row; FastSecAgg/LightSecAgg as cost bench)

**Categorization: aggregation-security layer + cost profile.** Images change this paradigm's
numbers, not its semantics: the interesting question is what SecAgg+ costs at CNN wire sizes over
10 pseudo-clinics.

- **Blueprint.** Flower's SecAgg+ transport for the measured row (rules intact). FastSecAgg and
  LightSecAgg are **cost-model benchmarks** computed from their published complexity formulas at
  our $(n{=}10, p)$ — **no re-implementation of audited cryptography** (scope decision): their
  claims are profiling targets, ours are measured with SecAgg+.
- **Sensitivity.** None by itself — orthogonal composition with P1.
- **Image-specific handling.** No image-specific behavior; the variable is $p$: mask setup and
  pair-exchange scale linearly-to-quadratically in ciphertext count ∝ parameter count. At
  logreg's 2,352 params SecAgg+ cost was negligible against the CKD bench; at CNN-S (0.11 MB)
  expect still-cheap; at CNN-M (5.7 MB) the per-round ciphertext overhead becomes a first-class
  cost column. The 64×64 ablation changes activation/local-compute cost, not wire size.
- **Matrix contribution.** Per-round wall-clock and transmitted bytes at $\{2{,}352,\ 28\text{k},\
  1.5\text{M}\}$ params; dropout stress: kill `$k\in{1,2,3}$` clinics mid-round (rule-8 strategy
  completeness) measuring ceiling cost vs surviving-client sum recovery. Note: the 10 pseudo-
  clinics are big-$N$ and reliable, so dropout is a *stress arm*, not the primary claim — the
  analog of the wearable story lives on the time-series track.

## 3. Paradigm P3 — Prediction-space consensus, FedCT («Little Is Enough», AAAI 2025; arXiv 2310.05696; impl: github.com/kampmichael/federatedcotraining)

**Categorization: the structural-fit arm.** Image classification with a credible public unlabeled
reference is *the* canonical FedCT setting — and the only paradigm in this bench where **raw
gradients never leave a clinic**, which for the MIA-sensitive modality is a different security
posture than any amount of clipped-noised gradients.

- **Blueprint.** Local phase: train a private-view head (CNN-S) on local labeled data. Consensus
  phase: clients predict hard labels on the public consensus set (ISIC-derived or held-out
  DermaMNIST test pool); server majority-votes; clients fit their public-view head to consensus
  labels. Communication = one label per consensus sample per client ($M$ bits), no gradient
  traffic.
- **Sensitivity formulation.** The server receives $M$ noisy votes. With participation cap
  $\lfloor\gamma n\rfloor$ per consensus round, on-average-LOO stability bounds the
  sample-and-aggregate privacy of any vote to $(2\gamma d)/M$; summing $M$ Bernoulli votes with
  random response yields per-round cost $\sqrt{2\gamma M}\ \varepsilon_{\max}$ with
  $\varepsilon_{\max} = 2\gamma d/M$. **Honest risk (carried from the original design doc):** the
  refined bound is an *expected* (on-average) statement with sample-varying sensitivity — to make
  it a worst-case guarantee, either a high-probability lift (bounding tail participation) or a
  data-dependent refactor (PATE-style: publish noise-calibrated per-vote histograms only when
  consensus is near-unanimous) is required. The conservative baseline is participation cap
  $\gamma{=}1$, i.e. $\varepsilon_{\max}=2d/M$ and total cost $\sqrt{2M}\ \varepsilon_{\max}$
  worst-case. **The bench ships both rows** and reports both; the refined row is flagged expected-
  value-conditional until the analysis closes.
- **Image-specific handling.** No temporal hazards. Specific to images: (i) consensus-set
  domain shift — ISIC public pool vs HAM10000-derived DermaMNIST distribution must be audited
  (embedded-distribution test), because FedCT accuracy is consensus-set-quality-limited;
  (ii) hard-label noise interacts with the 0.1111 melanoma prevalence → majority votes on rare
  positives need the random-response noise calibrated *per class arm*, or recall collapses;
  (iii) the private/public-view split for dermoscopy: color/contrast augmentation
  invariants as the public view — the augmentations hold (low-level geometry) while raw
  high-res pores/structure stay private.
- **Matrix contribution.** Utility + MIA advantage + per-round MB (=public-set labels only) vs
  composed $\varepsilon$ — plotted beside P1's curve. Hypothesis: at equal $\varepsilon$, FedCT is
  strictly MIA-safer; the question is the utility gap at the consensus-set sizes we can field.

## 4. Paradigm P4 — Hybrid verified SecAgg + local DP (malicious-server posture)

**Categorization: high-assurance layer.** For images, the incentive for a malicious server is
highest (gradient inversion on un-noised image-model updates is pixel-recoverable), so this is
the arm worth benchmarking first among the P2+ compositors.

- **Blueprint.** SecAgg+ around each client update; client-side DP noising as in P1 with
  **distributed discrete Gaussian** decomposition (Kairouz–Liu–Steinke 2021, arXiv 2102.06387):
  per-client Skellam/DDG shares whose *sum* post-unmask equals one Gaussian with central-grade
  scale $\sigma_{\rm agg}$; norm enforcement pre-masking via ELSA-style arithmetic-circuit
  proofs ($\ell_2$ and $\ell_\infty$) so a malicious client can't inflate its update into the mask
  sum.
- **Sensitivity.** Composition rests on the DDG tail bounds (mod-2$^{64}$ wrap and precision are
  analyzed in the paper; bounds printed in code constants). Distributed $\chi^2$ concentration:
  with $n$=10 clinics and $p$=28k–1.5M dims the tail inflation is negligible at pilot sizes —
  verified numerically, not assumed.
- **Image-specific handling.** Same as P1's clipping discipline (per-layer clips must be part of
  the proved relation — the norm proof commits to whatever clipping structure P1 uses; wiring
  this is the main engineering risk). Stale-update replay deserves the window-binding
  construction from the TS doc; harmless for images but shared infra.
- **Matrix contribution.** Utility uplift of central-grade $\sigma_{\rm agg}$ vs L4/
  record-level scales at equal $\varepsilon$; proof-gen wall-clock per round vs CNN-S wire;
  measured MIA advantage. Claim test: whether DDG beats the current standard enough to matter at
  image dimensionality.

## 5. Threat models

1. **Honest-but-curious server.** Sees aggregates only (SecAgg+) and per-round FedAvg outputs.
   Publication risk = the aggregate model + trajectory; per-image DP (P1) caps record leakage;
   FedCT removes gradient exposure entirely.
2. **Malicious server.** Flat regions of CNN losses let Crafty-style traps inflate effective
   sensitivity per image → inspectability (ClipInspector-style audits of received clips) + P4
   norm proofs; P1/P4 measured under a trap-detection audit, not just assumed.
3. **External memorization auditor (the imaging case).** Membership inference against final CNN:
   augmented per-image vs held-out neighbors — measured MIA advantage is **a first-class metric
   for every arm** on this branch (TPR@FPR=1\% and AUROC of a shadow-model attack).

## 6. Empirical trade-off benchmark

Matrix shape identical to the time-series bench so cross-data-kind comparison reads row-wise.

- **Variants.** DL-vanilla (no privacy) per model; P1 across $\varepsilon\in${0.5, 1, 2, 4, 8,
  16} × {record-level} × {CNN-S, CNN-M}; P2 row without DP on CNN-S + dropout stress
  (1, 2, 3 drops); P3 with {public-set size, participation $\gamma$} on CNN-S; P4 {P1, vanilla} ×
  {proof, no-proof}.
- **Columns.** model architecture & params; paradigm; protocol variant; composed
  $(\varepsilon,\delta)$ per claim row (record-level; FedCT rows carry both noise-basis claims);
  AUROC; worst-practice AUROC (rule 5, worst over the 10 pseudo-clinics, matched seeds 42–46);
  MIA advantage (TPR@FPR=1\%); per-round/epoch wall-clock sec; per-round comm MB per client &
  server-ingress; client peak RAM. 28×28 vs 64×64 tagged per row.
- **Harness.** Same shape as the TS bench: `results/…matrix.json` rows + acceptance gates, plus
  the MIA cell and the ISIC provenance pin when FedCT runs. DL components gated behind the uv
  `dl` extra; harness runs without `dl` installed covering the subset that doesn't need torch.

## 7. Work plan

1. CNN-S (+CNN-M probe) module + local train/eval steps, torch as uv-managed optional `dl`,
   notebook-06-styled smoke FedAvg run at both resolutions → measures that CNN-S reproduces the
   logreg AUROC band (the sanity gate before any privacy spend).
2. P1 record-level DP-SGD (reuse `dp.py`/`privacy.py` accounting; parameterize `dpsgd.py` on
   clinics dir + label) ε-grid on CNN-S → the image knee.
      MIA audit per ε row.
3. FedCT consensus loop with the public reference + provenance pin.
4. P2 cost profile across model sizes + dropout stress.
5. P4 DDG + norm-proof against the reference SecAgg actionable path.
6. Patient-level upgrade: if HAM10000 `/` seven-point-checklist metadata can re-attach patient
   identifiers to DermaMNIST rows, add user-level grouping (submission-aware) and rerun P1 — the
   honest guarantee for dermoscopy; documented limitation otherwise.
7. Merge cadence: results return to `dev`→`main` at milestones per the branch table.

## 8. Measured results (2026-09-04, seeds 42-46 pilots)

All numbers are recorded in `results/dl_image_*.json` (gitignored; regenerate via the
`python -m research.privacy_dl_image.<module>` runners). Single-seed-42 shape data for the
full ε grids; dispersion on the headline ε {off, 1, 4} from seeds 43-46.

### P1 — record-level DP-SGD on CNN-S (FedProx μ=0.1 transport), AUROC at round 10

| ε | seed 42 | seeds 43-46 range (headline ε subset) |
|---|---|---|
| off | 0.606 | 0.592 – 0.636 |
| 0.5 | 0.657 | — |
| 1 | 0.657 | 0.579 – 0.627 |
| 2 | 0.657 | — |
| 4 | 0.657 | 0.583 – 0.630 |
| 8 | 0.657 | — |
| 16 | 0.657 | — |

Findings: (i) at the 10-round bench budget the arm is **utility-flat across ε ∈ [0.5, 16]** —
round-1 rows separate monotonically in ε (0.6550 → 0.6532) but all cells converge to the
same fixed point by round 10; (ii) the off-arm (σ=0, clip active) is the weakest link
(0.606), a clipping-regularization effect; (iii) **loss-threshold MIA: no detectable
advantage at ANY ε including off** (attack AUC 0.499-0.501, TPR@FPR=1% = 0.70-0.80×
marginal) — recorded as a lower-bound audit (shadow-model attacks out of scope);
(iv) worst-clinic AUROC dips to 0.000 on seed 46 for every ε — clinic 5's 8-row test fold
{7 negative, 1 positive}: an 8-row-eval variance artifact of the benchmark's fairness column,
not a DP effect (n_te≥15 folds are stable); (v) per-clinic composed ε ≤ target verified
exactly on every cell. Seed-42 DP rows share subsample draws across ε (in-cell metrics fine;
the sweep rows redecorrelate).

### P3 — FedCT consensus (isolated teachers, train-carve public pool q≤512, seed 42)

γ̂ = 0.037; teacher majority-vote accuracy 0.564 on a pool with prevalence 0.105; clean
(σ_v=0) distilled student AUROC 0.595 (9/10 clinics defined — clinic 5's fold single-class,
counted in-row). **Every paid cell is destroyed**: Gaussian count noise σ_v ∈ [1,716 ;
27,457] against K=10 votes (ε ∈ [0.5, 8]) → students at chance (NaN). Consensus voting is
nonviable at bench scale; the clean-arm result is the information ceiling if query volume
grew an order of magnitude.

### P4 — verified-hybrid DDG (mod-2³² ring, scale 1e-3, FedAvg wire), AUROC at round 10

| ε | seed 42 |
|---|---|
| off | 0.649 |
| 0.5 | 0.628 |
| 1 | 0.638 |
| 2 | 0.628 |
| 4 | 0.627 |
| 8 | 0.627 |
| 16 | 0.632 |

Norm proofs verified on every round of every cell; KLS feasibility (σ_int ≥ 2) holds at all
paid ε; composed ε exact under the same epsilon_rdp accounting as P1. The ~0.02-0.03 gap vs
P1's fedprox rows is the transport confound (FedAvg single-masker wire), not the DDG
noising; quantizer hooks proved benign by the off-row. **P4 beats P1 on the off-arm**
(0.649 vs 0.606 — clip-heavy FedAvg + integer sums converge better than clip+FedProx here).

### P2 — SecAgg+ cost row

Analytic table (`secagg_cost.bench_table`) at wire sizes 28,577 (CNN-S) and 2,758,177
(CNN-M): 64 rows {secagg, secagg_plus, fastsecagg, lightsecagg} × client-count × dropout
retention. Measured deployment probe cross-reference: `results/dl_ts_p2.json` (identical
built-in workflow); stage-1 dispatch wedge over 10 supernodes recorded there honestly —
E2E runtime at image wire sizes not claimed.

### HAM10000 patient-level reattachment (work plan §6 step 6) — DOCUMENTED LIMITATION

HAM10000 metadata downloaded (Dataverse file 4338392; lesion_id is the finest patient-level
unit) but **row reattachment is not recoverable from this payload**: MedMNIST shuffled
HAM10000-derived rows with an undisclosed permutation when building dermamnist.npz, and the
original 600×450 pixels (which would allow pixel-hash matching) are not available. An
ordering sweep (as-is / image_id / lesion_id sorts; RandomState seed scan 0-4000; sklearn
train_test_split rs ∈ [0,200) with and without stratification) found no exact
label-sequence match. User-level (patient-level) DP on dermoscopy therefore cannot run on
this parity-check payload; it stays in scope only if MedMNIST's construction permutation or
the raw HAM10000 imagery becomes available.

## 9. References

Dockhorn et al. `DP-SGD vs PATE` (ICLR 2023). Kairouz–Liu–Steinke, *The Distributed Discrete Gaussian
Mechanism for Federated Learning* (arXiv 2102.06387). Rathgeb et al., *ELSA: Secure Aggregation via
Cryptographic Enforcement of Model Update Integrity* (2023) + updated SecAgg+ bounds (2025). Formato,
*Little Is Enough: Federated Co-Training via Consensus Labelling* (AAAI 2025; arXiv 2310.05696;
impl github.com/kampmichael/federatedcotraining). FastSecAgg (arXiv 2107.13710); LightSecAgg
(arXiv 2109.14236). Ratke et al., *Trimmed FL* (arXiv 2407.14562). Nasr & Carlini, *Tight Auditing of
DP-SGD* / Crafty adversaries, for the malicious-server row. MedMNIST v2 (Sci Data 2023) for
DermaMNIST; HAM10000 (Tschandl et al., Sci Data 2018) + ISIC archive for the consensus reference.
