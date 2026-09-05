# Privacy policy — image modality (DermaMNIST melanoma)

User-directed policy for `experiment/image`, grounded in the measured matrix
(`results/dl_image_matrix.json`; regenerators are the `research/privacy_dl_image/*`
modules). Fixed 2026-09-05 after the P1–P4 + randomized-response grids landed.

## 0. Review gate (user-directed)

Nothing on this branch merges into `main` or `dev` automatically. Milestone integration is
a manual user review: the user inspects branch state (research tree, notebooks, matrices)
and personally performs or approves any merge into `dev`/`main`. Automation in this repo
(watchdog auto-commit sweeps, scheduled jobs) may operate only within this working branch;
it must never create merge commits into `dev` or `main`, and those branches are never
pushed without the user's explicit action.

## 1. Protection unit

**The image.** One DermaMNIST row is one lesion image; the MedMNIST build applied an
undisclosed row permutation to HAM10000, so **patient identity is provably unrecoverable
from this payload** (ordering sweep exhausted, documented in the track README §8). The
strongest executable guarantee is per-image record-level DP. The patient-level path is
frozen, not abandoned: it reopens only on receipt of raw HAM10000 identity mapping (external
dependency — ISIC/MedMNIST provenance), at which point per-patient clipping replaces the
per-image unit and the whole ε-grid is re-measured.

## 2. Threat model

Default: **honest-but-curious server** (per-image DP-SGD is the baseline guarantee). Because
dermoscopy is THE MIA-sensitive modality, the **verified-hybrid arm (P4)** is this track's
flagship strong posture and is *recommended for consortium deployments once its transport
confound is disentangled* (one FedProx-wire P4 arm outstanding): it beats P1's off-arm
(0.649 vs 0.630) and holds 0.627–0.638 across paid ε with norm proofs verified every round.
SecAgg+ wire: analytic cost only (shared TS probe wedge, see TS policy §5).

## 3. Budget standard

**Fixed band: target ε ∈ [1, 4] per federation run per clinic**, δ=1e-5, RDP accounting via
`dp.py`, per-clinic standardized plans, composed ≤ target verified in-row. Within this band
the measured utility is **plateau-flat** (0.655–0.657 at seed 42; 0.579–0.631 across seeds
43–46): privacy is nearly free at bench budget, so the band's LOW end is the default
(ε=1 available at no measured cost) and ε=4 is the exploration ceiling.

**Regime caveat (binding):** the plateau is a property of this bench's budget/clip/scale
(C=1, 10 rounds, CNN-S). Any change to model size, local budget, or clip re-runs the
headline seed-sweep before the "ε=1 is free" reading applies again.

## 4. Mechanism + transport locks

- **Transport: FedProx (μ=0.1)** — measured parity-oscillation fix over FedAvg on
  Dirichlet-skewed pixels; locked by the transport-gate evidence.
- **Clipping: global per-image clip C=1.0**, Poisson batching. The per-layer variant (5
  module groups, σ·√5·C) is measured and retained (`dl_image_p1_layerwise.json`): equal or
  slightly better utility at √5× per-coordinate noise; optional, not default.
- **MIA floor audit on every paid cell** (loss-threshold attack; attack AUC + TPR@FPR=1%
  recorded per row). All current cells show no detectable advantage including the non-private
  arm — a lower-bound audit only; shadow-model attacks remain the upgrade path if the MIA
  claim ever needs to carry external weight.

## 5. Paradigm dispositions (with receipts)

| Paradigm | Disposition | Receipt |
|---|---|---|
| P1 per-image DP-SGD | **adopted base guarantee** | `dl_image_p1.json`, `dl_image_p1_seeds.json` |
| P1 per-layer variant | measured option | `dl_image_p1_layerwise.json` |
| P2 SecAgg+ | analytic cost accepted; deployment gated (TS probe wedge) | `dl_image_p2.json` |
| P3 FedCT consensus (Gaussian votes) | **retired** — σ_v 335–27,457 vs K=10 votes, all paid cells at chance | `dl_image_p3.json` |
| P3 FedCT consensus (randomized-response votes) | **retired** — lands 0.47–0.54 vs 0.595 clean ceiling WITH rare-class erosion (positive recall 0.42–0.64); rerun replication exact (max |ΔAUROC| = 0.0) | `dl_image_p3_rr.json` |
| P4 verified hybrid | **flagship strong posture** (transport disentangle pending) | `dl_image_p4.json` |
| Patient-level DP | **frozen: provably unrecoverable on this payload** | README §8 |

## 6. Release gates (any ε-claim on this track)

1. Seed-dispersion band (seeds 42–46) on the headline ε cells; single-seed grid shape data
   only otherwise (seed-42 DP rows share subsample draws across ε — recorded caveat).
2. Dual metrics: weighted AND worst clinic; NaN rows (single-class held-out folds, e.g.
   clinic 5's 8-row fold) are counted explicitly, never silently dropped; an 8-row-fold
   worst-clinic artifact is labeled a fold-variance artifact, not a DP effect.
3. MIA audit row attached to every paid cell (attack AUC + TPR@1%).
4. Plateau-regime recheck (§3 caveat) for any budget/scale/clip change.
5. Composed ε ≤ target verified per row.

## 7. Explicit non-claims

- No patient-level privacy claim: the protection unit is the image, on this payload, by
  construction.
- No P3 consensus at bench scale (K=10, q≤512), under EITHER vote-noise mechanism —
  measured twice, retired twice.
- No secure-wire deployment claim until the SecAgg+ wedge resolves.
- MIA no-advantage is an attack-floor result (weak attacker class), not a general
  membership-privacy certificate; it is recorded with that label everywhere it appears.
