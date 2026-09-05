# Privacy policy — timeseries modality (ECG classification + windowed forecasting)

User-directed policy for `experiment/timeseries`, grounded in the measured matrix
(`results/dl_ts_matrix.json`; regenerators are the `research/privacy_dl_ts/*` modules).
Fixed 2026-09-05 after the P1–P4 grids landed.

## 0. Review gate (user-directed)

Nothing on this branch merges into `main` or `dev` automatically. Milestone integration is
a manual user review: the user inspects branch state (research tree, notebooks, matrices)
and personally performs or approves any merge into `dev`/`main`. Automation in this repo
(watchdog auto-commit sweeps, scheduled jobs) may operate only within this working branch;
it must never create merge commits into `dev` or `main`, and those branches are never
pushed without the user's explicit action.

## 1. Protection unit

- **CinC-2017 AF screening**: one recording = one patient → **record-level DP IS
  patient-level DP** on this payload. Headline guarantee unit.
- **Forecasting (ETTh1/ETTm1/Weather)**: the honest unit is the **trajectory** (site/series
  over its window set). User-level DP-SGD is infeasible at RNN width (σ·√d = 1,385 at ε=1,
  d=39,041 → destroyed rows, measured in `results/dl_ts_forecast_p1.json`). The
  parameter-efficient route is **measured and closed** (2026-09-05): `LinearForecaster`
  (channel-shared in96→horizon map, d=9,312) cuts σ·√d to 676/193 at ε=1/ε=4 — and the paid
  cells still collapse (etth1: MSE 2,873 at ε=1, 229 at ε=4, vs 0.93 off; weather likewise;
  `results/dl_ts_forecast_p1_linear.json`). Model-side shrinking does not reopen
  trajectory-level DP at this protocol (3 rounds × 10 clinics, clip 1); remaining levers are
  protocol-side (more clinics, fewer rounds) and are NOT claimed. Forecasting claims are
  **event-level only** (per window).

## 2. Threat model

Default: **honest-but-curious server** (record-level DP-SGD over the federated run is the
baseline guarantee). The **verified-hybrid arm (P4: SecAgg-DDG + client norm proofs)** is
the documented malicious-server posture and remains available and measured on this track
(ε-rows 0.409–0.621 AUROC, proofs verified every round) — it is not the default because its
paid-band spread is the widest of any TS arm.

## 3. Budget standard

**Fixed band: target ε ∈ [1, 4] per federation run per clinic**, δ=1e-5, RDP accounting via
`dp.py` (`sigma_for_epsilon` / `epsilon_rdp`), per-clinic standardized plans (every clinic
composes the same target over its own rounds×steps at its own q), composed ≤ target verified
in-file on every row. Below ε=1 the TS utility band erodes toward the far end of the seed
dispersion band; above ε=4 the spend buys no measured utility.

## 4. Mechanism + transport locks

- **Local optimizer: DP-Adam only** — Adam moments applied to the noised clipped aggregate
  (pure post-processing; accounting unchanged). Plain-SGD-on-noised-means did not converge
  at any lr ∈ {0.05, 0.1, 0.5} (pass-1 evidence retained in `dl_ts_p1{,_seeds}.json`);
  P1 reruns without DP-Adam are a known bad configuration.
- **Transport: FedAvgM (β=0.6, built-in)** — measured non-convergence under FedAvg and
  over-suppression under FedProx (μ=0.1) on this modality; at ε=1 under DP noise the three
  transports are within 0.002 AUROC, so the lock costs nothing under noise.
- **Clipping**: global per-record clip C=1.0 with Poisson batching; per-layer geometry is an
  image-track variant and is not adopted here.

## 5. Paradigm dispositions (with receipts)

| Paradigm | Disposition | Receipt |
|---|---|---|
| P1 record-level DP-SGD/DP-Adam | **adopted base guarantee** | `dl_ts_p1_adam*.json` |
| P2 SecAgg+ | analytic cost accepted; **deployment gated** until the stage-1 dispatch wedge (`p2fab/` probe: RUNNING + "Secure aggregation commencing" then 55 min zero-node traffic) is fixed or reported upstream | `dl_ts_p2.json` |
| P3 FedCT consensus | **retired** — σ_v 432–27,457 vs K=10 votes, every paid cell at ε≤8 collapsed to NaN | `dl_ts_p3.json` |
| P4 verified hybrid | **available** malicious-server posture, not default | `dl_ts_p4.json` |
| P1-forecast event-level | adopted at event level only | `dl_ts_forecast_p1.json` |
| Linear forecaster (trajectory-DP probe) | **closed, negative** — d=9,312 still collapses (σ·√d 676 at ε=1; MSE 2,873 vs 0.93 off); weak clean reference too (MSE 0.65–1.38 vs GRU 0.055–0.24; ACF corr 0.98+) | `dl_ts_forecast_p1_linear.json`, `dl_ts_forecast_grid.json` (`model=linear`) |

## 6. Release gates (any ε-claim on this track)

1. **5-seed dispersion band** (seeds 42–46) for the headline cell — the single-seed reading
   is inadmissible evidence on this modality (ε=1 band measured 0.439–0.625).
2. Dual metrics everywhere: weighted AND worst-clinic, NaN-skipping with skips counted.
3. Composed ε ≤ target verified per row (exact accountant assertion).
4. NaN-collapse honesty: paid cells that destroy a task are reported as rows, never dropped.
5. ACF fidelity column for forecasting rows (forecast-vs-truth ACF correlation; 0.98+ band
   on the reference grids).

## 7. Explicit non-claims

- No trajectory-level guarantee is claimed for forecasting; the unit is the right one but no
  measured mechanism sustains it (RNN-width and d=9.3k linear both collapse at this protocol).
- No secure-wire deployment claim until the SecAgg+ wedge resolves.
- No P3-variant resurrection at the same (K, q) scale — both vote-noise mechanisms are
  measured dead at K=10, q≤512.
