# Reafference-FL: Project Spec and Claude Code Prompt

A research codebase to test whether federated clients receive a measurable echo of their own
contribution, and whether cancelling it with a learned adaptive filter improves on existing
drift corrections under asynchrony.

**How to use this document.** Do not paste the whole thing into Claude Code at once. Paste
Section 0 plus Milestone 1. Get Milestone 1 working and inspect the output yourself. Only then
paste Milestone 2. The gating is deliberate and is explained in Section 0.

---

## 0. Context prompt (paste this first, every session)

```
You are helping me build a research codebase for a machine learning paper. I am a postdoc
in applied machine learning. Assume graduate-level familiarity with PyTorch, federated
optimization, and linear algebra. Do not explain basic concepts.

RESEARCH QUESTION
In federated learning, a client sends an update and receives back an aggregate that contains
its own contribution. Call the self-generated part "reafference" and the peer-generated part
"exafference" (terminology from von Holst and Mittelstaedt 1950). The client observes only
the sum. I want to know:

  (Q1) Is the reafference component large enough to matter, specifically in the parameter
       subspace the client actually cares about?
  (Q2) If so, can a client cancel it using only its own update history, with no extra
       communication?
  (Q3) Does cancelling it improve convergence relative to SCAFFOLD, and does the improvement
       scale with asynchrony as theory predicts?

NOTATION (use these names in code)
  K            number of clients
  theta_t      global model at round t
  delta_k      client k's local update, theta_k_local - theta_t
  g_t          global increment received by clients, theta_{t+1} - theta_t
  r_k          reafference: the part of g_t caused by client k
  e_k          exafference: g_t - r_k
  rho_k        reafference ratio in client k's private subspace (defined in Milestone 1)
  W_ell        efference-copy filter taps, ell = 0..L

CRITICAL METHODOLOGICAL CONSTRAINT
Milestone 1 is a diagnostic that can kill this project. It must be possible for the answer
to be "no, rho_k tracks 1/K, the effect is negligible." Do NOT implement the filter, the
baselines, or any downstream training comparison until I have personally reviewed the
Milestone 1 plots and told you to proceed. If you find yourself writing filter code during
Milestone 1, stop.

WORKING STYLE
- Small, reviewable commits. One concept per commit.
- Write the test before the implementation for anything with a closed-form expected value.
- Every experiment writes a JSON config snapshot and a git SHA into its output directory.
- Seed everything. Determinism is non-negotiable for this project.
- No plotting inside training loops. Training writes artifacts; analysis reads artifacts.
- If a design decision has more than one defensible option, stop and ask me rather than
  picking one silently.
- Prefer numpy/torch primitives over adding dependencies. Justify any new dependency.

STACK
Python 3.11, PyTorch, numpy, scipy, matplotlib, pytest, hydra-core for configs, uv for
dependency management. No federated learning framework: write a minimal simulator so we
control every detail of the aggregation and the staleness model.
```

---

## 1. Repository structure

Ask Claude Code to scaffold exactly this:

```
reafference-fl/
  pyproject.toml
  README.md
  configs/
    base.yaml
    m1_synthetic.yaml
    m1_cifar.yaml
    m2_filter.yaml
    m3_downstream.yaml
  src/refl/
    __init__.py
    data/
      splits.py          # dirichlet and pathological non-IID partitioners
      loaders.py
    models/
      linear.py          # analytically tractable case
      smallcnn.py        # CIFAR-10 / FMNIST
    fl/
      client.py
      server.py
      aggregators.py     # fedavg, coordinate median, krum
      staleness.py       # configurable p(ell) delay model
      simulator.py       # the round loop
    filters/
      efference.py       # L-tap LMS filter (Milestone 2)
      scaffold.py        # baseline control variates
    diagnostics/
      subspace.py        # private-subspace estimators
      rho.py             # the reafference ratio measurement
      spectra.py         # Fisher / Gauss-Newton eigenstructure
    analysis/
      plots.py
      tables.py
  tests/
  scripts/
    run_m1.py
    run_m2.py
    run_m3.py
  results/               # gitignored
```

---

## 2. Milestone 1: the kill test

**Paste this after Section 0. Nothing else.**

```
MILESTONE 1: measure the reafference ratio.

GOAL
Determine empirically whether a federated client's own contribution occupies an O(1)
fraction of the received global increment when projected into that client's private
parameter subspace, or whether it decays as O(1/K).

DEFINITIONS
For client k at round t:
    g_t   = sum_j w_j * delta_j       (received global increment)
    r_k   = w_k * delta_k             (oracle reafference, computable in simulation)
    P_k   = projection onto client k's "private subspace"
    rho_k = || P_k r_k ||^2 / || P_k g_t ||^2

The claim under test is that rho_k is approximately constant in K, while the global ratio
|| r_k ||^2 / || g_t ||^2 decays as 1/K.

IMPLEMENT THREE ESTIMATORS FOR P_k
  (a) ORACLE. Let U_{-k} = span of {delta_j : j != k} at this round. Set P_k = I - proj onto
      U_{-k}. This is not deployable but it is the ground truth for the claim. Use a thin QR
      or SVD; do not form d x d matrices.
  (b) FISHER. Top-r eigenvectors of client k's local empirical Fisher (use the Gauss-Newton
      approximation via per-sample gradients on a held-out local batch). r is a config
      parameter; sweep r in {8, 32, 128}.
  (c) GRADIENT-CONTRAST. Directions where client k's local gradient magnitude exceeds the
      mean client gradient magnitude by a factor tau. Cheap and deployable.

Estimator (a) validates the phenomenon. Estimators (b) and (c) test whether a client could
detect it without oracle access. Report all three.

EXPERIMENTAL GRID
  K            in {2, 5, 10, 20, 50, 100, 200}
  heterogeneity: Dirichlet alpha in {0.1, 0.5, 5.0} plus one IID control
  model        : linear.py first, then smallcnn.py on FMNIST, then CIFAR-10
  rounds       : 100, measure rho_k every round for every client
  seeds        : 5

CONTROLS (implement these as pytest tests, they must pass before I look at any result)
  - NEGATIVE CONTROL: IID data, K = 100, synchronous FedAvg. Both the global ratio and the
    oracle rho_k should be close to 1/K. If oracle rho_k is far from 1/K here, the estimator
    is broken, not interesting.
  - POSITIVE CONTROL: K = 2. Global ratio should be close to 1/2. Oracle rho_k should be
    close to 1.
  - DEGENERATE CONTROL: all clients given identical data and identical seeds. The private
    subspace should be empty or near-empty; assert that the oracle projector has near-zero
    rank and that the code raises rather than silently returning garbage.

MEMORY
Do not store every delta_k for every round at full dimension. Store:
  - full delta_k for all clients on rounds {1, 5, 10, 25, 50, 100} only
  - per-layer norms and the rho_k scalars for every round
Document this decision in the README.

OUTPUT
  results/m1/<config_hash>/
    config.json, git_sha.txt
    rho.parquet          long format: seed, round, client, K, alpha, estimator, rho, global_ratio
    spectra.npz
  Then a single figure: rho vs K on log-log axes, one line per estimator, with the 1/K
  reference line drawn. Faceted by alpha.

DECISION RULE (write this in the README so it is on record before we see results)
  If oracle rho_k follows the 1/K reference line within noise for alpha <= 0.5, the central
  claim is false and the project stops.
  If oracle rho_k is flat in K and Fisher rho_k tracks it within a factor of 2, proceed to
  Milestone 2.
  Anything in between: we discuss before proceeding.

Start by writing tests/test_rho_controls.py and the three subspace estimators. Do not write
the training loop until the estimators pass their controls on synthetic inputs where I know
the answer analytically.
```

---

## 3. Milestone 2: the filter (only after Milestone 1 passes)

```
MILESTONE 2: learned efference-copy cancellation.

MODEL
The client maintains an L-tap filter over its own update history:
    r_hat_k(t) = sum_{ell=0}^{L} W_ell * delta_k(t - ell)
    e_hat_k(t) = g_t - r_hat_k(t)
LMS update:
    W_ell <- W_ell + eta_f * e_hat_k(t) * delta_k(t - ell)^T

PARAMETERISATION
Full W_ell is d x d and infeasible. Implement three variants behind one interface:
    - scalar-per-layer:   W_ell is one scalar per parameter tensor
    - diagonal:           W_ell is elementwise, same shape as the parameters
    - low-rank:           W_ell = A_ell B_ell^T with rank q, q a config parameter
Default to scalar-per-layer. Report parameter count for each.

STABILITY
Implement normalised LMS with step size eta_f / (epsilon + ||phi_k||^2) where phi_k is the
stacked efference history. Assert the classical bound 0 < eta_f < 2 / lambda_max(R_k) is
respected; estimate lambda_max by power iteration on the running autocorrelation.

VALIDATION EXPERIMENT (this is the important one)
Inject a KNOWN staleness distribution p(ell) via fl/staleness.py. Theory says the optimal
linear filter is W_ell* = w_k * p(ell) * I. So:
    - set p(ell) to a known geometric, uniform, and bimodal distribution
    - train the filter
    - plot learned scalar taps against w_k * p(ell)
This is a recovery test with an analytic ground truth. If the taps do not recover p(ell) on
the synthetic linear model, the implementation is wrong. Make this a pytest test with a
tolerance, not just a plot.

DECORRELATION CHECK
At convergence, assert that the empirical cross-correlation between e_hat_k(t) and
delta_k(t - ell) is near zero for all ell <= L. This is the fixed-point property and it is
the honest version of the claim: decorrelation, not independence. Say so in the README.

SCAFFOLD AS A SPECIAL CASE
Implement scaffold.py, then add a test asserting that constraining L = 0 and freezing
W_0 = w_k * I reproduces SCAFFOLD's correction to numerical tolerance. This is a claim in the
paper and it should be a test, not a sentence.
```

---

## 4. Milestone 3: downstream evaluation

```
MILESTONE 3: does cancellation help.

METHODS
  fedavg, fedprox, scaffold, fedbuff, reafference-filter (ours), and
  reafference-oracle (upper bound: cancel using the true r_k, only possible in simulation)

The oracle variant is essential. It bounds how much of the available gain the learned filter
captures, and separates "the idea is wrong" from "the filter is undertrained."

PRIMARY SWEEP
  x-axis: variance of the staleness distribution p(ell), from 0 (synchronous) upward
  y-axis: rounds to reach target accuracy, and final accuracy
  fixed:  K, alpha, model, budget

FALSIFIABLE PREDICTION (state in README before running)
  At zero staleness variance the gap between reafference-filter and SCAFFOLD should be
  approximately zero. The gap should grow monotonically with staleness variance. If we see a
  gap under perfect synchrony, that is a bug in our implementation or an unfair baseline,
  not a result. Investigate before celebrating.

SECONDARY SWEEPS
  - K in {5, 10, 20, 50}: where does the effect wash out
  - aggregator in {fedavg, coordinate-median, krum}: the non-invertible case. Here the filter
    should converge to the client's expected influence E[J_k]. Log the learned scalar taps
    against the empirical selection frequency of client k under krum. That correlation is a
    result in itself.
  - drift metric: track || P_k theta_k - P_k theta_global || over rounds and test whether
    cancellation reduces private-subspace drift specifically, which is the mechanism claimed.

BASELINE FAIRNESS
Tune the learning rate for every baseline with the same budget as ours. Log the tuning grid.
An unfair baseline is the fastest way to get a paper rejected and I would rather find out
now that the effect is small.
```

---

## 5. Things to tell Claude Code to avoid

Paste this as a standing instruction:

```
ANTI-PATTERNS FOR THIS PROJECT
- Do not add a new baseline, ablation, or mechanism unless I ask. Scope creep is the main
  risk here, not underpowered experiments.
- Do not smooth, clip, or normalise rho_k for presentation. If it is noisy, show it noisy.
- Do not silently fall back to a different subspace estimator if one fails. Raise.
- Do not use a federated learning framework. We need control over the staleness model and
  the aggregation, and framework abstractions will hide exactly the quantities we measure.
- Do not write a results section, an abstract, or a paper draft. Code and figures only.
- If a test is failing and the fix is not obvious in ten minutes, tell me rather than
  loosening the tolerance.
```

---

## 6. Suggested session sequence

| Session | Scope | Done when |
|---|---|---|
| 1 | Scaffold repo, configs, data splits, linear model, minimal simulator | A synchronous FedAvg run on synthetic data converges and is deterministic across two runs with the same seed |
| 2 | Subspace estimators plus control tests | All three controls in Milestone 1 pass |
| 3 | rho measurement, K sweep on linear model | The log-log plot exists and you have looked at it |
| 4 | Extend to FMNIST small CNN | Same plot, real model, decision rule applied |
| 5 | Gate: proceed or stop | You decide, not the model |
| 6 to 7 | Filter implementation plus p(ell) recovery test | Recovery test passes on synthetic |
| 8 to 10 | Downstream sweeps | Primary sweep figure exists |

Sessions 1 to 5 are roughly two weeks part-time. That is the two-week kill test.

---

## 7. First message to send Claude Code

Section 0, then:

```
Scaffold the repository per the structure below, set up pyproject.toml with uv, and
implement only: configs/base.yaml, src/refl/data/splits.py (Dirichlet partitioner),
src/refl/models/linear.py, and a minimal synchronous FedAvg simulator in
src/refl/fl/simulator.py.

Write tests/test_determinism.py asserting that two runs with the same seed produce
bitwise-identical global model trajectories.

Do not implement diagnostics, filters, or baselines yet. Stop when the determinism test
passes and show me the simulator's round loop.
```
