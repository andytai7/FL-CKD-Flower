"""Differential-privacy accounting — the single definition every consumer in this repo shares.

Three surfaces read the same accountant, so a quoted ε always means the same arithmetic:

- `privacy.py`    — the measured utility sweeps (`ckd-privacy` -> results/privacy.json)
- `audit.py`      — the membership-inference report's composed budgets
- `server_app.py` — the LIVE run, where the configured quantity is the budget itself
  (`central-dp-epsilon`) and the noise multiplier σ is derived from it by `sigma_for_epsilon`

Conventions (matching docs/PRIVACY.md §3.4):

- RDP accounting via Google's `dp_accounting` — independently maintained and widely reviewed,
  which is what makes it defensible to a data-protection reviewer (EDPB-facing documentation is
  expected to rest on a recognised accountant, not hand-rolled composition).
- Updates are clipped to `CLIPPING_NORM` at the server, so in units of the clipping norm the
  per-update sensitivity is 1.0 and the Gaussian mechanism's noise multiplier IS the accounted σ.
- δ is conventional, not consequential: set comfortably below 1/n (the pilot cohort is ~3.5k
  patients, so 1e-5 has ~30x headroom). The lawyer-facing number is ε; δ is justified by rule.
- No subsampling amplification is credited in the live path when `fraction-fit` < 1.0: FedAvg
  samples uniformly per round, not by Poisson inclusion, so the honest account composes every
  round at sampling_probability = 1. That bound holds for whichever practices participate; the
  accountant's `sampling_probability` parameter remains available to the measurement sweeps.
"""

from __future__ import annotations

import math

# δ is conventionally set below 1/n; the clinics cohort is ~3.5k patients, so 1e-5 is comfortable.
DELTA = 1e-5
# Server-side updates are clipped to this norm before noise; sensitivity is 1.0 in its units.
CLIPPING_NORM = 1.0


def epsilon_basic_composition(
    noise_multiplier: float, rounds: int, delta: float = DELTA
) -> float | None:
    """The naive Gaussian-mechanism bound: per-round eps1, composed linearly.

    Per-round eps1 = sqrt(2 ln(1.25/delta))/sigma for the Gaussian mechanism at sensitivity 1
    (updates are clipped to `CLIPPING_NORM`), composed over rounds by **basic composition**,
    eps = rounds * eps1.

    Retained only so the report can show what changed and why. **It is not a valid bound over the
    range this project sweeps**: the classical Gaussian-mechanism analysis
    eps1 = sqrt(2 ln(1.25/delta))/sigma holds only for eps1 <= 1, and every row of the old table
    violated that badly (at sigma=0.1 the per-round eps1 alone is ~48). So the old figures were not
    conservative, they were simply outside the regime where the formula says anything.

    Where both are meaningful (larger sigma, smaller eps) `epsilon_rdp` is 2-4x tighter at
    identical noise; at small sigma the naive expression actually reports *less* than the
    accountant. Use `epsilon_rdp`. This exists for the audit trail, not for quotation.
    """
    if noise_multiplier <= 0:
        return None
    per_round = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    return per_round * rounds


def epsilon_rdp(
    noise_multiplier: float,
    rounds: int,
    delta: float = DELTA,
    sampling_probability: float = 1.0,
) -> float | None:
    """(eps, delta) from a Renyi-DP accountant — the value this project reports.

    Uses Google's `dp_accounting` rather than our own arithmetic: an independently maintained,
    widely reviewed implementation is easier to defend to a data-protection reviewer than a
    hand-rolled composition, and it is what EDPB-facing documentation is expected to rest on.

    The mechanism is unchanged — same clipping norm, same Gaussian noise. Only the *accounting* is
    tighter: RDP tracks the full Renyi divergence curve and converts once at the end, instead of
    paying the union bound every round.

    `sampling_probability` < 1 credits privacy amplification by subsampling (Poisson inclusion).
    The measurement sweeps run at fraction-fit = 1.0, where there is no amplification to credit;
    the live server deliberately does NOT credit it either (see module docstring).
    """
    if noise_multiplier <= 0:
        return None

    from dp_accounting import dp_event, rdp

    gaussian = dp_event.GaussianDpEvent(noise_multiplier)
    if sampling_probability < 1.0:
        gaussian = dp_event.PoissonSampledDpEvent(sampling_probability, gaussian)

    accountant = rdp.RdpAccountant()
    accountant.compose(dp_event.SelfComposedDpEvent(gaussian, rounds))
    return float(accountant.get_epsilon(delta))


def sigma_for_epsilon(
    target_epsilon: float,
    rounds: int,
    delta: float = DELTA,
    sampling_probability: float = 1.0,
    bisection_iterations: int = 60,
) -> float:
    """The inverse of `epsilon_rdp`: the SMALLEST noise multiplier whose composed budget fits.

    This is the direction a deployment actually needs: the (ε, δ) pair is the number the privacy
    documentation commits to, and the noise is whatever that budget requires for the configured
    round count. Wiring the run this way keeps the lawyer-facing ε fixed while `num-server-rounds`
    is tuned — e.g. halving the rounds directly halves the required σ — instead of letting the
    effective budget drift with the training schedule.

    Guarantees the budget direction: the returned σ satisfies
    `epsilon_rdp(σ, rounds, delta, sampling_probability) <= target_epsilon`, so the composed
    budget never exceeds the target. ε(σ) is strictly decreasing in σ > 0, which is what makes
    the bisection sound.

    `rounds` must be the number of federated rounds the model will actually train for — pass the
    run's `num-server-rounds`, not a guess, or the guarantee quietly stops matching the run.

    `bisection_iterations` trades inversion speed for resolution. Fewer iterations return a
    slightly LARGER σ than the optimum — the budget direction (`<= target_epsilon`) is preserved
    either way, so coarse settings are safe for orchestrator grid searches that evaluate
    thousands of candidates and do not need 1e-18 resolution. The live server's quoted ε keeps
    the default.
    """
    if target_epsilon <= 0:
        raise ValueError("central DP needs a positive target epsilon; 0 requests no mechanism")
    if rounds <= 0:
        raise ValueError("central DP needs a positive round count to compose the budget over")

    # Bracket: ε(σ) → ∞ as σ → 0 (always over budget), and ε decreases monotonically with σ.
    hi = 1.0
    for _ in range(120):
        if epsilon_rdp(hi, rounds, delta, sampling_probability) <= target_epsilon:
            break
        hi *= 2.0
    else:
        raise ValueError(f"target epsilon {target_epsilon} could not be met for σ ≤ 2**120")

    lo = 0.0
    for _ in range(bisection_iterations):  # 60 ≈ 1e-18 relative resolution; `hi` stays under budget
        mid = (lo + hi) / 2.0
        if epsilon_rdp(mid, rounds, delta, sampling_probability) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi
