"""Level 3 federated orchestrator — the rule-based server agent for per-clinic DP-SGD settings.

The planning logic in this file is a rule-based server agent, not a learned component: there is
no LLM, no learned policy, and no stochastic choice anywhere in planning. Every decision is
deterministic if-then logic plus arithmetic, so the same census always yields the same plan.
The agent's complete rule table (single source of truth: the `AGENT_RULES` constant below):

Rule 1 — Inventory: at the start of the run, count the connected SuperNodes and read each
clinic's reported census N. N is the only datum exchanged.

Rule 2 — Epsilon target: one fixed global policy for the whole federation — the run-config
`dpsgd-epsilon` (epsilon*) plus delta = 1e-5. Same target for every clinic.

Rule 3 — Parameter calculation: for each clinic, invert the shared RDP accountant
(`dp.sigma_for_epsilon`): the smallest sigma_k with epsilon(sigma_k; q = b/N_k,
T = ceil(E*N_k/b) per round, R rounds) <= epsilon*, and pick b from the candidate grid to
minimise per-epoch injected noise. This is the honest form of the hand-tuned sketch
'if N >= 1000 then sigma=1.1, b=64; if N < 1000 then raise sigma or b' — a fixed sigma per size
band does NOT equalise epsilon (measured: uniform settings give epsilon 1.10 vs 16.04 for 50k
vs 500 patients; `uniform_settings_audit`), so the rule is an inversion, not a lookup.

Rule 4 — Dispatch: `ClinicPlan.to_config()` bundled into the ConfigRecord stamped per outgoing
instruction; clinics are generic executors. Change epsilon* -> edit one server config value;
zero phone calls.

The agent is a rule table, not a model: every decision above is deterministic if-then logic and
one accountant inversion — reproducible from the code, auditable line by line, and unchanged
run to run.

Why N drives the plan: patient-level DP-SGD with Poisson sampling composes to an epsilon that
depends on the clinic's dataset size N, through the sampling rate q = batch/N and the step count
T = ceil(epochs*N/batch). Hand every clinic in a heterogeneous federation the SAME config (same
batch, same sigma) and the privacy they actually deliver is incoherent: a 50k-patient urban centre
inherits aggressive amplification and finishes far below the target, while a 500-patient rural
clinic composes to a far larger epsilon than anyone signed for. `uniform_settings_audit` reproduces
that failure mode as the motivating number; `plan` is the fix.

The fix is server-side standardisation. The server collects exactly ONE number per clinic — the
census N — and inverts the shared accountant (`dp.sigma_for_epsilon`, `dp.epsilon_rdp`) to pick a
(batch, sigma) pair per clinic such that EVERY clinic composes to the same target epsilon over the
configured run, regardless of N. Among the batch candidates that fit inside N, it keeps the one
that minimises expected per-epoch gradient-noise variance, so the fairness guarantee costs as
little accuracy as possible.

Privacy of the plan itself: N is the only information that ever leaves a clinic for this purpose.
It is low-sensitivity metadata (a headcount, not a record), and the deployment runs SecAgg+ masking
over the updates proper, so a census share is the weakest link in the pipeline by design. If a
consortium contested even that, N could itself be released under a small counting-DP mechanism
before planning — the orchestrator's arithmetic is a monotone function of N, so a noised census
still produces a valid (slightly conservative if N is over-reported...) plan. That is a deliberate
NON-implementation: the pilot's data-protection agreement already contemplates census exchange.

Accounting contract (shared with dp.py — never hand-roll composition): the per-step event is
Poisson(q) * Gaussian(sigma) with q = batch/N; per-round steps T = ceil(epochs*N/batch); total
steps = fed_rounds * T. `dp.sigma_for_epsilon` returns the smallest sigma whose composed budget
fits the target, guaranteeing `achieved_epsilon <= target_epsilon` on every plan row.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import dp
from flwr.app import ArrayRecord, ConfigRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

# Batch sizes the orchestrator is willing to consider, ascending. The sweep criteria below assume
# this ordering: ties on the noise objective resolve toward the LARGER batch (fewer optimisation
# steps per epoch, and a better-conditioned noisy-average estimator).
BATCH_GRID = (8, 16, 32, 64, 128, 256)

_UNSET_CLINIC_MSG = "the orchestrator cannot instruct a clinic it has not surveyed"
# The rule-based server agent's complete decision table — the single source of truth that docs,
# notebooks, and diagrams quote. Four deterministic if-then rules and one accountant inversion;
# no LLM, no learned policy, no stochastic choice: the same census always yields the same plan.
AGENT_RULES: tuple[str, ...] = (
    "inventory: count the connected SuperNodes at run start and read each clinic's reported "
    "census N; N is the only datum exchanged",
    "epsilon-target: one fixed global policy for the whole federation — run-config "
    "dpsgd-epsilon (epsilon*) plus delta = 1e-5; same target for every clinic",
    "parameter-calculation: for each clinic invert the shared RDP accountant "
    "(dp.sigma_for_epsilon): the smallest sigma_k with epsilon(sigma_k; q = b/N_k, "
    "T = ceil(E*N_k/b) per round, R rounds) <= epsilon*, choosing b from the candidate grid to "
    "minimise per-epoch injected noise — an inversion, not a size-band lookup (uniform "
    "settings give epsilon 1.10 vs 16.04 for 50k vs 500 patients)",
    "dispatch: ClinicPlan.to_config() bundled into the ConfigRecord stamped per outgoing "
    "instruction; clinics are generic executors; change epsilon* = one server config edit",
)


@dataclass(frozen=True)
class ClinicPlan:
    """The DP-SGD settings one clinic must run so the whole federation lands on one epsilon.

    Every field except `sigma` and `achieved_epsilon` is either an input (census, grid candidate)
    or closed-form arithmetic on inputs; those two come from the shared accountant and are the
    plan's guarantee: `achieved_epsilon <= target_epsilon` under Poisson(q)*Gaussian(sigma)
    composed over `fed_rounds * steps_per_round` steps (`dp.epsilon_rdp`).
    """

    clinic: str
    n_patients: int
    batch_size: int
    sampling_rate: float  # q = batch_size / n_patients
    steps_per_round: int  # ceil(local_epochs * n_patients / batch_size)
    sigma: float
    local_epochs: int
    achieved_epsilon: float
    # Not a planner decision: the clipping norm the run already commits to (dp.CLIPPING_NORM),
    # carried so `to_config` can hand the executor a self-contained instruction block.
    clip: float = dp.CLIPPING_NORM

    def to_config(self) -> dict:
        """The instruction block stamped onto this clinic's outgoing train Message.

        Key names mirror the run-config conventions already used by the ClientApp
        (`local-epochs`), so a generic executor reads this dict with no orchestrator-specific
        code. `dp-sgd` is the explicit opt-in flag: a clinic that receives it MUST clip every
        per-sample gradient to `clipping-norm` before averaging and inject Gaussian noise at
        `noise-multiplier` — clipping is what bounds the weighted per-record contribution that
        `sigma` was calibrated against.
        """
        return {
            "dp-sgd": True,
            "batch-size": self.batch_size,
            "noise-multiplier": self.sigma,
            "clipping-norm": self.clip,
            "local-epochs": self.local_epochs,
        }


def _noise_variance(n_patients: int, batch: int, sigma: float, clip: float) -> float:
    """Expected per-epoch gradient-noise variance for a (batch, sigma) candidate.

    Per step, the Poisson batch has expected size `batch`, the mechanism adds noise of norm
    sigma*clip along the averaged direction, and an epoch runs n_patients/batch steps; in
    expectation the accumulated noise per epoch scales as (N/batch) * (sigma*clip/batch)^2.
    Smaller is better — this is the accuracy cost the fairness guarantee buys at this candidate.
    """
    return (n_patients / batch) * (sigma * clip / batch) ** 2


def plan(
    sizes: dict[str, int],
    *,
    target_epsilon: float,
    fed_rounds: int,
    epochs: int,
    delta: float = dp.DELTA,
    clip: float = dp.CLIPPING_NORM,
    batch_grid: tuple[int, ...] = BATCH_GRID,
    bisection_iterations: int = 35,
) -> list[ClinicPlan]:
    """Compute per-clinic DP-SGD settings so every clinic composes to `target_epsilon`.

    For each clinic and each grid batch with batch <= N: q = batch/N, T = ceil(epochs*N/batch),
    total steps = fed_rounds*T, and sigma is the smallest noise multiplier whose composed budget
    fits the target under Poisson(q)*Gaussian(sigma) (`dp.sigma_for_epsilon`). The winning
    candidate minimises `_noise_variance`; ties resolve toward the larger batch (see BATCH_GRID).
    Every returned row re-verifies itself: `achieved_epsilon <= target_epsilon`.

    Rows come back in the census dict's iteration order so the tally a deployment logs matches
    the census it was given. Raises ValueError if any clinic is smaller than the smallest grid
    batch — DP-SGD with batch > N degrades to full-batch noise at q=1 and this function refuses
    to pretend that is the same mechanism the budget was computed for.

    `bisection_iterations` trades planner wall-time for sigma resolution: 35 iterations is
    ~1e-10 relative resolution on sigma and the budget direction is preserved either way (fewer
    iterations return a slightly larger, still budget-valid sigma); if a deployment epsilon is
    ever published, re-verify the quote with `dp.sigma_for_epsilon`'s default iteration count.
    """
    min_batch = min(batch_grid)
    for clinic, n in sizes.items():
        if n < min_batch:
            raise ValueError(
                f"clinic '{clinic}' reports N={n}, below the smallest candidate batch "
                f"{min_batch}; Poisson DP-SGD needs at least one full batch to draw from"
            )

    plans: list[ClinicPlan] = []
    for clinic, n in sizes.items():
        best: ClinicPlan | None = None
        for batch in batch_grid:
            if batch > n:
                continue
            q = batch / n
            steps_per_round = math.ceil(epochs * n / batch)
            total_steps = fed_rounds * steps_per_round
            sigma = dp.sigma_for_epsilon(
                target_epsilon, total_steps, delta, q, bisection_iterations=bisection_iterations
            )
            achieved = dp.epsilon_rdp(sigma, total_steps, delta, q)
            assert achieved is not None and achieved <= target_epsilon, (
                f"sigma_for_epsilon returned sigma={sigma} but composed epsilon {achieved} "
                f"exceeds target {target_epsilon}; the accountant's inversion guarantee broke"
            )
            candidate = ClinicPlan(
                clinic=clinic,
                n_patients=n,
                batch_size=batch,
                sampling_rate=q,
                steps_per_round=steps_per_round,
                sigma=sigma,
                local_epochs=epochs,
                achieved_epsilon=achieved,
                clip=clip,
            )
            # Strict improvement replaces; exact ties keep going so the LARGER batch wins
            # (`<=` on an ascending grid). Floating-point ties are how two adjacent batches
            # frequently price out identically after sigma rounding.
            if best is None or (
                _noise_variance(n, batch, sigma, clip)
                <= _noise_variance(n, best.batch_size, best.sigma, clip)
            ):
                best = candidate
        plans.append(best)
    return plans


def uniform_settings_audit(
    sizes: dict[str, int],
    *,
    batch_size: int,
    sigma: float,
    epochs: int,
    fed_rounds: int,
    delta: float = dp.DELTA,
) -> list[dict]:
    """What one-size-fits-all DP-SGD actually delivers: identical config, incoherent epsilon.

    Every clinic runs the same (batch_size, sigma, epochs); the composed budget is still
    clinic-specific because q and the step count depend on N. Expect a monotone split across
    the cohort — the big clinic's amplification (small q) drags its epsilon far below the small
    clinic's, which composes near the unamplified value. This is the number that justifies the
    orchestrator: under uniform settings nobody in the federation knows which epsilon they
    personally trained under.

    For batch_size >= N the sampling rate clamps to 1.0 (no amplification to credit), matching
    the accountant's own convention that sampling_probability=1 composes the raw mechanism.
    """
    audit: list[dict] = []
    for clinic, n in sizes.items():
        q = min(1.0, batch_size / n)
        steps_per_round = math.ceil(epochs * n / batch_size)
        achieved = dp.epsilon_rdp(sigma, fed_rounds * steps_per_round, delta, q)
        audit.append(
            {
                "clinic": clinic,
                "n_patients": n,
                "sampling_rate": q,
                "steps_per_round": steps_per_round,
                "sigma": sigma,
                "achieved_epsilon": achieved,
            }
        )
    return audit


class DpsgdOrchestrator(FedAvg):
    """The rule-based server agent (server brain) of the DP-SGD deployment, built on FedAvg.

    Aggregation is inherited from Flower's FedAvg UNCHANGED (CLAUDE.md rule: never reimplement
    averaging) — this class adds standardisation of instructions, not a new federation
    algorithm. It executes the four AGENT_RULES as pure if-then logic: Rule 1 (inventory) via
    the census dict handed to `set_census`; Rule 2 (epsilon target) via the constructor's
    fixed (target_epsilon, delta); Rule 3 (parameter calculation) via `orchestrator.plan`'s
    accountant inversion; Rule 4 (dispatch) via `configure_train`'s per-message ConfigRecord
    stamping. There is no LLM, no learned policy, and no stochastic choice anywhere in this
    planning path — the same census always yields the same plan.

    Live deployment mapping: under the Message API the planning logic here runs inside
    `configure_train` — the strategy looks up the reply's node, and stamps each outgoing
    instruction Message's config ConfigRecord with `client_config(node)` (= the clinic's
    `ClinicPlan.to_config()`). The 15-25 SuperNodes stay generic executors: they read the
    standard keys (`dp-sgd`, `batch-size`, `noise-multiplier`, ...) and train accordingly, and
    they carry NO privacy policy of their own. The consequence a consortium actually cares
    about: changing the target epsilon is a server-side config edit — one line in the run
    config — not 25 phone calls and 25 redeploys.

    The ONLY cross-organisational input is the census dict handed to `set_census` (see the
    module docstring for why N is deliberately collected unnoised). Call it once when the
    baseline cohort is surveyed; `client_config` then serves per-clinic instructions, and
    raises KeyError for a clinic the census never covered rather than silently running it
    unplanned.
    """

    def __init__(
        self,
        *,
        target_epsilon: float,
        epochs: int,
        fed_rounds: int,
        delta: float = dp.DELTA,
        clip: float = dp.CLIPPING_NORM,
        **kwargs,
    ):
        # Everything else (fraction_train, evaluate_metrics_aggr_fn=weighted_and_worst, ...)
        # flows through to FedAvg untouched.
        super().__init__(**kwargs)
        self.target_epsilon = float(target_epsilon)
        self.epochs = int(epochs)
        self.fed_rounds = int(fed_rounds)
        self.delta = float(delta)
        self.clip = float(clip)
        self.plans: dict[str, ClinicPlan] = {}

    def set_census(self, sizes: dict[str, int]) -> list[ClinicPlan]:
        """Survey the cohort: compute and store one plan per clinic, keyed by clinic name."""
        rows = plan(
            sizes,
            target_epsilon=self.target_epsilon,
            fed_rounds=self.fed_rounds,
            epochs=self.epochs,
            delta=self.delta,
            clip=self.clip,
        )
        self.plans = {row.clinic: row for row in rows}
        return rows

    def client_config(self, clinic: str) -> dict:
        """The instruction block for one clinic, from its stored plan."""
        try:
            return self.plans[clinic].to_config()
        except KeyError:
            raise KeyError(_UNSET_CLINIC_MSG) from None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable:
        """FedAvg's round, then per-node stamping of each sampled clinic's DP-SGD plan.

        Flower's FedAvg packs ONE RecordDict into every node's Message
        (`FedAvg._construct_messages`): all outgoing messages carry the SAME content
        object, so mutating `msg.content` — or the ConfigRecord inside it — would hand
        every clinic the LAST clinic's plan. This override therefore never writes into
        the shared content: each message gets a FRESH RecordDict that re-references the
        ArrayRecord (identical for every node by construction) and holds its own
        ConfigRecord copy — the inherited keys (which include the `server-round` FedAvg
        injected) merged with this clinic's `ClinicPlan.to_config()` block.

        The lookup key is the reply-destination node id as a string — the same key the
        Phase-0 census reports (`server_app._census_round` keys replies by
        `str(src node id)`; in the legacy path `GridClientProxy.cid == str(node_id)`).
        A sampled node with no census row raises the unsurveyed KeyError from
        `client_config` rather than training unplanned.
        """
        stamped = []
        for msg in super().configure_train(server_round, arrays, config, grid):
            plan_config = self.client_config(str(msg.metadata.dst_node_id))
            record = RecordDict()
            for key, array_record in msg.content.array_records.items():
                record[key] = array_record
            node_config = ConfigRecord(dict(msg.content[self.configrecord_key]))
            for key, value in plan_config.items():
                node_config[key] = value
            record[self.configrecord_key] = node_config
            msg.content = record
            stamped.append(msg)
        return stamped
