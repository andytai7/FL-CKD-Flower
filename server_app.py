"""Flower ServerApp — FedAvg with dual-level metric logging (Message API, flwr 1.33).

Per CLAUDE.md §5, a non-IID federation can look strong on the global aggregate while collapsing on
an outlier practice. So the evaluate-metrics aggregator reports BOTH the sample-weighted global
mean AND the worst (min) client.

Privacy note (CLAUDE.md §9, layer L5): naming a practice alongside its AUROC and cohort size is
itself a disclosure channel to whoever operates the SuperLink. Per-practice lines are therefore
suppressed for cohorts below `min-cohort-size`, and can be switched to fully anonymous reporting
with `metric-privacy = true` — which keeps the worst-practice number (the thing rule 5 exists to
protect) while dropping the identity that makes it re-identifying.
"""

from __future__ import annotations

import math
from logging import INFO

from flwr.app import (
    ArrayRecord, ConfigRecord, Context, Message, MessageType, MetricRecord, RecordDict,
)
from flwr.common.logger import log
from flwr.compat.common.recorddict_compat import arrayrecord_to_parameters
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg

from data import NUM_FEATURES
from dp import CLIPPING_NORM, DELTA, epsilon_rdp, sigma_for_epsilon
from models import make_model
from orchestrator import DpsgdOrchestrator
from orchestrator import plan as plan_clinics


def _census_round(grid: Grid, initial_arrays: ArrayRecord) -> dict[str, int]:
    """Phase 0 of the DP-SGD standard (Message API): survey every connected node for N.

    One TRAIN Message per node carries the initial arrays plus `census: 1` in the
    config; the ClientApp answers with the broadcast weights unchanged and its local
    train-split size in the reply's num-examples. That headcount is the ONLY datum the
    round collects (see orchestrator.py's module docstring for why it is deliberately
    unnoised). Replies are keyed by `str(src node id)` — the key namespace
    `DpsgdOrchestrator.configure_train` looks plans up in.
    """
    messages = [
        Message(
            content=RecordDict({
                "arrays": initial_arrays,
                "config": ConfigRecord({"census": 1}),
            }),
            message_type=MessageType.TRAIN,
            dst_node_id=node_id,
        )
        for node_id in grid.get_node_ids()
    ]
    census: dict[str, int] = {}
    for reply in grid.send_and_receive(messages):
        if reply.has_error():
            raise RuntimeError(
                f"DP-SGD census: node {reply.metadata.src_node_id} answered with an "
                f"error: {reply.error}"
            )
        metrics = next(iter(reply.content.metric_records.values()))
        census[str(reply.metadata.src_node_id)] = int(float(metrics["num-examples"]))
    return census


def _log_plan_rows(rows: list, target_epsilon: float) -> None:
    """The plan's audit surface: one line per clinic — node id, N, batch b, q, sigma, eps."""
    log(INFO, "Rule-based privacy agent: planned %d clinics to compose to EPS<=%.4g each (rules 1-4; no LLM)", len(rows), target_epsilon)
    for row in rows:
        log(
            INFO,
            "  clinic node %s: N=%d -> batch=%d, q=%.6g, sigma=%.4f, epsilon=%.4g",
            row.clinic,
            row.n_patients,
            row.batch_size,
            row.sampling_rate,
            row.sigma,
            row.achieved_epsilon,
        )

app = ServerApp()

_AGG_KEYS = ("accuracy", "sensitivity", "auc")

# Defaults for the L5 metric-hygiene layer; overridden from run_config in server_fn.
_METRIC_PRIVACY = {"enabled": False, "min_cohort_size": 0}


def configure_metric_privacy(*, enabled: bool, min_cohort_size: int) -> None:
    """Set the disclosure policy for per-practice metric logging (privacy layer L5)."""
    _METRIC_PRIVACY["enabled"] = enabled
    _METRIC_PRIVACY["min_cohort_size"] = int(min_cohort_size)


def weighted_and_worst(records: list[RecordDict], weighting_key: str = "num-examples") -> MetricRecord:
    """Aggregate per-client evaluate metrics into global weighted means + worst-client values.

    Signature matches flwr 1.33's `evaluate_metrics_aggr_fn`: it receives the reply RecordDicts and
    the key to weight by.
    """
    rows = []
    for record in records:
        metrics = next(iter(record.metric_records.values()))
        rows.append(dict(metrics))

    out = MetricRecord()
    for key in _AGG_KEYS:
        pairs = [
            (float(m[weighting_key]), float(m[key]))
            for m in rows
            if key in m and not math.isnan(float(m[key]))
        ]
        if pairs:
            total = sum(n for n, _ in pairs)
            out[key] = sum(n * v for n, v in pairs) / total
            out[f"{key}_worst"] = min(v for _, v in pairs)

    _log_per_practice(rows, weighting_key)
    return out


def _log_per_practice(rows: list[dict], weighting_key: str) -> None:
    """Dual-level visibility, subject to the L5 disclosure policy."""
    anonymous = _METRIC_PRIVACY["enabled"]
    floor = _METRIC_PRIVACY["min_cohort_size"]
    suppressed = 0

    for m in sorted(rows, key=lambda r: r.get("partition-id", -1)):
        n = int(float(m.get(weighting_key, 0)))
        if n < floor:
            suppressed += 1
            continue
        auc = float(m.get("auc", float("nan")))
        sens = float(m.get("sensitivity", float("nan")))
        if anonymous:
            print(f"      practice ··: n={'·' * 4} auc={auc:.3f} sensitivity={sens:.3f}")
        else:
            pid = int(float(m.get("partition-id", -1)))
            print(f"      practice {pid:>2}: n={n:<4} auc={auc:.3f} sensitivity={sens:.3f}")

    if suppressed:
        print(f"      [{suppressed} practice(s) suppressed: cohort < {floor} rows]")

class CohortFloorFedAvg(FedAvg):
    """FedAvg with a participation floor: practices below `min_examples` never enter the average.

    Server-side enforcement of the `min-train-examples` run config. A reply weighted by fewer
    patients than the floor is removed BEFORE the weighted average is computed, so an under-size
    practice contributes nothing to the round even if its SuperNode trains anyway (the server
    cannot trust the client side to refuse). Replies carrying an error pass through untouched so
    the base class still logs them as failures. If every practice is below the floor, the round
    updates nothing: the base returns `(None, None)` for an empty result set, which
    `strategy.start()` treats as "keep the current parameters".

    The floor travels in the run config, so the ClientApp reads the same value and skips local
    training entirely below the floor (saving the practice's compute; see `client_app.train`).
    What neither side can save is the model-broadcast bandwidth: flwr 1.33's Message-API strategy
    samples nodes by id in `configure_train` and has no property poll before sending the model —
    the classic `get_properties` gating exists only in the legacy client path.

    SecAgg+ note: under the legacy `SecAggPlusWorkflow` the server CANNOT filter post-hoc, because
    by aggregation time the contributions are already masked into one sum. There the floor works
    only as client-side refusal — a zero `num_examples` sets the protocol's weighting factor to
    zero, which is exactly what removes the contribution from the masked sum.
    """

    def __init__(self, *args, min_examples: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_examples = int(min_examples)

    def aggregate_train(self, server_round, replies):
        """Drop below-floor replies, then let the base FedAvg compute the weighted average."""
        if self.min_examples <= 0:
            return super().aggregate_train(server_round, replies)

        kept, dropped = [], 0
        for msg in replies:
            if msg.has_error():
                kept.append(msg)
                continue
            metrics = next(iter(msg.content.metric_records.values()), {})
            n = int(float(metrics.get(self.weighted_by_key, 0)))
            if n >= self.min_examples:
                kept.append(msg)
            else:
                dropped += 1

        if dropped:
            aggregating = sum(1 for m in kept if not m.has_error())
            log(
                INFO,
                "Round %s: %d practice(s) below the participation floor (%d train patients) "
                "excluded; aggregating %d",
                server_round,
                dropped,
                self.min_examples,
                aggregating,
            )
        return super().aggregate_train(server_round, kept)


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    num_rounds = int(rc["num-server-rounds"])

    configure_metric_privacy(
        enabled=bool(rc.get("metric-privacy", False)),
        min_cohort_size=int(rc.get("min-cohort-size", 0)),
    )

    # Budgets are read up front so mutual exclusion raises BEFORE any wiring. The DP-SGD
    # standard (`dpsgd-epsilon > 0`) replaces BOTH update-level mechanisms: its reported
    # number is the composed patient-level budget, and stacking another mechanism would
    # double-spend it. It composes with secure-aggregation (the pair IS the standard).
    cdp_epsilon = float(rc.get("central-dp-epsilon", 0.0) or 0.0)
    dpsgd_epsilon = float(rc.get("dpsgd-epsilon", 0.0) or 0.0)
    local_dp_epsilon = float(rc.get("local-dp-epsilon", 0.0) or 0.0)
    if dpsgd_epsilon > 0 and (cdp_epsilon > 0 or local_dp_epsilon > 0):
        raise ValueError(
            "dpsgd-epsilon is mutually exclusive with central-dp-epsilon and "
            "local-dp-epsilon: the standard already guarantees a per-patient composed "
            "budget inside every clinic; adding update-level noising would spend budget "
            "twice without changing the released information."
        )

    # Initialize global parameters from a fresh model so all clients start from the same shapes.
    model = make_model(
        str(rc["model"]),
        class_weight_balanced=bool(rc["class-weight-balanced"]),
        seed=int(rc["seed"]),
    )
    model.initialize(NUM_FEATURES)
    initial_arrays = ArrayRecord(model.get_parameters())

    if bool(rc.get("secure-aggregation", False)):
        if cdp_epsilon > 0:
            raise ValueError(
                "central-dp-epsilon and secure-aggregation cannot both be set on this code path: "
                "central DP clips and noises individual updates in the clear, which SecAgg+ masks "
                "before the server reads them. For a released-model guarantee under SecAgg+, set "
                "local-dp-epsilon instead."
            )
        _run_secure_aggregation(grid, context, initial_arrays, num_rounds)
        return

    if dpsgd_epsilon > 0:
        _run_dpsgd(grid, context, initial_arrays, num_rounds, dpsgd_epsilon)
        return

    strategy = CohortFloorFedAvg(
        min_examples=int(rc.get("min-train-examples", 0)),
        fraction_train=float(rc["fraction-fit"]),
        fraction_evaluate=1.0,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=int(rc["num-practices"]),
        evaluate_metrics_aggr_fn=weighted_and_worst,
    )
    if cdp_epsilon > 0:
        # Privacy layer L4-central, budget-first: the run is configured by the ε the privacy
        # documentation commits to, and dp.sigma_for_epsilon derives the noise multiplier that
        # keeps the composed RDP budget under it for this run's round count. Clipping norm and δ
        # match the measured sweeps (privacy.py), and no subsampling amplification is credited —
        # see dp.py for the conventions.
        cdp_delta = float(rc.get("central-dp-delta", DELTA) or DELTA)
        sigma = sigma_for_epsilon(cdp_epsilon, rounds=num_rounds, delta=cdp_delta)
        sampled = max(int(int(rc["num-practices"]) * float(rc["fraction-fit"])), 1)
        log(
            INFO,
            "Central DP: target EPS=%.4g composed over %d rounds at delta=%.3g -> "
            "noise multiplier sigma=%.4f (accounted EPS=%.4g); clipping norm %g, "
            "%d practices sampled per round",
            cdp_epsilon,
            num_rounds,
            cdp_delta,
            sigma,
            epsilon_rdp(sigma, num_rounds, cdp_delta),
            CLIPPING_NORM,
            sampled,
        )
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=sigma,
            clipping_norm=CLIPPING_NORM,
            num_sampled_clients=sampled,
        )

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": int(rc["local-epochs"])}),
    )


def _run_dpsgd(
    grid: Grid, context: Context, initial_arrays, num_rounds: int, target_epsilon: float
) -> None:
    """The deployment standard on the Message-API rails: census -> plan -> planned FedAvg.

    Executes the rule-based server agent's four rules in order: rule 1 (inventory) is
    Phase 0 below — one number per clinic, its census N, the only datum exchanged; rule 2
    (epsilon target) is the run-config `dpsgd-epsilon` handed in as `target_epsilon`;
    rule 3 (parameter calculation) is `DpsgdOrchestrator.set_census` inverting the shared
    accountant so EVERY clinic composes to `target_epsilon` over the run; rule 4
    (dispatch) is `configure_train` stamping each round's outgoing Message config with
    that clinic's own (batch-size, noise-multiplier, clipping-norm, local-epochs) plan.
    The agent is a rule table, not a model — the practices stay generic executors and
    the epsilon target is this one run-config value.

    CohortFloorFedAvg is deliberately NOT composed into this path: the census DEFINES the
    federation, and `orchestrator.plan` already refuses any batch larger than a clinic's
    N — a second size gate here would only shadow that planner guarantee. Metric hygiene
    stays where it belongs: `weighted_and_worst` still reports the weighted global mean
    alongside the worst practice, under the L5 disclosure policy, at evaluate time.
    """
    rc = context.run_config
    census = _census_round(grid, initial_arrays)
    strategy = DpsgdOrchestrator(
        target_epsilon=target_epsilon,
        epochs=int(rc["local-epochs"]),
        fed_rounds=num_rounds,
        fraction_train=float(rc["fraction-fit"]),
        fraction_evaluate=1.0,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=int(rc["num-practices"]),
        evaluate_metrics_aggr_fn=weighted_and_worst,
    )
    rows = strategy.set_census(census)
    _log_plan_rows(rows, target_epsilon)
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": int(rc["local-epochs"])}),
    )


def _run_secure_aggregation(grid: Grid, context: Context, initial_arrays, num_rounds: int) -> None:
    """Privacy layer L3: run the round under SecAgg+, so the server can only open the sum.

    This is the configuration that answers the question counsel put to the project — *does the
    aggregating party ever hold one practice's update in isolation?* Under SecAgg+ it does not:
    each practice masks its update with pairwise secrets that cancel only once enough contributions
    are combined, so the server obtains the aggregate and never an individual vector.

    **Why this is a second code path rather than a flag.** In flwr 1.33 SecAgg+ ships only in the
    legacy namespaces: `secaggplus_mod` is absent from `flwr.clientapp.mod`, and
    `SecAggPlusWorkflow.__call__` demands a `LegacyContext`, so it cannot compose with
    `strategy.start()`. It is reachable — that is what this function does — by driving
    `DefaultWorkflow` with a `LegacyContext` from inside this modern `ServerApp`. The client side
    switches on the same `secure-aggregation` config key, and its handlers accept the legacy record
    shape (`fitins.parameters`) as well as this project's `arrays` — see `client_app.py`.

    Three limits that belong next to any claim made about this path:

    1. **Semi-honest threat model.** The guarantee holds against a server that follows the protocol
       but inspects what it receives. It is not a guarantee against a server that deviates — for
       instance by running a round with a single practice, whose "aggregate" is that practice.
       `min_fit_clients` below is the control that makes such a round fail rather than succeed.
    2. **Participation stays visible.** The server always learns which practices took part, and the
       aggregate. Only the individual contribution is hidden.
    3. **The aggregate is still model parameters.** SecAgg+ answers who may see one practice's
       update; it does not by itself make the released model anonymous. That is what DP and the
       leakage audit are for.

    It also cannot protect the XGBoost path at all: `FedXgbBagging` transmits serialized decision
    trees, and there is no numeric vector to mask. A SecAgg-protected deployment is necessarily a
    logistic-regression deployment.

    **Composing the DP-SGD standard** (`dpsgd-epsilon > 0`): the run opens with a 1-round
    Phase-0 census over the same SecAgg+ rails (practices report N only, the model is left
    unchanged — rule 1, inventory), the shared planner (`orchestrator.plan`, rule 3:
    parameter calculation) computes one (batch, sigma) block per clinic against the
    fixed run-config target (rule 2: `dpsgd-epsilon`), and every subsequent round's
    `configure_fit` stamps each sampled practice's FitIns with its own plan (rule 4,
    dispatch) — patient-level DP-SGD inside the clinic, masked aggregation on the wire,
    the pair the deployment documentation calls the standard. The agent deciding the
    plans is rule-based throughout; no LLM is in the loop. The plan table below the
    census is the audit surface for that guarantee.
    """
    from flwr.common import FitIns
    from flwr.server import LegacyContext, ServerConfig
    from flwr.server.strategy import FedAvg as LegacyFedAvg
    from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

    rc = context.run_config
    num_practices = int(rc["num-practices"])
    num_shares = int(rc.get("secagg-num-shares", 3))
    threshold = int(rc.get("secagg-reconstruction-threshold", 2))

    log(
        INFO,
        "SecAgg+ enabled: %s shares, reconstruction threshold %s, %s practices required per round",
        num_shares,
        threshold,
        num_practices,
    )

    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=num_shares,
            reconstruction_threshold=threshold,
        )
    )

    base_kwargs = dict(
        fraction_fit=float(rc["fraction-fit"]),
        fraction_evaluate=1.0,
        # A round with too few practices would defeat the masking, so it must not be allowed
        # to run at all. This is the mitigation for limit (1) above.
        min_fit_clients=num_practices,
        min_evaluate_clients=1,
        min_available_clients=num_practices,
        initial_parameters=arrayrecord_to_parameters(initial_arrays, keep_input=True),
        evaluate_metrics_aggregation_fn=_legacy_weighted_and_worst,
    )

    dpsgd_epsilon = float(rc.get("dpsgd-epsilon", 0.0) or 0.0)
    if dpsgd_epsilon <= 0:
        legacy = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=LegacyFedAvg(**base_kwargs),
        )
        workflow(grid, legacy)
        return

    # ── The standard composed with SecAgg+ (the deployment this path exists for) ──────
    # The legacy rails carry the two-phase plan naturally: `configure_fit` is where a
    # clinic's instructions are stamped, `aggregate_fit` is where the census tallies.
    # Client cids are grid node ids (`GridClientProxy` constructs with
    # `cid = str(node_id)`), so the survey keys below are the same namespace the
    # Message-API census uses, and plans keyed one way resolve on either rail.

    class _CensusFedAvg(LegacyFedAvg):
        """Phase 0 over the legacy/SecAgg+ rails: one masked round that reports only N.

        Under secure-aggregation=true every TRAIN exchange rides the SecAgg+ protocol
        stages (the client-side mod requires the stage record), so the census is a
        1-round DefaultWorkflow run whose fit config carries `census: 1`. Practices
        report their headcount and echo the broadcast weights; this strategy only
        tallies cid -> N and leaves the model untouched — returning None parameters
        means the workflow keeps the broadcast model unchanged.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.sizes: dict[str, int] = {}

        def aggregate_fit(self, server_round, results, failures):
            for client, fit_res in results:
                self.sizes[str(client.cid)] = fit_res.num_examples
            return None, {}

    census_strategy = _CensusFedAvg(
        on_fit_config_fn=lambda _round: {"census": 1},
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_practices,
        min_available_clients=num_practices,
        initial_parameters=base_kwargs["initial_parameters"],
    )
    workflow(
        grid,
        LegacyContext(
            context=context, config=ServerConfig(num_rounds=1), strategy=census_strategy
        ),
    )
    if not census_strategy.sizes:
        raise RuntimeError(
            "DP-SGD census round collected no headcounts: the SecAgg+ round did not "
            "complete (not enough practices or shares). Check num-practices against the "
            "connected SuperNodes."
        )

    rows = plan_clinics(
        census_strategy.sizes,
        target_epsilon=dpsgd_epsilon,
        fed_rounds=num_rounds,
        epochs=int(rc["local-epochs"]),
    )
    plans = {row.clinic: row for row in rows}
    _log_plan_rows(rows, dpsgd_epsilon)

    class _PlannedLegacyFedAvg(LegacyFedAvg):
        """Legacy FedAvg that stamps each sampled clinic's DP-SGD plan onto its FitIns.

        The base `configure_fit` builds ONE shared FitIns handed to every sampled
        client, so the plan must never be written into the shared `config` in place —
        every client would otherwise train under the LAST clinic's sigma. Each sampled
        client gets a NEW FitIns that re-references the same Parameters but owns its own
        config dict (the round's base keys merged with `ClinicPlan.to_config()`). Lookup is
        `str(client.cid)` — the census namespace; an unsurveyed client raises rather
        than training unplanned.
        """

        def configure_fit(self, server_round, parameters, client_manager):
            stamped = []
            for client, fit_ins in super().configure_fit(
                server_round, parameters, client_manager
            ):
                try:
                    clinic_plan = plans[str(client.cid)]
                except KeyError:
                    raise KeyError(
                        "the orchestrator cannot instruct a clinic it has not surveyed"
                    ) from None
                stamped.append((
                    client,
                    FitIns(fit_ins.parameters, {**fit_ins.config, **clinic_plan.to_config()}),
                ))
            return stamped

    legacy = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=_PlannedLegacyFedAvg(**base_kwargs),
    )
    workflow(grid, legacy)


def _legacy_weighted_and_worst(results: list[tuple[int, dict]]) -> dict:
    """`weighted_and_worst` for the legacy strategy's aggregation signature.

    The legacy path hands `[(num_examples, metrics), ...]` rather than reply RecordDicts, so this
    adapts the shape and delegates. The dual-level policy itself (CLAUDE.md §0 rule 5) — global
    sample-weighted mean plus worst practice, and the L5 disclosure controls — is defined once, in
    `weighted_and_worst`, and is not duplicated here.
    """
    records = [
        RecordDict({"metrics": MetricRecord({"num-examples": float(n), **{
            k: float(v) for k, v in m.items()
        }})})
        for n, m in results
    ]
    return dict(weighted_and_worst(records))
