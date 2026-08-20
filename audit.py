"""Leakage audit — does the released model reveal who was in its training data? (T2.5, layer L6)

EDPB Opinion 28/2024 does not accept that a model is anonymous because differential privacy is
configured. It asks for a case-by-case assessment of extraction and identification risk, and names
**membership inference**, **model inversion** and **reconstruction** as the tests that establish it.
This module builds the first of those three, so the project can report an attack result rather than
an argument from epsilon.

Two attacks, both against the *released global model* — the artefact that would actually leave the
federation:

1. **Loss threshold** (Yeom et al., 2018). A model fits its training data slightly better than
   unseen data, so per-record loss is itself a membership signal. Needs no attacker training and is
   the standard first-line test.
2. **Shadow models** (Shokri et al., 2017). The attacker trains their own models on data from the
   same distribution, learns how in/out records differ, and applies that to the target. Stronger,
   and closer to what a motivated adversary would do.

Both are scored as **attack AUROC**: 0.5 is a coin flip (no leakage detected), 1.0 is perfect
membership recovery. `advantage = 2*AUC - 1` is the same number on Yeom's scale.

    uv run ckd-audit                    # -> results/audit.json

⚠️ **What a near-0.5 result does and does not mean.** The model under test is logistic regression
with eleven parameters over a few hundred patients per practice: it has very little capacity to
memorise an individual, so a weak attack is the expected outcome and is *evidence about this
method*, not proof of anonymity for the pilot. Real practice data adds eGFR and richer features. A
passing audit here is a necessary condition, never a sufficient one — and it covers one of the three
attack families EDPB names. `positive_control()` exists to show the attack fires when leakage is
actually present, because an attack that never fires is not evidence of anything.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg

from data import load_clinic_frames
from messages import hushed, local_dp_train_reply, train_reply
from models.protocols.common import (
    from_flower_arrays,
    init_weights,
    predict_proba,
    to_flower_arrays,
)
from privacy import (
    CLIPPING_NORM,
    DELTA,
    LEARNING_RATE,
    LOCAL_DP_EPSILONS,
    NOISE_MULTIPLIERS,
    _prepare,
    _seed_dp_noise,
    epsilon_rdp,
)

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

NUM_SHADOWS = 16
_EPS = 1e-12


# ── scoring ──────────────────────────────────────────────────────────────────────────────────

def per_record_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Binary cross-entropy per row — the raw membership signal both attacks read."""
    p = np.clip(predict_proba(X, w), _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def attack_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """AUROC of a membership score, computed as the Mann-Whitney statistic.

    Written out rather than imported so that ties count as half — the loss-threshold attack
    produces them, and silently breaking ties in the attacker's favour would overstate leakage.
    """
    n_m, n_n = len(member_scores), len(nonmember_scores)
    if n_m == 0 or n_n == 0:
        return float("nan")
    combined = np.concatenate([member_scores, nonmember_scores])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    # average ranks within tie groups
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[:n_m].sum() - n_m * (n_m + 1) / 2) / (n_m * n_n))


def threshold_attack(w: np.ndarray, members, nonmembers) -> float:
    """Yeom: a member should sit at lower loss. Score = -loss, so higher means 'in'."""
    return attack_auc(
        -per_record_loss(members[0], members[1], w),
        -per_record_loss(nonmembers[0], nonmembers[1], w),
    )


def shadow_attack(w: np.ndarray, members, nonmembers, rng: np.random.Generator) -> float:
    """Shokri: calibrate the loss signal against shadow models before thresholding it.

    Each shadow model is trained on a random half of the pooled cohort, so the attacker learns the
    in/out loss distributions for *this* data distribution and model class. The calibrated score is
    the target record's loss standardised against the shadow-out distribution — the record-specific
    difficulty correction that makes this stronger than a raw threshold.
    """
    X = np.vstack([members[0], nonmembers[0]])
    y = np.concatenate([members[1], nonmembers[1]])
    n = len(X)

    out_losses = []
    for _ in range(NUM_SHADOWS):
        idx = rng.permutation(n)
        in_idx = idx[: n // 2]
        w_shadow = _fit_logreg(X[in_idx], y[in_idx], n_features=X.shape[1], epochs=40)
        losses = per_record_loss(X, y, w_shadow)
        mask = np.ones(n, dtype=bool)
        mask[in_idx] = False
        held = np.full(n, np.nan)
        held[mask] = losses[mask]
        out_losses.append(held)

    out = np.vstack(out_losses)
    mu = np.nanmean(out, axis=0)
    sd = np.nanstd(out, axis=0) + _EPS

    target = per_record_loss(X, y, w)
    calibrated = -(target - mu) / sd  # lower-than-expected loss => likely a member
    return attack_auc(calibrated[: len(members[1])], calibrated[len(members[1]) :])


def _fit_logreg(X, y, n_features: int, epochs: int, lr: float = LEARNING_RATE) -> np.ndarray:
    """Plain full-batch gradient descent on the shared logistic objective.

    The shadow models are the *attacker's* models, not federation participants, so they are built
    here directly rather than through a Flower strategy — nothing about the federated protocol is
    being reimplemented (CLAUDE.md §0 rule 1).
    """
    from models.protocols.common import LogRegLocal

    local = LogRegLocal(X, y, X[:1], y[:1])
    w = init_weights(n_features)
    return local.local_sgd(w, epochs=epochs, lr=lr)


# ── the target models ────────────────────────────────────────────────────────────────────────

def _train_federated(locals_, rounds: int, seed: int, epochs: int, sigma=None, local_eps=None):
    """One federated run, returning the released global model.

    `sigma` selects central DP (noise applied by the server during aggregation) and `local_eps`
    selects local DP (noise applied inside each SuperNode before transmission). Both use Flower's
    own mechanisms; the point of auditing them separately is that they protect different things.
    """
    base = FedAvg()
    strategy = base
    if sigma:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            base, noise_multiplier=sigma, clipping_norm=CLIPPING_NORM,
            num_sampled_clients=len(locals_),
        )

    mod = None
    if local_eps:
        from flwr.clientapp.mod import LocalDpMod

        mod = LocalDpMod(
            clipping_norm=CLIPPING_NORM, sensitivity=CLIPPING_NORM,
            epsilon=local_eps, delta=DELTA,
        )

    w = init_weights(locals_[0].n_features)
    for rnd in range(1, rounds + 1):
        if sigma:
            from privacy import _array_record

            strategy.current_arrays = _array_record(w)
        global_arrays = to_flower_arrays(w)
        replies = []
        for loc in locals_:
            updated = to_flower_arrays(loc.local_sgd(w, epochs=epochs, lr=LEARNING_RATE))
            if mod is None:
                replies.append(train_reply(updated, loc.num_examples))
            else:
                with hushed():
                    replies.append(
                        local_dp_train_reply(mod, global_arrays, updated, loc.num_examples)
                    )
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        if arrays is None:
            break
        w = from_flower_arrays(arrays.to_numpy_ndarrays())
    return w


def _cohort(locals_):
    """Members = every practice's training rows; non-members = their held-out rows.

    This is the attacker-favourable framing: the non-members come from the same practices and the
    same distribution, so any separation is memorisation rather than population shift.
    """
    Xm = np.vstack([loc.X for loc in locals_])
    ym = np.concatenate([loc.y for loc in locals_])
    Xn = np.vstack([loc.X_test for loc in locals_])
    yn = np.concatenate([loc.y_test for loc in locals_])
    return (Xm, ym), (Xn, yn)


# ── the audit ────────────────────────────────────────────────────────────────────────────────

def run_membership_inference(
    rounds: int = 20, seeds: tuple[int, ...] = (42,), epochs: int = 2
) -> list[dict]:
    """Attack the released model at every privacy setting the project can deploy."""
    settings: list[tuple[str, float | None, float | None]] = [("none", None, None)]
    settings += [("central-dp", s, None) for s in NOISE_MULTIPLIERS if s > 0]
    settings += [("local-dp", None, e) for e in LOCAL_DP_EPSILONS if e]

    results = []
    for kind, sigma, local_eps in settings:
        thr, shd = [], []
        for seed in seeds:
            _seed_dp_noise(seed, sigma or local_eps or 0.0)
            locals_ = _prepare(load_clinic_frames(), seed)
            w = _train_federated(locals_, rounds, seed, epochs, sigma, local_eps)
            members, nonmembers = _cohort(locals_)
            thr.append(threshold_attack(w, members, nonmembers))
            shd.append(shadow_attack(w, members, nonmembers, np.random.default_rng(seed)))

        composed = None
        if sigma:
            composed = epsilon_rdp(sigma, rounds)
        elif local_eps:
            import math

            composed = epsilon_rdp(math.sqrt(2 * math.log(1.25 / DELTA)) / local_eps, rounds)

        row = {
            "mechanism": kind,
            "noise_multiplier": sigma,
            "local_dp_epsilon_per_round": local_eps,
            "epsilon_composed": composed,
            "seeds": list(seeds),
            "threshold_attack_auc_mean": float(np.mean(thr)),
            "threshold_attack_auc_std": float(np.std(thr)),
            "shadow_attack_auc_mean": float(np.mean(shd)),
            "shadow_attack_auc_std": float(np.std(shd)),
            "threshold_advantage": float(2 * np.mean(thr) - 1),
            "shadow_advantage": float(2 * np.mean(shd) - 1),
        }
        row["passes"] = bool(_passes(thr) and _passes(shd))
        results.append(row)

        label = kind if not sigma and not local_eps else f"{kind} {sigma or local_eps:g}"
        print(
            f"  {label:<20} threshold AUC={row['threshold_attack_auc_mean']:.3f}"
            f"±{row['threshold_attack_auc_std']:.3f}  "
            f"shadow AUC={row['shadow_attack_auc_mean']:.3f}"
            f"±{row['shadow_attack_auc_std']:.3f}  "
            f"{'PASS' if row['passes'] else 'LEAKAGE'}"
        )
    return results


# Largest membership advantage treated as negligible. An attack AUC of 0.52 means an adversary
# who must guess "was this patient in the training set?" is right 52% of the time instead of 50%.
NEGLIGIBLE_AUC_MARGIN = 0.02


def _passes(aucs: list[float]) -> bool:
    """Pass = the 95% interval's upper bound stays within a negligible advantage over chance.

    **One-sided, and on effect size rather than significance** — both deliberate, and the first
    version of this function got both wrong. A two-sided significance test flags AUC = 0.495 as a
    failure once the seed variance is small enough, even though that is an attack performing
    *worse* than a coin flip; and with enough seeds it would flag any arbitrarily tiny deviation.
    Neither says anything about disclosure risk. What matters is whether an adversary gains a
    usable advantage, so the test asks whether the plausible upper end of the attack's performance
    is still negligible.

    Passing is not a certificate. It says these two attacks found no usable signal at this cohort
    size, on synthetic data, against an eleven-parameter model — see the module docstring.
    """
    a = np.asarray(aucs, dtype=float)
    if len(a) < 2:
        return bool(a.mean() <= 0.5 + NEGLIGIBLE_AUC_MARGIN)
    upper = a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return bool(upper <= 0.5 + NEGLIGIBLE_AUC_MARGIN)


def positive_control(seed: int = 42) -> dict:
    """Does the attack fire when leakage is genuinely present?

    Same model class and same attack, but trained the way memorisation happens: one small cohort,
    many passes, no federated averaging to wash out the individual fit. If the attack cannot detect
    membership here, a near-chance result on the real target says nothing about the model and
    everything about the test — which is exactly the objection this control forecloses.
    """
    locals_ = _prepare(load_clinic_frames(), seed)
    small = locals_[6]  # the smallest practice: least data to hide in
    n = min(40, len(small.X))
    Xs, ys = small.X[:n], small.y[:n]

    w = _fit_logreg(Xs, ys, n_features=Xs.shape[1], epochs=4000, lr=2.0)
    thr = threshold_attack(w, (Xs, ys), (small.X_test, small.y_test))
    shd = shadow_attack(
        w, (Xs, ys), (small.X_test, small.y_test), np.random.default_rng(seed)
    )
    out = {
        "cohort_size": int(n),
        "threshold_attack_auc": float(thr),
        "shadow_attack_auc": float(shd),
        "detects_leakage": bool(max(thr, shd) > 0.6),
    }
    print(
        f"  positive control (n={n}, overfit): threshold AUC={thr:.3f} shadow AUC={shd:.3f}  "
        f"{'attack fires' if out['detects_leakage'] else 'ATTACK BLIND - do not trust PASS rows'}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership-inference audit (T2.5 / layer L6)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out", default=str(RESULTS_DIR / "audit.json"))
    args = parser.parse_args()

    print("Membership inference against the released global model")
    control = positive_control()
    print()
    rows = run_membership_inference(rounds=args.rounds, seeds=tuple(args.seeds))

    RESULTS_DIR.mkdir(exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "config": {
                    "rounds": args.rounds,
                    "seeds": args.seeds,
                    "num_shadow_models": NUM_SHADOWS,
                    "pass_criterion": "95% CI upper bound of attack AUC <= 0.5 + "
                                      "{}".format(NEGLIGIBLE_AUC_MARGIN),
                    "attacks": ["loss-threshold (Yeom 2018)", "shadow-model (Shokri 2017)"],
                    "not_covered": ["model inversion", "reconstruction"],
                },
                "positive_control": control,
                "membership_inference": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
