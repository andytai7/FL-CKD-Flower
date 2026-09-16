"""Generate the FLIP-IT retreat deck charts from the folder's own result JSONs.

Reads:   ../data/benchmark_seed{42..46}.json, ../data/privacy.json, ../data/audit.json
Writes:  ../assets/fig0*.png  (1600x900 @200dpi)  and  ../data/benchmark_5seed_summary.json

Re-run from the repo root after regenerating results/:

    ./.venv/bin/python retreat_presentation/scripts/make_figures.py

The DP-SGD-standard scatter points in fig05 are the measured rows of
docs/PRIVACY.md §3.5 (notebooks/03_dpsgd_secagg_standard.ipynb, 5 seeds):
they are not re-computed here because that run lives in a notebook, not a CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ASSETS = HERE.parent / "assets"
SEEDS = (42, 43, 44, 45, 46)

# Deck palette (repeated in PROMPT.md so slide chrome matches the charts).
INK = "#1b4965"    # primary blue  — federated / headline
TEAL = "#2a9d8f"   # accent        — the deployment standard
AMBER = "#e9c46a"  # reference     — pooled ceiling / baselines
RED = "#e76f51"    # risk          — worst practice / collapse
GRAY = "#8d99ae"   # neutral

plt.rcParams.update({
    "font.size": 13,
    "axes.edgecolor": "#5c677d",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

FIGSIZE = (8, 4.5)
DPI = 200


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=DPI)
    plt.close(fig)
    print("wrote", ASSETS / name)


def load_benchmark_merged() -> dict:
    """Five-seed merge of the protocol benchmark (clinics + flat control)."""
    per_seed = {s: json.loads((DATA / f"benchmark_seed{s}.json").read_text()) for s in SEEDS}
    protos = ("fedprox", "fedmosaic", "local", "fedavg")
    merged: dict = {"seeds": list(SEEDS), "rounds": per_seed[SEEDS[0]]["config"]["rounds"],
                    "datasets": {}}
    for ds in ("clinics", "flat"):
        block = per_seed[SEEDS[0]]["datasets"][ds]
        out_ds: dict = {"cohort": block["cohort"], "runs": {}}
        ceil = [per_seed[s]["datasets"][ds]["centralized_ceiling"]["logreg"] for s in SEEDS]
        out_ds["ceiling"] = {
            "auc_mean": float(np.mean([c["auc"] for c in ceil])),
            "auc_std": float(np.std([c["auc"] for c in ceil])),
            "sens_mean": float(np.mean([c["sensitivity"] for c in ceil])),
            "auc_seed42": per_seed[42]["datasets"][ds]["centralized_ceiling"]["logreg"]["auc"],
        }
        for p in protos:
            runs = [per_seed[s]["datasets"][ds]["runs"][p] for s in SEEDS]
            curves = np.array([r["auc_curve"] for r in runs])
            out_ds["runs"][p] = {
                "auc_mean": float(np.mean([r["final_auc"] for r in runs])),
                "auc_std": float(np.std([r["final_auc"] for r in runs])),
                "worst_mean": float(np.mean([r["final_auc_worst"] for r in runs])),
                "worst_std": float(np.std([r["final_auc_worst"] for r in runs])),
                "sens_mean": float(np.mean([r["final_sensitivity"] for r in runs])),
                "uplink_bits": runs[0]["uplink_bits_per_client_per_round"],
                "curve_mean": curves.mean(axis=0).tolist(),
                "curve_std": curves.std(axis=0).tolist(),
            }
        merged["datasets"][ds] = out_ds
    (DATA / "benchmark_5seed_summary.json").write_text(json.dumps(merged, indent=2))
    return merged


def fig01_cohort(merged: dict) -> None:
    cohort = merged["datasets"]["clinics"]["cohort"]
    n = cohort["practice_sizes"]
    rate = [100 * r for r in cohort["ckd_rate_per_practice"]]
    x = np.arange(len(n))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, n, color=INK, alpha=0.85, label="panel size (patients)")
    ax2 = ax.twinx()
    ax2.scatter(x, rate, color=RED, zorder=3, s=55, label="CKD prevalence (%)")
    ax2.set_ylabel("CKD prevalence (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(x, [f"P{i}" for i in range(len(n))])
    ax.set_ylabel("patients")
    ax.set_title(
        f"10 synthetic GP practices, {cohort['patients_total']:,} patients — non-IID in size, prevalence, archetype",
        loc="left")
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, loc="upper left")
    _save(fig, "fig01_cohort.png")


def fig02_protocols(merged: dict) -> None:
    runs = merged["datasets"]["clinics"]["runs"]
    ceil = merged["datasets"]["clinics"]["ceiling"]
    order = ("local", "fedavg", "fedprox", "fedmosaic")
    labels = ("local\n(no federation)", "FedAvg", "FedProx (μ=0.1)", "FedMosaic")
    auc = [runs[p]["auc_mean"] for p in order]
    auc_e = [runs[p]["auc_std"] for p in order]
    worst = [runs[p]["worst_mean"] for p in order]
    worst_e = [runs[p]["worst_std"] for p in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w / 2, auc, w, yerr=auc_e, capsize=4, color=INK,
           label="global AUROC (weighted)")
    ax.bar(x + w / 2, worst, w, yerr=worst_e, capsize=4, color=RED,
           label="worst-practice AUROC")
    ax.axhline(ceil["auc_mean"], color=AMBER, ls="--", lw=2,
               label=f"pooled-data ceiling {ceil['auc_mean']:.3f} (not deployable)")
    for i, p in enumerate(order):
        ax.annotate(f"{runs[p]['uplink_bits']:.0f} bits/client/round", (x[i], 0.20),
                    ha="center", fontsize=10, color="white", rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("AUROC")
    ax.set_title("Federation recovers almost all of the pooled ceiling — and fixes the worst practice",
                 loc="left")
    ax.legend(loc="lower left", ncols=2)
    _save(fig, "fig02_protocols.png")


def fig03_convergence(merged: dict) -> None:
    runs = merged["datasets"]["clinics"]["runs"]
    ceil = merged["datasets"]["clinics"]["ceiling"]
    rounds = np.arange(1, len(runs["fedprox"]["curve_mean"]) + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for p, c, lbl in (("fedprox", INK, "FedProx / FedAvg (identical)"),
                      ("fedmosaic", TEAL, "FedMosaic"),
                      ("local", GRAY, "local (no federation)")):
        m = np.array(runs[p]["curve_mean"])
        s = np.array(runs[p]["curve_std"])
        ax.plot(rounds, m, color=c, lw=2.2, label=lbl)
        ax.fill_between(rounds, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(ceil["auc_mean"], color=AMBER, ls="--", lw=2, label="pooled ceiling")
    ax.axvspan(10, 20, color=GRAY, alpha=0.08)
    ax.annotate("converged by round ~10 —\nevery extra round only spends DP budget",
                (11.0, 0.845), fontsize=11, color="#5c677d")
    ax.set_xlabel("federation round")
    ax.set_ylabel("global AUROC (5-seed mean ± s.d.)")
    ax.set_ylim(0.75, 0.875)
    ax.set_title("Logistic protocols converge early; training alone drifts down", loc="left")
    ax.legend(loc="lower right")
    _save(fig, "fig03_convergence.png")


def fig04_dp_central(privacy: dict) -> None:
    rows = [r for r in privacy["dp_sweep"] if r["noise_multiplier"] > 0]
    base = next(r for r in privacy["dp_sweep"] if r["noise_multiplier"] == 0)
    sigma = [r["noise_multiplier"] for r in rows]
    eps = [r["epsilon"] for r in rows]
    auc = [r["auc_mean"] for r in rows]
    auc_e = [r["auc_std"] for r in rows]
    worst = [r["auc_worst_mean"] for r in rows]
    worst_e = [r["auc_worst_std"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, auc, "-o", color=INK, lw=2.2, label="global AUROC")
    ax.errorbar(x, auc, yerr=auc_e, color=INK, capsize=4, lw=1)
    ax.plot(x, worst, "-o", color=RED, lw=2.2, label="worst-practice AUROC")
    ax.errorbar(x, worst, yerr=worst_e, color=RED, capsize=4, lw=1)
    ax.axhline(base["auc_mean"], color=INK, ls=":", lw=1.5, alpha=0.6,
               label=f"no-DP global {base['auc_mean']:.3f}")
    for xi, s, e, wv in zip(x, sigma, eps, worst):
        ax.annotate(f"ε={e:.0f}" if e >= 100 else f"ε={e:.1f}", (xi, wv - 0.10),
                    ha="center", fontsize=10, color="#5c677d")
    ax.set_xticks(x, [f"σ={s:g}" for s in sigma])
    ax.set_xlabel("central-DP noise multiplier (server-side, clipping norm 1.0, 20 rounds)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Central DP: accuracy holds — the ε column is the catch (RDP, δ=1e-5)", loc="left")
    ax.annotate("ε=12.3 at σ=2.0 and the worst practice\nis already down to 0.608",
                (4.1, 0.68), fontsize=11, color=RED)
    _save(fig, "fig04_dp_central.png")


def fig05_central_vs_local(privacy: dict, merged: dict) -> None:
    central = [r for r in privacy["dp_sweep"] if r["noise_multiplier"] > 0]
    local = [r for r in privacy["local_dp_sweep"] if r["epsilon_per_round"] is not None]
    local_worst_alone = merged["datasets"]["clinics"]["runs"]["local"]["worst_mean"]
    ce = [r["epsilon"] for r in central]
    ca = [r["auc_mean"] for r in central]
    cw = [r["auc_worst_mean"] for r in central]
    le = [r["epsilon_composed"] for r in local]
    la = [r["auc_mean"] for r in local]
    lw = [r["auc_worst_mean"] for r in local]
    # Measured deployment-standard rows (docs/PRIVACY.md §3.5, notebook 03, 5 seeds).
    dpsgd_eps = [0.5, 2.0, 8.0]
    dpsgd_auc = [0.798, 0.797, 0.798]
    dpsgd_worst = [0.660, 0.649, 0.656]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ce, ca, "-o", color=INK, lw=2.2, label="central DP — global")
    ax.plot(ce, cw, "--o", color=INK, alpha=0.55, lw=1.8, label="central DP — worst practice")
    ax.plot(le, la, "-s", color=RED, lw=2.2, label="local DP — global")
    ax.plot(le, lw, "--s", color=RED, alpha=0.55, lw=1.8, label="local DP — worst practice")
    ax.scatter(dpsgd_eps, dpsgd_auc, marker="*", s=280, color=TEAL, zorder=5,
               label="deployment standard: record-level DP-SGD + SecAgg + ε-orchestrator")
    ax.scatter(dpsgd_eps, dpsgd_worst, marker="*", s=280, color=TEAL, alpha=0.55, zorder=5)
    ax.axhline(local_worst_alone, color=GRAY, ls=":", lw=1.5)
    ax.annotate("worst practice training ALONE: 0.495", (1.4, 0.51), fontsize=10, color=GRAY)
    ax.annotate("ε≈4.3: local DP worst practice 0.397\n— below not collaborating at all",
                (1.25, 0.375), fontsize=11, color=RED)
    ax.annotate("standard holds ≈0.798 at ε 0.5–8", (0.62, 0.815), fontsize=11, color=TEAL)
    ax.set_xscale("log")
    ax.set_xlabel("composed ε over the run (RDP, δ=1e-5) — smaller is more private")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.33, 0.85)
    ax.set_xlim(0.4, 1600)
    ax.set_title("Update-level local DP collapses at defensible ε; the record-level standard does not",
                 loc="left")
    ax.legend(loc="upper left", fontsize=10.5)
    _save(fig, "fig05_central_vs_local.png")


def fig06_audit(audit: dict) -> None:
    rows = audit["membership_inference"]
    labels, thr, shad = [], [], []
    for r in rows:
        if r["mechanism"] == "none":
            labels.append("no DP")
        elif r["mechanism"] == "central-dp":
            labels.append(f"central σ={r['noise_multiplier']:g}")
        else:
            labels.append(f"local ε={r['local_dp_epsilon_per_round']:g}/round")
        thr.append(r["threshold_attack_auc_mean"])
        shad.append(r["shadow_attack_auc_mean"])
    pc = audit["positive_control"]
    labels.append("positive control")
    thr.append(pc["threshold_attack_auc"])
    shad.append(pc["shadow_attack_auc"])
    x = np.arange(len(labels))
    w = 0.38
    colors = [INK] * (len(labels) - 1) + [RED]
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.bar(x - w / 2, thr, w, color=colors, label="loss-threshold attack")
    ax.bar(x + w / 2, shad, w, color=colors, alpha=0.55, label="shadow-model attack")
    ax.axhline(0.5, color=GRAY, ls="--", lw=1.5)
    ax.axhline(0.52, color=AMBER, ls=":", lw=1.5)
    ax.annotate("chance = 0.500   ·   pass bar = 0.520 (95% CI upper bound)", (3.1, 0.527),
                fontsize=10.5, color=GRAY)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("attack AUROC")
    ax.set_title("Membership-inference audit: no usable signal anywhere — control fires at 0.66",
                 loc="left")
    ax.legend(loc="lower right")
    _save(fig, "fig06_audit.png")


def main() -> None:
    merged = load_benchmark_merged()
    privacy = json.loads((DATA / "privacy.json").read_text())
    audit = json.loads((DATA / "audit.json").read_text())
    fig01_cohort(merged)
    fig02_protocols(merged)
    fig03_convergence(merged)
    fig04_dp_central(privacy)
    fig05_central_vs_local(privacy, merged)
    fig06_audit(audit)


if __name__ == "__main__":
    main()
