"""Build every figure, table and inline number in the FLIP-IT expert report from one pinned run.

The report goes to counsel. Every quantity in it must be traceable to code that can be re-run, and
no number may be typed into the prose by hand — so this module owns all of them:

    paper/figures/*.pdf      vector figures
    paper/tables/*.tex       booktabs tables
    paper/generated/macros.tex    a \\newcommand per inline number
    paper/generated/manifest.tex  run date, package versions, seeds
    paper/generated/data.json     the raw run, cached

Two phases, deliberately separable: `compute()` runs the experiments (slow) and caches; `render()`
turns the cache into LaTeX and PDF (fast). Iterating on a figure does not re-run the federation.

    uv run python paper/make_paper.py            # compute if no cache, then render
    uv run python paper/make_paper.py --recompute
    uv run python paper/make_paper.py --render-only

Everything routes through the project's own entry points — `run_simulation`, `run_centralized`,
`run_dp_sweep`, `run_local_dp_sweep`, `probe_secagg`, `run_membership_inference`. Nothing about the
federation is reimplemented here (CLAUDE.md §0 rules 1 and 8).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER = ROOT / "paper"
FIGURES, TABLES, GENERATED = PAPER / "figures", PAPER / "tables", PAPER / "generated"

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ── experiment configuration — copied from notebooks/01_explore_and_baselines.ipynb ──────────
SEEDS = (42, 43, 44, 45, 46)
ROUNDS = 50
CHECKPOINTS = (5, 10, 20, 35, 50)
HEADLINE_ROUND = 20
DP_ROUNDS = 20
PROTOCOLS = ("fedprox", "fedxgb", "fedmosaic")
BASELINES = ("local", "fedavg")
ALL_RUNS = PROTOCOLS + BASELINES

# ── design tokens ────────────────────────────────────────────────────────────────────────────
# Categorical slots 1-3 of the reference palette, one per protocol; the two reference baselines
# are neutral rather than a fourth and fifth hue, because "protocol vs baseline" is the real
# distinction and cycling hues would imply five peers. Validated for this three-hue subset:
# worst normal-vision dE 24.0 (floor 15), worst CVD dE 10.0 across protan/deutan/tritan (target 8).
# `fedxgb` sits at 2.82:1 on white, below the 3:1 contrast gate, so the relief rule applies and
# every figure carries direct value labels with a companion table in the report.
COLOR = {
    "fedmosaic": "#2a78d6",  # slot 1 blue
    "fedprox": "#eb6834",    # slot 2 orange
    "fedxgb": "#1baf7a",     # slot 3 aqua
    "fedavg": "#8695a1",     # baseline, neutral
    "local": "#b8c2ca",      # baseline, neutral
}
# Secondary encoding so every figure survives greyscale printing — this report gets printed.
DASH = {"fedmosaic": "-", "fedprox": "--", "fedxgb": "-.", "fedavg": (0, (1, 1.5)), "local": (0, (4, 2))}
MARKER = {"fedmosaic": "o", "fedprox": "s", "fedxgb": "^", "fedavg": "D", "local": "v"}
INK, MUTED, GRID, CEIL = "#141d24", "#5c6b77", "#e3e9ed", "#141d24"

LABEL = {
    "fedprox": "FedProx", "fedxgb": "FedXgbBagging", "fedmosaic": "FedMosaic",
    "fedavg": "FedAvg (baseline)", "local": "Local only (baseline)",
}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": 10.5, "axes.labelsize": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


# ── phase 1: compute ─────────────────────────────────────────────────────────────────────────

def compute() -> dict:
    """Run every experiment the report cites, once."""
    from audit import positive_control, run_membership_inference
    from centralized import run_centralized
    from data import load_clinic_frames, to_xy
    from privacy import probe_secagg, run_dp_sweep, run_local_dp_sweep
    from simulate import run_simulation

    print("negative-control correlation ...")
    from data import FEATURE_COLS, LABEL_COL, load_dataframe
    flat = load_dataframe()
    flat_corr = float(max(abs(flat[c].corr(flat[LABEL_COL])) for c in FEATURE_COLS))

    print("cohort ...")
    frames = load_clinic_frames()
    cohort = [
        {"practice": i, "patients": int(len(to_xy(df)[1])), "ckd_rate": float(to_xy(df)[1].mean())}
        for i, df in enumerate(frames)
    ]

    print(f"protocol runs: {len(ALL_RUNS)} comparators x {len(SEEDS)} seeds x 2 datasets ...")
    hist: dict[str, dict] = {}
    for dataset in ("clinics", "flat"):
        for run in ALL_RUNS:
            for seed in SEEDS:
                with contextlib.redirect_stdout(io.StringIO()):
                    h = run_simulation(
                        num_practices=12, num_rounds=ROUNDS, quiet=True,
                        clinics_dir=True if dataset == "clinics" else None,
                        protocol=run, seed=seed,
                    )
                # Drop wall-clock timings before caching. They are the only non-reproducible
                # field in the run, nothing in the report cites them, and leaving them in would
                # make the determinism gate (README) fail on noise rather than on a real change.
                hist[f"{dataset}|{run}|{seed}"] = [
                    {k: v for k, v in rnd.items() if k != "round-seconds"} for rnd in h
                ]
        print(f"  {dataset} done")

    print("pooled ceilings ...")
    ceiling = {}
    for dataset in ("clinics", "flat"):
        for model in ("logreg", "xgboost"):
            with contextlib.redirect_stdout(io.StringIO()):
                ceiling[f"{dataset}|{model}"] = run_centralized(
                    model, seed=42, clinics=(dataset == "clinics")
                )

    print("central DP sweep ...")
    dp = run_dp_sweep(rounds=DP_ROUNDS, seeds=SEEDS)
    print("local DP sweep ...")
    local_dp = run_local_dp_sweep(rounds=DP_ROUNDS, seeds=SEEDS)
    print("SecAgg+ probe ...")
    secagg = probe_secagg()
    print("membership-inference audit ...")
    control = positive_control()
    mia = run_membership_inference(rounds=DP_ROUNDS, seeds=SEEDS)

    import flwr
    import sklearn

    return {
        "manifest": {
            "date": date.today().isoformat(),
            "python": platform.python_version(),
            "flwr": flwr.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "seeds": list(SEEDS),
            "rounds": ROUNDS,
            "headline_round": HEADLINE_ROUND,
            "dp_rounds": DP_ROUNDS,
        },
        "cohort": cohort,
        "flat_max_abs_corr": flat_corr,
        "history": hist,
        "ceiling": ceiling,
        "dp_sweep": dp,
        "local_dp_sweep": local_dp,
        "secagg": secagg,
        "mia": {"positive_control": control, "settings": mia},
    }


# ── helpers over the cache ───────────────────────────────────────────────────────────────────

def at(data, dataset, run, seed, rnd, key="auc"):
    h = data["history"][f"{dataset}|{run}|{seed}"]
    return float(h[rnd - 1].get(key, np.nan))


def summarise(data, dataset, rnd=HEADLINE_ROUND):
    rows = {}
    for run in ALL_RUNS:
        a = [at(data, dataset, run, s, rnd) for s in SEEDS]
        w = [at(data, dataset, run, s, rnd, "auc_worst") for s in SEEDS]
        sn = [at(data, dataset, run, s, rnd, "sensitivity") for s in SEEDS]
        up = [at(data, dataset, run, s, rnd, "uplink-bits-per-client") for s in SEEDS]
        rows[run] = {
            "auc": float(np.mean(a)), "auc_sd": float(np.std(a)),
            "worst": float(np.mean(w)), "worst_sd": float(np.std(w)),
            "sensitivity": float(np.mean(sn)), "uplink": float(np.mean(up)),
        }
    return rows


def fmt(x, n=3, na="---"):
    return na if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


# ── phase 2: render ──────────────────────────────────────────────────────────────────────────

def render(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    _style()
    for d in (FIGURES, TABLES, GENERATED):
        d.mkdir(parents=True, exist_ok=True)

    clinics = summarise(data, "clinics")
    flat = summarise(data, "flat")
    ceil_auc = data["ceiling"]["clinics|logreg"]["auc"]

    macros: dict[str, str] = {}
    _fig_headline(data, clinics, ceil_auc)
    _fig_convergence(data, ceil_auc)
    _fig_dp(data, clinics["local"]["worst"])
    _fig_payload(clinics)
    _fig_mia(data)

    _tab_cohort(data)
    _tab_accuracy(clinics, data)
    _tab_rounds(data)
    _tab_negcontrol(flat, data)
    _tab_payload(clinics)
    _tab_dp_central(data)
    _tab_dp_local(data)
    _tab_mia(data)
    _tab_secagg(data)

    _macros(data, clinics, flat, ceil_auc, macros)
    _manifest(data)
    print(f"\nrendered -> {FIGURES}, {TABLES}, {GENERATED}")


def _save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", format="pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  figures/{name}.pdf")


def _fig_headline(data, clinics, ceil_auc):
    """Magnitude across a small set of named comparators -> horizontal bars, sorted."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = sorted(ALL_RUNS, key=lambda r: clinics[r]["auc"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y, h = np.arange(len(order)), 0.36

    for i, run in enumerate(order):
        c = COLOR[run]
        a, w = clinics[run]["auc"], clinics[run]["worst"]
        # y is inverted below, so the negative offset is the upper bar: global on top of worst.
        ax.barh(y[i] - h / 2, a, height=h, color=c, edgecolor="white", linewidth=1.6)
        ax.barh(y[i] + h / 2, w, height=h, color=c, alpha=0.45, edgecolor="white", linewidth=1.6)
        ax.text(a + 0.006, y[i] - h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK)
        ax.text(w + 0.006, y[i] + h / 2, f"{w:.3f}", va="center", fontsize=8, color=MUTED)

    ax.axvline(ceil_auc, color=CEIL, lw=1.1, ls=(0, (4, 3)))
    ax.text(ceil_auc - 0.008, -0.72, f"pooled ceiling {ceil_auc:.3f}",
            ha="right", va="center", fontsize=8, color=CEIL)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r] for r in order])
    ax.invert_yaxis()  # best comparator at the top, as the sort implies
    ax.set_xlim(0.42, 0.92)
    ax.set_ylim(len(order) - 0.4, -1.15)  # headroom for the legend above the bars
    ax.set_xlabel("AUROC")
    ax.set_title(f"Global vs worst-practice AUROC  ·  {HEADLINE_ROUND} rounds, {len(SEEDS)} seeds",
                 loc="left", pad=10)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=MUTED, label="global (sample-weighted mean)"),
                       Patch(facecolor=MUTED, alpha=0.45, label="worst single practice")],
              loc="upper left", fontsize=8, ncols=2)
    _save(fig, "headline")


def _fig_convergence(data, ceil_auc):
    """Change over time -> lines. Two panels, one measure each: never a second y-axis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    for ax, key, title in zip(axes, ("auc", "auc_worst"),
                              ("Global AUROC", "Worst-practice AUROC")):
        # Baselines first so the protocols draw on top; FedProx and FedAvg coincide to three
        # decimals (a real null result), so whichever is drawn second hides the other.
        for run in BASELINES + PROTOCOLS:
            arr = np.array([[at(data, "clinics", run, s, r, key) for r in range(1, ROUNDS + 1)]
                            for s in SEEDS], dtype=float)
            m, sd = arr.mean(0), arr.std(0)
            x = np.arange(1, ROUNDS + 1)
            ax.plot(x, m, color=COLOR[run], lw=1.8, ls=DASH[run], label=LABEL[run],
                    zorder=3 if run in PROTOCOLS else 2)
            ax.fill_between(x, m - sd, m + sd, color=COLOR[run], alpha=0.10, linewidth=0, zorder=1)
        if key == "auc":
            ax.axhline(ceil_auc, color=CEIL, lw=1.1, ls=(0, (4, 3)))
            ax.text(ROUNDS, ceil_auc + 0.004, "pooled ceiling", ha="right", fontsize=8, color=CEIL)
        ax.axvline(10, color=MUTED, lw=0.8, ls=":")
        lo, hi = ax.get_ylim()
        ax.text(11, lo + 0.06 * (hi - lo), "converged", fontsize=7.5, color=MUTED, va="bottom")
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("federation round")
        ax.set_xlim(1, ROUNDS)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncols=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    _save(fig, "convergence")


def _fig_dp(data, no_collab_worst: float):
    """The report's central comparison: utility against the SAME budget scale, both mechanisms."""
    import matplotlib.pyplot as plt

    global NO_COLLAB_WORST
    NO_COLLAB_WORST = no_collab_worst
    cen = [r for r in data["dp_sweep"] if r["epsilon"]]
    loc = [r for r in data["local_dp_sweep"] if r["epsilon_composed"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, key, title in zip(axes, ("auc_mean", "auc_worst_mean"),
                              ("Global AUROC", "Worst-practice AUROC")):
        sd_key = "auc_std" if key == "auc_mean" else "auc_worst_std"
        ax.errorbar([r["epsilon"] for r in cen], [r[key] for r in cen],
                    yerr=[r[sd_key] for r in cen], color=COLOR["fedmosaic"], lw=1.8,
                    marker="o", ms=5, capsize=2.5, label="central DP (server adds noise)")
        ax.errorbar([r["epsilon_composed"] for r in loc], [r[key] for r in loc],
                    yerr=[r[sd_key] for r in loc], color=COLOR["fedprox"], lw=1.8, ls="--",
                    marker="s", ms=5, capsize=2.5, label="local DP (SuperNode adds noise)")
        ax.set_xscale("log")
        ax.invert_xaxis()
        if key == "auc_worst_mean":
            # The line that decides deployability: a practice training entirely alone.
            # Placed in a blended transform (x in axes fraction, y in data) so it cannot
            # depend on the log axis limits.
            ax.axhline(NO_COLLAB_WORST, color=INK, lw=1.0, ls=(0, (4, 3)))
            ax.text(0.02, NO_COLLAB_WORST + 0.010,
                    f"worst practice training alone ({NO_COLLAB_WORST:.3f})",
                    fontsize=7.5, color=INK, ha="left",
                    transform=ax.get_yaxis_transform())
        ax.set_xlabel(r"composed $\varepsilon$ over %d rounds (RDP; stronger $\rightarrow$)"
                      % DP_ROUNDS)
        ax.set_title(title, loc="left", pad=8)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=8, loc="lower left")
    _save(fig, "dp_tradeoff")


def _fig_payload(clinics):
    """One measure across three named protocols, spanning two orders of magnitude -> log bars."""
    import matplotlib.pyplot as plt

    runs = ["fedprox", "fedmosaic", "fedxgb"]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    vals = [clinics[r]["uplink"] for r in runs]
    ax.barh(np.arange(len(runs)), vals, height=0.5,
            color=[COLOR[r] for r in runs], edgecolor="white", linewidth=1.6)
    for i, v in enumerate(vals):
        ax.text(v * 1.12, i, f"{v:,.0f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels([LABEL[r] for r in runs])
    ax.set_xlim(200, max(vals) * 3)
    ax.set_xlabel("uplink bits per practice per round (log scale)")
    ax.set_title("What each protocol puts on the wire", loc="left", pad=8)
    ax.grid(axis="y", visible=False)
    _save(fig, "payload")


def _fig_mia(data):
    """Attack strength against chance. A reference line at 0.5 is the whole point."""
    import matplotlib.pyplot as plt

    rows = data["mia"]["settings"]
    ctrl = data["mia"]["positive_control"]
    labels, thr, shd = [], [], []
    for r in rows:
        if r["mechanism"] == "none":
            labels.append("no DP")
        elif r["mechanism"] == "central-dp":
            labels.append(f"central $\\sigma$={r['noise_multiplier']:g}")
        else:
            labels.append(f"local $\\varepsilon$={r['local_dp_epsilon_per_round']:g}")
        thr.append(r["threshold_attack_auc_mean"])
        shd.append(r["shadow_attack_auc_mean"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.axhline(0.5, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(len(labels) - 0.6, 0.506, "chance", ha="right", fontsize=8, color=INK)
    ax.axhspan(0.5, 0.52, color=MUTED, alpha=0.10, linewidth=0)
    ax.plot(x, thr, color=COLOR["fedmosaic"], lw=1.6, ls="-", marker="o", ms=5,
            label="loss-threshold attack (Yeom)")
    ax.plot(x, shd, color=COLOR["fedprox"], lw=1.6, ls="--", marker="s", ms=5,
            label="shadow-model attack (Shokri)")
    ax.axhline(ctrl["shadow_attack_auc"], color=COLOR["fedxgb"], lw=1.4, ls="-.")
    ax.text(0, ctrl["shadow_attack_auc"] + 0.006,
            f"positive control (overfit model) {ctrl['shadow_attack_auc']:.3f}",
            fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("attack AUROC")
    ax.set_ylim(0.45, 0.70)
    ax.set_title("Membership inference against the released model", loc="left", pad=8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, "mia")


# ── tables ───────────────────────────────────────────────────────────────────────────────────

def _write_table(name: str, body: str):
    (TABLES / f"{name}.tex").write_text(body)
    print(f"  tables/{name}.tex")


def _tab_cohort(data):
    rows = "\n".join(
        f"    {r['practice']} & {r['patients']:,} & {r['ckd_rate'] * 100:.1f}\\% \\\\"
        for r in data["cohort"]
    )
    total = sum(r["patients"] for r in data["cohort"])
    _write_table("cohort", f"""\\begin{{tabular}}{{rrr}}
    \\toprule
    Practice & Patients & CKD prevalence \\\\
    \\midrule
{rows}
    \\midrule
    Total & {total:,} & \\\\
    \\bottomrule
\\end{{tabular}}
""")


def _tab_accuracy(clinics, data):
    order = sorted(ALL_RUNS, key=lambda r: clinics[r]["auc"], reverse=True)
    rows = []
    for r in order:
        c = clinics[r]
        kind = "protocol" if r in PROTOCOLS else "baseline"
        rows.append(
            f"    {LABEL[r]} & {kind} & {c['auc']:.3f} $\\pm$ {c['auc_sd']:.3f} & "
            f"{c['worst']:.3f} $\\pm$ {c['worst_sd']:.3f} & {c['sensitivity']:.3f} & "
            f"{c['uplink']:,.0f} \\\\"
        )
    ceil = data["ceiling"]["clinics|logreg"]
    rows.append("    \\midrule")
    rows.append(
        f"    \\emph{{Pooled ceiling}} & ceiling & \\emph{{{ceil['auc']:.3f}}} & --- & "
        f"\\emph{{{ceil['sensitivity']:.3f}}} & --- \\\\"
    )
    _write_table("accuracy", f"""\\setlength{{\\tabcolsep}}{{5pt}}\\small
\\begin{{tabular}}{{llrrrr}}
    \\toprule
    & & AUROC & Worst practice & Sensitivity & Uplink bits/round \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def _tab_rounds(data):
    rows = []
    for r in CHECKPOINTS:
        cells = " & ".join(
            f"{np.mean([at(data, 'clinics', run, s, r) for s in SEEDS]):.3f}" for run in ALL_RUNS
        )
        worst = " & ".join(
            f"{np.mean([at(data, 'clinics', run, s, r, 'auc_worst') for s in SEEDS]):.3f}"
            for run in ALL_RUNS
        )
        rows.append(f"    {r} & {cells} & {worst} \\\\")
    # Eleven columns is already at the page's limit, so the headers are abbreviated and the
    # column separation tightened rather than letting the table run into the margin.
    short = {"fedprox": "Prox", "fedxgb": "XGB", "fedmosaic": "Mosaic",
             "local": "Local", "fedavg": "Avg"}
    head = " & ".join(short[r] for r in ALL_RUNS)
    _write_table("rounds", f"""\\setlength{{\\tabcolsep}}{{4.5pt}}\\small
\\begin{{tabular}}{{r{'r' * len(ALL_RUNS)}{'r' * len(ALL_RUNS)}}}
    \\toprule
    & \\multicolumn{{{len(ALL_RUNS)}}}{{c}}{{Global AUROC}} & \\multicolumn{{{len(ALL_RUNS)}}}{{c}}{{Worst practice}} \\\\
    \\cmidrule(lr){{2-{1 + len(ALL_RUNS)}}} \\cmidrule(lr){{{2 + len(ALL_RUNS)}-{1 + 2 * len(ALL_RUNS)}}}
    Rounds & {head} & {head} \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def _tab_negcontrol(flat, data):
    order = sorted(ALL_RUNS, key=lambda r: flat[r]["auc"], reverse=True)
    rows = "\n".join(
        f"    {LABEL[r]} & {flat[r]['auc']:.3f} $\\pm$ {flat[r]['auc_sd']:.3f} & "
        f"{flat[r]['worst']:.3f} \\\\" for r in order
    )
    ceil = data["ceiling"]["flat|logreg"]["auc"]
    _write_table("negcontrol", f"""\\begin{{tabular}}{{lrr}}
    \\toprule
    & AUROC & Worst practice \\\\
    \\midrule
{rows}
    \\midrule
    \\emph{{Pooled ceiling}} & \\emph{{{ceil:.3f}}} & --- \\\\
    \\bottomrule
\\end{{tabular}}
""")


def _tab_payload(clinics):
    desc = {
        "fedprox": "model coefficients --- a parameter vector fitted to this practice's patients",
        "fedxgb": "serialized decision trees --- split thresholds are literal patient values",
        "fedmosaic": "predictions + expertise on a \\emph{public} cohort --- no parameter vector",
    }
    rows = "\n".join(
        f"    {LABEL[r]} & {desc[r]} & {clinics[r]['uplink']:,.0f} \\\\"
        for r in ("fedprox", "fedmosaic", "fedxgb")
    )
    _write_table("payload", f"""\\begin{{tabular}}{{lp{{7.4cm}}r}}
    \\toprule
    Protocol & What crosses the practice boundary & Bits/round \\\\
    \\midrule
{rows}
    \\bottomrule
\\end{{tabular}}
""")


def _tab_dp_central(data):
    rows = []
    base = data["dp_sweep"][0]["auc_mean"]
    for r in data["dp_sweep"]:
        eps = fmt(r["epsilon"], 1) if r["epsilon"] else "---"
        basic = f"{r['epsilon_basic_upper_bound']:.0f}" if r["epsilon_basic_upper_bound"] else "---"
        cost = "---" if r["noise_multiplier"] == 0 else f"{r['auc_mean'] - base:+.3f}"
        rows.append(
            f"    {r['noise_multiplier']:.2f} & {r['auc_mean']:.3f} $\\pm$ {r['auc_std']:.3f} & "
            f"{r['auc_worst_mean']:.3f} $\\pm$ {r['auc_worst_std']:.3f} & {eps} & {basic} & {cost} \\\\"
        )
    _write_table("dp_central", f"""\\begin{{tabular}}{{rrrrrr}}
    \\toprule
    Noise $\\sigma$ & AUROC & Worst practice & $\\varepsilon$ (RDP) & naive bound & AUROC cost \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def _tab_dp_local(data):
    rows = []
    base = data["local_dp_sweep"][0]["auc_mean"]
    for r in data["local_dp_sweep"]:
        eps = f"{r['epsilon_per_round']:g}" if r["epsilon_per_round"] else "off"
        comp = fmt(r["epsilon_composed"], 1) if r["epsilon_composed"] else "---"
        cost = "---" if not r["epsilon_per_round"] else f"{r['auc_mean'] - base:+.3f}"
        rows.append(
            f"    {eps} & {comp} & {r['auc_mean']:.3f} $\\pm$ {r['auc_std']:.3f} & "
            f"{r['auc_worst_mean']:.3f} $\\pm$ {r['auc_worst_std']:.3f} & {cost} \\\\"
        )
    _write_table("dp_local", f"""\\begin{{tabular}}{{rrrrr}}
    \\toprule
    $\\varepsilon$ per round & $\\varepsilon$ composed & AUROC & Worst practice & AUROC cost \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def _tab_mia(data):
    rows = []
    c = data["mia"]["positive_control"]
    rows.append(
        f"    \\emph{{Positive control}} (n={c['cohort_size']}, overfit) & --- & "
        f"\\emph{{{c['threshold_attack_auc']:.3f}}} & \\emph{{{c['shadow_attack_auc']:.3f}}} & "
        f"\\emph{{fires}} \\\\"
    )
    rows.append("    \\midrule")
    for r in data["mia"]["settings"]:
        if r["mechanism"] == "none":
            name, eps = "No DP at all", "---"
        elif r["mechanism"] == "central-dp":
            name = f"Central DP $\\sigma$={r['noise_multiplier']:g}"
            eps = fmt(r["epsilon_composed"], 1)
        else:
            name = f"Local DP $\\varepsilon$={r['local_dp_epsilon_per_round']:g}/round"
            eps = fmt(r["epsilon_composed"], 1)
        rows.append(
            f"    {name} & {eps} & {r['threshold_attack_auc_mean']:.3f} $\\pm$ "
            f"{r['threshold_attack_auc_std']:.3f} & {r['shadow_attack_auc_mean']:.3f} $\\pm$ "
            f"{r['shadow_attack_auc_std']:.3f} & {'pass' if r['passes'] else 'LEAK'} \\\\"
        )
    _write_table("mia", f"""\\begin{{tabular}}{{llrrl}}
    \\toprule
    Setting & $\\varepsilon$ composed & Threshold attack & Shadow attack & Verdict \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def _tex_escape(text: str) -> str:
    """Escape the characters LaTeX would otherwise interpret. Package and module names carry
    underscores, which are subscript operators in maths mode and an error outside it."""
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, "\\" + ch)
    return text


def _tab_secagg(data):
    s = data["secagg"]
    def yn(v):
        return "present" if v else "absent"
    rows = [
        f"    \\texttt{{secaggplus\\_mod}} in \\texttt{{flwr.clientapp.mod}} (Message API) & {yn(s['secaggplus_mod_in_new_namespace'])} \\\\",
        f"    \\texttt{{secaggplus\\_mod}} in \\texttt{{flwr.client.mod}} (legacy) & {yn(s['secaggplus_mod_in_legacy_namespace'])} \\\\",
        f"    \\texttt{{SecAggPlusWorkflow}} importable & {yn(s['secaggplus_workflow_available'])} \\\\",
        f"    \\texttt{{SecAggPlusWorkflow}} requires \\texttt{{LegacyContext}} & {'yes' if s['requires_legacy_context'] else 'no'} \\\\",
        "    DP mods in \\texttt{flwr.clientapp.mod} & "
        + ", ".join(f"\\texttt{{{_tex_escape(m)}}}" for m in s["dp_mods_in_new_namespace"])
        + " \\\\",
        f"    Implemented and verified in this repository & {'yes' if s.get('implemented_in_this_repo') else 'no'} \\\\",
    ]
    _write_table("secagg", f"""\\begin{{tabular}}{{lp{{5.2cm}}}}
    \\toprule
    Probe & Result \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


# ── macros & manifest ────────────────────────────────────────────────────────────────────────

def _macros(data, clinics, flat, ceil_auc, macros):
    dp = {r["noise_multiplier"]: r for r in data["dp_sweep"]}
    ldp = {r["epsilon_per_round"]: r for r in data["local_dp_sweep"]}
    mia = {("none" if r["mechanism"] == "none" else
            f"c{r['noise_multiplier']:g}" if r["mechanism"] == "central-dp" else
            f"l{r['local_dp_epsilon_per_round']:g}"): r for r in data["mia"]["settings"]}
    ctrl = data["mia"]["positive_control"]

    m = {
        "NumPractices": f"{len(data['cohort'])}",
        "NumPatients": f"{sum(r['patients'] for r in data['cohort']):,}",
        "PanelMin": f"{min(r['patients'] for r in data['cohort'])}",
        "PanelMax": f"{max(r['patients'] for r in data['cohort'])}",
        "PrevMin": f"{min(r['ckd_rate'] for r in data['cohort']) * 100:.1f}\\%",
        "PrevMax": f"{max(r['ckd_rate'] for r in data['cohort']) * 100:.1f}\\%",
        "NumSeeds": f"{len(SEEDS)}",
        "HeadlineRound": f"{HEADLINE_ROUND}",
        "MaxRounds": f"{ROUNDS}",
        "DPRounds": f"{DP_ROUNDS}",
        "CeilingAUC": f"{ceil_auc:.3f}",
        "CeilingXGB": f"{data['ceiling']['clinics|xgboost']['auc']:.3f}",
        "BestFedAUC": f"{clinics['fedprox']['auc']:.3f}",
        "BestFedWorst": f"{clinics['fedprox']['worst']:.3f}",
        "LocalAUC": f"{clinics['local']['auc']:.3f}",
        "LocalWorst": f"{clinics['local']['worst']:.3f}",
        "WorstGain": f"{clinics['fedprox']['worst'] - clinics['local']['worst']:+.3f}",
        "CeilingGapAbs": f"{abs(clinics['fedprox']['auc'] - ceil_auc):.3f}",
        "FedAvgAUC": f"{clinics['fedavg']['auc']:.3f}",
        "MosaicAUC": f"{clinics['fedmosaic']['auc']:.3f}",
        "XGBAUC": f"{clinics['fedxgb']['auc']:.3f}",
        "XGBWorst": f"{clinics['fedxgb']['worst']:.3f}",
        "XGBGlobalGap": f"{clinics['fedprox']['auc'] - clinics['fedxgb']['auc']:.3f}",
        "XGBWorstGap": f"{clinics['fedprox']['worst'] - clinics['fedxgb']['worst']:.3f}",
        "XGBBandwidthRatio": f"{clinics['fedxgb']['uplink'] / clinics['fedprox']['uplink']:.0f}",
        "UplinkLogreg": f"{clinics['fedprox']['uplink']:,.0f}",
        "UplinkXGB": f"{clinics['fedxgb']['uplink']:,.0f}",
        "UplinkMosaic": f"{clinics['fedmosaic']['uplink']:,.0f}",
        "NegControlMax": f"{max(flat[r]['auc'] for r in ALL_RUNS):.3f}",
        "NegControlCeiling": f"{data['ceiling']['flat|logreg']['auc']:.3f}",
        "EpsAtSigmaTwo": f"{dp[2.0]['epsilon']:.1f}",
        "EpsAtSigmaTwoNaive": f"{dp[2.0]['epsilon_basic_upper_bound']:.0f}",
        "AUCAtSigmaTwo": f"{dp[2.0]['auc_mean']:.3f}",
        "WorstAtSigmaTwo": f"{dp[2.0]['auc_worst_mean']:.3f}",
        "EpsAtSigmaOne": f"{dp[1.0]['epsilon']:.1f}",
        "AUCAtSigmaOne": f"{dp[1.0]['auc_mean']:.3f}",
        "WorstAtSigmaOne": f"{dp[1.0]['auc_worst_mean']:.3f}",
        "LocalEpsFive": f"{ldp[5.0]['epsilon_composed']:.1f}",
        "LocalAUCFive": f"{ldp[5.0]['auc_mean']:.3f}",
        "LocalWorstFive": f"{ldp[5.0]['auc_worst_mean']:.3f}",
        "LocalEpsOne": f"{ldp[1.0]['epsilon_composed']:.1f}",
        "LocalAUCOne": f"{ldp[1.0]['auc_mean']:.3f}",
        "LocalWorstOne": f"{ldp[1.0]['auc_worst_mean']:.3f}",
        "LocalVsCentralAUCGap": f"{dp[1.0]['auc_mean'] - ldp[5.0]['auc_mean']:.3f}",
        "LocalVsCentralWorstGap": f"{dp[1.0]['auc_worst_mean'] - ldp[5.0]['auc_worst_mean']:.3f}",
        "NegControlMaxCorr": f"{data['flat_max_abs_corr']:.3f}",
        "MIANoDPThreshold": f"{mia['none']['threshold_attack_auc_mean']:.3f}",
        "MIANoDPShadow": f"{mia['none']['shadow_attack_auc_mean']:.3f}",
        "MIAControlThreshold": f"{ctrl['threshold_attack_auc']:.3f}",
        "MIAControlShadow": f"{ctrl['shadow_attack_auc']:.3f}",
        "MIAControlN": f"{ctrl['cohort_size']}",
        "MIAMargin": f"{__import__('audit').NEGLIGIBLE_AUC_MARGIN:.2f}",
        "FlwrVersion": data["manifest"]["flwr"],
        "RunDate": data["manifest"]["date"],
    }
    m["NumFeatures"] = "10"
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    (GENERATED / "macros.tex").write_text(
        "% GENERATED by paper/make_paper.py -- do not edit.\n"
        "% Every number quoted in the report's prose resolves through one of these.\n"
        + body + "\n"
    )
    print(f"  generated/macros.tex ({len(m)} macros)")


def _manifest(data):
    mf = data["manifest"]
    (GENERATED / "manifest.tex").write_text(f"""% GENERATED by paper/make_paper.py -- do not edit.
\\begin{{tabular}}{{ll}}
    \\toprule
    Item & Value \\\\
    \\midrule
    Run date & {mf['date']} \\\\
    Python & {mf['python']} \\\\
    \\texttt{{flwr}} & {mf['flwr']} \\\\
    \\texttt{{numpy}} & {mf['numpy']} \\\\
    \\texttt{{scikit-learn}} & {mf['sklearn']} \\\\
    Seeds & {', '.join(str(s) for s in mf['seeds'])} \\\\
    Rounds recorded & {mf['rounds']} (headline read at {mf['headline_round']}) \\\\
    DP / audit rounds & {mf['dp_rounds']} \\\\
    \\bottomrule
\\end{{tabular}}
""")
    print("  generated/manifest.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FLIP-IT report's figures and tables")
    parser.add_argument("--recompute", action="store_true", help="ignore the cache and re-run")
    parser.add_argument("--render-only", action="store_true", help="render from cache; do not run")
    args = parser.parse_args()

    cache = GENERATED / "data.json"
    if args.render_only or (cache.exists() and not args.recompute):
        print(f"using cache {cache}")
        data = json.loads(cache.read_text())
    else:
        data = compute()
        GENERATED.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1))
        print(f"cached -> {cache}")

    render(data)


if __name__ == "__main__":
    main()
