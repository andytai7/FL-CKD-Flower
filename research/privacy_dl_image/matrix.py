"""Matrix emitter (image track): results/dl_image_matrix.json — same schema as the TS track.

Sources: dl_image_p1.json (seed-42 full grid), dl_image_p1_seeds.json (dispersion),
dl_image_p3.json (per-cell student rows), dl_image_p4.json, dl_image_p2.json (analytic +
measured cross-ref).
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "dl_image_matrix.json"


def _load(name):
    p = R / name
    return json.loads(p.read_text())["rows"] if p.exists() else []


def main() -> None:
    rows = []

    for src, scope in (("dl_image_p1.json", "seed42-fullgrid"),
                       ("dl_image_p1_seeds.json", "sweep")):
        for r in _load(src):
            mia = r.get("mia", {})
            rows.append({"track": "image", "paradigm": "P1", "task": "dermamnist_melanoma",
                         "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                         "transport": r.get("transport", "fedprox-0.1"), "metric": "auc",
                         "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                         "aux": {"scope": scope,
                                 "mia_attack_auc": mia.get("attack_auc"),
                                 "mia_tpr_at_fpr1pct": mia.get("tpr_at_fpr1pct"),
                                 "mia_tpr_ratio": mia.get("tpr_ratio_vs_marginal"),
                                 "clip_rate": r["history"][-1].get("clip_rate")}})

    for r in _load("dl_image_p3.json"):
        for c in r["cells"]:
            rows.append({"track": "image", "paradigm": "P3", "task": "dermamnist_melanoma",
                         "epsilon_target": r["epsilon"], "seed": r["seed"],
                         "transport": "consensus-ddg-votes", "metric": "auc",
                         "metric_value": c["auc"], "metric_worst": c["auc_worst"],
                         "aux": {"sensitivity": c["sensitivity"], "sigma_votes": c["sigma_votes"],
                                 "gamma_hat": r["gamma_hat"], "majority_acc": r["majority_acc"],
                                 "eval_clinics_defined": c["eval_clinics_defined"]}})
        rows.append({"track": "image", "paradigm": "P3-clean-ceiling", "task": "dermamnist_melanoma",
                     "epsilon_target": None, "seed": r["seed"],
                     "transport": "consensus-clean", "metric": "auc",
                     "metric_value": r["clean_student"]["auc"],
                     "metric_worst": r["clean_student"]["auc_worst"],
                     "aux": {"gamma_hat": r["gamma_hat"], "majority_acc": r["majority_acc"]}})

    for r in _load("dl_image_p4.json"):
        rows.append({"track": "image", "paradigm": "P4", "task": "dermamnist_melanoma",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": "fedavg-ddg", "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"norm_proofs_ok": r["norm_proofs_ok_all"],
                             "kls_feasible": r["feasible_all"], "scale": r.get("scale")}})

    p2 = R / "dl_image_p2.json"
    if p2.exists():
        d = json.loads(p2.read_text())
        for r in d["analytic_table"]:
            rows.append({"track": "image", "paradigm": "P2-cost", "task": "wire-cost",
                         "epsilon_target": None, "seed": None,
                         "transport": r["protocol"], "metric": None,
                         "metric_value": None, "metric_worst": None,
                         "aux": {k2: v for k2, v in r.items() if k2 != "protocol"}})
        rows.append({"track": "image", "paradigm": "P2-measured", "task": "deployment-probe",
                     "epsilon_target": None, "seed": None, "transport": "secagg_plus",
                     "metric": None, "metric_value": None, "metric_worst": None,
                     "aux": dict(d["measured_row"])})

    rows.append({"track": "image", "paradigm": "P1-user-level", "task": "dermamnist_melanoma",
                 "epsilon_target": None, "seed": None, "transport": None,
                 "metric": None, "metric_value": None, "metric_worst": None,
                 "aux": {"status": "not-runnable",
                         "reason": "HAM10000 patient-key reattachment unrecoverable from the MedMNIST payload (undisclosed row permutation; ordering sweep found no match) - see README section 8"}})

    OUT.write_text(json.dumps({"schema": "track/paradigm/task/epsilon/seed/transport/metric/value/worst/aux",
                               "version": "2026-09-04", "rows": rows}, indent=2))
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
