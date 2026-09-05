"""Matrix emitter (timeseries track): results/dl_ts_matrix.json.

One row per (paradigm, task, epsilon, seed) with the shared schema used by the image track,
so cross-data-kind comparison reads row-wise:
{track, paradigm, task, epsilon_target, seed, transport, metric, metric_worst, aux}.
Seeds: P1/forecast-P1 rows draw from the Adam-semantics regenerate (`*_adam*` for P1); the
superseded pass-1 files are NOT matrix inputs. P2 rows are analytic (cost-schema rows, not
utility) — flagged metric=None.
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "dl_ts_matrix.json"


def _load(name):
    p = R / name
    return json.loads(p.read_text())["rows"] if p.exists() else []


def main() -> None:
    rows = []

    for r in _load("dl_ts_p1_adam.json"):
        rows.append({"track": "timeseries", "paradigm": "P1", "task": "ecg_afib",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": r["transport"], "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"local": "DP-Adam lr=0.001", "clip_rate": r["history"][-1]["clip_rate"]}})
    for r in _load("dl_ts_p1_adam_seeds.json"):
        rows.append({"track": "timeseries", "paradigm": "P1", "task": "ecg_afib",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": r["transport"], "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"local": "DP-Adam lr=0.001"}})
    for r in _load("dl_ts_p1_adam_transport.json"):
        rows.append({"track": "timeseries", "paradigm": "P1-transport-arm", "task": "ecg_afib",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": r["transport_arm"], "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"local": "DP-Adam lr=0.001"}})

    for r in _load("dl_ts_p3.json"):
        rows.append({"track": "timeseries", "paradigm": "P3", "task": "ecg_afib",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": "consensus-ddg-votes", "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"sensitivity": r["sensitivity_kind"], "sigma_votes": r["vote_sigma"],
                             "gamma_hat": r["gamma_hat"],
                             "majority_acc": r["teacher_majority_acc"],
                             "vote_acc": r["noisy_vote_acc"]}})

    for r in _load("dl_ts_p4.json"):
        rows.append({"track": "timeseries", "paradigm": "P4", "task": "ecg_afib",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": r.get("transport", "fedavg-ddg"), "metric": "auc",
                     "metric_value": r["final_auc"], "metric_worst": r["final_auc_worst"],
                     "aux": {"norm_proofs_ok": r["norm_proofs_ok_all"],
                             "kls_feasible": r["feasible_all"], "scale": r.get("scale")}})

    seen_fc: dict[tuple, dict] = {}
    for r in _load("dl_ts_forecast_grid.json"):
        key = (r["suite"], r["model"], r["horizon"])
        if key not in seen_fc or r["round"] > seen_fc[key]["round"]:
            seen_fc[key] = r
    for r in seen_fc.values():
        rows.append({"track": "timeseries", "paradigm": "model-crosscheck", "task": f"forecast:{r['suite']}",
                     "epsilon_target": None, "seed": r.get("seed", 42),
                     "transport": "fedavg", "metric": "mse",
                     "metric_value": r["mse"], "metric_worst": r.get("mse_worst"),
                     "aux": {"model": r["model"], "horizon": r["horizon"],
                             "acf_corr": r.get("acf_corr")}})

    for r in _load("dl_ts_forecast_p1.json"):
        rows.append({"track": "timeseries", "paradigm": "P1-forecast", "task": f"forecast:{r['suite']}",
                     "epsilon_target": r["target_epsilon"], "seed": 42,
                     "transport": "fedavg", "metric": "mse",
                     "metric_value": r["final_mse"], "metric_worst": None,
                     "aux": {"dp_level": r["level"], "horizon": r["horizon"],
                             "scope": "pilot" if r["level"] == "event" else "full"}})

    for r in _load("dl_ts_forecast_p1_linear.json"):
        rows.append({"track": "timeseries", "paradigm": "P1-forecast", "task": f"forecast:{r['suite']}",
                     "epsilon_target": r["target_epsilon"], "seed": r["seed"],
                     "transport": "fedavg", "metric": "mse",
                     "metric_value": r["final_mse"], "metric_worst": None,
                     "aux": {"dp_level": "user", "horizon": r["horizon"], "scope": "full",
                             "model": "linear", "d_params": r["d_params"],
                             "sigma_sqrt_d": r["sigma_sqrt_d"],
                             "note": "parameter-efficient trajectory-DP feasibility arm"}})

    # P2 analytic cost rows (no utility metric — protocol cost rows, flagged)
    p2 = R / "dl_ts_p2.json"
    if p2.exists():
        d = json.loads(p2.read_text())
        for r in d["analytic_table"]:
            rows.append({"track": "timeseries", "paradigm": "P2-cost", "task": "wire-cost",
                         "epsilon_target": None, "seed": None,
                         "transport": r["protocol"], "metric": None,
                         "metric_value": None, "metric_worst": None,
                         "aux": {k2: v for k2, v in r.items() if k2 != "protocol"}})
        rows.append({"track": "timeseries", "paradigm": "P2-measured", "task": "deployment-probe",
                     "epsilon_target": None, "seed": None, "transport": "secagg_plus",
                     "metric": None, "metric_value": None, "metric_worst": None,
                     "aux": dict(d["measured_row"])})

    OUT.write_text(json.dumps({"schema": "track/paradigm/task/epsilon/seed/transport/metric/value/worst/aux",
                               "version": "2026-09-04", "rows": rows}, indent=2))
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
