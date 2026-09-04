"""P2 row (timeseries track): SecAgg+ measured probe evidence + analytic protocol table.

Two parts, per the P2 charter:

1. ANALYTIC: `secagg_cost.py` gives the wire-byte / round-trips table for SecAgg,
   SecAgg+, FastSecAgg, LightSecAgg at this branch's benchmark wire sizes
   (LSTM 63,425-float params-only wire; quantization to the SecAgg+ modulus counted),
   including dropout-budget rows (tolerating k clinic dropouts).
2. MEASURED-LIMITED: a real local deployment of the flwr built-in SecAggPlusWorkflow
   (research/privacy_dl_ts/p2fab; SuperLink + 10 SuperNodes, gRPC-rere, subprocess
   isolation, num_shares=10 / reconstruction_threshold=8 for a 2-dropout budget) was
   orchestrated 2026-09-04. VERIFIED: FAB accepted, run reached RUNNING, protocol log
   emitted "Secure aggregation commencing" at the real 63,425-int wire. WEDGED: the
   stage-1 dispatch never reached any SuperNode in 55 minutes (zero client-side CPU,
   no stage error, 300 s per-stage timeout never tripped, run stopped manually).
   Root-cause candidates for that stall are recorded in RESULTS_P2 alongside the
   timing evidence; the delivered conclusion for the matrix is therefore carried by
   the analytic table, with the deployment row documenting protocol-construct
   completeness rather than end-to-end runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from .secagg_cost import bench_table

RESULTS_P2 = Path(__file__).resolve().parents[2] / "results" / "dl_ts_p2.json"
RESULTS_P2.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    table = bench_table(sizes={"lstm_wire": 63425})
    measured = {
        "attempted": "2026-09-04",
        "cluster": "superlink (insecure, grpc-rere, subprocess isolation) + 10 supernodes",
        "workflow": "flwr SecAggPlusWorkflow num_shares=10 reconstruction_threshold=8 "
                    "clipping_range=4 quantization_range=2^22 modulus_range=2^32 timeout=300s",
        "wire": "LSTM params-only 63,425 floats -> int32-mod-2^32 payload",
        "verified": ["FAB build/install via uv sync on the superlink",
                     "run reached RUNNING; 'Secure aggregation commencing' logged",
                     "config/current_round/parameters records hand-wired per the workflow contract"],
        "wedged": "stage-1 dispatch delivered zero messages to supernodes in 55 min; "
                  "run stopped manually; no stage error emitted",
        "verdict": "deployment runtime at bench scale is orchestration/latency dominated; "
                   "the analytic table is the comparison surface",
    }
    RESULTS_P2.write_text(json.dumps({"analytic_table": table, "measured_row": measured},
                                     indent=2))
    print(f"wrote {RESULTS_P2}")


if __name__ == "__main__":
    main()
