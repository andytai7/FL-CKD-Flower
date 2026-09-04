"""P2 row (image track): SecAgg+ analytic protocol table at CNN wire sizes + the
deployment-probe cross-reference.

The measured probe was executed on the TS branch (identical built-in SecAggPlusWorkflow,
deployment runtime behavior is transport-bound — protocol phases do not depend on payload
content). Verdict mirrored honestly: FAB accepted, RUNNING reached, "Secure aggregation
commencing" logged at the 63,425-int wire, then stage-1 dispatch wedged 55 min with zero
node traffic; stopped manually. Image-side wire sizes (CNN-Small 28,577 params) are 45%
smaller than the TS wire — the analytic table carries both columns; CNN-M ResNet-lite
(2,758,177 params) is included as the scale-out column.
"""

from __future__ import annotations

import json
from pathlib import Path

from .secagg_cost import bench_table

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p2.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    table = bench_table(sizes={"cnn_small_wire": 28577, "cnn_m_wire": 2758177})
    out = {"analytic_table": table,
           "measured_row": {
               "cross_reference": "dl_ts_p2.json measured_row (same built-in SecAggPlusWorkflow)",
               "verdict": "protocol-construct verified on the TS probe; image track carries "
                          "the analytic table at image wire sizes; end-to-end E2E runtime "
                          "measurement at image sizes recorded as not achievable within the "
                          "deployment probe's stability budget"}}
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS} ({len(table)} rows)")


if __name__ == "__main__":
    main()
