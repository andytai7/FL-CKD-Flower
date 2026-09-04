"""P2 measured row — server side: built-in SecAggPlusWorkflow looping 2 measured rounds
around a LEGACY FedAvg strategy (the workflow's stage calls are legacy-strategy API).

num_shares=10, reconstruction_threshold=8 tolerates ≤2 clinic dropouts per round (the
resiliency property the P2 row measures). Wire: LSTM 63,425-float vector quantized to the
workflow modulus. The initial parameters are stashed in the run state for configure_fit.
"""

import time

import torch
from flwr.common import Array, ArrayRecord, Context
from flwr.server import ServerApp
from flwr.server.compat import LegacyContext
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow

from research.privacy_dl_ts.models import LSTMClassifier

ROUNDS = 2

torch.manual_seed(42)
_INIT = LSTMClassifier().param_vector().numpy()

workflow = SecAggPlusWorkflow(
    num_shares=10,
    reconstruction_threshold=8,       # tolerates dropout of ≤2 clinics per round
    clipping_range=4.0,
    quantization_range=4194304,
    modulus_range=4294967296,
    timeout=300.0,
)

app = ServerApp()


@app.main()
def main(grid, context: Context) -> None:
    from flwr.common import ConfigRecord, ndarrays_to_parameters
    from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD
    legacy_strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
                             min_available_clients=8,
                             initial_parameters=ndarrays_to_parameters([_INIT]))
    lctx = LegacyContext(context, config=ServerConfig(num_rounds=ROUNDS),
                         strategy=legacy_strategy)
    from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
    lctx.state.config_records[MAIN_CONFIGS_RECORD] = ConfigRecord({})  # workflow binds here
    lctx.state.array_records[MAIN_PARAMS_RECORD] = ArrayRecord([_INIT])
    from flwr.server.workflow.constant import Key as WorkflowKey
    cfg_rec = lctx.state.config_records[MAIN_CONFIGS_RECORD]
    for rnd in range(1, ROUNDS + 1):
        cfg_rec[WorkflowKey.CURRENT_ROUND] = rnd
        t0 = time.time()
        workflow(grid, lctx)
        print(f"[p2-measured] round {rnd} completed in {time.time() - t0:.1f}s", flush=True)
