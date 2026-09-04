"""P2 measured row — client side: real LSTM local epoch on the node's own ECG clinic,
carried through the built-in `secaggplus_mod` (ten-nodes wire, clipping/quantization
applied by the mod before masking).
"""

import numpy as np
from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from research.privacy_dl_ts.federated import local_train, load_clinics
from research.privacy_dl_ts.models import LSTMClassifier

_CLINICS = None


def _clinic(partition_id: int):
    global _CLINICS
    if _CLINICS is None:
        _CLINICS = load_clinics()
    return _CLINICS[partition_id]


app = ClientApp()


@app.train(mods=[secaggplus_mod])
def train(msg: Message, ctxt: Context) -> Message:
    pid = int(ctxt.node_config["partition-id"])
    X, y = _clinic(pid)
    carry = [a.numpy() for a in msg.content.get("arrays", ArrayRecord()).to_numpy_ndarrays()] \
        if "arrays" in msg.content.array_records else None
    if carry:
        vec = np.asarray(carry[0])
    else:
        import torch
        torch.manual_seed(42)
        vec = LSTMClassifier().param_vector().numpy()
    model = LSTMClassifier()
    import torch
    model.load_param_vector(torch.from_numpy(vec))
    out = local_train(model, X, y, epochs=1, seed=1234 + pid)
    content = RecordDict({
        "arrays": ArrayRecord([out.numpy()]),
        "metrics": MetricRecord({"num-examples": float(len(y))}),
    })
    return Message(content=content, reply_to=msg)
