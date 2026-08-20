"""Builders for the Flower `Message` objects the in-process runners exchange.

`simulate.py` and the research protocols both drive Flower's real strategies without the Ray
transport, which means they construct the same reply Messages a live `ClientApp` would return.
Those builders live here so there is exactly one definition of the wire shape, and so
`models/protocols/` can use them without importing `simulate` (which imports the protocols).

The shape mirrors `client_app.py` exactly: an `arrays` ArrayRecord plus a `metrics` MetricRecord
carrying `num-examples`, the key Flower's strategies weight by.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MessageType, MetricRecord, RecordDict


def train_reply(arrays: list[np.ndarray], num_examples: int, **extra) -> Message:
    """A train reply Message shaped exactly like a real ClientApp's."""
    content = RecordDict({
        "arrays": ArrayRecord(arrays),
        "metrics": MetricRecord({"num-examples": num_examples, **extra}),
    })
    return Message(content=content, message_type=MessageType.TRAIN, dst_node_id=1)


def _dummy_context() -> Context:
    """The Context a mod is handed. Mods in this project read the message, not the context."""
    return Context(run_id=0, node_id=1, node_config={}, state=RecordDict(), run_config={})


def local_dp_train_reply(
    mod, global_arrays: list[np.ndarray], updated: list[np.ndarray], num_examples: int, **extra
) -> Message:
    """A train reply put through Flower's real `LocalDpMod` before it leaves the practice.

    Local DP is the only placement that changes what the *aggregating party receives*: the update
    is clipped and Gaussian-noised inside the SuperNode, so the server never sees the un-noised
    vector. Measuring its utility cost therefore means applying the real mod, not an imitation of
    it — `LocalDpMod` owns both the clipping (against the broadcast model) and the noise scale
    `sensitivity * sqrt(2 ln(1.25/delta)) / epsilon`, and neither is reimplemented here
    (CLAUDE.md §0 rule 1).

    The mod's contract is `(incoming_msg, context, call_next) -> outgoing_msg`, so the in-process
    runner reconstructs exactly that: an incoming Message carrying the broadcast weights, and a
    `call_next` standing in for the local training that already happened.
    """
    incoming = Message(
        content=RecordDict({"arrays": ArrayRecord(global_arrays)}),
        message_type=MessageType.TRAIN,
        dst_node_id=1,
    )
    reply = train_reply(updated, num_examples, **extra)
    return mod(incoming, _dummy_context(), lambda _msg, _ctx: reply)


def evaluate_reply(metrics: dict, num_examples: int, partition_id: int) -> Message:
    """An evaluate reply carrying this practice's local metrics and cohort size."""
    content = RecordDict({
        "metrics": MetricRecord({
            "num-examples": num_examples,
            "partition-id": float(partition_id),
            **{k: float(v) for k, v in metrics.items()},
        }),
    })
    return Message(content=content, message_type=MessageType.EVALUATE, dst_node_id=1)


def hushed():
    """Silence Flower's per-call INFO logging inside a tight benchmark loop."""
    return contextlib.redirect_stdout(io.StringIO())
