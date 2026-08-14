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
from flwr.app import ArrayRecord, Message, MessageType, MetricRecord, RecordDict


def train_reply(arrays: list[np.ndarray], num_examples: int, **extra) -> Message:
    """A train reply Message shaped exactly like a real ClientApp's."""
    content = RecordDict({
        "arrays": ArrayRecord(arrays),
        "metrics": MetricRecord({"num-examples": num_examples, **extra}),
    })
    return Message(content=content, message_type=MessageType.TRAIN, dst_node_id=1)


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
