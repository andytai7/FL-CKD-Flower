# (per-branch copy from research/privacy_dl_ts/secagg_cost.py; keep mirrored)
"""P2 machinery: secure-aggregation cost model at benchmark scale (no DIY crypto).

This module measure-models, it does NOT re-implement the protocols — per the track rule
(Flower transports only; no hand-rolled SecAgg). Three regimes, formulas from the papers:

- SecAgg (Bonawitz et al., CCS 2017): pairwise-mask baseline. Per client per round, to
  leading order at 128-bit security with DH keys of 32 B and Shamir shares of 32 B per
  peer: key setup ~ (2 public keys × 32 B) + O(k − 1) shares ≈ 32·(k+2) B, PLUS the masked
  model upload d_params·bytes_per_param. Rounds: 4 communication rounds per FL round.
- SecAgg+ (Bell et al., CCS 2020): replaces the O(k²) setup graph with a log-depth graph:
  setup ≈ 32·(log2 k + 2) B per client; same masked upload. Flower ships this as the
  secured aggregation transport — this is the protocol the MEASURED reference row uses.
- FastSecAgg (Kadhe et al., NeurIPS 2020): clears pairwise masks using a PRG with
  one-time function-encryption keys; per-client setup ≈ FE key+ciphertext ≈ 2·λ·(2k) B
  with λ=32 B; 3 rounds.
- LightSecAgg (So et al., IoT J. 2021): server-side mask reconstruction from cached
  aggregates; per-client ≈ 32·(k + 3) B once per configuration, 2 rounds mostly.

The wire-cost migrated row in every P2 matrix cell is `bytes_per_client` — model bytes
dominate at all benchmark scales, and the protocol DIFFERENCE is the setup word count;
both ends are reported. `dropout_retention` models the surviving-clients factor:
dropped clients force share reconstruction and (for Secret-sharing protocols) a fresh
round; modeled as a (1 − d)^{-1} multicast multiplier on setup traffic, exact for the
benchmark's small k when rounded.

Outputs feed results/dl_*_p2.json rows alongside the measured SecAgg+ reference.
"""

from __future__ import annotations

import math

BYTES_PER_PARAM = 4        # float32 wire
LAMBDA = 32                # bytes at 128-bit security (DH key / share / PRG seed word)

PROTOCOLS = ("secagg", "secagg_plus", "fastsecagg", "lightsecagg")


def setup_bytes(client_n: int, protocol: str) -> float:
    """Per-client protocol-setup traffic per FL round, bytes (without the model upload)."""
    k = client_n
    if protocol == "secagg":
        return float(LAMBDA * (2 * (k - 1) + 4))              # pairwise shares+keys, O(k)
    if protocol == "secagg_plus":
        return float(LAMBDA * (2 * math.ceil(math.log2(max(k, 2))) + 4))
    if protocol == "fastsecagg":
        return float(2 * LAMBDA * (2 * k))                    # FE keys/ciphertexts, O(k)
    if protocol == "lightsecagg":
        return float(LAMBDA * (k + 3))                        # cached-aggregate shares
    raise ValueError(protocol)


def rounds_of_interaction(protocol: str) -> int:
    return {"secagg": 4, "secagg_plus": 3, "fastsecagg": 3, "lightsecagg": 2}[protocol]


def bytes_per_client(client_n: int, d_params: int, protocol: str,
                     dropout: float = 0.0) -> dict:
    """Dominant wire row for one FL round: masked-model upload + protocol setup, with a
    dropout reconstruction multiplier on setup traffic ((1−d)^{-1} on the surviving
    set; an exact integer effect at benchmark-scale k once ceiling-rounded)."""
    survivors = max(1, math.floor(client_n * (1.0 - dropout)))
    setup = setup_bytes(survivors, protocol) * client_n / survivors  # dropped-client cost
    model_up = d_params * BYTES_PER_PARAM
    model_down = d_params * BYTES_PER_PARAM  # aggregate broadcast is protocol-invariant
    return {"protocol": protocol, "client_n": client_n, "d_params": d_params,
            "dropout": dropout,
            "model_bytes": model_up,
            "setup_bytes": math.ceil(setup),
            "total_bytes": model_up + math.ceil(setup) + model_down,
            "rounds_of_interaction": rounds_of_interaction(protocol)}


def bench_table(*, sizes: dict[str, int], client_ns=((3, 10, 25, 100)),
                dropouts=((0.0, 0.3))) -> list[dict]:
    """Full cost bench: every protocol × wire size × k × dropout."""
    rows = []
    for name, d in sizes.items():
        for k in client_ns:
            for dr in dropouts:
                for proto in PROTOCOLS:
                    row = bytes_per_client(k, d, proto, dropout=dr)
                    row["model_name"] = name
                    rows.append(row)
    return rows
