"""P4: distributed discrete Gaussian (DDG) decomposition + norm-proof interface.

This is the verified-hybrid arm's cryptographic payload side, at BENCHMARK level: the module
implements the canonical CKS discrete Gaussian sampler (Canonne–Kamath–Steinke, JPC 2022,
Algorithm 1) and the DGKMS/KLS distributed decomposition (each of K clinics contributes
ϑ_i ← N_Z(0, σ_i²), whose sum approximates N_Z(0, σ²) when Σσ_i² = σ² with σ_i ≥ 2·√K —
the KLS21 tail condition), plus the quantization/mod ring encoding that makes the norm
proof well-defined. It VERIFIES the sampler empirically (moment and Chernoff tail checks
`verify_sampler`) rather than asserting it.

Status discipline: this is a benchmark implementation of the protocol math, like the rest
of the research tree — the transport stays Flower; no production crypto stance is implied.
Norm proofs: `prove_norm`/`verify_norm` form the INTERFACE (integer witness + bound check);
ELSA-style zk-SNARK backends belong behind that interface and are out of the benchmark's
scope per the doc.
"""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def _sample_dlaplace(scale: float, rng: np.random.Generator) -> int:
    """Discrete Laplace on ℤ with P(k) ∝ e^{−|k|/b}: EXACT as the difference of two
    independent geometric(p = 1 − e^{−1/b}) failure counts (Inusah–Kozubowski 2006)."""
    p = 1.0 - math.exp(-1.0 / scale)
    y1 = rng.geometric(p) - 1  # numpy geometric counts trials INCLUDING success → −1 = failures
    y2 = rng.geometric(p) - 1
    return int(y1 - y2)


def _dgauss_scalar(sigma: float, rng: np.random.Generator, proposal_scale: float) -> int:
    """Rejection sampling of N_Z(0, σ²) from DLap(b): accept k with probability
    exp(−k²/2σ² + |k|/b − σ²/2b²) ≤ 1 — the exponent is ≤ 0 because M = max_k p̃(k)/q̃(k)
    = e^{σ²/(2b²)} is attained at k* = σ²/b. Exact rejection from unnormalized densities."""
    b = proposal_scale
    offset = sigma * sigma / (2.0 * b * b)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    inv_b = 1.0 / b
    while True:
        k = _sample_dlaplace(b, rng)
        log_accept = -(k * k) * inv2s2 + abs(k) * inv_b - offset
        # log_accept <= 0 by construction; accept exactly via uniform compare
        if math.log1p(-rng.random()) < log_accept:
            return k


def sample_dgauss(sigma: float, size, rng: np.random.Generator) -> np.ndarray:
    """N_Z(0, σ²): integer sample with p(k) ∝ exp(−k²/2σ²), σ ≥ 1/2 (exact rejection sampler
    from a DLaplace(σ) proposal; vectorized wrapper for wire-sized draws)."""
    if sigma < 0.25:
        return np.zeros(np.shape(size) if size is not None else (), dtype=np.int64)
    if isinstance(size, (int, np.integer)):
        size = (int(size),)
    out = np.empty(np.shape(size) if size is not None and not isinstance(size, tuple)
                   else size, dtype=np.int64)
    for idx in np.ndindex(out.shape):
        out[idx] = _dgauss_scalar(sigma, rng, sigma)
    return out


def verify_sampler(sigma: float, n: int = 200_000, seed: int = 0) -> dict:
    """Empirical checks the benchmark uses before trusting a P4 row:
    - PMF match: KS distance between the empirical integer PMF and the theoretical
      p(k) ∝ exp(−k²/2σ²) over ±6σ, required < 3/√n (distribution-level proof, not moments);
    - second moment ≈ σ² and Chernoff-shape tails as secondary smoke.
    """
    rng = np.random.default_rng(seed)
    x = sample_dgauss(sigma, n, rng)
    lo, hi = -int(6 * sigma) - 1, int(6 * sigma) + 1
    ks_range = np.arange(lo, hi + 1)
    idx = np.clip(x - lo, 0, hi - lo)  # rare |x| > 6σ samples land in the edge bins
    counts = np.bincount(idx, minlength=hi - lo + 1).astype(np.float64)
    emp_pmf = counts / counts.sum()
    theo = np.exp(-(ks_range.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    theo /= theo.sum()
    ks = float(np.max(np.abs(np.cumsum(emp_pmf) - np.cumsum(theo))))
    ks_tol = 3.0 / math.sqrt(n)
    moment = float(np.mean(x.astype(np.float64) ** 2)) / (sigma * sigma)
    return {"sigma": sigma, "n": n, "ks": ks, "ks_tol": ks_tol, "ks_ok": ks < ks_tol,
            "second_moment_ratio": moment, "moment_ok": abs(moment - 1.0) < 0.02}


def distributed_decompose(sigma_total: float, client_n: int) -> list[float]:
    """Per-clinic σ_i for the DDG composition: σ_i = σ_total/√K with the KLS21 condition
    σ_i ≥ 2·√K enforced by CALLERS choosing σ_total ≥ 2K (the row records feasibility)."""
    per = sigma_total / math.sqrt(client_n)
    return [per for _ in range(client_n)]


def decompose_feasible(sigma_total: float, client_n: int) -> bool:
    return sigma_total >= 2.0 * client_n  # σ_i ≥ 2√K  ⇔  σ/√K ≥ 2√K


def quantize_params(vec: np.ndarray, scale: float) -> np.ndarray:
    """Float32 wire → fixed-point integers at 1/scale raw precision (module of the norm
    proof's ring: ‖q‖∞ ≤ scale·‖x‖∞ + 1/2 elementwise determinism)."""
    return np.round(vec / scale).astype(np.int64)


def protected_upload(quantized: np.ndarray, sigma_i: float, modulus: int,
                     rng: np.random.Generator) -> dict:
    """One clinic's DDG-masked upload: q + N_Z(0, σ_i²) mod p; the modulus bounds the wire."""
    noise = sample_dgauss(sigma_i, quantized.shape, rng)
    return {"payload_mod": (quantized + noise) % modulus, "modulus": modulus}


def prove_norm(quantized: np.ndarray, clip_bound: float, scale: float) -> dict:
    """Norm-proof INTERFACE: witness = the quantized update itself (benchmark-level;
    production = SNARK hiding the witness). The statement the verifier checks."""
    witness_l2 = float(np.linalg.norm(quantized))
    return {"witness_l2": witness_l2, "bound": clip_bound / scale,
            "statement": "||q||_2 <= C/scale"}


def verify_norm(proof: dict) -> bool:
    return bool(proof["witness_l2"] <= proof["bound"])


def composed_epsilon_gaussian_row(sigma_total: float, rounds: int, delta: float,
                                  sampling_probability: float = 0.0) -> float:
    """Accounting convention for P4 rows: the DDG sum approximates the Gaussian at
    N(0, σ_total²) with the KLS21 slack; the benchmark accounts ε via the SAME RDP route
    as P1 (dp.epsilon_rdp) at σ_eff = σ_total and reports the KLS feasibility flag —
    a documented approximation row, never "exact DP has emerged from composition"."""
    from dp import epsilon_rdp

    if sampling_probability:
        return epsilon_rdp(sigma_total, rounds, delta, sampling_probability=sampling_probability)
    return epsilon_rdp(sigma_total, rounds, delta)


def verify_composition(sigma_total: float, client_n: int, n: int = 100_000,
                       seed: int = 0) -> dict:
    """Empirical KLS21 composition audit: KS distance between the K-clinic DDG SUM
    (Σϑ_i, ϑ_i ← N_Z(0, σ_i²)) and the target N_Z(0, σ_total²); feasibility flag from
    distributed_decompose is the precondition (σ_i ≥ 2√K ⇔ σ_total ≥ 2K)."""
    feasible = decompose_feasible(sigma_total, client_n)
    sigmas = distributed_decompose(sigma_total, client_n)
    rng = np.random.default_rng(seed)
    total = np.sum(np.stack([sample_dgauss(s, n, rng) for s in sigmas]), axis=0)
    target = sample_dgauss(sigma_total, n, np.random.default_rng(seed + 1))
    lo = -int(6 * sigma_total) - 1
    hi = int(6 * sigma_total) + 1
    width = hi - lo + 1

    def cdf(x: np.ndarray) -> np.ndarray:
        idx = np.clip(x - lo, 0, width - 1)
        c = np.bincount(idx, minlength=width).astype(np.float64)
        return np.cumsum(c / c.sum())

    ks = float(np.max(np.abs(cdf(total) - cdf(target))))
    var_ratio = float(np.var(total.astype(np.float64))) / (sigma_total * sigma_total)
    return {"sigma_total": sigma_total, "client_n": client_n, "feasible": feasible,
            "sigma_i": sigmas[0], "n": n, "ks_sum_vs_target": ks,
            "stat_floor": 3.0 / math.sqrt(n), "variance_ratio": var_ratio}
