# Privacy-preserving federated deep learning on time series — benchmark design

> **Track charter (experiment/timeseries).** This document is the contract for the
> time-series branch of the privacy-preserving-FL benchmark. It lives **on this data-kind
> branch** (methods are categorized per data kind; no methodology sits on `dev`), under a
> **dated, user-directed, track-local relaxation of CLAUDE.md rule 2** (2026-09-04): recurrent
> deep models are permitted in the research tree **here and only here**. The deployable paths
> (`server_app.py`, `client_app.py`) stay logreg. Rules 1 and 8 still bind: all federation runs
> through Flower's real `ServerApp`/`ClientApp` and every protocol below is a `Strategy`
> subclass — no hand-rolled aggregation, no custom transport.
>
> **Picks for this data kind (user-directed, 2026-09-04).** Model: **RNN family** — LSTM for
> classification, GRU for seq2seq forecasting (PatchTST demoted to an optional attention
> ablation). Privacy model for this space: **user-level (per-patient-trajectory) DP with
> Rényi-DP accounting** as the primary guarantee — a patient's trajectory is the privacy unit —
> with **event-level (sliding-window) DP as the measured contrast** (§1.2); both reported side
> by side because they protect different adjacencies. FedCT is the gradient-free alternative,
> SecAgg+ the wire layer, and the verified hybrid the malicious-server arm.
>
> **Transport (aggregator) — measured (2026-09-04).** On the convex logreg proxy over
> `data/clinics_ecg` (309-col profile features; shared-learner protocol bench, 15 rounds),
> FedAvg and FedProx μ=0.1 are **indistinguishable** (r15 AUROC 0.625 / worst 0.581 both;
> max drawdown 0.001 both) — standardized, low-correlation feature frames do not trigger client
> drift, so FedAvg stays the default transport for the proxy runs. Under the RNN arms
> (non-convex local objectives, stronger drift pressure), the matrix keeps **FedAvg as baseline
> and FedProx μ=0.1 as the stabilization arm**; if drift appears (parity oscillation or worst-
> practice regression vs pooled baseline), FedAvgM is the next sanctioned `Strategy`-subclass
> arm. Accounting is transport-agnostic: clip-then-SecAgg-averaged is unchanged under FedProx
> (proximal term is local-objective-only), so P1/P4 sensitivity analysis below holds unchanged.
>
> **Status: design milestone.** The four paradigms are specified below with modular class
> blueprints, exact sensitivity formulations, temporal-handling choices, and the evaluation
> framework. Implementation lands per the work plan in §7; every claim about an external
> protocol is cited in §8.

---

## 0. Problem position

Federated deep learning over multivariate time series suffers two failure modes the tabular CKD
line never met:

1. **DP utility collapse in high-dimensional, non-convex landscapes.** Per-example clipping to
   norm $C$ injects $\mathcal N(0, \sigma^2C^2 I_p)$ into *all* $p$ coordinates while useful
   signal concentrates in a thin subspace — noise-to-signal ratio $\approx\sqrt p$ — and
   non-convex training needs far more composed steps than the 10-round convex logreg baseline.
2. **User-level sensitivity vs. sequential dependency.** A patient's contribution is a
   *trajectory*, not an i.i.d. record. Event-level clipping (per window) gives small sensitivity
   but severs long-range autocorrelation at window boundaries; user-level clipping (per
   trajectory) preserves the full sequence but makes the sensitivity of *one* patient's update
   large and forces either huge noise or participation caps.

The benchmark contrasts four paradigms that each relocate this tension: (1) gradient-space DP with
explicit event- vs user-level accounting, (2) scaling the aggregation primitive instead of
shrinking the model, (3) leaving gradient space entirely (FedCT consensus in prediction space,
Kamp lab), (4) combining SecAgg with *verified* local DP so the server can enforce the clipping
contract it cannot see.

### How the four paradigms categorize *for time series* (this branch's stance)

| Paradigm | Category for sequences | Why, for this data kind |
|---|---|---|
| **P1 gradient-space DP (RDP-accounted)** | **Primary privacy model** | Clinical time series are *trajectories*: the honest unit of protection is the patient (user level), and the event-level contrast is exactly the "sever autocorrelation vs over-count sensitivity" knob this data kind forces. RDP composition handles the long composed training traces non-convex RNNs need |
| **P2 SecAgg at scale** | Security layer + cost profile | ECG-like streams are the dropout-prone setting SecAgg variants were built for (wearables, intermittent uplinks); CinC-ECG's 8–10 pseudo-clinics are big-N/reliable, so dropout arms are a stress test, and RNN wire sizes (0.2–0.8 MB) are the cost-profile subject |
| **P3 FedCT consensus** | Gradient-free alternative | Removes gradient exposure entirely — important because sequences invert poorly without it: a released RNN gradient *trajectory* leaks autocorrelation structure. Needs a public consensus cohort (held-out CinC reference pool) |
| **P4 verified hybrid (SecAgg + DDG + norm proofs)** | Malicious-server assurance arm | Malicious-server traps on recurrent losses are the nastiest attack here (flat temporal error surfaces); distributed discrete Gaussian noise + proved clipping contracts is the only row that covers it |

### Threat models (fixed, per paradigm)

| Paradigm | Server | Clients | Admissible adversary |
|---|---|---|---|
| P1 gradient DP | honest-but-curious | honest (privacy constraint on what any observer can learn from released model) | external analyst of released model; no malicious client assumption |
| P2 SecAgg at scale | honest-but-curious | honest, **dropout-prone** | network adversary + curious server; availability threats from streams |
| P3 FedCT | honest-but-curious | honest | curious server sees only hard-label votes on a *public* cohort |
| P4 hybrid | **malicious-capable** server possible | **malicious-capable** clients (bounded fraction $<1/3$) | hidden malicious updates under masking (norm unverifiability) |

### Benchmark datasets (public, no credentials)

| Axis | Dataset | Task | Why |
|---|---|---|---|
| Classification | **CinC 2017 ECG** (in-repo, `data/clinics_ecg/`) | AF vs rest (binary; 4-class later) | primary-care modality, per-recording units, already partitioned |
| Classification (stress) | CinC 2017 4-class (N/A/O/~) | rhythm classification | tests temporal confusion of noised grads |
| Forecasting | **ETTh1 / ETTm1** (Informer release) | 96/192/336-step oil-temperature forecasting | the canonical recurrent-forecasting testbed (GRU spine, LSTM contrast); univariate target, multivariate covariates |
| Forecasting (stress) | **Weather** (Autoformer release, 21 channels) | multivariate forecasting | seasonality at multiple scales — the autocorrelation-preservation test |

MIMIC-IV credentialed access is deliberately excluded. Synthea longitudinal trajectories are a
later optional derivative (mapper work, not benchmark input).

### Model suite (shared interface) — RNN family is the benchmark; PatchTST is an optional ablation

One factory, RNN-first, all exposing `(params: StateDict) -> torch.Tensor` flattening so the
privacy machinery operates on a single vector (the design mirrors how `models/logreg.py` flattens
`coef_`/`intercept_`):

```python
# research/privacy_dl_ts/models.py (blueprint — lands in §7.1)
class TSModelFactory(Protocol):
    def build(self, cfg: ModelCfg) -> nn.Module: ...   # forward(x: (B, L, C)) -> (B, H) | (B,)

class LSTMClassifier(nn.Module):    # THE classification arm: hidden 64, 2 layers,
                                    # mean-pool over time -> logit
class GRUForecaster(nn.Module):     # THE forecasting arm: seq2seq decoder, horizon H
class PatchTST(nn.Module):          # OPTIONAL attention ablation only (not the benchmark model):
                                    # Nie et al. 2023: patch_len 16, stride 8, 3 layers,
                                    # d_model 128, channel-independent; ~1.3M params (small cfg)
```

Wire sizes used for all communication-cost calculations below (fp32):

| Model | params $p$ | update size $\approx 4p$ bytes |
|---|---|---|
| LSTM-Clf (64×2) | $\sim$ 50 k | 0.2 MB |
| GRU-Forecaster | $\sim$ 0.2 M | 0.8 MB |
| PatchTST-small | $\sim$ 1.3 M | 5.2 MB |

---

## 1. Paradigm 1 — Gradient-space DP with tight accounting

### 1.1 Architectural blueprint

Local DP-SGD at the client; the per-round release is one clipped, noised trajectory gradient.
Flower integration: a `Strategy` subclass (rule 8) that receives already-noised arrays — the
clipping/noise lives entirely client-side in the `ClientApp` train step.

```python
# research/privacy_dl_ts/p1_gradient_dp.py (blueprint)
@dataclass
class ClipSpec:
    level: Literal["event", "user"]   # per-window vs per-trajectory
    norm: float                        # C_e or C_u
    epochs_per_round: int

class TsDpClient:
    """Local step: gradient accumulation over the client's trajectory store."""
    def local_update(self, model, clip: ClipSpec) -> torch.Tensor:
        flat_grads = []
        if clip.level == "event":
            for window in self.sliding_windows(stride=1):        # L-window events
                g = grad(loss(model, window), model.parameters())
                flat_grads.append(clip_(flatten(g), clip.norm))  # C_e per window
            per_round = torch.stack(flat_grads).mean(0)          # window-mean update
        else:  # user-level: retain full-trajectory gradient path
            for traj_id, traj in self.trajectories.items():
                g_seq = 0
                for window in traj.windows():                     # BPTT unfolds full history
                    g_seq = g_seq + flatten(grad(loss(model, window), ...))
                flat_grads.append(clip_(g_seq, clip.norm))       # C_u per patient
            per_round = torch.stack(flat_grads).sum(0)
        noise = torch.normal(0, clip.sigma * clip.norm, size=per_round.shape)
        return per_round + noise                                  # the only wire payload

class DpTsStrategy(FedAvg):   # strategy subclass (rule 8); aggregation is Flower's own
    pass                      # accounting + clip dispatch live in run config
```

### 1.2 Sensitivity and privacy analysis

- **Event level** (protect one length-$L$ window). Adjacency: replace one window in one patient's
  record. Update $=\frac1{K}\sum_i \mathrm{clip}(\nabla_i; C_e)$ over $K$ windows $\Rightarrow$
  per-round sensitivity $\Delta = 2C_e/K$ (replace) or $C_e/(K+1)$ under add/remove with the
  unnormalized form. Each round is a Gaussian mech with std $\sigma C_e/K$; released model
  composes over $R$ rounds. RDP accountant: `dp-accounting`'s
  `RdpAccountant` composed over self-loops — exactly the machinery in `dp.py`
  (`sigma_for_epsilon` inverts budget → noise multiplier).
- **User level** (protect a whole trajectory). Adjacency: replace one patient's full record.
  $G_p=\mathrm{clip}\!\big(\sum_{i\in p}\nabla_i;\, C_u\big)$ bounds the patient's total
  influence at $C_u$ *independent of trajectory length* — the property event-level DP cannot
  give. Participation-capped FedAvg yields per-round $\Delta=2C_u/N_{\text{part}}$; Poisson
  client sampling gives amplification; compose over rounds as in McMahan et al. 2018
  (user-level DP for LSTM language models — the canonical antecedent).
- **Accounting:** RDP for both; report the event-level/user-level pair at the same query
  semantics — they protect different adjacencies, so the $\varepsilon$ columns in the matrix are
  **not directly comparable**; the benchmark reports both explicitly.

### 1.3 Temporal dynamics handling

- Event-level units are windows: to keep long-range structure usable, windows **overlap**
  (stride $L/4$) and the LSTM warms its hidden state from the preceding context window without
  back-propagating through it (truncated BPTT with warm start) — autocorrelation beyond $L$ is
  available to the *encoder state* though not to the *loss graph*.
- User level keeps full-sequence gradient paths; PatchTST's natural "event" is the **patch**
  (patch-len 16), so event-level DP at patch granularity is sub-sequence DP without window
  hard-occlusion; channel independence keeps cross-channel structure intact.
- Measured, not asserted: the matrix includes an autocorrelation-preservation metric — mean
  $\ell_2$ distance between the ACF of residuals on non-private vs private models at lags up to
  $4L$.

### 1.4 Trade-off contribution

Compute/client-round: $\mathcal O(B\cdot p)$ forward+backward, $\times$ clipping overhead
(~1.3×); uplink $4p$ bytes (no extra terms); server cost unchanged; $(\varepsilon,\delta)$ from
RDP; utility = AUROC/F1 (AF) and MSE/MAE (ETT) per $\varepsilon\in\{0.5,1,2,4,8\}$, both
adjacency levels.

---

## 2. Paradigm 2 — Efficient SecAgg at scale

Flower's SecAgg+ (the deployed L3) masks pairwise with per-client setup **quadratic-ish in client
count on the wire for key agreement** and, at $p=5\,$M, a $20\,$MB masked payload per client per
round. The question for deep TS models: at what $(n, p)$ does the *masking protocol* dominate
compute? Answer with the two scalable alternates — benchmarked as **overhead models**, not
production crypto (scope decision: we do not re-implement audited cryptography; we instrument
their cost profiles and keep Flower's real SecAgg+ as the executed baseline).

### 2.1 Architectural blueprint

**FastSecAgg** (Kadhe et al. 2020, arXiv:2009.11248): pairwise mask sharing replaced by a single
*multi-secret sharing* (FastShare) — clients share one master mask-seed via an FFT-based
packed-scheme (Reed–Solomon sharing evaluated/interpolated through FFT), cutting the per-client
exchange from $\mathcal O(n)$ pairings to $\mathcal O(n\log n)$-amortized shares; dropout
recovery via threshold reconstruction.

**LightSecAgg** (So et al., MLSys 2022, arXiv:2109.14236): instead of reconstructing *dropped
clients'* random seeds (whose cost grows with dropouts), the server reconstructs the **aggregate
mask of the active clients in one shot**: each client uploads one encoded composite mask-share;
a threshold $t$ of surviving clients suffices regardless of which clients dropped — the property
that matters for dropout-heavy sensor streams.

```python
# research/privacy_dl_ts/p2_secagg_scale.py (blueprint — cost model, not crypto)
@dataclass
class SecAggProfile:
    name: Literal["secaggplus", "fastsecagg", "lightsecagg"]
    def uplink_bytes(self, n: int, p: int, drop: float) -> int: ...
    def downlink_bytes(self, n: int, p: int, drop: float) -> int: ...
    def crypto_ops(self, n: int, p: int, drop: float) -> dict: ...   # FFT shares vs pairings
    def survives(self, n: int, t: int, drop: float, trials: int, seed: int) -> float: ...
        # Monte-Carlo dropout resilience: P(round completes) under per-client drop prob

class SecAggScaleBench:
    """Sweeps (n, p, drop) over the three profiles; reports MB/round and completion rate.
    Flower's real SecAgg+ (the L3 standard) provides the measured reference point; FastSecAgg/
    LightSecAgg numbers are computed from the papers' complexity formulae and marked modeled."""
```

### 2.2 Sensitivity and privacy analysis

SecAgg contributes **no DP guarantee** — it is the L3 confidential-aggregation layer. The privacy
semantics: the server observes only $\sum_i x_i$ mod $q$ (FastShare threshold $t$; LightSecAgg
aggregate-mask threshold $t$). Utility cost of the crypto itself is zero below the numerical
quantization noise of modular embedding (uint32/uint64 mapping of fp32 updates — the same
quantization our SecAgg+ path already pays).

### 2.3 Temporal dynamics handling

The content of the vector is irrelevant to SecAgg; the *streaming profile* is the temporal
surface: wearable/ECG-style clients straggle and drop mid-round, **truncating sequential
coverage** (a dropped ward leaves that horizon unlearned this round). Modeling: per-client drop
probability $d\in\{0, 0.1, 0.2, 0.4\}$; LightSecAgg's one-shot property tolerates churn without
re-masking rounds — report *rounds completed per wall-clock* as the temporal-coverage metric,
plus a staleness discount $\gamma^{\Delta t}$ on late windows in the aggregation weight.

### 2.4 Trade-off contribution

Columns: modeled uplink/downlink MB at $(n,p,d)$ grid $\{10,25,100\}\times\{0.05M,0.2M,1.3M\}
\times\{0,0.1,0.2,0.4\}$; crypto-op complexity class; dropout completion rate (Monte-Carlo,
5 seeds); the measured SecAgg+ reference row.

---

## 3. Paradigm 3 — Prediction-space consensus (FedCT paradigm)

**Ground truth:** FedCT = *"Little Is Enough: Boosting Privacy by Sharing Only Hard Labels in
Federated Learning"* (AAAI 2025; Kamp lab, RUB — Michael Kamp is the FLIP-IT consortium partner;
arXiv:2310.05696; reference implementation `github.com/kampmichael/federatedcotraining`).
Clients never share gradients: each trains a local deep model, predicts **hard labels** on a
shared, *unlabeled public* reference set, the server takes **majority-vote consensus**, clients
retrain on consensus pseudo-labels + private data. The repo already runs a cousin of this idea —
FedMosaic's public-cohort prediction exchange (`models/protocols/fedmosaic.py`).

### 3.1 Architectural blueprint

```python
# research/privacy_dl_ts/p3_fedct.py (blueprint)
class FedCtClient:
    def local_step(self) -> "np.ndarray (M,) uint8":
        self.model.fit(self.private_windows, self.current_pseudo_labels)
        return self.model.predict_hard(self.public_reference)      # M hard labels only

class FedCtConsensus(Strategy):                                    # rule 8 subclass
    def aggregate_train(self, rnd, replies):
        votes = np.stack([vote_vector(r) for r in replies])        # (n, M) labels
        noisy = self.add_discrete_noise(votes)                     # P3-DP, below
        pseudo = majority_vote_per_record(noisy)                   # (M,)
        return pseudo                                              # broadcast; no gradients
```

Forecasting variant: the continuous horizon is **discretized** into quantile bins fitted on the
*public* cohort (bin edges broadcast once, static) so majority vote is defined per horizon step;
the DP-free variant uses coordinate-wise median over client predictions (robust, but sensitivity
analysis differs — median has unbounded local sensitivity on unbounded data, hence the binning).

### 3.2 Sensitivity and privacy analysis

- Raw FedCT's privacy claim is *structural* (hard labels leak far less than gradients — the
  paper's label-only MIA evaluation). The benchmark adds a **formal layer**: release counts as
  the mechanism. Per record $j$, the count vector over classes changes by $\le 1$ on replacing
  one client's vote $\Rightarrow$ per-record $\ell_2$ sensitivity $\sqrt{2}$; over all $M$
  released records with participation adjacency, $\Delta_2=\sqrt{2M}$.
- **The stability refinement the user named:** FedCT's local models are trained with a stable
  learner; under on-average-leave-one-out stability with parameter $\gamma$ (expected fraction
  of votes that flip when one client's data leaves), the *effective* released-dimension count
  shrinks from $M$ to $\approx\gamma M$ — sensitivity $\Delta_2^{(\mathrm{stab})}
  =\sqrt{2\gamma M}$ in expectation. Noise scale is then the discrete Gaussian (or binomial)
  reaching target $(\varepsilon,\delta)$ under RDP composition over rounds on
  $\Delta_2^{(\mathrm{stab})}$. This is the **research-risk item** of the paradigm: on-average
  stability gives an *expected* bound, so the released guarantee needs either a high-probability
  lift of the stability parameter or a posterior-hybrid (data-dependent PATE-style) analysis;
  both are called out in the work plan as the analysis tasks, with the conservative
  $\sqrt{2M}$ bound as the baseline row in every result table.
- Note the decisive win: **no gradient clipping at all** — sensitivity is attached to $M$
  discrete outputs, not to $p$ continuous parameters; the $\sqrt p$ pathology disappears.

### 3.3 Temporal dynamics handling

The local deep models see raw, unwindowed-forward-limit sequences — the richest temporal
treatment of any paradigm, since nothing is clipped per-window. Temporal fidelity risk shifts
elsewhere: the **public reference cohort must cover the seasonal regimes** of the private data,
or consensus labels collapse to majority-season behavior. Handling: reference sets stratified by
calendar period and horizon step; per-bin consensus for forecasts keeps phase information
(votes are per horizon step, never pooled across time).

### 3.4 Trade-off contribution

Uplink: $M$ labels/round/client = $M$ bytes (uint8) + bin indices — **~KB scale at any model
size** (vs MB-scale gradient payloads); client compute unchanged (local training) + inference on
$M$; server compute $\mathcal O(nM)$; $(\varepsilon,\delta)$ from the vote-noise analysis
(baseline + stability-refined rows); utility AUROC/F1 (AF) and MSE/MAE vs the residual-Acf
preservation metric.

---

## 4. Paradigm 4 — Hybrid SecAgg + verified local DP

The hidden-malicious-update problem: under SecAgg the server cannot verify the clipping contract
$\|\Delta_i\|\le C$, and a malicious client can silently inflate its update (norm attack) or
shift the noise budget (adding calibrated noise is what *makes* the aggregate DP — a client that
skips noise degrades privacy without detection).

### 4.1 Architectural blueprint

```python
# research/privacy_dl_ts/p4_hybrid_secagg_dp.py (blueprint)
class VerifiedLocalNoiseClient:
    def local_update(self, clip_norm: float, sigma_c: float, n_clients: int):
        g = clip_(self.flat_grad(), clip_norm)                      # contract term 1
        xi = sample_distributed_discrete_gaussian(sigma_c**2 / n_clients)  # contract term 2
        return mask_for_secagg(g + xi), commit(g)                  # masked payload + proof data

class VerifiedSecAggStrategy(FedAvg):              # rule 8 subclass
    # 1) clients prove ||update|| <= C to the *aggregator's threshold committee* via a
    #    lightweight ZK norm proof (ELSA-style two-server variant when available);
    # 2) masked sum unmasks to sum_i g_i + sum_i xi_i — post-unmask total noise
    #    ~ DiscreteGaussian(0, sigma_c^2) EXACTLY, i.e. central-DP strength without a
    #    trusted server (Kairouz, Liu, Steinke ICML 2021, arXiv:2102.06387; Skellam alt:
    #    Agarwal et al. 2021);
    # 3) any client failing the norm proof is excluded and the round recomputes threshold,
    #    LightSecAgg-style (P2 reuse).
```

### 4.2 Sensitivity and privacy analysis

Clipping is trajectory-level (P1 user-level) with norm $C_u$; the norm **proof** makes the
sensitivity claim enforceable rather than presumed. Per-client noise
$\mathrm{DDG}(\sigma^2/n)$ sums to the exact discrete Gaussian $\mathrm{DDG}(\sigma^2)$
independent of client honesty, given conditional-RDP accounting per coordinate (Kairouz et al.
Thm. for sums of DDG under modular wraparound; Skellam as the communication-cheaper alternate
with composition penalties analyzed). Result: **the utility of central DP at the L3 trust
level** — no server sees anything but the noised sum, and no single client can silently weaken
the noise.

### 4.3 Temporal dynamics handling

Identical trajectory-level treatment to P1-user-level (full-sequence gradients, warm-start
states); the only temporal-specific addition is round-level **window freshness binding** in the
norm proof: the committed update hashes the covered window indices, preventing replay of stale
updates across horizons (a malpractice specific to streaming settings).

### 4.4 Trade-off contribution

Compute: + one ZK norm proof per client-round (ELSA-class: sub linear in $p$ verification,
$\mathcal O(p)$ proof); uplink $4p$ + proof overhead (KB); DP = central-grade $(\varepsilon,\delta)$
under RDP; robustness: excludes bounded malicious fraction assuming threshold committees /
two non-colluding servers where the protocol requires it; utility targets the non-private P1
user-level curve (the "free-lunch test": how close does verified-hybrid land to plain DP-SGD at
equal ε).

---

## 5. Empirical trade-off matrix (evaluation framework to build, §7.3)

Single harness, one CLI (`research/privacy_dl_ts/bench.py` → `results/dl_ts_matrix.json`),
Flower strategies only, seeds 42–46, dual-level metrics per CLAUDE.md rule 5 carried over.

| Paradigm | Server trust | Client trust | Uplink/round (n=25) | Extra compute/client | Guarantee (accountant) | Utility (AF AUROC / ETT MSE) | Dropout resilience | Temporal fidelity (ACF ℓ2) |
|---|---|---|---|---|---|---|---|---|
| P1 event-level DP-SGD | HBC | honest | $4p$ MB | clip+noise ~1.3× | RDP, event adjacency | sweep ε∈{0.5,1,2,4,8} | vanilla FedAvg | measured |
| P1 user-level DP-SGD | HBC | honest | $4p$ MB | clip+noise ~1.3× | RDP, user adjacency + Poisson sampling | same grid | participation cap | measured |
| P2 SecAgg+ / Fast / Light | HBC+network | honest, droppy | protocol bytes (§2.4 grid) | masking ops | none (L3-only) | identical to non-private reference row by construction | protocol completion P | n/a (content-agnostic) |
| P3 FedCT + noisy vote | HBC | honest | ≈$M$ KB | +inference on public M | vote-count DDG, baseline $\sqrt{2M}$ and stability-refined | same grid | consensus tolerates churn | reference-cohort coverage metric |
| P4 verified-hybrid | malicious-capable | bounded-malicious | $4p$ + proof KB | + noise gen + proof | conditional RDP, central-grade | parity target: P1-user row | norm-proof exclusion | freshness-bound |

Dropout dimension: per-client drop prob ∈ {0, 0.1, 0.2, 0.4}; temporal-coverage and
staleness-discount columns populated for P2/P4. Every utility cell also logs the **worst
practice** value (rule 5 analog: a secure average that kills one clinic is a failure).

## 6. What this benchmark will claim (and will not)

It claims comparative, measured statements about the four paradigms on shared data, seeds,
models, accounting code (`dp.py`'s RDP path reused) — at modest model scales that fit a CPU/GPU
workstation. It does **not** claim a production medical system, credential-only dataset results
(MIMIC), or implemented production cryptography for FastSecAgg/LightSecAgg (modeled overhead,
cited formulas, measured SecAgg+ reference only).

## 7. Work plan (ordered, one mergeable unit each)

1. `models.py` + torch optional extra (`uv add --optional dl torch`), unit-shape smoke tests.
2. P1 event/user DP-SGD on CinC-ECG (LSTM) — the ε-grid; **this is the spine**, everything references it.
3. Benchmark harness + matrix emitter (the §5 columns).
4. P3 FedCT on AF + ETT (Kamp-lab reference impl cross-check on public cohort splitting).
5. P2 SecAgg-scale cost bench (+ measured SecAgg+ reference row).
6. P4 verified-hybrid: DDG noise decomposition first (pure math, cheap), norm-proof interface behind it.
7. Forecasting arms (PatchTST vs GRU on ETT/Weather) once 1–4 hold on classification.

## 8. References

- FedCT — *Little Is Enough: Boosting Privacy by Sharing Only Hard Labels in Federated
  Learning*, AAAI 2025 (Kamp lab, RUB); arXiv:2310.05696; impl: `github.com/kampmichael/federatedcotraining`.
- FastSecAgg — Kadhe et al. 2020, arXiv:2009.11248 (FastShare: FFT-based multi-secret sharing).
- LightSecAgg — So et al., MLSys 2022, arXiv:2109.14236 (one-shot aggregate-mask reconstruction).
- Distributed discrete Gaussian — Kairouz, Liu, Steinke, ICML 2021, arXiv:2102.06387; Skellam
  alternate — Agarwal, Suresh, Yu, Kumar, McMahan 2021 (`google-research/federated/distributed_dp`).
- User-level DP FL for recurrent models — McMahan et al. 2018 (*Learning Differentially Private
  Recurrent Language Models*).
- On-average loss stability & DP lineage — Bassily, Feldman et al. (private ERM via stability;
  lifted here to vote-release sensitivity — flagged as the analysis risk item in §3.2).
- PatchTST — Nie et al., ICLR 2023, arXiv:2211.14730. Norm-verified secure aggregation — ELSA,
  Rathee et al., CCS 2023.

### 2.2 Measured model-suite + transport protocol (2026-09-04)

- **Model suite** (`research/privacy_dl_ts/models.py`, torch-optional `dl` extra):
  LSTM-Clf 63,425 params (conv stem stride-8 -> L≈128 RNN steps); GRU-Fcst 39,041
  (zero-init scalar head, step-ahead recurrence); PatchTST ablation probe 432,385.
  ParamVectorMixin round-trip contract checked at import.
- **Sanity gate verdict** (target: reproduce the logreg band 0.625 before privacy spend):
  v1 mean-pool LSTM FAILED (AUROC ~0.53 through r7); mean+max + epochs FAILED (~0.51);
  conv stem fixed feature dilution; **FedAvgM β=0.6 transport PASSED: r5-r10 0.668 -> 0.716,
  worst clinic 0.53-0.61**. Comparator rows: FedAvg same-config 0.601 final (peak 0.646 @ r5,
  mid-run drift), FedProx μ=0.1 0.525 (over-suppressed on these gradient magnitudes).
  **Transport decision: FedAvgM default on the TS track; FedAvg/FedProx stay comparator rows.**
- **P1 machinery**: `dpsgd_ts.py` honest per-record clip+Gaussian Poisson DP-SGD
  (clip-rate audit, FedProx-compatible proximal term) + `p1_spine.py` ε-grid {off,0.5,1,2,4,8}
  runner with per-clinic standardised plans (σ_k so every clinic composes the SAME target ε);
  unit-checked budget direction.
