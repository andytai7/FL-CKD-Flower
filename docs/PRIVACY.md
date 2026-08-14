# Privacy architecture

Evidence for **Milestone 4** (month 18), whose acceptance criterion in the Projektantrag is
*"Implementierung von Secure Aggregation und Differential Privacy"*, and for **T2.3** (Privacy
Enhancing Features) and **T2.5** (Datenschutztests und Bias-Analyse).

Reproduce everything here with:

```bash
uv run ckd-privacy --rounds 20 --seeds 42 43 44 45 46    # -> results/privacy.json
```

---

## 1. Defence in depth

Seven layers, each independently verifiable. "Status" is what is true in this repo **today**, not
what is planned.

| Layer | Mechanism | Protects against | Status |
|---|---|---|---|
| **L0** Minimisation | Drop direct identifiers at extraction; no patient id ever enters a feature frame | Direct re-identification | ⚠️ **Gap — see §2** |
| **L1** Transport | TLS on SuperLink↔SuperNode and SuperLink↔`flwr` CLI | Network interception | ✅ Documented in [DEPLOYMENT.md](DEPLOYMENT.md) |
| **L2** Identity | SuperNode public-key authentication; only registered practice keys admitted | A rogue node joining the federation | ✅ Documented in [DEPLOYMENT.md](DEPLOYMENT.md) |
| **L3** Confidential aggregation | SecAgg+ — the server sees only the sum, never one practice's update | An honest-but-curious SuperLink operator | ⚠️ **Achievable, but only on the legacy path — see §4** |
| **L4** Formal guarantee | Central DP: clipping + calibrated Gaussian noise | Reconstruction / membership inference from the released model | ✅ **Measured — see §3** |
| **L5** Metric hygiene | Suppress or anonymise per-practice metric lines below a cohort floor | Re-identification through the dual-level logs | ✅ Implemented (`--metric-privacy`, `--min-cohort-size`) |
| **L6** Audit | Fairness gaps by group; membership-inference testbed | Undetected bias / leakage (T2.5) | 🟡 Fairness implemented; MIA testbed **not yet built** |

---

## 2. L0 — the identifier gap, stated plainly

[`extract_features.sql:342`](../extract_features.sql) selects `ep.patientid` into the per-practice
export:

```sql
SELECT
    (SELECT val FROM stichtag)  AS t0,
    ep.patientid,                        -- ← direct identifier
    ep.alter_jahre,
```

Nothing downstream uses it — `data/loader.py` selects only `FEATURE_COLS`, so it never reaches a
model. But the **CSV on the practice's disk carries it**, which makes that file personal data under
DSGVO rather than a pseudonymous extract, and drags it into the scope of every access-control and
retention obligation.

**Fix before any real extraction run:** drop the column, or replace it with a per-practice salted
hash if a stable join key is genuinely needed for the T4.2 data collection. This is a one-line SQL
change and a legal-review item for Jorzig & Partner, not an engineering problem.

The FHIR path is already clean: `data/fhir_loader.py` builds rows positionally and never carries a
patient identifier into the frame.

---

## 3. L4 — what Differential Privacy actually costs

Flower's real `DifferentialPrivacyServerSideFixedClipping` wrapped around `FedAvg`, clipping norm
1.0, 10 non-IID practices, 20 rounds, **5 seeds** (each seed redraws both the local splits and the
DP noise). Mean ± std of the final round:

| Noise σ | AUROC | Worst practice | ε upper bound | AUROC cost |
|---:|---|---|---:|---:|
| 0.00 | **0.808 ± 0.008** | 0.675 ± 0.054 | — | — |
| 0.10 | 0.807 ± 0.008 | 0.674 ± 0.057 | 969 | −0.001 |
| 0.25 | 0.807 ± 0.007 | 0.672 ± 0.052 | 388 | −0.001 |
| 0.50 | 0.800 ± 0.010 | 0.670 ± 0.060 | 194 | −0.008 |
| 1.00 | 0.788 ± 0.013 | 0.663 ± 0.027 | 97 | −0.020 |
| 2.00 | 0.753 ± 0.029 | 0.566 ± 0.075 | 48 | −0.055 |

**Read this the right way.** The comfortable reading is "DP is nearly free" — at σ ≤ 0.25 the model
loses 0.001 AUROC. That reading is wrong, and the ε column is why.

> ### ⚠️ The headline risk for MS4
>
> Every ε in that table is **enormous**. A guarantee of ε ≈ 48 is not a privacy guarantee in any
> meaningful sense; the usual target is single digits. The direction of travel is brutal: pushing ε
> down means pushing σ up, and by σ = 2.0 the worst practice has already collapsed from 0.675 to
> 0.566 — worse than several practices achieve training alone.
>
> The cause is cohort size. ~3.5k patients across 10 practices is simply very little data to hide
> in. DP noise is calibrated to the *individual*, so the smaller the cohort, the more each patient
> must be obscured relative to signal.
>
> **This needs to be surfaced to the consortium now, not at month 18.** The realistic levers are:
> more practices (the funded pilot's 25 rather than today's 10), fewer rounds (each composition
> costs budget), a tighter accountant, and accepting a larger ε with SecAgg+ carrying more of the
> load.

### About the ε numbers

`privacy._epsilon()` uses the plain Gaussian mechanism, ε₁ = √(2·ln(1.25/δ))/σ at sensitivity 1
(updates are clipped to the clipping norm), composed over rounds by **basic composition**, δ = 1e-5.

This is deliberately simple and auditable, and it is a **loose upper bound, not a certified
budget**. It ignores subsampling amplification and uses the weakest composition theorem. A real MS4
submission must use a proper accountant (RDP or PLD — `dp-accounting`, `opacus`), which will report
a substantially smaller ε for the same σ. The shape of the utility curve is what this table
establishes; the exact ε is not to be quoted externally.

---

## 4. L3 — SecAgg+ is achievable, but not on the same code path

Probed against the installed `flwr` 1.33.0 (`uv run ckd-privacy` reports this):

| Probe | Result |
|---|---|
| `secaggplus_mod` in `flwr.clientapp.mod` (Message API) | ❌ absent |
| `secaggplus_mod` in `flwr.client.mod` (legacy) | ✅ present |
| `SecAggPlusWorkflow` importable | ✅ present |
| `SecAggPlusWorkflow.__call__` requires `LegacyContext` | ✅ yes — raises `TypeError` otherwise |
| DP mods in `flwr.clientapp.mod` | ✅ `fixedclipping_mod`, `adaptiveclipping_mod`, `LocalDpMod` |

**Conclusion.** SecAgg+ has not been ported to the Message API in 1.33. It cannot be composed with
`strategy.start()`, which is what this project's `ServerApp` uses. It **is** reachable by driving
the legacy workflow from inside a modern `ServerApp`:

```python
from flwr.server import LegacyContext, ServerConfig
from flwr.server.strategy import FedAvg as LegacyFedAvg      # legacy strategy namespace
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

@app.main()
def main(grid: Grid, context: Context) -> None:
    legacy = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=20),
        strategy=LegacyFedAvg(...),
    )
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(num_shares=3, reconstruction_threshold=2)
    )
    workflow(grid, legacy)      # takes the modern Grid
```

with `secaggplus_mod` from `flwr.client.mod` on the `ClientApp`.

So MS4 is deliverable — but it forces a choice the consortium should make consciously:

| Option | SecAgg+ | Message API | Note |
|---|---|---|---|
| **A** Message API (today's code) | ❌ | ✅ | Central DP works; SecAgg+ does not |
| **B** Legacy workflow path | ✅ | ❌ | Deprecated surface; SecAgg+ + DP both available |
| **C** Both, selected by config | ✅ | ✅ | Two server code paths to maintain |

**Recommendation: C**, with the Message API as default and a `--secure-aggregation` flag switching
to the legacy workflow. Also raise it with Flower directly — the Projektantrag records a Letter of
Intent from FlowerAI precisely so the project can ask when SecAgg+ lands on the Message API.

**SecAgg+ cannot protect the XGBoost path at all.** `FedXgbBagging` transmits serialized decision
trees, whose split thresholds are derived from patient values; there is no vector to mask. Any
SecAgg-protected deployment is a logistic-regression (or MLP) deployment. The benchmark's finding
that trees federate poorly here (§ [REPORT.md](REPORT.md)) makes that an easy trade.

---

## 5. What each protocol puts on the wire

| Protocol | Payload | Disclosure surface | Bits/client/round |
|---|---|---|---:|
| `fedprox` / `fedavg` | model coefficients | A parameter vector fit to this practice's patients — the object gradient-inversion attacks target | 352 |
| `fedxgb` | serialized decision trees | **Split thresholds are literal patient feature values.** The highest-disclosure payload here | ~20,900 |
| `fedmosaic` | binary predictions + expertise on a *public* cohort | Opinions about patients who are already public. No parameter vector exists for the server to invert | 3,600 |

FedMosaic's disclosure profile is qualitatively different, and it is why the paper motivates the
approach on privacy grounds. It also carries its own per-round DP construction (XOR mechanism on the
label matrix, Gaussian on the expertise vector) which this repo does **not** yet implement — a clear
next step for T2.3.

---

## 6. Honest gaps

| Gap | Consequence | Owner |
|---|---|---|
| `patientid` in the SQL export | Extract is personal data, not pseudonymous | docport + Jorzig & Partner |
| ε is a loose bound from basic composition | Cannot be quoted as the project's formal budget | IKIM / AG Kamp |
| No membership-inference testbed | L6 audit is incomplete; DP's benefit is argued, not demonstrated | AG Kamp (T2.5) |
| FedMosaic's own DP mechanisms unimplemented | Its privacy claim rests on payload shape alone | this repo (T2.3) |
| Fairness audited by age band, not sex | Antrag specifies sex; `geschlecht` exists only in the real schema | pending real data |
| DP measured on FedAvg only | FedProx/FedMosaic DP cost unmeasured | this repo |
