# Preliminary technical answer to the FLIP-IT legal assessment

**To:** Dr. Nicolas Conze (docport GmbH) / J. Flötotto (Jorzig & Partner)
**Re:** "Central open technical question", preliminary legal assessment of 10 August 2026
**Date:** 13 August 2026
**Status:** Preliminary engineering findings. **Not legal advice** — this document supplies facts for
the legal assessment; it does not draw legal conclusions.

---

## What this system is, and what it is not

`FL-CKD-Flower` is a **pre-kickoff test bed**. Its purpose is to have working, deployable code — an
app that can be pushed to a Flower SuperLink and run — before the funded project begins and before
any real practice data exists.

Stated plainly, so that nothing below is read as more than it is:

- **No live deployment exists.** No SuperLink has been stood up under TLS with node authentication.
  There are no server certificates and no per-practice keypairs in this repository.
- **No real patient data has ever entered this system.** Every figure comes from **synthetic** data:
  10 generated practices, 3,276 generated patients (161–532 each, CKD prevalence 8.3 %–46.0 %).
- **Everything measured was measured in simulation** — either in-process (`ckd-simulate`) or in
  Flower's Ray simulation engine (`flwr run .`), both on one machine.
- Therefore statements below about "the SuperLink" describe **the documented behaviour of the
  software on the code path this project has configured**, not observations of a running host.

This is why the document is titled *preliminary*. It answers what is knowable from the protocol
infrastructure we have actually built and run, and it says so explicitly where a question cannot be
answered until a deployment — and an operator — exists.

Verified against **Flower 1.33.0** as installed in this repository.

---

## The question

> *Does the SuperLink, or Flower Labs, at any point have isolated access to individual practices'
> gradients or other individual contributions — or, as a result of the secure aggregation
> implementation, is only the joint aggregate ever accessible?*

---

## 1. What we can answer now, and what we cannot

| ✅ Answerable from the test bed | ❓ Not answerable until deployment |
|---|---|
| Where in the protocol aggregation happens, relative to the server receiving updates | Who operates the SuperLink host (docport vs. IKIM/KITE) — a decision not yet taken, and the one that drives the Art. 26/28 role allocation |
| Whether an individual update is attributable to a specific practice | What logging, retention and access control are actually configured on that host |
| What each protocol places on the wire, and how large it is | Real cohort sizes per practice — which determine whether any meaningful ε is reachable at all |
| Whether central differential privacy changes what the aggregating party receives | Whether a released model is anonymous — requires membership-inference / inversion testing, on real data |
| Which privacy mechanisms are available on our current code path, and which are not | Whether the practices' own IT environments introduce further disclosure surfaces |
| What the utility cost of central DP is, at fixed clipping, across 5 seeds | The utility cost of **local** DP — mechanism confirmed available, cost not yet measured (§9) |

Everything that follows is drawn from the left-hand column.

---

## 2. The answer, scoped to the test bed

**Under the protocol configuration this test bed runs today, the aggregating party would receive each
practice's individual parameter update in the clear, attributable to that practice.** Secure
aggregation is not part of this code path. Counsel's premise — *"as a result of the secure
aggregation implementation"* — does not yet describe this project.

Two qualifications, and they pull in opposite directions, so both belong here:

**This is a configuration state, not an architectural limit.** Federated learning does not require
the server to see individual contributions; it is simply that the mechanisms which prevent it are
switched off in a research baseline, deliberately, so that reference numbers stay interpretable.
§4 sets out exactly which mechanisms are available to us and what each one changes.

**But nothing in the current implementation supports a claim of anonymity at the aggregation level.**
The conservative treatment the assessment already recommends — treating gradients as personal data,
and potentially as health data — is the technically correct treatment of the system as it stands. We
would not argue against it on the strength of what has been built so far.

On Flower Labs specifically: in the intended self-hosted deployment the company receives nothing and
has no access path (§7).

---

## 3. Why — the two structural facts

Both verified in the installed package rather than inferred from documentation.

**(a) Aggregation is the last step, not a property of transmission.** In Flower's deployment runtime
the SuperLink is the message broker: SuperNodes (the practices) push reply messages to the
SuperLink's Fleet API, and the ServerApp retrieves them. The aggregation function receives a *list of
individual messages*, each carrying that practice's own `ArrayRecord` — the raw parameter vector —
and only then combines them. Nothing about sending an update to the server aggregates it.

**(b) Each message is individually attributable.** Every Flower `Message` carries
`metadata.src_node_id`, and the framework's own aggregation code reads it. A practice's update is
therefore not an anonymous contribution to a pool; it arrives tagged with the identity of the
SuperNode that sent it, which under the authentication scheme in [DEPLOYMENT.md](DEPLOYMENT.md) is
bound by public key to a named practice.

Together: at the moment of aggregation, the aggregating party holds *this named practice's parameter
vector, derived from this named practice's patients.*

---

## 4. Adding noise: what is available to us today

The constructive half of the answer. Three mechanisms bear on the question, and they differ in a way
that matters legally — **only two of them change what the aggregating party receives**.

| Mechanism | Where it acts | Does the server still see the individual update? | Available on our code path? |
|---|---|---|---|
| **Central DP** (`DifferentialPrivacyServerSideFixedClipping`) | Server, *during* aggregation | **Yes — un-noised** | ✅ Yes — measured, [PRIVACY.md §3](PRIVACY.md) |
| **Local DP** (`LocalDpMod`) | **Inside the SuperNode, before transmission** | **No — noised at source** | ✅ **Yes — see below** |
| **SecAgg+** (`SecAggPlusWorkflow`) | Cryptographic masking across practices | No — server obtains only the sum | ⚠️ Legacy path only |

**Central DP does not answer the question posed.** The noise is applied by the server as it
aggregates, so the server necessarily receives every practice's un-noised update first. Central DP
constrains what can be inferred from the *published model*; it does not constrain what the
aggregating party sees. This distinction is easy to lose and it is the single most important
technical point in this document.

### Local DP is available to us now

Verified by inspecting the installed package:

```
flwr.clientapp.mod  ->  LocalDpMod, localdp_mod, fixedclipping_mod,
                        adaptiveclipping_mod, centraldp_mods, ...
```

`LocalDpMod` wraps the practice's ClientApp. It calls the local training step, then **clips and
Gaussian-noises the outgoing parameter record before the message leaves the SuperNode**, with
standard deviation `sensitivity · √(2·ln(1.25/δ)) / ε`. It is parameterised directly by **(ε, δ)**
rather than by a raw noise scale.

The decisive detail is the namespace. `LocalDpMod` sits in `flwr.clientapp.mod` — the current Message
API namespace — so unlike SecAgg+ it composes with `strategy.start()`, which is what this project's
`ServerApp` uses. `client_app.py` currently constructs `app = ClientApp()` with no mods; enabling
local DP is:

```python
app = ClientApp(mods=[LocalDpMod(clipping_norm=..., sensitivity=..., epsilon=..., delta=...)])
```

**So if noise at the practice is what the protocols need in order to comply, that is available to us
on the current code path and is a one-line change.** It is a decision to be taken, not an engineering
obstacle. Its utility cost on a cohort this size is a real question, and it is unmeasured — see §9.

### SecAgg+ — achievable, on a separate path, with limits

Probed against flwr 1.33.0 (`results/privacy.json`): `secaggplus_mod` is **absent** from
`flwr.clientapp.mod` and present only in the legacy `flwr.client.mod`; `SecAggPlusWorkflow.__call__`
requires a `LegacyContext`. It therefore cannot be composed with `strategy.start()`. It **is**
reachable by driving `DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))` with a `LegacyContext`
from inside a modern `ServerApp` — a second, parallel server implementation. Milestone 4 (month 18)
remains deliverable; it needs a deliberate engineering decision, not a later configuration flag.

Three limits that should not be overstated if SecAgg+ is relied on:

1. **The threat model is semi-honest.** The guarantee holds against a server that follows the
   protocol but inspects what it receives. It is not a guarantee against a server that actively
   deviates — for example by running a round with a single practice.
2. **Participation stays visible.** The server always learns which practices took part, and the
   aggregate. Only the individual contribution is hidden.
3. **The aggregate is still model parameters.** SecAgg+ answers "who can see one practice's
   gradient". It does not by itself make the resulting model anonymous.

**SecAgg+ cannot protect the XGBoost path at all.** `FedXgbBagging` transmits serialized decision
trees whose split thresholds are derived from patient values; there is no numeric vector to mask. Any
secure-aggregation deployment is necessarily a logistic-regression deployment. The accuracy results
in [REPORT.md](REPORT.md) independently recommend logistic regression, so this costs nothing.

---

## 5. What each protocol puts on the wire

Measured in the test bed, per practice per round:

| Protocol | Payload | Disclosure surface | Bits/round |
|---|---|---|---:|
| `fedprox` / `fedavg` | model coefficients | A parameter vector fitted to this practice's patients — the object gradient-inversion attacks target | 352 |
| `fedxgb` | serialized decision trees | **Split thresholds are literal patient feature values.** The highest-disclosure payload here | ~20,900 |
| `fedmosaic` | binary predictions + expertise weights on a *public* cohort | Opinions about patients who are already public. No parameter vector exists for the server to invert | 3,600 |

FedMosaic's disclosure profile is qualitatively different in kind, not just degree, and that is why
its source paper motivates the approach partly on privacy grounds. It also specifies its own per-round
DP construction which this test bed does **not** implement — noted as an open item, not a claim.

---

## 6. SuperLink capabilities — documented, not observed

The assessment asks about "administrative access, logging and export capabilities on the SuperLink".
These are flags of the software we would deploy, verified from `flower-superlink --help` and the
`flwr` CLI. **No such host is running, so nothing below is an observation of a live system** — it is
the capability surface an operator would inherit.

| Surface | Mechanism | What an operator would have |
|---|---|---|
| State database | `--database <path>` | Run state persisted to a SQLite file on the host. Omitting it keeps state in memory only, lost on restart |
| Log file | `--log-file`, `--log-rotation-interval-hours`, `--log-rotation-backup-count` | Server logs written to disk with rotation |
| Run log retrieval | `flwr log <run-id>` | Any operator with Control-API access can pull a run's logs |
| Run listing / control | `flwr ls`, `flwr stop` | Enumerate and stop runs |
| Node registry | `flwr supernode list / register / unregister` | The mapping of public keys to admitted practices is centrally held and administrable |

⚠️ **One disclosure channel is of our own making.** This project's dual-level metric logging prints,
every round and *per practice*, its identifier, cohort size, AUROC and sensitivity. For a small
practice that is itself a disclosure to whoever reads the logs — independently of the gradients.
Mitigation is implemented (`--metric-privacy`, `--min-cohort-size`, which anonymise the practice
identity and suppress cohorts below a floor) but is **off by default**, so that baseline research
numbers stay interpretable. It must be **on** in any deployment touching real patients.

---

## 7. Flower Labs specifically

In the intended deployment the **SuperLink is self-hosted** on consortium infrastructure. Flower is
open-source software running on that host. Flower Labs, the company, receives nothing and has no
access path.

⚠️ **One configuration default to close deliberately.** The Flower CLI generates `~/.flwr/config.toml`
containing, by default, an entry pointing at a Flower-operated endpoint:

```toml
[superlink.supergrid]
address = "supergrid.flower.ai"
```

It is inert unless someone explicitly runs `flwr run . supergrid`. But it is one word on a command
line between a self-hosted federation and transmission to a third party. Recommended control: remove
the entry from operator machines, pin the default to the consortium SuperLink, and put it on the
deployment checklist. Cheap to do, easy to forget.

Separately: the Letter of Intent from FlowerAI recorded in the Projektantrag concerns technical
support and expertise. It is **not** a data-processing arrangement and should not be relied on as one.

---

## 8. The four documentation items the assessment lists as missing

| # | Item | What the test bed supplies now | What still needs the real deployment |
|---|---|---|---|
| 1 | **Privacy budget ε/δ and DP configuration** | 🟡 Central DP measured across 5 seeds; the utility curve is established | A proper accountant and real cohort sizes — see below |
| 2 | **Number and size of practices per round** | ✅ 10 practices, 161–532 patients, 3,276 total, prevalence 8.3 %–46 %; `fraction-fit = 1.0`, so every practice participates every round | The funded pilot's 25 practices (MS5) |
| 3 | **Training and aggregation configuration** | ✅ Fully specified in `pyproject.toml` `[tool.flwr.app.config]`: 20 rounds, 2 local epochs, sample-weighted FedAvg/FedProx (μ = 0.1), balanced class weighting, fixed seed, local 80/20 split, local `StandardScaler` never transmitted | — |
| 4 | **Administrative access, logging, export on SuperLink** | ✅ Capability surface documented — §6 | The configuration actually chosen on the operator's host |

**On the privacy budget (1).** Measured at clipping norm 1.0, δ = 1e-5, 20 rounds, 5 seeds
([PRIVACY.md §3](PRIVACY.md)): noise σ from 0.0 to 2.0 costs between 0.001 and 0.055 AUROC — a small
utility cost. The corresponding **ε values range from 48 to 969**, far outside any range ordinarily
considered a meaningful guarantee. Two qualifications, both cutting the same way:

- Those ε figures use the plain Gaussian mechanism with **basic composition** — a deliberately simple,
  auditable **upper bound**, not a certified budget. A proper RDP/PLD accountant will report a
  smaller ε for the same σ. **They should not be quoted externally as the project's budget.**
- Even allowing for that, the underlying constraint is real: ~3,300 patients across 10 practices is
  very little population to hide an individual in. The funded pilot's 25 practices will help; it is
  unlikely to be sufficient alone.

One improvement is available immediately at no accuracy cost: the logistic-regression protocols are
**fully converged by round 10** ([REPORT.md §2](REPORT.md)). Since every additional round composes
additional privacy loss, cutting the round count from 50 to 10 reduces the budget roughly five-fold
for free.

---

## 9. What we are not claiming

The assessment correctly notes that "the technical effectiveness of the protective mechanisms
envisaged has so far not been demonstrated by a documented anonymisation or re-identification test."
That remains true. Specifically **not** done:

- **No membership-inference testbed.** The benefit of DP is argued from ε, not demonstrated against an
  attack. EDPB Opinion 28/2024 names membership inference, model inversion and reconstruction as the
  relevant tests; none has been run here.
- **No local-DP utility measurement.** §4 establishes that `LocalDpMod` is *available* on our code
  path. It does not establish what it would cost. The measured sweep is **central** DP, and the two
  compose differently — local DP noises each practice's update independently, so the noise entering
  the aggregate grows with the number of practices rather than being added once. Local DP will cost
  **more** utility than the central-DP table at matched ε, possibly much more at this cohort size.
  This is the first thing to measure, and until it is measured no local-DP claim should be made.
- **No documented anonymisation proof** for either data path.
- **No penetration or disclosure testing**, and no k-anonymity / minimum-cell analysis.
- **Fairness audited by age band, not sex.** The Projektantrag specifies demographic parity, equal
  opportunity, equalized odds and calibration by group **by sex**; the synthetic schema carries no
  `geschlecht` column, so that audit awaits the real `extract_features.sql` contract.

One further item, verified and worth flagging early because it is cheap to fix and expensive to
discover late: [`extract_features.sql:342`](../extract_features.sql) selects `ep.patientid` into the
per-practice export. Nothing downstream uses it, but the CSV on the practice's disk would carry a
direct identifier, making that file personal data rather than a pseudonymous extract. It should be
dropped, or replaced with a per-practice salted hash, before any real extraction run.

---

## 10. Next steps available on the test bed

Ordered by effect on the legal position per unit of effort. All are actions we can take here, before
any deployment.

| # | Action | Effect |
|---|---|---|
| 1 | **Measure local DP** — enable `LocalDpMod`, sweep (ε, δ), report the utility cost | Turns §4's availability finding into a defensible answer: *this* is the accuracy price of the server never seeing an individual update |
| 2 | **Drop `patientid`** from `extract_features.sql` | Removes a direct identifier from every future extract. One-line change |
| 3 | **Enable `--metric-privacy` / `--min-cohort-size`** as the default outside research runs | Closes the per-practice metric logging channel (§6) |
| 4 | **Reduce rounds to ~10** | ~5× privacy budget reduction at zero accuracy cost |
| 5 | **Pin the deployment path to logistic regression** | Makes SecAgg+ possible at all; independently the better model |
| 6 | **Build the membership-inference testbed** | Converts "DP is configured" into "DP is demonstrated" (T2.5) |
| 7 | **Implement the SecAgg+ legacy-workflow path** and decide it consciously | The change that would let the project answer counsel's question with "only the joint aggregate" |
| 8 | **Commission a proper DP accountant** (RDP/PLD) | Replaces the loose upper bound with a defensible ε |
| 9 | **Remove the `supergrid` entry** from operator config; add to the deployment checklist | Closes the third-party transmission default (§7) |
| 10 | **Raise SecAgg+ Message-API support with Flower** under the existing LoI | May remove the need for a parallel legacy code path entirely |

---

## 11. Summary

1. **This is a test bed.** No live deployment, no real patient data, all figures synthetic across 10
   simulated practices. The findings below describe the protocol infrastructure, not an operating
   system.
2. **On the code path built and tested, the aggregating party would receive individual,
   practice-attributable updates in the clear.** Aggregation is the last step, and every message
   carries its sender's identity.
3. **Central differential privacy does not change this.** It constrains what can be inferred from the
   published model, not what the aggregating party receives. This distinction is the crux.
4. **Noise at the practice is available to us now.** `LocalDpMod` is present on our current Message
   API path, is parameterised by (ε, δ), and noises the update before it leaves the SuperNode — a
   one-line change. Its utility cost is unmeasured and should be measured first.
5. **Secure aggregation is achievable**, subject to a semi-honest threat model, visible
   participation, and a parallel server implementation. It is a Milestone 4 decision best taken early.
6. **Flower Labs has no access** in a self-hosted deployment; one default configuration entry should
   nonetheless be removed as a precaution.
7. **The conservative treatment recommended in the assessment is technically correct** for the system
   as it stands today, and we do not argue against it.

Questions are welcome — particularly on items 1 and 7 of §10, which have the longest lead times.
