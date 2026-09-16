# Deploying to Flower SuperLink / SuperNode

How to get this app off a laptop and onto the real federation: one **SuperLink** on central
infrastructure, one **SuperNode** per GP practice, TLS everywhere, and only registered practice keys
admitted.

Every flag below was verified against the installed **flwr 1.33.0** (`flower-superlink --help`,
`flower-supernode --help`, `flwr supernode --help`). Where 1.33 deprecated an older flag, the
current one is used and the deprecation noted.

```
                   ┌──────────────────────────── central (docport / IKIM KITE) ────┐
   flwr run . ───► │  SuperLink                                                    │
   (your laptop)   │    Control API   :9093   ← the flwr CLI talks here            │
                   │    Fleet API     :9092   ← SuperNodes talk here               │
                   │    ServerApp     runs the strategy                            │
                   └───────────────────────────────────────────────────────────────┘
                            ▲                ▲                      ▲
                        TLS │            TLS │                  TLS │
                   ┌────────┴───┐   ┌────────┴───┐         ┌────────┴───┐
                   │ SuperNode  │   │ SuperNode  │   ...   │ SuperNode  │
                   │ Praxis 01  │   │ Praxis 02  │         │ Praxis 25  │
                   │  ClientApp │   │  ClientApp │         │  ClientApp │
                   │  FHIR db   │   │  FHIR db   │         │  FHIR db   │
                   └────────────┘   └────────────┘         └────────────┘
                    patient rows never leave this box
```

---

## Step 0 — Prerequisites

On the central host and on every practice machine:

```bash
uv sync --extra dev            # installs flwr 1.33 + the app's dependencies
uv run python -c "import flwr; print(flwr.__version__)"   # expect 1.33.x
```

The app is shipped as a **FAB** (Flower App Bundle) built from `pyproject.toml`. You do not copy
source to the practices by hand — `flwr run` builds and ships it.

---

## Step 1 — TLS certificates

The SuperNode→SuperLink link and the CLI→SuperLink link must both be TLS in production. You need a
CA certificate, a server certificate and its key.

```bash
# Development only — use your organisation's real PKI for the pilot.
openssl req -x509 -newkey rsa:4096 -days 3650 -nodes \
  -keyout ca.key -out ca.crt -subj "/CN=FLIP-IT Dev CA"

openssl req -newkey rsa:4096 -nodes \
  -keyout server.key -out server.csr -subj "/CN=superlink.flipit.local"

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -days 825 -out server.crt \
  -extfile <(printf "subjectAltName=DNS:superlink.flipit.local,IP:10.0.0.10")
```

> ⚠️ `subjectAltName` is not optional — gRPC rejects certificates without it. Use the exact hostname
> or IP the SuperNodes will dial.

Distribute **`ca.crt` only** to each practice. `server.key` never leaves the central host.

---

## Step 2 — One key pair per practice

SuperNode authentication uses **NIST elliptic-curve** keys; `flwr supernode register` expects a
P-384 public key.

```bash
# On each practice machine (the private key must never leave it):
openssl ecparam -name secp384r1 -genkey -noout -out praxis_01.key
openssl ec -in praxis_01.key -pubout -out praxis_01.pub
```

Send only the `.pub` to whoever administers the SuperLink, over a channel you trust.

---

## Step 3 — Start the SuperLink

```bash
flower-superlink \
  --ssl-ca-certfile   ca.crt \
  --ssl-certfile      server.crt \
  --ssl-keyfile       server.key \
  --enable-supernode-auth \
  --fleet-api-address    0.0.0.0:9092 \
  --control-api-address  0.0.0.0:9093 \
  --database /var/lib/flower/flipit.db
```

- `--enable-supernode-auth` turns on key-based admission. **Without it any node can join.**
- `--database` persists run state across restarts; omit it and everything is in-memory.
- `--insecure` exists for local testing only and is mutually exclusive with node auth — a SuperNode
  started with an auth key against a non-TLS link exits with
  [error 303](https://flower.ai/docs/framework/ref-exit-codes/303.html).

> The older `--auth-list-public-keys <csv>` flag is **deprecated in 1.33**. Register keys with the
> CLI instead (step 4).

Verify:

```bash
ss -lntp | grep -E '9092|9093'
```

---

## Step 4 — Register each practice's public key

From an administrator machine that can reach the Control API. First point the CLI at the SuperLink
(see step 6), then:

```bash
flwr supernode register praxis_01.pub flipit-prod
flwr supernode register praxis_02.pub flipit-prod
# …

flwr supernode list flipit-prod          # confirm what is admitted
```

Removing a practice — end of pilot, withdrawn consent, compromised key:

```bash
flwr supernode unregister <node-id> flipit-prod
```

Federation membership is managed separately with `flwr federation create` /
`flwr federation add-supernode` when you need more than one federation on one SuperLink.

---

## Step 5 — Start a SuperNode at each practice

```bash
flower-supernode \
  --superlink superlink.flipit.local:9092 \
  --root-certificates ca.crt \
  --auth-supernode-private-key praxis_01.key \
  --auth-supernode-public-key  praxis_01.pub \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0 num-partitions=25 fhir-base-url='http://127.0.0.1:8080'"
```

`--node-config` is how this app learns which practice it is and where its data lives:

| Key | Meaning |
|---|---|
| `partition-id` | 0-based practice index; unique per SuperNode |
| `num-partitions` | total practices in the federation (25 for the pilot) |
| `fhir-base-url` | this practice's own **Helios FHIR server** (sandbox host: `uv run ckd-helios`), read by `data/fhir_loader.py` / the SQL-on-FHIR views in `data/helios.py` |
| `pseudonym-salt` | optional (default off): enables the per-practice salted-HMAC join key on the extraction — the T4.2 mechanism named in [PRIVACY.md](PRIVACY.md) §2, not a raw identifier |

To read FHIR rather than CSV, run with `data-source=fhir` (step 7). `--clientappio-api-address`
only needs to change if several SuperNodes share one host.

The FHIR path builds the **canonical `extract_features.sql` contract** per patient — a different
prediction task from the synthetic baseline (16 model columns incl. lab missing-indicators, and a
CKD *incidence* label) — and de-identifies at the source: only the contract fields are read, rows
carry no identifier, and row order is a deterministic per-extraction hash order. Validate the
query against the real server before the run, on the practice machine:

```bash
uv run ckd-fhir-extract http://127.0.0.1:8080           # prints exclusion accounting + cohort summary
uv run ckd-fhir-extract http://127.0.0.1:8080 --t0 2025-08-26   # fixed landmark for reproducibility
```
In the sandbox, the server is the seeded local **Helios** instance: `uv run ckd-helios`
(serve + seed), `uv run ckd-helios extract` (round-trip proof), `uv run ckd-helios sql`
(SQL-on-FHIR views). The repo builds only from Helios (CLAUDE.md §0.9).

---

## Step 6 — Point the CLI at the federation

SuperLink *connection* config moved out of `pyproject.toml` in flwr 1.30+. It now lives in
`$HOME/.flwr/config.toml` (or `$FLWR_HOME/config.toml`):

```toml
[superlink]
default = "local-sim"

# Local simulation — no SuperLink process, for development.
[superlink.local-sim]
address = ":local:"
options.num-supernodes = 12

# The real federation.
[superlink.flipit-prod]
address = "superlink.flipit.local:9093"      # the CONTROL API port, not the Fleet port
root-certificates = "/etc/flower/ca.crt"
insecure = false
```

```bash
flwr config list        # shows every connection and which is default
```

### Remove the Flower-operated default (do this deliberately)

The Flower CLI writes a `supergrid` entry into a freshly generated `config.toml` pointing at an
endpoint Flower Labs operates:

```toml
[superlink.supergrid]
address = "supergrid.flower.ai"
```

It is inert unless someone explicitly runs `flwr run . supergrid` — but that is one word on a
command line between a self-hosted federation and transmission of model updates to a third party.
On every operator and practice machine:

```bash
flwr config list                     # confirm what is configured
# delete the [superlink.supergrid] block from $HOME/.flwr/config.toml
flwr config list                     # confirm it is gone and the default is the consortium SuperLink
```

Pin `default` to the consortium SuperLink so an omitted federation argument cannot reach outside the
consortium either. This is on the practice-side checklist below; it is cheap to do and easy to
forget.

---

## Step 7 — Push the app

```bash
flwr run . flipit-prod --stream
```

That builds the FAB from `pyproject.toml`, ships it to the SuperLink, and starts the run;
`--stream` follows the logs live.

Override run config without editing the file:

```bash
flwr run . flipit-prod --stream \
  --run-config "num-server-rounds=20 data-source='fhir' num-practices=25"
```

Managing runs:

```bash
flwr ls flipit-prod                # list runs
flwr log <run-id> flipit-prod      # fetch logs
flwr stop <run-id> flipit-prod     # stop a run
```

---

## Console runbook — plain-weights FHIR demo (what notebook 04 does, over the wire)

[`notebooks/04_helios_sql_federated_fedprox.ipynb`](../notebooks/04_helios_sql_federated_fedprox.ipynb)
proves this exact run in one process: Helios **SQL-on-FHIR** queries → the canonical
`extract_features.sql` frame → federated logreg via **Flower's real strategies**, with a plain
weights-only wire and no privacy layer. This is the same thing from the terminal. Every ε key is
left default (0) and `secure-aggregation` is not set, so the wire carries **only model
coefficients** — no DP, no SecAgg, exactly the demo scope.

1) **Practice Helios servers up + verified** (each practice, against its own Helios):
   ```bash
   uv run ckd-helios             # host + seed the practice Helios server on :8080
   curl -fsS http://127.0.0.1:8080/metadata > /dev/null && echo "helios ok"
   uv run ckd-helios extract     # round-trip proof: canonical extraction reproduces the seed
   uv run ckd-helios sql         # optional: the SQL-on-FHIR views -> data/helios/out/
   ```

2) **SuperLink up** (central host). Development rehearsal is `--insecure`; the pilot uses
   TLS + node auth (Steps 1–4):
   ```bash
   flower-superlink \
     --ssl-ca-certfile ca.crt --ssl-certfile server.crt --ssl-keyfile server.key \
     --enable-supernode-auth \
     --fleet-api-address 0.0.0.0:9092 --control-api-address 0.0.0.0:9093 \
     --database /var/lib/flower/flipit.db
   ```
   then admit each practice: `flwr supernode register praxis_NN.pub flipit-prod`.

3) **SuperNode per practice** (with its own Helios endpoint — Step 5):
   ```bash
   flower-supernode \
     --superlink superlink.flipit.local:9092 --root-certificates ca.crt \
     --auth-supernode-private-key praxis_01.key --auth-supernode-public-key praxis_01.pub \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config "partition-id=0 num-partitions=25 fhir-base-url='http://127.0.0.1:8080'"
   ```

4) **Push the plain-weights run**:
   ```bash
   flwr run . flipit-prod --stream --run-config \
     "fraction-fit=1.0 num-server-rounds=20 num-practices=25 \
      class-weight-balanced=true seed=42 local-epochs=2 data-source='fhir'"
   ```
   `server_app.main()` reads all of `fraction-fit`, `num-server-rounds`, `num-practices`,
   `class-weight-balanced`, `seed` and `local-epochs` with **no defaults** — leave none out.
   Optional here: `min-cohort-size`, `min-train-examples`, `metric-privacy`. With no ε key and
   no `secure-aggregation`, the ServerApp is plain `CohortFloorFedAvg` — plain weights only.

5) **Monitor / teardown**:
   ```bash
   flwr ls flipit-prod        # runs; `--stream` already followed logs
   flwr log <run-id> flipit-prod
   flwr stop <run-id> flipit-prod
   ```
   Teardown order: stop the run → Ctrl-C each SuperNode → stop the SuperLink. **Helios stays
   up** — it is the practice's own server, not part of the Flower stack.

> **FedAvg on the wire, FedProx on the research bench.** The live `ServerApp` ships
> `CohortFloorFedAvg`, a Flower `FedAvg` subclass. FedProx — Flower's built-in strategy, μ = 0.1 in
> `models/protocols` — is exercised **in-process** in `simulate.py` / `models/protocols.run_protocol`
> and in notebook 04 through the *same real `flwr.serverapp.strategy.FedProx` object* the
> SuperLink would run, on identical frames, comparably with `local` and `fedavg`. If the pilot picks
> FedProx, it is a run-config change on the live `server_app` (the strategy already ships in flwr
> 1.33) — it is just not currently the deployed default.

---

## Local rehearsal — do this before touching a practice

Two SuperNodes on one machine, insecure, proving the wiring end to end:

```bash
# terminal 1
flower-superlink --insecure

# terminal 2
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0 num-partitions=2"

# terminal 3
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1 num-partitions=2"

# terminal 4
flwr run . local-deployment --stream --run-config "num-practices=2 num-server-rounds=3"
```

with

```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

The pure-simulation path (`uv run flwr run .`, no SuperLink process) is verified working in this
repo and is the faster check while iterating on app code.

---

All privacy layers are **run-config keys, not code edits** — no wrapping in `server_app.py` and no
client mods by hand. `server_app.main()` reads them, derives every budget parameter itself, and
refuses contradicting combinations.

**| The deployment standard (recommended).** Record-level DP-SGD inside each clinic with the
ε shared across the federation:

```bash
flwr run . flipit-prod --stream --run-config "dpsgd-epsilon=3.0 num-server-rounds=10"
# …and, together on the production path, the masked wire:
flwr run . flipit-prod --stream --run-config "dpsgd-epsilon=3.0 secure-aggregation=true"
```

You set only the target ε*. A **rule-based server agent** (no LLM — the four deterministic rules
in `orchestrator.AGENT_RULES`, [PRIVACY.md §3.5](PRIVACY.md#35-the-deployment-standard-record-level-dp-sgd--secagg--rule-based-epsilon-orchestration)) runs at
round zero: it counts the connected SuperNodes (rule 1 — census N is the only datum exchanged),
enforces the fixed global policy (rule 2 — ε* with δ = 1e-5), inverts the shared RDP accountant
per clinic to pick `(batchₖ, σₖ)` — larger σ for smaller clinics, every composed budget ≤ ε*
(rule 3), and stamps each clinic's `ClinicPlan` into the ConfigRecord on its outgoing train
instructions (rule 4). SuperNodes stay generic executors; changing ε is one server-side edit.

Two hard rules a pilot must plan around:

- **The census has a floor.** A practice with N < 8 training patients gets **no unsafe plan**;
  the run refuses to start until the federation is censused above the smallest candidate batch.
  No SuperNode silently trains unprotected.
- **One budget mechanism per run.** `dpsgd-epsilon` is mutually exclusive with `central-dp-epsilon`
  and `local-dp-epsilon` (server raises at startup) — stacking them double-spends the same budget
  without changing what is released.

**Secure Aggregation needs the legacy workflow path** — `SecAggPlusWorkflow` requires a
`LegacyContext` and cannot compose with `strategy.start()` in 1.33; the working construction is in
[PRIVACY.md §4](PRIVACY.md#4-l3--secagg-is-implemented) and is what
`secure-aggregation=true` drives, **including for the standard's census round** (the headcount is
collected over the masked rails; on the legacy path the per-clinic plans are stamped into each
client's `FitIns`). Verified end-to-end: 12 practices, fit and evaluate with 0 failures.

**The two comparison baselines remain available** for audits/`PRIVACY.md` tables, decided by run
config, not code: `central-dp-epsilon=<ε>` (budget-first; σ derived by `dp.sigma_for_epsilon` for
the run's round count, server-side clipping — cannot combine with `secure-aggregation=true`, which
the log will refuse) and `local-dp-epsilon=<ε>` (update-level client-side clipping; message-API path only — flwr 1.33's `LocalDpMod` is record-key-shaped and the
SecAgg+ legacy rail cannot serve it, so the server refuses `secure-aggregation=true` with
`local-dp-epsilon > 0` — and measured unusable at defensible budgets — worst practice 0.397 at ε≈4.3,
[PRIVACY.md §3](PRIVACY.md#3-l4--what-differential-privacy-actually-costs)). The comparison is why
the standard exists; the standard is what ships.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Exit code 303 | SuperNode has an auth key but the link is not TLS | Drop `--insecure`, pass `--root-certificates` |
| `CERTIFICATE_VERIFY_FAILED` | `ca.crt` mismatch, or missing `subjectAltName` | Re-issue the server cert with a correct SAN |
| SuperNode connects, run never starts | `num-practices` > nodes actually connected | `min_available_nodes` blocks; match the counts |
| `flwr run` cannot reach the SuperLink | Pointed at the Fleet port | Use the **Control API** port (9093) in `config.toml` |
| Node rejected | Public key not registered | `flwr supernode register <key> flipit-prod` |
| `ConnectionError` from the FHIR loader | Practice **Helios** server down or wrong URL | Check `fhir-base-url` in `--node-config` (`uv run ckd-helios` in the sandbox) |
| `ValueError: ... below the smallest candidate batch 8` | A practice's census is under the DP-SGD floor (typical: low-`alpha` Dirichlet sims leave tiny partitions) | The rule-based agent refuses unsafe plans — federate only practices above the floor; in simulation, raise `alpha` (0.5 → 5.0) |

---

## Practice-side checklist

Per practice, before go-live:

- [ ] `uv sync --extra dev` succeeds; `flwr` reports 1.33.x
- [ ] `ca.crt` present; `praxis_NN.key` present and readable only by the service user
- [ ] Public key registered centrally (`flwr supernode list` shows it)
- [ ] FHIR server reachable: `curl -s $FHIR/metadata | head`; cohort validated:
      `uv run ckd-fhir-extract $FHIR` (prints inclusion accounting; no rows leave the machine)
- [ ] `--node-config` has the right `partition-id` and `num-partitions`
- [ ] Egress to the SuperLink on 9092 permitted; **no inbound** rule needed
- [ ] Confirmed: no patient rows in any outbound payload — only model parameters
      (or, for FedMosaic, predictions on the public cohort)
- [ ] `[superlink.supergrid]` removed from `$HOME/.flwr/config.toml`, and `default` pinned to the
      consortium SuperLink (Step 6) — no path to a Flower-operated endpoint
- [ ] `metric-privacy = true` and `min-cohort-size` at the agreed floor (deployment default since
      the L5 hardening) — per-practice metric lines are not identified in the SuperLink logs
- [ ] `dpsgd-epsilon` is the agreed target and the practice's train census clears the floor
      (N ≥ 8) — go-live starts with the orchestrator's census round; the server log's plan table
      must name this practice's own `(batch, σ)` before any weights move (a practice under the
      floor is refused, deliberately)
- [ ] Extract carries **no** `patientid` column (`extract_features.sql` §9) — the practice CSV is a
      pseudonymous extract, not personal data
