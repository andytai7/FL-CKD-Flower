# scripts/

Runnable shell wrappers around the project's `uv` commands. Run any of them from anywhere —
each `cd`s to the repo root first. The [Makefile](../Makefile) targets just call these.

| Script | What it does | Make equivalent |
|---|---|---|
| `./scripts/setup.sh` | Create/repair the venv + install everything | `make setup` |
| `./scripts/clinics.sh` | Generate synthetic per-clinic datasets into `data/clinics/` | `make clinics` |
| `./scripts/baseline.sh` | Centralized ceiling (logreg + mlp + xgboost) | `make baseline` |
| `./scripts/simulate.sh` | Federated simulation (logreg, non-IID) | `make simulate` |
| `./scripts/notebook.sh` | Re-run the exploration notebook headless | `make notebook` |
| `./scripts/lab.sh` | Open the notebook in JupyterLab | `make lab` |
| `./scripts/test.sh` | Run the test suite | `make test` |
| `./scripts/lint.sh` | Lint with ruff | `make lint` |
| `./scripts/clean.sh` | Remove caches + `.DS_Store` | `make clean` |

Most accept pass-through flags, e.g.:

```bash
./scripts/baseline.sh --model xgboost
./scripts/simulate.sh --model mlp --rounds 10
./scripts/simulate.sh --iid
./scripts/simulate.sh --practices 25 --alpha 0.1
```
