# retreat_presentation — FLIP-IT 10-minute retreat deck kit

Everything needed for Claude (Claude Code / Claude Design) to build an **editable PowerPoint**
of the 10-minute FLIP-IT retreat talk, grounded in this repository's measured results.

**Talk:** *FLIP-IT — federated learning for kidney-disease risk, without moving a single patient
record.* Audience: FL/ML research retreat. Language: English. Frame: FL has three problems —
(1) data modality → model, (2) model + privacy → protocol, (3) privacy protocol ↔ law.

## Quick start

1. Connect Claude to this GitHub repository.
2. Paste the full contents of **[`PROMPT.md`](PROMPT.md)** into Claude.
3. It writes `scripts/build_deck.py`, runs it, verifies, and produces
   **`FLIP-IT_retreat_deck.pptx`** (13 slides: 10 timed + 3 backup) with full speaker notes.

Open the `.pptx` in PowerPoint/Keynote, or drag it into Google Slides — all text stays editable.

## What's in here

| Path | Purpose |
|---|---|
| `PROMPT.md` | **The prompt.** Build instructions, design spec, acceptance checks. |
| `SLIDES.md` | Slide-by-slide content + **verbatim speaker notes** (~1,600 words ≈ 10 min) + timings. Read this yourself to rehearse. |
| `data/NUMBERS.md` | Every number in the deck + provenance (which live run or doc it came from). |
| `data/benchmark*.json` | Live protocol benchmark, seeds 42–46 + merged 5-seed summary (2026-09-16). |
| `data/privacy.json` | Live central-DP & local-DP sweeps + SecAgg probe (2026-09-16). |
| `data/audit.json` | Live membership-inference audit (2026-09-16). |
| `assets/fig01–fig06` | Generated charts (1600×900) from those JSONs. |
| `assets/Flipit-privacy.png`, `Flipit-process.png` | The camera-ready project diagrams. |
| `assets/01_patient_form.png`, `02_risk_score_result.png` | Physician-demo webapp screenshots. |
| `scripts/make_figures.py` | Regenerates the charts + merged summary from `data/*.json`. |

## Regenerating the evidence

```bash
# from the repo root — NOTE: use .venv/bin/... directly
./.venv/bin/ckd-clinics --clinics 10
for s in 42 43 44 45 46; do ./.venv/bin/ckd-benchmark --rounds 20 --seed $s --out results/benchmark_seed$s.json; done
./.venv/bin/ckd-privacy --rounds 20 --seeds 42 43 44 45 46
./.venv/bin/ckd-audit   --rounds 20 --seeds 42 43 44 45 46
cp results/*.json retreat_presentation/data/
./.venv/bin/python retreat_presentation/scripts/make_figures.py
```

## Environment caveats (important)

- ⚠️ **`uv run` is broken in this checkout**: the directory name contains `:`
  (`IKIM:Lamarr`), which uv's run resolution rejects (`error: path segment contains
  separator ':'`). Call the venv directly: `./.venv/bin/python`, `./.venv/bin/ckd-*`.
- `python-pptx` 1.0.2 is already installed in `.venv` (via
  `uv pip install --python .venv/bin/python python-pptx`). Project rule: uv only — no pip/conda.

## Honesty contract

This is a **pre-kickoff, synthetic-data** baseline. The deck's numbers are either the live
2026-09-16 re-runs in `data/*.json` or explicitly-marked measurements from `docs/`
(`PRIVACY.md §3.5` DP-SGD standard, `REPORT.md §2` historical XGBoost). Do not present them as
clinical results; the deck itself says so on every slide footer.
