# PROMPT — build the FLIP-IT retreat deck (paste this into Claude, repo connected)

> Paste this entire file into Claude (Claude Code / Claude Design) with this GitHub repository
> connected. It has everything needed: content source, exact numbers, assets, design spec,
> build requirements, and acceptance checks. Do not ask for more input — read the referenced
> files from the repo.

---

## Role

You are building a **conference-retreat slide deck as an editable PowerPoint** for a technical
member of the FLIP-IT consortium. Every content claim in this deck is backed by files in this
repository; your job is faithful, well-designed assembly — not copywriting, not number-finding.

## Objective

Create **`retreat_presentation/FLIP-IT_retreat_deck.pptx`** — a 16:9, 13-slide (10 timed + 3
backup), editable `.pptx` for a **10-minute talk** titled:

**"FLIP-IT — federated learning for kidney-disease risk, without moving a single patient record"**

Audience: **FL/ML researchers at a Lamarr/IKIM research retreat.** Language: **English.**

**The talk's frame (must structure the deck):** Federated learning has three problems, each with a
measured answer from this project:

1. **Data modality → model** — the shape of GP practice data chooses the model (logistic
   regression; trees measured and rejected).
2. **Model + privacy goal → protocol** — the wire payload IS the disclosure surface (FedAvg /
   FedProx coefficients vs FedMosaic predictions-on-public-data; rounds are privacy currency;
   SecAgg+ verified).
3. **Privacy protocol ↔ law** — central vs local DP, EDPB Opinion 28/2024, the record-level
   DP-SGD + SecAgg + ε-orchestrator deployment standard, and the leakage audit.

## Inputs — read these first (all paths relative to repo root)

| Path | What it is | Authority |
|---|---|---|
| `retreat_presentation/SLIDES.md` | **Slide-by-slide content + verbatim speaker notes, with per-slide timing.** | **Content source of truth.** Slide titles, bullets, and speaker notes come from here. You may tighten layout-wise but MUST NOT rewrite claims or change any number. |
| `retreat_presentation/data/NUMBERS.md` | Every measured number + provenance | Number oracle — if any conflict, NUMBERS.md wins |
| `retreat_presentation/assets/*.png` | 6 generated charts + 2 project diagrams + 2 webapp screenshots | Visuals, already measured/verified |
| `retreat_presentation/data/*.json` | Live-run raw results (2026-09-16) | Evidence trail (reference only) |
| `docs/REPORT.md`, `docs/PRIVACY.md` | Long-form write-ups | Context only |

## Number policy (hard rules)

- Use numbers **exactly** as in SLIDES.md / NUMBERS.md (e.g. `0.808 ± 0.008`, `0.397`,
  `ε ≈ 4.3`, `0.861`, `352` / `3,600` bits, `+0.180`). No re-rounding, no derived stats.
- Keep the provenance framing: benchmark/privacy/audit numbers are **live 2026-09-16 re-runs** on
  synthetic data; the DP-SGD-standard figures are from `docs/PRIVACY.md §3.5` (notebook 03).
- This is a pre-kickoff **synthetic-data** sandbox. No clinical claims anywhere.

## Environment & build requirements (hard rules)

- ⚠️ **Do NOT use `uv run`** in this repo — the directory name contains `:` (`IKIM:Lamarr`),
  which breaks uv's run resolution. The project venv works directly: use
  **`.venv/bin/python`** for everything.
- **`python-pptx` 1.0.2 is already installed** in that venv. If a reinstall is ever needed:
  `uv pip install --python .venv/bin/python python-pptx` (uv only; never bare pip).
- Write the generator as **`retreat_presentation/scripts/build_deck.py`**, run it with
  `.venv/bin/python retreat_presentation/scripts/build_deck.py`, and keep it in the folder
  (it is part of the deliverable: re-runnable deck generation).
- Do **not** modify any repo file outside `retreat_presentation/`.
- Fully editable deck: native text boxes/shapes for text and tables; PNG images inserted as
  pictures. **No flattened all-image slides** (the title slide included).

## Design spec

- **Format:** 13.333" × 7.5" (16:9). Margins ≥ 0.55". All figures at native 16:9-friendly aspect.
- **Palette (matches the charts):** ink blue `#1B4965` (primary/titles), teal `#2A9D8F`
  (deployed-standard accents), amber `#E9C46A` (ceiling/reference), red `#E76F51` (risk/worst
  practice), gray `#8D99AE` (secondary text), white page background.
- **Typography:** one sans family throughout (Arial/Helvetica fallback ok). Slide titles 28–30 pt
  bold ink blue; body 16–18 pt; captions/citations 11–12 pt gray. Generous whitespace; max ~7
  content lines per slide. Numbers in tables/figure callouts may use a mono face for alignment.
- **Title slide:** full ink-blue background, white title, gray-blue subtitle, small
  consortium/funding line. No stock imagery.
- **Chrome:** thin amber accent rule under each slide title; slide number bottom-right; tiny
  footer `FLIP-IT · NEXT.IN.NRW — synthetic-data baseline, 2026-09` bottom-left (skip on title).
- **Three-problem identity:** mark each content slide with a small chip tab top-right:
  `Problem 1 · modality → model` (ink), `Problem 2 · protocol` (teal), `Problem 3 · privacy ↔ law`
  (red) on slides 3–9; slide 2 displays all three chips.
- **Charts:** insert at full quality (they are 1600×900 PNG @200 dpi); never stretch/distort;
  never cover plot areas with text boxes. If a slide needs callouts, place them outside the axes.
- **Diagrams:** `Flipit-privacy.png` (1152×593) fits slide 6 landscape; `Flipit-process.png`
  (1740×2397, portrait) goes left third/fifth of slide 10-style slide 9 with bullets right;
  webapp screenshots (720×1200) pair on slide 9 right side, small.
- **Speaker notes:** put the verbatim `Speaker notes` block from SLIDES.md into each slide's
  **PowerPoint notes** (`slide.notes_slide.notes_text_frame.text = ...`) — markdown `>` stripped.
  Timed slides 1–10 carry their timing as the first notes line, e.g. `[0:30]`.
- **No:** stock photos, medical clip art, invented logos, unrequested decorative iconography,
  walls of text, claims not present in SLIDES.md.

## Slide list (13 total) & asset mapping

| # | Slide (title source: SLIDES.md) | Image(s) |
|---|---|---|
| 1 | Title | — |
| 2 | One promise, three problems | — (three numbered problem boxes as native shapes) |
| 3 | Problem 1 — data modality: what GP data actually looks like | `assets/fig01_cohort.png` |
| 4 | Federation recovers the ceiling — and fixes the worst practice | `assets/fig02_protocols.png` |
| 5 | Problem 2 — what actually crosses the wire | `assets/fig03_convergence.png` + native wire table |
| 6 | Problem 3 — the privacy protocol must answer to the law | `assets/Flipit-privacy.png` |
| 7 | Record-level DP inside the clinic survives ε = 0.5–8 | `assets/fig05_central_vs_local.png` |
| 8 | Membership inference finds nothing — for the right reason | `assets/fig06_audit.png` |
| 9 | From 10 synthetic practices to 25 real ones | `assets/Flipit-process.png` (+ 2 webapp screenshots) |
| 10 | Three problems, three measured answers + closer line | — |
| 11 | Backup — central DP sweep, row by row | `assets/fig04_dp_central.png` |
| 12 | Backup — is uniform ε* fair? (eras 14–15) | — |
| 13 | Backup — reproduce everything + references | — (monospace command block) |

Slide order and content: exactly SLIDES.md. Backup slides get a `Backup` chip instead of a
problem chip.

## Process

1. Read SLIDES.md and NUMBERS.md. Extract per-slide: title, body blocks, table blocks, figure
   path(s), speaker notes (verbatim), timing tag.
2. Write `retreat_presentation/scripts/build_deck.py` (python-pptx): helper functions
   (`add_title_slide`, `add_figure_slide`, `add_text_slide`, `chip`, `footer`), one builder per
   slide, notes writer, then save `retreat_presentation/FLIP-IT_retreat_deck.pptx`.
3. Run it with `.venv/bin/python`, then run this readback verification (include it as
   `scripts/verify_deck.py`, run with the same interpreter):

   - 13 slides exist.
   - Every slide has non-empty notes; slides 1–10 notes contain a `[m:ss]` timing tag.
   - Slides 3–9 and 11 each embed ≥ 1 picture; slide 9 embeds ≥ 2.
   - No picture's native aspect ratio is distorted > 2%.
   - The strings `0.808`, `0.861`, `0.397`, `352`, `3,600`, `+0.180`, `SecAgg`, `EDPB` each
     appear at least once across slide text.
   - Print a PASS/FAIL summary line.
4. If any check fails, fix the builder and re-run; report final check output, the deck path, and
   the slide count.

## Definition of done

`retreat_presentation/FLIP-IT_retreat_deck.pptx` opens in PowerPoint/Keynote/Google Slides with
13 editable slides, charts embedded crisply, full speaker notes on every slide, the verification
script passing, and zero numbers invented or re-rounded.
