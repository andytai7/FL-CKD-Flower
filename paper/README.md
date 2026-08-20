# FLIP-IT — Technical Expert Report (LaTeX source)

The report answering the preliminary legal assessment of 10 August 2026. Every section is one of
counsel's questions; every number resolves through a generated macro.

## Layout

```
main.tex              preamble, title page, \input of every section
refs.bib              bibliography
sections/             one .tex per question — the editing unit
figures/              generated vector PDFs      ← do not hand-edit
tables/               generated booktabs tables  ← do not hand-edit
generated/
  macros.tex          a \newcommand per inline number  ← do not hand-edit
  manifest.tex        run date, package versions, seeds
  data.json           the cached experiment run
make_paper.py         produces everything above
```

## Editing

Edit `sections/*.tex` freely — that is what they are for.

**Do not type a number into the prose.** Every quantity comes from `generated/macros.tex`, so the
document cannot drift from the data it describes. If you need a number that has no macro, add it to
`_macros()` in `make_paper.py` and regenerate. The check that keeps this honest:

```bash
grep -nE '[0-9]\.[0-9]{2,}' sections/*.tex     # should match only layout and formula constants
```

## Building

### Overleaf

```bash
./scripts/paper-zip.sh          # -> flip-it-expert-report.zip
```

Upload that zip. Set the compiler to **pdfLaTeX**; the bibliography uses biber, which Overleaf
selects automatically. Only standard CTAN packages are used.

The zip deliberately **includes** `figures/`, `tables/` and `generated/` — an Overleaf project must
typeset without running Python — and excludes the cached run (`generated/data.json`, several MB and
not needed to build) and any local build products.

### Locally

```bash
latexmk -pdf main.tex          # or: tectonic -X compile main.tex
```

## Regenerating figures, tables and numbers

```bash
uv run python paper/make_paper.py              # render from the cache (fast)
uv run python paper/make_paper.py --recompute  # re-run every experiment (~10 min), then render
```

`--recompute` re-runs the protocol benchmark, both DP sweeps, the SecAgg+ probe and the
membership-inference audit, then rebuilds every figure, table and macro from that one run.

### The determinism gate

The pipeline is fully seeded, so **two consecutive runs must produce byte-identical output**:

```bash
uv run python paper/make_paper.py --recompute && cp -r generated /tmp/gen-a
uv run python paper/make_paper.py --recompute && diff -r /tmp/gen-a generated
```

A difference means a seeding regression, not noise. This check failed before the fix described in
Appendix A of the report — Flower draws DP noise from NumPy's *global* RNG, which nothing was
seeding — and it is the reason the report's privacy figures supersede all earlier ones.

## Design notes

Figures use categorical slots 1–3 of the reference palette, one per protocol, with neutral greys for
the two reference baselines — "protocol vs baseline" is the real distinction, and cycling five hues
would imply five peers. Validated for this subset: worst normal-vision ΔE 24.0 (floor 15), worst CVD
ΔE 10.0 across protanopia/deuteranopia/tritanopia (target 8). One hue sits below the 3:1 contrast
gate on white, so every figure carries direct value labels and a companion table.

Line style and marker vary with colour throughout, so every figure survives greyscale printing —
this report gets printed.
