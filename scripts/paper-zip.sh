#!/usr/bin/env bash
# Package the expert report for upload to Overleaf.
#
# Includes the generated figures/tables/macros deliberately: an Overleaf project must build without
# running Python. Excludes the cached run (paper/generated/data.json, several MB and not needed to
# typeset) and any local LaTeX build products.
#
#   ./scripts/paper-zip.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-flip-it-expert-report.zip}"
rm -f "$OUT"
zip -r "$OUT" paper \
  -x 'paper/generated/data.json' \
     'paper/__pycache__/*' \
     'paper/*.aux' 'paper/*.log' 'paper/*.out' 'paper/*.toc' \
     'paper/*.bbl' 'paper/*.bcf' 'paper/*.blg' 'paper/*.fls' 'paper/*.run.xml' \
     'paper/main.pdf' \
  >/dev/null
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload to Overleaf, set the compiler to pdfLaTeX. Bibliography uses bibtex (automatic)."
