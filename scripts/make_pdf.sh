#!/usr/bin/env bash
# Render the single-file print edition (site/print.html) to a PDF with headless Chrome.
#
#   python3 build.py           # writes site/print.html (cover + TOC + all chapters)
#   scripts/make_pdf.sh        # -> site/the-llm-stack.pdf
#
# NOTE: a *snap-confined* Chromium cannot write to a repo on a non-home mount (e.g. /local-ssd)
# and will fail with an AppArmor "Permission denied". Use a distro google-chrome/chromium, or
# run this in CI, or point the output at a path the sandbox allows. Math (KaTeX) and figures
# render because Chrome executes the page's JS; interactive widgets print their initial state.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN="$ROOT/site/print.html"
OUT="${1:-$ROOT/site/the-llm-stack.pdf}"
[ -f "$IN" ] || { echo "Missing $IN — run: python3 build.py"; exit 1; }
BIN="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)"
[ -n "$BIN" ] || { echo "No Chrome/Chromium found on PATH."; exit 1; }
echo "Rendering $IN -> $OUT with $BIN ..."
"$BIN" --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=60000 --run-all-compositor-stages-before-draw \
  --print-to-pdf="$OUT" "file://$IN"
echo "Wrote $OUT"
