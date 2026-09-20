#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-.}"
for f in optimize_network.py routing_alns.py routing_quality.py run_portfolio.py REQUIREMENTS_TRACEABILITY.md README_HARDENED.md test_hardening.py .gitignore; do
  cp -f "$(dirname "$0")/$f" "$TARGET/$f"
done
printf 'Hardened update applied to %s\n' "$TARGET"
printf 'Next: uv sync && uv run python -m unittest -v\n'
