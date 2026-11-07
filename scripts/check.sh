#!/usr/bin/env bash
# Quality gate: lint, tests and a content check of tracked files.
# Prints only what failed. Ends with GATE PASS or GATE FAIL.
# Optional: CHECK_NOTEBOOKS=1 bash scripts/check.sh — also re-runs all notebooks.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"
LOG_DIR="$(mktemp -d)"; trap 'rm -rf "$LOG_DIR"' EXIT
FAILED=0

run_step() {
  local name="$1"; shift
  local log="$LOG_DIR/${name//[\/:]/_}.log"
  if "$@" >"$log" 2>&1; then echo "ok   $name"
  else
    echo "FAIL $name"; echo "----- $name (last 40 lines) -----"; tail -n 40 "$log"; echo "---------------------------------"
    FAILED=1
  fi
}

run_step "ruff" "$PY" -m ruff check src tests

if ls tests/test_*.py >/dev/null 2>&1; then
  run_step "pytest" "$PY" -m pytest -q
else
  echo "skip pytest (no tests yet)"
fi

# No secrets or local-only files tracked
if git rev-parse --git-dir >/dev/null 2>&1; then
  run_step "no-secrets" bash -c '! git ls-files --cached | grep -E "(^|/)\.env$"'
  if [ -x .git/hooks/forbidden-scan ]; then
    run_step "content" bash -c 'git ls-files -z | xargs -0 cat 2>/dev/null | .git/hooks/forbidden-scan >/dev/null'
  fi
fi

if [ "${CHECK_NOTEBOOKS:-0}" = "1" ] && ls notebooks/*.ipynb >/dev/null 2>&1; then
  for nb in notebooks/*.ipynb; do
    run_step "nb:$(basename "$nb")" "$PY" -m jupyter nbconvert --to notebook --execute --inplace "$nb"
  done
fi

if [ "$FAILED" = "0" ]; then echo "GATE PASS"; else echo "GATE FAIL"; exit 1; fi
