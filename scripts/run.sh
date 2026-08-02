#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime/python"

if [[ ! -f "${RUNTIME_DIR}/hawk-patch-manifest.json" ]]; then
  echo "Runtime is not initialized. Run scripts/setup_runtime.sh first." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "$@"
