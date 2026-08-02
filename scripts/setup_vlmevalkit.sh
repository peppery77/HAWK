#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_PYTHON="${VLMEVAL_BOOTSTRAP_PYTHON:-${PYTHON_BIN}}"
TARGET_ROOT="${PROJECT_ROOT}/.runtime/VLMEvalKit"

if [[ ! -d "${TARGET_ROOT}/vlmeval" ]]; then
  if [[ -n "${VLMEVAL_SOURCE:-}" ]]; then
    SOURCE_ROOT="${VLMEVAL_SOURCE}"
  else
    SOURCE_ROOT="$("${BOOTSTRAP_PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vlmeval")
if spec is None or spec.origin is None:
    raise SystemExit(
        "VLMEvalKit was not found. Install requirements.txt or set "
        "VLMEVAL_SOURCE=/path/to/VLMEvalKit/vlmeval."
    )
print(Path(spec.origin).resolve().parent)
PY
)"
  fi
  if [[ -f "${SOURCE_ROOT}/vlmeval/__init__.py" ]]; then
    SOURCE_ROOT="${SOURCE_ROOT}/vlmeval"
  fi
  if [[ ! -f "${SOURCE_ROOT}/__init__.py" ]]; then
    echo "Invalid VLMEvalKit package directory: ${SOURCE_ROOT}" >&2
    exit 1
  fi
  mkdir -p "${TARGET_ROOT}"
  "${PYTHON_BIN}" - "${SOURCE_ROOT}" "${TARGET_ROOT}/vlmeval" <<'PY'
from pathlib import Path
from shutil import copytree
import sys

source, target = map(Path, sys.argv[1:])
copytree(source, target, dirs_exist_ok=True)
PY
  printf '%s\n' \
    "VLMEvalKit-compatible source copied from ms-vlmeval 0.0.18." \
    > "${TARGET_ROOT}/SOURCE_VERSION.txt"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/patch_vlmevalkit.py" \
  --root "${TARGET_ROOT}"
