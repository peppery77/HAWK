#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime/python"

"${PYTHON_BIN}" - <<'PY'
missing = []
for module in ("torch", "PIL", "accelerate", "qwen_vl_utils"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing base-environment dependencies: " + ", ".join(missing)
        + ". Run `pip install -r requirements.txt` first."
    )
PY

mkdir -p "${RUNTIME_DIR}"

"${PYTHON_BIN}" -m pip install \
  --target "${RUNTIME_DIR}" \
  --upgrade \
  --no-deps \
  "transformers==4.52.0" \
  "tokenizers==0.21.1" \
  "huggingface-hub==0.31.4" \
  "qwen-vl-utils==0.0.11" \
  "pytest==8.3.5" \
  "pluggy==1.5.0" \
  "iniconfig==2.0.0" \
  "packaging==24.2" \
  "ruff==0.11.10"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/patch_transformers.py" \
  --target "${RUNTIME_DIR}/transformers"

PYTHONPATH="${PROJECT_ROOT}/src:${RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -c \
  "import hawk, transformers; assert transformers.__version__ == '4.52.0'; print('HAWK', hawk.__version__, 'Transformers', transformers.__version__)"
