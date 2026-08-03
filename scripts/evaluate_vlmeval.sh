#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
    if [[ ! -x "${TORCHRUN_BIN}" ]]; then
        TORCHRUN_BIN="$(command -v torchrun)"
    fi
fi
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
RUN_NAME="${RUN_NAME:-hawk_native_p80}"
KEEP_RATIO="${KEEP_RATIO:-0.20}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-RealWorldQA}"
HEAD_WEIGHTS_JSON="${HEAD_WEIGHTS_JSON:-}"
SCORE_NORMALIZATION="${SCORE_NORMALIZATION:-l2}"
CONFIG_PATH="${PROJECT_ROOT}/validation/vlmeval_${RUN_NAME}.json"
OUTPUT_ROOT="${PROJECT_ROOT}/vlmeval_results/${RUN_NAME}"
TRACE_PATH="${OUTPUT_ROOT}/traces/pruning"
DATA_ROOT="${PROJECT_ROOT}/data/vlmeval"

# Paper Table 3 / VLMEvalKit Qwen2.5-VL native-resolution defaults.
MIN_PIXELS=$((1280 * 28 * 28))
MAX_PIXELS=$((16384 * 28 * 28))

mkdir -p "${OUTPUT_ROOT}" "${TRACE_PATH%/*}" "${PROJECT_ROOT}/validation"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export LMUData="${DATA_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/.runtime/python:${PROJECT_ROOT}/.runtime/VLMEvalKit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/patch_vlmevalkit.py" \
  --root "${PROJECT_ROOT}/.runtime/VLMEvalKit"

"${PYTHON_BIN}" - "${CONFIG_PATH}" "${MODEL_PATH}" "${KEEP_RATIO}" \
  "${MIN_PIXELS}" "${MAX_PIXELS}" "${TRACE_PATH}" "${DATA_ROOT}" \
  "${OUTPUT_ROOT}" "${RUN_NAME}" "${CUDA_DEVICES}" "${NUM_PROCESSES}" \
  "${DATASETS}" "${HEAD_WEIGHTS_JSON}" "${SCORE_NORMALIZATION}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from hawk import QWEN2_5_VL_7B_HEAD_WEIGHTS

(
    config_path,
    model_path,
    keep_ratio,
    min_pixels,
    max_pixels,
    trace_path,
    data_root,
    output_root,
    run_name,
    cuda_devices,
    num_processes,
    datasets_csv,
    head_weights_json,
    score_normalization,
) = sys.argv[1:]
weights = (
    json.loads(head_weights_json)
    if head_weights_json
    else list(QWEN2_5_VL_7B_HEAD_WEIGHTS)
)
if len(weights) != 28 or not all(
    isinstance(weight, (int, float)) for weight in weights
):
    raise ValueError("HEAD_WEIGHTS_JSON must contain exactly 28 numeric values")
weights = [float(weight) for weight in weights]
dataset_names = [
    dataset.strip() for dataset in datasets_csv.split(",") if dataset.strip()
]
dataset_specs = {
    "RealWorldQA": {
        "class": "HawkLocalImageMCQDataset",
        "dataset": "RealWorldQA",
        "data_file": str(Path(data_root, "RealWorldQA_local.tsv").resolve()),
    },
    "ScienceQA_TEST": {
        "class": "HawkLocalImageMCQDataset",
        "dataset": "ScienceQA_TEST",
        "data_file": str(Path(data_root, "ScienceQA_TEST_local.tsv").resolve()),
    },
    "ChartQA_TEST": {
        "class": "HawkLocalImageVQADataset",
        "dataset": "ChartQA_TEST",
        "data_file": str(Path(data_root, "ChartQA_TEST_local.tsv").resolve()),
    },
    "TextVQA_VAL": {
        "class": "HawkLocalImageVQADataset",
        "dataset": "TextVQA_VAL",
        "data_file": str(Path(data_root, "TextVQA_VAL_local.tsv").resolve()),
    },
    "MME": {
        "class": "HawkLocalImageYORNDataset",
        "dataset": "MME",
        "data_file": str(Path(data_root, "MME_local.tsv").resolve()),
    },
}
unknown_datasets = sorted(set(dataset_names) - set(dataset_specs))
if not dataset_names or unknown_datasets:
    raise ValueError(
        f"DATASETS must select from {sorted(dataset_specs)}, "
        f"got {dataset_names}"
    )
if score_normalization not in {"l2", "softmax", "minmax"}:
    raise ValueError(
        "SCORE_NORMALIZATION must be one of: l2, softmax, minmax"
    )
keep_ratio_value = float(keep_ratio)
pruning_percent = round((1.0 - keep_ratio_value) * 100)
model_name = (
    "Qwen2.5-VL-7B-Baseline"
    if keep_ratio_value >= 1.0
    else f"Qwen2.5-VL-7B-HAWK-p{pruning_percent:02d}"
)
config = {
    "model": {
        model_name: {
            "class": "HawkQwen2VLChat",
            "model_path": str(Path(model_path).resolve()),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "max_new_tokens": 2048,
            "attn_implementation": "sdpa",
            "use_custom_prompt": False,
            "hawk_keep_ratio": keep_ratio_value,
            "hawk_head_weights": weights,
            "hawk_score_normalization": score_normalization,
            "hawk_trace_path": trace_path,
        }
    },
    "data": {dataset: dataset_specs[dataset] for dataset in dataset_names},
}
Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
manifest = {
    "framework": "VLMEvalKit",
    "run_name": run_name,
    "model_name": model_name,
    "model_path": str(Path(model_path).resolve()),
    "keep_ratio": keep_ratio_value,
    "pruning_ratio": 1.0 - keep_ratio_value,
    "resolution_mode": "native",
    "min_pixels": int(min_pixels),
    "max_pixels": int(max_pixels),
    "attn_implementation": "sdpa",
    "cuda_devices": cuda_devices.split(","),
    "num_processes": int(num_processes),
    "datasets": dataset_names,
    "score_normalization": score_normalization,
    "head_weights": weights,
    "head_weights_sum": sum(weights),
    "head_weights_sha256": hashlib.sha256(
        json.dumps(weights, separators=(",", ":")).encode()
    ).hexdigest(),
}
output_path = Path(output_root)
output_path.mkdir(parents=True, exist_ok=True)
(output_path / "hawk-vlmeval-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "-1" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc-per-node "${NUM_PROCESSES}" \
  "${PROJECT_ROOT}/scripts/run_vlmeval.py" \
  --config "${CONFIG_PATH}" \
  --work-dir "${OUTPUT_ROOT}" \
  --mode all \
  "${LIMIT_ARGS[@]}"
