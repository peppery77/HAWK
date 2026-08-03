#!/usr/bin/env python3
"""Download one benchmark and run its native-resolution VLMEvalKit evaluation."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    prepare_command: tuple[str, ...]


TASKS = {
    "realworldqa": TaskSpec(
        "RealWorldQA",
        (
            "scripts/run.sh",
            "scripts/prepare_vlmeval_datasets.py",
            "--project-root",
            ".",
            "--datasets",
            "RealWorldQA",
        ),
    ),
    "scienceqa": TaskSpec(
        "ScienceQA_TEST",
        (
            "scripts/run.sh",
            "scripts/prepare_vlmeval_datasets.py",
            "--project-root",
            ".",
            "--datasets",
            "ScienceQA_TEST",
        ),
    ),
    "chartqa": TaskSpec(
        "ChartQA_TEST",
        ("scripts/run.sh", "scripts/prepare_chartqa_hf.py", "--project-root", "."),
    ),
    "textvqa": TaskSpec(
        "TextVQA_VAL",
        ("scripts/run.sh", "scripts/prepare_textvqa_hf.py", "--project-root", "."),
    ),
    "mme": TaskSpec(
        "MME",
        ("scripts/run.sh", "scripts/prepare_mme_hf.py", "--project-root", "."),
    ),
}


def normalize_task(value: str) -> str:
    task = value.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "chartqatest": "chartqa",
        "textvqaval": "textvqa",
        "scienceqatest": "scienceqa",
    }
    task = aliases.get(task, task)
    if task not in TASKS:
        raise argparse.ArgumentTypeError(
            f"unknown task {value!r}; choose from {', '.join(TASKS)}"
        )
    return task


def keep_ratio_from_pruning(pruning_ratio: float) -> float:
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError("pruning_ratio must be in [0, 1)")
    return 1.0 - pruning_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a benchmark from Hugging Face and evaluate HAWK with "
            "VLMEvalKit. pruning_ratio is the fraction of visual tokens removed."
        )
    )
    parser.add_argument("--task", type=normalize_task, required=True)
    parser.add_argument("--pruning_ratio", type=float, required=True)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num_processes", type=int)
    parser.add_argument("--run_name")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--score_normalization",
        choices=("l2", "softmax", "minmax"),
        default="l2",
    )
    parser.add_argument("--head_weights_json")
    parser.add_argument("--skip_data_download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def execute(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    keep_ratio = keep_ratio_from_pruning(args.pruning_ratio)
    spec = TASKS[args.task]
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("gpus must contain at least one CUDA device")
    num_processes = args.num_processes or len(gpu_ids)
    if num_processes < 1 or num_processes > len(gpu_ids):
        raise ValueError("num_processes must be between 1 and the number of GPUs")

    model_path = (PROJECT_ROOT / args.model_path).resolve()
    runtime_manifest = PROJECT_ROOT / ".runtime/python/hawk-patch-manifest.json"
    if not args.dry_run and not runtime_manifest.is_file():
        raise RuntimeError("HAWK runtime is missing; run `make setup` first")
    if not args.dry_run and not model_path.is_dir():
        raise FileNotFoundError(f"model not found: {model_path}; run `make download`")

    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    vlmeval_package = PROJECT_ROOT / ".runtime/VLMEvalKit/vlmeval"
    if not vlmeval_package.is_dir():
        execute(["scripts/setup_vlmevalkit.sh"], env, args.dry_run)
    if not args.skip_data_download:
        execute(list(spec.prepare_command), env, args.dry_run)

    pruning_percent = round(args.pruning_ratio * 100)
    run_name = args.run_name or f"{args.task}_prune{pruning_percent:03d}_native"
    env.update(
        {
            "MODEL_PATH": str(model_path),
            "RUN_NAME": run_name,
            "KEEP_RATIO": f"{keep_ratio:.12g}",
            "CUDA_DEVICES": ",".join(gpu_ids),
            "NUM_PROCESSES": str(num_processes),
            "DATASETS": spec.dataset,
            "SCORE_NORMALIZATION": args.score_normalization,
        }
    )
    if args.limit is None:
        env.pop("LIMIT", None)
    else:
        env["LIMIT"] = str(args.limit)
    if args.head_weights_json:
        env["HEAD_WEIGHTS_JSON"] = args.head_weights_json

    print(
        f"task={args.task} pruning_ratio={args.pruning_ratio:g} "
        f"keep_ratio={keep_ratio:g} gpus={','.join(gpu_ids)}",
        flush=True,
    )
    execute(["scripts/evaluate_vlmeval.sh"], env, args.dry_run)


if __name__ == "__main__":
    main()
