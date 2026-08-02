#!/usr/bin/env python3
"""Summarize paired native-resolution VLMEvalKit baseline and HAWK runs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from hawk import QWEN2_5_VL_7B_HEAD_WEIGHTS


DATASETS = ("RealWorldQA", "ScienceQA_TEST")
PAPER = {
    "RealWorldQA": {"baseline": 67.7, "hawk_r080": 65.0},
    "ScienceQA_TEST": {"baseline": 72.8, "hawk_r080": 73.2},
}


def one_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {matches}")
    return matches[0]


def load_traces(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(root / "traces" / "*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def accuracy(root: Path, dataset: str) -> float:
    frame = pd.read_csv(one_file(root, f"*_{dataset}_acc.csv"))
    return float(frame["Overall"].iloc[0]) * 100


def prediction_frame(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_excel(one_file(root, f"*_{dataset}.xlsx"))


def scored_frame(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_excel(one_file(root, f"*_{dataset}_openai_result.xlsx"))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hawk", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("validation/vlmeval_native_comparison.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("validation/VLMEVAL_NATIVE_RESULTS.md"),
    )
    args = parser.parse_args()

    baseline_traces = load_traces(args.baseline)
    hawk_traces = load_traces(args.hawk)
    weights = list(QWEN2_5_VL_7B_HEAD_WEIGHTS)
    weight_hash = hashlib.sha256(
        json.dumps(weights, separators=(",", ":")).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "framework": "VLMEvalKit",
        "resolution": {
            "mode": "native",
            "min_pixels": 1280 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
        },
        "requested_keep_ratio": 0.2,
        "requested_pruning_ratio": 0.8,
        "head_weights": weights,
        "head_weights_sum": sum(weights),
        "head_weights_sha256": weight_hash,
        "datasets": {},
    }

    for dataset in DATASETS:
        baseline_pred = prediction_frame(args.baseline, dataset)
        hawk_pred = prediction_frame(args.hawk, dataset)
        baseline_score = scored_frame(args.baseline, dataset).set_index("index")
        hawk_score = scored_frame(args.hawk, dataset).set_index("index")
        joined = baseline_score[["prediction", "hit"]].join(
            hawk_score[["prediction", "hit"]],
            how="inner",
            lsuffix="_baseline",
            rsuffix="_hawk",
        )
        traces = [
            row["pruning"]
            for row in hawk_traces
            if row.get("dataset") == dataset and row.get("pruning") is not None
        ]
        baseline_dataset_traces = [
            row["pruning"]
            for row in baseline_traces
            if row.get("dataset") == dataset and row.get("pruning") is not None
        ]
        original_visual = [int(row["original_visual_tokens"]) for row in traces]
        kept_visual = [int(row["kept_visual_tokens"]) for row in traces]
        ratio_actual = [float(row["keep_ratio_actual"]) for row in traces]
        diagnostic_traces = [
            row
            for row in traces
            if all(
                key in row
                for key in (
                    "score_l2_norm_min",
                    "score_l2_norm_mean",
                    "score_l2_norm_max",
                )
            )
        ]
        l2_min = [
            float(row["score_l2_norm_min"]) for row in diagnostic_traces
        ]
        l2_mean = [
            float(row["score_l2_norm_mean"]) for row in diagnostic_traces
        ]
        l2_max = [
            float(row["score_l2_norm_max"]) for row in diagnostic_traces
        ]
        baseline_acc = accuracy(args.baseline, dataset)
        hawk_acc = accuracy(args.hawk, dataset)
        report["datasets"][dataset] = {
            "samples": len(baseline_pred),
            "baseline_accuracy": baseline_acc,
            "hawk_accuracy": hawk_acc,
            "delta_pp": hawk_acc - baseline_acc,
            "paper": PAPER[dataset],
            "prediction_rows": {
                "baseline": len(baseline_pred),
                "hawk": len(hawk_pred),
                "paired": len(joined),
            },
            "empty_predictions": {
                "baseline": int(
                    baseline_pred["prediction"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                ),
                "hawk": int(
                    hawk_pred["prediction"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                ),
            },
            "prediction_changed": int(
                (
                    joined["prediction_baseline"].astype(str)
                    != joined["prediction_hawk"].astype(str)
                ).sum()
            ),
            "correctness_transitions": {
                "baseline_correct_hawk_wrong": int(
                    (
                        (joined["hit_baseline"] == 1)
                        & (joined["hit_hawk"] == 0)
                    ).sum()
                ),
                "baseline_wrong_hawk_correct": int(
                    (
                        (joined["hit_baseline"] == 0)
                        & (joined["hit_hawk"] == 1)
                    ).sum()
                ),
            },
            "trace_rows": {
                "baseline": len(baseline_dataset_traces),
                "hawk": len(traces),
            },
            "visual_tokens_original": stats(original_visual),
            "visual_tokens_kept": stats(kept_visual),
            "actual_keep_ratio": stats(ratio_actual),
            "actual_pruning_ratio_micro": 1 - sum(kept_visual) / sum(original_visual),
            "post_normalize_head_l2_norm": (
                {
                    "trace_rows": len(diagnostic_traces),
                    "min": min(l2_min),
                    "mean": statistics.fmean(l2_mean),
                    "max": max(l2_max),
                }
                if diagnostic_traces
                else {
                    "trace_rows": 0,
                    "status": "unavailable_in_legacy_trace",
                }
            ),
        }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Native-resolution VLMEvalKit results",
        "",
        "Configuration: Qwen2.5-VL-7B, SDPA, native aspect-ratio-preserving "
        "processing, 80% target pruning (`keep_ratio=0.20`).",
        "",
        "| Dataset | Samples | Baseline | HAWK | Delta | Paper baseline | Paper HAWK |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = report["datasets"][dataset]
        lines.append(
            f"| {dataset} | {row['samples']} | {row['baseline_accuracy']:.2f} | "
            f"{row['hawk_accuracy']:.2f} | {row['delta_pp']:+.2f} | "
            f"{row['paper']['baseline']:.1f} | {row['paper']['hawk_r080']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains per-dataset visual-token distributions, "
            "micro pruning ratios, answer transitions, empty-output checks, and "
            "post-normalization head L2 norms.",
        ]
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
