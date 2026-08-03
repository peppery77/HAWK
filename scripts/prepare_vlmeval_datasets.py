#!/usr/bin/env python3
"""Prepare RealWorldQA and ScienceQA for project-local VLMEvalKit runs."""

from __future__ import annotations

import argparse
import ast
import string
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


REALWORLD_REPO = "lmms-lab/RealWorldQA"
REALWORLD_METADATA_REPO = "jefehern/vlmevalkit_inference"
REALWORLD_METADATA = (
    "InternVL-Chat-V1-5/InternVL-Chat-V1-5_RealWorldQA.xlsx"
)
SCIENCEQA_REPO = "lmms-lab/ScienceQA"
SCIENCEQA_CONFIG = "ScienceQA-FULL"
SUPPORTED_DATASETS = ("RealWorldQA", "ScienceQA_TEST")


def image_suffix(image: Any, source_path: object, default: str) -> str:
    suffix = Path(str(source_path or "")).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    image_format = str(getattr(image, "format", "") or "").lower()
    return f".{image_format}" if image_format else default


def save_image(image: Any, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        image.save(target)
    return str(target.resolve())


def canonical_question(text: object) -> str:
    return "".join(character for character in str(text).lower() if character.isalnum())


def prepare_realworld(cache_root: Path, output_root: Path) -> Path:
    source: Dataset = load_dataset(
        REALWORLD_REPO,
        split="test",
        cache_dir=str(cache_root),
    )
    if len(source) != 765:
        raise ValueError(f"expected 765 RealWorldQA rows, got {len(source)}")

    metadata_root = cache_root.parents[1] / "vlmeval-bootstrap"
    metadata_path = hf_hub_download(
        repo_id=REALWORLD_METADATA_REPO,
        filename=REALWORLD_METADATA,
        repo_type="dataset",
        local_dir=metadata_root,
    )
    metadata = pd.read_excel(metadata_path).drop(
        columns=["prediction"], errors="ignore"
    )
    if len(metadata) != len(source):
        raise ValueError(
            f"RealWorldQA row mismatch: {len(source)} != {len(metadata)}"
        )

    image_root = output_root / "images" / "RealWorldQA"
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(source):
        meta_row = metadata.iloc[row_number]
        if not canonical_question(row["question"]).startswith(
            canonical_question(meta_row["question"])
        ):
            raise ValueError(f"RealWorldQA question mismatch at row {row_number}")
        image = row["image"]
        suffix = image_suffix(image, row.get("image_path"), ".webp")
        image_path = save_image(
            image,
            image_root / f"{meta_row['index']}{suffix}",
        )
        record = meta_row.to_dict()
        record["image_path"] = image_path
        records.append(record)

    output_path = output_root / "RealWorldQA_local.tsv"
    pd.DataFrame.from_records(records).to_csv(output_path, sep="\t", index=False)
    return output_path


def parse_choices(value: object) -> list[str]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a choice list, got {type(value).__name__}")
    return [str(choice) for choice in value]


def prepare_scienceqa(cache_root: Path, output_root: Path) -> Path:
    source: Dataset = load_dataset(
        SCIENCEQA_REPO,
        SCIENCEQA_CONFIG,
        split="test",
        cache_dir=str(cache_root),
    )
    if len(source) != 4241:
        raise ValueError(f"expected 4241 ScienceQA test rows, got {len(source)}")

    image_root = output_root / "images" / "ScienceQA_TEST"
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(source):
        if row["image"] is None:
            continue
        # VLMEvalKit uses the one-based index of the unfiltered ScienceQA test
        # split. Loading ScienceQA-IMG directly would lose that stable index.
        source_index = row_number + 1
        choices = parse_choices(row["choices"])
        answer_index = int(row["answer"])
        if answer_index < 0 or answer_index >= len(choices):
            raise ValueError(
                f"row {source_index} has invalid answer index {answer_index}"
            )
        image = row["image"]
        suffix = image_suffix(image, None, ".png")
        image_path = save_image(image, image_root / f"{source_index}{suffix}")
        option_columns = {
            string.ascii_uppercase[choice_index]: choice
            for choice_index, choice in enumerate(choices)
        }
        records.append(
            {
                "question": str(row["question"]),
                "answer": string.ascii_uppercase[answer_index],
                "hint": str(row.get("hint", "")),
                "task": str(row.get("task", "")),
                "grade": str(row.get("grade", "")),
                "subject": str(row.get("subject", "")),
                "topic": str(row.get("topic", "")),
                "category": str(row.get("category", "")),
                "skill": str(row.get("skill", "")),
                "lecture": str(row.get("lecture", "")),
                "solution": str(row.get("solution", "")),
                "index": source_index,
                "split": "test",
                **option_columns,
                "image_path": image_path,
            }
        )

    if len(records) != 2017:
        raise ValueError(
            f"expected 2017 ScienceQA image rows, got {len(records)}"
        )

    output_path = output_root / "ScienceQA_TEST_local.tsv"
    pd.DataFrame.from_records(records).to_csv(output_path, sep="\t", index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    cache_root = project_root / ".cache" / "huggingface" / "datasets"
    output_root = project_root / "data" / "vlmeval"
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    preparers = {
        "RealWorldQA": prepare_realworld,
        "ScienceQA_TEST": prepare_scienceqa,
    }
    for dataset in args.datasets:
        output = preparers[dataset](cache_root, output_root)
        frame = pd.read_csv(output, sep="\t")
        print(f"{output}: {len(frame)} rows")


if __name__ == "__main__":
    main()
