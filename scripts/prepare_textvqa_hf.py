#!/usr/bin/env python3
"""Prepare the HF-mirror TextVQA validation split for local VLMEvalKit use."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "lmms-lab-encoder/textvqa"
VALIDATION_SHARDS = (
    "data/validation-00000-of-00003.parquet",
    "data/validation-00001-of-00003.parquet",
    "data/validation-00002-of-00003.parquet",
)


def image_suffix(payload: bytes, source_path: str | None = None) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith((b"RIFF",)) and payload[8:12] == b"WEBP":
        return ".webp"
    suffix = Path(source_path or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    raise ValueError("unsupported TextVQA image encoding")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    download_root = project_root / ".cache" / "textvqa-hf"
    output_root = project_root / "data" / "vlmeval"
    image_root = output_root / "images" / "TextVQA_VAL"
    download_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    shard_paths = [
        Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=download_root,
                endpoint=args.endpoint,
            )
        )
        for filename in VALIDATION_SHARDS
    ]
    source = pd.concat(
        [pd.read_parquet(path) for path in shard_paths], ignore_index=True
    )
    required = {"image_id", "question_id", "question", "image", "answers"}
    if not required.issubset(source.columns):
        raise ValueError(f"missing columns: {sorted(required - set(source.columns))}")
    if len(source) != 5000:
        raise ValueError(f"expected 5000 validation questions, got {len(source)}")

    image_paths: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for row_number, row in source.iterrows():
        image = row["image"]
        if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
            raise TypeError(f"row {row_number} has no embedded image bytes")
        payload = image["bytes"]
        image_id = str(row["image_id"])
        digest = hashlib.sha256(payload).hexdigest()[:16]
        suffix = image_suffix(payload, image.get("path"))
        target = image_root / f"{image_id}_{digest}{suffix}"
        if image_id not in image_paths:
            if not target.exists():
                target.write_bytes(payload)
            image_paths[image_id] = str(target)

        answers = [str(answer) for answer in row["answers"]]
        if len(answers) != 10:
            raise ValueError(
                f"question {row['question_id']} has {len(answers)} answers, expected 10"
            )
        records.append(
            {
                "index": row_number,
                "image_path": image_paths[image_id],
                "question": str(row["question"]),
                "answer": repr(answers),
                "split": "validation",
                "question_id": int(row["question_id"]),
                "image_id": image_id,
            }
        )

    table = pd.DataFrame.from_records(records)
    output = output_root / "TextVQA_VAL_local.tsv"
    table.to_csv(output, sep="\t", index=False)
    print(f"wrote {len(table)} questions and {len(image_paths)} images")
    print(output)


if __name__ == "__main__":
    main()
