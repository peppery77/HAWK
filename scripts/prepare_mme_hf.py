#!/usr/bin/env python3
"""Prepare the HF-mirror MME test split for local VLMEvalKit use."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "lmms-lab-encoder/MME"
TEST_SHARDS = (
    "data/test-00000-of-00004-a25dbe3b44c4fda6.parquet",
    "data/test-00001-of-00004-7d22c7f1aba6fca4.parquet",
    "data/test-00002-of-00004-594798fd3f5b029c.parquet",
    "data/test-00003-of-00004-53ae1794f93b1e35.parquet",
)


def image_suffix(payload: bytes, source_path: str | None = None) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    suffix = Path(source_path or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    raise ValueError("unsupported MME image encoding")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    download_root = project_root / ".cache" / "mme-hf"
    output_root = project_root / "data" / "vlmeval"
    image_root = output_root / "images" / "MME"
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
        for filename in TEST_SHARDS
    ]
    source = pd.concat(
        [pd.read_parquet(path) for path in shard_paths], ignore_index=True
    )
    required = {"question_id", "image", "question", "answer", "category"}
    if not required.issubset(source.columns):
        raise ValueError(f"missing columns: {sorted(required - set(source.columns))}")
    if len(source) != 2374:
        raise ValueError(f"expected 2374 MME questions, got {len(source)}")

    image_paths: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for row_number, row in source.iterrows():
        image = row["image"]
        if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
            raise TypeError(f"row {row_number} has no embedded image bytes")
        payload = image["bytes"]
        digest = hashlib.sha256(payload).hexdigest()
        suffix = image_suffix(payload, image.get("path"))
        target = image_root / f"{digest[:20]}{suffix}"
        if digest not in image_paths:
            if not target.exists():
                target.write_bytes(payload)
            image_paths[digest] = str(target)

        answer = str(row["answer"]).strip().capitalize()
        if answer not in {"Yes", "No"}:
            raise ValueError(f"unexpected MME answer at row {row_number}: {answer}")
        records.append(
            {
                "index": row_number,
                "image_path": image_paths[digest],
                "question": str(row["question"]),
                "answer": answer,
                "category": str(row["category"]),
                "question_id": str(row["question_id"]),
            }
        )

    table = pd.DataFrame.from_records(records)
    pair_counts = Counter(zip(table["category"], table["image_path"]))
    if set(pair_counts.values()) != {2}:
        raise ValueError("MME requires exactly two questions per category/image pair")
    output = output_root / "MME_local.tsv"
    table.to_csv(output, sep="\t", index=False)
    print(
        f"wrote {len(table)} questions, {len(image_paths)} images, "
        f"and {table['category'].nunique()} categories"
    )
    print(output)


if __name__ == "__main__":
    main()
