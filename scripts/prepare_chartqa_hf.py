#!/usr/bin/env python3
"""Prepare the Hugging Face ChartQA parquet as a local VLMEvalKit TSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "lmms-lab-encoder/chartqa"
TEST_PARQUET = "data/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    raise ValueError("unsupported ChartQA image encoding")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if args.parquet is None:
        parquet = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=TEST_PARQUET,
                repo_type="dataset",
                local_dir=project_root / ".cache" / "chartqa-hf",
            )
        )
    else:
        parquet = args.parquet.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "data" / "vlmeval"
    )
    image_root = output_root / "images" / "ChartQA_TEST"
    image_root.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(parquet)
    required = {"question", "answer", "image"}
    if not required.issubset(source.columns):
        raise ValueError(f"missing columns: {sorted(required - set(source.columns))}")

    records = []
    for index, row in source.reset_index(drop=True).iterrows():
        image = row["image"]
        payload = image["bytes"] if isinstance(image, dict) else None
        if not isinstance(payload, bytes):
            raise TypeError(f"row {index} has no embedded image bytes")
        digest = hashlib.sha256(payload).hexdigest()[:20]
        path = image_root / f"{digest}{image_suffix(payload)}"
        if not path.exists():
            path.write_bytes(payload)
        records.append(
            {
                "index": index,
                "image_path": str(path),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "split": str(row.get("type", "test")),
            }
        )

    table = pd.DataFrame.from_records(records)
    output = output_root / "ChartQA_TEST_local.tsv"
    table.to_csv(output, sep="\t", index=False)
    print(f"wrote {len(table)} rows and {table.image_path.nunique()} images")
    print(output)


if __name__ == "__main__":
    main()
