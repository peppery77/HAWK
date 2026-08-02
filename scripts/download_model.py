#!/usr/bin/env python3
"""Download Qwen2.5-VL-7B through HF Mirror."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    if endpoint.rstrip("/") != "https://hf-mirror.com":
        raise ValueError(
            "This release script uses https://hf-mirror.com; "
            f"received HF_ENDPOINT={endpoint!r}"
        )
    os.environ["HF_ENDPOINT"] = endpoint

    destination = args.local_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        max_workers=args.max_workers,
        ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.onnx", "*.tflite"],
    )
    print(result)


if __name__ == "__main__":
    main()
