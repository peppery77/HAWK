#!/usr/bin/env python3
"""Apply the small, idempotent VLMEvalKit compatibility patch used by HAWK."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


MARKER = "# HAWK_VLMEVAL_ATTN_IMPLEMENTATION"
ANCHOR = """\
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
            )
"""
REPLACEMENT = """\
            # HAWK_VLMEVAL_ATTN_IMPLEMENTATION
            self.model = MODEL_CLS.from_pretrained(
                model_path,
                torch_dtype='auto',
                device_map="auto",
                attn_implementation=kwargs.get('attn_implementation', 'flash_attention_2'),
            )
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    target = args.root.resolve() / "vlmeval/vlm/qwen2_vl/model.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print("already-patched")
        return
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"expected one Qwen2.5-VL load anchor in {target}")
    if args.check:
        print("would-patch")
        return
    target.write_text(text.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    print("patched")


if __name__ == "__main__":
    main()
