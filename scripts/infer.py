#!/usr/bin/env python3
"""Run one-image Qwen2.5-VL inference with HAWK pruning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from hawk import HawkConfig, configure_model, get_last_pruning_stats


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Describe the image carefully and identify the most important objects.",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.198)
    parser.add_argument("--min-visual-tokens", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dtype = DTYPES[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(args.device).eval()
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)
    configure_model(
        model,
        HawkConfig(
            keep_ratio=args.keep_ratio,
            min_visual_tokens=args.min_visual_tokens,
        ),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(args.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started

    stats = get_last_pruning_stats(model)
    prompt_length = (
        int(stats["kept_total_tokens"]) if stats else int(inputs.input_ids.shape[1])
    )
    generated_only = generated_ids[:, prompt_length:]
    response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    peak_gib = (
        torch.cuda.max_memory_allocated(args.device) / (1024**3)
        if args.device.startswith("cuda")
        else None
    )

    report = {
        "model_path": str(model_path),
        "image": str(image_path),
        "device": args.device,
        "dtype": args.dtype,
        "elapsed_seconds": round(elapsed, 4),
        "peak_allocated_gib": round(peak_gib, 4) if peak_gib is not None else None,
        "original_prompt_tokens": int(inputs.input_ids.shape[1]),
        "kept_prompt_tokens": prompt_length,
        "generated_sequence_tokens": int(generated_ids.shape[1]),
        "generated_new_tokens": int(generated_ids.shape[1]) - prompt_length,
        "pruning": stats,
        "response": response,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
