#!/usr/bin/env python3
"""Run VLMEvalKit after registering the project-local HAWK adapters."""

from __future__ import annotations

import os


def bind_one_gpu_before_torch_import() -> None:
    """Give each torchrun worker one physical GPU before importing torch."""

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if local_world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    visible = [device.strip() for device in visible if device.strip()]
    if len(visible) < local_world_size:
        raise RuntimeError(
            f"need {local_world_size} visible GPUs, got CUDA_VISIBLE_DEVICES={visible}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    # Prevent VLMEvalKit from remapping the already isolated worker again.
    os.environ["LOCAL_WORLD_SIZE"] = "1"


def main() -> None:
    bind_one_gpu_before_torch_import()

    import torch

    torch.cuda.set_device(0)
    import vlmeval.dataset
    import vlmeval.vlm
    from hawk.vlmeval_adapter import (
        HawkLocalImageMCQDataset,
        HawkLocalImageVQADataset,
        HawkLocalImageYORNDataset,
        HawkQwen2VLChat,
    )
    from vlmeval import run

    vlmeval.vlm.HawkQwen2VLChat = HawkQwen2VLChat
    vlmeval.dataset.HawkLocalImageMCQDataset = HawkLocalImageMCQDataset
    vlmeval.dataset.HawkLocalImageVQADataset = HawkLocalImageVQADataset
    vlmeval.dataset.HawkLocalImageYORNDataset = HawkLocalImageYORNDataset
    run.load_env()
    args = run.parse_args()
    if args.config is None:
        assert args.data, "--data should be a list of data files"
    run.run_task(args)


if __name__ == "__main__":
    main()
