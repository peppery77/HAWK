"""VLMEvalKit adapters for HAWK and locally prepared benchmark metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from .config import HawkConfig
from .integration import configure_model, get_last_pruning_stats


class HawkQwen2VLChat:
    """Factory-compatible HAWK wrapper around VLMEvalKit's Qwen2.5-VL class.

    The concrete base class is resolved lazily so importing :mod:`hawk` does not
    require VLMEvalKit to be installed.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

        class _ConfiguredHawkQwen2VLChat(Qwen2VLChat):
            def __init__(
                self,
                *model_args: Any,
                hawk_keep_ratio: float = 0.2,
                hawk_head_weights: list[float] | tuple[float, ...] | None = None,
                hawk_score_normalization: str = "l2",
                hawk_min_visual_tokens: int = 1,
                hawk_trace_path: str | None = None,
                **model_kwargs: Any,
            ) -> None:
                self._hawk_keep_ratio = hawk_keep_ratio
                self._hawk_trace_path = hawk_trace_path
                super().__init__(*model_args, **model_kwargs)
                # This VLMEvalKit release passes device_map="auto", but the
                # pinned Transformers runtime leaves the model on CPU. Each
                # torchrun worker sees exactly one GPU, so move the complete
                # model there explicitly and avoid CPU offload.
                self.model.to("cuda")
                configure_model(
                    self.model,
                    HawkConfig(
                        keep_ratio=hawk_keep_ratio,
                        **(
                            {"head_weights": tuple(hawk_head_weights)}
                            if hawk_head_weights is not None
                            else {}
                        ),
                        score_normalization=hawk_score_normalization,
                        min_visual_tokens=hawk_min_visual_tokens,
                    ),
                )

            def _restore_original_prompt_for_vlmeval(
                self, original_generate: Any, *generate_args: Any, **generate_kwargs: Any
            ) -> torch.Tensor:
                generated = original_generate(*generate_args, **generate_kwargs)
                stats = get_last_pruning_stats(self.model)
                input_ids = generate_kwargs.get("input_ids")
                if (
                    stats is None
                    or input_ids is None
                    or generated.ndim != 2
                    or input_ids.ndim != 2
                ):
                    return generated

                kept_prompt = int(stats["kept_total_tokens"])
                generated_only = generated[:, kept_prompt:]
                return torch.cat(
                    [input_ids.to(generated.device), generated_only],
                    dim=1,
                )

            def generate_inner_transformers(
                self, message: list[dict[str, str]], dataset: str | None = None
            ) -> str:
                original_generate = self.model.generate

                def generate_with_original_prompt(
                    *generate_args: Any, **generate_kwargs: Any
                ) -> torch.Tensor:
                    return self._restore_original_prompt_for_vlmeval(
                        original_generate, *generate_args, **generate_kwargs
                    )

                self.model.generate = generate_with_original_prompt
                try:
                    response = super().generate_inner_transformers(
                        message,
                        dataset=dataset,
                    )
                finally:
                    self.model.generate = original_generate

                self._write_hawk_trace(message, dataset, response)
                return response

            def _write_hawk_trace(
                self,
                message: list[dict[str, str]],
                dataset: str | None,
                response: str,
            ) -> None:
                if not self._hawk_trace_path:
                    return
                rank = int(os.environ.get("RANK", "0"))
                trace_base = Path(self._hawk_trace_path)
                trace_path = trace_base.with_name(
                    f"{trace_base.name}.rank{rank}.jsonl"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                text = "\n".join(
                    str(part["value"])
                    for part in message
                    if part.get("type") == "text"
                )
                images = [
                    str(part["value"])
                    for part in message
                    if part.get("type") == "image"
                ]
                record = {
                    "dataset": dataset,
                    "question_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "images": images,
                    "response": response,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "pruning": get_last_pruning_stats(self.model),
                }
                with trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        return _ConfiguredHawkQwen2VLChat(*args, **kwargs)


class HawkLocalImageMCQDataset:
    """Factory for an ImageMCQDataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        # Not reached because __new__ returns the concrete VLMEvalKit dataset.
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:
        from vlmeval.dataset.image_mcq import ImageMCQDataset
        from vlmeval.smp import load

        class _LocalImageMCQDataset(ImageMCQDataset):
            def __init__(self) -> None:
                self._hawk_data_file = str(Path(data_file).expanduser().resolve())
                super().__init__(dataset=dataset, skip_noimg=skip_noimg)

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                return load(self._hawk_data_file)

        return _LocalImageMCQDataset()


class HawkLocalImageVQADataset:
    """Factory for an ImageVQADataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:
        from vlmeval.dataset.image_vqa import ImageVQADataset
        from vlmeval.smp import load

        class _LocalImageVQADataset(ImageVQADataset):
            def __init__(self) -> None:
                self._hawk_data_file = str(Path(data_file).expanduser().resolve())
                super().__init__(dataset=dataset, skip_noimg=skip_noimg)

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                return load(self._hawk_data_file)

        return _LocalImageVQADataset()


class HawkLocalImageYORNDataset:
    """Factory for an ImageYORNDataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:
        from vlmeval.dataset.image_yorn import ImageYORNDataset
        from vlmeval.smp import load

        class _LocalImageYORNDataset(ImageYORNDataset):
            def __init__(self) -> None:
                self._hawk_data_file = str(Path(data_file).expanduser().resolve())
                super().__init__(dataset=dataset, skip_noimg=skip_noimg)

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                return load(self._hawk_data_file)

        return _LocalImageYORNDataset()
