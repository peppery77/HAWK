"""Public integration helpers for patched Qwen2.5-VL models."""

from __future__ import annotations

from typing import Any

from .config import HawkConfig


def configure_model(model: Any, config: HawkConfig | None = None) -> Any:
    """Enable HAWK on a loaded ``Qwen2_5_VLForConditionalGeneration`` model."""

    config = config or HawkConfig()
    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "language_model"):
        raise TypeError("expected a Qwen2.5-VL conditional-generation model")

    num_heads = inner_model.language_model.layers[0].self_attn.num_heads
    if len(config.head_weights) != num_heads:
        raise ValueError(
            f"HAWK received {len(config.head_weights)} weights, but the model has {num_heads} heads"
        )

    inner_model._hawk_config = config
    inner_model._hawk_generation_state = None
    inner_model._hawk_last_stats = None
    return model


def disable_model(model: Any) -> Any:
    """Disable pruning without unloading the model."""

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        inner_model._hawk_config = HawkConfig(enabled=False)
        inner_model._hawk_generation_state = None
    return model


def get_last_pruning_stats(model: Any) -> dict[str, Any] | None:
    """Return JSON-serializable statistics from the last prefill."""

    inner_model = getattr(model, "model", None)
    return getattr(inner_model, "_hawk_last_stats", None)
