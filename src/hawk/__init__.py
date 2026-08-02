"""HAWK: head importance-aware visual-token pruning."""

from .config import (
    HawkConfig,
    QWEN2_5_VL_7B_HEAD_WEIGHTS,
    QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY,
)
from .integration import configure_model, disable_model, get_last_pruning_stats

__all__ = [
    "HawkConfig",
    "QWEN2_5_VL_7B_HEAD_WEIGHTS",
    "QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY",
    "configure_model",
    "disable_model",
    "get_last_pruning_stats",
]

__version__ = "0.1.0"
