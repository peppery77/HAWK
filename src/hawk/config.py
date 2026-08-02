"""Configuration for HAWK visual-token pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# Raw Qwen2.5-VL-7B head-importance vector. Equation (3) in the paper defines
# head importance through L1-normalized per-dataset scores, so the public
# runtime uses the sum-to-one vector derived from these values below.
QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY = (
    0.0875,
    0.3000,
    0.0480,
    0.0321,
    0.0104,
    0.0102,
    0.0336,
    0.0460,
    0.0540,
    0.0472,
    0.0234,
    0.0537,
    0.0109,
    0.0573,
    0.0276,
    0.0403,
    0.0429,
    0.0352,
    0.0162,
    0.0519,
    0.0553,
    0.0266,
    0.0222,
    0.0480,
    0.0337,
    0.0250,
    0.0000,
    0.0224,
)

_QWEN2_5_VL_7B_HEAD_WEIGHT_SUM = sum(QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY)
QWEN2_5_VL_7B_HEAD_WEIGHTS = tuple(
    weight / _QWEN2_5_VL_7B_HEAD_WEIGHT_SUM
    for weight in QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY
)


@dataclass(frozen=True)
class HawkConfig:
    """Runtime settings for training-free HAWK pruning.

    ``keep_ratio`` is the fraction of visual tokens retained. The fixed 1008
    paper setting uses 0.198; the corresponding native-resolution setting
    uses 0.20.
    """

    keep_ratio: float = 0.198
    head_weights: Sequence[float] = QWEN2_5_VL_7B_HEAD_WEIGHTS
    score_normalization: str = "l2"
    min_visual_tokens: int = 1
    im_end_token_id: int = 151645
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least 1")
        if not self.head_weights:
            raise ValueError("head_weights cannot be empty")
        if self.score_normalization not in {"l2", "softmax", "minmax"}:
            raise ValueError(
                "score_normalization must be one of: l2, softmax, minmax"
            )
