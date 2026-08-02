import pytest

from hawk.config import (
    HawkConfig,
    QWEN2_5_VL_7B_HEAD_WEIGHTS,
    QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY,
)


def test_head_weight_count() -> None:
    assert len(QWEN2_5_VL_7B_HEAD_WEIGHTS) == 28
    assert len(QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY) == 28


def test_default_head_weights_are_l1_normalized() -> None:
    assert sum(QWEN2_5_VL_7B_HEAD_WEIGHTS) == pytest.approx(1.0)
    assert QWEN2_5_VL_7B_HEAD_WEIGHTS[1] == pytest.approx(0.3 / 1.2616)
    assert QWEN2_5_VL_7B_HEAD_WEIGHTS[26] == 0.0


def test_normalization_preserves_head_ratios() -> None:
    for raw, normalized in zip(
        QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY,
        QWEN2_5_VL_7B_HEAD_WEIGHTS,
        strict=True,
    ):
        assert normalized == pytest.approx(raw / 1.2616)


@pytest.mark.parametrize("ratio", [0.0, -0.2, 1.01])
def test_invalid_keep_ratio(ratio: float) -> None:
    with pytest.raises(ValueError):
        HawkConfig(keep_ratio=ratio)


def test_paper_default_ratio() -> None:
    assert HawkConfig().keep_ratio == 0.198


def test_invalid_score_normalization() -> None:
    with pytest.raises(ValueError):
        HawkConfig(score_normalization="unknown")
