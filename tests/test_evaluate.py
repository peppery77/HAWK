import argparse

import pytest

from scripts.evaluate import keep_ratio_from_pruning, normalize_task


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RealWorldQA", "realworldqa"),
        ("ChartQA_TEST", "chartqa"),
        ("TextVQA_VAL", "textvqa"),
        ("MME", "mme"),
    ],
)
def test_normalize_task(value: str, expected: str) -> None:
    assert normalize_task(value) == expected


def test_normalize_task_rejects_unknown_task() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        normalize_task("unknown")


@pytest.mark.parametrize(
    ("pruning_ratio", "keep_ratio"),
    [(0.0, 1.0), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)],
)
def test_keep_ratio_from_pruning(
    pruning_ratio: float, keep_ratio: float
) -> None:
    assert keep_ratio_from_pruning(pruning_ratio) == pytest.approx(keep_ratio)


@pytest.mark.parametrize("pruning_ratio", [-0.1, 1.0])
def test_keep_ratio_rejects_invalid_values(pruning_ratio: float) -> None:
    with pytest.raises(ValueError):
        keep_ratio_from_pruning(pruning_ratio)
