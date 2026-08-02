import math

import torch

from hawk.pruning import compute_hawk_scores, select_keep_indices


def test_compute_scores_matches_manual_aggregation() -> None:
    queries = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]]
    )
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]]]
    )
    weights = torch.tensor([0.25, 0.75])
    diagnostics = {}

    actual = compute_hawk_scores(queries, keys, weights, diagnostics=diagnostics)
    raw = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(2)
    per_head = torch.nn.functional.normalize(raw.mean(dim=2), p=2, dim=-1)
    expected = (per_head * weights.view(1, 2, 1)).sum(dim=1).squeeze(0)
    torch.testing.assert_close(actual, expected)
    assert math.isclose(diagnostics["score_l2_norm_min"], 1.0, abs_tol=1e-6)
    assert math.isclose(diagnostics["score_l2_norm_mean"], 1.0, abs_tol=1e-6)
    assert math.isclose(diagnostics["score_l2_norm_max"], 1.0, abs_tol=1e-6)


def test_softmax_score_normalization() -> None:
    queries = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    weights = torch.tensor([0.4, 0.6])
    actual = compute_hawk_scores(
        queries,
        keys,
        weights,
        normalization="softmax",
    )
    raw = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(2)
    expected = (
        torch.softmax(raw.mean(dim=2), dim=-1)
        * weights.view(1, 2, 1)
    ).sum(dim=1).squeeze(0)
    torch.testing.assert_close(actual, expected)


def test_minmax_score_normalization() -> None:
    queries = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    weights = torch.tensor([0.4, 0.6])
    actual = compute_hawk_scores(
        queries,
        keys,
        weights,
        normalization="minmax",
    )
    raw = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(2)
    raw = raw.mean(dim=2)
    minimum = raw.amin(dim=-1, keepdim=True)
    expected = (raw - minimum) / (
        raw.amax(dim=-1, keepdim=True) - minimum
    )
    expected = (
        expected * weights.view(1, 2, 1)
    ).sum(dim=1).squeeze(0)
    torch.testing.assert_close(actual, expected)


def test_selection_keeps_nonvisual_tokens_and_original_order() -> None:
    visual_indices = torch.tensor([2, 3, 5, 7])
    visual_scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
    keep_indices, selected_visual = select_keep_indices(
        visual_indices,
        visual_scores,
        sequence_length=9,
        keep_ratio=0.5,
    )

    assert selected_visual.tolist() == [3, 5]
    assert keep_indices.tolist() == [0, 1, 3, 4, 5, 6, 8]


def test_selection_uses_ceil_and_minimum() -> None:
    visual_indices = torch.tensor([1, 2, 3])
    visual_scores = torch.tensor([0.2, 0.3, 0.1])
    _, selected_visual = select_keep_indices(
        visual_indices,
        visual_scores,
        sequence_length=5,
        keep_ratio=0.01,
    )
    assert selected_visual.tolist() == [2]
