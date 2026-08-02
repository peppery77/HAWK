"""Core HAWK scoring and visual-token selection.

The implementation follows equations (4)-(7) in the HAWK paper and the
released Qwen2.5-VL inference path:

* project first-layer, pre-RoPE Q/K states;
* average text-to-vision scores over instruction tokens;
* L2-normalize each head's visual-token score vector;
* aggregate heads with the normalized static importance weights;
* preserve the original order of the selected visual tokens.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import HawkConfig


@dataclass(frozen=True)
class PruningStats:
    applied: bool
    original_total_tokens: int
    kept_total_tokens: int
    original_visual_tokens: int
    kept_visual_tokens: int
    removed_visual_tokens: int
    keep_ratio_requested: float
    keep_ratio_actual: float
    query_tokens: int
    score_normalization: str | None = None
    score_l2_norm_min: float | None = None
    score_l2_norm_mean: float | None = None
    score_l2_norm_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrunedPrefill:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    cache_position: torch.Tensor | None
    keep_indices: torch.Tensor
    stats: PruningStats


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat grouped-query K states without materializing intermediate copies."""

    batch, kv_heads, sequence_length, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence_length, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence_length, head_dim)


def compute_hawk_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    head_weights: torch.Tensor,
    normalization: str = "l2",
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return one importance score per visual token.

    Args:
        query_states: ``[1, heads, text_tokens, head_dim]``.
        key_states: ``[1, heads, visual_tokens, head_dim]``.
        head_weights: ``[heads]``.
    """

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("query_states and key_states must both be rank-4")
    if query_states.shape[:2] != key_states.shape[:2]:
        raise ValueError("query_states and key_states must share batch/head dimensions")
    if query_states.shape[0] != 1:
        raise ValueError("HAWK currently supports batch size 1")
    if query_states.shape[2] == 0 or key_states.shape[2] == 0:
        raise ValueError("HAWK requires at least one text query and one visual key")
    if head_weights.numel() != query_states.shape[1]:
        raise ValueError(
            f"expected {query_states.shape[1]} head weights, got {head_weights.numel()}"
        )

    head_dim = query_states.shape[-1]
    scores = torch.matmul(query_states, key_states.transpose(2, 3))
    scores = scores / math.sqrt(head_dim)
    scores = scores.mean(dim=2)

    # Match the reference float16 guard while also making bfloat16/float32
    # behavior deterministic in the presence of an invalid projection value.
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if normalization == "l2":
        scores = F.normalize(scores, p=2, dim=-1).float()
    elif normalization == "softmax":
        scores = F.softmax(scores, dim=-1).float()
    elif normalization == "minmax":
        scores = scores.float()
        score_min = scores.amin(dim=-1, keepdim=True)
        score_range = scores.amax(dim=-1, keepdim=True) - score_min
        scores = (scores - score_min) / score_range.clamp_min(
            torch.finfo(scores.dtype).eps
        )
    else:
        raise ValueError(
            f"normalization must be l2, softmax, or minmax; got {normalization}"
        )
    if diagnostics is not None:
        norms = torch.linalg.vector_norm(scores, ord=2, dim=-1)
        diagnostics.update(
            score_l2_norm_min=float(norms.min().item()),
            score_l2_norm_mean=float(norms.mean().item()),
            score_l2_norm_max=float(norms.max().item()),
        )

    weights = head_weights.to(device=scores.device, dtype=torch.float32).view(1, -1, 1)
    return (scores * weights).sum(dim=1).squeeze(0)


def select_keep_indices(
    visual_indices: torch.Tensor,
    visual_scores: torch.Tensor,
    sequence_length: int,
    keep_ratio: float,
    min_visual_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-scoring visual tokens and return ordered absolute indices."""

    if visual_indices.ndim != 1 or visual_scores.ndim != 1:
        raise ValueError("visual_indices and visual_scores must be one-dimensional")
    if visual_indices.numel() != visual_scores.numel():
        raise ValueError("visual_indices and visual_scores must have equal length")
    if visual_indices.numel() == 0:
        raise ValueError("no visual tokens were found")

    visual_count = visual_indices.numel()
    keep_count = max(min_visual_tokens, math.ceil(visual_count * keep_ratio))
    keep_count = min(visual_count, keep_count)
    relative = torch.topk(visual_scores, k=keep_count, sorted=False).indices
    selected_visual = visual_indices.index_select(0, relative).sort().values

    keep_mask = torch.ones(sequence_length, dtype=torch.bool, device=visual_indices.device)
    keep_mask[visual_indices] = False
    keep_mask[selected_visual] = True
    return torch.where(keep_mask)[0].long(), selected_visual


def _query_indices(
    input_ids: torch.Tensor,
    visual_indices: torch.Tensor,
    vision_end_token_id: int,
    im_end_token_id: int,
) -> torch.Tensor:
    """Locate instruction tokens after the final vision separator.

    This is the semantic range used by HAWK:
    ``vision_end + 1 : final_im_end``. If a custom template leaves that range
    empty, the vision-end separator itself is used as a safe fallback.
    """

    row = input_ids[0]
    vision_end = torch.where(row == vision_end_token_id)[0]
    if vision_end.numel() > 0:
        start = int(vision_end[-1].item()) + 1
    else:
        start = int(visual_indices[-1].item()) + 1

    later_im_end = torch.where((row == im_end_token_id) & (torch.arange(row.numel(), device=row.device) > start))[0]
    end = int(later_im_end[-1].item()) if later_im_end.numel() > 0 else row.numel()
    candidates = torch.arange(start, end, device=row.device)

    if candidates.numel() > 0:
        return candidates
    if vision_end.numel() > 0:
        return vision_end[-1:].long()
    raise ValueError("could not locate any text query tokens")


@torch.no_grad()
def prune_qwen_prefill(
    model: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    config: HawkConfig,
) -> PrunedPrefill:
    """Prune a Qwen2.5-VL prefill sequence and retain mRoPE position IDs."""

    if input_ids is None:
        raise ValueError("HAWK requires input_ids during multimodal prefill")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("HAWK currently supports one sample per generation call")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("HAWK expects a 2D attention mask during prefill")

    image_token_id = int(getattr(model.config, "image_token_id", 151655))
    video_token_id = int(getattr(model.config, "video_token_id", 151656))
    vision_end_token_id = int(getattr(model.config, "vision_end_token_id", 151653))

    row = input_ids[0]
    visual_mask = (row == image_token_id) | (row == video_token_id)
    if attention_mask is not None:
        visual_mask &= attention_mask[0].bool()
    visual_indices = torch.where(visual_mask)[0].long()
    sequence_length = inputs_embeds.shape[1]

    if visual_indices.numel() == 0:
        keep_indices = torch.arange(sequence_length, device=inputs_embeds.device)
        stats = PruningStats(
            applied=False,
            original_total_tokens=sequence_length,
            kept_total_tokens=sequence_length,
            original_visual_tokens=0,
            kept_visual_tokens=0,
            removed_visual_tokens=0,
            keep_ratio_requested=config.keep_ratio,
            keep_ratio_actual=1.0,
            query_tokens=0,
        )
        _store_state(model, keep_indices, stats)
        return PrunedPrefill(
            inputs_embeds, attention_mask, position_ids, cache_position, keep_indices, stats
        )

    if config.keep_ratio >= 1.0:
        keep_indices = torch.arange(sequence_length, device=inputs_embeds.device)
        visual_count = int(visual_indices.numel())
        stats = PruningStats(
            applied=False,
            original_total_tokens=sequence_length,
            kept_total_tokens=sequence_length,
            original_visual_tokens=visual_count,
            kept_visual_tokens=visual_count,
            removed_visual_tokens=0,
            keep_ratio_requested=config.keep_ratio,
            keep_ratio_actual=1.0,
            query_tokens=0,
        )
        _store_state(model, keep_indices, stats)
        return PrunedPrefill(
            inputs_embeds, attention_mask, position_ids, cache_position, keep_indices, stats
        )

    queries = _query_indices(
        input_ids, visual_indices, vision_end_token_id, config.im_end_token_id
    )
    layer0 = model.language_model.layers[0]
    attention = layer0.self_attn
    normalized = layer0.input_layernorm(inputs_embeds)

    batch_size, q_len, _ = normalized.shape
    num_heads = attention.num_heads
    num_kv_heads = attention.num_key_value_heads
    head_dim = attention.head_dim

    query_states = attention.q_proj(normalized)
    key_states = attention.k_proj(normalized)
    query_states = query_states.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, num_kv_heads, head_dim).transpose(1, 2)
    key_states = repeat_kv(key_states, attention.num_key_value_groups)

    text_queries = query_states.index_select(2, queries)
    visual_keys = key_states.index_select(2, visual_indices)
    weights = torch.as_tensor(config.head_weights, device=inputs_embeds.device)
    score_diagnostics: dict[str, float] = {}
    visual_scores = compute_hawk_scores(
        text_queries,
        visual_keys,
        weights,
        normalization=config.score_normalization,
        diagnostics=score_diagnostics,
    )
    keep_indices, selected_visual = select_keep_indices(
        visual_indices,
        visual_scores,
        sequence_length,
        config.keep_ratio,
        config.min_visual_tokens,
    )

    kept_visual = int(selected_visual.numel())
    original_visual = int(visual_indices.numel())
    kept_total = int(keep_indices.numel())
    stats = PruningStats(
        applied=kept_visual < original_visual,
        original_total_tokens=sequence_length,
        kept_total_tokens=kept_total,
        original_visual_tokens=original_visual,
        kept_visual_tokens=kept_visual,
        removed_visual_tokens=original_visual - kept_visual,
        keep_ratio_requested=config.keep_ratio,
        keep_ratio_actual=kept_visual / original_visual,
        query_tokens=int(queries.numel()),
        score_normalization=config.score_normalization,
        **score_diagnostics,
    )

    pruned_attention_mask = (
        attention_mask.index_select(1, keep_indices.to(attention_mask.device))
        if attention_mask is not None
        else None
    )
    pruned_position_ids = (
        position_ids.index_select(2, keep_indices.to(position_ids.device))
        if position_ids is not None
        else None
    )
    pruned_cache_position = (
        cache_position[:kept_total] if cache_position is not None else None
    )
    _store_state(model, keep_indices, stats)

    return PrunedPrefill(
        inputs_embeds=inputs_embeds.index_select(1, keep_indices),
        attention_mask=pruned_attention_mask,
        position_ids=pruned_position_ids,
        cache_position=pruned_cache_position,
        keep_indices=keep_indices,
        stats=stats,
    )


def _store_state(model: Any, keep_indices: torch.Tensor, stats: PruningStats) -> None:
    model._hawk_generation_state = {
        "keep_indices": keep_indices,
        "stats": stats.to_dict(),
        "consumed": False,
    }
    model._hawk_last_stats = stats.to_dict()
