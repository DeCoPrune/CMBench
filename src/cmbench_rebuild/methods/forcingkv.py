"""ForcingKV's method-specific online cache update.

The shared runtime calls :func:`update_forcingkv_cache` after each clean
causal chunk.  This module owns the complete ForcingKV rule: score adjacent
patches, keep the lowest-similarity memory patches, and compact the static and
dynamic head banks.  Model execution and attention remain runtime concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.budgets import forcingkv_middle_window_start_frame
from .selectors import lowest_similarity_indices
from .spec import MethodSpec
from .torch_kernels import forcingkv_adjacent_patch_moments, forcingkv_scores_from_moments


SPEC = MethodSpec(
    "forcingkv",
    "ForcingKV",
    "forcingkv_reindex",
    "forcingkv",
    {
        "static_sink_frames": 4,
        "static_recent_chunks": 1,
        "dynamic_sink_frames": 4,
        "dynamic_recent_chunks": 1,
        "selected_patch_count": 256,
        "middle_window_chunks": 32,
        "middle_candidate_scope": "local-boundary",
        "layer_zero_policy": "streaming",
    },
)


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


@dataclass(frozen=True)
class ForcingKVUpdate:
    """State carried from one ForcingKV chunk to the next."""

    scores: tuple[float, ...]
    patch_starts: tuple[int, ...]
    event: dict[str, Any]


def update_forcingkv_cache(
    *,
    cache: list[dict[str, Any]],
    geometry: Any,
    parameters: Mapping[str, Any],
    chunk_idx: int,
    phase: str,
    previous_scores: Sequence[float],
    previous_patch_starts: Sequence[int],
) -> ForcingKVUpdate:
    """Apply one ForcingKV transition and return its small persistent state."""
    import torch
    import torch.distributed as dist
    from ..runtime.compaction import keep_global_source_ids

    frame_tokens = int(geometry.frame_tokens)
    chunk_size = int(geometry.chunk_size)
    patch_count = 6
    patch_tokens = frame_tokens // patch_count
    reference = cache[1 if len(cache) > 1 else 0]
    live_end = _int(reference["dynamic_local_end_index"])
    source_ids = reference["dynamic_source_ids"][0, :live_end]

    current_start = int(chunk_idx) * chunk_size * frame_tokens
    current_ids = torch.arange(
        current_start,
        current_start + chunk_size * frame_tokens,
        device=source_ids.device,
    )
    predecessor_ids = torch.arange(
        current_start - frame_tokens,
        current_start,
        device=source_ids.device,
    )
    has_predecessor = current_start >= frame_tokens and bool(
        torch.isin(predecessor_ids, source_ids).all()
    )

    new_scores: list[float] = []
    new_patch_starts: list[int] = []
    if has_predecessor:
        predecessor_positions = torch.searchsorted(source_ids, predecessor_ids)
        current_positions = torch.searchsorted(source_ids, current_ids)
        local_heads = int(reference["dynamic_k"].shape[2])
        if local_heads:
            predecessor_k = reference["dynamic_k"][0].index_select(0, predecessor_positions)
            current_k = reference["dynamic_k"][0].index_select(0, current_positions)
            current_k = current_k.reshape(chunk_size, frame_tokens, local_heads, -1)
            moments = forcingkv_adjacent_patch_moments(
                predecessor_k,
                current_k,
                num_patches=patch_count,
            )
        else:
            shape = (chunk_size, patch_count, patch_tokens)
            moments = tuple(
                torch.zeros(shape, dtype=torch.float32, device=source_ids.device)
                for _ in range(3)
            )
        if dist.is_available() and dist.is_initialized():
            for moment in moments:
                dist.all_reduce(moment, op=dist.ReduceOp.SUM)
        score_values = (
            forcingkv_scores_from_moments(*moments)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        new_scores = [float(value) for value in score_values]
        for patch_index in range(len(new_scores)):
            source_frame = int(chunk_idx) * chunk_size - 1 + patch_index // patch_count
            new_patch_starts.append(
                source_frame * frame_tokens + (patch_index % patch_count) * patch_tokens
            )

    window_start = forcingkv_middle_window_start_frame(
        chunk_index=int(chunk_idx),
        chunk_size=chunk_size,
        dynamic_recent_chunks=int(parameters["dynamic_recent_chunks"]),
        middle_window_chunks=int(parameters["middle_window_chunks"]),
    )
    candidates = [
        (float(score), int(start))
        for score, start in zip(previous_scores, previous_patch_starts)
        if int(start) // frame_tokens >= window_start
    ]
    candidates.extend(zip(new_scores, new_patch_starts))
    keep_count = min(int(parameters["selected_patch_count"]), len(candidates))
    selected_indices = lowest_similarity_indices(
        [score for score, _ in candidates],
        keep_count=keep_count,
    )
    selected = [candidates[index] for index in selected_indices]
    scores = tuple(score for score, _ in selected)
    patch_starts = tuple(start for _, start in selected)

    frames_seen = (int(chunk_idx) + 1) * chunk_size
    static_ids = _bank_source_ids(
        frames_seen=frames_seen,
        frame_tokens=frame_tokens,
        sink_frames=int(parameters["static_sink_frames"]),
        recent_frames=int(parameters["static_recent_chunks"]) * chunk_size,
    )
    dynamic_ids = _bank_source_ids(
        frames_seen=frames_seen,
        frame_tokens=frame_tokens,
        sink_frames=int(parameters["dynamic_sink_frames"]),
        recent_frames=int(parameters["dynamic_recent_chunks"]) * chunk_size,
    )
    for start in patch_starts:
        dynamic_ids.update(range(start, start + patch_tokens))

    static_tensor = torch.tensor(
        sorted(static_ids), device=source_ids.device, dtype=torch.long
    )
    dynamic_tensor = torch.tensor(
        sorted(dynamic_ids), device=source_ids.device, dtype=torch.long
    )
    keep_global_source_ids(cache, static_tensor, prefix="static_")
    keep_global_source_ids(cache, dynamic_tensor, prefix="dynamic_")

    event = {
        "event": "forcingkv_physical_allocation",
        "phase": str(phase),
        "chunk_idx": int(chunk_idx),
        "candidate_patches_added": len(new_scores),
        "logical_selected_patches": len(selected),
        "static_tokens": len(static_ids),
        "dynamic_tokens": len(dynamic_ids),
        "middle_window_start_frame": window_start,
    }
    return ForcingKVUpdate(scores=scores, patch_starts=patch_starts, event=event)


def _bank_source_ids(
    *,
    frames_seen: int,
    frame_tokens: int,
    sink_frames: int,
    recent_frames: int,
) -> set[int]:
    """Return absolute token IDs for one sink-plus-recent head bank."""
    result = set(range(0, min(frames_seen, sink_frames) * frame_tokens))
    recent_start = max(sink_frames, frames_seen - recent_frames)
    result.update(range(recent_start * frame_tokens, frames_seen * frame_tokens))
    return result
