"""Framework-independent token identity selectors for all eight methods."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TokenSelection:
    token_ids: tuple[int, ...]
    total_tokens: int
    source: str
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if tuple(sorted(set(self.token_ids))) != self.token_ids:
            raise ValueError("token_ids must be unique and in source order")
        if self.token_ids and (self.token_ids[0] < 0 or self.token_ids[-1] >= self.total_tokens):
            raise ValueError("token selection is outside the source timeline")

    @property
    def kept_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def prune_ratio(self) -> float | None:
        return 1.0 - self.kept_tokens / self.total_tokens if self.total_tokens else None


def _timeline_tokens(frame_indices: Iterable[int], frame_tokens: int) -> tuple[int, ...]:
    per_frame = int(frame_tokens)
    if per_frame <= 0:
        raise ValueError("frame_tokens must be positive")
    return tuple(token for frame in frame_indices for token in range(int(frame) * per_frame, (int(frame) + 1) * per_frame))


def fullkv_selection(*, context_frames: int, frame_tokens: int) -> TokenSelection:
    total = int(context_frames) * int(frame_tokens)
    if total < 0:
        raise ValueError("context dimensions must be non-negative")
    return TokenSelection(tuple(range(total)), total, "fullkv", {"implicit": True})


def streaming_selection(*, context_frames: int, frame_tokens: int, sink_frames: int, local_frames: int) -> TokenSelection:
    context, sink, local = map(int, (context_frames, sink_frames, local_frames))
    if context < 0 or sink < 0 or local < 0:
        raise ValueError("streaming frame counts must be non-negative")
    frames = sorted(set(range(min(context, sink))) | set(range(max(0, context - local), context)))
    total = context * int(frame_tokens)
    return TokenSelection(_timeline_tokens(frames, frame_tokens), total, "streaming_sink_plus_local", {"frame_indices": frames})


def threshold_selection(scores: Sequence[float], *, threshold: float) -> tuple[int, ...]:
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    return tuple(index for index, score in enumerate(scores) if float(score) > value)


def deterministic_random_indices(*, candidate_count: int, keep_count: int, seed: int) -> tuple[int, ...]:
    count, keep = int(candidate_count), int(keep_count)
    if count < 0 or not 0 <= keep <= count:
        raise ValueError("keep_count is outside the candidate range")
    ranked = sorted(range(count), key=lambda index: hashlib.sha256(f"cmbench-random-v1:{int(seed)}:{index}".encode()).digest())
    return tuple(sorted(ranked[:keep]))


def random_matching_threshold(scores: Sequence[float], *, threshold: float, seed: int) -> tuple[int, ...]:
    return deterministic_random_indices(candidate_count=len(scores), keep_count=len(threshold_selection(scores, threshold=threshold)), seed=seed)


def lowest_similarity_indices(scores: Sequence[float], *, keep_count: int) -> tuple[int, ...]:
    keep = int(keep_count)
    if not 0 <= keep <= len(scores):
        raise ValueError("keep_count is outside the candidate range")
    selected = sorted(range(len(scores)), key=lambda index: (float(scores[index]), index))[:keep]
    return tuple(sorted(selected))


def forcingkv_patch_token_ids(selected_patch_indices: Sequence[int], *, sink_frames: int, frame_tokens: int, num_patches: int) -> tuple[int, ...]:
    frame, patches = int(frame_tokens), int(num_patches)
    if frame <= 0 or patches <= 0 or frame % patches:
        raise ValueError("frame_tokens must be divisible by num_patches")
    patch_tokens = frame // patches
    values: list[int] = []
    for raw_index in sorted(set(int(value) for value in selected_patch_indices)):
        if raw_index < 0:
            raise ValueError("selected patch indices must be non-negative")
        candidate_frame = raw_index // patches + int(sink_frames)
        patch_in_frame = raw_index % patches
        start = candidate_frame * frame + patch_in_frame * patch_tokens
        values.extend(range(start, start + patch_tokens))
    return tuple(values)


def patchification_block_token_ids(selected_block_indices: Sequence[int], *, sink_frames: int, token_height: int, token_width: int, grid_rows: int, grid_cols: int) -> tuple[int, ...]:
    height, width, rows, cols = map(int, (token_height, token_width, grid_rows, grid_cols))
    if min(height, width, rows, cols) <= 0 or height % rows or width % cols:
        raise ValueError("2-D block grid must evenly divide the token grid")
    block_h, block_w = height // rows, width // cols
    blocks_per_frame = rows * cols
    values: list[int] = []
    for raw_index in sorted(set(int(value) for value in selected_block_indices)):
        if raw_index < 0:
            raise ValueError("selected block indices must be non-negative")
        frame = raw_index // blocks_per_frame + int(sink_frames)
        block = raw_index % blocks_per_frame
        block_row, block_col = divmod(block, cols)
        frame_start = frame * height * width
        for row in range(block_h):
            start = frame_start + (block_row * block_h + row) * width + block_col * block_w
            values.extend(range(start, start + block_w))
    return tuple(values)


def dummyforcing_context_banks(*, context_frames: int, frame_tokens: int, chunk_size: int, first_frames: int, middle_bank_strategy: str, middle_bank_frames: int, middle_recent_frames: int, last_frames: int) -> dict[str, Any]:
    context, frame, chunk, first, bank, recent, last = map(int, (context_frames, frame_tokens, chunk_size, first_frames, middle_bank_frames, middle_recent_frames, last_frames))
    if min(context, frame, chunk, first, bank, recent, last) <= 0:
        raise ValueError("Dummy bank sizes must be positive")
    if any(value % chunk for value in (first, bank, recent, last)):
        raise ValueError("Dummy bank sizes must contain complete chunks")
    if middle_bank_strategy not in {"center", "rolling-suffix"}:
        raise ValueError("unknown Dummy middle bank strategy")
    first_indices = list(range(0, min(context, first)))
    recent_start = max(0, context - recent)
    recent_indices = list(range(recent_start, context))
    if middle_bank_strategy == "rolling-suffix":
        bank_start = max(0, recent_start - bank)
    else:
        eligible_start = min(context, first)
        eligible_end = max(eligible_start, recent_start)
        available = eligible_end - eligible_start
        bank = min(bank, (available // chunk) * chunk)
        chunk_slack = (available - bank) // chunk if bank else 0
        bank_start = eligible_start + (chunk_slack // 2) * chunk
    bank_indices = list(range(bank_start, min(context, bank_start + bank)))
    middle_indices = sorted(set(bank_indices + recent_indices))
    last_indices = list(range(max(0, context - last), context))
    return {
        "first": _timeline_tokens(first_indices, frame), "middle": _timeline_tokens(middle_indices, frame), "last": _timeline_tokens(last_indices, frame),
        "first_frame_indices": first_indices, "middle_bank_frame_indices": bank_indices, "middle_recent_frame_indices": recent_indices,
        "middle_frame_indices": middle_indices, "last_frame_indices": last_indices,
    }


def dummyforcing_group_assignment(scores: Sequence[Sequence[Sequence[float]]], *, last_group_count: int) -> tuple[tuple[int, ...], ...]:
    layers = len(scores)
    heads = len(scores[0]) if layers else 0
    if layers == 0 or heads == 0 or any(len(layer) != heads for layer in scores) or any(len(head) != 3 for layer in scores for head in layer):
        raise ValueError("scores must have shape [layers, heads, 3]")
    first = [float(scores[layer][head][0]) for layer in range(layers) for head in range(heads)]
    middle = [float(scores[layer][head][1]) for layer in range(layers) for head in range(heads)]
    first_total, middle_total = sum(first), sum(middle)
    if first_total <= 0 or middle_total <= 0:
        raise ValueError("first and middle attention mass must be positive")
    first_norm = [value / first_total for value in first]
    middle_norm = [value / middle_total for value in middle]
    count = int(last_group_count)
    if not 0 <= count <= layers * heads:
        raise ValueError("last_group_count is outside the layer-head range")
    cost = [max(left, right) for left, right in zip(first_norm, middle_norm)]
    assignment = [0] * (layers * heads)
    for index in sorted(range(len(cost)), key=lambda value: (cost[value], value))[:count]:
        assignment[index] = 2
    for index in range(len(assignment)):
        if assignment[index] != 2 and first_norm[index] < middle_norm[index]:
            assignment[index] = 1
    return tuple(tuple(assignment[layer * heads:(layer + 1) * heads]) for layer in range(layers))
