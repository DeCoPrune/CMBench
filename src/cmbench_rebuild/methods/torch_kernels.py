"""Torch scoring kernels isolated from orchestration and cache mutation."""
from __future__ import annotations

import math
from typing import Any


def framewise_attention_mass(query: Any, key: Any, *, frame_tokens: int, query_block_size: int, query_indices: Any | None = None) -> Any:
    import torch

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have shape [batch,tokens,heads,dim]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key batch, head and channel shapes must match")
    frame, block_size = int(frame_tokens), int(query_block_size)
    if frame <= 0 or key.shape[1] % frame:
        raise ValueError("key tokens must contain complete latent frames")
    if block_size <= 0:
        raise ValueError("query_block_size must be positive")
    if query_indices is not None:
        if query_indices.ndim != 1 or query_indices.numel() == 0:
            raise ValueError("query_indices must be a non-empty vector")
        indices = query_indices.to(device=query.device, dtype=torch.long)
        if int(indices.min()) < 0 or int(indices.max()) >= query.shape[1]:
            raise ValueError("query_indices are outside the query sequence")
        query = query.index_select(1, indices)
    query_count = int(query.shape[1])
    if query_count <= 0:
        raise ValueError("at least one query row is required")
    frame_count = int(key.shape[1]) // frame
    mass = torch.zeros(int(query.shape[2]), frame_count, device=query.device, dtype=torch.float32)
    scale = int(query.shape[-1]) ** -0.5
    key_float = key.float()
    for start in range(0, query_count, block_size):
        block = query[:, start:start + block_size].float()
        scores = torch.einsum("bqhd,bkhd->bhqk", block, key_float) * scale
        probabilities = torch.softmax(scores, dim=-1)
        mass += probabilities.reshape(probabilities.shape[0], probabilities.shape[1], probabilities.shape[2], frame_count, frame).sum(dim=-1).sum(dim=(0, 2))
    return mass / (int(query.shape[0]) * query_count)


def forcingkv_adjacent_patch_moments(previous_frame_k: Any, current_chunk_k: Any, *, num_patches: int) -> tuple[Any, Any, Any]:
    import torch

    if previous_frame_k.ndim != 3:
        raise ValueError("previous_frame_k must have shape [tokens,heads,dim]")
    if current_chunk_k.ndim != 4 or current_chunk_k.shape[0] <= 0:
        raise ValueError("current_chunk_k must have shape [frames>0,tokens,heads,dim]")
    if tuple(previous_frame_k.shape) != tuple(current_chunk_k.shape[1:]):
        raise ValueError("previous/current K shapes do not match")
    frame_tokens, patches = int(previous_frame_k.shape[0]), int(num_patches)
    if patches <= 0 or frame_tokens % patches:
        raise ValueError("frame token count must be divisible by num_patches")
    pair_count = int(current_chunk_k.shape[0])
    chain = torch.cat([previous_frame_k.unsqueeze(0), current_chunk_k], dim=0)
    values = chain.reshape(pair_count + 1, patches, frame_tokens // patches, -1).float()
    left, right = values[:-1], values[1:]
    return (left * right).sum(dim=-1), left.square().sum(dim=-1), right.square().sum(dim=-1)


def forcingkv_scores_from_moments(dot: Any, left_sq: Any, right_sq: Any) -> Any:
    if dot.shape != left_sq.shape or dot.shape != right_sq.shape:
        raise ValueError("dot and norm moments must share a shape")
    denominator = left_sq.float().sqrt().clamp_min(1.0e-8) * right_sq.float().sqrt().clamp_min(1.0e-8)
    return (dot.float() / denominator).mean(dim=-1)


def forcingkv_adjacent_patch_scores(previous_frame_k: Any, current_chunk_k: Any, *, num_patches: int) -> Any:
    return forcingkv_scores_from_moments(*forcingkv_adjacent_patch_moments(previous_frame_k, current_chunk_k, num_patches=num_patches))


def patchification_block_moments(context_k: Any, *, sink_frames: int, recent_frames: int, token_height: int, token_width: int, grid_rows: int, grid_cols: int) -> tuple[Any, Any, Any]:
    import torch

    if context_k.ndim != 4:
        raise ValueError("context_k must have shape [frames,tokens,heads,dim]")
    frames, frame_tokens = int(context_k.shape[0]), int(context_k.shape[1])
    height, width, rows, cols = map(int, (token_height, token_width, grid_rows, grid_cols))
    sink, recent = int(sink_frames), int(recent_frames)
    middle_end = frames - recent
    if height * width != frame_tokens:
        raise ValueError("token_height * token_width must equal frame_tokens")
    if rows <= 0 or cols <= 0 or height % rows or width % cols:
        raise ValueError("2-D block grid must evenly divide the token grid")
    if sink < 0 or recent <= 0 or sink >= middle_end:
        raise ValueError("sink/recent leave no middle candidate frames")
    block_h, block_w = height // rows, width // cols

    def blocks(values: Any) -> Any:
        values = values.reshape(values.shape[0], height, width, int(values.shape[2]) * int(values.shape[3]))
        return values.reshape(values.shape[0], rows, block_h, cols, block_w, values.shape[-1]).permute(0, 1, 3, 2, 4, 5).reshape(values.shape[0], rows * cols, block_h * block_w, -1).float()

    dots, left_norms, right_norms = [], [], []
    for start in range(sink, middle_end):
        left, right = blocks(context_k[start:start + 1]), blocks(context_k[start + 1:start + 2])
        dots.append((left * right).sum(dim=-1))
        left_norms.append(left.square().sum(dim=-1))
        right_norms.append(right.square().sum(dim=-1))
    return torch.cat(dots), torch.cat(left_norms), torch.cat(right_norms)


def dummyforcing_region_attention(query: Any, key: Any, *, frame_tokens: int, ar_start: int, sampled_rows: Any, query_block_size: int = 64, first_region_frames: int | None = None, recent_region_frames: int | None = None) -> Any:
    import torch

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have shape [batch,tokens,heads,dim]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key batch, head and dim shapes must match")
    frame = int(frame_tokens)
    if frame <= 0 or query.shape[1] % frame:
        raise ValueError("query length must contain whole latent frames")
    rows = sampled_rows.to(device=query.device, dtype=torch.long)
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("sampled_rows must be a non-empty vector")
    if int(rows.min()) < 0 or int(rows.max()) >= query.shape[1]:
        raise ValueError("sampled row is outside the query")
    block_size = int(query_block_size)
    if block_size <= 0:
        raise ValueError("query_block_size must be positive")
    query_tokens = int(query.shape[1])
    first_tokens = int(first_region_frames) * frame if first_region_frames is not None else query_tokens // 3 if int(ar_start) == 1 else query_tokens
    last_tokens = int(recent_region_frames) * frame if recent_region_frames is not None else query_tokens
    if first_region_frames is not None or recent_region_frames is not None:
        first_tokens = min(int(key.shape[1]), first_tokens)
        last_start = max(first_tokens, int(key.shape[1]) - min(int(key.shape[1]), last_tokens))
    else:
        if key.shape[1] < first_tokens + last_tokens:
            raise ValueError("key timeline is too short for first/middle/last regions")
        last_start = int(key.shape[1]) - last_tokens
    mass = torch.zeros(query.shape[0], query.shape[2], 3, dtype=torch.float32, device=query.device)
    key_transposed = key.transpose(1, 2).transpose(-2, -1)
    for start in range(0, int(rows.numel()), block_size):
        block_rows = rows[start:start + block_size]
        sampled = query.index_select(1, block_rows).transpose(1, 2)
        attention = torch.softmax(torch.matmul(sampled, key_transposed) / math.sqrt(float(query.shape[-1])), dim=-1, dtype=torch.float32)
        mass[..., 0] += attention[..., :first_tokens].sum(dim=(-2, -1), dtype=torch.float32)
        middle = attention[..., first_tokens:last_start] if last_start != first_tokens else attention[..., last_start:]
        mass[..., 1] += middle.sum(dim=(-2, -1), dtype=torch.float32)
        mass[..., 2] += attention[..., last_start:].sum(dim=(-2, -1), dtype=torch.float32)
    return (mass / float(rows.numel())).mean(dim=0)


def dummyforcing_reference_evidence(dense_history: Any, *, frame_tokens: int, chunk_size: int) -> Any:
    import torch

    frame, chunk = int(frame_tokens), int(chunk_size)
    if dense_history.ndim != 4 or int(dense_history.shape[1]) != 3 * chunk * frame:
        raise ValueError("DummyForcing classification requires dense C0+C1+C2 history")
    return torch.cat([dense_history[:, :frame], dense_history[:, chunk * frame:2 * chunk * frame], dense_history[:, 2 * chunk * frame:]], dim=1)
