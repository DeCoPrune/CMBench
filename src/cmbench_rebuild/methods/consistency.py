"""Consistency-pruning scores and deterministic spatial safety floor."""
from __future__ import annotations

import math
from typing import Any, Sequence


MSE_SCORE_TYPES = frozenset({"mse", "normalized_mse", "chunk_normalized_mse"})
SCORE_TYPES = MSE_SCORE_TYPES | {"cosine"}


def token_mse_scores(predicted_x0: Any, target_x0: Any) -> Any:
    if tuple(predicted_x0.shape) != tuple(target_x0.shape):
        raise ValueError(f"x0/target shape mismatch: {tuple(predicted_x0.shape)} vs {tuple(target_x0.shape)}")
    values = (predicted_x0.float() - target_x0.float()).pow(2)
    channels, frames, height, width = values.shape
    if height % 2 or width % 2:
        raise ValueError(f"consistency scores require even latent dimensions, got {height}x{width}")
    return values.reshape(channels, frames, height // 2, 2, width // 2, 2).mean(dim=(0, 3, 5)).reshape(-1)


def token_scores(predicted_x0: Any, target_x0: Any, *, score_type: str = "mse") -> Any:
    import torch

    kind = str(score_type)
    if kind not in SCORE_TYPES:
        raise ValueError(f"unsupported consistency score type: {kind}")
    if kind == "mse":
        return token_mse_scores(predicted_x0, target_x0)
    if kind == "normalized_mse":
        squared = token_mse_scores(predicted_x0, target_x0)
        energy = token_mse_scores(target_x0, torch.zeros_like(target_x0))
        return squared / energy.clamp_min(1.0e-8)
    if kind == "chunk_normalized_mse":
        squared = token_mse_scores(predicted_x0, target_x0)
        return squared / target_x0.float().pow(2).mean().clamp_min(1.0e-8)
    if tuple(predicted_x0.shape) != tuple(target_x0.shape):
        raise ValueError("x0/target shape mismatch")
    channels, frames, height, width = predicted_x0.shape
    if height % 2 or width % 2:
        raise ValueError(f"consistency scores require even latent dimensions, got {height}x{width}")

    def flatten(value: Any) -> Any:
        return value.float().reshape(channels, frames, height // 2, 2, width // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(-1, channels * 4)

    return torch.nn.functional.cosine_similarity(flatten(predicted_x0), flatten(target_x0), dim=1, eps=1.0e-8)


def consistency_keep_mask(scores: Sequence[float], *, threshold: float, score_type: str = "mse") -> tuple[bool, ...]:
    kind = str(score_type)
    if kind not in SCORE_TYPES:
        raise ValueError(f"unsupported consistency score type: {kind}")
    limit = float(threshold)
    if not math.isfinite(limit):
        raise ValueError("threshold must be finite")
    result = []
    for raw in scores:
        score = float(raw)
        # An unstable diagnostic is not evidence that a token is disposable.
        result.append(True if not math.isfinite(score) else score > limit if kind in MSE_SCORE_TYPES else score <= limit)
    return tuple(result)


def apply_spatial_floor(
    keep_mask: Sequence[bool],
    scores: Sequence[float],
    *,
    frames: int,
    height: int,
    width: int,
    min_keep_per_block: int,
    block_size: int = 4,
    floor_threshold: float = -math.inf,
) -> tuple[bool, ...]:
    frames, height, width = map(int, (frames, height, width))
    minimum, block = int(min_keep_per_block), int(block_size)
    expected = frames * height * width
    if min(frames, height, width, block) <= 0 or minimum < 0:
        raise ValueError("spatial floor dimensions must be positive and minimum non-negative")
    if len(keep_mask) != expected or len(scores) != expected:
        raise ValueError(f"mask/scores need {expected} values")
    output = [bool(value) for value in keep_mask]
    values = [float(value) for value in scores]
    for frame in range(frames):
        frame_start = frame * height * width
        for y0 in range(0, height, block):
            for x0 in range(0, width, block):
                indices = [frame_start + y * width + x for y in range(y0, min(height, y0 + block)) for x in range(x0, min(width, x0 + block))]
                missing = minimum - sum(output[index] for index in indices)
                if missing <= 0 or max(values[index] for index in indices) <= float(floor_threshold):
                    continue
                eligible = [index for index in indices if not output[index]]
                for index in sorted(eligible, key=lambda item: (-values[item], item))[:missing]:
                    output[index] = True
    return tuple(output)


def build_consistency_mask(
    scores: Sequence[float],
    *,
    threshold: float,
    score_type: str,
    frames: int,
    height: int,
    width: int,
    min_keep_per_block: int = 0,
    block_size: int = 4,
    floor_threshold: float = -math.inf,
) -> tuple[bool, ...]:
    mask = consistency_keep_mask(scores, threshold=threshold, score_type=score_type)
    if not min_keep_per_block:
        return mask
    return apply_spatial_floor(mask, scores, frames=frames, height=height, width=width, min_keep_per_block=min_keep_per_block, block_size=block_size, floor_threshold=floor_threshold)
