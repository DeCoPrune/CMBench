"""Pure cache-visibility, sequence-PR and throughput contracts."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _visible_bank_token_head_units(*, live_tokens: int, capacity_tokens: int, current_noisy_tokens: int, num_heads: int) -> int:
    live, capacity, current, heads = map(int, (live_tokens, capacity_tokens, current_noisy_tokens, num_heads))
    if min(live, capacity, current, heads) < 0:
        raise ValueError("cache accounting values must be non-negative")
    if heads == 0:
        return 0
    if live > capacity:
        raise ValueError(f"live history {live} exceeds capacity {capacity}")
    if current > capacity:
        raise ValueError(f"current noisy tokens {current} exceed capacity {capacity}")
    evicted = max(0, live + current - capacity)
    return (live - evicted) * heads


def active_history_token_head_units_before_current(kv_cache: Sequence[dict[str, Any]], *, current_noisy_tokens: int) -> int:
    current = int(current_noisy_tokens)
    if current < 0:
        raise ValueError("current_noisy_tokens must be non-negative")
    total = 0
    for cache in kv_cache:
        mode = cache.get("cache_mode")
        if mode in {"physical_head_split", "physical_forcingkv"}:
            prefixes = ("static", "dynamic")
        elif mode == "physical_dummyforcing":
            prefixes = ("dummy_first", "dummy_middle", "dummy_last")
        else:
            prefixes = ("",)
        for prefix in prefixes:
            key_name = f"{prefix}_k" if prefix else "k"
            end_name = f"{prefix}_local_end_index" if prefix else "local_end_index"
            tensor = cache[key_name]
            total += _visible_bank_token_head_units(
                live_tokens=int(cache[end_name].item()),
                capacity_tokens=int(tensor.shape[1]),
                current_noisy_tokens=current,
                num_heads=int(tensor.shape[2]),
            )
    return total


def active_history_token_head_units_for_source_range(
    kv_cache: Sequence[dict[str, Any]], *, current_noisy_tokens: int, source_start: int, source_stop: int
) -> int:
    """Count visible cache units whose original source-token IDs lie in [start, stop)."""
    start, stop, current = int(source_start), int(source_stop), int(current_noisy_tokens)
    if start < 0 or stop < start or current < 0:
        raise ValueError("invalid source range or current token count")
    total = 0
    for cache in kv_cache:
        mode = cache.get("cache_mode")
        if mode in {"physical_head_split", "physical_forcingkv"}:
            prefixes = ("static", "dynamic")
        elif mode == "physical_dummyforcing":
            prefixes = ("dummy_first", "dummy_middle", "dummy_last")
        else:
            prefixes = ("",)
        for prefix in prefixes:
            key_name = f"{prefix}_k" if prefix else "k"
            end_name = f"{prefix}_local_end_index" if prefix else "local_end_index"
            ids_name = f"{prefix}_source_ids" if prefix else "source_ids"
            live = int(cache[end_name].item()) if hasattr(cache[end_name], "item") else int(cache[end_name])
            tensor = cache[key_name]
            evicted = max(0, live + current - int(tensor.shape[1]))
            ids = cache[ids_name][0, evicted:live]
            count = int(((ids >= start) & (ids < stop)).sum().item())
            total += count * int(tensor.shape[2])
    return total


def validate_generation_domain(*, context_latent_frames: int, output_latent_frames: int, chunk_size: int, pure_i2v_generation: bool) -> str:
    context, output, chunk = map(int, (context_latent_frames, output_latent_frames, chunk_size))
    if context < 0 or output <= 0 or chunk <= 0:
        raise ValueError("context must be non-negative and output/chunk must be positive")
    if output % chunk:
        raise ValueError("output latent frames must contain complete chunks")
    if pure_i2v_generation:
        if context != 0:
            raise ValueError("pure I2V generation requires zero context latents")
        return "pure-i2v"
    if context <= 0:
        raise ValueError("context continuation requires context latents")
    if context % chunk:
        raise ValueError("context latent frames must contain complete chunks")
    return "context-continuation"


def plan_i2v_scored_timeline(*, scored_pixel_frames: int, chunk_size: int, vae_temporal_stride: int) -> dict[str, int]:
    scored, chunk, stride = map(int, (scored_pixel_frames, chunk_size, vae_temporal_stride))
    if min(scored, chunk, stride) <= 0:
        raise ValueError("scored_pixel_frames, chunk_size and vae_temporal_stride must be positive")
    required_latents = math.ceil(scored / stride) + 1
    generated_latents = math.ceil(required_latents / chunk) * chunk
    return {
        "generated_latent_frames": generated_latents,
        "raw_decoded_pixel_frames": (generated_latents - 1) * stride + 1,
        "score_slice_start": 1,
        "score_slice_stop": 1 + scored,
        "scored_pixel_frames": scored,
        "conditioning_anchor_pixel_frames": 1,
        "chunk_padding_latent_frames": generated_latents - required_latents,
    }


def make_sequence_pr_step(*, chunk_index: int, context_latent_frames: int, completed_generated_latent_frames: int, current_chunk_size: int, active_token_head_layer_units: int, frame_tokens: int, num_heads: int, num_layers: int) -> dict[str, Any]:
    values = {
        "chunk_index": int(chunk_index),
        "context_latent_frames": int(context_latent_frames),
        "completed_generated_latent_frames": int(completed_generated_latent_frames),
        "current_noisy_latent_frames": int(current_chunk_size),
        "active_token_head_layer_units": int(active_token_head_layer_units),
        "frame_tokens": int(frame_tokens),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"sequence PR values must be non-negative: {values}")
    if values["current_noisy_latent_frames"] == 0:
        raise ValueError("current_chunk_size must be positive")
    dense_frames = values["context_latent_frames"] + values["completed_generated_latent_frames"]
    dense = dense_frames * values["frame_tokens"] * values["num_heads"] * values["num_layers"]
    active = values["active_token_head_layer_units"]
    if active > dense:
        raise ValueError(f"active units {active} exceed dense history {dense}")
    return {
        **values,
        "dense_history_latent_frames": dense_frames,
        "dense_token_head_layer_units": dense,
        "valid_for_seqPR": dense > 0,
        "prune_ratio": 1.0 - active / dense if dense else None,
        "current_noisy_excluded_from_numerator": True,
        "current_noisy_excluded_from_denominator": True,
    }


def aggregate_sequence_pr(steps: Sequence[dict[str, Any]], *, generation_kv_policy: str) -> dict[str, Any]:
    rows = list(steps)
    valid = [row for row in rows if int(row["dense_token_head_layer_units"]) > 0]
    active = sum(int(row["active_token_head_layer_units"]) for row in valid)
    dense = sum(int(row["dense_token_head_layer_units"]) for row in valid)
    if active > dense:
        raise ValueError(f"active units {active} exceed dense history {dense}")
    ratio = 1.0 - active / dense if dense else 0.0
    return {
        "schema_version": 1,
        "metric": "seqPR",
        "generation_kv_policy": str(generation_kv_policy),
        "generation_chunks": len(rows),
        "valid_generation_chunks": len(valid),
        "active_token_head_layer_units": active,
        "dense_token_head_layer_units": dense,
        "seqPR": ratio,
        "seq_prune_ratio": ratio,
        "aggregation": "ratio_of_generation_chunk_totals_within_video",
        "steps": rows,
    }


def ar_frames_per_second(continuation_latent_frames: int, ar_elapsed_seconds: float) -> float:
    frames, elapsed = int(continuation_latent_frames), float(ar_elapsed_seconds)
    if frames < 0 or elapsed <= 0:
        raise ValueError("frames must be non-negative and elapsed time positive")
    return 4 * frames / elapsed


def actual_decoded_frames_per_second(continuation_pixel_frames: int, ar_elapsed_seconds: float) -> float:
    frames, elapsed = int(continuation_pixel_frames), float(ar_elapsed_seconds)
    if frames < 0 or elapsed <= 0:
        raise ValueError("frames must be non-negative and elapsed time positive")
    return frames / elapsed


def aggregate_frames_per_second(frame_count_per_case: int, elapsed_seconds: Sequence[float]) -> float:
    frames = int(frame_count_per_case)
    elapsed = [float(value) for value in elapsed_seconds]
    if frames < 0 or not elapsed or any(value <= 0 for value in elapsed):
        raise ValueError("frame count must be non-negative and elapsed times positive")
    return frames * len(elapsed) / sum(elapsed)
