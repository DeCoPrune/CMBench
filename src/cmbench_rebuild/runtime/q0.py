"""q0 finalization: selection, cache qualification, and RoPE retargeting."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..core.rope import RopeReindexPlan
from ..methods.selectors import (
    fullkv_selection,
    lowest_similarity_indices,
    patchification_block_token_ids,
)
from ..methods.torch_kernels import forcingkv_scores_from_moments, patchification_block_moments
from .cache_factory import CacheGeometry
from .compaction import keep_global_source_ids, resize_live_banks
from .kv_cache import cache_banks, prepare_append_only_generation


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _identity_digest(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_identity_snapshot(cache_layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Small auditable description of every physical bank at q0."""
    layers = []
    for layer_index, cache in enumerate(cache_layers):
        banks = []
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            live = _int(cache[bank.end_name])
            values = cache[f"{prefix}source_ids"][0, :live].detach().cpu().tolist()
            if any(right <= left for left, right in zip(values, values[1:])):
                raise RuntimeError(f"source-token identity is not ordered at layer {layer_index}, bank {bank.prefix or 'dense'}")
            banks.append({
                "bank": bank.prefix or "dense",
                "live_tokens": live,
                "source_id_min": int(values[0]) if values else None,
                "source_id_max": int(values[-1]) if values else None,
                "source_ids_sha256": _identity_digest(values),
                "heads": int(cache[bank.key_name].shape[2]),
            })
        layers.append({"layer_idx": layer_index, "cache_mode": str(cache.get("cache_mode") or "native"), "banks": banks})
    return {"schema_version": 1, "capture": "q0_before_first_generated_chunk", "layers": layers}


def token_union(cache_layers: list[dict[str, Any]]) -> tuple[int, ...]:
    values: set[int] = set()
    for cache in cache_layers:
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            if int(cache[bank.key_name].shape[2]) == 0:
                continue
            live = _int(cache[bank.end_name])
            values.update(int(value) for value in cache[f"{prefix}source_ids"][0, :live].detach().cpu().tolist())
    return tuple(sorted(values))


def patchification_q0(
    cache_layers: list[dict[str, Any]],
    geometry: CacheGeometry,
    parameters: dict[str, Any],
    online_policy: Any | None = None,
) -> dict[str, Any]:
    """Select lowest-cosine 2-D middle blocks once from dense context KV."""
    import torch
    import torch.distributed as dist

    sink, recent = int(parameters["sink_frames"]), int(parameters["recent_frames"])
    rows, cols, topk = int(parameters["grid_rows"]), int(parameters["grid_cols"]), int(parameters["topk_blocks"])
    token_height = int(round(geometry.frame_tokens ** 0.5))
    while token_height > 1 and geometry.frame_tokens % token_height:
        token_height -= 1
    token_width = geometry.frame_tokens // token_height
    # LingBot's canonical 480x832 patch grid is 30x52; do not silently accept
    # a transposed/factorized grid that changes 2-D block membership.
    if geometry.frame_tokens == 1560:
        token_height, token_width = 30, 52
    reference = cache_layers[1 if len(cache_layers) > 1 else 0]
    live = _int(reference["local_end_index"])
    expected = geometry.context_frames * geometry.frame_tokens
    if live != expected:
        raise RuntimeError(f"Patchification q0 needs dense context KV: {live} != {expected}")
    compute_device = reference.get("compute_device", reference["k"].device)
    context_key = reference["k"][0, :live].to(compute_device).reshape(
        geometry.context_frames,
        geometry.frame_tokens,
        reference["k"].shape[2],
        reference["k"].shape[3],
    )
    moments = patchification_block_moments(
        context_key,
        sink_frames=sink,
        recent_frames=recent,
        token_height=token_height,
        token_width=token_width,
        grid_rows=rows,
        grid_cols=cols,
    )
    if dist.is_available() and dist.is_initialized():
        for moment in moments:
            dist.all_reduce(moment, op=dist.ReduceOp.SUM)
    scores = forcingkv_scores_from_moments(*moments).reshape(-1)
    keep_blocks = min(topk, int(scores.numel()))
    selected_blocks = lowest_similarity_indices(scores.detach().cpu().tolist(), keep_count=keep_blocks)
    middle_ids = patchification_block_token_ids(
        selected_blocks,
        sink_frames=sink,
        token_height=token_height,
        token_width=token_width,
        grid_rows=rows,
        grid_cols=cols,
    )
    fixed_ids = set(fullkv_selection(context_frames=sink, frame_tokens=geometry.frame_tokens).token_ids)
    recent_start = max(sink, geometry.context_frames - recent)
    fixed_ids.update(range(recent_start * geometry.frame_tokens, geometry.context_frames * geometry.frame_tokens))
    selected_ids = tuple(sorted(fixed_ids | set(middle_ids)))
    if bool(parameters.get("update_each_chunk", False)):
        if online_policy is None:
            raise RuntimeError("online Patchification requires the generation policy runtime")
        # Candidate IDs encode the left frame and 2-D block, so new
        # adjacent-frame scores can be added incrementally during generation.
        block_count = rows * cols
        global_candidate_scores = {
            (sink + pair_index) * block_count + block_index: float(value)
            for pair_index in range(int(scores.numel()) // block_count)
            for block_index, value in enumerate(scores.detach().cpu().tolist()[pair_index * block_count:(pair_index + 1) * block_count])
        }
        online_policy.initialize_online_patchification(
            candidate_scores=global_candidate_scores,
            last_scored_start=sink + int(scores.numel()) // block_count - 1,
            token_height=token_height,
            token_width=token_width,
            grid_rows=rows,
            grid_cols=cols,
            sink_frames=sink,
            recent_frames=recent,
            topk_blocks=topk,
        )
    selection = torch.tensor(selected_ids, dtype=torch.long, device=reference["k"].device)
    compaction = keep_global_source_ids(cache_layers, selection)
    # Generation writes must append after the compacted physical prefix, not
    # at their much larger absolute source-token offset on the original dense
    # timeline.
    for cache in cache_layers:
        cache["cache_mode"] = "physical_compacted"
    resize_live_banks(
        cache_layers,
        capacity=len(selected_ids) + geometry.output_frames * geometry.frame_tokens,
    )
    return {
        "allocator": "layer1_all_middle_2d_block_cosine_lowest",
        "candidate_blocks": int(scores.numel()),
        "selected_blocks": len(selected_blocks),
        "selected_block_indices_sha256": _identity_digest(selected_blocks),
        "selected_tokens": len(selected_ids),
        "update_each_chunk": bool(parameters.get("update_each_chunk", False)),
        "selected_source_ids_sha256": _identity_digest(selected_ids),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        **compaction,
    }


def apply_rope_reindex(
    cache_layers: list[dict[str, Any]],
    *,
    plan: RopeReindexPlan,
    context_frames: int,
    frame_tokens: int,
    temporal_rope_frequencies: Any,
    block_size: int = 8192,
) -> dict[str, Any]:
    """Retarget RoPE-baked context keys using source-token identity."""
    import torch

    plan.validate()
    if plan.mode == "off":
        return {"mode": "off", "rotated_banks": 0, "rotated_tokens": 0}
    virtual = torch.tensor(
        plan.virtual_positions(int(context_frames)),
        dtype=torch.float64,
        device="cpu",
    )
    theta = torch.angle(temporal_rope_frequencies[1].to(torch.complex128)).cpu()
    pair_count = plan.temporal_pair_count(int(temporal_rope_frequencies.shape[1]))
    theta = theta[:pair_count]
    rotated_banks = rotated_tokens = 0
    for cache in cache_layers:
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            keys = cache[bank.key_name]
            if int(keys.shape[2]) == 0:
                continue
            live = _int(cache[bank.end_name])
            ids = cache[f"{prefix}source_ids"][0, :live]
            if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= context_frames * frame_tokens):
                raise RuntimeError("q0 RoPE re-index found a non-context source token")
            real_frames = ids.div(int(frame_tokens), rounding_mode="floor").to(torch.long)
            dtype = keys.dtype
            compute_device = cache.get("compute_device", keys.device)
            device_virtual = virtual.to(compute_device)
            device_theta = theta.to(compute_device)
            for start in range(0, live, int(block_size)):
                stop = min(start + int(block_size), live)
                real = real_frames[start:stop].to(compute_device)
                delta = device_virtual.index_select(0, real) - real.to(torch.float64)
                rotation = torch.polar(
                    torch.ones(
                        stop - start,
                        pair_count,
                        dtype=torch.float64,
                        device=compute_device,
                    ),
                    delta[:, None] * device_theta[None, :],
                ).to(torch.complex64)
                block = keys[:, start:stop].to(compute_device).to(torch.float32).clone()
                complex_block = torch.view_as_complex(block.reshape(1, stop - start, keys.shape[2], -1, 2))
                complex_block[..., :pair_count] *= rotation[None, :, None, :]
                rotated = torch.view_as_real(complex_block).flatten(3).to(dtype)
                keys[:, start:stop].copy_(rotated.to(keys.device))
            rotated_banks += 1
            rotated_tokens += live
    return {
        "mode": plan.mode,
        "virtual_span": plan.virtual_span,
        "recent_frames": plan.recent_frames,
        "temporal_pair_count": pair_count,
        "rotated_banks": rotated_banks,
        "rotated_tokens": rotated_tokens,
    }


def finalize_q0(
    cache_layers: list[dict[str, Any]],
    *,
    geometry: CacheGeometry,
    method: str,
    parameters: dict[str, Any],
    generation_kv_policy: str,
    rope_plan: RopeReindexPlan,
    temporal_rope_frequencies: Any,
    online_policy: Any | None = None,
) -> dict[str, Any]:
    policy = patchification_q0(cache_layers, geometry, parameters, online_policy=online_policy) if method == "patchification" else {"allocator": "online_or_implicit"}
    selected = token_union(cache_layers)
    rope = apply_rope_reindex(
        cache_layers,
        plan=rope_plan,
        context_frames=geometry.context_frames,
        frame_tokens=geometry.frame_tokens,
        temporal_rope_frequencies=temporal_rope_frequencies,
    )
    if generation_kv_policy == "append-only":
        prepare_append_only_generation(cache_layers, output_tokens=geometry.output_frames * geometry.frame_tokens)
    elif generation_kv_policy != "method-native":
        raise ValueError("unknown generation KV policy")
    return {
        "schema_version": 1,
        "method": method,
        "context_tokens": geometry.context_frames * geometry.frame_tokens,
        "selected_tokens": len(selected),
        "selected_source_ids_sha256": _identity_digest(selected),
        "token_prune_ratio": 1.0 - len(selected) / (geometry.context_frames * geometry.frame_tokens),
        "policy": policy,
        "rope_reindex": rope,
        "cache": cache_identity_snapshot(cache_layers),
    }
