"""Physical KV writes and attention, independent of LingBot's SP transport."""
from __future__ import annotations

from typing import Any, Callable

from .compaction import compact_static_predecessor

from ..methods.selectors import deterministic_random_indices
from ..methods.torch_kernels import (
    dummyforcing_reference_evidence,
    dummyforcing_region_attention,
)


Attender = Callable[[Any, Any, Any], Any]


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _run_from_offloaded_cache(
    query: Any,
    *,
    cache: dict[str, Any],
    call: Callable[[dict[str, Any]], Any],
) -> Any:
    """Stage one layer's cache for attention, then return it to host memory.

    Only one transformer layer is staged at a time.  This keeps the complete
    FullKV semantics without holding every layer's history on the GPU.
    """
    import torch

    owner = getattr(cache, "owner", None)
    if owner is not None and owner.stager is not None:
        return owner.stager.run(cache.index, int(query.shape[1]), call)

    compute_device = query.device
    sequence_tokens = int(query.shape[1])

    def staged_tensor(name: str, item: Any) -> Any:
        if not isinstance(item, torch.Tensor):
            return item
        if item.ndim < 2 or not (
            name in {"k", "v", "tpos", "source_ids"}
            or name.endswith(("_k", "_v", "_tpos", "_source_ids"))
        ):
            return item.to(compute_device)
        prefix = ""
        if name not in {"k", "v", "tpos", "source_ids"}:
            for suffix in ("k", "v", "tpos", "source_ids"):
                marker = f"_{suffix}"
                if name.endswith(marker):
                    prefix = name[: -len(suffix)]
                    break
        end_name = f"{prefix}local_end_index" if prefix else "local_end_index"
        live = _int(cache[end_name])
        staged_tokens = min(int(item.shape[1]), live + sequence_tokens)
        return item[:, :staged_tokens].to(compute_device)

    working = {name: staged_tensor(name, item) for name, item in cache.items()}
    output = call(working)

    # Attention may consume transient flags and update score buffers as well
    # as K/V cursors. Mirror the complete per-layer state back to its storage
    # device before the next transformer layer runs.
    for name in tuple(cache):
        if name not in working:
            del cache[name]
    for name, item in working.items():
        original = cache.get(name)
        if isinstance(item, torch.Tensor) and isinstance(original, torch.Tensor):
            if item.ndim >= 2 and original.ndim >= 2 and int(original.shape[1]) >= int(item.shape[1]):
                original[:, : item.shape[1]].copy_(item.to(original.device))
            else:
                original.copy_(item.to(original.device))
        elif not isinstance(item, torch.Tensor):
            cache[name] = item
        else:
            cache[name] = item.cpu()
    working.clear()
    return output


def _write_tpos(cache: dict[str, Any], name: str, start: int, end: int, *, frame_tokens: int, start_frame: int) -> None:
    import torch

    target = cache.get(name)
    if target is None:
        return
    positions = torch.arange(end - start, device=target.device, dtype=torch.int32)
    positions = positions.div(int(frame_tokens), rounding_mode="floor").add_(int(start_frame))
    target[:, start:end] = positions.view(1, -1, 1, 1)


def _write_source_ids(cache: dict[str, Any], name: str, start: int, end: int, *, current_start: int) -> None:
    import torch

    target = cache.get(name)
    if target is None:
        return
    target[:, start:end] = torch.arange(
        int(current_start),
        int(current_start) + int(end - start),
        device=target.device,
        dtype=torch.long,
    ).view(1, -1)


def _write_group(
    cache: dict[str, Any],
    *,
    prefix: str,
    indices: Any,
    key: Any,
    value: Any,
    is_new: bool,
    sequence_tokens: int,
    frame_tokens: int,
    start_frame: int,
    current_start: int,
) -> int:
    end_name = f"{prefix}local_end_index"
    end = _int(cache[end_name])
    start = end if is_new else end - sequence_tokens
    if start < 0:
        raise RuntimeError(f"{prefix}KV cannot overwrite a missing current chunk")
    if is_new:
        end += sequence_tokens
    key_buffer, value_buffer = cache[f"{prefix}k"], cache[f"{prefix}v"]
    if end > int(key_buffer.shape[1]):
        raise RuntimeError(f"{prefix}KV cache overflow: need {end}, capacity={key_buffer.shape[1]}")
    if int(indices.numel()):
        key_buffer[:, start:end] = key.index_select(2, indices)
        value_buffer[:, start:end] = value.index_select(2, indices)
    _write_tpos(cache, f"{prefix}tpos", start, end, frame_tokens=frame_tokens, start_frame=start_frame)
    _write_source_ids(cache, f"{prefix}source_ids", start, end, current_start=current_start)
    cache[end_name].fill_(end)
    return end


def _group_attention(query: Any, cache: dict[str, Any], groups: tuple[str, ...], attender: Attender) -> tuple[Any, list[int]]:
    output = query.new_zeros(query.shape)
    ends: list[int] = []
    for prefix in groups:
        indices = cache[f"{prefix}local_head_indices"]
        end = _int(cache[f"{prefix}local_end_index"])
        ends.append(end)
        if int(indices.numel()):
            attended = attender(
                query.index_select(2, indices),
                cache[f"{prefix}k"][:, :end],
                cache[f"{prefix}v"][:, :end],
            )
            output.index_copy_(2, indices, attended)
    return output, ends


def _dense_dummy_score(cache: dict[str, Any], query: Any, key: Any, *, frame_tokens: int, sequence_tokens: int, current_end: int) -> None:
    target_end = cache.get("dummyforcing_score_at_global_end")
    armed = bool(cache.pop("dummyforcing_score_pending", False))
    # The third causal chunk is the method definition, not merely an external
    # arm flag. Keep this structural trigger so wrappers/checkpointing cannot
    # silently lose transient Python metadata between model calls.
    at_structural_target = (
        str(cache.get("cache_mode")) == "dummyforcing_warmup"
        and int(current_end) == 3 * int(sequence_tokens)
    )
    at_target = (target_end is not None and int(current_end) == int(target_end)) or at_structural_target
    valid = cache.get("dummyforcing_group_scores_valid")
    already_valid = bool(valid.item()) if valid is not None else "dummyforcing_group_scores" in cache
    if not armed and not (at_target and not already_valid):
        return
    sample_count = sequence_tokens // 3
    if sample_count <= 0:
        raise RuntimeError("DummyForcing requires at least three query tokens")
    rows = deterministic_random_indices(
        candidate_count=sequence_tokens,
        keep_count=sample_count,
        seed=int(cache["dummyforcing_sample_seed"]),
    )
    import torch

    sampled_rows = torch.tensor(rows, device=query.device, dtype=torch.long)
    evidence_mode = str(cache.get("dummyforcing_classification_evidence", "original_packed"))
    score_key = (
        dummyforcing_reference_evidence(
            key,
            frame_tokens=frame_tokens,
            chunk_size=int(cache["dummyforcing_chunk_size"]),
        )
        if evidence_mode == "original_packed"
        else key
    )
    if evidence_mode not in {"original_packed", "chunkwise_dense", "q0-context-regions"}:
        raise RuntimeError(f"unknown DummyForcing evidence mode: {evidence_mode}")
    scores = dummyforcing_region_attention(
        query,
        score_key,
        frame_tokens=frame_tokens,
        ar_start=int(cache.get("dummyforcing_ar_start", 2)),
        sampled_rows=sampled_rows,
        query_block_size=int(cache.get("dummyforcing_query_block_size", 64)),
        first_region_frames=(int(cache["dummyforcing_first_region_frames"]) if evidence_mode == "q0-context-regions" else None),
        recent_region_frames=(int(cache["dummyforcing_recent_region_frames"]) if evidence_mode == "q0-context-regions" else None),
    ).detach()
    buffer = cache.get("dummyforcing_group_scores_buffer")
    if buffer is not None:
        buffer.copy_(scores)
        cache["dummyforcing_group_scores_valid"].fill_(True)
    else:
        cache["dummyforcing_group_scores"] = scores


def cached_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    cache: dict[str, Any],
    current_start: int,
    frame_tokens: int,
    local_attn_size: int,
    sink_size: int,
    max_attention_size: int,
    attender: Attender,
) -> Any:
    """Write one RoPE-applied K/V chunk and attend to the physical live state."""
    import torch

    first_key = next(
        item
        for name, item in cache.items()
        if isinstance(item, torch.Tensor) and (name == "k" or name.endswith("_k"))
    )
    if first_key.device != query.device:
        return _run_from_offloaded_cache(
            query,
            cache=cache,
            call=lambda staged: cached_attention(
                query,
                key,
                value,
                cache=staged,
                current_start=current_start,
                frame_tokens=frame_tokens,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                max_attention_size=max_attention_size,
                attender=attender,
            ),
        )
    sequence_tokens = int(query.shape[1])
    if sequence_tokens <= 0 or int(key.shape[1]) != sequence_tokens or int(value.shape[1]) != sequence_tokens:
        raise ValueError("query/key/value must have the same positive sequence length")
    current_start = int(current_start)
    current_end = current_start + sequence_tokens
    start_frame = current_start // int(frame_tokens)
    previous_global = _int(cache["global_end_index"])
    if current_end < previous_global:
        raise RuntimeError(f"non-monotonic absolute KV write: {current_end} < {previous_global}")
    is_new = current_end > previous_global
    mode = str(cache.get("cache_mode") or "native")

    if mode == "physical_dummyforcing":
        groups = ("dummy_first_", "dummy_middle_", "dummy_last_")
        for prefix in groups:
            _write_group(
                cache,
                prefix=prefix,
                indices=cache[f"{prefix}local_head_indices"],
                key=key,
                value=value,
                is_new=is_new,
                sequence_tokens=sequence_tokens,
                frame_tokens=frame_tokens,
                start_frame=start_frame,
                current_start=current_start,
            )
        output, ends = _group_attention(query, cache, groups, attender)
        cache["local_end_index"].fill_(max(ends))
    elif mode in {"physical_head_split", "physical_forcingkv"}:
        if is_new and mode == "physical_head_split" and not bool(cache.get("append_only_generation", False)):
            compact_static_predecessor(cache)
        groups = ("static_", "dynamic_")
        for prefix in groups:
            _write_group(
                cache,
                prefix=prefix,
                indices=cache[f"{prefix}local_head_indices"],
                key=key,
                value=value,
                is_new=is_new,
                sequence_tokens=sequence_tokens,
                frame_tokens=frame_tokens,
                start_frame=start_frame,
                current_start=current_start,
            )
        output, ends = _group_attention(query, cache, groups, attender)
        cache["local_end_index"].fill_(max(ends))
    else:
        capacity = int(cache["k"].shape[1])
        previous_local = _int(cache["local_end_index"])
        if mode in {"physical_compacted", "physical_frozen_dense"}:
            local_start = previous_local if is_new else previous_local - sequence_tokens
            local_end = local_start + sequence_tokens
            if local_start < 0 or local_end > capacity:
                raise RuntimeError(f"physical KV cache overflow/underflow: [{local_start},{local_end}) of {capacity}")
        elif int(local_attn_size) == -1:
            local_start, local_end = current_start, current_end
            if local_end > capacity:
                raise RuntimeError(f"dense KV cache overflow: need {local_end}, capacity={capacity}")
        elif is_new and sequence_tokens + previous_local > capacity:
            evicted = sequence_tokens + previous_local - capacity
            sink_tokens = int(sink_size) * int(frame_tokens)
            rolled = previous_local - evicted - sink_tokens
            if rolled < 0:
                raise RuntimeError("streaming window is smaller than its sink plus incoming chunk")
            for suffix in ("k", "v", "tpos", "source_ids"):
                buffer = cache.get(suffix)
                if buffer is not None:
                    buffer[:, sink_tokens:sink_tokens + rolled] = buffer[:, sink_tokens + evicted:sink_tokens + evicted + rolled].clone()
            local_end = previous_local + sequence_tokens - evicted
            local_start = local_end - sequence_tokens
        else:
            delta = current_end - previous_global
            local_end = previous_local + delta
            local_start = local_end - sequence_tokens
        cache["k"][:, local_start:local_end] = key
        cache["v"][:, local_start:local_end] = value
        _write_tpos(cache, "tpos", local_start, local_end, frame_tokens=frame_tokens, start_frame=start_frame)
        _write_source_ids(cache, "source_ids", local_start, local_end, current_start=current_start)
        left = max(0, local_end - int(max_attention_size))
        live_key, live_value = cache["k"][:, left:local_end], cache["v"][:, left:local_end]
        _dense_dummy_score(
            cache,
            query,
            live_key,
            frame_tokens=frame_tokens,
            sequence_tokens=sequence_tokens,
            current_end=current_end,
        )
        output = attender(query, live_key, live_value)
        cache["local_end_index"].fill_(local_end)

    cache["global_end_index"].fill_(current_end)
    return output
