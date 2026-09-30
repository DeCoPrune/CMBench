"""Per-case physical KV allocation for every registered method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..methods.head_map import HeadMap


@dataclass(frozen=True)
class CacheGeometry:
    num_layers: int
    num_heads: int
    head_dim: int
    world_size: int
    rank: int
    context_frames: int
    output_frames: int
    chunk_size: int
    frame_tokens: int

    def validate(self) -> "CacheGeometry":
        positive = (self.num_layers, self.num_heads, self.head_dim, self.world_size, self.chunk_size, self.frame_tokens)
        if any(int(value) <= 0 for value in positive):
            raise ValueError("model/cache geometry dimensions must be positive")
        if int(self.context_frames) <= 0 or int(self.output_frames) <= 0:
            raise ValueError("context and output frames must be positive")
        if int(self.num_heads) % int(self.world_size):
            raise ValueError("num_heads must be divisible by world_size")
        if not 0 <= int(self.rank) < int(self.world_size):
            raise ValueError("rank is outside world_size")
        if int(self.context_frames) % int(self.chunk_size) or int(self.output_frames) % int(self.chunk_size):
            raise ValueError("context/output frames must contain complete causal chunks")
        return self

    @property
    def local_heads(self) -> int:
        return int(self.num_heads) // int(self.world_size)

    @property
    def tokens_per_chunk(self) -> int:
        return int(self.chunk_size) * int(self.frame_tokens)


def _buffer(torch: Any, geometry: CacheGeometry, capacity: int, heads: int, *, dtype: Any, device: Any) -> Any:
    return torch.empty(1, int(capacity), int(heads), int(geometry.head_dim), dtype=dtype, device=device)


def _counter(torch: Any, *, device: Any) -> Any:
    return torch.zeros(1, dtype=torch.long, device=device)


def _dense_layers(torch: Any, geometry: CacheGeometry, capacity: int, *, dtype: Any, device: Any, mode: str = "native") -> list[dict[str, Any]]:
    return [
        {
            "cache_mode": mode,
            "layer_index": layer,
            "k": _buffer(torch, geometry, capacity, geometry.local_heads, dtype=dtype, device=device),
            "v": _buffer(torch, geometry, capacity, geometry.local_heads, dtype=dtype, device=device),
            "global_end_index": _counter(torch, device=device),
            "local_end_index": _counter(torch, device=device),
        }
        for layer in range(int(geometry.num_layers))
    ]


def _local_ownership(head_map: HeadMap, geometry: CacheGeometry, layer: int, *, all_dynamic_layer_zero: bool) -> tuple[list[int], list[int]]:
    if all_dynamic_layer_zero and layer == 0:
        return [], list(range(geometry.local_heads))
    global_start = int(geometry.rank) * geometry.local_heads
    static_global = set(head_map.layer(layer).static_heads)
    static = [index for index in range(geometry.local_heads) if global_start + index in static_global]
    dynamic = [index for index in range(geometry.local_heads) if index not in static]
    return static, dynamic


def _head_split_layers(
    torch: Any,
    geometry: CacheGeometry,
    *,
    head_map: HeadMap,
    method: str,
    parameters: dict[str, Any],
    dtype: Any,
    device: Any,
) -> list[dict[str, Any]]:
    if head_map.num_layers != geometry.num_layers or head_map.num_heads != geometry.num_heads:
        raise ValueError("head map and cache geometry disagree")
    if method == "forcingkv":
        static_sink = int(parameters["static_sink_frames"])
        static_recent = int(parameters["static_recent_chunks"]) * geometry.chunk_size
        dynamic_sink = int(parameters["dynamic_sink_frames"])
        dynamic_recent = int(parameters["dynamic_recent_chunks"]) * geometry.chunk_size
        patch_count = int(parameters["selected_patch_count"])
        if geometry.frame_tokens % 6:
            raise ValueError("ForcingKV frame token count must be divisible by six patches")
        memory_tokens = patch_count * (geometry.frame_tokens // 6)
        mode = "physical_forcingkv"
        all_dynamic_layer_zero = False
        static_capacity = (static_sink + static_recent + geometry.chunk_size) * geometry.frame_tokens
        dynamic_capacity = (dynamic_sink + dynamic_recent + geometry.chunk_size) * geometry.frame_tokens + memory_tokens
    else:
        static_sink = static_recent = dynamic_sink = dynamic_recent = geometry.chunk_size
        mode = "physical_head_split"
        all_dynamic_layer_zero = True
        static_capacity = (static_sink + static_recent + geometry.chunk_size) * geometry.frame_tokens
        # A threshold policy is allowed to retain every token. Capacity must
        # therefore be correct independently of an assumed prune ratio.
        dynamic_capacity = (geometry.context_frames + geometry.output_frames) * geometry.frame_tokens
        memory_tokens = 0
    layers = []
    for layer in range(geometry.num_layers):
        static, dynamic = _local_ownership(head_map, geometry, layer, all_dynamic_layer_zero=all_dynamic_layer_zero)
        static_indices = torch.tensor(static, dtype=torch.long, device=device)
        dynamic_indices = torch.tensor(dynamic, dtype=torch.long, device=device)
        layers.append({
            "cache_mode": mode,
            "layer_index": layer,
            "global_end_index": _counter(torch, device=device),
            "local_end_index": _counter(torch, device=device),
            "static_local_end_index": _counter(torch, device=device),
            "dynamic_local_end_index": _counter(torch, device=device),
            "static_local_head_indices": static_indices,
            "dynamic_local_head_indices": dynamic_indices,
            "static_sink_tokens": static_sink * geometry.frame_tokens,
            "static_recent_tokens": static_recent * geometry.frame_tokens,
            "dynamic_sink_tokens": dynamic_sink * geometry.frame_tokens,
            "dynamic_recent_tokens": dynamic_recent * geometry.frame_tokens,
            "dynamic_memory_tokens": memory_tokens,
            "static_k": _buffer(torch, geometry, static_capacity, len(static), dtype=dtype, device=device),
            "static_v": _buffer(torch, geometry, static_capacity, len(static), dtype=dtype, device=device),
            "dynamic_k": _buffer(torch, geometry, dynamic_capacity, len(dynamic), dtype=dtype, device=device),
            "dynamic_v": _buffer(torch, geometry, dynamic_capacity, len(dynamic), dtype=dtype, device=device),
        })
    return layers


def allocate_case_cache(
    geometry: CacheGeometry,
    *,
    method: str,
    parameters: dict[str, Any],
    dtype: Any,
    device: Any,
    compute_device: Any | None = None,
    head_map: HeadMap | None = None,
    with_provenance: bool = True,
) -> list[dict[str, Any]]:
    """Allocate only physical state addressable by a method's lifecycle."""
    geometry.validate()
    import torch

    method = str(method)
    total_tokens = (geometry.context_frames + geometry.output_frames) * geometry.frame_tokens
    context_tokens = geometry.context_frames * geometry.frame_tokens
    if method == "fullkv":
        layers = _dense_layers(torch, geometry, total_tokens, dtype=dtype, device=device)
    elif method == "streaming":
        local = int(parameters["local_attn_size"])
        sink = int(parameters["sink_size"])
        if not 0 <= sink < local:
            raise ValueError("Streaming sink must be smaller than its local window")
        layers = _dense_layers(
            torch,
            geometry,
            (local + geometry.chunk_size) * geometry.frame_tokens,
            dtype=dtype,
            device=device,
        )
    elif method == "dummy_forcing":
        layers = _dense_layers(torch, geometry, 3 * geometry.tokens_per_chunk, dtype=dtype, device=device, mode="dummyforcing_warmup")
    elif method in {"forcingkv", "decoprune_hs"}:
        if head_map is None:
            raise ValueError(f"{method} requires a validated head map")
        layers = _head_split_layers(
            torch,
            geometry,
            head_map=head_map,
            method=method,
            parameters=parameters,
            dtype=dtype,
            device=device,
        )
    elif method == "patchification":
        layers = _dense_layers(torch, geometry, context_tokens, dtype=dtype, device=device)
    elif method in {"random", "decoprune"}:
        capacity = total_tokens
        layers = _dense_layers(torch, geometry, capacity, dtype=dtype, device=device, mode="physical_compacted")
    else:
        raise ValueError(f"unsupported cache method: {method}")
    if with_provenance:
        prefixes = {
            "physical_forcingkv": ("static_", "dynamic_"),
            "physical_head_split": ("static_", "dynamic_"),
        }
        for cache in layers:
            for prefix in prefixes.get(str(cache.get("cache_mode")), ("",)):
                key = cache[f"{prefix}k"]
                cache[f"{prefix}tpos"] = torch.full(
                    (1, int(key.shape[1]), 1, 1),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
                cache[f"{prefix}source_ids"] = torch.full(
                    (1, int(key.shape[1])),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
    for cache in layers:
        cache["method"] = method
        cache["storage_device"] = str(device)
        cache["compute_device"] = str(compute_device if compute_device is not None else device)
    return layers


def cache_memory_plan(cache_layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe persistent storage plus the two reusable GPU staging slots."""
    import torch

    layer_bytes: list[int] = []
    gpu_bytes = 0
    host_layer_bytes = []
    storage_devices: set[str] = set()
    for cache in cache_layers:
        total = 0
        for value in cache.values():
            if not isinstance(value, torch.Tensor):
                continue
            total += int(value.numel()) * int(value.element_size())
            storage_devices.add(str(value.device))
        layer_bytes.append(total)
        if cache.get("storage_device") == "cpu":
            host_layer_bytes.append(total)
        else:
            gpu_bytes += total
    offloaded = bool(host_layer_bytes)
    # The stager alternates even/odd slots, including without prefetch.
    staging_bytes = sum(
        max((size for index, size in enumerate(layer_bytes)
             if index % 2 == slot and cache_layers[index].get("storage_device") == "cpu"), default=0)
        for slot in (0, 1)
    )
    return {
        "storage_devices": sorted(storage_devices),
        "persistent_bytes": sum(layer_bytes),
        "largest_layer_bytes": max(layer_bytes, default=0),
        "staging_bytes": staging_bytes,
        "peak_gpu_cache_bytes": gpu_bytes + staging_bytes,
        "gpu_resident_bytes": gpu_bytes,
        "host_resident_bytes": sum(host_layer_bytes),
        "gpu_layers": len(layer_bytes) - len(host_layer_bytes),
        "host_layers": len(host_layer_bytes),
        "layer_bytes": layer_bytes,
        "offloaded": offloaded,
    }
