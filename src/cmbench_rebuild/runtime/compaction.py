"""Identity-preserving physical cache compaction primitives."""
from __future__ import annotations

from typing import Any, Iterable


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def movable_names(cache: dict[str, Any], prefix: str = "") -> tuple[str, ...]:
    return tuple(
        f"{prefix}{suffix}"
        for suffix in ("k", "v", "tpos", "source_ids")
        if f"{prefix}{suffix}" in cache
    )


def compact_static_predecessor(cache: dict[str, Any]) -> None:
    """Prepare the static sink/recent window before any reversible HS probe."""
    import torch

    end = _int(cache["static_local_end_index"])
    sink = min(_int(cache["static_sink_tokens"]), end)
    recent = min(_int(cache["static_recent_tokens"]), max(0, end - sink))
    recent_start = end - recent
    if recent_start <= sink:
        return
    compacted = sink + recent
    for name in movable_names(cache, "static_"):
        buffer = cache[name]
        pieces = [buffer[:, :sink].clone(), buffer[:, recent_start:end].clone()]
        buffer[:, :compacted] = torch.cat(pieces, dim=1)
    cache["static_local_end_index"].fill_(compacted)


def compact_range(cache_layers: Iterable[dict[str, Any]], *, token_start: int, keep_mask: Any, prefix: str = "") -> int:
    """Pack one shared physical range while preserving the remaining suffix."""
    import torch

    layers = list(cache_layers)
    if not layers:
        return 0
    end_name = f"{prefix}local_end_index" if prefix else "local_end_index"
    device_name = f"{prefix}k" if prefix else "k"
    mask = keep_mask.to(dtype=torch.bool).reshape(-1)
    total, kept = int(mask.numel()), int(mask.sum().item())
    start, stop = int(token_start), int(token_start) + total
    for cache in layers:
        layer_mask = mask.to(cache[device_name].device)
        live = _int(cache[end_name])
        if start < 0 or stop > live:
            raise ValueError(f"compaction range [{start},{stop}) is outside {prefix or 'dense'} live prefix {live}")
        for name in movable_names(cache, prefix):
            values = cache[name]
            selected = values[:, start:stop][:, layer_mask].clone()
            values[:, start:start + kept] = selected
            if stop < live:
                suffix = values[:, stop:live].clone()
                values[:, start + kept:start + kept + suffix.shape[1]] = suffix
            if name.endswith("tpos") or name.endswith("source_ids"):
                values[:, live - (total - kept):live] = -1
        new_live = live - (total - kept)
        cache[end_name].fill_(new_live)
        if prefix in {"static_", "dynamic_"}:
            other = "dynamic_local_end_index" if prefix == "static_" else "static_local_end_index"
            cache["local_end_index"].fill_(max(new_live, _int(cache[other])))
    return total - kept


def keep_global_source_ids(cache_layers: Iterable[dict[str, Any]], source_ids: Any, *, prefix: str = "") -> dict[str, Any]:
    """At q0, retain an explicit set of global token identities in source order."""
    import torch

    layers = list(cache_layers)
    if not layers:
        raise ValueError("cache layer list is empty")
    source_name = f"{prefix}source_ids" if prefix else "source_ids"
    end_name = f"{prefix}local_end_index" if prefix else "local_end_index"
    wanted = source_ids.to(device=layers[0][source_name].device, dtype=torch.long).reshape(-1)
    if wanted.numel() and not bool((wanted[1:] > wanted[:-1]).all()):
        raise ValueError("source IDs must be strictly increasing")
    removed_by_layer = []
    retained_reference = None
    wanted_by_device = {}
    reference_by_device = {}
    for cache in layers:
        live = _int(cache[end_name])
        physical = cache[source_name][0, :live]
        if physical.device not in wanted_by_device:
            wanted_by_device[physical.device] = wanted.to(physical.device)
        mask = torch.isin(physical, wanted_by_device[physical.device])
        retained = physical[mask]
        if retained_reference is None:
            retained_reference = retained
        else:
            if retained.device not in reference_by_device:
                reference_by_device[retained.device] = retained_reference.to(retained.device)
            if not torch.equal(retained, reference_by_device[retained.device]):
                raise RuntimeError("cache layers disagree on source-token identity at q0")
        removed_by_layer.append(compact_range([cache], token_start=0, keep_mask=mask, prefix=prefix))
    missing = wanted if retained_reference is None else wanted[~torch.isin(wanted, retained_reference)]
    if int(missing.numel()):
        raise ValueError(f"requested {missing.numel()} source IDs absent from physical cache")
    return {
        "kept_tokens": int(wanted.numel()),
        "removed_tokens_by_layer": removed_by_layer,
        "source_id_min": int(wanted[0].item()) if wanted.numel() else None,
        "source_id_max": int(wanted[-1].item()) if wanted.numel() else None,
    }


def resize_live_banks(cache_layers: Iterable[dict[str, Any]], *, capacity: int, prefixes: tuple[str, ...] = ("",)) -> None:
    """Resize one bank at a time after q0, never dropping live state."""
    import torch

    target = int(capacity)
    if target <= 0:
        raise ValueError("cache capacity must be positive")
    for cache in cache_layers:
        for prefix in prefixes:
            end_name = f"{prefix}local_end_index" if prefix else "local_end_index"
            live = _int(cache[end_name])
            if live > target:
                raise ValueError(f"cannot resize {prefix or 'dense'} bank below live tokens")
            for name in movable_names(cache, prefix):
                source = cache[name]
                shape = list(source.shape)
                shape[1] = target
                resized = torch.empty(shape, dtype=source.dtype, device=source.device,
                                      pin_memory=source.device.type == "cpu" and source.is_pinned())
                resized[:, :live].copy_(source[:, :live])
                if name.endswith("tpos") or name.endswith("source_ids"):
                    resized[:, live:].fill_(-1)
                cache[name] = resized
