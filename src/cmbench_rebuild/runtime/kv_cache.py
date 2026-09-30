"""Physical KV bank discovery and q0-to-generation lifecycle transition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


HEAD_SPLIT_MODES = frozenset({"physical_head_split", "physical_forcingkv"})
DUMMY_MODE = "physical_dummyforcing"


@dataclass(frozen=True)
class CacheBank:
    prefix: str
    key_name: str
    value_name: str
    end_name: str


def cache_banks(cache: dict[str, Any]) -> tuple[CacheBank, ...]:
    mode = str(cache.get("cache_mode") or "native")
    if mode in HEAD_SPLIT_MODES:
        prefixes = ("static", "dynamic")
    elif mode == DUMMY_MODE:
        prefixes = ("dummy_first", "dummy_middle", "dummy_last")
    else:
        prefixes = ("",)
    return tuple(
        CacheBank(
            prefix=prefix,
            key_name=f"{prefix}_k" if prefix else "k",
            value_name=f"{prefix}_v" if prefix else "v",
            end_name=f"{prefix}_local_end_index" if prefix else "local_end_index",
        )
        for prefix in prefixes
    )


def _scalar(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _grow_live_prefix(tensor: Any, *, live_tokens: int, output_tokens: int) -> Any:
    if len(tensor.shape) < 2:
        raise ValueError("cache buffers must have a token dimension at axis 1")
    live, extra = int(live_tokens), int(output_tokens)
    if live < 0 or extra < 0 or live > int(tensor.shape[1]):
        raise ValueError(f"invalid cache resize: live={live}, output={extra}, capacity={tensor.shape[1]}")
    required = live + extra
    if int(tensor.shape[1]) >= required:
        return tensor
    shape = list(tensor.shape)
    shape[1] = required
    result = tensor.new_empty(shape)
    if live:
        result[:, :live].copy_(tensor[:, :live])
    return result


def describe_cache_layer(cache: dict[str, Any], *, layer_idx: int) -> dict[str, Any]:
    banks = []
    for bank in cache_banks(cache):
        key = cache[bank.key_name]
        value = cache[bank.value_name]
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError(f"K/V shape mismatch for bank {bank.prefix or 'dense'}")
        live = _scalar(cache[bank.end_name])
        if live < 0 or live > int(key.shape[1]):
            raise ValueError(f"invalid live prefix for bank {bank.prefix or 'dense'}")
        banks.append({
            "bank": bank.prefix or "dense", "live_tokens": live,
            "capacity_tokens": int(key.shape[1]), "heads": int(key.shape[2]), "head_dim": int(key.shape[3]),
        })
    return {"layer_idx": int(layer_idx), "cache_mode": str(cache.get("cache_mode") or "native"), "banks": banks}


def prepare_append_only_generation(kv_cache: Iterable[dict[str, Any]], *, output_tokens: int) -> list[dict[str, Any]]:
    """Freeze every q0 context bank and reserve an unpruned output tail."""
    output = int(output_tokens)
    if output <= 0:
        raise ValueError("output_tokens must be positive")
    caches = list(kv_cache)
    for cache in caches:
        banks = cache_banks(cache)
        ends: list[int] = []
        cache["append_only_generation"] = True
        cache["generation_tokens_reserved"] = output
        for bank in banks:
            end = _scalar(cache[bank.end_name])
            ends.append(end)
            frozen_name = f"frozen_{bank.prefix}_context_end" if bank.prefix else "frozen_context_end"
            cache[frozen_name] = end
            cache[bank.key_name] = _grow_live_prefix(cache[bank.key_name], live_tokens=end, output_tokens=output)
            cache[bank.value_name] = _grow_live_prefix(cache[bank.value_name], live_tokens=end, output_tokens=output)
            provenance_name = f"{bank.prefix}_tpos" if bank.prefix else "tpos"
            if provenance_name in cache:
                cache[provenance_name] = _grow_live_prefix(
                    cache[provenance_name], live_tokens=end, output_tokens=output
                )
            source_name = f"{bank.prefix}_source_ids" if bank.prefix else "source_ids"
            if source_name in cache:
                cache[source_name] = _grow_live_prefix(
                    cache[source_name], live_tokens=end, output_tokens=output
                )
        cache["frozen_context_end"] = max(ends)
        if str(cache.get("cache_mode") or "native") not in HEAD_SPLIT_MODES | {DUMMY_MODE}:
            cache["cache_mode"] = "physical_frozen_dense"
    return caches
