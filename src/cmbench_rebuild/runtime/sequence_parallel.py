"""LingBot Ulysses adapter whose physical KV semantics live in this project."""
from __future__ import annotations

import math
from typing import Any

from .cache_attention import cached_attention


def policy_attn_forward_causal(
    self: Any,
    x: Any,
    seq_lens: Any,
    grid_sizes: Any,
    freqs: Any,
    kv_cache: dict[str, Any] | None = None,
    current_start: int = 0,
    max_attention_size: int = 1_000_000,
    frame_seqlen: int | None = None,
    seq_lens_int: int | None = None,
) -> Any:
    """Single-device equivalent of the policy-aware Ulysses path."""
    from wan.modules.attention import attention
    from wan.modules.model_fast import causal_rope_apply

    del seq_lens
    if kv_cache is None:
        raise ValueError("causal attention requires a KV cache")
    batch, sequence = int(x.shape[0]), int(x.shape[1])
    heads, head_dim = int(self.num_heads), int(self.head_dim)
    query = self.norm_q(self.q(x)).view(batch, sequence, heads, head_dim)
    key = self.norm_k(self.k(x)).view(batch, sequence, heads, head_dim)
    value = self.v(x).view(batch, sequence, heads, head_dim)
    valid_sequence = sequence if seq_lens_int is None else int(seq_lens_int)
    if frame_seqlen is None:
        frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
    start_frame = int(current_start) // int(frame_seqlen)
    query = causal_rope_apply(query, grid_sizes, freqs, start_frame=start_frame).type_as(value)[:, :valid_sequence]
    key = causal_rope_apply(key, grid_sizes, freqs, start_frame=start_frame).type_as(value)[:, :valid_sequence]
    value = value[:, :valid_sequence]
    attended = cached_attention(
        query,
        key,
        value,
        cache=kv_cache,
        current_start=int(current_start),
        frame_tokens=int(frame_seqlen),
        local_attn_size=int(self.local_attn_size),
        sink_size=int(self.sink_size),
        max_attention_size=int(max_attention_size),
        attender=attention,
    )
    return self.o(attended.flatten(2))


def sp_policy_attn_forward_causal(
    self: Any,
    x: Any,
    seq_lens: Any,
    grid_sizes: Any,
    freqs: Any,
    kv_cache: dict[str, Any] | None = None,
    current_start: int = 0,
    max_attention_size: int = 1_000_000,
    frame_seqlen: int | None = None,
    seq_lens_int: int | None = None,
) -> Any:
    """Ulysses all-to-all plus the method-neutral physical-cache operator."""
    import torch
    from wan.distributed.sequence_parallel import causal_rope_apply
    from wan.distributed.ulysses import all_to_all
    from wan.distributed.util import get_world_size
    from wan.modules.attention import flash_attention

    if kv_cache is None:
        raise ValueError("causal sequence-parallel attention requires a KV cache")
    batch, local_sequence = int(x.shape[0]), int(x.shape[1])
    heads, head_dim = int(self.num_heads), int(self.head_dim)
    query = self.norm_q(self.q(x)).view(batch, local_sequence, heads, head_dim)
    key = self.norm_k(self.k(x)).view(batch, local_sequence, heads, head_dim)
    value = self.v(x).view(batch, local_sequence, heads, head_dim)
    query = all_to_all(query, scatter_dim=2, gather_dim=1)
    key = all_to_all(key, scatter_dim=2, gather_dim=1)
    value = all_to_all(value, scatter_dim=2, gather_dim=1)
    parallel_size = int(get_world_size())
    padded_sequence = local_sequence * parallel_size
    valid_sequence = int(seq_lens) if seq_lens_int is None else int(seq_lens_int)
    if frame_seqlen is None:
        frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
    start_frame = int(current_start) // int(frame_seqlen)
    query = causal_rope_apply(query, grid_sizes, freqs, start_frame=start_frame).type_as(value)[:, :valid_sequence]
    key = causal_rope_apply(key, grid_sizes, freqs, start_frame=start_frame).type_as(value)[:, :valid_sequence]
    value = value[:, :valid_sequence]
    attended = cached_attention(
        query,
        key,
        value,
        cache=kv_cache,
        current_start=int(current_start),
        frame_tokens=int(frame_seqlen),
        local_attn_size=int(self.local_attn_size),
        sink_size=int(self.sink_size),
        max_attention_size=int(max_attention_size),
        attender=flash_attention,
    )
    padding = padded_sequence - valid_sequence
    if padding > 0:
        attended = torch.cat(
            [attended, attended.new_zeros(batch, padding, attended.shape[2], head_dim)],
            dim=1,
        )
    output = all_to_all(attended, scatter_dim=1, gather_dim=2).flatten(2)
    return self.o(output)
