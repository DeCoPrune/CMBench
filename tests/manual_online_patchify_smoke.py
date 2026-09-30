from types import SimpleNamespace

import torch

from cmbench_rebuild.methods.registry import resolve
from cmbench_rebuild.runtime.cache_attention import cached_attention
from cmbench_rebuild.runtime.cache_factory import CacheGeometry, allocate_case_cache
from cmbench_rebuild.runtime.online_policies import OnlinePolicyRuntime
from cmbench_rebuild.runtime.q0 import patchification_q0


geometry = CacheGeometry(2, 2, 2, 1, 0, 8, 4, 2, 16)
parameters = {
    **dict(resolve("patchification").defaults),
    "sink_frames": 1,
    "recent_frames": 1,
    "grid_rows": 2,
    "grid_cols": 2,
    "topk_blocks": 2,
    "update_each_chunk": True,
}
cache = allocate_case_cache(
    geometry,
    method="patchification",
    parameters=parameters,
    dtype=torch.float32,
    device="cpu",
)


def write_chunk(chunk_idx):
    for layer_idx, layer in enumerate(cache):
        generator = torch.Generator().manual_seed(100 * layer_idx + chunk_idx)
        count = geometry.tokens_per_chunk
        key = torch.randn(1, count, geometry.local_heads, geometry.head_dim, generator=generator)
        value = torch.randn(key.shape, generator=generator)
        query = torch.randn(key.shape, generator=generator)
        cached_attention(
            query,
            key,
            value,
            cache=layer,
            current_start=chunk_idx * count,
            frame_tokens=geometry.frame_tokens,
            local_attn_size=-1,
            sink_size=0,
            max_attention_size=(geometry.context_frames + geometry.output_frames) * geometry.frame_tokens,
            attender=lambda q, k, v: q,
        )


for chunk in range(geometry.context_frames // geometry.chunk_size):
    write_chunk(chunk)
runtime = OnlinePolicyRuntime(
    cache=cache,
    geometry=geometry,
    policy=SimpleNamespace(name="patchification", parameters=parameters),
    request=SimpleNamespace(seed=2, raw={}, generation={"generation_kv_policy": "method-native"}),
    pipeline=SimpleNamespace(),
    timesteps=torch.arange(4),
    emit=lambda _row: None,
)
q0 = patchification_q0(cache, geometry, parameters, online_policy=runtime)
before = runtime.diagnostics()["patchification_online_candidates"]
write_chunk(geometry.context_frames // geometry.chunk_size)
runtime.after_clean(
    chunk_idx=geometry.context_frames // geometry.chunk_size,
    clean_x=torch.empty(1),
    context_scores=None,
    phase="generation",
)
after = runtime.diagnostics()["patchification_online_candidates"]
assert q0["update_each_chunk"] is True
assert before == (geometry.context_frames - 1 - parameters["recent_frames"]) * parameters["grid_rows"] * parameters["grid_cols"]
assert after > before
assert runtime.events[-1]["event"] == "patchification_online_update"
assert runtime.events[-1]["new_pair_starts_scored"] > 0
print({"candidates_before": before, "candidates_after": after, "update": runtime.events[-1]})
