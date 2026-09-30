from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cmbench_rebuild.methods.head_map import HeadMap, LayerHeadOwnership
from cmbench_rebuild.methods.registry import resolve
from cmbench_rebuild.runtime.cache_attention import cached_attention
from cmbench_rebuild.runtime.cache_factory import CacheGeometry, allocate_case_cache
from cmbench_rebuild.runtime.online_policies import (
    ConsistencyRecord,
    OnlinePolicyRuntime,
    empty_physical_cache_at,
    isolated_probe_cache,
    restore_cache_cursors,
    restore_physical_cache,
    snapshot_cache_cursors,
    snapshot_physical_cache,
)
from cmbench_rebuild.runtime.q0 import patchification_q0


def _geometry(method: str = "fullkv") -> CacheGeometry:
    del method
    return CacheGeometry(2, 2, 2, 1, 0, 12, 4, 2, 6)


def _head_map() -> HeadMap:
    layers = (
        LayerHeadOwnership(0, (0,), (1,)),
        LayerHeadOwnership(1, (1,), (0,)),
    )
    return HeadMap("model", 2, 2, layers, Path("map.json"), "hash", {})


def _request(generation_policy: str = "method-native") -> SimpleNamespace:
    return SimpleNamespace(seed=2, raw={}, generation={"generation_kv_policy": generation_policy})


def _runtime(method: str, cache, geometry, *, emit=None) -> OnlinePolicyRuntime:
    spec = resolve(method)
    return OnlinePolicyRuntime(
        cache=cache,
        geometry=geometry,
        policy=SimpleNamespace(name=method, parameters=dict(spec.defaults)),
        request=_request(),
        pipeline=SimpleNamespace(),
        timesteps=torch.arange(4),
        emit=emit or (lambda row: None),
    )


def _write_chunk(cache, geometry, chunk_idx: int) -> None:
    tokens = geometry.tokens_per_chunk
    for layer_idx, layer in enumerate(cache):
        generator = torch.Generator().manual_seed(100 * layer_idx + chunk_idx)
        key = torch.randn(1, tokens, geometry.local_heads, geometry.head_dim, generator=generator)
        value = torch.randn(1, tokens, geometry.local_heads, geometry.head_dim, generator=generator)
        query = torch.randn(1, tokens, geometry.local_heads, geometry.head_dim, generator=generator)
        cached_attention(
            query,
            key,
            value,
            cache=layer,
            current_start=chunk_idx * tokens,
            frame_tokens=geometry.frame_tokens,
            local_attn_size=-1,
            sink_size=0,
            max_attention_size=(geometry.context_frames + geometry.output_frames) * geometry.frame_tokens,
            attender=lambda q, k, v: q,
        )


def test_patchification_can_update_block_ranking_after_each_generation_chunk():
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
    for chunk_idx in range(geometry.context_frames // geometry.chunk_size):
        _write_chunk(cache, geometry, chunk_idx)
    policy = OnlinePolicyRuntime(
        cache=cache,
        geometry=geometry,
        policy=SimpleNamespace(name="patchification", parameters=parameters),
        request=_request(),
        pipeline=SimpleNamespace(),
        timesteps=torch.arange(4),
        emit=lambda row: None,
    )
    q0 = patchification_q0(cache, geometry, parameters, online_policy=policy)
    initial_candidates = policy.diagnostics()["patchification_online_candidates"]
    assert q0["update_each_chunk"] is True
    assert initial_candidates == (8 - 1 - 1) * 4
    _write_chunk(cache, geometry, chunk_idx=4)
    policy.after_clean(chunk_idx=4, clean_x=torch.empty(1), context_scores=None, phase="generation")
    state = policy.diagnostics()
    assert state["patchification_online_candidates"] > initial_candidates
    assert policy.events[-1]["event"] == "patchification_online_update"
    assert policy.events[-1]["new_pair_starts_scored"] > 0


def test_physical_snapshot_restores_head_split_probe_exactly():
    geometry = _geometry()
    cache = allocate_case_cache(
        geometry,
        method="decoprune_hs",
        parameters=dict(resolve("decoprune_hs").defaults),
        dtype=torch.float32,
        device="cpu",
        head_map=_head_map(),
    )
    _write_chunk(cache, geometry, 0)
    snapshot = snapshot_physical_cache(cache)
    empty_physical_cache_at(cache, geometry.tokens_per_chunk)
    _write_chunk(cache, geometry, 1)
    restore_physical_cache(cache, snapshot)
    assert cache[0]["global_end_index"].item() == geometry.tokens_per_chunk
    assert torch.equal(cache[1]["dynamic_k"][:, :geometry.tokens_per_chunk], snapshot[1]["dynamic_k"])
    assert torch.equal(cache[0]["static_source_ids"][:, :geometry.tokens_per_chunk], snapshot[0]["static_source_ids"])


def test_isolated_probe_cache_preserves_live_history_and_head_layout():
    geometry = _geometry()
    cache = allocate_case_cache(
        geometry,
        method="decoprune_hs",
        parameters=dict(resolve("decoprune_hs").defaults),
        dtype=torch.float32,
        device="cpu",
        head_map=_head_map(),
    )
    _write_chunk(cache, geometry, 0)
    original_dynamic = cache[0]["dynamic_k"][:, :geometry.tokens_per_chunk].clone()
    cursors = snapshot_cache_cursors(cache)
    scratch = isolated_probe_cache(cache, capacity_tokens=geometry.tokens_per_chunk)
    empty_physical_cache_at(scratch, geometry.tokens_per_chunk)
    _write_chunk(scratch, geometry, 1)
    restore_cache_cursors(cache, cursors)
    assert scratch[0]["dynamic_k"].data_ptr() != cache[0]["dynamic_k"].data_ptr()
    assert scratch[0]["dynamic_k"].shape[2] == cache[0]["dynamic_k"].shape[2]
    assert torch.equal(cache[0]["dynamic_k"][:, :geometry.tokens_per_chunk], original_dynamic)
    assert cache[0]["global_end_index"].item() == geometry.tokens_per_chunk


def test_head_split_probe_cannot_turn_its_noisy_tail_into_static_history():
    from cmbench_rebuild.runtime.q0 import cache_identity_snapshot

    geometry = _geometry()
    cache = allocate_case_cache(geometry, method="decoprune_hs",
        parameters=dict(resolve("decoprune_hs").defaults), dtype=torch.float32,
        device="cpu", head_map=_head_map())
    for chunk in range(3):
        _write_chunk(cache, geometry, chunk)
    tokens = geometry.tokens_per_chunk
    expected = torch.cat([cache[1]["static_k"][:, :tokens], cache[1]["static_k"][:, 2*tokens:3*tokens]], dim=1).clone()
    runtime = _runtime("decoprune_hs", cache, geometry)

    def probe_model(*, x, t, kv_cache, current_start, **unused):
        _write_chunk(kv_cache, geometry, current_start // tokens)
        for layer in kv_cache:
            for prefix in ("static_", "dynamic_"):
                end = int(layer[prefix + "local_end_index"].item())
                layer[prefix + "k"][:, end-tokens:end].add_(1000)  # unmistakably noisy probe values
        return [torch.zeros_like(x[0])]

    runtime.pipeline = SimpleNamespace(model=probe_model,
        scheduler=SimpleNamespace(add_noise=lambda clean, noise, step: clean),
        _convert_flow_pred_to_x0=lambda flow, noisy, step, scheduler: flow)
    runtime.context_probe(chunk_idx=3, clean_x=torch.zeros(1, 2, 4, 6),
                          kwargs={"kv_cache": cache, "current_start": 3*tokens})
    assert cache[1]["static_local_end_index"].item() == 2*tokens
    assert torch.equal(cache[1]["static_k"][:, :2*tokens], expected)
    _write_chunk(cache, geometry, 3)
    cache_identity_snapshot(cache)  # Includes the zero-static-head first layer.
    assert torch.equal(cache[1]["static_k"][:, :2*tokens], expected)
    assert cache[1]["static_source_ids"][0, :3*tokens].tolist() == list(range(tokens)) + list(range(2*tokens, 4*tokens))


def test_consistency_compaction_uses_source_identity_not_stale_offsets():
    geometry = _geometry()
    cache = allocate_case_cache(
        geometry,
        method="decoprune",
        parameters=dict(resolve("decoprune").defaults),
        dtype=torch.float32,
        device="cpu",
    )
    for chunk_idx in range(4):
        _write_chunk(cache, geometry, chunk_idx)
    runtime = _runtime("decoprune", cache, geometry)
    ids = torch.arange(geometry.tokens_per_chunk, 2 * geometry.tokens_per_chunk)
    scores = torch.tensor([0.0, 0.2] * (geometry.tokens_per_chunk // 2))
    runtime.records.append(ConsistencyRecord(1, ids, scores, "context"))
    runtime._compact_consistency(next_chunk_idx=4)
    live = cache[0]["local_end_index"].item()
    retained = cache[0]["source_ids"][0, :live]
    assert not torch.isin(ids[::2], retained).any()
    assert torch.isin(ids[1::2], retained).all()
    assert runtime.records[0].applied


def test_forcingkv_online_bank_is_bounded_and_identity_ordered():
    geometry = _geometry()
    params = dict(resolve("forcingkv").defaults)
    params["selected_patch_count"] = 3
    cache = allocate_case_cache(
        geometry,
        method="forcingkv",
        parameters=params,
        dtype=torch.float32,
        device="cpu",
        head_map=_head_map(),
    )
    runtime = OnlinePolicyRuntime(
        cache=cache,
        geometry=geometry,
        policy=SimpleNamespace(name="forcingkv", parameters=params),
        request=_request(),
        pipeline=SimpleNamespace(),
        timesteps=torch.arange(4),
        emit=lambda row: None,
    )
    for chunk_idx in range(4):
        _write_chunk(cache, geometry, chunk_idx)
        runtime.after_clean(chunk_idx=chunk_idx, clean_x=torch.zeros(1), context_scores=None, phase="context")
        for layer in cache:
            live = layer["dynamic_local_end_index"].item()
            ids = layer["dynamic_source_ids"][0, :live]
            assert bool((ids[1:] > ids[:-1]).all())
            assert live <= int(layer["dynamic_k"].shape[1])
    assert len(runtime.forcing_patch_starts) <= 3


def test_dummy_center_bank_survives_rolling_recent_updates():
    geometry = _geometry()
    params = dict(resolve("dummy_forcing").defaults)
    cache = allocate_case_cache(
        geometry,
        method="dummy_forcing",
        parameters=params,
        dtype=torch.float32,
        device="cpu",
    )
    runtime = OnlinePolicyRuntime(
        cache=cache,
        geometry=geometry,
        policy=SimpleNamespace(name="dummy_forcing", parameters=params),
        request=_request(),
        pipeline=SimpleNamespace(),
        timesteps=torch.arange(4),
        emit=lambda row: None,
    )
    for chunk_idx in range(3):
        runtime.prepare_chunk(chunk_idx)
        if chunk_idx == 2:
            # The method remains correct even if transient wrapper metadata is
            # stripped; absolute third-chunk structure is authoritative.
            for layer in cache:
                layer.pop("dummyforcing_score_pending", None)
                layer.pop("dummyforcing_score_at_global_end", None)
        _write_chunk(cache, geometry, chunk_idx)
        runtime.capture_model_state()
    runtime.after_clean(chunk_idx=2, clean_x=torch.zeros(1), context_scores=None, phase="context")
    fixed = set(runtime._dummy_middle_bank_ids)
    assert cache[0]["cache_mode"] == "physical_dummyforcing"
    _write_chunk(cache, geometry, 3)
    runtime.after_clean(chunk_idx=3, clean_x=torch.zeros(1), context_scores=None, phase="context")
    live = cache[0]["dummy_middle_local_end_index"].item()
    retained = set(cache[0]["dummy_middle_source_ids"][0, :live].tolist())
    assert fixed <= retained
    assert set(range(3 * geometry.tokens_per_chunk, 4 * geometry.tokens_per_chunk)) <= retained
