from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cmbench_rebuild.core.rope import RopeReindexPlan
from cmbench_rebuild.core.metrics import active_history_token_head_units_for_source_range
from cmbench_rebuild.runtime.cache_factory import CacheGeometry, allocate_case_cache
from cmbench_rebuild.runtime.q0 import apply_rope_reindex, cache_identity_snapshot, patchification_q0, token_union


def _filled_dense(geometry: CacheGeometry, *, head_dim: int | None = None):
    if head_dim is not None:
        geometry = CacheGeometry(**{**vars(geometry), "head_dim": head_dim})
    layers = allocate_case_cache(
        geometry,
        method="fullkv",
        parameters={},
        dtype=torch.float32,
        device="cpu",
    )
    live = geometry.context_frames * geometry.frame_tokens
    for layer_index, layer in enumerate(layers):
        layer["local_end_index"].fill_(live)
        layer["global_end_index"].fill_(live)
        layer["source_ids"][0, :live] = torch.arange(live)
        layer["tpos"][0, :live, 0, 0] = torch.arange(live).div(geometry.frame_tokens, rounding_mode="floor")
        layer["k"][:, :live] = torch.arange(live).reshape(1, live, 1, 1) + layer_index
        layer["v"][:, :live] = layer["k"][:, :live]
    return geometry, layers


def test_q0_snapshot_and_union_validate_ordered_source_identity():
    geometry = CacheGeometry(2, 1, 1, 1, 0, 4, 2, 2, 3)
    _, layers = _filled_dense(geometry)
    assert token_union(layers) == tuple(range(12))
    snapshot = cache_identity_snapshot(layers)
    assert snapshot["layers"][0]["banks"][0]["live_tokens"] == 12
    assert len(snapshot["layers"][0]["banks"][0]["source_ids_sha256"]) == 64


def test_rope_reindex_changes_old_keys_but_preserves_recent_tail():
    geometry = CacheGeometry(1, 1, 4, 1, 0, 4, 2, 2, 1)
    geometry, layers = _filled_dense(geometry, head_dim=4)
    layers[0]["k"][:, :4].fill_(0)
    layers[0]["k"][:, :4, :, 0] = 1
    layers[0]["k"][:, :4, :, 2] = 1
    before = layers[0]["k"].clone()
    frequencies = torch.polar(torch.ones(16, 2, dtype=torch.float64), torch.outer(torch.arange(16, dtype=torch.float64), torch.tensor([1.0, 0.5])))
    result = apply_rope_reindex(
        layers,
        plan=RopeReindexPlan(mode="all_bands", virtual_span=2.5, recent_frames=1),
        context_frames=4,
        frame_tokens=1,
        temporal_rope_frequencies=frequencies,
    )
    assert result["rotated_tokens"] == 4
    assert not torch.equal(layers[0]["k"][:, 0], before[:, 0])
    assert torch.equal(layers[0]["k"][:, 3], before[:, 3])


def test_patchification_q0_selects_fixed_sink_recent_and_lowest_blocks():
    geometry = CacheGeometry(2, 1, 2, 1, 0, 6, 2, 2, 24)
    layers = allocate_case_cache(
        geometry,
        method="patchification",
        parameters={},
        dtype=torch.float32,
        device="cpu",
    )
    live = geometry.context_frames * geometry.frame_tokens
    for layer_index, layer in enumerate(layers):
        layer["local_end_index"].fill_(live)
        layer["global_end_index"].fill_(live)
        layer["source_ids"][0, :live] = torch.arange(live)
        layer["tpos"][0, :live, 0, 0] = torch.arange(live).div(geometry.frame_tokens, rounding_mode="floor")
        values = torch.arange(live * 2, dtype=torch.float32).reshape(1, live, 1, 2) + layer_index
        layer["k"][:, :live] = values
        layer["v"][:, :live] = values
    result = patchification_q0(
        layers,
        geometry,
        {"sink_frames": 1, "recent_frames": 1, "grid_rows": 2, "grid_cols": 3, "topk_blocks": 2},
    )
    assert all(layer["cache_mode"] == "physical_compacted" for layer in layers)
    assert result["candidate_blocks"] == 24
    assert result["selected_blocks"] == 2
    assert result["selected_tokens"] == 56
    assert all(layer["local_end_index"].item() == 56 for layer in layers)


def test_source_range_accounting_splits_context_from_generated_history():
    geometry = CacheGeometry(2, 1, 1, 1, 0, 4, 2, 2, 3)
    _, layers = _filled_dense(geometry)
    # Keep original source IDs 0..11; 0..5 are context and 6..11 are generated.
    # Accounting is in token-head-layer units, so each six-token range counts
    # once for each of the two layers (and one local head).
    assert active_history_token_head_units_for_source_range(
        layers, current_noisy_tokens=0, source_start=0, source_stop=6
    ) == 12
    assert active_history_token_head_units_for_source_range(
        layers, current_noisy_tokens=0, source_start=6, source_stop=12
    ) == 12
