"""Validated ownership map for heterogeneous static/dynamic head policies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerHeadOwnership:
    layer_idx: int
    static_heads: tuple[int, ...]
    dynamic_heads: tuple[int, ...]


@dataclass(frozen=True)
class HeadMap:
    model: str
    num_layers: int
    num_heads: int
    layers: tuple[LayerHeadOwnership, ...]
    source: Path
    sha256: str
    metadata: dict[str, Any]

    def layer(self, layer_idx: int) -> LayerHeadOwnership:
        index = int(layer_idx)
        if not 0 <= index < self.num_layers:
            raise ValueError(f"layer_idx {index} is outside [0, {self.num_layers})")
        return self.layers[index]


def load_head_map(path: Path, *, expected_model: str, num_layers: int, num_heads: int) -> HeadMap:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    model = str(payload.get("model") or "")
    layers_count, heads_count = int(payload.get("num_layers", -1)), int(payload.get("num_heads", -1))
    if model != expected_model:
        raise ValueError(f"head-map model mismatch: {model!r} != {expected_model!r}")
    if (layers_count, heads_count) != (int(num_layers), int(num_heads)):
        raise ValueError(f"head-map shape mismatch: {layers_count}x{heads_count} != {num_layers}x{num_heads}")
    indexed: dict[int, LayerHeadOwnership] = {}
    for raw in payload.get("layers", []):
        layer_idx = int(raw["layer_idx"])
        if layer_idx in indexed:
            raise ValueError(f"duplicate head-map layer: {layer_idx}")
        static = tuple(sorted({int(value) for value in raw["static_head"]}))
        dynamic = tuple(sorted({int(value) for value in raw["dynamic_head"]}))
        if set(static) & set(dynamic) or set(static) | set(dynamic) != set(range(heads_count)):
            raise ValueError(f"head ownership is not a partition at layer {layer_idx}")
        indexed[layer_idx] = LayerHeadOwnership(layer_idx, static, dynamic)
    if set(indexed) != set(range(layers_count)):
        raise ValueError("head map does not cover every model layer")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "profile": payload.get("profile"),
        "calibration": payload.get("calibration", {}),
        "stability": payload.get("stability", {}),
        "ownership_summary": payload.get("ownership_summary", {}),
    }
    return HeadMap(model, layers_count, heads_count, tuple(indexed[index] for index in range(layers_count)), source, digest, metadata)
