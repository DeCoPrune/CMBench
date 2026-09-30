"""Single, strict JSON configuration schema for every run."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from .methods import resolve

@dataclass(frozen=True)
class RunConfig:
    protocol: str
    method: str
    case_id: str
    seed: int
    checkpoint: str
    dataset_version: str
    request_file: str
    head_map_file: str | None = None
    output_root: str = "runs"
    height: int = 480
    width: int = 832
    sampling_shift: float = 5.0
    chunk_size: int = 4
    timesteps_index: tuple[int, ...] = (0, 179, 358, 679)
    model_fps: float = 16.0
    vae_temporal_stride: int = 4
    world_size: int = 4
    kv_storage_mode: str = "auto_80gb"
    gpu_memory_limit_gib: float = 80.0
    generation_kv_policy: str = "method-native"
    rope_reindex: dict[str, Any] | None = None
    method_params: dict[str, Any] | None = None

    def normalized(self) -> dict[str, Any]:
        data = asdict(self)
        spec = resolve(self.method)
        supplied_method = self.method_params or {}
        unknown_method = set(supplied_method) - set(spec.defaults)
        if unknown_method:
            raise ValueError(f"unknown {spec.id} method parameters: {sorted(unknown_method)}")
        default_rope = {"mode": "all_bands", "virtual_span": 17, "recent_frames": 8, "fast_band_pairs": 11}
        supplied_rope = self.rope_reindex or {}
        unknown_rope = set(supplied_rope) - set(default_rope)
        if unknown_rope:
            raise ValueError(f"unknown RoPE parameters: {sorted(unknown_rope)}")
        data["method"] = spec.id
        data["legacy_method"] = spec.legacy_id
        data["method_params"] = spec.defaults | supplied_method
        data["rope_reindex"] = default_rope | supplied_rope
        if data["height"] <= 0 or data["width"] <= 0 or data["seed"] < 0 or data["world_size"] <= 0:
            raise ValueError("height/width/world_size must be positive and seed must be non-negative")
        if int(data["chunk_size"]) <= 0 or float(data["model_fps"]) <= 0 or int(data["vae_temporal_stride"]) <= 0:
            raise ValueError("chunk_size/model_fps/VAE temporal stride must be positive")
        if not data["timesteps_index"] or any(int(value) < 0 for value in data["timesteps_index"]):
            raise ValueError("timesteps_index must be a non-empty sequence of non-negative values")
        if data["generation_kv_policy"] not in {"append-only", "method-native"}:
            raise ValueError("generation_kv_policy must be append-only or method-native")
        if data["kv_storage_mode"] not in {"auto_80gb", "cuda", "cpu_offload"}:
            raise ValueError("kv_storage_mode must be auto_80gb, cuda, or cpu_offload")
        if not math.isfinite(float(data["gpu_memory_limit_gib"])) or data["gpu_memory_limit_gib"] <= 0:
            raise ValueError("gpu_memory_limit_gib must be finite and positive")
        rope = data["rope_reindex"]
        if rope["mode"] not in {"off", "all_bands", "fast_bands"}:
            raise ValueError("invalid RoPE re-index mode")
        if int(rope["recent_frames"]) < 0 or int(rope["fast_band_pairs"]) < 1:
            raise ValueError("invalid RoPE recent frames or fast band pairs")
        if rope["mode"] != "off" and float(rope["virtual_span"]) <= int(rope["recent_frames"]):
            raise ValueError("RoPE virtual span must exceed recent frames")
        params = data["method_params"]
        if spec.id == "streaming" and not (0 <= int(params["sink_size"]) < int(params["local_attn_size"])):
            raise ValueError("streaming requires 0 <= sink_size < local_attn_size")
        if spec.id in {"random", "decoprune", "decoprune_hs"}:
            if not 0 <= int(params["step_index"]) <= 3 or not 0.0 <= float(params["threshold"]) <= 1.0:
                raise ValueError("consistency methods require step_index in [0,3] and threshold in [0,1]")
        heterogeneous = spec.id in {"forcingkv", "decoprune_hs"}
        if heterogeneous and not data["head_map_file"]:
            raise ValueError(f"{spec.id} requires an explicit head_map_file")
        if not heterogeneous and data["head_map_file"] is not None:
            raise ValueError("head_map_file is exclusive to heterogeneous methods")
        return data

def load(path: Path) -> RunConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - set(RunConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    config = RunConfig(**raw)
    config.normalized()
    return config
