"""Lossless request parsing with explicit prompt/camera/generation views."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import CaseRequest


CAMERA_FIELDS = frozenset({"camera_condition", "camera_pose_source", "camera_target_reviewed", "camera_yaw_degrees_per_latent", "camera_yaw_keyframes_degrees", "camera_pitch_keyframes_degrees", "camera_translation_keyframes_m", "camera_keyframe_latent_indices", "camera_rotation_vector", "camera_translation_vector"})
GENERATION_FIELDS = frozenset({"num_output_latent_frames", "num_output_pixel_frames", "generation_kv_policy", "rope_reindex_mode", "rope_reindex_virtual_span", "rope_reindex_recent_frames", "rope_reindex_fast_band_pairs", "matched_context_policy"})


def request_from_mapping(value: Mapping[str, Any]) -> CaseRequest:
    required = ("case_id", "seed", "prompt", "clip_file")
    missing = [name for name in required if value.get(name) is None]
    if missing:
        raise ValueError(f"request is missing required fields: {missing}")
    prompt = str(value["prompt"])
    camera = {name: value[name] for name in CAMERA_FIELDS if name in value}
    generation = {name: value[name] for name in GENERATION_FIELDS if name in value}
    raw = dict(value)
    raw["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return CaseRequest(case_id=str(value["case_id"]), seed=int(value["seed"]), prompt=prompt, clip_file=Path(str(value["clip_file"])), camera=camera, generation=generation, raw=raw)


def load_requests(path: Path) -> tuple[CaseRequest, ...]:
    requests = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            requests.append(request_from_mapping(json.loads(line)))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not requests:
        raise ValueError(f"request file is empty: {path}")
    identities = [(request.case_id, request.seed) for request in requests]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate case/seed request identity in {path}")
    return tuple(requests)
