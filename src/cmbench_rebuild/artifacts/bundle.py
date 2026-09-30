"""Normalize one completed case directory into the stable artifact contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import sha256_file
from .video import inspect_video
from ..evaluation.seqpr import audit_mask_accounting, audit_seqpr


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact_bundle(
    case_dir: Path,
    *,
    independent_dino: Path | None = None,
    official_dino: Path | None = None,
) -> dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    seqpr_path = case_dir / "seq_pr_metrics.json"
    new_trajectory = case_dir / "token_trajectory.jsonl"
    masks_path = new_trajectory if new_trajectory.is_file() else case_dir / "mask_accounting.jsonl"
    video_path = case_dir / "continuation.mp4"
    required = (metadata_path, seqpr_path, masks_path, video_path)
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"incomplete case artifacts: {missing_files}")
    metadata = _json(metadata_path)
    seqpr = audit_seqpr(seqpr_path)
    masks = audit_mask_accounting(masks_path)
    mask_rows = [json.loads(line) for line in masks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    video = inspect_video(video_path)
    selection_candidates = [case_dir / name for name in ("selection_trajectory.json", "ours_q0_decisions.json", "forcingkv_allocator_events.jsonl", "patchify_retrieve_allocator_events.jsonl", "dummyforcing_decisions.json")]
    selection = next((path for path in selection_candidates if path.is_file()), None)
    peaks = [item.get("generation_peak_allocated", item.get("generation_peak_reserved")) for item in metadata.get("gpu_memory_bytes_by_rank", [])]
    result = {
        "schema_version": 1,
        "status": "complete",
        "oracle_state": "candidate",
        "independent_evidence": ["seqpr_formula_recomputed", "mask_accounting_conservation", "decoded_frame_sha256"],
        "known_deviations": [],
        "identity": {"case_id": metadata.get("case_id"), "method": metadata.get("method"), "seed": metadata.get("seed"), "profile": metadata.get("matched_prune_profile") or metadata.get("profile")},
        "metrics": {"seqpr": seqpr["computed_seqpr"], "generation_seconds": metadata.get("generation_elapsed_s"), "generation_fps": metadata.get("generation_fps_forcingkv_public"), "peak_memory_bytes": max((int(value) for value in peaks if value is not None), default=None), "dino_overall": None, "dino_object": None, "dino_scene": None},
        "audits": {"seqpr": seqpr, "mask_accounting": masks},
        "artifacts": {"metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)}, "video": {"path": str(video_path), "file_sha256": sha256_file(video_path), "decoded": video}, "token_trajectory": {"kind": "per_layer" if masks_path == new_trajectory else "legacy_aggregate", "path": str(masks_path), "sha256": sha256_file(masks_path)}, "selection_trajectory": ({"kind": "explicit", "path": str(selection), "sha256": sha256_file(selection)} if selection else {"kind": "implicit_policy", "method": metadata.get("method")})},
    }
    if independent_dino is not None:
        dino = _json(independent_dino)
        if dino.get("official_compatible") is not False:
            raise ValueError("independent DINO evidence must explicitly declare official_compatible=false")
        if dino.get("case_id") != metadata.get("case_id"):
            raise ValueError("independent DINO case_id does not match metadata")
        if dino.get("generated_video", {}).get("sha256") != result["artifacts"]["video"]["file_sha256"]:
            raise ValueError("independent DINO video hash does not match bundle video")
        result["artifacts"]["independent_dino_ablation"] = {
            "path": str(independent_dino),
            "sha256": sha256_file(independent_dino),
            "metric": dino.get("metric"),
            "score": dino.get("dino_score"),
            "official_compatible": False,
        }
        result["independent_evidence"].append("direct_bbox_dinov2_reference_best_ablation")
        result["known_deviations"].append("independent DINO ablation is evidence only; official OWL/SAM/DINO score remains unqualified")
    if official_dino is not None:
        dino = _json(official_dino)
        if dino.get("official_compatible") is not True or dino.get("scoring_path") != "owl_sam_dino":
            raise ValueError("official DINO evidence must be verified OWL/SAM/DINO output")
        if dino.get("case_id") != metadata.get("case_id"):
            raise ValueError("official DINO case_id does not match metadata")
        if dino.get("generated_video", {}).get("sha256") != result["artifacts"]["video"]["file_sha256"]:
            raise ValueError("official DINO video hash does not match bundle video")
        score = float(dino["dino_score"])
        level = str(dino.get("memory_level") or "").lower()
        result["metrics"]["dino_overall"] = score
        if level in {"object", "scene"}:
            result["metrics"][f"dino_{level}"] = score
        result["artifacts"]["official_dino"] = {
            "path": str(official_dino),
            "sha256": sha256_file(official_dino),
            "metric": dino.get("metric"),
            "score": score,
        }
        result["independent_evidence"].append("verified_official_owl_sam_dino")
    result["missing_required_fields"] = [name for name, value in result["metrics"].items() if name.startswith("dino_") and value is None]
    if not any(row.get("layer_idx") is not None for row in mask_rows):
        result["missing_required_fields"].append("per_layer_token_trajectory")
    return result
