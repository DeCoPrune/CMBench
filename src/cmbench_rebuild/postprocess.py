"""VAE-only rank-0 post-processing for a completed distributed generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .artifacts.manifest import sha256_file, validate_directory_content_identity
from .artifacts.video import write_video_tensor
from .identity import validate_file_content_identity
from .runtime.vae_decode import decode_continuation


def continuation_pixel_slice(*, context_latent_frames: int, stride: int, output_pixel_frames: int) -> slice:
    """Return the continuation span in a fully decoded causal latent timeline."""
    if int(context_latent_frames) <= 0 or int(stride) <= 0 or int(output_pixel_frames) <= 0:
        raise ValueError("causal VAE timeline dimensions must be positive")
    context_pixel_frames = (int(context_latent_frames) - 1) * int(stride) + 1
    return slice(context_pixel_frames, context_pixel_frames + int(output_pixel_frames))


def finish_case(input_path: Path) -> dict[str, Any]:
    source = Path(input_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    output = Path(payload["output"]).resolve()
    config = dict(payload["config"])
    request = dict(payload["request"])
    diagnostics = dict(payload["generation"])
    resources = dict(payload["resources"])
    production_source = dict(payload["production_source"])
    checkpoint_identity = dict(payload["checkpoint_identity"])
    input_video_identity = dict(payload["input_video_identity"])
    raw_head_map_identity = payload.get("head_map_identity")
    head_map_identity = dict(raw_head_map_identity) if raw_head_map_identity is not None else None
    validate_directory_content_identity(checkpoint_identity, verify_content=False)
    validate_file_content_identity(input_video_identity)
    if head_map_identity is not None:
        validate_file_content_identity(head_map_identity)

    import torch
    from wan.modules.vae2_1 import Wan2_1_VAE

    method = str(config["method"])
    torch.cuda.set_device(0)
    if config["kv_storage_mode"] == "auto_80gb":
        total = torch.cuda.get_device_properties(0).total_memory
        limit = float(config.get("gpu_memory_limit_gib", 80.0))
        torch.cuda.set_per_process_memory_fraction(min(limit * 2**30, total) * 0.95 / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    decode_started = time.perf_counter()
    latent = torch.load(
        output / "context_plus_continuation_latents.pt",
        map_location="cuda:0",
        weights_only=True,
    )
    vae = Wan2_1_VAE(
        vae_pth=str(Path(config["checkpoint"]) / "Wan2.1_VAE.pth"),
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )
    expected_pixel_frames = int(request["generation"].get(
        "num_output_pixel_frames",
        int(request["generation"].get("num_output_latent_frames", 16)) * int(config["vae_temporal_stride"]),
    ))
    pixel_slice = continuation_pixel_slice(
        context_latent_frames=int(diagnostics["context_latent_frames"]),
        stride=int(config["vae_temporal_stride"]),
        output_pixel_frames=expected_pixel_frames,
    )
    decoded, decoded_frames = decode_continuation(vae, latent, pixel_slice)
    torch.cuda.synchronize(0)
    resources["causal_vae_load_and_decode_seconds"] = time.perf_counter() - decode_started
    resources["causal_vae_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    resources["causal_vae_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
    video_audit = write_video_tensor(decoded, output / "continuation.mp4", fps=float(config["model_fps"]))
    trajectory = list(diagnostics.pop("token_trajectory"))
    seqpr = dict(diagnostics["seq_pr_metrics"])
    (output / "token_trajectory.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trajectory), encoding="utf-8"
    )
    (output / "seq_pr_metrics.json").write_text(json.dumps(seqpr, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = {
        "schema_version": 1,
        "method": method,
        "case_id": request["case_id"],
        "seed": request["seed"],
        "q0": diagnostics["q0"],
        "online_policy": diagnostics["online_policy"],
    }
    (output / "selection_trajectory.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request_digest = hashlib.sha256(
        json.dumps(request["raw"], sort_keys=True, allow_nan=True).encode("utf-8")
    ).hexdigest()
    ar_seconds = float(resources["autoregressive_generation_seconds"])
    metadata = {
        "schema_version": 1,
        "case_id": request["case_id"],
        "method": method,
        "legacy_method": config["legacy_method"],
        "seed": request["seed"],
        "profile": config["protocol"],
        "dataset_version": config["dataset_version"],
        "request_sha256": request_digest,
        "generation_kv_policy": request["generation"].get("generation_kv_policy", config["generation_kv_policy"]),
        "context_latent_frames": int(diagnostics["context_latent_frames"]),
        "output_latent_frames": int(diagnostics["output_latent_frames"]),
        "output_pixel_frames": int(video_audit["decoded_frame_count"]),
        "causal_vae_decode": {
            "full_timeline_decoded_frames": decoded_frames,
            "context_pixel_boundary": int(pixel_slice.start),
            "continuation_slice_start": int(pixel_slice.start),
            "continuation_slice_stop": int(pixel_slice.stop),
        },
        "generation_elapsed_s": ar_seconds,
        "generation_total_elapsed_s": float(resources["generation_total_seconds"]),
        "context_preprocessing_elapsed_s": float(resources["context_preprocessing_seconds"]),
        "generation_fps": int(video_audit["decoded_frame_count"]) / ar_seconds,
        "generation_fps_forcingkv_public": int(video_audit["decoded_frame_count"]) / ar_seconds,
        "gpu_memory_bytes_by_rank": [
            {"rank": int(row["rank"]), "generation_peak_allocated": int(row["peak_allocated_bytes"])}
            for row in payload["resources_by_rank"]
        ],
        "video": video_audit,
        "diagnostic_disqualified": bool(payload["diagnostic"]),
        "production_source_sha256": production_source["sha256"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "input_video_sha256": input_video_identity["sha256"],
        "head_map_sha256": (head_map_identity or {}).get("sha256"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "status": "diagnostic_completed" if payload["diagnostic"] else "completed",
        "disqualified": bool(payload["diagnostic"]),
        "config": config,
        "request": {"case_id": request["case_id"], "seed": request["seed"]},
        "generation": diagnostics,
        "resources": resources,
        "backend": payload["backend"],
        "production_source": production_source,
        "checkpoint_identity": checkpoint_identity,
        "input_video_identity": input_video_identity,
        "head_map_identity": head_map_identity,
        "artifacts": {
            "video": "continuation.mp4", "latents": "continuation_latents.pt",
            "causal_decode_latents": "context_plus_continuation_latents.pt",
            "metadata": "metadata.json", "seqpr": "seq_pr_metrics.json",
            "token_trajectory": "token_trajectory.jsonl", "selection": "selection_trajectory.json",
            "postprocess_input": "postprocess-input.json",
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_names = [
        "continuation.mp4", "continuation_latents.pt", "context_plus_continuation_latents.pt", "metadata.json",
        "seq_pr_metrics.json", "token_trajectory.jsonl", "selection_trajectory.json",
        "postprocess-input.json", "result.json",
    ]
    manifest = {
        "schema_version": 1,
        "case_id": request["case_id"],
        "method": method,
        "seed": request["seed"],
        "production_source_sha256": production_source["sha256"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "input_video_sha256": input_video_identity["sha256"],
        "head_map_sha256": (head_map_identity or {}).get("sha256"),
        "files": {
            name: {"bytes": int((output / name).stat().st_size), "sha256": sha256_file(output / name)}
            for name in artifact_names
        },
    }
    (output / "artifacts.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmbench-postprocess")
    parser.add_argument("input", type=Path, nargs="+")
    args = parser.parse_args()
    for input_path in args.input:
        result = finish_case(input_path)
        print(json.dumps({"status": result["status"], "output": str(input_path.parent), "resources": result["resources"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
