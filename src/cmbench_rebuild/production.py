"""Distributed production entry point for the rebuilt LingBot runtime."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import load
from .identity import (
    directory_content_identity,
    file_content_identity,
    validate_directory_content_identity,
)
from .core.rope import RopeReindexPlan
from .methods.head_map import load_head_map
from .methods.policy import policy_from_config
from .matrix import production_source_identity
from .runtime.backend import LingBotBackend, LingBotLoadOptions
from .runtime.context import TorchVideoContextProvider
from .runtime.contracts import PreparedContext
from .runtime.camera import validate_explicit_camera
from .runtime.generation import LingBotGenerationOptions
from .runtime.requests import load_requests
from .runtime.output import write_generation_input


def _select_request(path: Path, case_id: str, seed: int):
    matches = [request for request in load_requests(path) if request.case_id == case_id and request.seed == int(seed)]
    if len(matches) != 1:
        raise ValueError(f"expected one request for case={case_id}, seed={seed}; found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmbench-production")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-latent-limit", type=int, help="diagnostic smoke only; disqualifies benchmark output")
    parser.add_argument("--output-latent-frames", type=int, help="diagnostic smoke only; disqualifies benchmark output")
    parser.add_argument("--case-id", help="matrix runner override; case must exist in the frozen request file")
    parser.add_argument("--seed", type=int, help="matrix runner seed override; source request remains provenance")
    args = parser.parse_args()

    production_source = production_source_identity(Path.cwd())
    planned_source_sha256 = os.environ.get("CMBENCH_PRODUCTION_SOURCE_SHA256")
    if planned_source_sha256 and planned_source_sha256 != production_source["sha256"]:
        raise RuntimeError(
            "launcher production source identity differs from the running checkout: "
            f"planned={planned_source_sha256}, actual={production_source['sha256']}"
        )
    config_path = args.config.resolve()
    config = load(config_path).normalized()
    checkpoint_path = Path(str(config["checkpoint"])).expanduser().resolve()
    planned_checkpoint_json = os.environ.get("CMBENCH_CHECKPOINT_IDENTITY_JSON")
    planned_input_video_json = os.environ.get("CMBENCH_INPUT_VIDEO_IDENTITY_JSON")
    planned_head_map_json = os.environ.get("CMBENCH_HEAD_MAP_IDENTITY_JSON")
    checkpoint_identity = None
    if planned_checkpoint_json:
        checkpoint_identity = json.loads(planned_checkpoint_json)
        if not isinstance(checkpoint_identity, dict):
            raise ValueError("launcher checkpoint identity must be a JSON object")
        if Path(str(checkpoint_identity.get("path") or "")).resolve() != checkpoint_path:
            raise RuntimeError("launcher checkpoint identity points at a different checkpoint")
        validate_directory_content_identity(checkpoint_identity, verify_content=False)
    head_map_path = None
    head_map_identity = None
    if config.get("head_map_file"):
        raw_head_map_path = Path(str(config["head_map_file"])).expanduser()
        head_map_path = (
            raw_head_map_path.resolve()
            if raw_head_map_path.is_absolute()
            else (Path.cwd() / raw_head_map_path).resolve()
        )
        head_map_identity = file_content_identity(head_map_path)
    if planned_head_map_json:
        planned_head_map_identity = json.loads(planned_head_map_json)
        if not isinstance(planned_head_map_identity, dict):
            raise ValueError("launcher head-map identity must be a JSON object")
        if planned_head_map_identity != head_map_identity:
            raise RuntimeError("launcher head-map content identity changed")

    import torch
    import torch.distributed as dist
    from wan.distributed.util import init_distributed_group

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
            device_id=torch.device("cuda", local_rank),
        )
        init_distributed_group()
    try:
        if checkpoint_identity is None:
            checkpoint_identity = directory_content_identity(checkpoint_path) if rank == 0 else None
            if world_size > 1:
                values = [checkpoint_identity]
                dist.broadcast_object_list(values, src=0)
                checkpoint_identity = values[0]
        if not isinstance(checkpoint_identity, dict):
            raise RuntimeError("checkpoint identity broadcast failed")
        if int(config["world_size"]) != world_size:
            raise ValueError(f"config world_size={config['world_size']} does not match launcher world_size={world_size}")
        request_path = Path(config["request_file"])
        if not request_path.is_absolute():
            request_path = Path.cwd() / request_path
        source_seed = int(config["seed"])
        effective_case_id = str(args.case_id or config["case_id"])
        effective_seed = int(args.seed if args.seed is not None else source_seed)
        request = _select_request(request_path, effective_case_id, source_seed)
        if effective_seed != source_seed:
            request_raw = dict(request.raw)
            request_raw["seed"] = effective_seed
            request = replace(request, seed=effective_seed, raw=request_raw)
        input_video_path = request.clip_file.expanduser()
        if not input_video_path.is_absolute():
            input_video_path = (Path.cwd() / input_video_path).resolve()
        else:
            input_video_path = input_video_path.resolve()
        request = replace(request, clip_file=input_video_path)
        input_video_identity = file_content_identity(input_video_path)
        if planned_input_video_json:
            planned_input_video_identity = json.loads(planned_input_video_json)
            if not isinstance(planned_input_video_identity, dict):
                raise ValueError("launcher input video identity must be a JSON object")
            if planned_input_video_identity != input_video_identity:
                raise RuntimeError("launcher input video content identity changed")
        config["case_id"] = effective_case_id
        config["seed"] = effective_seed
        config["source_request_seed"] = source_seed
        diagnostic = args.context_latent_limit is not None or args.output_latent_frames is not None
        if args.output_latent_frames is not None:
            generation = dict(request.generation)
            generation["num_output_latent_frames"] = int(args.output_latent_frames)
            generation["num_output_pixel_frames"] = int(args.output_latent_frames) * int(config["vae_temporal_stride"])
            camera = dict(request.camera)
            positions = camera.get("camera_keyframe_latent_indices")
            if positions is not None:
                old_last = int(positions[-1])
                new_last = int(args.output_latent_frames) - 1
                scaled = [round(int(value) * new_last / old_last) for value in positions] if old_last else [0 for _ in positions]
                if any(right <= left for left, right in zip(scaled, scaled[1:])):
                    raise ValueError("diagnostic output shortening collapses camera keyframes")
                camera["camera_keyframe_latent_indices"] = scaled
            request = replace(request, generation=generation, camera=camera)
        if request.camera.get("camera_condition") == "explicit_pose_continuation":
            validate_explicit_camera(
                request.camera,
                output_frames=int(request.generation.get("num_output_latent_frames", 16)),
            )
        if args.context_latent_limit is not None:
            diagnostic_raw = dict(request.raw)
            schedule = diagnostic_raw.get("context_prompt_schedule")
            if isinstance(schedule, list):
                context_chunks = int(args.context_latent_limit) // int(config["chunk_size"])
                diagnostic_raw["context_prompt_schedule"] = [
                    item for item in schedule if int(item.get("start_chunk", -1)) < context_chunks
                ]
            request = replace(request, raw=diagnostic_raw)
        output = args.output.resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        if world_size > 1:
            dist.barrier()

        events_path = output / f"events.rank{rank}.jsonl"

        def emit(event: dict[str, Any]) -> None:
            row = {"rank": rank, **dict(event)}
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

        method = str(config["method"])
        parameters = dict(config["method_params"])
        local_attn_size = int(parameters["local_attn_size"]) if method == "streaming" else -1
        sink_size = int(parameters["sink_size"]) if method == "streaming" else 0
        head_map = None
        if head_map_path is not None:
            head_map = load_head_map(
                head_map_path,
                expected_model="lingbot-world-v2-14b-causal-fast",
                num_layers=40,
                num_heads=40,
            )
        backend = LingBotBackend(
            LingBotLoadOptions(
                device_id=local_rank,
                rank=rank,
                world_size=world_size,
                use_sequence_parallel=world_size > 1,
                dit_fsdp=world_size > 1,
                t5_cpu=True,
                init_on_cpu=False,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            ),
            generation_options=LingBotGenerationOptions(
                chunk_size=int(config["chunk_size"]),
                timesteps_index=tuple(int(value) for value in config["timesteps_index"]),
                sampling_shift=float(config["sampling_shift"]),
                vae_temporal_stride=int(config["vae_temporal_stride"]),
                generation_kv_policy=str(config["generation_kv_policy"]),
                kv_storage_mode=str(config["kv_storage_mode"]),
                gpu_memory_limit_gib=float(config["gpu_memory_limit_gib"]),
            ),
            head_map=head_map,
        )
        emit({"event": "model_load_started"})
        backend.load(Path(config["checkpoint"]))
        emit({"event": "model_loaded", "manifest": backend.load_manifest})
        provider = TorchVideoContextProvider(
            height=int(config["height"]),
            width=int(config["width"]),
            target_fps=float(config["model_fps"]),
            chunk_size=int(config["chunk_size"]),
            vae_temporal_stride=int(config["vae_temporal_stride"]),
            device=f"cuda:{local_rank}",
        )
        prepared = provider.prepare(request)
        if args.context_latent_limit is not None:
            latent_limit = int(args.context_latent_limit)
            if latent_limit <= 0 or latent_limit % int(config["chunk_size"]):
                raise ValueError("context-latent-limit must be a positive chunk multiple")
            pixel_limit = (latent_limit - 1) * int(config["vae_temporal_stride"]) + 1
            if pixel_limit > prepared.pixel_frames:
                raise ValueError("context-latent-limit exceeds available context")
            diagnostics = {**prepared.diagnostics, "diagnostic_context_latent_limit": latent_limit, "disqualified": True}
            pixels = prepared.payload[:, :pixel_limit].contiguous() if prepared.payload is not None else None
            prepared = PreparedContext(pixels, pixel_limit, latent_limit, diagnostics)
        torch.cuda.reset_peak_memory_stats(torch.device(f"cuda:{local_rank}"))
        generated = backend.generate(
            request,
            prepared,
            policy_from_config(config),
            RopeReindexPlan(**config["rope_reindex"]),
            emit,
        )
        resources_by_rank = [None for _ in range(world_size)]
        local_resources = {"rank": rank, **dict(generated.resource_metrics)}
        if world_size > 1:
            dist.all_gather_object(resources_by_rank, local_resources)
        else:
            resources_by_rank[0] = local_resources
        if rank == 0:
            write_generation_input(output, generated=generated, request=request, config=config,
                resources_by_rank=resources_by_rank, backend_manifest=backend.load_manifest,
                production_source=production_source, checkpoint_identity=checkpoint_identity,
                input_video_identity=input_video_identity, head_map_identity=head_map_identity,
                diagnostic=diagnostic)
        if world_size > 1:
            dist.barrier()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if rank != 0:
            return
        # Replace the rank-0 process rather than mutating live FSDP objects.
        # exec releases the entire DiT/CUDA interpreter before a VAE-only
        # process loads, while torchrun continues tracking the same PID.
        os.execv(
            sys.executable,
            [sys.executable, "-m", "cmbench_rebuild.postprocess", str(output / "postprocess-input.json")],
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
