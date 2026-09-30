"""Run an explicit serial job list with one DiT load and deferred VAE decode."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import load
from .identity import directory_content_identity, file_content_identity, validate_directory_content_identity
from .matrix import production_source_identity
from .runtime.output import write_generation_input
from .runtime.session import RuntimeSession


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names) or any(not name or name in {".", ".."} or Path(name).name != name for name in names):
        raise ValueError("job names must be unique simple directory names")
    unknown = set().union(*(set(job) for job in jobs)) - {"name", "config", "case_id", "seed", "context_limit", "output_frames"}
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    return jobs


def main():
    import torch
    import torch.distributed as dist
    from wan.distributed.util import init_distributed_group

    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    jobs = validate_jobs(json.loads(args.jobs.read_text()))
    configs = [load(Path(job["config"])).normalized() for job in jobs]
    checkpoints = {str(Path(config["checkpoint"]).resolve()) for config in configs}
    if len(checkpoints) != 1:
        raise ValueError("one worker must use one checkpoint")
    source = production_source_identity(Path.cwd())
    rank, world, local = [int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))]
    if any(config["world_size"] != world for config in configs):
        raise ValueError("config world size must match the worker")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local))
    init_distributed_group()
    pending = []
    try:
        if rank == 0:
            args.output_root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        checkpoint = Path(next(iter(checkpoints)))
        identities = [directory_content_identity(checkpoint) if rank == 0 else None]
        dist.broadcast_object_list(identities, src=0)
        checkpoint_identity = identities[0]
        session = RuntimeSession(checkpoint, rank=rank, world_size=world, device_id=local)
        for job, config in zip(jobs, configs):
            if production_source_identity(Path.cwd())["sha256"] != source["sha256"]:
                raise RuntimeError("production source changed during the worker")
            validate_directory_content_identity(checkpoint_identity, verify_content=False)
            total = torch.cuda.get_device_properties(local).total_memory
            limit = float(config["gpu_memory_limit_gib"]) * 2**30
            fraction = min(limit, total) * 0.95 / total if config["kv_storage_mode"] == "auto_80gb" else 1.0
            torch.cuda.set_per_process_memory_fraction(fraction, local)
            output = (args.output_root / job["name"]).resolve()
            if rank == 0:
                output.mkdir(exist_ok=False)
            dist.barrier()
            def emit(event):
                with (output / f"events.rank{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps({"rank": rank, "utc": datetime.now(timezone.utc).isoformat(), **event}, sort_keys=True) + "\n")
            emit({"event": "worker_job_started", "job": job["name"]})
            generated = session.run(config, emit=emit, context_limit=job.get("context_limit"),
                output_frames=job.get("output_frames"), case_id=job.get("case_id"), seed=job.get("seed"))
            request = session.last_request
            input_identity = file_content_identity(request.clip_file)
            if input_identity["sha256"] != session.last_input_sha256:
                raise RuntimeError("input video changed during generation")
            head_identity = file_content_identity(Path(config["head_map_file"])) if config.get("head_map_file") else None
            if head_identity is not None and head_identity["sha256"] != session.last_head_map_sha256:
                raise RuntimeError("head map changed during generation")
            by_rank = [None] * world
            dist.all_gather_object(by_rank, {"rank": rank, **generated.resource_metrics})
            cache_by_rank = [None] * world
            dist.all_gather_object(cache_by_rank, {"rank": rank, "q0": generated.diagnostics["q0"],
                "memory_plan": generated.diagnostics["kv_cache_memory_plan"]})
            generated.diagnostics["cache_by_rank"] = cache_by_rank
            if rank == 0:
                resolved = {**config, "source_request_seed": config["seed"], "case_id": request.case_id, "seed": request.seed}
                path = write_generation_input(output, generated=generated, request=request, config=resolved,
                    resources_by_rank=by_rank, backend_manifest=session.backend.load_manifest,
                    production_source=source, checkpoint_identity=checkpoint_identity,
                    input_video_identity=input_identity, head_map_identity=head_identity,
                    diagnostic=job.get("context_limit") is not None or job.get("output_frames") is not None)
                pending.append(str(path))
            emit({"event": "worker_job_completed", "job": job["name"]})
            del generated
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        # Process replacement releases all DiT weights before causal VAE decode.
        os.execv(sys.executable, [sys.executable, "-m", "cmbench_rebuild.postprocess", *pending])


if __name__ == "__main__":
    main()
