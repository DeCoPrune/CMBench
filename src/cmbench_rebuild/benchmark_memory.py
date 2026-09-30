"""Explicit diagnostic job list; reuse one model and report every rank's costs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path

from .config import load
from .runtime.generation import LingBotGenerationOptions
from .runtime.session import RuntimeSession
from .matrix import production_source_identity


def main():
    import torch
    import torch.distributed as dist
    from wan.distributed.util import init_distributed_group

    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocator-limit-gib", type=float)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if not jobs or len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("jobs must have unique names")
    for job in jobs:
        if job["name"] in {"", ".", ".."} or Path(job["name"]).name != job["name"]:
            raise ValueError("job name must be a filename")
    rank, world, local = [int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))]
    torch.cuda.set_device(local)
    if args.allocator_limit_gib:
        total = torch.cuda.get_device_properties(local).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.allocator_limit_gib * 2**30 / total), local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local))
    init_distributed_group()
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        first = load(Path(jobs[0]["config"])).normalized()
        source = production_source_identity(Path.cwd())
        began = time.perf_counter()
        session = RuntimeSession(first["checkpoint"], rank=rank, world_size=world, device_id=local)
        torch.cuda.synchronize()
        dist.barrier()
        load_seconds = time.perf_counter() - began
        if rank == 0:
            (args.output / "session.json").write_text(json.dumps({
                "status": "diagnostic_only", "torch": torch.__version__, "jobs": jobs,
                "model_load_seconds": load_seconds, "backend": session.backend.load_manifest,
                "allocator_limit_gib": args.allocator_limit_gib,
                "production_source": source,
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }, indent=2))
        for job in jobs:
            limit = job.get("allocator_limit_gib", args.allocator_limit_gib)
            total = torch.cuda.get_device_properties(local).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, limit * 2**30 / total) if limit is not None else 1.0, local)
            directory = args.output / job["name"]
            if rank == 0:
                directory.mkdir(exist_ok=False)
            dist.barrier()
            def emit(event):
                with (directory / f"events.rank{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps({"rank": rank, **event}) + "\n")
            config = load(Path(job["config"])).normalized()
            if config["world_size"] != world:
                raise ValueError("config world size must match the diagnostic worker")
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            for module in session.backend.pipeline.model.modules():
                if isinstance(module, FSDP):
                    module.forward_prefetch = bool(job.get("fsdp_prefetch", False))
            options = LingBotGenerationOptions(
                chunk_size=config["chunk_size"], timesteps_index=tuple(config["timesteps_index"]),
                sampling_shift=config["sampling_shift"], vae_temporal_stride=config["vae_temporal_stride"],
                generation_kv_policy=config["generation_kv_policy"],
                **({"kv_storage_mode": config["kv_storage_mode"],
                    "gpu_memory_limit_gib": config["gpu_memory_limit_gib"]} | job.get("options", {})))
            started_utc = datetime.now(timezone.utc).isoformat()
            began = time.perf_counter()
            generated = session.run(config, emit=emit, options=options,
                context_limit=job.get("context_limit"), output_frames=job.get("output_frames"),
                case_id=job.get("case_id"), seed=job.get("seed"))
            torch.cuda.synchronize()
            dist.barrier()
            resources = {**generated.resource_metrics, "case_wall_seconds": time.perf_counter() - began, "rank": rank}
            ranks = [None] * world
            dist.all_gather_object(ranks, resources)
            diagnostics = [None] * world
            dist.all_gather_object(diagnostics, {"q0": generated.diagnostics["q0"],
                                               "memory_plan": generated.diagnostics["kv_cache_memory_plan"],
                                               "transfer": generated.diagnostics["kv_transfer"]})
            if rank == 0:
                latent = generated.continuation_latents.detach().cpu()
                if not torch.isfinite(latent).all():
                    raise RuntimeError("non-finite diagnostic output")
                torch.save(latent, directory / "continuation_latents.pt")
                result = {"status": "diagnostic_completed", "job": job,
                    "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "resources_by_rank": ranks, "generation": dict(generated.diagnostics),
                    "cache_by_rank": diagnostics,
                    "tensor_sha256": hashlib.sha256(latent.numpy().tobytes()).hexdigest()}
                (directory / "result.json").write_text(json.dumps(result, indent=2))
                print(json.dumps({"job": job["name"], "resources": resources,
                                  "input_cache_hit": generated.diagnostics["input_cache_hit"]}), flush=True)
            del generated
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
