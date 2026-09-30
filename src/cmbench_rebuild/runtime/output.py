"""Shared latent and provenance handoff to the VAE-only decoding process."""
from pathlib import Path
import json


def write_generation_input(output, *, generated, request, config, resources_by_rank,
                           backend_manifest, production_source, checkpoint_identity,
                           input_video_identity, head_map_identity, diagnostic):
    import torch

    output = Path(output).resolve()
    context = generated.context_latents.detach().cpu()
    continuation = generated.continuation_latents.detach().cpu()
    torch.save(continuation, output / "continuation_latents.pt")
    torch.save(torch.cat([context, continuation], dim=1), output / "context_plus_continuation_latents.pt")
    payload = {
        "schema_version": 1, "output": str(output), "diagnostic": diagnostic, "config": config,
        "request": {"case_id": request.case_id, "seed": request.seed,
                    "raw": dict(request.raw), "generation": dict(request.generation)},
        "generation": dict(generated.diagnostics), "resources": dict(generated.resource_metrics),
        "resources_by_rank": resources_by_rank, "backend": backend_manifest,
        "production_source": production_source, "checkpoint_identity": checkpoint_identity,
        "input_video_identity": input_video_identity, "head_map_identity": head_map_identity,
    }
    path = output / "postprocess-input.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
