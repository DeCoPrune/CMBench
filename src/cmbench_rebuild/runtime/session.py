"""One loaded model, bounded preprocessing reuse, independent KV for each job."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from ..core.rope import RopeReindexPlan
from ..methods.head_map import load_head_map
from ..methods.policy import policy_from_config
from .backend import LingBotBackend, LingBotLoadOptions
from .context import TorchVideoContextProvider
from .contracts import PreparedContext
from .generation import LingBotGenerationEngine, LingBotGenerationOptions
from .requests import load_requests


class RuntimeSession:
    """Serial jobs only. Reusing a session never reuses a case's KV cache."""

    def __init__(self, checkpoint, *, rank, world_size, device_id):
        self.backend = LingBotBackend(LingBotLoadOptions(
            rank=rank, world_size=world_size, device_id=device_id,
            use_sequence_parallel=world_size > 1, dit_fsdp=world_size > 1,
        ))
        self.backend.load(Path(checkpoint))
        self.contexts = OrderedDict()
        self.head_maps = {}

    def run(self, config, *, emit, context_limit=None, output_frames=None,
            options=None, case_id=None, seed=None):
        import torch
        from ..identity import file_content_identity

        if Path(config["checkpoint"]).resolve() != self.backend._layout.root:
            raise ValueError("a runtime session is bound to one checkpoint")
        identifier = case_id or config["case_id"]
        requests = [r for r in load_requests(Path(config["request_file"]))
                    if r.case_id == identifier and r.seed == config["seed"]]
        if len(requests) != 1:
            raise ValueError("request must identify exactly one case")
        request = requests[0]
        request = replace(request, clip_file=request.clip_file.resolve())
        if seed is not None:
            request = replace(request, seed=seed, raw={**request.raw, "seed": seed})
        if output_frames is not None:
            camera = dict(request.camera)
            positions = camera.get("camera_keyframe_latent_indices")
            if positions:
                last = positions[-1]
                scaled = [round(i * (output_frames - 1) / last) for i in positions]
                if any(b <= a for a, b in zip(scaled, scaled[1:])):
                    raise ValueError("diagnostic camera shortening collapses keyframes")
                camera["camera_keyframe_latent_indices"] = scaled
            request = replace(request, camera=camera, generation={**request.generation,
                "num_output_latent_frames": output_frames,
                "num_output_pixel_frames": output_frames * config["vae_temporal_stride"]})
        key = (file_content_identity(request.clip_file)["sha256"], config["height"],
               config["width"], config["model_fps"], config["chunk_size"], config["vae_temporal_stride"])
        self.last_input_sha256 = key[0]
        self.last_head_map_sha256 = None
        context_hit = key in self.contexts
        if not context_hit:
            provider = TorchVideoContextProvider(
                height=config["height"], width=config["width"], target_fps=config["model_fps"],
                chunk_size=config["chunk_size"], vae_temporal_stride=config["vae_temporal_stride"],
                device=str(self.backend.pipeline.device))
            self.contexts[key] = provider.prepare(request)
            while len(self.contexts) > 2:
                self.contexts.popitem(last=False)
        self.contexts.move_to_end(key)
        context = self.contexts[key]
        if context_limit is not None:
            if context_limit <= 0 or context_limit % config["chunk_size"]:
                raise ValueError("context_limit must be a positive chunk multiple")
            if context_limit > context.latent_frames:
                raise ValueError("context_limit exceeds the input timeline")
            pixels = (context_limit - 1) * config["vae_temporal_stride"] + 1
            context = PreparedContext(
                context.payload[:, :pixels].contiguous() if context.payload is not None else None,
                pixels, context_limit, {**context.diagnostics, "diagnostic_context_latent_limit": context_limit})
            schedule = request.raw.get("context_prompt_schedule")
            if schedule:
                request = replace(request, raw={**request.raw, "context_prompt_schedule": [
                    entry for entry in schedule if entry["start_chunk"] < context_limit // config["chunk_size"]]})
        params = config["method_params"]
        local = int(params["local_attn_size"]) if config["method"] == "streaming" else -1
        sink = int(params["sink_size"]) if config["method"] == "streaming" else 0
        for block in self.backend.pipeline.model.blocks:
            block.self_attn.local_attn_size = local
            block.self_attn.sink_size = sink
        head_map = None
        if config.get("head_map_file"):
            path = Path(config["head_map_file"])
            identity = file_content_identity(path)["sha256"]
            self.last_head_map_sha256 = identity
            if identity not in self.head_maps:
                self.head_maps[identity] = load_head_map(path,
                    expected_model="lingbot-world-v2-14b-causal-fast", num_layers=40, num_heads=40)
            head_map = self.head_maps[identity]
        generation_options = options or LingBotGenerationOptions(
            chunk_size=config["chunk_size"], timesteps_index=tuple(config["timesteps_index"]),
            sampling_shift=config["sampling_shift"], vae_temporal_stride=config["vae_temporal_stride"],
            generation_kv_policy=config["generation_kv_policy"], kv_storage_mode=config["kv_storage_mode"],
            gpu_memory_limit_gib=config.get("gpu_memory_limit_gib", 80.0))
        engine = LingBotGenerationEngine(self.backend.pipeline, load_options=self.backend.options,
            options=generation_options, head_map=head_map)
        self.last_request = request
        torch.cuda.reset_peak_memory_stats(self.backend.pipeline.device)
        emit({"event": "session_case_started", "case_id": identifier, "context_cache_hit": context_hit})
        return engine.generate(request, context, policy_from_config(config),
                               RopeReindexPlan(**config["rope_reindex"]), emit)
