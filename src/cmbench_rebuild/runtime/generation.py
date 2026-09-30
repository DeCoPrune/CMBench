"""Method-neutral LingBot context replay and autoregressive continuation."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..core.accounting import LayerTokenStep, SequenceTokenLedger
from ..core.metrics import active_history_token_head_units_before_current, active_history_token_head_units_for_source_range
from ..core.rope import RopeReindexPlan
from ..methods.head_map import HeadMap
from ..methods.streaming import finish_chunk as finish_streaming_chunk
from .cache_factory import CacheGeometry, allocate_case_cache, cache_memory_plan
from .cache_handles import CacheHandles
from .cache_growth import CacheGrowth, start_small
from .cache_placement import place_cache, promote_if_fits
from .input_cache import prepare_inputs
from .kv_offload import KVStager
from .camera import explicit_camera_plan, tail_motion_camera_plan
from .camera_estimation import estimate_tail_rotation
from .contracts import CaseRequest, GeneratedCase, PreparedContext
from .online_policies import OnlinePolicyRuntime
from .q0 import finalize_q0


@dataclass(frozen=True)
class LingBotGenerationOptions:
    chunk_size: int = 4
    timesteps_index: tuple[int, ...] = (0, 179, 358, 679)
    sampling_shift: float = 5.0
    vae_temporal_stride: int = 4
    generation_kv_policy: str = "method-native"
    kv_storage_mode: str = "auto_80gb"
    gpu_memory_limit_gib: float = 80.0
    kv_prefetch: bool = True
    kv_delta_writeback: bool = True
    reuse_inputs: bool = True
    gpu_cache_budget_gib: float | None = None
    cpu_cursors: bool = True
    gpu_safety_gib: float = 8.0
    initial_cache_frames: int = 128

    def validate(self) -> "LingBotGenerationOptions":
        if int(self.chunk_size) <= 0 or int(self.vae_temporal_stride) <= 0:
            raise ValueError("chunk size and VAE temporal stride must be positive")
        if not self.timesteps_index or any(int(value) < 0 for value in self.timesteps_index):
            raise ValueError("timesteps_index must be non-empty and non-negative")
        if self.generation_kv_policy not in {"append-only", "method-native"}:
            raise ValueError("unknown generation KV policy")
        if self.kv_storage_mode not in {"auto_80gb", "cuda", "cpu_offload"}:
            raise ValueError("unknown KV storage mode")
        if not math.isfinite(self.gpu_memory_limit_gib) or self.gpu_memory_limit_gib <= 0:
            raise ValueError("gpu_memory_limit_gib must be finite and positive")
        if self.gpu_safety_gib < 4 or (self.gpu_cache_budget_gib is not None and self.gpu_cache_budget_gib <= 0):
            raise ValueError("GPU safety margin must be at least 4 GiB and cache budget positive")
        if self.initial_cache_frames <= 0 or self.initial_cache_frames % self.chunk_size:
            raise ValueError("initial cache frames must be a positive chunk multiple")
        return self

    def tensor_budget_bytes(self, physical_bytes: int) -> int:
        """Keep 5% outside Torch for allocator, CUDA and communication overhead."""
        return int(min(self.gpu_memory_limit_gib * 1024**3, physical_bytes) * 0.95)


def parse_prompt_schedule(raw: Any, *, default_prompt: str, total_chunks: int, name: str) -> tuple[tuple[int, str], ...]:
    if raw is None:
        return ((0, str(default_prompt)),)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    parsed = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"every {name} item must be an object")
        start, prompt = int(item.get("start_chunk", -1)), str(item.get("prompt") or "").strip()
        if not 0 <= start < int(total_chunks) or not prompt:
            raise ValueError(f"invalid {name} entry")
        parsed.append((start, prompt))
    parsed.sort()
    if parsed[0][0] != 0 or len({start for start, _ in parsed}) != len(parsed):
        raise ValueError(f"{name} must start at zero and contain unique boundaries")
    return tuple(parsed)


def active_prompt(schedule: tuple[tuple[int, str], ...], chunk_index: int) -> tuple[int, str]:
    return max((item for item in schedule if item[0] <= int(chunk_index)), key=lambda item: item[0])


def _broadcast_object(value: Any, *, source_rank_value: bool) -> Any:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    values = [value if source_rank_value else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _encode_prompt(pipeline: Any, prompt: str) -> list[Any]:
    import torch

    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cached = pipeline._t5_cache.get(key)
    if cached is None:
        if pipeline.t5_cpu:
            cached = pipeline.text_encoder([prompt], torch.device("cpu"))
        else:
            pipeline.text_encoder.model.to(pipeline.device)
            cached = [value.to("cpu") for value in pipeline.text_encoder([prompt], pipeline.device)]
            pipeline.text_encoder.model.to("cpu")
        pipeline._t5_cache[key] = cached
    return [value.to(pipeline.device, non_blocking=True) for value in cached]


def _cross_cache(pipeline: Any, *, text_len: int, heads: int, head_dim: int) -> list[dict[str, Any]]:
    return pipeline._initialize_crossattn_cache(
        num_layers=int(pipeline.model.config.num_layers),
        shape=[1, int(text_len), int(heads), int(head_dim)],
        dtype=pipeline.pipe_dtype,
        device=pipeline.device,
    )


class LingBotGenerationEngine:
    def __init__(
        self,
        pipeline: Any,
        *,
        load_options: Any,
        options: LingBotGenerationOptions,
        head_map: HeadMap | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.load_options = load_options
        self.options = options.validate()
        self.head_map = head_map

    def _camera_plan(self, request: CaseRequest, context: PreparedContext, *, context_frames: int, output_frames: int) -> tuple[Any, dict[str, Any]]:
        import torch.distributed as dist

        camera = dict(request.camera)
        condition = str(camera.get("camera_condition") or "static_identity")
        height = int(context.diagnostics.get("height", context.payload.shape[2] if context.payload is not None else 480))
        width = int(context.diagnostics.get("width", context.payload.shape[3] if context.payload is not None else 832))
        if condition == "explicit_pose_continuation":
            return explicit_camera_plan(camera, context_frames=context_frames, output_frames=output_frames, height=height, width=width), {"status": "request_explicit_pose"}
        if condition == "tail_motion":
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            diagnostics = estimate_tail_rotation(
                context.payload,
                tail_frames=max(5, int(round((4.0 / 3.0) * 16.0))),
                stride=int(self.options.vae_temporal_stride),
            ) if rank == 0 else None
            diagnostics = _broadcast_object(diagnostics, source_rank_value=rank == 0)
            active = max(self.options.chunk_size, (output_frames // 2 // self.options.chunk_size) * self.options.chunk_size)
            return tail_motion_camera_plan(
                diagnostics["c2w_rotvec_radians_per_latent"],
                context_frames=context_frames,
                output_frames=output_frames,
                active_output_frames=active,
                height=height,
                width=width,
            ), diagnostics
        if condition == "static_identity":
            return tail_motion_camera_plan(
                (0.0, 0.0, 0.0),
                context_frames=context_frames,
                output_frames=output_frames,
                active_output_frames=0,
                height=height,
                width=width,
            ), {"status": "static_identity"}
        raise ValueError(f"unsupported camera condition: {condition}")

    def generate(
        self,
        request: CaseRequest,
        context: PreparedContext,
        policy: Any,
        rope_plan: RopeReindexPlan,
        emit: Callable[[Mapping[str, Any]], None],
    ) -> GeneratedCase:
        import torch
        import torch.distributed as dist

        pipeline = self.pipeline
        memory_stages: list[dict[str, int | float | str]] = []

        def record_memory(stage: str) -> None:
            snapshot = {
                "stage": stage,
                "elapsed_seconds": time.perf_counter() - started,
                "allocated_bytes": int(torch.cuda.memory_allocated(pipeline.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(pipeline.device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(pipeline.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(pipeline.device)),
            }
            memory_stages.append(snapshot)
            emit({"event": "gpu_memory_stage", **snapshot})

        chunk = int(self.options.chunk_size)
        output_frames = int(request.generation.get("num_output_latent_frames", 16))
        if output_frames <= 0 or output_frames % chunk:
            raise ValueError("output latent frames must be a positive multiple of chunk_size")
        started = time.perf_counter()
        record_memory("generate_started")
        context_latent, condition, input_cache_hit = prepare_inputs(
            pipeline, context, output_frames=output_frames,
            stride=int(self.options.vae_temporal_stride), reuse=self.options.reuse_inputs,
        )
        record_memory("inputs_prepared")
        context_frames = int(context_latent.shape[1])
        if context_frames != int(context.latent_frames) or context_frames % chunk:
            raise RuntimeError("context VAE layout disagrees with prepared causal timeline")
        if str(policy.name) == "dummy_forcing" and context_frames < 3 * chunk:
            raise ValueError(
                "DummyForcing requires at least three complete context chunks "
                f"for Q-K head classification; got {context_frames} latent frames"
            )
        total_frames = context_frames + output_frames
        torch.cuda.empty_cache()
        record_memory("vae_released")
        camera, camera_diagnostics = self._camera_plan(
            request,
            context,
            context_frames=context_frames,
            output_frames=output_frames,
        )
        model_config = pipeline.model.config
        frame_tokens = int(context_latent.shape[2] * context_latent.shape[3] // 4)
        geometry = CacheGeometry(
            num_layers=int(model_config.num_layers),
            num_heads=int(model_config.num_heads),
            head_dim=int(model_config.dim // model_config.num_heads),
            world_size=int(self.load_options.world_size),
            rank=int(self.load_options.rank),
            context_frames=context_frames,
            output_frames=output_frames,
            chunk_size=chunk,
            frame_tokens=frame_tokens,
        )
        use_cpu_cache = self.options.kv_storage_mode != "cuda"
        cache_device = torch.device("cpu") if use_cpu_cache else pipeline.device
        cache = allocate_case_cache(
            geometry,
            method=str(policy.name),
            parameters=dict(policy.parameters),
            dtype=pipeline.pipe_dtype,
            device=cache_device,
            compute_device=pipeline.device,
            head_map=self.head_map,
        )
        growable = self.options.kv_storage_mode == "auto_80gb" and policy.name in {
            "random", "decoprune", "decoprune_hs"}
        growing_prefix = start_small(cache, method=policy.name,
            initial_tokens=self.options.initial_cache_frames * frame_tokens) if growable else None
        # Reserve CUDA/NCCL/allocator overhead outside the tensor budget, also
        # when exercising the 80 GiB profile on a larger development GPU.
        target_bytes = self.options.tensor_budget_bytes(torch.cuda.get_device_properties(pipeline.device).total_memory)
        safety_bytes = int(self.options.gpu_safety_gib * 1024**3)
        base_bytes = int(torch.cuda.memory_allocated(pipeline.device))
        if self.options.kv_storage_mode == "auto_80gb":
            budget = target_bytes - base_bytes - safety_bytes
            if self.options.gpu_cache_budget_gib is not None:
                budget = min(budget, int(self.options.gpu_cache_budget_gib * 1024**3))
            place_cache(cache, device=pipeline.device, budget_bytes=budget)
        elif self.options.kv_storage_mode == "cpu_offload":
            place_cache(cache, device=pipeline.device, budget_bytes=None)
        memory_plan = cache_memory_plan(cache)
        if self.options.cpu_cursors:
            for layer in cache:
                layer["cpu_cursors"] = True
                for name, value in layer.items():
                    if name.endswith("end_index"):
                        layer[name] = value.cpu()
        memory_plan.update(target_bytes=target_bytes, safety_bytes=safety_bytes,
                           admission_is_estimate=True)
        stager = KVStager(cache, pipeline.device, prefetch=self.options.kv_prefetch,
                          delta_writeback=self.options.kv_delta_writeback)
        attention_cache = CacheHandles(cache, stager)
        growth = CacheGrowth(cache, stager, prefix=growing_prefix,
            maximum_tokens=total_frames * frame_tokens, device=pipeline.device,
            target_bytes=target_bytes, safety_bytes=safety_bytes,
            cache_budget_bytes=int(self.options.gpu_cache_budget_gib * 1024**3)
                if self.options.gpu_cache_budget_gib is not None else None) if growable else None
        emit({"event": "kv_cache_memory_planned", **memory_plan})
        record_memory("kv_cache_allocated")
        context_schedule = parse_prompt_schedule(
            request.raw.get("context_prompt_schedule"),
            default_prompt=request.prompt,
            total_chunks=context_frames // chunk,
            name="context_prompt_schedule",
        )
        generation_schedule = parse_prompt_schedule(
            request.raw.get("prompt_schedule"),
            default_prompt=request.prompt,
            total_chunks=output_frames // chunk,
            name="prompt_schedule",
        )
        prompt_values = {prompt for _, prompt in context_schedule + generation_schedule}
        prompt_context = {prompt: _encode_prompt(pipeline, prompt) for prompt in prompt_values}
        record_memory("prompts_encoded")
        pipeline.scheduler.set_timesteps(pipeline.num_train_timesteps, shift=float(self.options.sampling_shift))
        timesteps = pipeline.scheduler.timesteps[list(self.options.timesteps_index)]
        text_len = int(getattr(pipeline.config, "text_len", 512))
        head_dim = int(model_config.dim // model_config.num_heads)
        cross = _cross_cache(pipeline, text_len=text_len, heads=int(model_config.num_heads), head_dim=head_dim)
        online = OnlinePolicyRuntime(
            cache=cache,
            geometry=geometry,
            policy=policy,
            request=request,
            pipeline=pipeline,
            timesteps=timesteps,
            emit=emit,
        )
        generation_kv_policy = str(request.generation.get("generation_kv_policy", self.options.generation_kv_policy))
        ledger = SequenceTokenLedger(
            num_layers=int(model_config.num_layers),
            generation_kv_policy=generation_kv_policy,
        )
        current_prompt_start = None
        cross_first = True
        zero_t = torch.zeros(1, device=pipeline.device, dtype=torch.float32)
        seq_len = chunk * frame_tokens
        max_attention = total_frames * frame_tokens
        emit({"event": "context_replay_started", "context_latent_frames": context_frames, "frame_tokens": frame_tokens})
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
            for frame_start in range(0, context_frames, chunk):
                if growth is not None:
                    growth.ensure(geometry.tokens_per_chunk, emit)
                chunk_index = frame_start // chunk
                prompt_start, prompt_value = active_prompt(context_schedule, chunk_index)
                if current_prompt_start is not None and prompt_start != current_prompt_start:
                    cross = _cross_cache(pipeline, text_len=text_len, heads=int(model_config.num_heads), head_dim=head_dim)
                    cross_first = True
                current_prompt_start = prompt_start
                kwargs = {
                    "context": [prompt_context[prompt_value][0]],
                    "seq_len": seq_len,
                    "y": [condition[:, frame_start:frame_start + chunk]],
                    "dit_cond_dict": {"c2ws_plucker_emb": (camera.packed_chunk(frame_start, frame_start + chunk, device=pipeline.device, dtype=pipeline.param_dtype),)},
                    "kv_cache": attention_cache,
                    "crossattn_cache": cross,
                    "current_start": frame_start * frame_tokens,
                    "max_attention_size": max_attention,
                    "frame_seqlen": frame_tokens,
                    "cross_attn_first_call": cross_first,
                }
                online.prepare_chunk(chunk_index)
                context_scores = online.context_probe(
                    chunk_idx=chunk_index,
                    clean_x=context_latent[:, frame_start:frame_start + chunk],
                    kwargs=kwargs,
                )
                if context_scores is not None:
                    kwargs["cross_attn_first_call"] = False
                online.before_clean(chunk_index, phase="context")
                pipeline.model(x=[context_latent[:, frame_start:frame_start + chunk]], t=zero_t, **kwargs)
                if frame_start == 0:
                    record_memory("first_context_forward")
                online.capture_model_state()
                if policy.name == "fullkv":
                    expected_end = (frame_start + chunk) * frame_tokens
                    for layer in cache:
                        if int(layer["local_end_index"].item()) != expected_end:
                            raise RuntimeError("FullKV context cache did not persist across the model boundary")
                cross_first = False
                if policy.name == "streaming":
                    finish_streaming_chunk(
                        cache,
                        frame_tokens=frame_tokens,
                        sink_frames=int(policy.parameters["sink_size"]),
                        local_frames=int(policy.parameters["local_attn_size"]),
                    )
                online.after_clean(
                    chunk_idx=chunk_index,
                    clean_x=context_latent[:, frame_start:frame_start + chunk],
                    context_scores=context_scores,
                    phase="context",
                )
                emit({"event": "context_chunk_replayed", "chunk_idx": chunk_index})
            torch.cuda.synchronize(pipeline.device)
            record_memory("context_replay_completed")
            if growth is not None:
                growth.ensure(output_frames * frame_tokens, emit)
            matched_context = str(request.generation.get("matched_context_policy", "native"))
            q0_method = str(policy.name)
            q0_parameters = dict(policy.parameters)
            q0_online_policy = online
            if matched_context == "patchification_q0":
                from ..methods.patchification import SPEC as PATCHIFICATION_SPEC

                shared_parameters = dict(PATCHIFICATION_SPEC.defaults)
                if policy.name == "patchification":
                    for key in ("sink_frames", "recent_frames", "grid_rows", "grid_cols", "topk_blocks"):
                        if int(policy.parameters[key]) != int(shared_parameters[key]):
                            raise ValueError(f"matched context requires shared Patchification q0 parameter {key}={shared_parameters[key]}")
                    shared_parameters["update_each_chunk"] = bool(policy.parameters.get("update_each_chunk", False))
                else:
                    shared_parameters["update_each_chunk"] = False
                    q0_online_policy = None
                q0_method = "patchification"
                q0_parameters = shared_parameters
            q0 = finalize_q0(
                cache,
                geometry=geometry,
                method=q0_method,
                parameters=q0_parameters,
                generation_kv_policy=generation_kv_policy,
                rope_plan=rope_plan,
                temporal_rope_frequencies=pipeline.model.freqs,
                online_policy=q0_online_policy,
            )
            q0["method"] = str(policy.name)
            q0["context_selection_method"] = q0_method
            q0["matched_context_policy"] = matched_context
            online.begin_generation()
            q0["online_policy"] = online.diagnostics()
            if policy.name == "patchification" and self.options.kv_storage_mode == "auto_80gb":
                if stager.pending:
                    raise RuntimeError("q0 cannot move an in-flight prefetched cache")
                stager.buffers = [{}, {}]
                plan = cache_memory_plan(cache)
                base = torch.cuda.memory_allocated(pipeline.device) - plan["gpu_resident_bytes"]
                budget = target_bytes - base - safety_bytes
                if self.options.gpu_cache_budget_gib is not None:
                    budget = min(budget, int(self.options.gpu_cache_budget_gib * 1024**3))
                promoted = promote_if_fits(cache, device=pipeline.device, budget_bytes=budget)
                emit({"event": "q0_cache_placement", "all_layers_promoted": promoted, **cache_memory_plan(cache)})
            emit({"event": "q0_finalized", "selected_tokens": q0["selected_tokens"]})
            if dist.is_available() and dist.is_initialized():
                torch.cuda.synchronize(pipeline.device)
                dist.barrier()
            context_elapsed = time.perf_counter() - started
            record_memory("q0_completed")
            ar_started = time.perf_counter()
            generator = torch.Generator(device=pipeline.device)
            generator.manual_seed(int(request.seed))
            noise = torch.randn(16, output_frames, context_latent.shape[2], context_latent.shape[3], device=pipeline.device, dtype=torch.float32, generator=generator)
            generated = []
            current_prompt_start = None
            for output_start in range(0, output_frames, chunk):
                generation_chunk = output_start // chunk
                prompt_start, prompt_value = active_prompt(generation_schedule, generation_chunk)
                if current_prompt_start is None or prompt_start != current_prompt_start:
                    cross = _cross_cache(pipeline, text_len=text_len, heads=int(model_config.num_heads), head_dim=head_dim)
                    cross_first = True
                current_prompt_start = prompt_start
                current = noise[:, output_start:output_start + chunk]
                timeline_start = context_frames + output_start
                kwargs = {
                    "context": [prompt_context[prompt_value][0]],
                    "seq_len": seq_len,
                    "y": [condition[:, timeline_start:timeline_start + chunk]],
                    "dit_cond_dict": {"c2ws_plucker_emb": (camera.packed_chunk(timeline_start, timeline_start + chunk, device=pipeline.device, dtype=pipeline.param_dtype),)},
                    "kv_cache": attention_cache,
                    "crossattn_cache": cross,
                    "current_start": timeline_start * frame_tokens,
                    "max_attention_size": max_attention,
                    "frame_seqlen": frame_tokens,
                    "cross_attn_first_call": cross_first,
                }
                timeline_chunk = timeline_start // chunk
                online.prepare_chunk(timeline_chunk)
                local_active = torch.tensor(
                    [
                        active_history_token_head_units_before_current(
                            [layer], current_noisy_tokens=geometry.tokens_per_chunk
                        )
                        for layer in cache
                    ],
                    device=pipeline.device,
                    dtype=torch.long,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_active, op=dist.ReduceOp.SUM)
                matched_context = str(request.generation.get("matched_context_policy", "native")) == "patchification_q0"
                if matched_context:
                    context_stop = context_frames * frame_tokens
                    generated_stop = context_stop + output_start * frame_tokens
                    local_context_active = torch.tensor(
                        [active_history_token_head_units_for_source_range([layer], current_noisy_tokens=geometry.tokens_per_chunk, source_start=0, source_stop=context_stop) for layer in cache],
                        device=pipeline.device, dtype=torch.long,
                    )
                    local_generated_active = torch.tensor(
                        [active_history_token_head_units_for_source_range([layer], current_noisy_tokens=geometry.tokens_per_chunk, source_start=context_stop, source_stop=generated_stop) for layer in cache],
                        device=pipeline.device, dtype=torch.long,
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(local_context_active, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_generated_active, op=dist.ReduceOp.SUM)
                dense_per_layer = (context_frames + output_start) * frame_tokens * int(model_config.num_heads)
                for layer_idx, active_units in enumerate(local_active.tolist()):
                    ledger.add(LayerTokenStep(
                        chunk_idx=generation_chunk,
                        layer_idx=layer_idx,
                        context_latent_frames=context_frames,
                        completed_generated_latent_frames=output_start,
                        current_noisy_latent_frames=chunk,
                        active_token_head_units=int(active_units),
                        frame_tokens=frame_tokens,
                        num_heads=int(model_config.num_heads),
                        active_context_token_head_units=int(local_context_active[layer_idx].item()) if matched_context else None,
                        active_generated_token_head_units=int(local_generated_active[layer_idx].item()) if matched_context else None,
                    ))
                emit({
                    "event": "generation_history_accounted",
                    "chunk_idx": generation_chunk,
                    "active_token_head_layer_units": int(local_active.sum().item()),
                    "dense_token_head_layer_units": dense_per_layer * int(model_config.num_layers),
                })
                for step_index, step in enumerate(timesteps):
                    flow = pipeline.model(x=[current], t=step.reshape(1).to(pipeline.device), **kwargs)[0]
                    online.capture_model_state()
                    cross_first = False
                    kwargs["cross_attn_first_call"] = False
                    x0 = pipeline._convert_flow_pred_to_x0(flow, current, step, pipeline.scheduler)
                    online.capture_generation_prediction(step_index, x0)
                    if step_index < len(timesteps) - 1:
                        current = pipeline.scheduler.add_noise(
                            x0,
                            torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype),
                            timesteps[step_index + 1],
                        )
                online.before_clean(timeline_chunk, phase="generation")
                pipeline.model(x=[x0], t=zero_t, **kwargs)
                online.capture_model_state()
                if policy.name == "streaming" and str(request.generation.get("generation_kv_policy", self.options.generation_kv_policy)) == "method-native":
                    finish_streaming_chunk(
                        cache,
                        frame_tokens=frame_tokens,
                        sink_frames=int(policy.parameters["sink_size"]),
                        local_frames=int(policy.parameters["local_attn_size"]),
                    )
                online.after_clean(
                    chunk_idx=timeline_chunk,
                    clean_x=x0,
                    context_scores=None,
                    phase="generation",
                )
                generated.append(x0)
                emit({"event": "generation_chunk_completed", "chunk_idx": generation_chunk})
        continuation = torch.cat(generated, dim=1)
        if dist.is_available() and dist.is_initialized():
            torch.cuda.synchronize(pipeline.device)
            dist.barrier()
        ar_elapsed = time.perf_counter() - ar_started
        elapsed = time.perf_counter() - started
        seqpr = ledger.aggregate(expected_chunks=output_frames // chunk)
        memory_plan.update(cache_memory_plan(cache))
        return GeneratedCase(
            context_latents=context_latent,
            continuation_latents=continuation,
            diagnostics={
                "context_latent_frames": context_frames,
                "output_latent_frames": output_frames,
                "frame_tokens": frame_tokens,
                "q0": q0,
                "camera": camera_diagnostics,
                "prompt_count": len(prompt_values),
                "online_policy": online.diagnostics(),
                "kv_cache_memory_plan": memory_plan,
                "seq_pr_metrics": seqpr,
                "token_trajectory": ledger.rows(),
                "gpu_memory_stages": memory_stages,
                "input_cache_hit": input_cache_hit,
                "kv_transfer": stager.diagnostics(),
                "runtime_options": vars(self.options),
            },
            resource_metrics={
                "generation_total_seconds": elapsed,
                "context_preprocessing_seconds": context_elapsed,
                "autoregressive_generation_seconds": ar_elapsed,
                "decoded_frame_fps_nominal": (output_frames * int(self.options.vae_temporal_stride)) / ar_elapsed,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(pipeline.device)),
            },
        )
