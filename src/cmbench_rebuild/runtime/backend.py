"""Loading boundary for the pinned LingBot World v2 causal-fast model."""
from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointLayout, inspect_checkpoint
from .generation import LingBotGenerationEngine, LingBotGenerationOptions
from .sequence_parallel import policy_attn_forward_causal, sp_policy_attn_forward_causal
from .vendor import verify_vendored_source


@dataclass(frozen=True)
class LingBotLoadOptions:
    device_id: int = 0
    rank: int = 0
    world_size: int = 1
    use_sequence_parallel: bool = False
    dit_fsdp: bool = False
    t5_fsdp: bool = False
    t5_cpu: bool = True
    init_on_cpu: bool = False
    local_attn_size: int = -1
    sink_size: int = 0

    def validate(self) -> "LingBotLoadOptions":
        if min(int(self.device_id), int(self.rank)) < 0 or int(self.world_size) <= 0:
            raise ValueError("device/rank must be non-negative and world_size positive")
        if int(self.rank) >= int(self.world_size):
            raise ValueError("rank must be smaller than world_size")
        if bool(self.use_sequence_parallel) != (int(self.world_size) > 1):
            raise ValueError("sequence parallel must be enabled exactly when world_size > 1")
        if int(self.local_attn_size) == 0 or int(self.local_attn_size) < -1:
            raise ValueError("local_attn_size must be -1 or positive")
        if int(self.sink_size) < 0 or (int(self.local_attn_size) > 0 and int(self.sink_size) >= int(self.local_attn_size)):
            raise ValueError("sink_size must fit inside the local attention window")
        return self


PipelineFactory = Callable[[Path, LingBotLoadOptions], Any]


def _official_pipeline_factory(checkpoint: Path, options: LingBotLoadOptions) -> Any:
    from wan.configs import i2v_A14B
    import wan.image2video as image2video

    from .fsdp_inference import shard_for_inference

    original_shard_model = image2video.shard_model
    image2video.shard_model = shard_for_inference
    try:
        return image2video.WanI2VCausal(
            config=i2v_A14B,
            checkpoint_dir=str(checkpoint),
            device_id=int(options.device_id),
            rank=int(options.rank),
            t5_fsdp=bool(options.t5_fsdp),
            dit_fsdp=bool(options.dit_fsdp),
            use_sp=bool(options.use_sequence_parallel),
            t5_cpu=bool(options.t5_cpu),
            init_on_cpu=bool(options.init_on_cpu),
            local_attn_size=int(options.local_attn_size),
            sink_size=int(options.sink_size),
            infer_mode="causal_fast",
        )
    finally:
        image2video.shard_model = original_shard_model


class LingBotBackend:
    """Owns model loading; generation state is always created per case."""

    def __init__(self, options: LingBotLoadOptions, *, generation_options: LingBotGenerationOptions | None = None, head_map: Any | None = None, pipeline_factory: PipelineFactory | None = None) -> None:
        self.options = options.validate()
        self._pipeline_factory = pipeline_factory or _official_pipeline_factory
        self.generation_options = (generation_options or LingBotGenerationOptions()).validate()
        self.head_map = head_map
        self._pipeline: Any | None = None
        self._layout: CheckpointLayout | None = None
        self.load_manifest: dict[str, Any] | None = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError("LingBot backend is not loaded")
        return self._pipeline

    def load(self, checkpoint: Path) -> None:
        source = verify_vendored_source()
        layout = inspect_checkpoint(checkpoint)
        if self._layout is not None:
            if layout.root != self._layout.root:
                raise RuntimeError("one backend instance cannot switch checkpoints")
            return
        pipeline = self._pipeline_factory(layout.root, self.options)
        if not hasattr(pipeline, "model") or not hasattr(pipeline.model, "blocks"):
            raise TypeError("LingBot pipeline factory returned no causal transformer blocks")
        attention_forward = sp_policy_attn_forward_causal if self.options.use_sequence_parallel else policy_attn_forward_causal
        for layer_index, block in enumerate(pipeline.model.blocks):
            block.self_attn.forward = types.MethodType(attention_forward, block.self_attn)
            block.self_attn.cmbench_layer_index = layer_index
        if hasattr(pipeline.model, "register_forward_pre_hook"):
            from .cache_handles import protect_cache_inputs
            pipeline.model.register_forward_pre_hook(protect_cache_inputs, with_kwargs=True)
        self._pipeline = pipeline
        self._layout = layout
        self.load_manifest = {
            "backend": "lingbot_world_v2_causal_fast",
            "checkpoint": layout.manifest(),
            "source": source,
            "options": vars(self.options),
            "attention_backend": (
                "cmbench_policy_ulysses" if self.options.use_sequence_parallel else "cmbench_policy_single_device"
            ),
            "bound_attention_module": attention_forward.__module__,
            "bound_attention_function": attention_forward.__name__,
            "fsdp": {"strategy": str(getattr(pipeline.model, "sharding_strategy", "none")),
                     "forward_prefetch": bool(getattr(pipeline.model, "forward_prefetch", False))},
        }

    def generate(self, request: Any, context: Any, policy: Any, rope_plan: Any, emit: Callable[[dict[str, Any]], None]) -> Any:
        engine = LingBotGenerationEngine(
            self.pipeline,
            load_options=self.options,
            options=self.generation_options,
            head_map=self.head_map,
        )
        return engine.generate(request, context, policy, rope_plan, emit)
