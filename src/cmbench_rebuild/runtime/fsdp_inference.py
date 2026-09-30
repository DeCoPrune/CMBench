"""Inference-only FSDP policy kept outside the pinned LingBot source tree."""
from __future__ import annotations

from functools import partial
import os
from typing import Any


def shard_for_inference(
    model: Any,
    device_id: int,
    *,
    param_dtype: Any = None,
    reduce_dtype: Any = None,
    buffer_dtype: Any = None,
    process_group: Any = None,
    sync_module_states: bool = True,
    use_lora: bool = False,
) -> Any:
    """Fully shard each transformer block and reshard it after every forward."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    strategy_name = os.environ.get("CMBENCH_FSDP_STRATEGY", "full_shard").strip().lower()
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }
    if strategy_name not in strategies:
        raise ValueError(
            "CMBENCH_FSDP_STRATEGY must be full_shard or shard_grad_op; "
            f"got {strategy_name!r}"
        )

    return FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=strategies[strategy_name],
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module in model.blocks,
        ),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype or torch.bfloat16,
            reduce_dtype=reduce_dtype or torch.float32,
            buffer_dtype=buffer_dtype or torch.float32,
        ),
        device_id=device_id,
        limit_all_gathers=True,
        forward_prefetch=os.environ.get("CMBENCH_FSDP_PREFETCH", "1") == "1",
        sync_module_states=sync_module_states,
        use_orig_params=bool(use_lora),
    )
