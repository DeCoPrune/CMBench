"""Grow threshold-policy banks losslessly, spilling only when they really fill."""
from .cache_factory import cache_memory_plan
from .cache_placement import resident_layer_indices
from .compaction import movable_names, resize_live_banks
from .kv_offload import tensor_prefix


def start_small(layers, *, method, initial_tokens):
    prefix = "dynamic_" if method == "decoprune_hs" else ""
    capacity = min(initial_tokens, layers[0][prefix + "k"].shape[1])
    resize_live_banks(layers, capacity=capacity, prefixes=(prefix,))
    return prefix


class CacheGrowth:
    def __init__(self, layers, stager, *, prefix, maximum_tokens, device, target_bytes, safety_bytes, cache_budget_bytes=None):
        self.layers, self.stager, self.prefix = layers, stager, prefix
        self.maximum_tokens, self.device = maximum_tokens, device
        self.target_bytes, self.safety_bytes = target_bytes, safety_bytes
        self.cache_budget_bytes = cache_budget_bytes

    def ensure(self, extra_tokens, emit):
        import torch

        prefix = self.prefix
        current = self.layers[0][prefix + "k"].shape[1]
        required = max(int(layer[prefix + "local_end_index"].item()) for layer in self.layers) + extra_tokens
        if required <= current:
            return
        if required > self.maximum_tokens:
            raise RuntimeError("live KV would exceed the complete case timeline")
        capacity = min(self.maximum_tokens, max(required, current + max(current // 2, extra_tokens)))
        torch.cuda.synchronize(self.device)
        if self.stager.pending:
            raise RuntimeError("cannot resize KV during a prefetched forward")
        self.stager.buffers = [{}, {}]
        old_plan = cache_memory_plan(self.layers)
        base = torch.cuda.memory_allocated(self.device) - old_plan["gpu_resident_bytes"]
        sizes = []
        for layer, old_size in zip(self.layers, old_plan["layer_bytes"]):
            extra = sum(layer[name].numel() * layer[name].element_size() // current * (capacity - current)
                        for name in movable_names(layer, prefix))
            sizes.append(old_size + extra)
        budget = self.target_bytes - base - self.safety_bytes
        if self.cache_budget_bytes is not None:
            budget = min(budget, self.cache_budget_bytes)
        resident = resident_layer_indices(sizes, budget_bytes=budget)
        # Spill first, freeing old GPU banks before allocating larger resident banks.
        order = sorted(range(len(self.layers)), key=lambda index: index in resident)
        for index in order:
            layer = self.layers[index]
            target = self.device if index in resident else torch.device("cpu")
            growing = movable_names(layer, prefix)
            for name, value in layer.items():
                if not isinstance(value, torch.Tensor):
                    continue
                device = torch.device("cpu") if layer.get("cpu_cursors") and name.endswith("end_index") else target
                shape = list(value.shape)
                if name in growing:
                    shape[1] = capacity
                if value.device == device and list(value.shape) == shape:
                    continue
                replacement = torch.empty(shape, dtype=value.dtype, device=device, pin_memory=device.type == "cpu")
                bank = tensor_prefix(name)
                if bank is not None and value.ndim >= 2:
                    live = int(layer[bank + "local_end_index"].item())
                    replacement[:, :live].copy_(value[:, :live])
                    if name.endswith(("source_ids", "tpos")):
                        replacement[:, live:].fill_(-1)
                else:
                    replacement.copy_(value)
                layer[name] = replacement
            layer["storage_device"] = str(target)
        emit({"event": "kv_cache_grown", "old_capacity_tokens": current,
              "capacity_tokens": capacity, **cache_memory_plan(self.layers)})
