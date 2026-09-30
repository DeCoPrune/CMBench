"""Fit as many complete layers as possible in a measured GPU memory budget."""
from __future__ import annotations


def resident_layer_indices(layer_bytes, *, budget_bytes, staging_slots=2):
    """Reserve staging space first; never assume a method will prune tokens."""
    if sum(layer_bytes) <= budget_bytes:
        return set(range(len(layer_bytes)))
    available = budget_bytes - staging_slots * max(layer_bytes, default=0)
    if available < 0:
        raise RuntimeError("GPU cache budget cannot fit staging buffers")
    resident = set()
    for index, size in enumerate(layer_bytes):
        if size <= available:
            resident.add(index)
            available -= size
    return resident


def place_cache(layers, *, device, budget_bytes):
    import torch
    from .cache_factory import cache_memory_plan

    sizes = cache_memory_plan(layers)["layer_bytes"]
    resident = set() if budget_bytes is None else resident_layer_indices(sizes, budget_bytes=budget_bytes)
    for index, layer in enumerate(layers):
        target = device if index in resident else torch.device("cpu")
        for name, value in layer.items():
            if isinstance(value, torch.Tensor):
                # Buffers are empty at placement time. Only metadata is initialized.
                if name == "k" or name == "v" or name.endswith(("_k", "_v")):
                    layer[name] = torch.empty_like(value, device=target, pin_memory=target.type == "cpu")
                else:
                    layer[name] = value.to(target)
                    if target.type == "cpu" and not layer[name].is_pinned():
                        layer[name] = layer[name].pin_memory()
        layer["storage_device"] = str(target)
    return resident


def promote_if_fits(layers, *, device, budget_bytes):
    """After q0 shrinking, move a now-small cache back to GPU without reallocation of resident banks."""
    import torch
    from .cache_factory import cache_memory_plan

    if cache_memory_plan(layers)["persistent_bytes"] > budget_bytes:
        return False
    for layer in layers:
        if layer["storage_device"] != "cpu":
            continue
        for name, value in layer.items():
            if not isinstance(value, torch.Tensor):
                continue
            if layer.get("cpu_cursors") and name.endswith("end_index"):
                continue
            layer[name] = value.to(device, non_blocking=value.is_pinned())
        layer["storage_device"] = str(device)
    return True
