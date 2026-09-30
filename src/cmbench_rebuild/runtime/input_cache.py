"""Bounded CPU reuse of exact VAE outputs within one loaded checkpoint."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


class InputCache:
    def __init__(self, maximum_bytes=512 * 1024**2):
        self.maximum_bytes = maximum_bytes
        self.items = OrderedDict()
        self.bytes = 0

    def get(self, key):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key, tensors):
        size = sum(t.numel() * t.element_size() for t in tensors)
        if size > self.maximum_bytes:
            return
        if key in self.items:
            self.bytes -= sum(t.numel() * t.element_size() for t in self.items.pop(key))
        while self.items and self.bytes + size > self.maximum_bytes:
            _, removed = self.items.popitem(last=False)
            self.bytes -= sum(t.numel() * t.element_size() for t in removed)
        self.items[key] = tuple(t.detach().to("cpu", copy=True) for t in tensors)
        self.bytes += size


def input_key(context, *, total_frames, stride, dtype):
    source = context.diagnostics.get("source_sha256")
    if not source:
        return None
    identity = {"source": source, "diagnostics": dict(context.diagnostics),
                "pixel_frames": context.pixel_frames, "latent_frames": context.latent_frames,
                "total_frames": total_frames, "stride": stride, "dtype": str(dtype)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def move_vae(vae, device):
    vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [value.to(device) for value in vae.scale]
    vae.device = device


def prepare_inputs(pipeline, context, *, output_frames, stride, reuse=True):
    from .conditioning import _distributed, encode_context_rank0, build_i2v_condition_rank0

    total = context.latent_frames + output_frames
    key = input_key(context, total_frames=total, stride=stride, dtype=pipeline.pipe_dtype)
    if not hasattr(pipeline, "_cmbench_inputs"):
        pipeline._cmbench_inputs = InputCache()
    store = pipeline._cmbench_inputs
    cached = store.get(key) if reuse and key is not None else None
    if cached is not None:
        latent, condition = [tensor.to(pipeline.device) for tensor in cached]
        return latent, condition, True
    _, rank = _distributed()
    # Only rank zero encodes. Other ranks only receive the resulting latents.
    move_vae(pipeline.vae, pipeline.device if rank == 0 else "cpu")
    latent = encode_context_rank0(pipeline.vae, context.payload, device=pipeline.device).float()
    condition = build_i2v_condition_rank0(
        pipeline.vae, context.payload[:, 0] if rank == 0 else None,
        total_latent_frames=total, temporal_stride=stride,
        device=pipeline.device, output_dtype=pipeline.pipe_dtype,
    )
    move_vae(pipeline.vae, "cpu")
    if reuse and key is not None:
        store.put(key, (latent, condition))
    return latent, condition, False
