"""Two reusable GPU staging slots; host KV remains the source of truth."""
from __future__ import annotations


def tensor_prefix(name):
    for suffix in ("source_ids", "tpos", "k", "v"):
        if name == suffix:
            return ""
        if name.endswith("_" + suffix):
            return name[:-len(suffix)]
    return None


class KVStager:
    def __init__(self, layers, device, *, prefetch=False, delta_writeback=True):
        import torch

        self.layers = layers
        self.device = device
        self.prefetch = prefetch
        self.delta_writeback = delta_writeback
        self.stream = torch.cuda.Stream(device=device) if prefetch else None
        self.buffers = [{}, {}]
        self.pending = {}
        self.h2d_bytes = 0
        self.d2h_bytes = 0

    def _stage(self, index, tokens):
        import torch

        cache = self.layers[index]
        slot = self.buffers[index % 2]
        working = {}
        stream = self.stream or torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            for name, value in cache.items():
                if not isinstance(value, torch.Tensor):
                    working[name] = value
                    continue
                if cache.get("cpu_cursors") and name.endswith("end_index"):
                    working[name] = value.clone()
                    continue
                prefix = tensor_prefix(name)
                source = value
                if prefix is not None and value.ndim >= 2:
                    live = int(cache[prefix + "local_end_index"].item())
                    source = value[:, :min(value.shape[1], live + tokens)]
                buffer = slot.get(name)
                if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                    buffer = torch.empty_like(value, device=self.device)
                    slot[name] = buffer
                target = buffer[:, :source.shape[1]] if prefix is not None and value.ndim >= 2 else buffer
                target.copy_(source, non_blocking=source.is_pinned())
                self.h2d_bytes += source.numel() * source.element_size()
                working[name] = target
            ready = torch.cuda.Event()
            ready.record(stream)
        self.pending[index] = (working, ready)

    def run(self, index, tokens, call):
        import torch

        if index not in self.pending:
            self._stage(index, tokens)
        working, ready = self.pending.pop(index)
        torch.cuda.current_stream(self.device).wait_event(ready)
        if self.prefetch and index + 1 < len(self.layers):
            next_cache = self.layers[index + 1]
            if next_cache["storage_device"] == "cpu":
                self._stage(index + 1, tokens)
        output = call(working)
        cache = self.layers[index]
        delta = self.delta_writeback and cache.get("method") in {"fullkv", "patchification", "random", "decoprune"}
        for name in tuple(cache):
            if name not in working:
                del cache[name]
        for name, value in working.items():
            original = cache.get(name)
            if not isinstance(value, torch.Tensor):
                cache[name] = value
                continue
            if not isinstance(original, torch.Tensor):
                cache[name] = value.cpu()
                self.d2h_bytes += value.numel() * value.element_size()
                continue
            prefix = tensor_prefix(name)
            if prefix is not None and value.ndim >= 2:
                end = int(working[prefix + "local_end_index"].item())
                start = max(0, end - tokens) if delta and prefix == "" else 0
                original[:, start:end].copy_(value[:, start:end])
                self.d2h_bytes += value[:, start:end].numel() * value.element_size()
            else:
                original.copy_(value)
                self.d2h_bytes += value.numel() * value.element_size()
        return output

    def diagnostics(self):
        return {"prefetch": self.prefetch, "delta_writeback": self.delta_writeback,
                "h2d_bytes": self.h2d_bytes, "d2h_bytes": self.d2h_bytes}
