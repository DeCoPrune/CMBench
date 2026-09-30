"""Keep mutable KV state opaque to FSDP's recursive input conversion."""
from collections.abc import MutableMapping, Sequence
import weakref


class LayerCache(MutableMapping):
    """A reference to one original cache dictionary, never a tensor copy."""

    def __init__(self, data, index, owner):
        self.data = data
        self.index = index
        self.owner = weakref.proxy(owner)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class CacheHandles(Sequence):
    """Opaque at both the whole-model and individual-block FSDP boundaries."""

    def __init__(self, layers, stager=None):
        self.layers = layers
        self.stager = stager
        self.handles = [LayerCache(layer, index, self) for index, layer in enumerate(layers)]

    def __getitem__(self, index):
        return self.handles[index]

    def __len__(self):
        return len(self.handles)


def protect_cache_inputs(module, args, kwargs):
    """Also protect auxiliary policy probes that pass an ordinary cache list."""
    cache = kwargs.get("kv_cache")
    if isinstance(cache, list):
        kwargs = dict(kwargs)
        kwargs["kv_cache"] = CacheHandles(cache)
    return args, kwargs
