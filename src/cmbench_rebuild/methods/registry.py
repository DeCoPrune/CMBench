"""Discover the eight method modules by their small public specs."""
from __future__ import annotations

from .consistency_prune import SPEC as CONSISTENCY_PRUNE
from .consistency_prune_hs import SPEC as CONSISTENCY_PRUNE_HS
from .dummy_forcing import SPEC as DUMMY_FORCING
from .forcingkv import SPEC as FORCINGKV
from .fullkv import SPEC as FULLKV
from .patchification import SPEC as PATCHIFICATION
from .random import SPEC as RANDOM
from .spec import MethodSpec
from .streaming import SPEC as STREAMING


_SPECS = (
    FULLKV,
    STREAMING,
    DUMMY_FORCING,
    FORCINGKV,
    PATCHIFICATION,
    RANDOM,
    CONSISTENCY_PRUNE,
    CONSISTENCY_PRUNE_HS,
)
REGISTRY = {spec.id: spec for spec in _SPECS}
LEGACY_REGISTRY = {spec.legacy_id: spec for spec in _SPECS}

ALIASES = {"consistency_prune": "decoprune", "consistency_prune_hs": "decoprune_hs",
           "DeCoPrune": "decoprune", "DeCoPrune-HS": "decoprune_hs"}

def resolve(name: str) -> MethodSpec:
    name = ALIASES.get(name, name)
    try:
        return REGISTRY.get(name) or LEGACY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown method {name!r}; choices: {', '.join(REGISTRY)}") from exc
