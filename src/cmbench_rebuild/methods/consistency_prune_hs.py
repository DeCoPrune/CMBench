"""DeCoPrune-HS: apply consistency pruning only to configured dynamic heads."""
from .consistency_prune import DEFAULTS
from .spec import MethodSpec

SPEC = MethodSpec(
    "decoprune_hs",
    "DeCoPrune-HS",
    "ours_fkv_reindex",
    "consistency_prune_fkv",
    {**DEFAULTS, "head_map_required": True},
)

