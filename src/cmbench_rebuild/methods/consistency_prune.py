"""DeCoPrune: retain tokens whose clean-x0 consistency error exceeds a threshold."""
from .spec import MethodSpec

DEFAULTS = {
    "step_index": 1,
    "generation_step_index": 2,
    "threshold": 0.10,
    "score_type": "mse",
    "local_window_chunks": 1,
    "min_keep_per_spatial_block": 0,
    "spatial_floor_threshold": float("-inf"),
}

SPEC = MethodSpec(
    "decoprune",
    "DeCoPrune",
    "ours_reindex",
    "consistency_prune",
    DEFAULTS,
)

