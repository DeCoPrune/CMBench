"""Random control: match Ours' keep count with deterministic random tokens."""
from .spec import MethodSpec

SPEC = MethodSpec(
    "random",
    "Random",
    "ours_random_reindex",
    "consistency_random",
    {
        "step_index": 1,
        "generation_step_index": 2,
        "threshold": 0.10,
        "score_type": "mse",
        "local_window_chunks": 1,
        "min_keep_per_spatial_block": 0,
        "spatial_floor_threshold": float("-inf"),
        "reference_method": "decoprune",
    },
)

