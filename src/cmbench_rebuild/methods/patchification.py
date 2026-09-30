"""Patchification: select low-similarity spatial blocks at q0."""
from .spec import MethodSpec

SPEC = MethodSpec(
    "patchification",
    "Patchification",
    "patchify_reindex",
    "patchify_retrieve",
    {
        "sink_frames": 4,
        "recent_frames": 4,
        "grid_rows": 6,
        "grid_cols": 13,
        "topk_blocks": 1657,
        "update_each_chunk": False,
    },
)
