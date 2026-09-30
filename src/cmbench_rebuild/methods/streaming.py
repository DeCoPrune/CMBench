"""StreamingKV: retain a fixed sink and rolling recent window."""
from .spec import MethodSpec

SPEC = MethodSpec(
    "streaming",
    "Streaming",
    "streaming_reindex",
    "streaming",
    {"sink_size": 4, "local_attn_size": 8},
)


def finish_chunk(
    cache_layers,
    *,
    frame_tokens: int,
    sink_frames: int,
    local_frames: int,
) -> None:
    """Compact every layer to the shared sink-plus-recent token set."""
    from ..runtime.compaction import keep_global_source_ids

    reference = cache_layers[0]
    live = int(reference["local_end_index"].item())
    source_ids = reference["source_ids"][0, :live]
    source_frames = source_ids.div(int(frame_tokens), rounding_mode="floor")
    newest_frame = int(source_frames.max().item())
    recent_start = max(0, newest_frame - (int(local_frames) - int(sink_frames)) + 1)
    keep = (source_frames < int(sink_frames)) | (source_frames >= recent_start)
    keep_global_source_ids(cache_layers, source_ids[keep])
