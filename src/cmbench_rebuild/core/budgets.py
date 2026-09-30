"""Pure budget arithmetic shared by strategy implementations."""
from __future__ import annotations


def forcingkv_middle_window_start_frame(*, chunk_index: int, chunk_size: int, dynamic_recent_chunks: int, middle_window_chunks: int) -> int:
    values = (chunk_size, dynamic_recent_chunks, middle_window_chunks)
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    if any(value <= 0 for value in values):
        raise ValueError("chunk and window sizes must be positive")
    current_total = (chunk_index + 1) * chunk_size
    excluded_recent = dynamic_recent_chunks * chunk_size
    eligible_middle = middle_window_chunks * chunk_size
    return max(0, current_total - excluded_recent - eligible_middle)
