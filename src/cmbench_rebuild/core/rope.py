"""Framework-independent RoPE re-index planning."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RopeReindexPlan:
    mode: str = "all_bands"
    virtual_span: float = 17.0
    recent_frames: int = 8
    fast_band_pairs: int = 11

    def validate(self) -> "RopeReindexPlan":
        if self.mode not in {"off", "all_bands", "fast_bands"}:
            raise ValueError(f"unknown RoPE re-index mode: {self.mode}")
        if self.recent_frames < 0 or self.fast_band_pairs < 1:
            raise ValueError("recent_frames must be non-negative and fast_band_pairs positive")
        if self.mode != "off" and self.virtual_span <= self.recent_frames:
            raise ValueError("virtual_span must exceed recent_frames")
        return self

    def virtual_positions(self, context_frames: int) -> tuple[float, ...]:
        """Return the legacy-equivalent virtual position for every real frame."""
        self.validate()
        if context_frames < 0:
            raise ValueError("context_frames must be non-negative")
        if self.mode == "off" or context_frames <= self.recent_frames:
            return tuple(float(value) for value in range(context_frames))
        q0 = context_frames
        result = []
        for real in range(q0):
            offset = q0 - real
            virtual_offset = offset if offset <= self.recent_frames else self.recent_frames + (offset - self.recent_frames) * ((self.virtual_span - self.recent_frames) / (q0 - self.recent_frames))
            result.append(q0 - virtual_offset)
        return tuple(result)

    def temporal_pair_count(self, rope_pair_count: int) -> int:
        if rope_pair_count < 0:
            raise ValueError("rope_pair_count must be non-negative")
        temporal = rope_pair_count - 2 * (rope_pair_count // 3)
        return temporal if self.mode == "all_bands" else min(self.fast_band_pairs, temporal)
