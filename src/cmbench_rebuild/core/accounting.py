"""Method-independent token accounting and per-layer trajectory ledger."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def account_token_union_q0(retained_token_ids_by_group: Iterable[Iterable[int]], total_context_tokens: int) -> dict[str, Any]:
    total = int(total_context_tokens)
    if total < 0:
        raise ValueError("total_context_tokens must be non-negative")
    retained: set[int] = set()
    for group in retained_token_ids_by_group:
        for value in group:
            token_id = int(value)
            if token_id < 0 or token_id >= total:
                raise ValueError(f"retained token ID {token_id} is outside [0, {total})")
            retained.add(token_id)
    kept = len(retained)
    return {
        "retained_unique_context_tokens_q0": kept,
        "pruned_unique_context_tokens_q0": total - kept,
        "total_unique_context_tokens_q0": total,
        "token_prune_ratio_context_q0": (total - kept) / total if total else None,
    }


@dataclass(frozen=True)
class LayerTokenStep:
    chunk_idx: int
    layer_idx: int
    context_latent_frames: int
    completed_generated_latent_frames: int
    current_noisy_latent_frames: int
    active_token_head_units: int
    frame_tokens: int
    num_heads: int
    active_context_token_head_units: int | None = None
    active_generated_token_head_units: int | None = None

    def normalized(self) -> dict[str, Any]:
        values = asdict(self)
        split_active = values.pop("active_context_token_head_units"), values.pop("active_generated_token_head_units")
        if any(int(value) < 0 for value in values.values()):
            raise ValueError(f"layer token values must be non-negative: {values}")
        if self.current_noisy_latent_frames <= 0 or self.frame_tokens <= 0 or self.num_heads <= 0:
            raise ValueError("current chunk, frame_tokens and num_heads must be positive")
        dense_frames = self.context_latent_frames + self.completed_generated_latent_frames
        dense = dense_frames * self.frame_tokens * self.num_heads
        active = int(self.active_token_head_units)
        if active > dense:
            raise ValueError(f"active units {active} exceed dense history {dense}")
        result = {
            "schema_version": 1,
            "metric": "per_layer_token_trajectory",
            **values,
            "dense_history_latent_frames": dense_frames,
            "dense_token_head_units": dense,
            "masked_token_head_units": dense - active,
            "prune_ratio": 1.0 - active / dense if dense else None,
            "current_noisy_excluded_from_numerator": True,
            "current_noisy_excluded_from_denominator": True,
        }
        context_active, generated_active = split_active
        if context_active is not None or generated_active is not None:
            context_active = int(context_active or 0)
            generated_active = int(generated_active or 0)
            context_dense = int(self.context_latent_frames) * int(self.frame_tokens) * int(self.num_heads)
            generated_dense = int(self.completed_generated_latent_frames) * int(self.frame_tokens) * int(self.num_heads)
            if context_active > context_dense or generated_active > generated_dense:
                raise ValueError("context/generated active units exceed their dense history")
            result.update({
                "active_context_token_head_units": context_active,
                "dense_context_token_head_units": context_dense,
                "context_prune_ratio": 1.0 - context_active / context_dense if context_dense else None,
                "active_generated_token_head_units": generated_active,
                "dense_generated_token_head_units": generated_dense,
                "generated_prune_ratio": 1.0 - generated_active / generated_dense if generated_dense else None,
            })
        return result


class SequenceTokenLedger:
    """Require one accounting snapshot for every chunk/layer pair."""

    def __init__(self, *, num_layers: int, generation_kv_policy: str) -> None:
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.generation_kv_policy = str(generation_kv_policy)
        self._rows: dict[tuple[int, int], dict[str, Any]] = {}

    def add(self, step: LayerTokenStep) -> dict[str, Any]:
        row = step.normalized()
        if step.layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {step.layer_idx} is outside [0, {self.num_layers})")
        key = (step.chunk_idx, step.layer_idx)
        if key in self._rows:
            raise ValueError(f"duplicate token trajectory row for chunk/layer {key}")
        self._rows[key] = row
        return row

    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[key] for key in sorted(self._rows)]

    def aggregate(self, *, expected_chunks: int | None = None) -> dict[str, Any]:
        rows = self.rows()
        chunks = sorted({int(row["chunk_idx"]) for row in rows})
        if expected_chunks is not None and chunks != list(range(int(expected_chunks))):
            raise ValueError(f"chunk coverage mismatch: expected 0..{int(expected_chunks) - 1}, got {chunks}")
        missing = [
            (chunk, layer)
            for chunk in chunks
            for layer in range(self.num_layers)
            if (chunk, layer) not in self._rows
        ]
        if missing:
            raise ValueError(f"missing per-layer token rows: {missing}")
        active = sum(int(row["active_token_head_units"]) for row in rows if int(row["dense_token_head_units"]) > 0)
        dense = sum(int(row["dense_token_head_units"]) for row in rows if int(row["dense_token_head_units"]) > 0)
        if active > dense:
            raise ValueError(f"active units {active} exceed dense history {dense}")
        result = {
            "schema_version": 2,
            "metric": "seqPR",
            "generation_kv_policy": self.generation_kv_policy,
            "generation_chunks": len(chunks),
            "num_layers": self.num_layers,
            "per_layer_trajectory_complete": True,
            "active_token_head_layer_units": active,
            "dense_token_head_layer_units": dense,
            "seqPR": 1.0 - active / dense if dense else 0.0,
            "seq_prune_ratio": 1.0 - active / dense if dense else 0.0,
            "aggregation": "ratio_of_chunk_layer_totals_within_video",
            "steps": rows,
        }
        if rows and all("active_context_token_head_units" in row for row in rows):
            context_active = sum(int(row["active_context_token_head_units"]) for row in rows)
            context_dense = sum(int(row["dense_context_token_head_units"]) for row in rows)
            generated_active = sum(int(row["active_generated_token_head_units"]) for row in rows)
            generated_dense = sum(int(row["dense_generated_token_head_units"]) for row in rows)
            result.update({
                "context_seqPR": 1.0 - context_active / context_dense if context_dense else 0.0,
                "continuation_only_seqPR": 1.0 - generated_active / generated_dense if generated_dense else 0.0,
                "active_context_token_head_layer_units": context_active,
                "dense_context_token_head_layer_units": context_dense,
                "active_generated_token_head_layer_units": generated_active,
                "dense_generated_token_head_layer_units": generated_dense,
            })
        return result

    def write(self, output_dir: Path, *, expected_chunks: int | None = None) -> dict[str, Any]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        summary = self.aggregate(expected_chunks=expected_chunks)
        trajectory = root / "token_trajectory.jsonl"
        trajectory.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows()),
            encoding="utf-8",
        )
        (root / "seq_pr_metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
