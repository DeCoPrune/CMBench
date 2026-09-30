"""Online method state machines operating on physical KV token identity.

The model-facing attention path only writes and reads banks.  This module owns
the method lifecycle between clean causal chunks: diagnostic probes, delayed
consistency compaction, ForcingKV memory updates, and DummyForcing allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..methods.consistency import build_consistency_mask
from ..methods.forcingkv import update_forcingkv_cache
from ..methods.selectors import (
    deterministic_random_indices,
    dummyforcing_context_banks,
    dummyforcing_group_assignment,
    patchification_block_token_ids,
)
from ..methods.torch_kernels import forcingkv_scores_from_moments, patchification_block_moments
from .compaction import compact_static_predecessor, keep_global_source_ids
from .kv_cache import cache_banks


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def snapshot_physical_cache(cache_layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Capture exactly the live self-attention state for a reversible probe."""
    snapshots: list[dict[str, Any]] = []
    for cache in cache_layers:
        item: dict[str, Any] = {"global_end_index": _int(cache["global_end_index"])}
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            live = _int(cache[bank.end_name])
            item[bank.end_name] = live
            for suffix in ("k", "v", "tpos", "source_ids"):
                name = f"{prefix}{suffix}"
                if name in cache:
                    item[name] = cache[name][:, :live].clone()
        item["local_end_index"] = _int(cache["local_end_index"])
        snapshots.append(item)
    return snapshots


def restore_physical_cache(cache_layers: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    if len(cache_layers) != len(snapshots):
        raise ValueError("cache snapshot layer count changed")
    for cache, item in zip(cache_layers, snapshots):
        cache["global_end_index"].fill_(int(item["global_end_index"]))
        cache["local_end_index"].fill_(int(item["local_end_index"]))
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            live = int(item[bank.end_name])
            cache[bank.end_name].fill_(live)
            for suffix in ("k", "v", "tpos", "source_ids"):
                name = f"{prefix}{suffix}"
                if name in item:
                    cache[name][:, :live].copy_(item[name])


def snapshot_cache_cursors(cache_layers: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Capture live boundaries without cloning the potentially huge KV payload."""
    cursors: list[dict[str, int]] = []
    for cache in cache_layers:
        item = {
            "global_end_index": _int(cache["global_end_index"]),
            "local_end_index": _int(cache["local_end_index"]),
        }
        for bank in cache_banks(cache):
            item[bank.end_name] = _int(cache[bank.end_name])
        cursors.append(item)
    return cursors


def restore_cache_cursors(cache_layers: list[dict[str, Any]], cursors: list[dict[str, int]]) -> None:
    if len(cache_layers) != len(cursors):
        raise ValueError("cache cursor layer count changed")
    for cache, item in zip(cache_layers, cursors):
        cache["global_end_index"].fill_(int(item["global_end_index"]))
        cache["local_end_index"].fill_(int(item["local_end_index"]))
        for bank in cache_banks(cache):
            cache[bank.end_name].fill_(int(item[bank.end_name]))


def isolated_probe_cache(cache_layers: list[dict[str, Any]], *, capacity_tokens: int) -> list[dict[str, Any]]:
    """Build a one-chunk cache with the same head layout but no shared payload."""
    import torch

    capacity = int(capacity_tokens)
    if capacity <= 0:
        raise ValueError("probe cache capacity must be positive")
    isolated: list[dict[str, Any]] = []
    for cache in cache_layers:
        clone = dict(cache)
        clone["global_end_index"] = cache["global_end_index"].clone()
        clone["local_end_index"] = torch.zeros_like(cache["local_end_index"])
        for bank in cache_banks(cache):
            prefix = f"{bank.prefix}_" if bank.prefix else ""
            clone[bank.end_name] = torch.zeros_like(cache[bank.end_name])
            for suffix in ("k", "v", "tpos", "source_ids"):
                name = f"{prefix}{suffix}"
                if name not in cache:
                    continue
                shape = list(cache[name].shape)
                shape[1] = capacity
                clone[name] = torch.empty(shape, dtype=cache[name].dtype, device=cache[name].device)
                if suffix in {"tpos", "source_ids"}:
                    clone[name].fill_(-1)
        isolated.append(clone)
    return isolated


def empty_physical_cache_at(cache_layers: list[dict[str, Any]], global_start: int) -> None:
    """Give a diagnostic call its own current chunk but no predecessor KV."""
    for cache in cache_layers:
        cache["global_end_index"].fill_(int(global_start))
        cache["local_end_index"].zero_()
        for bank in cache_banks(cache):
            cache[bank.end_name].zero_()


def _request_parameter(raw: Mapping[str, Any], parameters: Mapping[str, Any], request_name: str, default_name: str) -> Any:
    return raw[request_name] if request_name in raw else parameters[default_name]


@dataclass
class ConsistencyRecord:
    chunk_idx: int
    source_ids: Any
    scores: Any
    source: str
    applied: bool = False


class OnlinePolicyRuntime:
    """One per-case lifecycle object shared by context replay and generation."""

    def __init__(
        self,
        *,
        cache: list[dict[str, Any]],
        geometry: Any,
        policy: Any,
        request: Any,
        pipeline: Any,
        timesteps: Any,
        emit: Callable[[Mapping[str, Any]], None],
    ) -> None:
        self.cache = cache
        self.geometry = geometry
        self.policy = policy
        self.request = request
        self.pipeline = pipeline
        self.timesteps = timesteps
        self.emit = emit
        self.name = str(policy.name)
        self.parameters = dict(policy.parameters)
        self.events: list[dict[str, Any]] = []
        self.records: list[ConsistencyRecord] = []
        self.forcing_scores: list[float] = []
        self.forcing_patch_starts: list[int] = []
        self.probe_calls = 0
        self._generation_prediction = None
        self._dummy_local_scores = None
        self._dummy_capture_attempts = 0
        self._dummy_middle_bank_ids: tuple[int, ...] = ()
        self._patchification_state: dict[str, Any] | None = None
        self._isolated_probe_cache: list[dict[str, Any]] | None = None
        self._append_only_generation = str(request.generation.get("generation_kv_policy", "method-native")) == "append-only"
        self.matched_context_policy = str(request.generation.get("matched_context_policy", "native"))
        if self.matched_context_policy not in {"native", "patchification_q0"}:
            raise ValueError(f"unknown matched context policy: {self.matched_context_policy}")
        if self.is_consistency:
            raw = request.raw
            self.context_step_index = int(_request_parameter(raw, self.parameters, "ours_step_index", "step_index"))
            self.generation_step_index = int(self.parameters.get("generation_step_index", 2))
            self.threshold = float(_request_parameter(raw, self.parameters, "ours_threshold", "threshold"))
            self.score_type = str(raw.get("ours_score_type", self.parameters.get("score_type", "mse")))
            self.local_window_chunks = int(raw.get("ours_local_window_chunks", self.parameters.get("local_window_chunks", 1)))
            self.min_keep_per_block = int(raw.get("ours_min_keep_per_spatial_block", self.parameters.get("min_keep_per_spatial_block", 0)))
            self.spatial_floor_threshold = float(raw.get("ours_spatial_floor_threshold", self.parameters.get("spatial_floor_threshold", float("-inf"))))
            if not 0 <= self.context_step_index < len(timesteps) or not 0 <= self.generation_step_index < len(timesteps):
                raise ValueError("consistency probe step is outside the configured timestep schedule")
            if self.local_window_chunks < 0:
                raise ValueError("consistency local_window_chunks must be non-negative")
        if self.name == "dummy_forcing":
            self._configure_dummy_warmup()

    @property
    def is_consistency(self) -> bool:
        return self.name in {"random", "decoprune", "decoprune_hs"}

    @property
    def consistency_prefix(self) -> str:
        return "dynamic_" if self.name == "decoprune_hs" else ""

    def _configure_dummy_warmup(self) -> None:
        import torch

        for layer, cache in enumerate(self.cache):
            cache.update({
                "dummyforcing_ar_start": 2,
                "dummyforcing_query_block_size": 64,
                "dummyforcing_chunk_size": int(self.geometry.chunk_size),
                "dummyforcing_classification_evidence": "original_packed",
                "dummyforcing_sample_seed": int(self.request.seed) + 32452843 * (layer + 1),
                "dummyforcing_first_region_frames": int(self.geometry.chunk_size),
                "dummyforcing_recent_region_frames": int(self.geometry.chunk_size),
                "dummyforcing_score_at_global_end": 3 * int(self.geometry.tokens_per_chunk),
                # FSDP may copy the surrounding Python dict. Preallocated
                # tensors preserve attention-side mutations across that copy.
                "dummyforcing_group_scores_buffer": torch.full(
                    (int(self.geometry.local_heads), 3),
                    float("nan"),
                    dtype=torch.float32,
                    device=cache["k"].device,
                ),
                "dummyforcing_group_scores_valid": torch.zeros(
                    (), dtype=torch.bool, device=cache["k"].device
                ),
            })

    def initialize_online_patchification(
        self,
        *,
        candidate_scores: dict[int, float],
        last_scored_start: int,
        token_height: int,
        token_width: int,
        grid_rows: int,
        grid_cols: int,
        sink_frames: int,
        recent_frames: int,
        topk_blocks: int,
    ) -> None:
        self._patchification_state = {
            "candidate_scores": dict(candidate_scores),
            "last_scored_start": int(last_scored_start),
            "token_height": int(token_height),
            "token_width": int(token_width),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "sink_frames": int(sink_frames),
            "recent_frames": int(recent_frames),
            "topk_blocks": int(topk_blocks),
        }

    def _patchification_update(self, *, chunk_idx: int, phase: str) -> None:
        if phase != "generation" or self._patchification_state is None:
            return
        import torch
        import torch.distributed as dist

        state = self._patchification_state
        frame_tokens = int(self.geometry.frame_tokens)
        chunk_size = int(self.geometry.chunk_size)
        frames_seen = (int(chunk_idx) + 1) * chunk_size
        recent = int(state["recent_frames"])
        # A left frame becomes an eligible candidate only after its right
        # neighbor has aged out of the protected recent window.
        max_pair_start = frames_seen - recent - 2
        first_pair_start = int(state["last_scored_start"]) + 1
        if max_pair_start >= first_pair_start:
            reference = self.cache[1 if len(self.cache) > 1 else 0]
            live = _int(reference["local_end_index"])
            source_ids = reference["source_ids"][0, :live]
            frame_start = first_pair_start
            frame_end = max_pair_start + 1
            wanted = torch.arange(
                frame_start * frame_tokens,
                (frame_end + 1) * frame_tokens,
                device=source_ids.device,
                dtype=torch.long,
            )
            positions = torch.searchsorted(source_ids, wanted)
            if positions.numel() and (
                int(positions.max().item()) >= live
                or not torch.equal(source_ids.index_select(0, positions), wanted)
            ):
                raise RuntimeError("online Patchification needs complete keys for the newly eligible frame pairs")
            compute_device = reference.get("compute_device", reference["k"].device)
            frame_keys = reference["k"][0, :live].index_select(0, positions).to(compute_device).reshape(
                frame_end - frame_start + 1,
                frame_tokens,
                reference["k"].shape[2],
                reference["k"].shape[3],
            )
            moments = patchification_block_moments(
                frame_keys,
                sink_frames=0,
                recent_frames=1,
                token_height=int(state["token_height"]),
                token_width=int(state["token_width"]),
                grid_rows=int(state["grid_rows"]),
                grid_cols=int(state["grid_cols"]),
            )
            if dist.is_available() and dist.is_initialized():
                for moment in moments:
                    dist.all_reduce(moment, op=dist.ReduceOp.SUM)
            scores = forcingkv_scores_from_moments(*moments).detach().float().cpu()
            block_count = int(state["grid_rows"]) * int(state["grid_cols"])
            for pair_offset in range(int(scores.shape[0])):
                pair_start = frame_start + pair_offset
                for block_idx, value in enumerate(scores[pair_offset].tolist()):
                    state["candidate_scores"][pair_start * block_count + block_idx] = float(value)
            state["last_scored_start"] = max_pair_start

        candidates = state["candidate_scores"]
        selected_blocks = sorted(
            sorted(candidates, key=lambda key: (candidates[key], key))[:int(state["topk_blocks"])]
        )
        selected_ids = patchification_block_token_ids(
            selected_blocks,
            sink_frames=0,
            token_height=int(state["token_height"]),
            token_width=int(state["token_width"]),
            grid_rows=int(state["grid_rows"]),
            grid_cols=int(state["grid_cols"]),
        )
        sink = int(state["sink_frames"])
        # Keep one complete predecessor frame just before the protected
        # recent window. It becomes the left side of a newly eligible pair
        # after the next chunk advances the window.
        recent_start = max(sink, frames_seen - recent - 1)
        fixed_ids = set(range(sink * frame_tokens))
        fixed_ids.update(range(recent_start * frame_tokens, frames_seen * frame_tokens))
        wanted_ids = torch.tensor(sorted(fixed_ids | set(selected_ids)), dtype=torch.long, device=self.cache[0]["source_ids"].device)
        compacted = keep_global_source_ids(self.cache, wanted_ids)
        event = {
            "event": "patchification_online_update",
            "phase": phase,
            "chunk_idx": int(chunk_idx),
            "frames_seen": frames_seen,
            "new_pair_starts_scored": max(0, max_pair_start - first_pair_start + 1),
            "candidate_blocks": len(candidates),
            "selected_blocks": len(selected_blocks),
            "selected_tokens": int(wanted_ids.numel()),
            **compacted,
        }
        self.events.append(event)
        self.emit(event)

    def prepare_chunk(self, chunk_idx: int) -> None:
        if self.name == "dummy_forcing" and str(self.cache[0].get("cache_mode")) == "dummyforcing_warmup" and int(chunk_idx) == 2:
            for cache in self.cache:
                cache["dummyforcing_score_pending"] = True
            self.emit({
                "event": "dummyforcing_warmup_armed",
                "chunk_idx": int(chunk_idx),
                "cache_mode": str(self.cache[0].get("cache_mode")),
                "global_end": _int(self.cache[0]["global_end_index"]),
                "local_end": _int(self.cache[0]["local_end_index"]),
                "all_layers_armed": all(bool(cache.get("dummyforcing_score_pending")) for cache in self.cache),
            })

    def context_probe(self, *, chunk_idx: int, clean_x: Any, kwargs: dict[str, Any]) -> Any | None:
        """Score a clean context chunk with and without genuine predecessor KV."""
        if not self.is_consistency or int(chunk_idx) == 0 or self.matched_context_policy == "patchification_q0":
            return None
        import torch

        # Static-window compaction is destructive. Commit it before the cursor
        # snapshot so restoring the probe discards only its appended noisy tail.
        if self.name == "decoprune_hs":
            for layer in self.cache:
                compact_static_predecessor(layer)
        cursors = snapshot_cache_cursors(self.cache)
        if self._isolated_probe_cache is None:
            self._isolated_probe_cache = isolated_probe_cache(
                self.cache,
                capacity_tokens=int(self.geometry.tokens_per_chunk),
            )
        generator = torch.Generator(device=clean_x.device)
        generator.manual_seed(int(self.request.seed) + 104729 * (int(chunk_idx) + 1))
        noise = torch.randn(clean_x.shape, generator=generator, device=clean_x.device, dtype=clean_x.dtype)
        step = self.timesteps[self.context_step_index]
        noisy = self.pipeline.scheduler.add_noise(clean_x, noise, step)
        probe_kwargs = dict(kwargs)
        try:
            with_flow = self.pipeline.model(x=[noisy], t=step.reshape(1).to(clean_x.device), **probe_kwargs)[0]
            probe_kwargs["cross_attn_first_call"] = False
            with_x0 = self.pipeline._convert_flow_pred_to_x0(with_flow, noisy, step, self.pipeline.scheduler)
            with_scores = self._scores(with_x0, clean_x)
            # The with-KV call only appended into the live tail. Resetting its
            # cursors is enough because the following clean replay overwrites
            # that tail. The without-KV call must use isolated storage so it
            # cannot overwrite the retained history prefix.
            restore_cache_cursors(self.cache, cursors)
            empty_physical_cache_at(self._isolated_probe_cache, int(kwargs["current_start"]))
            probe_kwargs["kv_cache"] = self._isolated_probe_cache
            without_flow = self.pipeline.model(x=[noisy], t=step.reshape(1).to(clean_x.device), **probe_kwargs)[0]
            without_x0 = self.pipeline._convert_flow_pred_to_x0(without_flow, noisy, step, self.pipeline.scheduler)
            without_scores = self._scores(without_x0, clean_x)
            self.probe_calls += 2
            row = {
                "event": "context_consistency_probe",
                "chunk_idx": int(chunk_idx),
                "step_index": self.context_step_index,
                "timestep": float(step.item()),
                "score_type": self.score_type,
                "with_kv_mean": float(with_scores.mean().item()),
                "without_kv_mean": float(without_scores.mean().item()),
                "gap_without_minus_with": float((without_scores.mean() - with_scores.mean()).item()),
                "shared_noise": True,
            }
            self.events.append(row)
            self.emit(row)
            return with_scores.detach()
        finally:
            restore_cache_cursors(self.cache, cursors)

    def _scores(self, predicted: Any, target: Any) -> Any:
        from ..methods.consistency import token_scores

        return token_scores(predicted, target, score_type=self.score_type)

    def before_clean(self, chunk_idx: int, *, phase: str) -> None:
        if self.is_consistency and not (phase == "generation" and self._append_only_generation) and not (phase == "context" and self.matched_context_policy == "patchification_q0"):
            self._compact_consistency(next_chunk_idx=int(chunk_idx) + 1)

    def begin_generation(self) -> None:
        """Discard context-only consistency records after a shared q0 selection."""
        if self.matched_context_policy != "patchification_q0" or not self.is_consistency:
            return
        context_records = sum(record.source == "context" for record in self.records)
        self.records = [record for record in self.records if record.source != "context"]
        event = {
            "event": "matched_context_generation_started",
            "context_policy": self.matched_context_policy,
            "discarded_context_consistency_records": int(context_records),
        }
        self.events.append(event)
        self.emit(event)

    def capture_generation_prediction(self, step_index: int, x0: Any) -> None:
        if self.is_consistency and not self._append_only_generation and int(step_index) == self.generation_step_index:
            self._generation_prediction = x0.detach()

    def capture_model_state(self) -> None:
        """Persist transient attention evidence at the model-call boundary."""
        if self.name != "dummy_forcing" or str(self.cache[0].get("cache_mode")) != "dummyforcing_warmup":
            return
        self._dummy_capture_attempts += 1
        complete = all(
            bool(cache.get("dummyforcing_group_scores_valid", False).item())
            if hasattr(cache.get("dummyforcing_group_scores_valid"), "item")
            else "dummyforcing_group_scores" in cache
            for cache in self.cache
        )
        if self._dummy_capture_attempts == 3 or complete:
            self.emit({
                "event": "dummyforcing_warmup_capture_state",
                "capture_attempt": self._dummy_capture_attempts,
                "cache_mode": str(self.cache[0].get("cache_mode")),
                "global_end": _int(self.cache[0]["global_end_index"]),
                "local_end": _int(self.cache[0]["local_end_index"]),
                "first_layer_armed": bool(self.cache[0].get("dummyforcing_score_pending")),
                "layers_with_scores": sum(
                    bool(cache.get("dummyforcing_group_scores_valid", False).item())
                    if hasattr(cache.get("dummyforcing_group_scores_valid"), "item")
                    else "dummyforcing_group_scores" in cache
                    for cache in self.cache
                ),
            })
        if complete:
            import torch

            self._dummy_local_scores = torch.stack(
                [
                    cache.get("dummyforcing_group_scores_buffer", cache.get("dummyforcing_group_scores")).detach().clone()
                    for cache in self.cache
                ]
            )

    def after_clean(self, *, chunk_idx: int, clean_x: Any, context_scores: Any | None, phase: str) -> None:
        if self.is_consistency and not (phase == "generation" and self._append_only_generation):
            scores = context_scores
            if phase == "generation":
                if self._generation_prediction is None:
                    raise RuntimeError("generation consistency prediction was not captured")
                scores = self._scores(self._generation_prediction, clean_x).detach()
                self._generation_prediction = None
            if scores is not None:
                start = int(chunk_idx) * int(self.geometry.tokens_per_chunk)
                import torch

                ids = torch.arange(start, start + int(self.geometry.tokens_per_chunk), device=clean_x.device, dtype=torch.long)
                self.records.append(ConsistencyRecord(int(chunk_idx), ids, scores.detach(), phase))
        if phase == "generation" and self._append_only_generation:
            return
        if self.name == "forcingkv":
            self._forcingkv_compact(int(chunk_idx), phase=phase)
        elif self.name == "patchification" and bool(self.parameters.get("update_each_chunk", False)):
            self._patchification_update(chunk_idx=int(chunk_idx), phase=phase)
        elif self.name == "dummy_forcing":
            if str(self.cache[0].get("cache_mode")) == "dummyforcing_warmup" and int(chunk_idx) == 2:
                self._dummy_allocate()
            elif str(self.cache[0].get("cache_mode")) == "physical_dummyforcing":
                self._dummy_compact(int(chunk_idx), phase=phase)

    def _consistency_mask(self, record: ConsistencyRecord) -> Any:
        import torch

        scores = record.scores.detach().float().cpu().tolist()
        # The safety floor requires the real 2-D token grid. Canonical LingBot
        # is 30x52; otherwise choose the closest exact factorization.
        height = int(round(self.geometry.frame_tokens ** 0.5))
        while height > 1 and self.geometry.frame_tokens % height:
            height -= 1
        width = self.geometry.frame_tokens // height
        if self.geometry.frame_tokens == 1560:
            height, width = 30, 52
        reference = build_consistency_mask(
            scores,
            threshold=self.threshold,
            score_type=self.score_type,
            frames=int(self.geometry.chunk_size),
            height=height,
            width=width,
            min_keep_per_block=self.min_keep_per_block,
            floor_threshold=self.spatial_floor_threshold,
        )
        if self.name == "random":
            chosen = deterministic_random_indices(
                candidate_count=len(reference),
                keep_count=sum(reference),
                seed=int(self.request.seed) + 1000003 * (record.chunk_idx + 1),
            )
            mask = torch.zeros(len(reference), dtype=torch.bool, device=record.source_ids.device)
            if chosen:
                mask[torch.tensor(chosen, device=mask.device)] = True
            return mask
        return torch.tensor(reference, dtype=torch.bool, device=record.source_ids.device)

    def _compact_consistency(self, *, next_chunk_idx: int) -> None:
        # Preserve the most recent completed local window and the immediately
        # preceding scored chunk needed by the successor diagnostic.
        cutoff = int(next_chunk_idx) - 2 * int(self.local_window_chunks)
        prefix = self.consistency_prefix
        source_name = f"{prefix}source_ids" if prefix else "source_ids"
        end_name = f"{prefix}local_end_index" if prefix else "local_end_index"
        for record in sorted(self.records, key=lambda item: item.chunk_idx):
            if record.applied or record.chunk_idx >= cutoff:
                continue
            mask = self._consistency_mask(record)
            rejected = record.source_ids[~mask]
            reference = self.cache[0]
            live = _int(reference[end_name])
            physical = reference[source_name][0, :live]
            rejected = rejected.to(physical.device)
            wanted = physical[~__import__("torch").isin(physical, rejected)]
            before = live
            keep_global_source_ids(self.cache, wanted, prefix=prefix)
            after = _int(reference[end_name])
            record.applied = True
            event = {
                "event": "consistency_physical_compaction",
                "method": self.name,
                "phase": record.source,
                "chunk_idx": record.chunk_idx,
                "next_chunk_idx": int(next_chunk_idx),
                "score_type": self.score_type,
                "threshold": self.threshold,
                "kept_tokens": int(mask.sum().item()),
                "total_tokens": int(mask.numel()),
                "removed_tokens": before - after,
                "head_scope": "dynamic" if prefix else "all",
            }
            self.events.append(event)
            self.emit(event)

    def _forcingkv_compact(self, chunk_idx: int, *, phase: str) -> None:
        update = update_forcingkv_cache(
            cache=self.cache,
            geometry=self.geometry,
            parameters=self.parameters,
            chunk_idx=int(chunk_idx),
            phase=phase,
            previous_scores=self.forcing_scores,
            previous_patch_starts=self.forcing_patch_starts,
        )
        self.forcing_scores = list(update.scores)
        self.forcing_patch_starts = list(update.patch_starts)
        self.events.append(update.event)
        self.emit(update.event)

    def _dummy_allocate(self) -> None:
        import torch
        import torch.distributed as dist

        self.capture_model_state()
        if self._dummy_local_scores is None:
            missing = [
                index for index, cache in enumerate(self.cache)
                if not bool(cache["dummyforcing_group_scores_valid"].item())
            ]
            raise RuntimeError(f"DummyForcing did not capture complete warmup scores; missing layers={missing}")
        local = self._dummy_local_scores
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            global_scores = torch.cat(gathered, dim=1)
            rank = dist.get_rank()
        else:
            global_scores, rank = local, 0
        layers, heads = int(global_scores.shape[0]), int(global_scores.shape[1])
        assignment = dummyforcing_group_assignment(global_scores.detach().cpu().tolist(), last_group_count=round(layers * heads * 0.5))
        dense_end = _int(self.cache[0]["local_end_index"])
        frames_seen = dense_end // int(self.geometry.frame_tokens)
        banks = dummyforcing_context_banks(
            context_frames=frames_seen,
            frame_tokens=int(self.geometry.frame_tokens),
            chunk_size=int(self.geometry.chunk_size),
            first_frames=int(self.parameters["first_history_frames"]),
            middle_bank_strategy="center",
            middle_bank_frames=int(self.parameters["middle_bank_frames"]),
            middle_recent_frames=int(self.parameters["middle_recent_frames"]),
            last_frames=int(self.parameters["last_history_frames"]),
        )
        self._dummy_middle_bank_ids = tuple(
            token
            for frame_idx in banks["middle_bank_frame_indices"]
            for token in range(frame_idx * int(self.geometry.frame_tokens), (frame_idx + 1) * int(self.geometry.frame_tokens))
        )
        group_names = ("first", "middle", "last")
        counts = {name: 0 for name in group_names}
        local_heads = int(local.shape[1])
        for layer_idx, cache in enumerate(self.cache):
            source_ids = cache["source_ids"][0, :dense_end]
            for group_idx, name in enumerate(group_names):
                local_assignment = assignment[layer_idx][rank * local_heads:(rank + 1) * local_heads]
                head_indices = torch.tensor([i for i, value in enumerate(local_assignment) if value == group_idx], device=source_ids.device, dtype=torch.long)
                wanted = torch.tensor(banks[name], device=source_ids.device, dtype=torch.long)
                positions = torch.searchsorted(source_ids, wanted)
                capacity_frames = {
                    "first": int(self.parameters["first_history_frames"]),
                    "middle": int(self.parameters["middle_bank_frames"]) + int(self.parameters["middle_recent_frames"]),
                    "last": int(self.parameters["last_history_frames"]),
                }[name] + int(self.geometry.chunk_size)
                capacity = capacity_frames * int(self.geometry.frame_tokens)
                for suffix in ("k", "v"):
                    dense = cache[suffix][:, :dense_end].index_select(1, positions).index_select(2, head_indices)
                    target = torch.empty(1, capacity, int(head_indices.numel()), dense.shape[-1], device=dense.device, dtype=dense.dtype)
                    target[:, :wanted.numel()].copy_(dense)
                    cache[f"dummy_{name}_{suffix}"] = target
                for suffix in ("tpos", "source_ids"):
                    dense = cache[suffix][:, :dense_end].index_select(1, positions)
                    shape = list(dense.shape)
                    shape[1] = capacity
                    target = torch.full(shape, -1, device=dense.device, dtype=dense.dtype)
                    target[:, :wanted.numel()].copy_(dense)
                    cache[f"dummy_{name}_{suffix}"] = target
                cache[f"dummy_{name}_local_head_indices"] = head_indices
                cache[f"dummy_{name}_local_end_index"] = torch.tensor([wanted.numel()], device=source_ids.device, dtype=torch.long)
                counts[name] += int(head_indices.numel())
            cache["cache_mode"] = "physical_dummyforcing"
            cache["local_end_index"].fill_(max(len(banks[name]) for name in group_names))
            for name in ("k", "v", "tpos", "source_ids"):
                cache.pop(name, None)
            cache.pop("dummyforcing_group_scores", None)
            cache.pop("dummyforcing_group_scores_buffer", None)
            cache.pop("dummyforcing_group_scores_valid", None)
        event = {"event": "dummyforcing_head_group_allocation", "group_local_head_layer_counts": counts, "frames_seen": frames_seen, "middle_policy": "fixed_center_bank_plus_rolling_recent"}
        self.events.append(event)
        self.emit(event)

    def _dummy_compact(self, chunk_idx: int, *, phase: str) -> None:
        import torch

        frame = int(self.geometry.frame_tokens)
        frames_seen = (chunk_idx + 1) * int(self.geometry.chunk_size)
        first_frames = int(self.parameters["first_history_frames"])
        recent_frames = int(self.parameters["middle_recent_frames"])
        last_frames = int(self.parameters["last_history_frames"])
        first = tuple(range(0, min(frames_seen, first_frames) * frame))
        recent = tuple(range(max(0, frames_seen - recent_frames) * frame, frames_seen * frame))
        middle = tuple(sorted(set(self._dummy_middle_bank_ids) | set(recent)))
        last = tuple(range(max(0, frames_seen - last_frames) * frame, frames_seen * frame))
        banks = {"first": first, "middle": middle, "last": last}
        for name, wanted in banks.items():
            device = self.cache[0][f"dummy_{name}_source_ids"].device
            keep_global_source_ids(self.cache, torch.tensor(wanted, device=device, dtype=torch.long), prefix=f"dummy_{name}_")
        event = {"event": "dummyforcing_group_compaction", "phase": phase, "chunk_idx": chunk_idx, "group_tokens": {name: len(ids) for name, ids in banks.items()}}
        self.events.append(event)
        self.emit(event)

    def diagnostics(self) -> dict[str, Any]:
        unapplied = sum(not record.applied for record in self.records)
        return {
            "method": self.name,
            "events": list(self.events),
            "probe_model_calls": self.probe_calls,
            "consistency_records": len(self.records),
            "unapplied_consistency_records": unapplied,
            "forcingkv_selected_patches": len(self.forcing_patch_starts),
            "patchification_online_candidates": len((self._patchification_state or {}).get("candidate_scores", {})),
        }
