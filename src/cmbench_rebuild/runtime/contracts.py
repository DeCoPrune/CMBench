"""Narrow runtime contracts; tensor implementations remain backend-owned."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class CaseRequest:
    case_id: str
    seed: int
    prompt: str
    clip_file: Path
    camera: Mapping[str, Any]
    generation: Mapping[str, Any]
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedContext:
    payload: Any
    pixel_frames: int
    latent_frames: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class GeneratedCase:
    context_latents: Any
    continuation_latents: Any
    diagnostics: Mapping[str, Any]
    resource_metrics: Mapping[str, Any]


class ContextProvider(Protocol):
    def prepare(self, request: CaseRequest) -> PreparedContext: ...


class KVPolicy(Protocol):
    name: str
    parameters: Mapping[str, Any]


class ModelBackend(Protocol):
    def load(self, checkpoint: Path) -> None: ...
    def generate(self, request: CaseRequest, context: PreparedContext, policy: KVPolicy, rope_plan: Any, emit: Callable[[Mapping[str, Any]], None]) -> GeneratedCase: ...


class VideoWriter(Protocol):
    def write(self, generated: GeneratedCase, output_dir: Path) -> Mapping[str, Path]: ...


class Evaluator(Protocol):
    name: str
    def evaluate(self, request: CaseRequest, artifacts: Mapping[str, Path], output_dir: Path) -> Mapping[str, Any]: ...
