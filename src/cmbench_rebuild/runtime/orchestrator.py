"""One method-neutral experiment flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import CaseRequest, ContextProvider, Evaluator, KVPolicy, ModelBackend, VideoWriter


class ExperimentRunner:
    def __init__(self, *, backend: ModelBackend, context: ContextProvider, video: VideoWriter, evaluators: Iterable[Evaluator]) -> None:
        self.backend = backend
        self.context = context
        self.video = video
        self.evaluators = tuple(evaluators)

    def run(self, *, request: CaseRequest, checkpoint: Path, policy: KVPolicy, rope_plan: Any, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=False)
        events_path = output_dir / "events.jsonl"

        def emit(event: Mapping[str, Any]) -> None:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event), sort_keys=True) + "\n")

        emit({"event": "case_started", "case_id": request.case_id, "method": policy.name, "seed": request.seed})
        self.backend.load(checkpoint)
        prepared = self.context.prepare(request)
        emit({"event": "context_prepared", "pixel_frames": prepared.pixel_frames, "latent_frames": prepared.latent_frames})
        generated = self.backend.generate(request, prepared, policy, rope_plan, emit)
        artifacts = self.video.write(generated, output_dir)
        metrics = {evaluator.name: evaluator.evaluate(request, artifacts, output_dir) for evaluator in self.evaluators}
        result = {"schema_version": 1, "status": "completed", "case_id": request.case_id, "method": policy.name, "seed": request.seed, "context": dict(prepared.diagnostics), "generation": dict(generated.diagnostics), "resources": dict(generated.resource_metrics), "artifacts": {name: str(path) for name, path in artifacts.items()}, "metrics": metrics, "events": str(events_path)}
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit({"event": "case_completed", "case_id": request.case_id})
        return result
