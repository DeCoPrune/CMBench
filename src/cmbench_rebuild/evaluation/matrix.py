"""Resumable, provenance-bound evaluation and aggregation for a production matrix."""
from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts.manifest import sha256_file, write_immutable_manifest
from ..artifacts.video import inspect_video
from ..evidence import (
    artifact_qualified_generation_attempt,
    audit_mask_accounting,
    audit_seqpr,
    qualified_generation_attempt,
)
from ..identity import directory_content_identity
from ..gpu import parse_cuda_visible_devices, require_idle_cuda_devices
from ..matrix import validate_matrix_plan_inputs
from .cmbench import combine_dino_summaries, eligible_case_ids, summarize_dino_rows
from .official import resolve_reference_videos, run_official_dino, run_official_dino_batch


OFFICIAL_MATRIX_PLAN_SCHEMA_VERSION = 2


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_identity(path: Path, *, preserve_symlink: bool = False) -> dict[str, str]:
    # A venv's python is commonly a symlink.  Invoking its resolved base
    # interpreter can drop the venv package context, so retain that path while
    # still hashing the bytes reached through the link.
    resolved = path.absolute() if preserve_symlink else path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def evaluation_source_identity(source_root: Path) -> dict[str, Any]:
    """Hash every project module that can launch, qualify, or aggregate evaluation."""
    root = source_root.resolve()
    candidates = [
        *sorted((root / "src/cmbench_rebuild/evaluation").rglob("*.py")),
        root / "src/cmbench_rebuild/artifacts/manifest.py",
        root / "src/cmbench_rebuild/artifacts/video.py",
        root / "src/cmbench_rebuild/evidence.py",
        root / "src/cmbench_rebuild/dataset.py",
        root / "src/cmbench_rebuild/gpu.py",
        root / "src/cmbench_rebuild/identity.py",
    ]
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if not files:
        raise FileNotFoundError(f"evaluation source tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"evaluation source escapes root: {path}") from error
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "algorithm": "sha256-relative-path-null-content-digest-v1",
        "root": str(root),
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


def _model_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    config = resolved / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    return directory_content_identity(resolved)


def _validate_matrix_coverage(plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove the cartesian method/seed/case coverage, not merely its total size."""
    declared_methods = int(plan.get("methods", -1))
    declared_cases = int(plan.get("cases_per_method", -1))
    declared_seeds = tuple(int(seed) for seed in plan.get("seeds", ()))
    if declared_methods <= 0 or declared_cases <= 0:
        raise ValueError("matrix plan has invalid method/case declarations")
    if not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("matrix plan seeds must be non-empty and unique")
    methods = sorted({str(row["method"]) for row in entries})
    seeds = sorted({int(row["seed"]) for row in entries})
    if len(methods) != declared_methods:
        raise ValueError(f"matrix method coverage mismatch: {len(methods)} != {declared_methods}")
    if seeds != sorted(declared_seeds):
        raise ValueError(f"matrix seed coverage mismatch: {seeds} != {sorted(declared_seeds)}")
    expected_tasks = declared_methods * declared_cases * len(declared_seeds)
    if len(entries) != expected_tasks or int(plan.get("tasks", -1)) != expected_tasks:
        raise ValueError(f"matrix cartesian task count mismatch: {len(entries)} != {expected_tasks}")
    universes: dict[str, set[str]] = {}
    for method in methods:
        for seed in sorted(declared_seeds):
            key = f"{method}/seed-{seed}"
            rows = [row for row in entries if str(row["method"]) == method and int(row["seed"]) == seed]
            cases = [str(row["case_id"]) for row in rows]
            if len(cases) != declared_cases or len(set(cases)) != declared_cases:
                raise ValueError(f"matrix partition coverage mismatch: {key} has {len(set(cases))}/{len(cases)} cases")
            universes[key] = set(cases)
    reference_key = next(iter(universes))
    reference = universes[reference_key]
    for key, universe in universes.items():
        if universe != reference:
            missing = sorted(reference - universe)
            extra = sorted(universe - reference)
            raise ValueError(f"matrix case universe mismatch: {key} missing={missing[:3]} extra={extra[:3]}")
    return {
        "methods": methods,
        "seeds": sorted(declared_seeds),
        "cases_per_partition": declared_cases,
        "partitions": len(universes),
        "tasks": expected_tasks,
        "case_universe_sha256": hashlib.sha256(
            "\n".join(sorted(reference)).encode("utf-8")
        ).hexdigest(),
    }


def _evaluation_dataset_identity(
    *, benchmark_root: Path, metadata: Path, annotations: Path, expected_case_ids: Sequence[str]
) -> dict[str, Any]:
    expected = tuple(sorted(str(case_id) for case_id in expected_case_ids))
    eligible = tuple(sorted(eligible_case_ids(annotations, expected)))
    if eligible != expected:
        missing = sorted(set(expected) - set(eligible))
        raise ValueError(f"official annotations ineligible for planned cases: {missing[:5]}")
    metadata_ids: set[str] = set()
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{metadata}:{line_number}: expected object")
        for key in ("task_id", "case_id", "source_case_id"):
            value = str(row.get(key) or "")
            if value:
                metadata_ids.add(value)
    missing_metadata = sorted(set(expected) - metadata_ids)
    if missing_metadata:
        raise ValueError(f"official metadata missing planned cases: {missing_metadata[:5]}")
    reference_paths = resolve_reference_videos(
        benchmark_root=benchmark_root,
        metadata=metadata,
        annotations=annotations,
        case_ids=expected,
    )
    reference_videos = {
        case_id: {
            "path": str(reference_paths[case_id]),
            "bytes": reference_paths[case_id].stat().st_size,
            "sha256": sha256_file(reference_paths[case_id]),
        }
        for case_id in expected
    }
    digest = hashlib.sha256("\n".join(expected).encode("utf-8")).hexdigest()
    return {
        "planned_cases": len(expected),
        "eligible_annotation_cases": len(eligible),
        "metadata_covered_cases": len(expected),
        "case_universe_sha256": digest,
        "reference_videos": reference_videos,
    }


def build_official_matrix_plan(
    generation_plan_path: Path,
    *,
    benchmark_root: Path,
    metadata: Path,
    annotations: Path,
    evaluator: Path,
    python: Path,
    owl_model: Path,
    sam_model: Path,
    dino_model: Path,
    eval_root: Path,
    cuda_visible_devices: str,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Bind every generation task to one official OWL/SAM/DINO evaluation task."""
    generation_plan_path = generation_plan_path.resolve()
    parse_cuda_visible_devices(cuda_visible_devices, expected_count=1)
    if str(device) not in {"cuda", "cuda:0"}:
        raise ValueError("official matrix must address its one visible GPU as cuda:0")
    generation = _load_json(generation_plan_path)
    production_source = validate_matrix_plan_inputs(generation)
    evaluation_source = evaluation_source_identity(Path(production_source["root"]))
    entries = generation.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("generation plan has no entries")
    declared_tasks = int(generation.get("tasks", -1))
    if declared_tasks != len(entries):
        raise ValueError(f"generation plan tasks mismatch: {declared_tasks} != {len(entries)}")
    coverage = _validate_matrix_coverage(generation, entries)
    expected_cases = sorted({str(row["case_id"]) for row in entries})
    benchmark = benchmark_root.resolve()
    if not benchmark.is_dir():
        raise FileNotFoundError(benchmark)
    inputs = {
        "benchmark_root": str(benchmark),
        "metadata": _file_identity(metadata),
        "annotations": _file_identity(annotations),
        "evaluator": _file_identity(evaluator),
        "python": _file_identity(python, preserve_symlink=True),
        "models": {
            "owl": _model_identity(owl_model),
            "sam": _model_identity(sam_model),
            "dino": _model_identity(dino_model),
        },
        "device": str(device),
        "cuda_visible_devices": str(cuda_visible_devices),
    }
    evaluation_dataset = _evaluation_dataset_identity(
        benchmark_root=benchmark,
        metadata=Path(inputs["metadata"]["path"]),
        annotations=Path(inputs["annotations"]["path"]),
        expected_case_ids=expected_cases,
    )
    root = eval_root.resolve()
    planned: list[dict[str, Any]] = []
    identities: set[tuple[str, int, str]] = set()
    for row in entries:
        method = str(row["method"])
        seed = int(row["seed"])
        case_id = str(row["case_id"])
        identity = (method, seed, case_id)
        if identity in identities:
            raise ValueError(f"duplicate generation task: {identity}")
        identities.add(identity)
        planned.append({
            "method": method,
            "seed": seed,
            "case_id": case_id,
            "generation_case_root": str(Path(row["case_root"]).resolve()),
            "evaluation_case_root": str((root / method / f"seed-{seed}" / case_id).resolve()),
            "reference_video": dict(evaluation_dataset["reference_videos"][case_id]),
            "production_source_sha256": production_source["sha256"],
            "evaluation_source_sha256": evaluation_source["sha256"],
            "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "input_video_path": str(row["input_video_path"]),
            "input_video_bytes": int(row["input_video_bytes"]),
            "input_video_sha256": str(row["input_video_sha256"]),
            "head_map_path": row.get("head_map_path"),
            "head_map_bytes": row.get("head_map_bytes"),
            "head_map_sha256": row.get("head_map_sha256"),
        })
    return {
        "schema_version": OFFICIAL_MATRIX_PLAN_SCHEMA_VERSION,
        "protocol": generation.get("protocol"),
        "generation_plan": {"path": str(generation_plan_path), "sha256": sha256_file(generation_plan_path)},
        "production_source": production_source,
        "evaluation_source": evaluation_source,
        "evaluation_root": str(root),
        "tasks": len(planned),
        "methods": int(generation.get("methods", len({row["method"] for row in planned}))),
        "cases_per_method": int(generation.get("cases_per_method", 0)),
        "seeds": list(generation.get("seeds", sorted({row["seed"] for row in planned}))),
        "inputs": inputs,
        "coverage": coverage,
        "evaluation_dataset": evaluation_dataset,
        "entries": planned,
    }


def validate_official_matrix_plan_inputs(plan: Mapping[str, Any]) -> None:
    """Fail closed if any immutable evaluator input changed after planning."""
    if plan.get("schema_version") != OFFICIAL_MATRIX_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported official evaluation plan schema: {plan.get('schema_version')!r}"
        )
    inputs = plan["inputs"]
    parse_cuda_visible_devices(str(inputs.get("cuda_visible_devices") or ""), expected_count=1)
    if str(inputs.get("device")) not in {"cuda", "cuda:0"}:
        raise ValueError("official matrix must address its one visible GPU as cuda:0")
    identities = [plan["generation_plan"], inputs["metadata"], inputs["annotations"], inputs["evaluator"], inputs["python"]]
    for identity in identities:
        path = Path(identity["path"])
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"official evaluation plan input changed: {path}")
    generation = _load_json(Path(plan["generation_plan"]["path"]))
    production_source = validate_matrix_plan_inputs(generation)
    if dict(plan.get("production_source") or {}) != production_source:
        raise ValueError("official evaluation plan production source identity changed")
    evaluation_source = evaluation_source_identity(Path(production_source["root"]))
    if dict(plan.get("evaluation_source") or {}) != evaluation_source:
        raise ValueError("official evaluation plan evaluation source identity changed")
    if not Path(inputs["benchmark_root"]).is_dir():
        raise ValueError(f"benchmark root missing: {inputs['benchmark_root']}")
    for name in ("owl", "sam", "dino"):
        identity = inputs["models"][name]
        if directory_content_identity(Path(identity["path"])) != identity:
            raise ValueError(f"official evaluation model identity changed: {name}")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != int(plan.get("tasks", -1)):
        raise ValueError("official evaluation plan task count changed")
    if any(
        row.get("production_source_sha256") != production_source["sha256"]
        for row in entries
    ):
        raise ValueError("official evaluation plan entries do not share the production source identity")
    if any(
        row.get("evaluation_source_sha256") != evaluation_source["sha256"]
        for row in entries
    ):
        raise ValueError("official evaluation plan entries do not share the evaluation source identity")
    coverage = _validate_matrix_coverage(plan, entries)
    if dict(plan.get("coverage") or {}) != coverage:
        raise ValueError("official evaluation plan coverage identity changed")
    evaluation_dataset = _evaluation_dataset_identity(
        benchmark_root=Path(inputs["benchmark_root"]),
        metadata=Path(inputs["metadata"]["path"]),
        annotations=Path(inputs["annotations"]["path"]),
        expected_case_ids=sorted({str(row["case_id"]) for row in entries}),
    )
    if dict(plan.get("evaluation_dataset") or {}) != evaluation_dataset:
        raise ValueError("official evaluation dataset identity changed")
    generation_entries = generation.get("entries") or []
    generation_by_identity = {
        (str(row["method"]), int(row["seed"]), str(row["case_id"])): row
        for row in generation_entries
    }
    official_by_identity: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in entries:
        identity = (str(row["method"]), int(row["seed"]), str(row["case_id"]))
        if identity in official_by_identity:
            raise ValueError(f"official evaluation plan contains duplicate entry: {identity}")
        official_by_identity[identity] = row
    if set(official_by_identity) != set(generation_by_identity):
        raise ValueError("official evaluation entries differ from the generation plan identities")
    evaluation_root_value = plan.get("evaluation_root")
    if not isinstance(evaluation_root_value, str) or not evaluation_root_value:
        raise ValueError("official evaluation plan lacks an evaluation root")
    evaluation_root = Path(evaluation_root_value).resolve()
    for identity, row in official_by_identity.items():
        method, seed, case_id = identity
        generation_row = generation_by_identity[identity]
        expected_evaluation_root = (evaluation_root / method / f"seed-{seed}" / case_id).resolve()
        if (
            Path(str(row.get("generation_case_root"))).resolve()
            != Path(str(generation_row["case_root"])).resolve()
            or Path(str(row.get("evaluation_case_root"))).resolve() != expected_evaluation_root
            or dict(row.get("reference_video") or {})
            != dict(evaluation_dataset["reference_videos"][case_id])
            or Path(str(row.get("checkpoint_path") or "")).resolve()
            != Path(str(generation_row.get("checkpoint_path") or "")).resolve()
            or str(row.get("checkpoint_sha256"))
            != str(generation_row.get("checkpoint_sha256"))
            or Path(str(row.get("input_video_path"))).resolve()
            != Path(str(generation_row.get("input_video_path"))).resolve()
            or int(row.get("input_video_bytes", -1))
            != int(generation_row.get("input_video_bytes", -1))
            or str(row.get("input_video_sha256"))
            != str(generation_row.get("input_video_sha256"))
            or row.get("head_map_path") != generation_row.get("head_map_path")
            or row.get("head_map_bytes") != generation_row.get("head_map_bytes")
            or row.get("head_map_sha256") != generation_row.get("head_map_sha256")
        ):
            raise ValueError(f"official evaluation entry provenance changed: {identity}")


def _generation_evidence_context(generation: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the complete identities that every admitted attempt must repeat."""
    checkpoints = {
        str(Path(str(identity["path"])).resolve()): dict(identity)
        for identity in generation["checkpoints"]
    }
    input_videos = {
        str(Path(str(identity["path"])).resolve()): dict(identity)
        for identity in generation["input_videos"]
    }
    if len(checkpoints) != len(generation["checkpoints"]):
        raise ValueError("generation checkpoint identities are duplicated")
    if len(input_videos) != len(generation["input_videos"]):
        raise ValueError("generation input video identities are duplicated")
    return {
        "production_source": dict(generation["production_source"]),
        "checkpoints": checkpoints,
        "input_videos": input_videos,
    }


def _qualification_identities(
    row: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_path = str(Path(str(row["checkpoint_path"])).resolve())
    input_video_path = str(Path(str(row["input_video_path"])).resolve())
    checkpoint = dict(context["checkpoints"][checkpoint_path])
    input_video = dict(context["input_videos"][input_video_path])
    if str(checkpoint["sha256"]) != str(row["checkpoint_sha256"]):
        raise ValueError("generation row checkpoint identity differs from plan")
    if (
        int(input_video["bytes"]) != int(row["input_video_bytes"])
        or str(input_video["sha256"]) != str(row["input_video_sha256"])
    ):
        raise ValueError("generation row input video identity differs from plan")
    return dict(context["production_source"]), checkpoint, input_video


def _qualified_generation_attempt(
    row: Mapping[str, Any], context: Mapping[str, Any]
) -> Path | None:
    production_source, checkpoint, input_video = _qualification_identities(row, context)
    return qualified_generation_attempt(
        row,
        production_source=production_source,
        checkpoint=checkpoint,
        input_video=input_video,
    )


def _artifact_qualified_generation_attempt(
    row: Mapping[str, Any], context: Mapping[str, Any]
) -> Path | None:
    production_source, checkpoint, input_video = _qualification_identities(row, context)
    return artifact_qualified_generation_attempt(
        row,
        production_source=production_source,
        checkpoint=checkpoint,
        input_video=input_video,
    )


def _provenance_matches(
    path: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    generation_attempt: Path | None = None,
) -> bool:
    try:
        value = _load_json(path)
        generation = generation_attempt
        if generation is None:
            return False
        video = generation / "continuation.mp4"
        inputs = plan["inputs"]
        score = float(value["dino_score"])
        decoded_frames = int(inspect_video(video)["decoded_frame_count"])
        reference = Path(row["reference_video"]["path"])
        return (
            value.get("schema_version") == 1
            and value.get("official_compatible") is True
            and value.get("scoring_path") == "owl_sam_dino"
            and value.get("mask_mode") == "sam"
            and int(value.get("frame_stride", -1)) == 1
            and str(value.get("case_id")) == str(row["case_id"])
            and str(value.get("method")) == str(row["method"])
            and dict(value.get("evaluation_source") or {}) == dict(plan["evaluation_source"])
            and math.isfinite(score)
            and int(value.get("num_eval_frames", 0)) == decoded_frames
            and value["generated_video"]["sha256"] == sha256_file(video)
            and dict(value["reference_video"]) == dict(row["reference_video"])
            and reference.is_file()
            and reference.stat().st_size == int(row["reference_video"]["bytes"])
            and sha256_file(reference) == row["reference_video"]["sha256"]
            and value["benchmark"]["metadata"]["sha256"] == inputs["metadata"]["sha256"]
            and value["benchmark"]["annotations"]["sha256"] == inputs["annotations"]["sha256"]
            and value["evaluator"]["sha256"] == inputs["evaluator"]["sha256"]
            and str(value.get("cuda_visible_devices")) == str(inputs["cuda_visible_devices"])
            and all(dict(value["models"][name]) == dict(inputs["models"][name]) for name in ("owl", "sam", "dino"))
            and isinstance(value.get("summary"), dict)
            and str(value["summary"].get("selected_case_id")) == str(row["case_id"])
            and value["summary"].get("scoring_path") == "owl_sam_dino"
            and int(value["summary"].get("num_eval_frames", -1)) == int(value["num_eval_frames"])
            and isinstance(value["summary"].get("references"), list)
            and bool(value["summary"]["references"])
            and not any(bool(reference.get("used_direct_dino")) for reference in value["summary"]["references"])
            and Path(value["summary_jsonl"]["path"]).is_file()
            and sha256_file(Path(value["summary_jsonl"]["path"])) == value["summary_jsonl"]["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _successful_evaluation_attempt(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    generation_attempt: Path | None = None,
) -> Path | None:
    root = Path(str(row["evaluation_case_root"]))
    for attempt in sorted(path for path in root.glob("attempt-*") if path.is_dir()):
        if _provenance_matches(
            attempt / "official_dino.provenance.json", row, plan, generation_attempt
        ):
            return attempt
    return None


def _next_attempt(root: Path) -> Path:
    numbers: list[int] = []
    for path in root.glob("attempt-*"):
        if not path.is_dir():
            continue
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except ValueError:
            continue
    return root / f"attempt-{max(numbers, default=0) + 1:03d}"


def _selected_entries(
    plan: Mapping[str, Any],
    *,
    methods: Iterable[str] | None,
    case_ids: Iterable[str] | None,
    seeds: Iterable[int] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    allowed_methods = set(methods or ())
    allowed_cases = set(case_ids or ())
    allowed_seeds = {int(seed) for seed in (seeds or ())}
    entries = list(plan["entries"])
    known_methods = {str(row["method"]) for row in entries}
    known_cases = {str(row["case_id"]) for row in entries}
    known_seeds = {int(row["seed"]) for row in entries}
    unknown_methods = sorted(allowed_methods - known_methods)
    unknown_cases = sorted(allowed_cases - known_cases)
    unknown_seeds = sorted(allowed_seeds - known_seeds)
    if unknown_methods:
        raise ValueError(f"unknown official matrix methods: {unknown_methods}")
    if unknown_cases:
        raise ValueError(f"unknown official matrix case IDs: {unknown_cases}")
    if unknown_seeds:
        raise ValueError(f"unknown official matrix seeds: {unknown_seeds}")
    selected = [
        dict(row) for row in entries
        if (not allowed_methods or row["method"] in allowed_methods)
        and (not allowed_cases or row["case_id"] in allowed_cases)
        and (not allowed_seeds or int(row["seed"]) in allowed_seeds)
    ]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("official matrix selection is empty")
    return selected


def run_official_matrix(
    plan: Mapping[str, Any],
    *,
    methods: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Evaluate selected generated cases, preserving attempts and skipping verified successes."""
    validate_official_matrix_plan_inputs(plan)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    selected = _selected_entries(plan, methods=methods, case_ids=case_ids, seeds=seeds, limit=limit)
    generation_plan = _load_json(Path(plan["generation_plan"]["path"]))
    evidence_context = _generation_evidence_context(generation_plan)
    generation_attempts = {
        (row["method"], int(row["seed"]), row["case_id"]): _qualified_generation_attempt(
            row, evidence_context
        )
        for row in selected
    }
    missing = [
        {key: row[key] for key in ("method", "seed", "case_id")}
        for row in selected
        if generation_attempts[(row["method"], int(row["seed"]), row["case_id"])] is None
    ]
    if missing:
        raise ValueError(f"generation matrix incomplete for {len(missing)} selected tasks; first={missing[0]}")
    inputs = plan["inputs"]
    completed = skipped = failed = 0
    failures: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for row in selected:
        generation_attempt = generation_attempts[(row["method"], int(row["seed"]), row["case_id"])]
        assert generation_attempt is not None
        if _successful_evaluation_attempt(row, plan, generation_attempt) is not None:
            skipped += 1
            continue
        pending.append((row, generation_attempt))

    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], Path]]] = {}
    for row, generation_attempt in pending:
        grouped.setdefault((row["method"], int(row["seed"])), []).append((row, generation_attempt))
    stop = False
    for (method, seed), group in grouped.items():
        for offset in range(0, len(group), int(batch_size)):
            chunk = group[offset : offset + int(batch_size)]
            require_idle_cuda_devices(
                str(inputs["cuda_visible_devices"]), expected_count=1
            )
            attempts: dict[str, Path] = {}
            for row, _ in chunk:
                evaluation_root = Path(row["evaluation_case_root"])
                evaluation_root.mkdir(parents=True, exist_ok=True)
                attempt = _next_attempt(evaluation_root)
                attempt.mkdir(parents=True, exist_ok=False)
                attempts[row["case_id"]] = attempt
            batch_root: Path | None = None
            if len(chunk) > 1:
                batch_parent = Path(plan["evaluation_root"]) / "_batches" / method / f"seed-{seed}"
                batch_parent.mkdir(parents=True, exist_ok=True)
                batch_root = _next_attempt(batch_parent)
                batch_root.mkdir(parents=True, exist_ok=False)
            try:
                common = {
                    "benchmark_root": Path(inputs["benchmark_root"]),
                    "metadata": Path(inputs["metadata"]["path"]),
                    "annotations": Path(inputs["annotations"]["path"]),
                    "evaluator": Path(inputs["evaluator"]["path"]),
                    "python": Path(inputs["python"]["path"]),
                    "owl_model": Path(inputs["models"]["owl"]["path"]),
                    "sam_model": Path(inputs["models"]["sam"]["path"]),
                    "dino_model": Path(inputs["models"]["dino"]["path"]),
                    "device": str(inputs["device"]),
                    "cuda_visible_devices": str(inputs["cuda_visible_devices"]),
                    "evaluation_source": dict(plan["evaluation_source"]),
                }
                if len(chunk) == 1:
                    row, generation_attempt = chunk[0]
                    result = run_official_dino(
                        case_dir=generation_attempt,
                        eval_root=attempts[row["case_id"]],
                        **common,
                    )
                    results = {row["case_id"]: result}
                else:
                    assert batch_root is not None
                    results = run_official_dino_batch(
                        case_dirs=[generation_attempt for _, generation_attempt in chunk],
                        eval_root=batch_root,
                        **common,
                    )
                    for row, _ in chunk:
                        write_immutable_manifest(
                            attempts[row["case_id"]] / "official_dino.provenance.json",
                            results[row["case_id"]],
                        )
                for row, generation_attempt in chunk:
                    provenance = attempts[row["case_id"]] / "official_dino.provenance.json"
                    if not _provenance_matches(provenance, row, plan, generation_attempt):
                        raise ValueError("official evaluation provenance does not match matrix plan")
                completed += len(chunk)
            except Exception as error:  # preserve a resumable, inspectable failed attempt
                failed += len(chunk)
                for row, _ in chunk:
                    failure = {
                        "schema_version": 1,
                        "method": row["method"],
                        "seed": row["seed"],
                        "case_id": row["case_id"],
                        "attempt": str(attempts[row["case_id"]]),
                        "batch_attempt": str(batch_root) if batch_root else None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    failure_path = attempts[row["case_id"]] / "failure.json"
                    if not failure_path.exists():
                        write_immutable_manifest(failure_path, failure)
                    failures.append(failure)
                if not continue_on_error:
                    stop = True
                    break
        if stop:
            break
    return {
        "schema_version": 1,
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dino = summarize_dino_rows(
        ({"case_id": row["case_id"], "dino_score": row["dino_score"], "memory_level": row["memory_level"]} for row in rows),
        [row["case_id"] for row in rows],
    )
    active = sum(int(row["active_units"]) for row in rows)
    dense = sum(int(row["dense_units"]) for row in rows)
    elapsed = sum(float(row["autoregressive_seconds"]) for row in rows)
    frames = sum(int(row["decoded_frames"]) for row in rows)
    return {
        **dino,
        "seqPR": 1.0 - active / dense,
        "seqPR_aggregation": "ratio_of_generation_chunk_totals_across_cases",
        "active_token_head_layer_units": active,
        "dense_token_head_layer_units": dense,
        "decoded_frames": frames,
        "autoregressive_generation_seconds": elapsed,
        "decoded_frames_per_second": frames / elapsed,
        "peak_allocated_bytes_max": max(int(row["peak_allocated_bytes"]) for row in rows),
    }


def aggregate_official_matrix(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly audit and aggregate a complete official evaluation matrix."""
    validate_official_matrix_plan_inputs(plan)
    generation_plan = _load_json(Path(plan["generation_plan"]["path"]))
    evidence_context = _generation_evidence_context(generation_plan)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in plan["entries"]:
        generation = _qualified_generation_attempt(row, evidence_context)
        evaluation = _successful_evaluation_attempt(row, plan, generation)
        if generation is None or evaluation is None:
            missing.append({key: row[key] for key in ("method", "seed", "case_id")})
            continue
        seqpr_path = generation / "seq_pr_metrics.json"
        trajectory_path = generation / "token_trajectory.jsonl"
        seqpr_audit = audit_seqpr(seqpr_path)
        accounting_audit = audit_mask_accounting(trajectory_path)
        if not seqpr_audit["passed"] or not accounting_audit["passed"]:
            raise ValueError(f"independent accounting audit failed: {generation}")
        seqpr = _load_json(seqpr_path)
        result = _load_json(generation / "result.json")
        provenance = _load_json(evaluation / "official_dino.provenance.json")
        resources = result["resources"]
        records.append({
            "method": row["method"],
            "seed": int(row["seed"]),
            "case_id": row["case_id"],
            "memory_level": provenance.get("memory_level"),
            "dino_score": float(provenance["dino_score"]),
            "decoded_frames": int(provenance["num_eval_frames"]),
            "active_units": int(seqpr["active_token_head_layer_units"]),
            "dense_units": int(seqpr["dense_token_head_layer_units"]),
            "autoregressive_seconds": float(resources["autoregressive_generation_seconds"]),
            "peak_allocated_bytes": int(resources["peak_allocated_bytes"]),
            "evidence": {
                "generation_attempt": str(generation),
                "evaluation_attempt": str(evaluation),
                "result_sha256": sha256_file(generation / "result.json"),
                "artifact_manifest_sha256": sha256_file(generation / "artifacts.manifest.json"),
                "seqpr_sha256": sha256_file(seqpr_path),
                "token_trajectory_sha256": sha256_file(trajectory_path),
                "official_provenance_sha256": sha256_file(evaluation / "official_dino.provenance.json"),
                "generated_video_sha256": provenance["generated_video"]["sha256"],
                "reference_video_sha256": provenance["reference_video"]["sha256"],
                "input_video_sha256": row["input_video_sha256"],
                "head_map_sha256": row.get("head_map_sha256"),
            },
        })
    if missing:
        raise ValueError(f"official matrix incomplete: missing {len(missing)} tasks; first={missing[0]}")
    if len(records) != int(plan["tasks"]):
        raise ValueError(f"audited task count mismatch: {len(records)} != {plan['tasks']}")

    by_method_seed: dict[str, dict[str, Any]] = {}
    partitions: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        partitions.setdefault(f"{record['method']}/seed-{record['seed']}", []).append(record)
    for key, values in sorted(partitions.items()):
        by_method_seed[key] = _aggregate_rows(values)

    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in records}):
        method_rows = [row for row in records if row["method"] == method]
        seed_groups = [values for key, values in partitions.items() if key.startswith(f"{method}/seed-")]
        dino = combine_dino_summaries([_aggregate_rows(values) for values in seed_groups])
        active = sum(int(row["active_units"]) for row in method_rows)
        dense = sum(int(row["dense_units"]) for row in method_rows)
        elapsed = sum(float(row["autoregressive_seconds"]) for row in method_rows)
        frames = sum(int(row["decoded_frames"]) for row in method_rows)
        by_method[method] = {
            **dino,
            "seeds": len(seed_groups),
            "seqPR": 1.0 - active / dense,
            "seqPR_aggregation": "ratio_of_generation_chunk_totals_across_cases_and_seeds",
            "active_token_head_layer_units": active,
            "dense_token_head_layer_units": dense,
            "decoded_frames": frames,
            "autoregressive_generation_seconds": elapsed,
            "decoded_frames_per_second": frames / elapsed,
            "peak_allocated_bytes_max": max(int(row["peak_allocated_bytes"]) for row in method_rows),
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "official_compatible": True,
        "scoring_path": "owl_sam_dino",
        "tasks": len(records),
        "methods": len(by_method),
        "seeds": sorted({row["seed"] for row in records}),
        "generation_plan": dict(plan["generation_plan"]),
        "production_source": dict(plan["production_source"]),
        "evaluation_source": dict(plan["evaluation_source"]),
        "inputs": dict(plan["inputs"]),
        "coverage": dict(plan["coverage"]),
        "evaluation_dataset": dict(plan["evaluation_dataset"]),
        "aggregator": {
            "module": __name__,
            "source": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "case_records": sorted(records, key=lambda row: (row["method"], row["seed"], row["case_id"])),
        "by_method_seed": by_method_seed,
        "by_method": by_method,
    }


def bind_aggregate_to_official_plan(aggregate: Mapping[str, Any], plan_path: Path) -> dict[str, Any]:
    """Bind a completed aggregate to the exact official-plan bytes used to compute it."""
    resolved = plan_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    result = dict(aggregate)
    result["official_evaluation_plan"] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    return result


def audit_generation_matrix(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Audit current generation progress without treating result-file presence as success."""
    validate_matrix_plan_inputs(dict(plan))
    evidence_context = _generation_evidence_context(plan)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("generation plan has no entries")
    if len(entries) != int(plan.get("tasks", -1)):
        raise ValueError("generation plan task count mismatch")
    coverage = _validate_matrix_coverage(plan, entries)
    identities: set[tuple[str, int, str]] = set()
    partitions: dict[str, dict[str, int]] = {}
    audited = artifact_qualified = invalid_completed = attempts = nonqualified_attempts = 0
    artifact_bytes = 0
    accounting_failures: list[dict[str, Any]] = []
    for source_row in entries:
        row = {
            "method": str(source_row["method"]),
            "seed": int(source_row["seed"]),
            "case_id": str(source_row["case_id"]),
            "generation_case_root": str(source_row["case_root"]),
            "production_source_sha256": str(source_row["production_source_sha256"]),
            "checkpoint_path": str(source_row["checkpoint_path"]),
            "checkpoint_sha256": str(source_row["checkpoint_sha256"]),
            "input_video_path": str(source_row["input_video_path"]),
            "input_video_bytes": int(source_row["input_video_bytes"]),
            "input_video_sha256": str(source_row["input_video_sha256"]),
            "head_map_path": source_row.get("head_map_path"),
            "head_map_bytes": source_row.get("head_map_bytes"),
            "head_map_sha256": source_row.get("head_map_sha256"),
        }
        identity = (row["method"], row["seed"], row["case_id"])
        if identity in identities:
            raise ValueError(f"duplicate generation task: {identity}")
        identities.add(identity)
        key = f"{row['method']}/seed-{row['seed']}"
        partition = partitions.setdefault(key, {"tasks": 0, "audited_successes": 0, "invalid_completed": 0})
        partition["tasks"] += 1
        case_root = Path(row["generation_case_root"])
        task_attempts = sorted(path for path in case_root.glob("attempt-*") if path.is_dir())
        attempts += len(task_attempts)
        artifact_generation = _artifact_qualified_generation_attempt(row, evidence_context)
        generation = _qualified_generation_attempt(row, evidence_context)
        nonqualified_attempts += len(task_attempts) - int(generation is not None)
        if artifact_generation is None:
            has_completed = False
            for attempt in task_attempts:
                result_path = attempt / "result.json"
                try:
                    has_completed = has_completed or _load_json(result_path).get("status") == "completed"
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            if has_completed:
                invalid_completed += 1
                partition["invalid_completed"] += 1
            continue
        artifact_qualified += 1
        selected_generation = generation or artifact_generation
        manifest = _load_json(selected_generation / "artifacts.manifest.json")
        artifact_bytes += sum(int(identity["bytes"]) for identity in manifest["files"].values())
        seqpr = audit_seqpr(selected_generation / "seq_pr_metrics.json")
        accounting = audit_mask_accounting(selected_generation / "token_trajectory.jsonl")
        accounting_errors = list(accounting["errors"])
        if int(accounting.get("per_layer_events", 0)) <= 0:
            accounting_errors.append("token trajectory contains no per-layer events")
        if generation is None or not seqpr["passed"] or not accounting["passed"] or accounting_errors:
            accounting_failures.append({
                "method": row["method"],
                "seed": row["seed"],
                "case_id": row["case_id"],
                "attempt": str(selected_generation),
                "seqpr_errors": seqpr["errors"],
                "accounting_errors": accounting_errors,
            })
            continue
        audited += 1
        partition["audited_successes"] += 1
    task_count = len(entries)
    complete = audited == task_count and not accounting_failures and invalid_completed == 0
    status = "complete" if complete else "invalid" if accounting_failures or invalid_completed else "in_progress"
    return {
        "schema_version": 1,
        "status": status,
        "complete": complete,
        "tasks": task_count,
        "audited_successes": audited,
        "artifact_qualified": artifact_qualified,
        "invalid_completed_tasks": invalid_completed,
        "unresolved_tasks": task_count - artifact_qualified,
        "attempts": attempts,
        "nonqualified_attempts": nonqualified_attempts,
        "artifact_bytes": artifact_bytes,
        "accounting_failures": accounting_failures,
        "coverage": coverage,
        "partitions": dict(sorted(partitions.items())),
    }


def audit_official_matrix_progress(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Audit provenance-bound official-evaluation progress without aggregating early."""
    validate_official_matrix_plan_inputs(plan)
    generation_plan = _load_json(Path(plan["generation_plan"]["path"]))
    evidence_context = _generation_evidence_context(generation_plan)
    evaluated = generation_ready = invalid_attempts = 0
    partitions: dict[str, dict[str, int]] = {}
    for row in plan["entries"]:
        key = f"{row['method']}/seed-{row['seed']}"
        partition = partitions.setdefault(key, {"tasks": 0, "generation_ready": 0, "evaluated": 0})
        partition["tasks"] += 1
        generation = _qualified_generation_attempt(row, evidence_context)
        if generation is None:
            continue
        generation_ready += 1
        partition["generation_ready"] += 1
        evaluation_root = Path(row["evaluation_case_root"])
        attempts = sorted(path for path in evaluation_root.glob("attempt-*") if path.is_dir())
        evaluation = _successful_evaluation_attempt(row, plan, generation)
        invalid_attempts += len(attempts) - int(evaluation is not None)
        if evaluation is not None:
            evaluated += 1
            partition["evaluated"] += 1
    tasks = int(plan["tasks"])
    return {
        "schema_version": 1,
        "status": "complete" if evaluated == tasks else "in_progress",
        "complete": evaluated == tasks,
        "tasks": tasks,
        "generation_ready": generation_ready,
        "evaluated": evaluated,
        "invalid_evaluation_attempts": invalid_attempts,
        "coverage": dict(plan["coverage"]),
        "evaluation_dataset": dict(plan["evaluation_dataset"]),
        "partitions": dict(sorted(partitions.items())),
    }
