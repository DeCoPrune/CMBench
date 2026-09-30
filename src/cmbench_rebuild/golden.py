"""Golden protocol snapshots and explicit, reportable equivalence checks."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .methods import LEGACY_REGISTRY, REGISTRY
from .artifacts.manifest import sha256_file

REQUIRED_METRICS = ("dino_overall", "dino_object", "dino_scene", "seqpr", "generation_seconds", "generation_fps", "peak_memory_bytes")
REQUIRED_ARTIFACTS = ("video", "token_trajectory", "selection_trajectory", "log")
GOLDEN_STATES = {"candidate", "qualified", "verified", "rejected"}
REQUIRED_QUALIFICATION_EVIDENCE = {
    "request_identity",
    "artifact_invariants",
    "representative_runtime",
    "full_matrix",
    "official_evaluation",
    "multi_seed",
}


def _evidence_path(golden: Path, raw: object) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("evidence path is empty")
    path = Path(text)
    return path if path.is_absolute() else (golden.parent / path).resolve()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_bound_path(document_path: Path, raw: object) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("bound path is empty")
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    candidates = [(parent / path).resolve() for parent in (document_path.parent, *document_path.parents)]
    matches = [candidate for candidate in candidates if candidate.exists()]
    if not matches:
        raise FileNotFoundError(f"cannot resolve bound path {text!r} from {document_path}")
    return matches[0]


def _request_identity_semantics(value: dict[str, Any], document_path: Path) -> bool:
    methods = value.get("methods") or {}
    if not isinstance(methods, dict) or set(methods) != set(REGISTRY):
        return False
    universes: list[set[str]] = []
    for method, row in methods.items():
        if not isinstance(row, dict) or row.get("legacy_method") != REGISTRY[method].legacy_id:
            return False
        request_path = _resolve_bound_path(document_path, row.get("request_file"))
        lines = [line for line in request_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        requests = [json.loads(line) for line in lines]
        if (
            len(requests) != 103
            or int(row.get("request_count", -1)) != 103
            or row.get("request_sha256") != sha256_file(request_path)
            or any(request.get("method") != row["legacy_method"] for request in requests)
            or any(int(request.get("seed", -1)) != 2 for request in requests)
        ):
            return False
        case_ids = [str(request.get("case_id") or "") for request in requests]
        if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
            return False
        universes.append(set(case_ids))
    return bool(universes) and all(universe == universes[0] for universe in universes[1:])


def _representative_runtime_semantics(value: dict[str, Any], document_path: Path) -> bool:
    from .artifacts.video import inspect_video
    from .evidence import generation_attempt_qualified
    from .identity import validate_directory_content_identity, validate_file_content_identity
    from .matrix import production_source_identity

    config = value.get("config") or {}
    source = (value.get("backend") or {}).get("source") or {}
    production_source = value.get("production_source") or {}
    checkpoint_identity = value.get("checkpoint_identity") or {}
    input_video_identity = value.get("input_video_identity") or {}
    head_map_identity = value.get("head_map_identity")
    artifacts = value.get("artifacts") or {}
    required_artifact_mapping = {
        "metadata": "metadata.json",
        "video": "continuation.mp4",
        "seqpr": "seq_pr_metrics.json",
        "token_trajectory": "token_trajectory.jsonl",
        "selection": "selection_trajectory.json",
    }
    if not (
        value.get("status") == "completed"
        and value.get("disqualified") is False
        and config.get("method") in REGISTRY
        and int(config.get("seed", -1)) in {2, 3}
        and source.get("verified") is True
        and _is_sha256(source.get("subtree_sha256"))
        and _is_sha256(production_source.get("sha256"))
        and _is_sha256(checkpoint_identity.get("sha256"))
        and _is_sha256(input_video_identity.get("sha256"))
        and all(artifacts.get(key) == name for key, name in required_artifact_mapping.items())
    ):
        return False
    if production_source_identity(Path(str(production_source.get("root")))) != production_source:
        return False
    if head_map_identity is not None and not isinstance(head_map_identity, dict):
        return False
    try:
        validate_directory_content_identity(checkpoint_identity, verify_content=True)
        validate_file_content_identity(input_video_identity)
        if head_map_identity is not None:
            validate_file_content_identity(head_map_identity)
    except (OSError, TypeError, ValueError):
        return False
    if bool(config.get("head_map_file")) != (head_map_identity is not None):
        return False
    attempt = document_path.parent.resolve()
    metadata_path = attempt / str(artifacts["metadata"])
    video_path = attempt / str(artifacts["video"])
    selection_path = attempt / str(artifacts["selection"])
    manifest_path = attempt / "artifacts.manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_matches = (
        manifest.get("method") == config.get("method") == metadata.get("method")
        and str(manifest.get("case_id")) == str(config.get("case_id")) == str(metadata.get("case_id"))
        and int(manifest.get("seed", -1)) == int(config.get("seed", -1)) == int(metadata.get("seed", -1))
        and manifest.get("production_source_sha256") == production_source["sha256"]
        and metadata.get("production_source_sha256") == production_source["sha256"]
        and manifest.get("checkpoint_sha256") == checkpoint_identity["sha256"]
        and metadata.get("checkpoint_sha256") == checkpoint_identity["sha256"]
        and manifest.get("input_video_sha256") == input_video_identity["sha256"]
        and metadata.get("input_video_sha256") == input_video_identity["sha256"]
        and manifest.get("head_map_sha256") == (head_map_identity or {}).get("sha256")
        and metadata.get("head_map_sha256") == (head_map_identity or {}).get("sha256")
    )
    if not identity_matches or bool(metadata.get("diagnostic_disqualified")):
        return False
    if (
        int(metadata.get("context_latent_frames", -1)) != 244
        or int(metadata.get("output_latent_frames", -1)) != 8
        or int(metadata.get("output_pixel_frames", -1)) != 32
    ):
        return False
    video = inspect_video(video_path)
    recorded_video = metadata.get("video") or {}
    recorded_frame_hashes = recorded_video.get("decoded_frame_sha256")
    if (
        int(video.get("decoded_frame_count", -1)) != 32
        or not isinstance(recorded_frame_hashes, list)
        or len(recorded_frame_hashes) != 32
        or any(not _is_sha256(frame_hash) for frame_hash in recorded_frame_hashes)
        or video.get("decoded_frame_sha256") != recorded_frame_hashes
        or int(recorded_video.get("decoded_frame_count", -1)) != 32
    ):
        return False
    row = {
        "generation_case_root": str(attempt.parent),
        "method": config["method"],
        "case_id": config["case_id"],
        "seed": int(config["seed"]),
        "production_source_sha256": production_source["sha256"],
        "checkpoint_path": checkpoint_identity["path"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "input_video_path": input_video_identity["path"],
        "input_video_bytes": input_video_identity["bytes"],
        "input_video_sha256": input_video_identity["sha256"],
        "head_map_path": (head_map_identity or {}).get("path"),
        "head_map_bytes": (head_map_identity or {}).get("bytes"),
        "head_map_sha256": (head_map_identity or {}).get("sha256"),
    }
    return selection_path.is_file() and generation_attempt_qualified(
        attempt,
        row,
        production_source=production_source,
        checkpoint=checkpoint_identity,
        input_video=input_video_identity,
    )


def _official_evaluation_semantics(value: dict[str, Any], document_path: Path) -> bool:
    from .evaluation.matrix import aggregate_official_matrix, bind_aggregate_to_official_plan

    if not (
        value.get("status") == "complete"
        and value.get("official_compatible") is True
        and value.get("scoring_path") == "owl_sam_dino"
        and int(value.get("tasks", -1)) == 1648
        and _coverage_is_complete(value)
        and set(value.get("by_method") or {}) == set(REGISTRY)
        and len(value.get("case_records") or []) == 1648
    ):
        return False
    aggregator = value.get("aggregator") or {}
    source_path = _resolve_bound_path(document_path, aggregator.get("source"))
    if aggregator.get("module") != "cmbench_rebuild.evaluation.matrix":
        return False
    if aggregator.get("source_sha256") != sha256_file(source_path):
        return False
    generation_plan = value.get("generation_plan") or {}
    generation_plan_path = _resolve_bound_path(document_path, generation_plan.get("path"))
    if generation_plan.get("sha256") != sha256_file(generation_plan_path):
        return False
    official_plan = value.get("official_evaluation_plan") or {}
    official_plan_path = _resolve_bound_path(document_path, official_plan.get("path"))
    if official_plan.get("sha256") != sha256_file(official_plan_path):
        return False
    plan = json.loads(official_plan_path.read_text(encoding="utf-8"))
    recomputed = bind_aggregate_to_official_plan(
        aggregate_official_matrix(plan), official_plan_path
    )
    return recomputed == value


def _coverage_is_complete(value: dict[str, Any]) -> bool:
    coverage = value.get("coverage") or {}
    return (
        isinstance(coverage, dict)
        and int(coverage.get("tasks", -1)) == 1648
        and int(coverage.get("partitions", -1)) == 16
        and int(coverage.get("cases_per_partition", -1)) == 103
        and coverage.get("seeds") == [2, 3]
        and set(coverage.get("methods") or []) == set(REGISTRY)
        and _is_sha256(coverage.get("case_universe_sha256"))
    )


def _evidence_semantics(kind: str, value: dict[str, Any], document_path: Path) -> bool:
    if kind == "request_identity":
        return _request_identity_semantics(value, document_path)
    if kind == "representative_runtime":
        return _representative_runtime_semantics(value, document_path)
    if kind in {"artifact_invariants", "full_matrix", "multi_seed"}:
        return (
            value.get("complete") is True
            and int(value.get("tasks", -1)) == 1648
            and int(value.get("audited_successes", -1)) == 1648
            and int(value.get("invalid_completed_tasks", -1)) == 0
            and not value.get("accounting_failures")
            and _coverage_is_complete(value)
        )
    if kind == "official_evaluation":
        return _official_evaluation_semantics(value, document_path)
    return False

def validate_golden(path: Path) -> dict[str, Any]:
    """Reject treating an unreviewed legacy result as a migration oracle."""
    value = json.loads(path.read_text(encoding="utf-8"))
    state = value.get("oracle_state")
    evidence = value.get("independent_evidence", [])
    problems: list[str] = []
    validated_evidence: list[dict[str, Any]] = []
    if state not in GOLDEN_STATES:
        problems.append("oracle_state must be candidate, qualified, verified, or rejected")
    if state in {"qualified", "verified"}:
        if not isinstance(evidence, list) or not evidence:
            problems.append("qualified/verified golden requires independent_evidence")
        else:
            by_kind: dict[str, list[dict[str, Any]]] = {}
            for index, item in enumerate(evidence):
                if not isinstance(item, dict):
                    problems.append(f"independent_evidence[{index}] must be an object")
                    continue
                kind = str(item.get("kind") or "")
                if kind not in REQUIRED_QUALIFICATION_EVIDENCE:
                    problems.append(f"independent_evidence[{index}] has unknown kind {kind!r}")
                    continue
                by_kind.setdefault(kind, []).append(item)
            missing = sorted(REQUIRED_QUALIFICATION_EVIDENCE - set(by_kind))
            duplicates = sorted(kind for kind, items in by_kind.items() if len(items) != 1)
            if missing:
                problems.append(f"missing qualification evidence kinds: {missing}")
            if duplicates:
                problems.append(f"duplicate qualification evidence kinds: {duplicates}")
            for kind, items in sorted(by_kind.items()):
                if len(items) != 1:
                    continue
                item = items[0]
                try:
                    evidence_path = _evidence_path(path, item.get("path"))
                    declared_sha = str(item.get("sha256") or "")
                    if not evidence_path.is_file():
                        raise ValueError(f"file missing: {evidence_path}")
                    actual_sha = sha256_file(evidence_path)
                    if not _is_sha256(declared_sha) or declared_sha != actual_sha:
                        raise ValueError(f"sha256 mismatch: {declared_sha} != {actual_sha}")
                    document = json.loads(evidence_path.read_text(encoding="utf-8"))
                    if not isinstance(document, dict) or not _evidence_semantics(kind, document, evidence_path):
                        raise ValueError("semantic qualification check failed")
                    validated_evidence.append({"kind": kind, "path": str(evidence_path), "sha256": actual_sha})
                except (OSError, KeyError, IndexError, TypeError, AttributeError, ValueError, json.JSONDecodeError) as error:
                    problems.append(f"{kind}: {error}")
            coverage_documents = [
                json.dumps(json.loads(Path(item["path"]).read_text(encoding="utf-8"))["coverage"], sort_keys=True)
                for item in validated_evidence
                if item["kind"] in {"artifact_invariants", "full_matrix", "official_evaluation", "multi_seed"}
            ]
            if coverage_documents and len(set(coverage_documents)) != 1:
                problems.append("qualification evidence does not share one exact matrix coverage identity")
    if state == "verified" and value.get("known_deviations"):
        problems.append("verified golden cannot retain known_deviations")
    qualified = state in {"qualified", "verified"} and not problems
    return {
        "path": str(path),
        "oracle_state": state,
        "passed": not problems,
        "qualified_for_comparison": qualified,
        "validated_evidence": validated_evidence,
        "problems": problems,
    }

def request_digest(path: Path) -> tuple[int, str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        row = json.loads(line)
        if row.get("method") not in LEGACY_REGISTRY:
            raise ValueError(f"{path}: unknown legacy method {row.get('method')!r}")
    return len(lines), sha256_file(path)

def snapshot_protocol(request_dir: Path, output: Path, protocol: str) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for legacy, spec in LEGACY_REGISTRY.items():
        candidates = sorted(request_dir.glob(f"{legacy}*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"missing request snapshot for {legacy}")
        count, digest = request_digest(candidates[0])
        methods[spec.id] = {"legacy_method": legacy, "request_file": str(candidates[0]), "request_count": count, "request_sha256": digest}
    snapshot = {"schema_version": 1, "protocol": protocol, "methods": methods, "comparability": {"determinism": {"required": ["fixed seed", "fixed request JSONL", "fixed checkpoint", "fixed world size", "fixed scheduler"], "video": "exact decoded-frame hash preferred; perceptual hash only when decoder metadata differs"}, "tolerances": {"metrics_absolute": 1e-6, "token_counts_exact": True, "selection_trajectory_exact": True, "video_frame_hash_exact": True, "fps_relative": 0.10, "peak_memory_relative": 0.10}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot

def _number_delta(expected: Any, actual: Any) -> float | None:
    if expected is None or actual is None: return None
    return abs(float(expected) - float(actual))

def compare(expected: Path, actual: Path, output: Path) -> dict[str, Any]:
    old, new = json.loads(expected.read_text()), json.loads(actual.read_text())
    oracle = validate_golden(expected)
    if not oracle["qualified_for_comparison"]:
        report = {"schema_version": 1, "expected": str(expected), "actual": str(actual), "passed": False, "failures": ["golden_not_qualified"], "oracle_validation": oracle, "checks": {}}
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    tol = old.get("tolerances", {"metrics_absolute": 1e-6, "fps_relative": .10, "peak_memory_relative": .10})
    failures: list[str] = []
    checks: dict[str, Any] = {}
    for field in REQUIRED_METRICS:
        delta = _number_delta(old.get("metrics", {}).get(field), new.get("metrics", {}).get(field))
        limit = tol.get("metrics_absolute", 1e-6)
        if field in {"generation_fps", "peak_memory_bytes"} and old.get("metrics", {}).get(field):
            limit = abs(float(old["metrics"][field])) * tol.get("fps_relative" if field == "generation_fps" else "peak_memory_relative", .10)
        passed = delta is not None and delta <= limit
        checks[field] = {"delta": delta, "limit": limit, "passed": passed}
        if not passed: failures.append(field)
    for field in REQUIRED_ARTIFACTS:
        same = old.get("artifacts", {}).get(field) == new.get("artifacts", {}).get(field)
        checks[field] = {"passed": same}
        if not same: failures.append(field)
    report = {"schema_version": 1, "expected": str(expected), "actual": str(actual), "passed": not failures, "failures": failures, "oracle_validation": oracle, "checks": checks}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def collect_legacy_summary(summary: Path, *, case_id: str, output: Path) -> dict[str, Any]:
    """Freeze what a completed legacy summary proves, and name what it does not."""
    rows = [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row.get("case_id") == case_id and row.get("status") == "ok"]
    if len(matches) != 1:
        raise ValueError(f"{summary}: expected one successful row for {case_id}, found {len(matches)}")
    row = matches[0]
    peaks = [item.get("generation_peak_reserved") for item in row.get("gpu_memory_bytes_by_rank", [])]
    result = {
        "schema_version": 1,
        "status": "partial_not_admissible_for_migration",
        "oracle_state": "candidate",
        "independent_evidence": [],
        "known_deviations": ["Legacy summary is observational evidence, not independently validated ground truth."],
        "source": {"summary": str(summary), "summary_sha256": sha256_file(summary), "legacy_method": row.get("method"), "case_id": case_id, "seed": row.get("seed"), "checkpoint": row.get("checkpoint")},
        "metrics": {"dino_overall": row.get("dino_overall"), "dino_object": row.get("dino_object"), "dino_scene": row.get("dino_scene"), "seqpr": row.get("seqPR"), "generation_seconds": row.get("generation_elapsed_s"), "generation_fps": row.get("generation_fps_forcingkv_public"), "peak_memory_bytes": max((int(value) for value in peaks if value is not None), default=None)},
        "artifacts": {"video": row.get("continuation_video"), "token_trajectory": row.get("mask_accounting"), "selection_trajectory": row.get("ours_q0_decisions"), "log": row.get("worker_log")},
        "missing_required_fields": ["DINO overall/object/scene" if row.get("dino_overall") is None else None, "decoded frame hashes", "normalized per-layer retained-token trajectory"],
    }
    result["missing_required_fields"] = [item for item in result["missing_required_fields"] if item]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
