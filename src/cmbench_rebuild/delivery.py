"""Fail-closed completion audit for the full CMBench rebuild delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .artifacts.manifest import sha256_file
from .evaluation import (
    aggregate_official_matrix,
    audit_generation_matrix,
    audit_official_matrix_progress,
    bind_aggregate_to_official_plan,
    evaluation_source_identity,
)
from .golden import request_digest, validate_golden
from .methods import LEGACY_REGISTRY, REGISTRY
from .matrix import validate_matrix_plan_inputs
from .runtime.vendor import verify_vendored_source


EXPECTED_METHODS = tuple(sorted(REGISTRY))
EXPECTED_SEEDS = [2, 3]
EXPECTED_CASES_PER_PARTITION = 103
EXPECTED_PARTITIONS = len(EXPECTED_METHODS) * len(EXPECTED_SEEDS)
EXPECTED_TASKS = EXPECTED_PARTITIONS * EXPECTED_CASES_PER_PARTITION


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(project_root: Path, raw: object) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty path")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _coverage_matches_contract(coverage: object) -> bool:
    if not isinstance(coverage, dict):
        return False
    digest = str(coverage.get("case_universe_sha256") or "")
    return (
        coverage.get("methods") == list(EXPECTED_METHODS)
        and coverage.get("seeds") == EXPECTED_SEEDS
        and int(coverage.get("cases_per_partition", -1)) == EXPECTED_CASES_PER_PARTITION
        and int(coverage.get("partitions", -1)) == EXPECTED_PARTITIONS
        and int(coverage.get("tasks", -1)) == EXPECTED_TASKS
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _derive_plan_coverage(plan: dict[str, Any]) -> dict[str, Any]:
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("generation plan has no entries")
    methods = sorted({str(row["method"]) for row in entries})
    seeds = sorted({int(row["seed"]) for row in entries})
    universes: list[set[str]] = []
    for method in methods:
        for seed in seeds:
            cases = [
                str(row["case_id"])
                for row in entries
                if str(row["method"]) == method and int(row["seed"]) == seed
            ]
            if len(cases) != len(set(cases)):
                raise ValueError(f"duplicate cases in {method}/seed-{seed}")
            universes.append(set(cases))
    if not universes or any(universe != universes[0] for universe in universes[1:]):
        raise ValueError("generation partitions do not share one case universe")
    return {
        "methods": methods,
        "seeds": seeds,
        "cases_per_partition": len(universes[0]),
        "partitions": len(universes),
        "tasks": len(entries),
        "case_universe_sha256": hashlib.sha256(
            "\n".join(sorted(universes[0])).encode("utf-8")
        ).hexdigest(),
    }


def audit_protocol_identity(protocol_snapshot: Path, project_root: Path) -> dict[str, Any]:
    snapshot = _load_json(protocol_snapshot)
    methods = snapshot.get("methods")
    problems: list[str] = []
    evidence: dict[str, Any] = {}
    if not isinstance(methods, dict) or set(methods) != set(EXPECTED_METHODS):
        problems.append("protocol must contain exactly the eight registered methods")
        methods = methods if isinstance(methods, dict) else {}
    for method in EXPECTED_METHODS:
        row = methods.get(method)
        if not isinstance(row, dict):
            continue
        try:
            request_path = _resolve(project_root, row.get("request_file"))
            count, digest = request_digest(request_path)
            if count != EXPECTED_CASES_PER_PARTITION or int(row.get("request_count", -1)) != count:
                problems.append(f"{method}: request count is not 103")
            if row.get("request_sha256") != digest:
                problems.append(f"{method}: request SHA-256 changed")
            expected_legacy = REGISTRY[method].legacy_id
            if row.get("legacy_method") != expected_legacy:
                problems.append(f"{method}: legacy method mapping changed")
            evidence[method] = {"path": str(request_path), "count": count, "sha256": digest}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"{method}: {error}")
    return {
        "passed": not problems and len(evidence) == len(EXPECTED_METHODS),
        "protocol": snapshot.get("protocol"),
        "snapshot": str(protocol_snapshot.resolve()),
        "snapshot_sha256": sha256_file(protocol_snapshot),
        "methods": evidence,
        "problems": problems,
    }


def audit_legacy_candidates(golden_dir: Path) -> dict[str, Any]:
    summaries = sorted(golden_dir.glob("*.summary.json"))
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    legacy_names: list[str] = []
    if len(summaries) != len(EXPECTED_METHODS):
        problems.append(f"expected eight legacy summaries, found {len(summaries)}")
    for path in summaries:
        try:
            value = _load_json(path)
            validation = validate_golden(path)
            state = validation.get("oracle_state")
            legacy_name = str((value.get("source") or {}).get("legacy_method") or "")
            legacy_names.append(legacy_name)
            if not validation.get("passed"):
                problems.append(f"{path.name}: malformed golden state/evidence")
            if state not in {"candidate", "rejected", "qualified", "verified"}:
                problems.append(f"{path.name}: invalid oracle state")
            if state in {"qualified", "verified"} and not validation.get("qualified_for_comparison"):
                problems.append(f"{path.name}: labeled {state} without admissible evidence")
            if state == "candidate" and not value.get("known_deviations"):
                problems.append(f"{path.name}: candidate must document why it is not an oracle")
            rows.append({
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "legacy_method": legacy_name,
                "oracle_state": state,
                "qualified_for_comparison": bool(validation.get("qualified_for_comparison")),
            })
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"{path.name}: {error}")
    if set(legacy_names) != set(LEGACY_REGISTRY):
        problems.append("legacy summaries do not cover exactly the eight legacy method names")
    return {
        "passed": not problems,
        "summaries": rows,
        "qualified": sum(int(row["qualified_for_comparison"]) for row in rows),
        "problems": problems,
    }


def audit_source_boundary(
    project_root: Path,
    generation_plan: dict[str, Any],
    official_plan: dict[str, Any],
    legacy_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    legacy_root = legacy_root.resolve()
    problems: list[str] = []
    if project_root == legacy_root or _is_within(project_root, legacy_root):
        problems.append("new project root overlaps the legacy checkout")
    def check_input_path(path: Path, label: str, *, require_project: bool = False) -> None:
        if _is_within(path, legacy_root):
            problems.append(f"{label} depends on the read-only legacy checkout: {path}")
        if require_project and not _is_within(path, project_root):
            problems.append(f"{label} escapes new project root: {path}")

    config_paths: list[str] = []
    for row in generation_plan.get("entries") or []:
        try:
            config = _resolve(project_root, row.get("config_path"))
            config_paths.append(str(config))
            check_input_path(config, "generation config", require_project=True)
            if _is_within(config, project_root) and not config.is_file():
                problems.append(f"generation config is missing: {config}")
            case_root = _resolve(project_root, row.get("case_root"))
            if not _is_within(case_root, project_root):
                problems.append(f"generation case root escapes new project root: {case_root}")
        except (TypeError, ValueError) as error:
            problems.append(f"invalid generation entry path: {error}")
    for row in generation_plan.get("sources") or []:
        try:
            request = _resolve(project_root, row.get("request_path"))
            check_input_path(request, "generation request", require_project=True)
            if _is_within(request, project_root) and not request.is_file():
                problems.append(f"generation request is missing: {request}")
            head_map = row.get("head_map_identity")
            if head_map is not None:
                head_map_path = _resolve(project_root, head_map.get("path"))
                check_input_path(head_map_path, "generation head map", require_project=True)
                if _is_within(head_map_path, project_root) and not head_map_path.is_file():
                    problems.append(f"generation head map is missing: {head_map_path}")
        except (AttributeError, TypeError, ValueError) as error:
            problems.append(f"invalid generation source path: {error}")
    for label, identities in (
        ("generation checkpoint", generation_plan.get("checkpoints") or []),
        ("generation input video", generation_plan.get("input_videos") or []),
    ):
        for identity in identities:
            try:
                check_input_path(_resolve(project_root, identity.get("path")), label)
            except (AttributeError, TypeError, ValueError) as error:
                problems.append(f"invalid {label} path: {error}")
    for row in official_plan.get("entries") or []:
        for field in ("generation_case_root", "evaluation_case_root"):
            try:
                case_root = _resolve(project_root, row.get(field))
                if not _is_within(case_root, project_root):
                    problems.append(f"official {field} escapes new project root: {case_root}")
            except (TypeError, ValueError) as error:
                problems.append(f"invalid official {field}: {error}")
    evaluator = _resolve(project_root, ((official_plan.get("inputs") or {}).get("evaluator") or {}).get("path"))
    check_input_path(evaluator, "official evaluator", require_project=True)
    if _is_within(evaluator, project_root) and not evaluator.is_file():
        problems.append(f"official evaluator is missing: {evaluator}")
    production_sources = [
        project_root / "src/cmbench_rebuild/production.py",
        *sorted((project_root / "src/cmbench_rebuild/runtime").glob("*.py")),
        *sorted((project_root / "src/cmbench_rebuild/methods").glob("*.py")),
        evaluator,
    ]
    forbidden_literal = str(legacy_root)
    contaminated: list[str] = []
    for path in production_sources:
        text = path.read_text(encoding="utf-8")
        if forbidden_literal in text or "cmbench_rebuild.execution" in text:
            contaminated.append(str(path))
    if contaminated:
        problems.append("production dependency surface references the legacy execution adapter")
    vendored = verify_vendored_source()
    return {
        "passed": not problems and vendored.get("verified") is True,
        "project_root": str(project_root),
        "legacy_root": str(legacy_root),
        "generation_configs": sorted(set(config_paths)),
        "official_evaluator": str(evaluator),
        "vendored_source": vendored,
        "contaminated_sources": contaminated,
        "problems": problems,
    }


def audit_plan_binding(
    generation_plan_path: Path,
    generation_plan: dict[str, Any],
    official_plan: dict[str, Any],
    source_root: Path | None = None,
) -> dict[str, Any]:
    problems: list[str] = []
    production_source = validate_matrix_plan_inputs(
        generation_plan,
        source_root or Path(str((generation_plan.get("production_source") or {}).get("root"))),
    )
    if official_plan.get("production_source") != production_source:
        problems.append("official plan is not bound to the generation production source")
    if any(
        row.get("production_source_sha256") != production_source["sha256"]
        for row in official_plan.get("entries") or []
    ):
        problems.append("official plan entries do not share the production source identity")
    evaluation_source = evaluation_source_identity(
        source_root or Path(str(production_source["root"]))
    )
    if official_plan.get("evaluation_source") != evaluation_source:
        problems.append("official plan is not bound to the evaluation adapter source")
    if any(
        row.get("evaluation_source_sha256") != evaluation_source["sha256"]
        for row in official_plan.get("entries") or []
    ):
        problems.append("official plan entries do not share the evaluation source identity")
    generation_identity = official_plan.get("generation_plan") or {}
    actual_sha = sha256_file(generation_plan_path)
    if generation_identity.get("sha256") != actual_sha:
        problems.append("official plan is not bound to the supplied generation plan bytes")
    declared_generation_path = Path(str(generation_identity.get("path") or "")).resolve()
    if declared_generation_path != generation_plan_path.resolve():
        problems.append("official plan points at a different generation plan path")
    generation_coverage = _derive_plan_coverage(generation_plan)
    official_entry_coverage = _derive_plan_coverage(official_plan)
    official_coverage = official_plan.get("coverage")
    if not _coverage_matches_contract(generation_coverage):
        problems.append("generation plan does not encode the exact 8x103x2 coverage contract")
    if not _coverage_matches_contract(official_coverage):
        problems.append("official plan does not encode the exact 8x103x2 coverage contract")
    if not _coverage_matches_contract(official_entry_coverage):
        problems.append("official plan entries do not encode the exact 8x103x2 coverage contract")
    if generation_coverage != official_coverage:
        problems.append("generation and official plans do not share one coverage identity")
    if official_entry_coverage != official_coverage:
        problems.append("official plan coverage summary does not match its entries")
    generation_entries = {
        (str(row["method"]), int(row["seed"]), str(row["case_id"])): row
        for row in generation_plan.get("entries") or []
    }
    official_entries = {
        (str(row["method"]), int(row["seed"]), str(row["case_id"])): row
        for row in official_plan.get("entries") or []
    }
    for identity, generation_row in generation_entries.items():
        official_row = official_entries.get(identity)
        if official_row is None:
            problems.append(f"official plan is missing generation task {identity}")
            break
        if (
            Path(str(official_row.get("generation_case_root") or "")).resolve()
            != Path(str(generation_row.get("case_root") or "")).resolve()
            or str(official_row.get("production_source_sha256"))
            != str(generation_row.get("production_source_sha256"))
            or Path(str(official_row.get("checkpoint_path") or "")).resolve()
            != Path(str(generation_row.get("checkpoint_path") or "")).resolve()
            or str(official_row.get("checkpoint_sha256"))
            != str(generation_row.get("checkpoint_sha256"))
            or Path(str(official_row.get("input_video_path") or "")).resolve()
            != Path(str(generation_row.get("input_video_path") or "")).resolve()
            or int(official_row.get("input_video_bytes", -1))
            != int(generation_row.get("input_video_bytes", -1))
            or str(official_row.get("input_video_sha256"))
            != str(generation_row.get("input_video_sha256"))
            or official_row.get("head_map_path") != generation_row.get("head_map_path")
            or official_row.get("head_map_bytes") != generation_row.get("head_map_bytes")
            or official_row.get("head_map_sha256") != generation_row.get("head_map_sha256")
        ):
            problems.append(f"official entry provenance differs from generation task {identity}")
            break
    if int(generation_plan.get("tasks", -1)) != EXPECTED_TASKS:
        problems.append("generation plan task total is not 1,648")
    if int(generation_plan.get("methods", -1)) != len(EXPECTED_METHODS):
        problems.append("generation plan method count is not eight")
    if int(generation_plan.get("cases_per_method", -1)) != EXPECTED_CASES_PER_PARTITION:
        problems.append("generation plan cases-per-method count is not 103")
    if generation_plan.get("seeds") != EXPECTED_SEEDS:
        problems.append("generation plan seeds are not exactly [2, 3]")
    if int(official_plan.get("tasks", -1)) != EXPECTED_TASKS:
        problems.append("official plan task total is not 1,648")
    if int(official_plan.get("methods", -1)) != len(EXPECTED_METHODS):
        problems.append("official plan method count is not eight")
    if int(official_plan.get("cases_per_method", -1)) != EXPECTED_CASES_PER_PARTITION:
        problems.append("official plan cases-per-method count is not 103")
    if official_plan.get("seeds") != EXPECTED_SEEDS:
        problems.append("official plan seeds are not exactly [2, 3]")
    return {
        "passed": not problems,
        "generation_plan": {"path": str(generation_plan_path.resolve()), "sha256": actual_sha},
        "coverage": generation_coverage,
        "problems": problems,
    }


def _captured(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return operation()
    except (OSError, TypeError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def _compact_generation(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version", "status", "complete", "tasks", "audited_successes",
        "artifact_qualified", "invalid_completed_tasks", "unresolved_tasks",
        "attempts", "nonqualified_attempts", "artifact_bytes", "accounting_failures",
        "coverage", "passed", "error",
    )
    return {key: value[key] for key in keys if key in value}


def _compact_official(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version", "status", "complete", "tasks", "generation_ready",
        "evaluated", "invalid_evaluation_attempts", "coverage", "passed", "error",
    )
    result = {key: value[key] for key in keys if key in value}
    dataset = value.get("evaluation_dataset")
    if isinstance(dataset, dict):
        references = dataset.get("reference_videos")
        summary = {key: item for key, item in dataset.items() if key != "reference_videos"}
        if isinstance(references, dict):
            encoded = json.dumps(references, sort_keys=True, separators=(",", ":")).encode("utf-8")
            summary["reference_video_cases"] = len(references)
            summary["reference_video_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
        result["evaluation_dataset"] = summary
    return result


def audit_delivery(
    *,
    project_root: Path,
    generation_plan_path: Path,
    official_plan_path: Path,
    aggregate_path: Path,
    protocol_snapshot: Path,
    golden_dir: Path,
    legacy_root: Path,
) -> dict[str, Any]:
    """Audit code/provenance readiness separately from still-pending experiments."""
    project_root = project_root.resolve()
    generation_plan = _load_json(generation_plan_path)
    official_plan = _load_json(official_plan_path)
    checks = {
        "protocol_identity": _captured(lambda: audit_protocol_identity(protocol_snapshot, project_root)),
        "legacy_golden_policy": _captured(lambda: audit_legacy_candidates(golden_dir)),
        "source_boundary": _captured(
            lambda: audit_source_boundary(project_root, generation_plan, official_plan, legacy_root)
        ),
        "plan_binding": _captured(
            lambda: audit_plan_binding(
                generation_plan_path, generation_plan, official_plan, project_root
            )
        ),
    }
    code_ready = all(bool(check.get("passed")) for check in checks.values())
    generation = _captured(lambda: audit_generation_matrix(generation_plan))
    generation["passed"] = bool(generation.get("complete")) and not generation.get("error")
    official = _captured(lambda: audit_official_matrix_progress(official_plan))
    official["passed"] = bool(official.get("complete")) and not official.get("error")
    aggregate: dict[str, Any]
    if not aggregate_path.is_file():
        aggregate = {"passed": False, "status": "pending", "path": str(aggregate_path.resolve())}
    elif not official["passed"]:
        aggregate = {
            "passed": False,
            "status": "invalid",
            "path": str(aggregate_path.resolve()),
            "error": "aggregate exists before the official matrix is complete",
        }
    else:
        def verify_aggregate() -> dict[str, Any]:
            saved = _load_json(aggregate_path)
            recomputed = bind_aggregate_to_official_plan(
                aggregate_official_matrix(official_plan), official_plan_path
            )
            if saved != recomputed:
                raise ValueError("saved aggregate differs from a fresh strict aggregation")
            return {
                "passed": True,
                "status": "complete",
                "path": str(aggregate_path.resolve()),
                "sha256": sha256_file(aggregate_path),
                "tasks": saved.get("tasks"),
                "coverage": saved.get("coverage"),
            }
        aggregate = _captured(verify_aggregate)
    experiment_complete = generation["passed"] and official["passed"] and aggregate.get("passed") is True
    if code_ready and experiment_complete:
        status = "complete"
    elif code_ready and not any(check.get("error") for check in (generation, official, aggregate)):
        status = "experiments_pending"
    else:
        status = "invalid"
    return {
        "schema_version": 1,
        "status": status,
        "passed": status == "complete",
        "code_ready": code_ready,
        "experiment_complete": experiment_complete,
        "contract": {
            "methods": list(EXPECTED_METHODS),
            "seeds": EXPECTED_SEEDS,
            "cases_per_partition": EXPECTED_CASES_PER_PARTITION,
            "partitions": EXPECTED_PARTITIONS,
            "tasks": EXPECTED_TASKS,
        },
        "code_checks": checks,
        "experiments": {
            "generation": _compact_generation(generation),
            "official_evaluation": _compact_official(official),
            "aggregate": aggregate,
        },
    }
