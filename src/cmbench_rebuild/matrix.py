"""Resumable production-matrix planning and execution."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import load
from .evidence import qualified_generation_attempt
from .gpu import parse_cuda_visible_devices, require_idle_cuda_devices
from .identity import directory_content_identity, file_content_identity
from .runtime.requests import load_requests


MATRIX_PLAN_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MatrixTask:
    method: str
    case_id: str
    seed: int
    config_path: str
    source_request_seed: int
    case_root: str
    production_source_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    input_video_path: str
    input_video_bytes: int
    input_video_sha256: str
    head_map_path: str | None
    head_map_bytes: int | None
    head_map_sha256: str | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_video_path(raw_path: Path, source_root: Path) -> Path:
    path = raw_path.expanduser()
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def _case_video_identities(
    requests: Sequence[Any],
    *,
    source_root: Path,
    cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for request in requests:
        path = _request_video_path(request.clip_file, source_root)
        key = str(path)
        identity = cache.get(key)
        if identity is None:
            identity = file_content_identity(path)
            cache[key] = identity
        values[str(request.case_id)] = identity
    return values


def _case_video_set_sha256(values: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_SHARED_CONFIG_FIELDS = (
    "protocol",
    "dataset_version",
    "height",
    "width",
    "sampling_shift",
    "chunk_size",
    "timesteps_index",
    "model_fps",
    "vae_temporal_stride",
    "world_size",
    "kv_storage_mode",
    "generation_kv_policy",
    "rope_reindex",
)


def _shared_config_contract(config: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    value = {field: config[field] for field in _SHARED_CONFIG_FIELDS}
    value["source_request_seed"] = int(config["seed"])
    value["checkpoint_path"] = str(checkpoint_path.resolve())
    return json.loads(json.dumps(value, sort_keys=True))


def _configured_file_identity(
    raw_path: object, *, source_root: Path, cache: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
    key = str(resolved)
    identity = cache.get(key)
    if identity is None:
        identity = file_content_identity(resolved)
        cache[key] = identity
    return identity


_METHOD_REQUEST_FIELDS = frozenset({
    "method",
    "local_attn_size",
    "sink_size",
    "dummy_first_history_frames",
    "dummy_middle_history_frames",
    "dummy_last_history_frames",
    "ours_step_index",
    "ours_threshold",
    "ours_score_type",
    "ours_local_window_chunks",
    "ours_probe_local_window_chunks",
    "ours_posthoc_no_local_probe",
    "ours_min_keep_per_spatial_block",
    "ours_spatial_floor_threshold",
    "patchify_retrieve_grid_rows",
    "patchify_retrieve_grid_cols",
    "patchify_retrieve_topk_blocks",
})


def _shared_request_contract(requests: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Remove only registered method knobs, leaving the shared case protocol."""
    return {
        str(request.case_id): {
            key: value
            for key, value in request.raw.items()
            if key not in _METHOD_REQUEST_FIELDS
        }
        for request in requests
    }


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_REQUEST_PARAMETER_MAP = {
    "streaming": {"local_attn_size": "local_attn_size", "sink_size": "sink_size"},
    "dummy_forcing": {
        "dummy_first_history_frames": "first_history_frames",
        "dummy_middle_history_frames": "middle_history_frames",
        "dummy_last_history_frames": "last_history_frames",
    },
    "patchification": {
        "patchify_retrieve_grid_rows": "grid_rows",
        "patchify_retrieve_grid_cols": "grid_cols",
        "patchify_retrieve_topk_blocks": "topk_blocks",
    },
    "random": {
        "ours_step_index": "step_index",
        "ours_threshold": "threshold",
        "ours_score_type": "score_type",
        "ours_local_window_chunks": "local_window_chunks",
        "ours_min_keep_per_spatial_block": "min_keep_per_spatial_block",
        "ours_spatial_floor_threshold": "spatial_floor_threshold",
    },
    "decoprune": {
        "ours_step_index": "step_index",
        "ours_threshold": "threshold",
        "ours_score_type": "score_type",
        "ours_local_window_chunks": "local_window_chunks",
        "ours_min_keep_per_spatial_block": "min_keep_per_spatial_block",
        "ours_spatial_floor_threshold": "spatial_floor_threshold",
    },
    "decoprune_hs": {
        "ours_step_index": "step_index",
        "ours_threshold": "threshold",
        "ours_score_type": "score_type",
        "ours_local_window_chunks": "local_window_chunks",
        "ours_min_keep_per_spatial_block": "min_keep_per_spatial_block",
        "ours_spatial_floor_threshold": "spatial_floor_threshold",
    },
}


def _validate_method_request_contract(
    config: Mapping[str, Any], requests: Sequence[Any]
) -> None:
    method = str(config["method"])
    parameter_map = _REQUEST_PARAMETER_MAP.get(method, {})
    allowed = set(parameter_map)
    literals: dict[str, Any] = {}
    if method in {"random", "decoprune", "decoprune_hs"}:
        literals = {
            "ours_probe_local_window_chunks": 0,
            "ours_posthoc_no_local_probe": False,
        }
        allowed.update(literals)
    parameters = config["method_params"]
    for request in requests:
        raw = request.raw
        # Public benchmark requests are method-neutral; the frozen config owns all policy knobs.
        if raw.get("task_id") == request.case_id and not (set(raw) & _METHOD_REQUEST_FIELDS):
            continue
        if raw.get("method") != config["legacy_method"]:
            raise ValueError(
                f"{method}/{request.case_id}: request method does not match config"
            )
        present = (set(raw) & _METHOD_REQUEST_FIELDS) - {"method"}
        if present != allowed:
            raise ValueError(
                f"{method}/{request.case_id}: request method parameters differ: "
                f"expected={sorted(allowed)} actual={sorted(present)}"
            )
        for request_name, parameter_name in parameter_map.items():
            if raw.get(request_name) != parameters[parameter_name]:
                raise ValueError(
                    f"{method}/{request.case_id}: {request_name} differs from config"
                )
        for name, expected in literals.items():
            if raw.get(name) != expected:
                raise ValueError(f"{method}/{request.case_id}: {name} differs from protocol")


def production_source_identity(source_root: Path) -> dict[str, Any]:
    """Hash the exact code surface allowed to influence production generation."""
    root = source_root.resolve()
    candidates = [
        root / "src/cmbench_rebuild/__init__.py",
        root / "src/cmbench_rebuild/config.py",
        root / "src/cmbench_rebuild/dataset.py",
        root / "src/cmbench_rebuild/evidence.py",
        root / "src/cmbench_rebuild/gpu.py",
        root / "src/cmbench_rebuild/identity.py",
        root / "src/cmbench_rebuild/matrix.py",
        root / "src/cmbench_rebuild/postprocess.py",
        root / "src/cmbench_rebuild/production.py",
        root / "src/cmbench_rebuild/worker.py",
        root / "src/cmbench_rebuild/artifacts/__init__.py",
        root / "src/cmbench_rebuild/artifacts/manifest.py",
        root / "src/cmbench_rebuild/artifacts/video.py",
        *sorted((root / "src/cmbench_rebuild/core").rglob("*.py")),
        *sorted((root / "src/cmbench_rebuild/methods").rglob("*.py")),
        *sorted((root / "src/cmbench_rebuild/runtime").rglob("*.py")),
        *sorted((root / "src/wan").rglob("*.py")),
        root / "third_party/lingbot-world-v2/ORIGIN.json",
        root / "pyproject.toml",
    ]
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if not files:
        raise FileNotFoundError(f"production source tree is empty: {root}")
    digest = hashlib.sha256()
    relative_files: list[str] = []
    for path in files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"production source escapes root: {path}") from error
        relative_files.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return {
        "algorithm": "sha256-relative-path-null-content-digest-v1",
        "root": str(root),
        "files": len(relative_files),
        "sha256": digest.hexdigest(),
    }


def validate_matrix_plan_inputs(
    plan: dict[str, Any], source_root: Path | None = None
) -> dict[str, Any]:
    """Re-derive and reject any stale or structurally altered production plan."""
    if plan.get("schema_version") != MATRIX_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported generation plan schema: {plan.get('schema_version')!r}"
        )
    declared_source = plan.get("production_source")
    if not isinstance(declared_source, dict) or not declared_source.get("root") or not declared_source.get("sha256"):
        raise ValueError("generation plan lacks a production source identity")
    current_root = (source_root or Path(str(declared_source["root"]))).resolve()
    current_source = production_source_identity(current_root)
    if declared_source != current_source:
        raise ValueError("generation plan production source identity changed")

    seeds = plan.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("generation plan seeds are invalid")
    output_value = plan.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("generation plan lacks an output root")
    output_root = Path(output_value).resolve()

    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != int(plan.get("methods", -1)):
        raise ValueError("generation plan source list changed")
    method_sources: dict[str, dict[str, Any]] = {}
    used_checkpoint_paths: set[str] = set()
    checkpoint_rows = plan.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or not checkpoint_rows:
        raise ValueError("generation plan lacks checkpoint identities")
    checkpoints: dict[str, dict[str, Any]] = {}
    for identity in checkpoint_rows:
        checkpoint_path = str(Path(str(identity.get("path") or "")).resolve())
        if checkpoint_path in checkpoints:
            raise ValueError(f"generation plan contains duplicate checkpoint: {checkpoint_path}")
        current_checkpoint = directory_content_identity(Path(checkpoint_path))
        if identity != current_checkpoint:
            raise ValueError(f"generation checkpoint content identity changed: {checkpoint_path}")
        checkpoints[checkpoint_path] = current_checkpoint
    if len(checkpoints) != 1:
        raise ValueError("generation matrix must use one shared checkpoint")
    head_map_rows = plan.get("head_maps")
    if not isinstance(head_map_rows, list):
        raise ValueError("generation plan lacks head-map identities")
    head_maps: dict[str, dict[str, Any]] = {}
    for identity in head_map_rows:
        if not isinstance(identity, dict):
            raise ValueError("generation plan head-map identity is invalid")
        current_head_map = file_content_identity(Path(str(identity.get("path") or "")))
        if identity != current_head_map or current_head_map["path"] in head_maps:
            raise ValueError(f"generation head-map content identity changed: {identity.get('path')}")
        head_maps[current_head_map["path"]] = current_head_map
    if len(head_maps) > 1:
        raise ValueError("heterogeneous methods must use one shared head map")
    input_video_rows = plan.get("input_videos")
    if not isinstance(input_video_rows, list) or not input_video_rows:
        raise ValueError("generation plan lacks input video identities")
    input_videos: dict[str, dict[str, Any]] = {}
    for identity in input_video_rows:
        if not isinstance(identity, dict):
            raise ValueError("generation plan input video identity is invalid")
        current_video = file_content_identity(Path(str(identity.get("path") or "")))
        if identity != current_video or current_video["path"] in input_videos:
            raise ValueError(f"generation input video content identity changed: {identity.get('path')}")
        input_videos[current_video["path"]] = current_video
    reference_cases: tuple[str, ...] | None = None
    reference_case_videos: dict[str, dict[str, Any]] | None = None
    reference_shared_requests: dict[str, dict[str, Any]] | None = None
    declared_shared_request_sha256 = plan.get("shared_request_sha256")
    if not isinstance(declared_shared_request_sha256, str):
        raise ValueError("generation plan lacks a shared request identity")
    shared_config = plan.get("shared_config")
    if not isinstance(shared_config, dict):
        raise ValueError("generation plan lacks a shared runtime configuration")
    if plan.get("protocol") != shared_config.get("protocol"):
        raise ValueError("generation plan protocol differs from shared runtime configuration")
    used_input_video_paths: set[str] = set()
    used_head_map_paths: set[str] = set()
    current_video_cache: dict[str, dict[str, Any]] = {}
    current_head_map_cache: dict[str, dict[str, Any]] = {}
    for row in sources:
        config_path = Path(str(row["config_path"])).resolve()
        request_path = Path(str(row["request_path"])).resolve()
        if not config_path.is_file() or _sha256(config_path) != row.get("config_sha256"):
            raise ValueError(f"generation config identity changed: {config_path}")
        if not request_path.is_file() or _sha256(request_path) != row.get("request_sha256"):
            raise ValueError(f"generation request identity changed: {request_path}")
        config = load(config_path).normalized()
        method = str(config["method"])
        if row.get("method") != method or method in method_sources:
            raise ValueError("generation plan methods are duplicated or differ from their configs")
        configured_request = Path(str(config["request_file"]))
        if not configured_request.is_absolute():
            configured_request = current_root / configured_request
        if configured_request.resolve() != request_path:
            raise ValueError(f"generation request path differs from config: {config_path}")
        source_seed = int(config["seed"])
        checkpoint_path = str(Path(str(config["checkpoint"])).expanduser().resolve())
        checkpoint = checkpoints.get(checkpoint_path)
        if (
            checkpoint is None
            or Path(str(row.get("checkpoint_path") or "")).resolve() != Path(checkpoint["path"])
            or row.get("checkpoint_sha256") != checkpoint["sha256"]
        ):
            raise ValueError(f"generation checkpoint binding changed: {method}")
        used_checkpoint_paths.add(checkpoint_path)
        if _shared_config_contract(config, Path(checkpoint_path)) != shared_config:
            raise ValueError(f"generation shared runtime configuration differs: {method}")
        head_map = _configured_file_identity(
            config.get("head_map_file"),
            source_root=current_root,
            cache=current_head_map_cache,
        )
        if head_map is not None:
            used_head_map_paths.add(head_map["path"])
            if head_map["path"] not in head_maps:
                raise ValueError(f"generation source references an unbound head map: {method}")
        if row.get("head_map_identity") != head_map:
            raise ValueError(f"generation source head-map identity changed: {method}")
        requests = load_requests(request_path)
        method_requests = tuple(request for request in requests if request.seed == source_seed)
        if not method_requests:
            raise ValueError(f"request seed has no cases: {request_path} seed={source_seed}")
        _validate_method_request_contract(config, method_requests)
        current_shared_requests = _shared_request_contract(method_requests)
        current_shared_request_sha256 = _mapping_sha256(current_shared_requests)
        if row.get("shared_request_sha256") != current_shared_request_sha256:
            raise ValueError(f"generation source shared request identity changed: {method}")
        if reference_shared_requests is None:
            reference_shared_requests = current_shared_requests
        elif current_shared_requests != reference_shared_requests:
            raise ValueError(f"generation shared request contract differs: {method}")
        cases = tuple(sorted(request.case_id for request in requests if request.seed == source_seed))
        if (
            row.get("source_request_seed") != source_seed
            or row.get("cases") != len(cases)
        ):
            raise ValueError(f"generation source summary changed: {method}")
        if reference_cases is None:
            reference_cases = cases
        elif reference_cases != cases:
            raise ValueError(f"generation source case universe differs: {method}")
        case_videos = _case_video_identities(
            method_requests, source_root=current_root, cache=current_video_cache
        )
        if any(identity["path"] not in input_videos for identity in case_videos.values()):
            raise ValueError(f"generation source references an unbound input video: {method}")
        if row.get("input_video_set_sha256") != _case_video_set_sha256(case_videos):
            raise ValueError(f"generation source input video mapping changed: {method}")
        if reference_case_videos is None:
            reference_case_videos = case_videos
        elif reference_case_videos != case_videos:
            raise ValueError(f"generation source input videos differ across methods: {method}")
        used_input_video_paths.update(identity["path"] for identity in case_videos.values())
        method_sources[method] = {
            "config_path": str(config_path),
            "source_request_seed": source_seed,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "head_map_identity": head_map,
            "case_videos": case_videos,
        }

    if used_checkpoint_paths != set(checkpoints):
        raise ValueError("generation plan checkpoint list contains unused or missing roots")
    if used_head_map_paths != set(head_maps):
        raise ValueError("generation plan head-map list contains unused or missing files")
    if used_input_video_paths != set(input_videos):
        raise ValueError("generation plan input video list contains unused or missing files")
    if _mapping_sha256(reference_shared_requests or {}) != declared_shared_request_sha256:
        raise ValueError("generation plan shared request identity changed")

    cases = reference_cases or ()
    expected_tasks = len(method_sources) * len(seeds) * len(cases)
    if (
        int(plan.get("cases_per_method", -1)) != len(cases)
        or int(plan.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("generation plan dimensions differ from its immutable inputs")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_tasks:
        raise ValueError("generation plan entry list changed")
    actual: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in entries:
        method = row.get("method")
        seed = row.get("seed")
        case_id = row.get("case_id")
        if not isinstance(method, str) or type(seed) is not int or not isinstance(case_id, str):
            raise ValueError("generation plan entry identity is invalid")
        identity = (method, seed, case_id)
        if identity in actual:
            raise ValueError(f"generation plan contains duplicate entry: {identity}")
        actual[identity] = row
    expected = {
        (method, seed, case_id)
        for method in method_sources
        for seed in seeds
        for case_id in cases
    }
    if set(actual) != expected:
        raise ValueError("generation plan entries differ from the complete input cartesian product")
    for (method, seed, case_id), row in actual.items():
        source = method_sources[method]
        expected_case_root = (output_root / method / f"seed-{seed}" / case_id).resolve()
        video = source["case_videos"][case_id]
        head_map = source["head_map_identity"]
        if (
            Path(str(row.get("config_path"))).resolve() != Path(source["config_path"])
            or row.get("source_request_seed") != source["source_request_seed"]
            or Path(str(row.get("checkpoint_path") or "")).resolve()
            != Path(source["checkpoint_path"])
            or row.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or Path(str(row.get("case_root"))).resolve() != expected_case_root
            or Path(str(row.get("input_video_path"))).resolve() != Path(video["path"])
            or int(row.get("input_video_bytes", -1)) != int(video["bytes"])
            or str(row.get("input_video_sha256")) != str(video["sha256"])
            or row.get("head_map_path") != (head_map or {}).get("path")
            or row.get("head_map_bytes") != (head_map or {}).get("bytes")
            or row.get("head_map_sha256") != (head_map or {}).get("sha256")
        ):
            raise ValueError(f"generation plan entry provenance changed: {(method, seed, case_id)}")
        if row.get("production_source_sha256") != current_source["sha256"]:
            raise ValueError("generation plan entries do not share the production source identity")
    return current_source


def build_matrix_plan(
    config_paths: Sequence[Path],
    *,
    seeds: Sequence[int],
    output_root: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if not config_paths:
        raise ValueError("at least one method config is required")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds) or any(seed < 0 for seed in normalized_seeds):
        raise ValueError("seeds must be unique non-negative integers")
    tasks: list[MatrixTask] = []
    sources: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    input_videos: dict[str, dict[str, Any]] = {}
    head_maps: dict[str, dict[str, Any]] = {}
    source = (source_root or Path.cwd()).resolve()
    production_source = production_source_identity(source)
    reference_cases: tuple[str, ...] | None = None
    reference_case_videos: dict[str, dict[str, Any]] | None = None
    reference_shared_requests: dict[str, dict[str, Any]] | None = None
    shared_config: dict[str, Any] | None = None
    seen_methods: set[str] = set()
    for raw_config_path in config_paths:
        config_path = raw_config_path.resolve()
        config = load(config_path).normalized()
        method = str(config["method"])
        if method in seen_methods:
            raise ValueError(f"duplicate method config: {method}")
        seen_methods.add(method)
        request_path = Path(config["request_file"])
        if not request_path.is_absolute():
            request_path = source / request_path
        requests = load_requests(request_path)
        source_seed = int(config["seed"])
        checkpoint_path = Path(str(config["checkpoint"])).expanduser().resolve()
        checkpoint_key = str(checkpoint_path)
        if checkpoint_key not in checkpoints:
            checkpoints[checkpoint_key] = directory_content_identity(checkpoint_path)
        checkpoint = checkpoints[checkpoint_key]
        current_shared_config = _shared_config_contract(config, checkpoint_path)
        if shared_config is None:
            shared_config = current_shared_config
        elif current_shared_config != shared_config:
            raise ValueError(f"shared runtime configuration differs for {method}")
        head_map = _configured_file_identity(
            config.get("head_map_file"), source_root=source, cache=head_maps
        )
        method_requests = tuple(request for request in requests if request.seed == source_seed)
        if not method_requests:
            raise ValueError(f"request seed has no cases: {request_path} seed={source_seed}")
        _validate_method_request_contract(config, method_requests)
        current_shared_requests = _shared_request_contract(method_requests)
        if reference_shared_requests is None:
            reference_shared_requests = current_shared_requests
        elif current_shared_requests != reference_shared_requests:
            raise ValueError(f"shared request contract differs for {method}")
        cases = tuple(sorted(request.case_id for request in method_requests))
        if len(cases) != len(set(cases)):
            raise ValueError(f"duplicate cases in {request_path}")
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError(f"method case universe differs for {method}")
        case_videos = _case_video_identities(
            method_requests, source_root=source, cache=input_videos
        )
        if reference_case_videos is None:
            reference_case_videos = case_videos
        elif case_videos != reference_case_videos:
            raise ValueError(f"method input video mapping differs for {method}")
        sources.append({
            "method": method,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "request_path": str(request_path.resolve()),
            "request_sha256": _sha256(request_path),
            "source_request_seed": source_seed,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "head_map_identity": head_map,
            "cases": len(cases),
            "shared_request_sha256": _mapping_sha256(current_shared_requests),
            "input_video_set_sha256": _case_video_set_sha256(case_videos),
        })
        for seed in normalized_seeds:
            for case_id in cases:
                root = output_root / method / f"seed-{seed}" / case_id
                tasks.append(MatrixTask(
                    method=method, case_id=case_id, seed=seed,
                    config_path=str(config_path), source_request_seed=source_seed,
                    case_root=str(root.resolve()),
                    production_source_sha256=production_source["sha256"],
                    checkpoint_path=str(checkpoint["path"]),
                    checkpoint_sha256=checkpoint["sha256"],
                    input_video_path=str(case_videos[case_id]["path"]),
                    input_video_bytes=int(case_videos[case_id]["bytes"]),
                    input_video_sha256=str(case_videos[case_id]["sha256"]),
                    head_map_path=(head_map or {}).get("path"),
                    head_map_bytes=(head_map or {}).get("bytes"),
                    head_map_sha256=(head_map or {}).get("sha256"),
                ))
    if len(checkpoints) != 1:
        raise ValueError("generation matrix must use one shared checkpoint")
    if len(head_maps) > 1:
        raise ValueError("heterogeneous methods must use one shared head map")
    return {
        "schema_version": MATRIX_PLAN_SCHEMA_VERSION,
        "protocol": shared_config["protocol"] if shared_config else None,
        "shared_config": shared_config,
        "shared_request_sha256": _mapping_sha256(reference_shared_requests or {}),
        "seeds": list(normalized_seeds),
        "methods": len(config_paths),
        "cases_per_method": len(reference_cases or ()),
        "tasks": len(tasks),
        "output_root": str(output_root.resolve()),
        "production_source": production_source,
        "checkpoints": [checkpoints[path] for path in sorted(checkpoints)],
        "head_maps": [head_maps[path] for path in sorted(head_maps)],
        "input_videos": [input_videos[path] for path in sorted(input_videos)],
        "sources": sources,
        "entries": [asdict(task) for task in tasks],
    }


def _successful_attempt(
    row: Mapping[str, Any],
    production_source: dict[str, Any],
    checkpoint: dict[str, Any],
) -> Path | None:
    input_video = {
        "path": row["input_video_path"],
        "bytes": row["input_video_bytes"],
        "sha256": row["input_video_sha256"],
    }
    return qualified_generation_attempt(
        row,
        production_source=production_source,
        checkpoint=checkpoint,
        input_video=input_video,
    )


def _next_attempt(case_root: Path) -> tuple[int, Path]:
    numbers = []
    for path in case_root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except ValueError:
            continue
    number = max(numbers, default=0) + 1
    return number, case_root / f"attempt-{number:03d}"


def run_matrix(
    plan: dict[str, Any],
    *,
    python: Path,
    cuda_visible_devices: str,
    source_root: Path,
    limit: int | None = None,
    methods: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    parse_cuda_visible_devices(cuda_visible_devices, expected_count=4)
    source = source_root.resolve()
    production_source = validate_matrix_plan_inputs(plan, source)
    allowed_methods = set(methods or ())
    allowed_cases = set(case_ids or ())
    selected = [
        row for row in plan["entries"]
        if (not allowed_methods or row["method"] in allowed_methods)
        and (not allowed_cases or row["case_id"] in allowed_cases)
    ]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:int(limit)]
    known_methods = {str(row["method"]) for row in plan["entries"]}
    known_cases = {str(row["case_id"]) for row in plan["entries"]}
    unknown_methods = sorted(allowed_methods - known_methods)
    unknown_cases = sorted(allowed_cases - known_cases)
    if unknown_methods:
        raise ValueError(f"unknown production matrix methods: {unknown_methods}")
    if unknown_cases:
        raise ValueError(f"unknown production matrix case IDs: {unknown_cases}")
    if not selected:
        raise ValueError("production matrix selection is empty")
    checkpoint_by_path = {
        str(Path(str(identity["path"])).resolve()): identity
        for identity in plan["checkpoints"]
    }
    base_environment = dict(os.environ)
    base_environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    existing_pythonpath = base_environment.get("PYTHONPATH")
    base_environment["PYTHONPATH"] = str(source / "src") + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    base_environment["CMBENCH_PRODUCTION_SOURCE_SHA256"] = str(production_source["sha256"])
    completed = skipped = failed = 0
    failures: list[dict[str, Any]] = []
    for row in selected:
        case_root = Path(row["case_root"])
        checkpoint = checkpoint_by_path[str(Path(row["checkpoint_path"]).resolve())]
        if _successful_attempt(row, production_source, checkpoint) is not None:
            skipped += 1
            continue
        require_idle_cuda_devices(cuda_visible_devices, expected_count=4)
        case_root.mkdir(parents=True, exist_ok=True)
        attempt_number, attempt = _next_attempt(case_root)
        log_path = case_root / f"attempt-{attempt_number:03d}.worker.log"
        command = [
            str(python.absolute()), "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=4", "-m", "cmbench_rebuild.production",
            row["config_path"], "--output", str(attempt),
            "--case-id", row["case_id"], "--seed", str(row["seed"]),
        ]
        environment = dict(base_environment)
        environment["CMBENCH_CHECKPOINT_IDENTITY_JSON"] = json.dumps(
            checkpoint, sort_keys=True, separators=(",", ":")
        )
        environment["CMBENCH_INPUT_VIDEO_IDENTITY_JSON"] = json.dumps(
            {
                "path": row["input_video_path"],
                "bytes": row["input_video_bytes"],
                "sha256": row["input_video_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if row.get("head_map_path") is not None:
            environment["CMBENCH_HEAD_MAP_IDENTITY_JSON"] = json.dumps(
                {
                    "path": row["head_map_path"],
                    "bytes": row["head_map_bytes"],
                    "sha256": row["head_map_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=source, env=environment, text=True, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode == 0 and _successful_attempt(row, production_source, checkpoint) == attempt:
            completed += 1
        else:
            failed += 1
            failures.append({"method": row["method"], "case_id": row["case_id"], "seed": row["seed"], "attempt": str(attempt), "returncode": process.returncode, "log": str(log_path)})
            if not continue_on_error:
                break
    return {"schema_version": 1, "selected": len(selected), "completed": completed, "skipped": skipped, "failed": failed, "failures": failures}
