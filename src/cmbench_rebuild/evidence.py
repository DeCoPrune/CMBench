"""Dependency-light, fail-closed qualification for generated case evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


REQUIRED_GENERATION_ARTIFACTS = frozenset({
    "context_plus_continuation_latents.pt",
    "continuation.mp4",
    "continuation_latents.pt",
    "metadata.json",
    "postprocess-input.json",
    "result.json",
    "selection_trajectory.json",
    "seq_pr_metrics.json",
    "token_trajectory.jsonl",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_seqpr(path: Path, *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Independently recompute seqPR totals and every declared step ratio."""
    data = _load_json(path)
    raw_steps = data.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    errors: list[str] = []
    if not steps:
        errors.append("seqPR contains no generation steps")

    def active_units(step: Mapping[str, Any]) -> int:
        return int(step.get("active_token_head_layer_units", step.get("active_token_head_units")))

    def dense_units(step: Mapping[str, Any]) -> int:
        return int(step.get("dense_token_head_layer_units", step.get("dense_token_head_units")))

    active = dense = 0
    seen_steps: set[tuple[int, int | None]] = set()
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            errors.append(f"step {index}: expected object")
            continue
        try:
            kept = active_units(raw_step)
            total = dense_units(raw_step)
        except (TypeError, ValueError):
            errors.append(f"step {index}: token totals are missing or invalid")
            continue
        active += kept
        dense += total
        if not 0 <= kept <= total:
            errors.append(f"step {index}: active units outside [0,dense]")
        expected = 1.0 - kept / total if total else None
        declared_step = raw_step.get("prune_ratio")
        try:
            ratio_matches = (
                (expected is None and declared_step is None)
                or (expected is not None and declared_step is not None and abs(expected - float(declared_step)) <= tolerance)
            )
        except (TypeError, ValueError):
            ratio_matches = False
        if not ratio_matches:
            errors.append(f"step {index}: prune ratio inconsistent")
        if raw_step.get("current_noisy_excluded_from_numerator") is not True:
            errors.append(f"step {index}: current noisy tokens are not excluded from numerator")
        if raw_step.get("current_noisy_excluded_from_denominator") is not True:
            errors.append(f"step {index}: current noisy tokens are not excluded from denominator")
        chunk = raw_step.get("chunk_idx", raw_step.get("chunk_index"))
        layer = raw_step.get("layer_idx")
        try:
            identity = (int(chunk), int(layer) if layer is not None else None)
        except (TypeError, ValueError):
            errors.append(f"step {index}: chunk/layer identity is invalid")
        else:
            if identity in seen_steps:
                errors.append(f"step {index}: duplicate chunk/layer identity {identity}")
            seen_steps.add(identity)

    computed = 1.0 - active / dense if dense else None
    declared = data.get("seqPR")
    if active != data.get("active_token_head_layer_units"):
        errors.append("active total does not equal sum of steps")
    if dense != data.get("dense_token_head_layer_units"):
        errors.append("dense total does not equal sum of steps")
    try:
        summary_matches = computed is not None and declared is not None and abs(computed - float(declared)) <= tolerance
    except (TypeError, ValueError):
        summary_matches = False
    if not summary_matches:
        errors.append("declared seqPR does not equal 1-active/dense")
    return {
        "schema_version": 1,
        "source": str(path),
        "passed": not errors,
        "declared_seqpr": declared,
        "computed_seqpr": computed,
        "generation_steps": len(steps),
        "errors": errors,
    }


def audit_mask_accounting(path: Path, *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Check every token-ledger conservation equation and require layer evidence."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if not rows:
        errors.append("token trajectory contains no events")
    seen_layers: set[tuple[int, int]] = set()
    layer_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: expected object")
            continue
        try:
            if row.get("metric") == "per_layer_token_trajectory":
                total = int(row["dense_token_head_units"])
                kept = int(row["active_token_head_units"])
                masked = int(row["masked_token_head_units"])
                layer_rows.append(row)
                identity = (int(row["chunk_idx"]), int(row["layer_idx"]))
                if identity in seen_layers:
                    errors.append(f"row {index}: duplicate chunk/layer identity {identity}")
                seen_layers.add(identity)
            else:
                total = int(row["total_token_head_units"])
                kept = int(row["kept_token_head_units"])
                masked = int(row["masked_token_head_units"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"row {index}: token totals or identity are missing or invalid")
            continue
        if total < 0 or kept < 0 or masked < 0:
            errors.append(f"row {index}: token counts must be non-negative")
        if kept + masked != total:
            errors.append(f"row {index}: kept + masked != total")
        expected = masked / total if total else None
        try:
            ratio_matches = expected is not None and abs(expected - float(row["prune_ratio"])) <= tolerance
        except (KeyError, TypeError, ValueError):
            ratio_matches = False
        if not ratio_matches:
            errors.append(f"row {index}: prune ratio inconsistent")
    return {
        "schema_version": 1,
        "source": str(path),
        "passed": not errors,
        "events": len(rows),
        "per_layer_events": len(layer_rows),
        "errors": errors,
    }


def artifact_manifest_valid(attempt: Path, row: Mapping[str, Any]) -> bool:
    """Validate identity and every content hash in a case artifact manifest."""
    try:
        root = attempt.resolve()
        manifest = _load_json(root / "artifacts.manifest.json")
        files = manifest["files"]
        if not isinstance(files, dict) or not REQUIRED_GENERATION_ARTIFACTS.issubset(files):
            return False
        if (
            manifest.get("schema_version") != 1
            or str(manifest.get("method")) != str(row["method"])
            or str(manifest.get("case_id")) != str(row["case_id"])
            or int(manifest.get("seed", -1)) != int(row["seed"])
            or str(manifest.get("production_source_sha256")) != str(row["production_source_sha256"])
            or str(manifest.get("checkpoint_sha256")) != str(row["checkpoint_sha256"])
            or str(manifest.get("input_video_sha256")) != str(row["input_video_sha256"])
            or manifest.get("head_map_sha256") != row.get("head_map_sha256")
        ):
            return False
        for name, identity in files.items():
            if not isinstance(name, str) or not name or not isinstance(identity, dict):
                return False
            path = (root / name).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return False
            if (
                not path.is_file()
                or int(identity["bytes"]) != path.stat().st_size
                or str(identity["sha256"]) != _sha256_file(path)
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _case_root(row: Mapping[str, Any]) -> Path:
    value = row.get("generation_case_root", row.get("case_root"))
    if not isinstance(value, str) or not value:
        raise ValueError("generation row has no case root")
    return Path(value)


def _artifact_attempt_valid(
    attempt: Path,
    row: Mapping[str, Any],
    *,
    production_source: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    input_video: Mapping[str, Any] | None = None,
) -> bool:
    try:
        result = _load_json(attempt / "result.json")
        metadata = _load_json(attempt / "metadata.json")
        expected_head_map = None
        if row.get("head_map_path") is not None:
            expected_head_map = {
                "path": row.get("head_map_path"),
                "bytes": row.get("head_map_bytes"),
                "sha256": row.get("head_map_sha256"),
            }
        if (
            result.get("schema_version") != 1
            or metadata.get("schema_version") != 1
            or result.get("status") != "completed"
            or bool(result.get("disqualified"))
        ):
            return False
        if bool(metadata.get("diagnostic_disqualified")):
            return False
        if (
            str((result.get("production_source") or {}).get("sha256"))
            != str(row.get("production_source_sha256"))
            or str(metadata.get("production_source_sha256"))
            != str(row.get("production_source_sha256"))
            or str((result.get("checkpoint_identity") or {}).get("sha256"))
            != str(row.get("checkpoint_sha256"))
            or str((result.get("checkpoint_identity") or {}).get("path"))
            != str(row.get("checkpoint_path"))
            or str(metadata.get("checkpoint_sha256")) != str(row.get("checkpoint_sha256"))
            or str((result.get("input_video_identity") or {}).get("path"))
            != str(row.get("input_video_path"))
            or int((result.get("input_video_identity") or {}).get("bytes", -1))
            != int(row.get("input_video_bytes", -1))
            or str((result.get("input_video_identity") or {}).get("sha256"))
            != str(row.get("input_video_sha256"))
            or str(metadata.get("input_video_sha256")) != str(row.get("input_video_sha256"))
            or result.get("head_map_identity") != expected_head_map
            or metadata.get("head_map_sha256") != row.get("head_map_sha256")
            or str(metadata.get("method")) != str(row["method"])
            or str(metadata.get("case_id")) != str(row["case_id"])
            or int(metadata.get("seed", -1)) != int(row["seed"])
        ):
            return False
        if production_source is not None and result.get("production_source") != dict(production_source):
            return False
        if checkpoint is not None and result.get("checkpoint_identity") != dict(checkpoint):
            return False
        if input_video is not None and result.get("input_video_identity") != dict(input_video):
            return False
        return artifact_manifest_valid(attempt, row)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _artifact_qualified_attempts(
    row: Mapping[str, Any],
    *,
    production_source: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    input_video: Mapping[str, Any] | None = None,
) -> Iterator[Path]:
    for attempt in sorted(path for path in _case_root(row).glob("attempt-*") if path.is_dir()):
        if _artifact_attempt_valid(
            attempt,
            row,
            production_source=production_source,
            checkpoint=checkpoint,
            input_video=input_video,
        ):
            yield attempt


def artifact_qualified_generation_attempt(
    row: Mapping[str, Any],
    *,
    production_source: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    input_video: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the first identity- and manifest-qualified attempt, if any."""
    return next(
        _artifact_qualified_attempts(
            row,
            production_source=production_source,
            checkpoint=checkpoint,
            input_video=input_video,
        ),
        None,
    )


def qualified_generation_attempt(
    row: Mapping[str, Any],
    *,
    production_source: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    input_video: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the first attempt that also passes both independent ledgers."""
    for attempt in _artifact_qualified_attempts(
        row,
        production_source=production_source,
        checkpoint=checkpoint,
        input_video=input_video,
    ):
        if generation_attempt_qualified(
            attempt,
            row,
            production_source=production_source,
            checkpoint=checkpoint,
            input_video=input_video,
        ):
            return attempt
    return None


def generation_attempt_qualified(
    attempt: Path,
    row: Mapping[str, Any],
    *,
    production_source: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    input_video: Mapping[str, Any] | None = None,
) -> bool:
    """Qualify one explicit attempt, including standalone representative runs."""
    if not _artifact_attempt_valid(
        attempt.resolve(),
        row,
        production_source=production_source,
        checkpoint=checkpoint,
        input_video=input_video,
    ):
        return False
    try:
        seqpr = audit_seqpr(attempt / "seq_pr_metrics.json")
        accounting = audit_mask_accounting(attempt / "token_trajectory.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        seqpr["passed"]
        and accounting["passed"]
        and int(accounting["per_layer_events"]) > 0
    )
