"""Controlled adapter for the canonical CMBench OWL/SAM/DINO evaluator."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.manifest import sha256_file
from ..artifacts.video import inspect_video
from ..identity import directory_content_identity
from ..gpu import require_idle_cuda_devices
from ..dataset import task_id, public_video_path, evaluator_inputs


_METHOD_ALIASES = {
    "fullkv": "full_kv",
    "decoprune": "consistency_prune",
    "decoprune_hs": "consistency_prune_hs",
    "dummy_forcing": "dummy_forcing",
    "patchification": "patchification",
}


def _jsonl_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        case_id = task_id(row)
        if case_id:
            if "task_id" in row and case_id in latest:
                raise ValueError(f"duplicate public task_id: {case_id}")
            latest[case_id] = row
    return latest


def resolve_reference_videos(
    *, benchmark_root: Path, metadata: Path, annotations: Path, case_ids: Sequence[str]
) -> dict[str, Path]:
    """Resolve the exact source clip selected by the frozen evaluator."""
    metadata_rows = _jsonl_latest(metadata)
    metadata_by_key = dict(metadata_rows)
    for row in metadata_rows.values():
        source_case_id = str(row.get("source_case_id") or "")
        if source_case_id:
            metadata_by_key[source_case_id] = row
    annotation_rows = _jsonl_latest(annotations)
    resolved: dict[str, Path] = {}
    for raw_case_id in case_ids:
        case_id = str(raw_case_id)
        annotation = annotation_rows.get(case_id)
        if annotation is None:
            raise ValueError(f"official annotation missing planned case: {case_id}")
        metadata_row = metadata_by_key.get(case_id, {})
        public_row = annotation if annotation.get("video") else metadata_row
        if public_row.get("video"):
            resolved[case_id] = public_video_path(benchmark_root, public_row)
            continue
        annotation_clip = str(
            annotation.get("clip_file") or annotation.get("clip_path") or annotation.get("video_file") or ""
        ).strip()
        metadata_clip = str(
            metadata_row.get("clip_file") or metadata_row.get("clip_path") or metadata_row.get("video_file") or ""
        ).strip()
        if annotation_clip and not Path(annotation_clip).is_absolute() and Path(metadata_clip).is_absolute():
            annotation_clip = metadata_clip
        selected = annotation_clip or metadata_clip
        if not selected:
            raise ValueError(f"official source clip missing from metadata/annotation: {case_id}")
        source = Path(selected)
        if not source.is_absolute():
            source = benchmark_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"official source clip does not exist for {case_id}: {source}")
        resolved[case_id] = source
    return resolved


def _model_identity(path: Path) -> dict[str, Any]:
    model = path.resolve()
    config = model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"model config does not exist: {config}")
    return directory_content_identity(model)


def run_official_dino(
    *,
    case_dir: Path,
    benchmark_root: Path,
    metadata: Path,
    annotations: Path,
    evaluator: Path,
    python: Path,
    owl_model: Path,
    sam_model: Path,
    dino_model: Path,
    eval_root: Path,
    device: str = "cuda:0",
    cuda_visible_devices: str | None = None,
    evaluation_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and verify the canonical localization evaluator for exactly one case."""
    results = run_official_dino_batch(
        case_dirs=[case_dir],
        benchmark_root=benchmark_root,
        metadata=metadata,
        annotations=annotations,
        evaluator=evaluator,
        python=python,
        owl_model=owl_model,
        sam_model=sam_model,
        dino_model=dino_model,
        eval_root=eval_root,
        device=device,
        cuda_visible_devices=cuda_visible_devices,
        evaluation_source=evaluation_source,
    )
    result = next(iter(results.values()))
    destination = eval_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "official_dino.provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_official_dino_batch(
    *,
    case_dirs: Sequence[Path],
    benchmark_root: Path,
    metadata: Path,
    annotations: Path,
    evaluator: Path,
    python: Path,
    owl_model: Path,
    sam_model: Path,
    dino_model: Path,
    eval_root: Path,
    device: str = "cuda:0",
    cuda_visible_devices: str | None = None,
    evaluation_source: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run one model load for a batch of uniquely named generated cases."""
    if not case_dirs:
        raise ValueError("official DINO batch requires at least one case")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_case_dir in case_dirs:
        case_root = raw_case_dir.resolve()
        case_metadata_path = case_root / "metadata.json"
        video_path = case_root / "continuation.mp4"
        if not case_metadata_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(f"case artifacts missing: {case_root}")
        case_metadata = json.loads(case_metadata_path.read_text(encoding="utf-8"))
        case_id = str(case_metadata.get("case_id") or "")
        method = str(case_metadata.get("method") or "")
        if not case_id or not method:
            raise ValueError("case metadata must contain case_id and method")
        if case_id in seen:
            raise ValueError(f"official DINO batch has duplicate case_id: {case_id}")
        seen.add(case_id)
        cases.append({
            "root": case_root,
            "metadata": case_metadata,
            "case_id": case_id,
            "method": method,
            "official_method": _METHOD_ALIASES.get(method, method),
            "video": video_path,
        })

    required = (metadata, annotations, evaluator, python)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"official DINO inputs missing: {missing}")
    destination = eval_root.resolve()
    evaluator_metadata, evaluator_annotations = evaluator_inputs(
        benchmark_root, metadata, annotations, destination / "benchmark-inputs",
        [case["case_id"] for case in cases],
    )
    # The frozen evaluator filters legacy method tags in per-video metadata.
    evaluator_roots = []
    for case in cases:
        if case["method"] == case["official_method"]:
            evaluator_roots.append(case["root"])
            continue
        view = destination / "generated-inputs" / case["case_id"]
        view.mkdir(parents=True, exist_ok=True)
        adapted_metadata = {**case["metadata"], "method": case["official_method"]}
        encoded = json.dumps(adapted_metadata, indent=2, sort_keys=True) + "\n"
        meta_path = view / "metadata.json"
        if meta_path.exists() and meta_path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"frozen generated metadata changed: {meta_path}")
        meta_path.write_text(encoded, encoding="utf-8")
        link = view / "continuation.mp4"
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != case["video"].resolve():
                raise ValueError(f"generated input link mismatch: {link}")
        else:
            link.symlink_to(case["video"].resolve())
        evaluator_roots.append(view)
    official_methods = sorted({case["official_method"] for case in cases})
    command = [
        # Keep the virtual-environment interpreter path intact.  Resolving its
        # symlink would invoke the base interpreter without the venv packages.
        str(python.absolute()), str(evaluator.resolve()),
        "--benchmark-root", str(benchmark_root.resolve()),
        "--metadata", str(evaluator_metadata.resolve()),
        "--annotations", str(evaluator_annotations.resolve()),
        "--output-roots", *[str(path) for path in evaluator_roots],
        "--eval-root", str(destination),
        "--methods", *official_methods,
        "--cases", *[case["case_id"] for case in cases],
        "--video-name", "continuation.mp4",
        "--owl-model", str(owl_model.resolve()),
        "--sam-model", str(sam_model.resolve()),
        "--dino-model", str(dino_model.resolve()),
        "--device", device,
        "--mask-mode", "sam",
        "--scoring-path", "owl_sam_dino",
        "--frame-stride", "1",
        "--no-overlays", "--no-comparison-videos",
    ]
    evaluator_environment = dict(os.environ)
    # The adapter itself is commonly launched with PYTHONPATH=src.  Do not
    # leak that harness-only path into a frozen evaluator environment, whose
    # interpreter owns its own model dependencies.
    evaluator_environment.pop("PYTHONPATH", None)
    if cuda_visible_devices is not None:
        require_idle_cuda_devices(cuda_visible_devices, expected_count=1)
        evaluator_environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        cwd=evaluator.resolve().parent.parent,
        env=evaluator_environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"official DINO evaluator failed with exit code {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    summary_path = destination / "video_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"official evaluator did not write {summary_path}")
    rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    common = {
        "benchmark": {
            "root": str(benchmark_root.resolve()),
            "metadata": {"path": str(metadata.resolve()), "sha256": sha256_file(metadata)},
            "annotations": {"path": str(annotations.resolve()), "sha256": sha256_file(annotations)},
        },
        "evaluator": {"path": str(evaluator.resolve()), "sha256": sha256_file(evaluator)},
        "models": {
            "owl": _model_identity(owl_model),
            "sam": _model_identity(sam_model),
            "dino": _model_identity(dino_model),
        },
        "command": command,
        "cuda_visible_devices": cuda_visible_devices,
        "summary_jsonl": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
    }
    results: dict[str, dict[str, Any]] = {}
    reference_videos = resolve_reference_videos(
        benchmark_root=benchmark_root,
        metadata=metadata,
        annotations=annotations,
        case_ids=[case["case_id"] for case in cases],
    )
    for case in cases:
        case_id = case["case_id"]
        matches = [row for row in rows if str(row.get("selected_case_id")) == case_id]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one official DINO row for {case_id}, found {len(matches)}")
        row = matches[0]
        if str(row.get("scoring_path")) != "owl_sam_dino":
            raise ValueError("official DINO result did not use owl_sam_dino")
        references = row.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError("official DINO result has no structured references")
        if any(bool(reference.get("used_direct_dino")) for reference in references):
            raise ValueError("official DINO result used forbidden direct-DINO fallback")
        score = float(row["dino_score"])
        if not math.isfinite(score):
            raise ValueError("official DINO score is non-finite")
        video_path = case["video"]
        decoded_frames = int(inspect_video(video_path)["decoded_frame_count"])
        if int(row.get("num_eval_frames", -1)) != decoded_frames:
            raise ValueError(
                f"official DINO did not score every decoded frame: {row.get('num_eval_frames')} != {decoded_frames}"
            )
        result = {
            "schema_version": 1,
            "metric": "cmbench_owl_sam_dino",
            "official_compatible": True,
            "case_id": case_id,
            "method": case["method"],
            "memory_level": row.get("memory_level"),
            "dino_score": score,
            "scoring_path": "owl_sam_dino",
            "mask_mode": "sam",
            "frame_stride": 1,
            "num_eval_frames": decoded_frames,
            "generated_video": {"path": str(video_path), "sha256": sha256_file(video_path)},
            "reference_video": {
                "path": str(reference_videos[case_id]),
                "bytes": reference_videos[case_id].stat().st_size,
                "sha256": sha256_file(reference_videos[case_id]),
            },
            **common,
            "summary": row,
        }
        if evaluation_source is not None:
            result["evaluation_source"] = dict(evaluation_source)
        results[case_id] = result
    return results
