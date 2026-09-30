"""Load the public CMBench release and adapt it to generation/evaluation inputs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    return rows


def task_id(row: Mapping[str, Any]) -> str:
    return str(row.get("task_id") or row.get("case_id") or "")


def normalized_box(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy_normalized must have four coordinates")
    box = tuple(float(x) for x in value)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
        raise ValueError("normalized box coordinates must be finite and in [0, 1]")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("reference box must have positive area")
    return box


def public_video_path(root: Path, row: Mapping[str, Any]) -> Path:
    relative = Path(str(row.get("video") or ""))
    if not row.get("video") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{task_id(row)}: video must be relative to the benchmark root")
    path = (Path(root) / relative).resolve()
    # Hugging Face snapshots may contain symlinks to their content-addressed cache.
    if not path.is_file():
        raise FileNotFoundError(f"{task_id(row)}: missing context video: {path}")
    return path


def load_benchmark(root: Path, metadata: Path | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(metadata or Path(root) / "metadata.jsonl")
    seen = set()
    for row in rows:
        identifier = task_id(row)
        if not identifier or identifier in seen:
            raise ValueError(f"missing or duplicate task_id: {identifier!r}")
        seen.add(identifier)
        if row.get("type") not in {"synthetic", "real"} or row.get("task_type") not in {"reappear", "revisit"}:
            raise ValueError(f"{identifier}: invalid type/task_type")
        public_video_path(root, row)
        clips = row.get("context_clip_prompts")
        if not isinstance(clips, list) or len(clips) != 6 or any(not c.get("clip_prompt") for c in clips):
            raise ValueError(f"{identifier}: six clip descriptions are required")
        if not row.get("continue_prompt") or not row.get("target_label"):
            raise ValueError(f"{identifier}: continuation prompt and target label are required")
        if not row.get("references"):
            raise ValueError(f"{identifier}: references are required")
        for reference in row["references"]:
            normalized_box(reference.get("bbox_xyxy_normalized"))
            stamp = float(reference["timestamp_seconds"])
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError(f"{identifier}: invalid reference timestamp")
            frame = reference.get("frame_index")
            if frame is not None and (type(frame) is not int or frame < 0):
                raise ValueError(f"{identifier}: invalid reference frame index")
    return rows


def prepare_requests(root: Path, output: Path, seeds: Sequence[int], *,
                     metadata: Path | None = None, output_latent_frames: int = 16) -> dict[str, Any]:
    rows = load_benchmark(root, metadata)
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) < 0 for s in seeds):
        raise ValueError("seeds must be non-negative and unique")
    if output_latent_frames <= 0:
        raise ValueError("output_latent_frames must be positive")
    requests = []
    for row in rows:
        for seed in seeds:
            requests.append({
                **row,
                "case_id": row["task_id"],
                "seed": int(seed),
                "clip_file": str(public_video_path(root, row)),
                "prompt": row["continue_prompt"],
                "num_output_latent_frames": int(output_latent_frames),
            })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in requests)
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"request file differs; choose a new output path: {output}")
    output.write_text(encoded, encoding="utf-8")
    return {"tasks": len(rows), "requests": len(requests), "seeds": list(seeds), "output": str(output.resolve())}


def pixel_box(box: Sequence[float], width: int, height: int) -> list[int]:
    values = [x * size for x, size in zip(normalized_box(box), (width, height, width, height))]
    # Do not expand an exact pixel box because of floating-point roundoff.
    values = [round(x) if abs(x - round(x)) < 1e-8 else x for x in values]
    return [math.floor(values[0]), math.floor(values[1]), math.ceil(values[2]), math.ceil(values[3])]


def decode_reference(video: Path, reference: Mapping[str, Any]) -> tuple[Any, int, list[int]]:
    """Honor explicit frame indices; seek timestamp annotations without assuming FPS."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open reference video: {video}")
        index = reference.get("frame_index")
        if index is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(reference["timestamp_seconds"]) * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot decode reference in {video}: {reference}")
        decoded_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if decoded_index < 0 or (index is not None and decoded_index != index):
            raise ValueError(f"reference frame seek mismatch: {video}")
        height, width = frame.shape[:2]
        return frame, decoded_index, pixel_box(reference["bbox_xyxy_normalized"], width, height)
    finally:
        capture.release()


def evaluator_inputs(root: Path, metadata: Path, annotations: Path, destination: Path,
                     case_ids: Sequence[str]) -> tuple[Path, Path]:
    """Materialize pixel-box inputs for the frozen external OWL/SAM/DINO evaluator."""
    metadata_rows, annotation_rows = read_jsonl(metadata), read_jsonl(annotations)
    if not any("task_id" in r and "video" in r for r in metadata_rows + annotation_rows):
        return metadata, annotations
    converted = []
    for path in (metadata, annotations):
        rows = load_benchmark(root, path)
        selected = [r for r in rows if r["task_id"] in set(case_ids)]
        if {r["task_id"] for r in selected} != set(case_ids):
            raise ValueError("benchmark metadata does not cover all evaluation tasks")
        compatible = []
        for row in selected:
            video = public_video_path(root, row)
            refs = []
            for offset, ref in enumerate(row["references"]):
                _, frame_index, box = decode_reference(video, ref)
                refs.append({"ref_id": f"ref_{offset:02d}", "frame": frame_index,
                             "bbox_xyxy": box, "timestamp_seconds": ref["timestamp_seconds"]})
            compatible.append({**row, "case_id": row["task_id"], "clip_file": str(video),
                               "video_file": str(video), "prompt": row["continue_prompt"],
                               "memory_level": "object" if row["task_type"] == "reappear" else "scene",
                               "references": refs})
        converted.append(compatible)
    destination.mkdir(parents=True, exist_ok=True)
    paths = (destination / "metadata.jsonl", destination / "annotations.jsonl")
    for path, rows in zip(paths, converted):
        encoded = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"frozen evaluator input changed: {path}")
        path.write_text(encoded, encoding="utf-8")
    return paths
