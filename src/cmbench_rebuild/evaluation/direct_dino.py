"""Auditable direct-bbox DINOv2 ablation for one CMBench case.

The official benchmark path localizes an object before feature comparison.
This cheaper fixed-location path is useful as independent evidence, especially
for near-full-frame scene references, but must not be reported as the official
OWL/SAM/DINO score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.manifest import sha256_file
from .cmbench import normalize_memory_level
from ..dataset import task_id, decode_reference


def scale_box_xyxy(box: Sequence[object], source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"box must contain four coordinates: {box!r}")
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("frame dimensions must be positive")
    x1, y1, x2, y2 = (float(value) for value in box)
    values = (
        round(x1 * target_width / source_width),
        round(y1 * target_height / source_height),
        round(x2 * target_width / source_width),
        round(y2 * target_height / source_height),
    )
    left = max(0, min(target_width - 1, values[0]))
    top = max(0, min(target_height - 1, values[1]))
    right = max(left + 1, min(target_width, values[2]))
    bottom = max(top + 1, min(target_height, values[3]))
    return left, top, right, bottom


def reference_best_summary(reference_rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    if not reference_rows:
        raise ValueError("at least one reference is required")
    best_scores: list[float] = []
    summaries: list[dict[str, Any]] = []
    for row in reference_rows:
        scores = [float(value) for value in row["per_frame_scores"]]
        frame_indices = [int(value) for value in row["generated_frame_indices"]]
        if not scores or len(scores) != len(frame_indices):
            raise ValueError("each reference needs equally sized non-empty scores and frame indices")
        best_offset = max(range(len(scores)), key=scores.__getitem__)
        best_scores.append(scores[best_offset])
        summaries.append(
            {
                "ref_id": row.get("ref_id"),
                "best_score": scores[best_offset],
                "best_generated_frame": frame_indices[best_offset],
            }
        )
    return {
        "dino_score": sum(best_scores) / len(best_scores),
        "dino_ref_best_mean": sum(best_scores) / len(best_scores),
        "dino_ref_best_min": min(best_scores),
        "reference_best": summaries,
    }


def _read_annotation(path: Path, case_id: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if task_id(row) == case_id:
                found.append(row)
    if len(found) != 1:
        raise ValueError(f"expected exactly one annotation for {case_id!r}, found {len(found)}")
    if found[0].get("discard") or str(found[0].get("status") or "ok").lower() in {"skip", "discard"}:
        raise ValueError(f"case is not eligible: {case_id}")
    return found[0]


def _video_frame(path: Path, frame_index: int) -> Any:
    import cv2

    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode frame {frame_index} from {path}")
    return frame


def _sample_video(path: Path, stride: int) -> tuple[list[int], list[Any]]:
    import cv2

    if stride <= 0:
        raise ValueError("frame_stride must be positive")
    capture = cv2.VideoCapture(str(path))
    indices: list[int] = []
    frames: list[Any] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            indices.append(index)
            frames.append(frame)
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"no decoded frames: {path}")
    return indices, frames


def _crop_rgb(frame: Any, box: tuple[int, int, int, int]) -> Any:
    import cv2

    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"empty crop for box {box}")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _features(images: Sequence[Any], processor: Any, model: Any, device: Any, batch_size: int) -> Any:
    import torch

    values = []
    for offset in range(0, len(images), batch_size):
        inputs = processor(images=list(images[offset : offset + batch_size]), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            feature = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        values.append(torch.nn.functional.normalize(feature.float(), dim=-1).cpu())
    return torch.cat(values)


def evaluate_direct_bbox_dino(
    *,
    annotation_path: Path,
    case_id: str,
    generated_video: Path,
    model_path: Path,
    frame_stride: int = 4,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    annotation = _read_annotation(annotation_path, case_id)
    source_video = Path(str(annotation.get("clip_file") or annotation.get("video_file") or ""))
    if not source_video.is_file():
        source_video = annotation_path.parent / source_video
    if not source_video.is_file():
        raise FileNotFoundError(f"source video does not exist: {source_video}")
    references = annotation.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError(f"case has no references: {case_id}")

    generated_indices, generated_frames = _sample_video(generated_video, frame_stride)
    target_height, target_width = generated_frames[0].shape[:2]
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    torch_device = torch.device(device)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(torch_device)
    reference_rows: list[dict[str, Any]] = []
    for offset, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"invalid reference at offset {offset}")
        if "bbox_xyxy_normalized" in reference:
            source_frame, frame_index, box = decode_reference(source_video, reference)
        else:
            frame_index = int(reference.get("frame") if reference.get("frame") is not None else reference.get("ref_frame"))
            box = reference.get("bbox_xyxy") or reference.get("ref_box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"reference has no valid bbox: {reference!r}")
            source_frame = _video_frame(source_video, frame_index)
        source_height, source_width = source_frame.shape[:2]
        source_box = scale_box_xyxy(box, (source_width, source_height), (source_width, source_height))
        generated_box = scale_box_xyxy(box, (source_width, source_height), (target_width, target_height))
        images = [_crop_rgb(source_frame, source_box)] + [_crop_rgb(frame, generated_box) for frame in generated_frames]
        features = _features(images, processor, model, torch_device, batch_size)
        scores = (features[0:1] * features[1:]).sum(dim=-1).tolist()
        reference_rows.append(
            {
                "ref_id": reference.get("ref_id") or f"ref_{offset:02d}",
                "reference_frame": frame_index,
                "reference_box_xyxy": list(source_box),
                "generated_box_xyxy": list(generated_box),
                "generated_frame_indices": generated_indices,
                "per_frame_scores": scores,
            }
        )
    summary = reference_best_summary(reference_rows)
    config_path = model_path / "config.json"
    return {
        "schema_version": 1,
        "metric": "direct_bbox_dinov2_reference_best_ablation",
        "official_compatible": False,
        "qualification_note": "independent evidence only; official benchmark uses the configured OWL/SAM/DINO localization path",
        "case_id": case_id,
        "memory_level": normalize_memory_level(annotation.get("memory_level")),
        "frame_stride": frame_stride,
        "feature": "pooler_output_else_cls_l2_normalized_cosine",
        "annotation": {"path": str(annotation_path), "sha256": sha256_file(annotation_path)},
        "source_video": {"path": str(source_video), "sha256": sha256_file(source_video)},
        "generated_video": {"path": str(generated_video), "sha256": sha256_file(generated_video)},
        "model": {"path": str(model_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "references": reference_rows,
        **summary,
    }
