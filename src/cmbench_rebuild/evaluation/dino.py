"""Independent DINOv2 comparison for corresponding decoded video frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_frames(path: Path) -> list[Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"no decoded frames: {path}")
    return frames


def _embeddings(images: list[Any], processor: Any, model: Any, device: Any, batch_size: int) -> Any:
    import torch

    chunks = []
    for offset in range(0, len(images), batch_size):
        inputs = processor(images=images[offset : offset + batch_size], return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state[:, 0]
        chunks.append(torch.nn.functional.normalize(values, dim=-1).cpu())
    return torch.cat(chunks)


def compare_dino(expected: Path, actual: Path, *, model_path: Path, device: str = "cuda", batch_size: int = 8) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    left = _read_frames(expected)
    right = _read_frames(actual)
    if len(left) != len(right):
        raise ValueError(f"frame count differs: {len(left)} != {len(right)}")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(torch.device(device))
    left_features = _embeddings(left, processor, model, torch.device(device), batch_size)
    right_features = _embeddings(right, processor, model, torch.device(device), batch_size)
    scores = (left_features * right_features).sum(dim=-1).tolist()
    config_path = model_path / "config.json"
    return {
        "schema_version": 1,
        "metric": "corresponding_frame_dinov2_cls_cosine",
        "expected": str(expected),
        "actual": str(actual),
        "model": str(model_path),
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": device,
        "frames": len(scores),
        "mean": sum(scores) / len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "per_frame": scores,
    }
