"""Deterministic video resampling and LingBot context preparation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CaseRequest, PreparedContext


def nearest_fps_indices(*, source_frames: int, source_fps: float, target_fps: float) -> tuple[int, ...]:
    frames, source_rate, target_rate = int(source_frames), float(source_fps), float(target_fps)
    if frames <= 0 or source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_frames and frame rates must be positive")
    target_frames = int((frames - 1) * target_rate / source_rate) + 1
    return tuple(min(frames - 1, max(0, round(index * source_rate / target_rate))) for index in range(target_frames))


def aligned_context_timeline(*, input_pixel_frames: int, chunk_size: int, vae_temporal_stride: int = 4) -> dict[str, int]:
    pixels, chunk, stride = int(input_pixel_frames), int(chunk_size), int(vae_temporal_stride)
    if pixels <= 0 or chunk <= 0 or stride <= 0:
        raise ValueError("pixel frames, chunk size and VAE stride must be positive")
    # One causal anchor plus enough stride cells to contain every observation.
    predicted_latents = (pixels + (2 * stride - 2)) // stride
    aligned_latents = ((predicted_latents + chunk - 1) // chunk) * chunk
    aligned_pixels = (aligned_latents - 1) * stride + 1
    if aligned_pixels < pixels:
        raise AssertionError("context alignment rounded the observed video down")
    return {
        "input_pixel_frames": pixels,
        "predicted_context_latents": predicted_latents,
        "aligned_context_latents": aligned_latents,
        "aligned_context_pixel_frames": aligned_pixels,
        "context_alignment_padding_frames": aligned_pixels - pixels,
    }


@dataclass
class TorchVideoContextProvider:
    height: int = 480
    width: int = 832
    target_fps: float = 16.0
    chunk_size: int = 4
    vae_temporal_stride: int = 4
    resize_batch_frames: int = 32
    device: str = "cuda"

    def prepare(self, request: CaseRequest) -> PreparedContext:
        import cv2
        import torch
        import torch.distributed as dist
        import torch.nn.functional as functional

        path = Path(request.clip_file)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        payload: dict[str, Any] | None = None
        video = None
        if rank == 0:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"could not decode {path}")
            source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or self.target_fps)
            indices = nearest_fps_indices(source_frames=source_frames, source_fps=source_fps, target_fps=self.target_fps)
            cursor = 0
            pending, batches = [], []

            def flush() -> None:
                if not pending:
                    return
                batch = torch.stack(pending).to(device=self.device, dtype=torch.float32) / 127.5 - 1.0
                batches.append(functional.interpolate(batch, size=(int(self.height), int(self.width)), mode="bilinear", align_corners=False).cpu())
                pending.clear()

            try:
                for source_index in range(source_frames):
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break
                    while cursor < len(indices) and indices[cursor] == source_index:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pending.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
                        cursor += 1
                        if len(pending) >= int(self.resize_batch_frames):
                            flush()
                    if cursor == len(indices):
                        break
            finally:
                capture.release()
            flush()
            if cursor != len(indices):
                raise RuntimeError(f"decoded only {cursor}/{len(indices)} selected frames from {path}")
            video = torch.cat(batches).permute(1, 0, 2, 3).contiguous()
            alignment = aligned_context_timeline(input_pixel_frames=int(video.shape[1]), chunk_size=self.chunk_size, vae_temporal_stride=self.vae_temporal_stride)
            padding = alignment["context_alignment_padding_frames"]
            if padding:
                video = torch.cat([video, video[:, -1:].repeat(1, padding, 1, 1)], dim=1)
            diagnostics = {
                "height": int(self.height),
                "width": int(self.width),
                "source_path": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_fps": source_fps,
                "source_pixel_frames": source_frames,
                "model_fps": float(self.target_fps),
                "resampled_pixel_frames": len(indices),
                "first_source_frame_index": indices[0],
                "last_source_frame_index": indices[-1],
                "unique_source_frames": len(set(indices)),
                "decoder": "opencv_streaming_selected_frames",
                "resize": "torch_bilinear_align_corners_false",
                "decode_rank": 0,
                **alignment,
            }
            payload = {"shape": tuple(int(value) for value in video.shape), "diagnostics": diagnostics}
        if dist.is_available() and dist.is_initialized():
            values = [payload]
            dist.broadcast_object_list(values, src=0)
            payload = values[0]
        if payload is None or (rank == 0 and video is None):
            raise RuntimeError("context preparation produced no payload")
        return PreparedContext(
            payload=video,
            pixel_frames=int(payload["shape"][1]),
            latent_frames=int(payload["diagnostics"]["aligned_context_latents"]),
            diagnostics=payload["diagnostics"],
        )
