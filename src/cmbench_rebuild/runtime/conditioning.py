"""Causal-VAE encoding and official 20-channel I2V conditioning contract."""
from __future__ import annotations

from typing import Any


def _distributed() -> tuple[Any, int]:
    import torch.distributed as dist

    initialized = dist.is_available() and dist.is_initialized()
    return dist, dist.get_rank() if initialized else 0


def broadcast_tensor_from_rank0(value: Any | None, *, device: Any, dtype: Any) -> Any:
    import torch

    dist, rank = _distributed()
    if not (dist.is_available() and dist.is_initialized()):
        if value is None:
            raise RuntimeError("rank-zero tensor is missing")
        return value.to(device=device, dtype=dtype)
    metadata = [tuple(int(item) for item in value.shape) if rank == 0 and value is not None else None]
    dist.broadcast_object_list(metadata, src=0)
    if rank != 0:
        value = torch.empty(metadata[0], device=device, dtype=dtype)
    else:
        value = value.to(device=device, dtype=dtype)
    dist.broadcast(value, src=0)
    return value


def encode_context_rank0(vae: Any, video_cthw: Any, *, device: Any) -> Any:
    """Run the replicated VAE once, then broadcast the exact latent tensor."""
    import torch
    _, rank = _distributed()
    latent = vae.encode([video_cthw.to(device)])[0] if rank == 0 else None
    return broadcast_tensor_from_rank0(latent, device=device, dtype=torch.float32)


def build_i2v_condition_rank0(
    vae: Any,
    first_frame_chw: Any,
    *,
    total_latent_frames: int,
    temporal_stride: int,
    device: Any,
    output_dtype: Any,
) -> Any:
    """Encode [first observed frame, zero pixels] and prepend the 4-channel mask."""
    import torch

    _, rank = _distributed()
    total, stride = int(total_latent_frames), int(temporal_stride)
    if total <= 0 or stride <= 0:
        raise ValueError("latent frames and temporal stride must be positive")
    if rank == 0 and (first_frame_chw.ndim != 3 or int(first_frame_chw.shape[0]) != 3):
        raise ValueError("first frame must have shape [3,H,W]")
    result = None
    if rank == 0:
        pixel_frames = (total - 1) * stride + 1
        condition_video = torch.cat(
            [
                first_frame_chw[:, None].to(device),
                torch.zeros(
                    3,
                    pixel_frames - 1,
                    first_frame_chw.shape[1],
                    first_frame_chw.shape[2],
                    device=device,
                    dtype=first_frame_chw.dtype,
                ),
            ],
            dim=1,
        )
        encoded = vae.encode([condition_video])[0]
        if int(encoded.shape[1]) != total:
            raise RuntimeError(f"I2V condition VAE length mismatch: {encoded.shape[1]} != {total}")
        mask_pixels = torch.ones(1, pixel_frames, encoded.shape[2], encoded.shape[3], device=device, dtype=encoded.dtype)
        mask_pixels[:, 1:] = 0
        mask_pixels = torch.cat([mask_pixels[:, :1].repeat(1, stride, 1, 1), mask_pixels[:, 1:]], dim=1)
        mask_latent = mask_pixels.view(1, total, stride, encoded.shape[2], encoded.shape[3]).transpose(1, 2)[0]
        result = torch.cat([mask_latent, encoded], dim=0).to(output_dtype)
    return broadcast_tensor_from_rank0(result, device=device, dtype=output_dtype)


def decode_all_latents_rank0(vae: Any, latents: Any) -> Any | None:
    _, rank = _distributed()
    return vae.decode([latents])[0] if rank == 0 else None
