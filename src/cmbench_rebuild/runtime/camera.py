"""Global camera-pose planning with lazy per-chunk Plucker packing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_INTRINSICS_832X480 = (502.9116, 503.10812, 415.77786, 239.77779)


def validate_explicit_camera(camera: Mapping[str, Any], *, output_frames: int) -> None:
    yaw_rate = camera.get("camera_yaw_degrees_per_latent")
    yaw_keys = camera.get("camera_yaw_keyframes_degrees")
    if yaw_rate is None and yaw_keys is None:
        raise ValueError("explicit pose camera requires a yaw rate or yaw keyframes")
    if yaw_keys is None:
        return
    if len(yaw_keys) < 2:
        raise ValueError("yaw keyframes need at least two values")
    for field in ("camera_pitch_keyframes_degrees", "camera_translation_keyframes_m"):
        values = camera.get(field)
        if values is not None and len(values) != len(yaw_keys):
            raise ValueError(f"{field} must match yaw keyframes")
    indices = camera.get("camera_keyframe_latent_indices")
    if indices is not None:
        if len(indices) != len(yaw_keys) or indices[0] != 0 or indices[-1] != int(output_frames) - 1:
            raise ValueError("camera keyframe indices must match yaw keys and span [0, output_frames-1]")
        if any(int(right) <= int(left) for left, right in zip(indices, indices[1:])):
            raise ValueError("camera keyframe indices must be strictly increasing")


def _relative_poses(c2ws: Any) -> Any:
    import torch

    rotations = c2ws[:, :3, :3]
    translations = c2ws[:, :3, 3:]
    inverse = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device).unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
    inverse[:, :3, :3] = rotations.transpose(-1, -2)
    inverse[:, :3, 3:] = -torch.bmm(rotations.transpose(-1, -2), translations)
    relative = torch.matmul(inverse[0:1], c2ws)
    relative[0] = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device)
    step_inverse = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device).unsqueeze(0).repeat(relative.shape[0] - 1, 1, 1)
    step_inverse[:, :3, :3] = relative[:-1, :3, :3].transpose(-1, -2)
    step_inverse[:, :3, 3:] = -torch.bmm(relative[:-1, :3, :3].transpose(-1, -2), relative[:-1, :3, 3:])
    relative[1:] = torch.bmm(step_inverse, relative[1:])
    translations = relative[:, :3, 3]
    max_norm = torch.norm(translations, dim=-1).max()
    if max_norm > 0:
        relative[:, :3, 3] = translations / max_norm
    return relative


def _plucker(poses: Any, intrinsics: Any, *, height: int, width: int) -> Any:
    import torch

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=poses.device, dtype=poses.dtype) + 0.5,
        torch.arange(width, device=poses.device, dtype=poses.dtype) + 0.5,
        indexing="ij",
    )
    fx, fy, cx, cy = intrinsics[0]
    directions = torch.stack(((grid_x.reshape(-1) - cx) / fx, (grid_y.reshape(-1) - cy) / fy, torch.ones(height * width, device=poses.device, dtype=poses.dtype)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = (poses[:, :3, :3] @ directions.T).transpose(1, 2)
    rays_o = poses[:, :3, 3].unsqueeze(1).expand_as(rays_d)
    return torch.cat((rays_o, rays_d), dim=-1).reshape(poses.shape[0], height, width, 6)


@dataclass
class CameraPosePlan:
    relative_poses: Any
    intrinsics: Any
    height: int
    width: int
    spatial_stride: tuple[int, int]

    @property
    def frames(self) -> int:
        return int(self.relative_poses.shape[0])

    def packed_chunk(self, start: int, stop: int, *, device: str | Any = "cpu", dtype: Any | None = None) -> Any:
        left, right = int(start), int(stop)
        if not 0 <= left < right <= self.frames:
            raise ValueError(f"camera chunk [{left},{right}) outside [0,{self.frames})")
        poses = self.relative_poses[left:right].to(device=device)
        intrinsics = self.intrinsics[left:right].to(device=device)
        values = _plucker(poses, intrinsics, height=self.height, width=self.width)
        stride_h, stride_w = self.spatial_stride
        if self.height % stride_h or self.width % stride_w:
            raise ValueError("camera resolution must be divisible by VAE spatial stride")
        latent_h, latent_w = self.height // stride_h, self.width // stride_w
        packed = values.reshape(right - left, latent_h, stride_h, latent_w, stride_w, 6).permute(5, 2, 4, 0, 1, 3).reshape(6 * stride_h * stride_w, right - left, latent_h, latent_w).unsqueeze(0)
        return packed.to(dtype=dtype) if dtype is not None else packed


def explicit_camera_plan(
    camera: Mapping[str, Any],
    *,
    context_frames: int,
    output_frames: int,
    height: int,
    width: int,
    spatial_stride: Sequence[int] = (8, 8),
) -> CameraPosePlan:
    import torch

    validate_explicit_camera(camera, output_frames=output_frames)
    context, output = int(context_frames), int(output_frames)
    total = context + output
    frame_indices = torch.arange(total, dtype=torch.float32)
    yaw_keys = camera.get("camera_yaw_keyframes_degrees")
    if yaw_keys is None:
        yaw = frame_indices * float(camera["camera_yaw_degrees_per_latent"])
        lower = upper = fraction = None
    else:
        relative_indices = (frame_indices - float(context)).clamp_(0.0, float(max(output - 1, 0)))
        positions = camera.get("camera_keyframe_latent_indices")
        key_positions = torch.linspace(0, max(output - 1, 0), len(yaw_keys), dtype=torch.float32) if positions is None else torch.tensor(positions, dtype=torch.float32)
        keys = torch.tensor(yaw_keys, dtype=torch.float32)
        upper = torch.bucketize(relative_indices, key_positions, right=True).clamp(max=len(yaw_keys) - 1)
        lower = (upper - 1).clamp(min=0)
        interval = (key_positions[upper] - key_positions[lower]).clamp_min(1.0)
        fraction = ((relative_indices - key_positions[lower]) / interval).clamp_(0.0, 1.0)
        yaw = keys[lower] * (1.0 - fraction) + keys[upper] * fraction
    pitch_keys = camera.get("camera_pitch_keyframes_degrees")
    pitch = torch.zeros_like(yaw) if pitch_keys is None else torch.tensor(pitch_keys, dtype=torch.float32)[lower] * (1.0 - fraction) + torch.tensor(pitch_keys, dtype=torch.float32)[upper] * fraction
    yaw_radians, pitch_radians = torch.deg2rad(yaw), torch.deg2rad(pitch)
    cosine, sine = torch.cos(yaw_radians), torch.sin(yaw_radians)
    pitch_cosine, pitch_sine = torch.cos(pitch_radians), torch.sin(pitch_radians)
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(total, 1, 1)
    poses[:, 0, 0] = cosine
    poses[:, 0, 1] = sine * pitch_sine
    poses[:, 0, 2] = sine * pitch_cosine
    poses[:, 1, 1] = pitch_cosine
    poses[:, 1, 2] = -pitch_sine
    poses[:, 2, 0] = -sine
    poses[:, 2, 1] = cosine * pitch_sine
    poses[:, 2, 2] = cosine * pitch_cosine
    translations = camera.get("camera_translation_keyframes_m")
    if translations is not None:
        values = torch.tensor(translations, dtype=torch.float32)
        poses[:, :3, 3] = values[lower] * (1.0 - fraction[:, None]) + values[upper] * fraction[:, None]
    intrinsics = torch.tensor(
        [DEFAULT_INTRINSICS_832X480[0] / 832.0 * width, DEFAULT_INTRINSICS_832X480[1] / 480.0 * height, DEFAULT_INTRINSICS_832X480[2] / 832.0 * width, DEFAULT_INTRINSICS_832X480[3] / 480.0 * height],
        dtype=torch.float32,
    ).repeat(total, 1)
    return CameraPosePlan(_relative_poses(poses), intrinsics, int(height), int(width), (int(spatial_stride[0]), int(spatial_stride[1])))


def tail_motion_camera_plan(
    rotation_vector: Sequence[float],
    *,
    context_frames: int,
    output_frames: int,
    active_output_frames: int,
    height: int,
    width: int,
    translation_vector: Sequence[float] | None = None,
    spatial_stride: Sequence[int] = (8, 8),
) -> CameraPosePlan:
    """Identity context followed by repeated framewise motion then a hold."""
    import torch

    if len(rotation_vector) != 3 or (translation_vector is not None and len(translation_vector) != 3):
        raise ValueError("rotation/translation vectors must have three values")
    context, output, active = int(context_frames), int(output_frames), int(active_output_frames)
    if context <= 0 or output <= 0 or not 0 <= active <= output:
        raise ValueError("invalid tail-motion timeline")
    vector = torch.tensor(rotation_vector, dtype=torch.float32)
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=torch.float32)
    skew = torch.stack((torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero))))
    rotation = torch.linalg.matrix_exp(skew)
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(context + output, 1, 1)
    if active:
        poses[context:context + active, :3, :3] = rotation
        if translation_vector is not None:
            poses[context:context + active, :3, 3] = torch.tensor(translation_vector, dtype=torch.float32)
    intrinsics = torch.tensor(
        [DEFAULT_INTRINSICS_832X480[0] / 832.0 * width, DEFAULT_INTRINSICS_832X480[1] / 480.0 * height, DEFAULT_INTRINSICS_832X480[2] / 832.0 * width, DEFAULT_INTRINSICS_832X480[3] / 480.0 * height],
        dtype=torch.float32,
    ).repeat(context + output, 1)
    return CameraPosePlan(poses, intrinsics, int(height), int(width), (int(spatial_stride[0]), int(spatial_stride[1])))
