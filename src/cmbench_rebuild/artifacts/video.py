"""Container metadata and decoded-frame hashes for reproducible video checks."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def write_video_tensor(tensor: Any, path: Path, *, fps: float = 16.0) -> dict[str, Any]:
    """Encode a single RGB video tensor strictly, then decode-audit it."""
    import torch

    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach()
    if value.ndim == 5 and int(value.shape[0]) == 1:
        value = value[0]
    if value.ndim != 4 or int(value.shape[0]) != 3:
        raise ValueError(f"video tensor must have shape [3,F,H,W], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("video tensor contains non-finite values")
    frames = value.float().clamp(-1, 1).add_(1).mul_(127.5).round_().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    frame_count, height, width, channels = map(int, frames.shape)
    if frame_count <= 0 or channels != 3 or float(fps) <= 0:
        raise ValueError("video dimensions and fps must be positive RGB")
    command = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(float(fps)), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.run(command, input=frames.tobytes(), capture_output=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, process.stdout, process.stderr)
    audited = inspect_video(output)
    if int(audited["decoded_frame_count"]) != frame_count:
        raise RuntimeError(f"encoded video frame mismatch: {audited['decoded_frame_count']} != {frame_count}")
    return audited


def inspect_video(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    frames = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    stream = metadata.get("streams", [{}])[0]
    frame_bytes = int(stream["width"]) * int(stream["height"]) * 3
    hashes = []
    assert frames.stdout is not None
    while True:
        raw = frames.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            frames.kill()
            raise ValueError(f"partial decoded frame: {len(raw)} of {frame_bytes} bytes")
        hashes.append(hashlib.sha256(raw).hexdigest())
    returncode = frames.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, frames.args)
    return {"schema_version": 1, "path": str(path), "probe": metadata, "decoded_pixel_format": "rgb24", "decoded_frame_count": len(hashes), "decoded_frame_sha256": hashes}


def compare_videos(expected: Path, actual: Path) -> dict[str, Any]:
    expected_info = inspect_video(expected)
    actual_info = inspect_video(actual)
    expected_hashes = expected_info["decoded_frame_sha256"]
    actual_hashes = actual_info["decoded_frame_sha256"]
    process = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(expected), "-i", str(actual), "-lavfi", "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args, process.stdout, process.stderr)
    frame_psnr = []
    for line in process.stderr.splitlines():
        match = re.match(r"n:(\d+) .* psnr_avg:(\S+)", line)
        if match:
            value = None if match.group(2) == "inf" else float(match.group(2))
            frame_psnr.append({"frame": int(match.group(1)) - 1, "psnr": value})
    summary = re.search(r"PSNR y:(\S+) u:(\S+) v:(\S+) average:(\S+) min:(\S+) max:(\S+)", process.stderr)
    return {
        "schema_version": 1,
        "expected": str(expected),
        "actual": str(actual),
        "expected_probe": expected_info["probe"],
        "actual_probe": actual_info["probe"],
        "expected_frames": len(expected_hashes),
        "actual_frames": len(actual_hashes),
        "exact_matching_frames": sum(left == right for left, right in zip(expected_hashes, actual_hashes)),
        "all_frames_exact": expected_hashes == actual_hashes,
        "frame_psnr": frame_psnr,
        "psnr_summary": ({"y": float(summary.group(1)), "u": float(summary.group(2)), "v": float(summary.group(3)), "average": float(summary.group(4)), "minimum": float(summary.group(5)), "maximum": None if summary.group(6) == "inf" else float(summary.group(6))} if summary else None),
    }
