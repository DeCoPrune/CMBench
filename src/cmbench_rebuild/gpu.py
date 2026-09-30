"""Fail-closed local GPU selection and idleness checks for benchmark jobs."""
from __future__ import annotations

import subprocess
from typing import Any


def parse_cuda_visible_devices(value: str, *, expected_count: int) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("CUDA_VISIBLE_DEVICES must be an explicit comma-separated index list")
    try:
        devices = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError("CUDA_VISIBLE_DEVICES entries must be non-negative integer GPU indices") from error
    if any(device < 0 for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES entries must be non-negative integer GPU indices")
    if len(devices) != len(set(devices)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU indices")
    if len(devices) != int(expected_count):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES must select exactly {int(expected_count)} GPUs; got {len(devices)}"
        )
    return devices


def require_idle_cuda_devices(
    value: str,
    *,
    expected_count: int,
    maximum_used_mib: int = 4096,
) -> dict[str, Any]:
    """Refuse a launch when any selected physical GPU is already materially used."""
    devices = parse_cuda_visible_devices(value, expected_count=expected_count)
    if int(maximum_used_mib) < 0:
        raise ValueError("maximum_used_mib must be non-negative")
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    inventory: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(completed.stdout.splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row {line_number}: {line!r}")
        try:
            index = int(parts[0])
            row = {
                "index": index,
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
        except ValueError as error:
            raise RuntimeError(f"invalid nvidia-smi row {line_number}: {line!r}") from error
        inventory[index] = row
    missing = [device for device in devices if device not in inventory]
    if missing:
        raise RuntimeError(f"selected GPUs do not exist: {missing}")
    selected = [inventory[device] for device in devices]
    busy = [row for row in selected if int(row["memory_used_mib"]) > int(maximum_used_mib)]
    if busy:
        raise RuntimeError(f"refusing to use busy GPUs: {busy}")
    return {
        "passed": True,
        "selected": selected,
        "maximum_used_mib": int(maximum_used_mib),
    }
