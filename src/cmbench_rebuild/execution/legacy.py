"""Explicit adapter for running the read-only legacy LingBot entry point on NAV."""
from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..methods import resolve


@dataclass(frozen=True)
class NavSettings:
    host: str = "nav"
    legacy_root: str = "/mnt/workspace/users/xiaozeqi/code/forcingkv_private_lingbotworld_lql_d206858"
    python: str = "/mnt/workspace/users/xiaozeqi/envs/forcingkv_private_lingbotworld_lql_d206858/bin/python"
    remote_run_root: str = "/mnt/workspace/users/xiaozeqi/runs/forcingkv_cmbench_rebuild"
    rebuild_root: str = "/mnt/workspace/users/xiaozeqi/code/forcingkv_cmbench_rebuild"
    master_port: int = 29571


_FLAG_MAP = {
    "dummy_forcing": {"first_history_frames": "--dummy-first-history-frames", "middle_history_frames": "--dummy-middle-history-frames", "last_history_frames": "--dummy-last-history-frames"},
    "forcingkv": {"static_sink_frames": "--forcingkv-static-sink-frames", "static_recent_chunks": "--forcingkv-static-recent-chunks", "dynamic_sink_frames": "--forcingkv-dynamic-sink-frames", "dynamic_recent_chunks": "--forcingkv-dynamic-recent-chunks", "selected_patch_count": "--forcingkv-selected-patch-count", "middle_window_chunks": "--forcingkv-middle-window-chunks", "middle_candidate_scope": "--forcingkv-middle-candidate-scope", "layer_zero_policy": "--forcingkv-layer-zero-policy"},
    "patchification": {"grid_rows": "--patchify-retrieve-grid-rows", "grid_cols": "--patchify-retrieve-grid-cols", "topk_blocks": "--patchify-retrieve-topk-blocks"},
    "random": {"step_index": "--ours-step-index", "threshold": "--ours-threshold"},
    "decoprune": {"step_index": "--ours-step-index", "threshold": "--ours-threshold"},
    "decoprune_hs": {"step_index": "--ours-step-index", "threshold": "--ours-threshold"},
}


def select_request(source: Path, case_id: str, output: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one request for {case_id}, found {len(matches)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matches[0], separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return matches[0]


def build_legacy_command(config: dict[str, Any], run_id: str, remote_request: str, settings: NavSettings) -> list[str]:
    spec = resolve(config["method"])
    runner = str(PurePosixPath(settings.legacy_root) / "vendor/lingbot_world2/run_context_memory_lingbot_sp.py")
    remote_output = str(PurePosixPath(settings.remote_run_root) / run_id)
    command = [settings.python, "-m", "torch.distributed.run", f"--nproc_per_node={config['world_size']}", f"--master_port={settings.master_port}", runner, "--checkpoint-dir", config["checkpoint"], "--output-root", remote_output, "--profile-name", config["protocol"], "--method", spec.legacy_id, "--height", str(config["height"]), "--width", str(config["width"]), "--seed", str(config["seed"]), "--sampling-shift", str(config["sampling_shift"]), "--ulysses-size", str(config["world_size"]), "--generation-kv-policy", config["generation_kv_policy"], "--requests-jsonl", remote_request, "--skip-existing"]
    rope = config["rope_reindex"]
    command += ["--rope-reindex-mode", str(rope["mode"]), "--rope-reindex-virtual-span", str(rope["virtual_span"]), "--rope-reindex-recent-frames", str(rope["recent_frames"]), "--rope-reindex-fast-band-pairs", str(rope["fast_band_pairs"])]
    if spec.id == "streaming":
        command += ["--local-attn-size", str(config["method_params"]["local_attn_size"]), "--sink-size", str(config["method_params"]["sink_size"])]
    else:
        command += ["--local-attn-size", "-1", "--sink-size", "0"]
    for key, flag in _FLAG_MAP.get(spec.id, {}).items():
        if spec.id == "streaming":
            continue
        value = config["method_params"].get(key)
        if value is not None and not isinstance(value, bool):
            command += [flag, str(value)]
    if spec.id in {"forcingkv", "decoprune_hs"}:
        configured = PurePosixPath(str(config["head_map_file"]))
        head_map = configured if configured.is_absolute() else PurePosixPath(settings.rebuild_root) / configured
        command += ["--head-map-file", str(head_map)]
    if spec.id == "random":
        command += ["--ours-random-reference-root", str(PurePosixPath(settings.remote_run_root) / run_id / "consistency_prune")]
    return command


def remote_plan(config: dict[str, Any], run_id: str, local_request: Path, settings: NavSettings) -> dict[str, Any]:
    remote_dir = PurePosixPath(settings.remote_run_root) / run_id
    remote_request = str(remote_dir / "request.jsonl")
    command = build_legacy_command(config, run_id, remote_request, settings)
    return {"host": settings.host, "local_request": str(local_request), "remote_request": remote_request, "remote_output": str(remote_dir), "command": command, "shell_preview": shlex.join(command)}


def probe_nav(settings: NavSettings) -> dict[str, Any]:
    script = f"test -r {shlex.quote(settings.legacy_root)} && {shlex.quote(settings.python)} --version && test -r {shlex.quote(settings.python)}"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", settings.host, script], capture_output=True, text=True)
    return {"host": settings.host, "reachable": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "legacy_root": settings.legacy_root, "python": settings.python}


def preflight_gpus(host: str, cuda_ids: str, *, maximum_used_mib: int = 4096) -> dict[str, Any]:
    selected = [int(value) for value in cuda_ids.split(",")]
    if len(selected) != len(set(selected)):
        raise ValueError("cuda_ids contains duplicates")
    command = ["ssh", "-o", "BatchMode=yes", host, "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    inventory = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        inventory[int(parts[0])] = {"index": int(parts[0]), "name": parts[1], "memory_total_mib": int(parts[2]), "memory_used_mib": int(parts[3]), "utilization_percent": int(parts[4])}
    missing = [value for value in selected if value not in inventory]
    busy = [inventory[value] for value in selected if value in inventory and inventory[value]["memory_used_mib"] > maximum_used_mib]
    if missing:
        raise RuntimeError(f"selected GPUs do not exist on {host}: {missing}")
    if busy:
        raise RuntimeError(f"refusing to use busy GPUs on {host}: {busy}")
    return {"host": host, "selected": [inventory[value] for value in selected], "maximum_used_mib": maximum_used_mib, "passed": True}


def execute_legacy(plan: dict[str, Any], *, cuda_ids: str, log_path: Path) -> dict[str, Any]:
    """Stage one immutable request and execute only inside the dedicated new run root."""
    if not cuda_ids or any(part.strip() == "" for part in cuda_ids.split(",")):
        raise ValueError("cuda_ids must be an explicit comma-separated list")
    host = str(plan["host"])
    nproc = next(int(value.split("=", 1)[1]) for value in plan["command"] if value.startswith("--nproc_per_node="))
    if len(cuda_ids.split(",")) != nproc:
        raise ValueError(f"selected GPU count must equal world size: {len(cuda_ids.split(','))} != {nproc}")
    preflight = preflight_gpus(host, cuda_ids)
    remote_output = PurePosixPath(str(plan["remote_output"]))
    remote_request = str(plan["remote_request"])
    local_request = str(plan["local_request"])
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", str(remote_output)], check=True)
    subprocess.run(["scp", "-q", local_request, f"{host}:{remote_request}"], check=True)
    remote_shell = "CUDA_VISIBLE_DEVICES=" + shlex.quote(cuda_ids) + " " + shlex.join(plan["command"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with log_path.open("wb") as log:
        process = subprocess.run(["ssh", "-o", "BatchMode=yes", host, remote_shell], stdout=log, stderr=subprocess.STDOUT)
    ended = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "cuda_visible_devices": cuda_ids,
        "local_log": str(log_path),
        "remote_output": str(remote_output),
        "gpu_preflight": preflight,
    }


def sync_legacy_output(plan: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Copy a completed isolated remote run without deleting or overwriting old data."""
    if destination.exists():
        raise FileExistsError(f"sync destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = f"{plan['host']}:{plan['remote_output']}"
    subprocess.run(["scp", "-q", "-r", source, str(destination)], check=True)
    return {"source": source, "destination": str(destination), "status": "synced"}
