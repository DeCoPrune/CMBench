"""Append-only manifests and content hashes; never mutate a completed run."""
from __future__ import annotations
import hashlib, json, os, platform, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..identity import directory_content_identity, sha256_file, validate_directory_content_identity

def code_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned:" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]

def code_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    try:
        names = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).decode().split("\0")
        names = sorted(name for name in names if name)
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]).strip())
    except (OSError, subprocess.CalledProcessError):
        names = sorted(str(path.relative_to(root)) for path in root.rglob("*.py") if ".git" not in path.parts and "runs" not in path.parts)
        dirty = None
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return {"git_revision": code_revision(root), "git_dirty": dirty, "content_sha256": digest.hexdigest(), "tracked_and_untracked_file_count": len(names)}

def write_immutable_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}; choose a new run id")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def new_manifest(config: dict[str, Any], *, command: list[str], source_root: Path) -> dict[str, Any]:
    request = Path(config["request_file"])
    code = code_snapshot(source_root)
    return {"schema_version": 1, "status": "created", "created_at": datetime.now(timezone.utc).isoformat(), "code_hash": code["content_sha256"], "code": code, "command": command, "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unspecified"), "checkpoint": config["checkpoint"], "dataset_version": config["dataset_version"], "prompt_request_sha256": sha256_file(request), "config": config, "artifacts": {}, "metrics": {}}
