"""Dependency-free content identities shared by production and evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_content_identity(root: Path) -> dict[str, Any]:
    """Return a deterministic content manifest for every regular file in a tree."""
    base = root.resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    paths = sorted(path for path in base.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"content directory is empty: {base}")
    digest = hashlib.sha256()
    entries = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(base).as_posix()
        stat = path.stat()
        size = stat.st_size
        checksum = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
        entries.append({
            "path": relative,
            "bytes": size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": checksum,
        })
        total_bytes += size
    return {
        "algorithm": "sha256-relative-path-null-content-digest-v1",
        "path": str(base),
        "files": len(entries),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


def validate_directory_content_identity(
    identity: dict[str, Any], *, verify_content: bool = True
) -> dict[str, Any]:
    """Validate an immutable tree fully, or by its exact file/stat inventory."""
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("directory identity has no root path")
    base = Path(raw_path).resolve()
    expected = identity.get("entries")
    if not isinstance(expected, list) or not expected:
        raise ValueError("directory identity has no file entries")
    paths = sorted(path for path in base.rglob("*") if path.is_file())
    current_inventory = [
        {
            "path": path.relative_to(base).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    expected_inventory = [
        {key: row[key] for key in ("path", "bytes", "mtime_ns")}
        for row in expected
    ]
    if current_inventory != expected_inventory:
        raise ValueError(f"directory file inventory changed: {base}")
    if verify_content and directory_content_identity(base) != identity:
        raise ValueError(f"directory content identity changed: {base}")
    return identity


def file_content_identity(path: Path) -> dict[str, Any]:
    """Bind one immutable input by resolved path, byte length, and contents."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def validate_file_content_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Rehash one bound file and reject path, size, or content changes."""
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("file identity has no path")
    if file_content_identity(Path(raw_path)) != identity:
        raise ValueError(f"file content identity changed: {raw_path}")
    return identity
