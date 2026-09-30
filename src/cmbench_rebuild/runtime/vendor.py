"""Integrity checks for the pinned, unmodified LingBot World v2 source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "wan"


def origin_file() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "lingbot-world-v2" / "ORIGIN.json"


def subtree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    # Python imports may create __pycache__ beside the pinned files. Runtime
    # artifacts are not part of the upstream source subtree.
    files = sorted(path for path in Path(root).rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"vendored source tree is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def verify_vendored_source() -> dict[str, Any]:
    origin = json.loads(origin_file().read_text(encoding="utf-8"))
    actual = subtree_sha256(source_root())
    expected = str(origin["subtree_sha256"])
    if actual != expected:
        raise RuntimeError(
            "vendored LingBot source differs from its pinned upstream subtree: "
            f"expected {expected}, got {actual}"
        )
    return {**origin, "verified": True, "source_root": str(source_root())}
