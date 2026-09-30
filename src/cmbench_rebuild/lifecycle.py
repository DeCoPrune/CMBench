"""Immutable run identity plus append-only execution attempts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts.manifest import write_immutable_manifest


@dataclass(frozen=True)
class Attempt:
    number: int
    root: Path

    @property
    def start_path(self) -> Path:
        return self.root / "start.json"

    @property
    def completion_path(self) -> Path:
        return self.root / "completion.json"


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def identity_path(self) -> Path:
        return self.root / "identity.json"

    def ensure_identity(self, identity: dict[str, Any]) -> bool:
        if self.identity_path.exists():
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing.get("config") != identity.get("config"):
                raise ValueError(f"run id already belongs to a different config: {self.root.name}")
            return False
        write_immutable_manifest(self.identity_path, identity)
        return True

    def attempts(self) -> list[Attempt]:
        parent = self.root / "attempts"
        if not parent.is_dir():
            return []
        values = []
        for path in sorted(parent.iterdir()):
            if path.is_dir() and path.name.isdigit():
                values.append(Attempt(int(path.name), path))
        return values

    def successful_attempt(self) -> Attempt | None:
        for attempt in reversed(self.attempts()):
            if not attempt.completion_path.is_file():
                continue
            result = json.loads(attempt.completion_path.read_text(encoding="utf-8"))
            if result.get("status") == "completed" and int(result.get("returncode", 1)) == 0:
                return attempt
        return None

    def start_attempt(self, payload: dict[str, Any]) -> Attempt:
        attempts = self.attempts()
        number = attempts[-1].number + 1 if attempts else 1
        attempt = Attempt(number, self.root / "attempts" / f"{number:04d}")
        write_immutable_manifest(attempt.start_path, payload)
        return attempt

    def finish_attempt(self, attempt: Attempt, payload: dict[str, Any]) -> None:
        write_immutable_manifest(attempt.completion_path, payload)

    def status(self) -> dict[str, Any]:
        attempts = self.attempts()
        success = self.successful_attempt()
        return {"run_id": self.root.name, "identity": str(self.identity_path), "attempt_count": len(attempts), "status": "completed" if success else ("not_started" if not attempts else "incomplete"), "successful_attempt": success.number if success else None}
