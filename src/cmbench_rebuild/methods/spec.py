"""Small public description shared by every KV method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MethodSpec:
    """Everything configuration code needs to know about one method."""

    id: str
    display_name: str
    legacy_id: str
    policy: str
    defaults: Mapping[str, Any]

