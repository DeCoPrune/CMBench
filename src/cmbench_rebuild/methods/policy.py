"""Configured policy objects consumed by the method-neutral runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import resolve


@dataclass(frozen=True)
class ConfiguredPolicy:
    name: str
    implementation: str
    parameters: Mapping[str, Any]


def policy_from_config(config: Mapping[str, Any]) -> ConfiguredPolicy:
    spec = resolve(str(config["method"]))
    parameters = spec.defaults | dict(config.get("method_params") or {})
    unknown = set(parameters) - set(spec.defaults)
    if unknown:
        raise ValueError(f"unknown policy parameters for {spec.id}: {sorted(unknown)}")
    return ConfiguredPolicy(name=spec.id, implementation=spec.policy, parameters=parameters)
