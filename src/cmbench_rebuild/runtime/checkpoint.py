"""Read-only validation of the LingBot World v2 checkpoint layout."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    transformer: Path
    transformer_config: Path
    vae: Path
    text_encoder: Path
    tokenizer: Path
    weight_files: tuple[Path, ...]

    def manifest(self) -> dict[str, Any]:
        config_bytes = self.transformer_config.read_bytes()
        return {
            "root": str(self.root),
            "transformer": str(self.transformer),
            "transformer_config": str(self.transformer_config),
            "transformer_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "vae": str(self.vae),
            "text_encoder": str(self.text_encoder),
            "tokenizer": str(self.tokenizer),
            "weight_files": [str(path) for path in self.weight_files],
            "weight_bytes": sum(path.stat().st_size for path in self.weight_files),
        }


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def inspect_checkpoint(root: Path) -> CheckpointLayout:
    base = Path(root).expanduser().resolve()
    _require_dir(base, "checkpoint directory")
    transformer = _require_dir(base / "transformers", "causal-fast transformer")
    transformer_config = _require_file(transformer / "config.json", "transformer config")
    try:
        config = json.loads(transformer_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid transformer config: {transformer_config}") from error
    if int(config.get("num_layers", -1)) != 40 or int(config.get("num_heads", -1)) != 40:
        raise ValueError("checkpoint must be the 40-layer, 40-head LingBot causal-fast model")
    weights = tuple(sorted(transformer.glob("*.safetensors"))) + tuple(sorted(transformer.glob("*.bin")))
    if not weights:
        raise FileNotFoundError(f"no transformer weights found in {transformer}")
    vae = _require_file(base / "Wan2.1_VAE.pth", "Wan VAE")
    text_encoder = _require_file(base / "models_t5_umt5-xxl-enc-bf16.pth", "T5 encoder")
    tokenizer = _require_dir(base / "google" / "umt5-xxl", "T5 tokenizer")
    return CheckpointLayout(base, transformer, transformer_config, vae, text_encoder, tokenizer, weights)
