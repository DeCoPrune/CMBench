"""Build a queryable, deterministic index without a database dependency."""
from __future__ import annotations
import json
from pathlib import Path
from .lifecycle import RunStore

def build(run_root: Path, output: Path) -> list[dict]:
    rows = []
    for identity_path in sorted(run_root.glob("*/identity.json")):
        data = json.loads(identity_path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        store = RunStore(identity_path.parent)
        state = store.status()
        success = store.successful_attempt()
        completion_path = success.completion_path if success else None
        completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path else {}
        rows.append({"run_id": identity_path.parent.name, "case_id": config.get("case_id"), "method": config.get("method"), "seed": config.get("seed"), "status": state["status"], "manifest": str(identity_path), "completion_manifest": str(completion_path) if completion_path else None, "attempt_count": state["attempt_count"], "metrics": completion.get("metrics", data.get("metrics", {})), "artifacts": completion.get("artifacts", data.get("artifacts", {}))})
    manifests = sorted(run_root.glob("*/manifest.json")) + sorted(run_root.glob("*/manifest.start.json"))
    for manifest in manifests:
        if (manifest.parent / "identity.json").exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        config = data.get("config", {})
        completion_path = manifest.parent / "manifest.complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
        rows.append({"run_id": manifest.parent.name, "case_id": config.get("case_id"), "method": config.get("method"), "seed": config.get("seed"), "status": completion.get("status", data.get("status")), "manifest": str(manifest), "completion_manifest": str(completion_path) if completion else None, "metrics": data.get("metrics", {}), "artifacts": data.get("artifacts", {})})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows
