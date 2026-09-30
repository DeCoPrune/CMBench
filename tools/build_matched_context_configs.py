from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired common-q0 context experiment from existing method requests.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in (("decoprune", "decoprune"), ("patchification", "patchify")):
        if source_name == "decoprune" and not (args.source_config_dir / "decoprune.json").exists():
            source_name = "consistency_prune"
        config = json.loads((args.source_config_dir / f"{source_name}.json").read_text())
        if output_name == "decoprune":
            config["method"] = "decoprune"
        source_requests = args.source_config_dir / f"{source_name}.jsonl"
        rows = [json.loads(line) for line in source_requests.read_text().splitlines() if line.strip()]
        selected = [row for row in rows if row["case_id"] in set(args.case_ids)]
        if [row["case_id"] for row in selected] != list(args.case_ids):
            raise SystemExit(f"{source_name}: requested cases are missing or not in requested order")
        for row in selected:
            row["matched_context_policy"] = "patchification_q0"
        request_path = args.output_dir / f"{output_name}.jsonl"
        request_path.write_text("".join(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n" for row in selected))
        config["case_id"] = selected[0]["case_id"]
        config["request_file"] = str(request_path)
        (args.output_dir / f"{output_name}.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({"cases": list(args.case_ids), "methods": ["decoprune", "patchification"], "matched_context_policy": "patchification_q0", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
