"""Execute one four-GPU lane from an already frozen speed-matrix plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmbench_rebuild.matrix import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--case-ids", nargs="+", help="Optional common case subset for a shorter paired retest.")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result = run_matrix(
        plan,
        python=args.python,
        cuda_visible_devices=args.cuda_visible_devices,
        source_root=Path.cwd(),
        methods=args.methods,
        case_ids=args.case_ids,
        continue_on_error=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
