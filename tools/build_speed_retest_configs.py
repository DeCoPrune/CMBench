"""Create an isolated, paired 10-case speed matrix on the nav2 snapshot."""
from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("/mnt/workspace/users/xiaozeqi/code/forcingkv_cmbench_ablation_reindex_20260911")
SNAPSHOT = Path("/mnt/workspace/users/xiaozeqi/code/forcingkv_cmbench_speed_online_20260916")
OUT = SNAPSHOT / "configs/experiments/speed_online_10cases_20260916"
SPECS = {
    "fullkv": ("reindex39_seed2", "fullkv-with.json"),
    "decoprune": ("reindex39_seed2", "ours-with.json"),
    "decoprune_hs": ("reindex39_seed2_other", "consistency_prune_hs-with.json"),
    "random": ("reindex39_seed2_other", "random-with.json"),
    "patchification": ("reindex39_seed2_other", "patchification-with.json"),
    "forcingkv": ("reindex39_seed2_other", "forcingkv-with.json"),
    "streaming": ("reindex39_seed2_other", "streaming-with.json"),
    "dummy_forcing": ("reindex39_seed2_other", "dummy_forcing-with.json"),
}


def read_requests(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    loaded: dict[str, tuple[dict, list[dict]]] = {}
    for method, (folder, name) in SPECS.items():
        config = json.loads((SOURCE / "configs/experiments" / folder / name).read_text())
        requests = read_requests(Path(config["request_file"]))
        loaded[method] = (config, requests)

    reference_ids = [row["case_id"] for row in loaded["fullkv"][1]]
    if len(reference_ids) != 39 or len(set(reference_ids)) != 39:
        raise RuntimeError("expected 39 distinct benchmark cases in the source matrix")
    for method, (_, requests) in loaded.items():
        if [row["case_id"] for row in requests] != reference_ids:
            raise RuntimeError(f"{method} does not share the same ordered 39-case input set")

    # Deterministically spread 10 paired timing cases across the 39-case set.
    indices = [round(i * (len(reference_ids) - 1) / 9) for i in range(10)]
    selected_ids = [reference_ids[index] for index in indices]
    OUT.mkdir(parents=True, exist_ok=True)
    for method, (config, requests) in loaded.items():
        selected = [row for row in requests if row["case_id"] in set(selected_ids)]
        selected.sort(key=lambda row: selected_ids.index(row["case_id"]))
        if method in {"decoprune", "decoprune_hs", "random"}:
            for row in selected:
                row["ours_step_index"] = 2
        request_path = OUT / f"{method}.jsonl"
        request_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
        config["request_file"] = str(request_path)
        if method in {"decoprune", "decoprune_hs", "random"}:
            config["method_params"] = {**dict(config.get("method_params") or {}), "step_index": 2}
        if method == "patchification":
            config["method_params"] = {**dict(config.get("method_params") or {}), "update_each_chunk": True}
        (OUT / f"{method}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"methods": list(SPECS), "cases": selected_ids, "seed": 2, "ours_context_step_index": 2}, indent=2))


if __name__ == "__main__":
    main()
