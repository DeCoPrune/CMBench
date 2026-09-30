from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmbench_rebuild.config import RunConfig
from cmbench_rebuild.dataset import (
    decode_reference, evaluator_inputs, load_benchmark, pixel_box, prepare_requests,
)
from cmbench_rebuild.evaluation.cmbench import eligible_case_ids
from cmbench_rebuild.evaluation.official import resolve_reference_videos
from cmbench_rebuild.matrix import build_matrix_plan, _validate_method_request_contract
from cmbench_rebuild.methods.registry import resolve
from cmbench_rebuild.runtime.requests import load_requests


@pytest.fixture
def benchmark(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = tmp_path / "bench"
    (root / "videos/real").mkdir(parents=True)
    video = root / "videos/real/real_001.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((48, 64, 3), 20 * i, dtype=np.uint8))
    writer.release()
    rows = []
    for kind in ["reappear", "revisit"]:
        rows.append({
            "type": "real", "task_id": f"real_001_{kind}_01", "scene": "Living Room",
            "video": "videos/real/real_001.mp4",
            "context_clip_prompts": [{"clip_id": f"{i:02d}", "clip_prompt": f"Part {i}."} for i in range(1, 7)],
            "continue_prompt": "Show the same cup." if kind == "reappear" else "Turn left to revisit the cup.",
            "task_type": kind, "target_label": "blue cup", "target_aliases": ["cup"],
            "references": [{"timestamp_seconds": 0.08, "frame_index": None,
                            "bbox_xyxy_normalized": [0.25, 0.25, 0.75, 0.75]}],
        })
    (root / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, rows


def test_export_and_official_inputs_preserve_public_annotations(benchmark, tmp_path):
    root, rows = benchmark
    path = root / "metadata.jsonl"
    original = path.read_bytes()
    requests_path = tmp_path / "requests.jsonl"
    assert prepare_requests(root, requests_path, [2, 3])["requests"] == 4
    requests = load_requests(requests_path)
    assert {r.case_id for r in requests} == {r["task_id"] for r in rows}
    assert all(r.clip_file == (root / rows[0]["video"]).resolve() for r in requests)
    assert all(r.raw["references"] == rows[0]["references"] for r in requests)
    assert requests[0].prompt == rows[0]["continue_prompt"]
    assert requests[0].camera == {}
    ids = [r["task_id"] for r in rows]
    assert eligible_case_ids(path, ids) == tuple(ids)
    videos = resolve_reference_videos(benchmark_root=root, metadata=path, annotations=path, case_ids=ids)
    assert len(videos) == 2
    metadata, annotations = evaluator_inputs(root, path, path, tmp_path / "eval", ids)
    converted = [json.loads(line) for line in annotations.read_text().splitlines()]
    assert converted[0]["references"][0]["bbox_xyxy"] == [16, 12, 48, 36]
    assert converted[0]["references"][0]["frame"] == 2
    assert converted[0]["references"][0]["timestamp_seconds"] == 0.08
    assert [r["memory_level"] for r in converted] == ["object", "scene"]
    assert metadata.read_bytes() == annotations.read_bytes()
    assert path.read_bytes() == original


def test_explicit_frame_takes_precedence_over_timestamp(benchmark):
    root, rows = benchmark
    ref = {**rows[0]["references"][0], "frame_index": 0}
    _, frame_index, _ = decode_reference(root / rows[0]["video"], ref)
    assert frame_index == 0


def test_pixel_box_avoids_one_pixel_roundoff():
    assert pixel_box([337/832, 2/480, 494/832, 1.0], 832, 480) == [337, 2, 494, 480]


@pytest.mark.parametrize("mutation", ["duplicate", "path", "empty_box", "missing_video"])
def test_invalid_public_records_fail_before_planning(benchmark, mutation):
    root, rows = benchmark
    if mutation == "duplicate":
        rows[1]["task_id"] = rows[0]["task_id"]
    elif mutation == "path":
        rows[0]["video"] = "../outside.mp4"
    elif mutation == "empty_box":
        rows[0]["references"][0]["bbox_xyxy_normalized"] = [0.5, 0.2, 0.5, 0.7]
    else:
        rows[0]["video"] = "videos/missing.mp4"
    (root / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_benchmark(root)


def test_public_requests_build_a_multi_method_multi_seed_plan(benchmark, tmp_path):
    root, rows = benchmark
    requests = tmp_path / "requests.jsonl"
    prepare_requests(root, requests, [2, 3])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"fixture": true}')
    configs = []
    for method in ["fullkv", "decoprune"]:
        config = dict(protocol="test", method=method, case_id=rows[0]["task_id"], seed=2,
                      checkpoint=str(checkpoint), dataset_version="Aoraku/CMBench", request_file=str(requests))
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps(config))
        configs.append(path)
    plan = build_matrix_plan(configs, seeds=[2, 3], output_root=tmp_path / "runs",
                             source_root=Path(__file__).resolve().parents[1])
    assert plan["tasks"] == 8
    assert plan["cases_per_method"] == 2
    assert {r["method"] for r in plan["entries"]} == {"fullkv", "decoprune"}
    # An incomplete method-bound request must still fail the frozen protocol contract.
    bad = load_requests(requests)[0]
    bad.raw["ours_threshold"] = 0.8
    normalized = RunConfig(**config).normalized()
    with pytest.raises(ValueError, match="request method does not match"):
        _validate_method_request_contract(normalized, [bad])


def test_method_names_normalize_at_the_config_boundary():
    assert resolve("decoprune").display_name == "DeCoPrune"
    assert resolve("DeCoPrune-HS").id == "decoprune_hs"
    assert resolve("consistency_prune").id == "decoprune"
    assert resolve("consistency_prune_hs").id == "decoprune_hs"


def test_official_adapter_passes_converted_inputs_without_relabeling_run(benchmark, tmp_path, monkeypatch):
    import shutil
    import sys
    from types import SimpleNamespace
    from cmbench_rebuild.evaluation import official

    root, rows = benchmark
    case_id = rows[0]["task_id"]
    case = tmp_path / "generated"
    case.mkdir()
    original_metadata = {"case_id": case_id, "method": "decoprune"}
    (case / "metadata.json").write_text(json.dumps(original_metadata))
    shutil.copyfile(root / rows[0]["video"], case / "continuation.mp4")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# External evaluator fixture\n")
    evaluation = tmp_path / "evaluation"

    def fake_evaluator(command, **kwargs):
        annotations = Path(command[command.index("--annotations") + 1])
        converted = json.loads(annotations.read_text().splitlines()[0])
        assert converted["case_id"] == case_id
        assert converted["references"][0]["bbox_xyxy"] == [16, 12, 48, 36]
        generated = Path(command[command.index("--output-roots") + 1])
        assert json.loads((generated / "metadata.json").read_text())["method"] == "consistency_prune"
        assert (generated / "continuation.mp4").resolve() == case / "continuation.mp4"
        result = {"selected_case_id": case_id, "scoring_path": "owl_sam_dino",
                  "references": [{"used_direct_dino": False}], "dino_score": 0.75, "num_eval_frames": 10}
        (evaluation / "video_summary.jsonl").write_text(json.dumps(result) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_evaluator)
    monkeypatch.setattr(official, "_model_identity", lambda path: {"fixture": True})
    monkeypatch.setattr(official, "inspect_video", lambda path: {"decoded_frame_count": 10})
    result = official.run_official_dino_batch(
        case_dirs=[case], benchmark_root=root, metadata=root / "metadata.jsonl",
        annotations=root / "metadata.jsonl", evaluator=evaluator, python=Path(sys.executable),
        owl_model=tmp_path, sam_model=evaluator, dino_model=tmp_path, eval_root=evaluation,
    )[case_id]
    assert result["method"] == "decoprune" and result["dino_score"] == 0.75
    assert json.loads((case / "metadata.json").read_text()) == original_metadata
