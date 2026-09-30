from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import build_artifact_bundle, compare_videos, inspect_video
from .artifacts.manifest import new_manifest, write_immutable_manifest
from .config import load
from .dataset import prepare_requests
from .delivery import audit_delivery
from .evaluation import (
    aggregate_official_matrix,
    bind_aggregate_to_official_plan,
    audit_generation_matrix,
    audit_mask_accounting,
    audit_official_matrix_progress,
    audit_seqpr,
    build_official_matrix_plan,
    compare_dino,
    evaluate_direct_bbox_dino,
    run_official_dino,
    run_official_matrix,
)
from .execution import NavSettings, execute_legacy, probe_nav, remote_plan, select_request, sync_legacy_output
from .golden import collect_legacy_summary, compare, snapshot_protocol, validate_golden
from .index import build
from .lifecycle import RunStore
from .matrix import build_matrix_plan, run_matrix


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmbench")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-benchmark", help="Build generation requests from the public CMBench release")
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seeds", type=int, nargs="+", default=[2])
    prepare.add_argument("--output-latent-frames", type=int, default=16)

    snapshot = commands.add_parser("snapshot-protocol")
    snapshot.add_argument("--requests", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--protocol", required=True)

    initialize = commands.add_parser("init-run")
    initialize.add_argument("config", type=Path)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--source-root", type=Path, default=Path.cwd())

    plan = commands.add_parser("plan-legacy-run")
    plan.add_argument("config", type=Path)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--run-dir", type=Path, default=Path("runs"))
    plan.add_argument("--host", default="nav")
    plan.add_argument("--master-port", type=int, default=29571)

    run = commands.add_parser("run-legacy")
    run.add_argument("config", type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--cuda-ids", required=True)
    run.add_argument("--run-dir", type=Path, default=Path("runs"))
    run.add_argument("--host", default="nav")
    run.add_argument("--master-port", type=int, default=29571)
    run.add_argument("--sync", action="store_true")
    run.add_argument("--skip-existing", action="store_true")

    nav = commands.add_parser("nav-probe")
    nav.add_argument("--host", default="nav")

    difference = commands.add_parser("compare")
    difference.add_argument("--expected", type=Path, required=True)
    difference.add_argument("--actual", type=Path, required=True)
    difference.add_argument("--output", type=Path, required=True)

    index = commands.add_parser("build-index")
    index.add_argument("--runs", type=Path, default=Path("runs"))
    index.add_argument("--output", type=Path, default=Path("runs/index.json"))

    collect = commands.add_parser("collect-legacy-summary")
    collect.add_argument("--summary", type=Path, required=True)
    collect.add_argument("--case-id", required=True)
    collect.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-golden")
    validate.add_argument("golden", type=Path)

    seqpr = commands.add_parser("audit-seqpr")
    seqpr.add_argument("path", type=Path)

    masks = commands.add_parser("audit-mask-accounting")
    masks.add_argument("path", type=Path)

    video = commands.add_parser("inspect-video")
    video.add_argument("path", type=Path)
    video.add_argument("--output", type=Path)

    video_diff = commands.add_parser("compare-videos")
    video_diff.add_argument("--expected", type=Path, required=True)
    video_diff.add_argument("--actual", type=Path, required=True)
    video_diff.add_argument("--output", type=Path)

    dino = commands.add_parser("compare-dino")
    dino.add_argument("--expected", type=Path, required=True)
    dino.add_argument("--actual", type=Path, required=True)
    dino.add_argument("--model", type=Path, required=True)
    dino.add_argument("--device", default="cuda")
    dino.add_argument("--batch-size", type=int, default=8)
    dino.add_argument("--output", type=Path)

    bundle = commands.add_parser("build-artifact-bundle")
    bundle.add_argument("case_dir", type=Path)
    bundle.add_argument("--independent-dino", type=Path)
    bundle.add_argument("--official-dino", type=Path)
    bundle.add_argument("--output", type=Path, required=True)

    official_dino = commands.add_parser("run-official-dino")
    official_dino.add_argument("case_dir", type=Path)
    official_dino.add_argument("--benchmark-root", type=Path, required=True)
    official_dino.add_argument("--metadata", type=Path, help="Default: BENCHMARK_ROOT/metadata.jsonl")
    official_dino.add_argument("--annotations", type=Path, help="Default: the same public metadata.jsonl")
    official_dino.add_argument("--evaluator", type=Path, required=True)
    official_dino.add_argument("--python", type=Path, required=True)
    official_dino.add_argument("--owl-model", type=Path, required=True)
    official_dino.add_argument("--sam-model", type=Path, required=True)
    official_dino.add_argument("--dino-model", type=Path, required=True)
    official_dino.add_argument("--eval-root", type=Path, required=True)
    official_dino.add_argument("--device", default="cuda:0")
    official_dino.add_argument("--cuda-visible-devices", required=True)

    official_matrix = commands.add_parser("run-official-matrix")
    official_matrix.add_argument("--generation-plan", type=Path, required=True)
    official_matrix.add_argument("--benchmark-root", type=Path, required=True)
    official_matrix.add_argument("--metadata", type=Path, help="Default: BENCHMARK_ROOT/metadata.jsonl")
    official_matrix.add_argument("--annotations", type=Path, help="Default: the same public metadata.jsonl")
    official_matrix.add_argument("--evaluator", type=Path, required=True)
    official_matrix.add_argument("--python", type=Path, required=True)
    official_matrix.add_argument("--owl-model", type=Path, required=True)
    official_matrix.add_argument("--sam-model", type=Path, required=True)
    official_matrix.add_argument("--dino-model", type=Path, required=True)
    official_matrix.add_argument("--eval-root", type=Path, required=True)
    official_matrix.add_argument("--device", default="cuda:0")
    official_matrix.add_argument("--cuda-visible-devices", required=True)
    official_matrix.add_argument("--methods", nargs="*")
    official_matrix.add_argument("--case-ids", nargs="*")
    official_matrix.add_argument("--seeds", type=int, nargs="*")
    official_matrix.add_argument("--limit", type=int)
    official_matrix.add_argument("--batch-size", type=int, default=16)
    official_matrix.add_argument("--continue-on-error", action="store_true")
    official_mode = official_matrix.add_mutually_exclusive_group()
    official_mode.add_argument(
        "--execute", action="store_true",
        help="explicitly start GPU evaluation after writing the immutable plan",
    )
    official_mode.add_argument("--plan-only", action="store_true", help=argparse.SUPPRESS)

    aggregate_matrix = commands.add_parser("aggregate-production-matrix")
    aggregate_matrix.add_argument("--evaluation-plan", type=Path, required=True)
    aggregate_matrix.add_argument("--output", type=Path, required=True)

    audit_generation = commands.add_parser("audit-production-matrix")
    audit_generation.add_argument("--plan", type=Path, required=True)
    audit_generation.add_argument("--output", type=Path)

    audit_official = commands.add_parser("audit-official-matrix")
    audit_official.add_argument("--plan", type=Path, required=True)
    audit_official.add_argument("--output", type=Path)

    delivery = commands.add_parser("audit-delivery")
    delivery.add_argument("--project-root", type=Path, default=Path.cwd())
    delivery.add_argument("--generation-plan", type=Path, required=True)
    delivery.add_argument("--official-plan", type=Path, required=True)
    delivery.add_argument("--aggregate", type=Path, required=True)
    delivery.add_argument("--protocol-snapshot", type=Path, required=True)
    delivery.add_argument("--golden-dir", type=Path, required=True)
    delivery.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("/mnt/workspace/users/xiaozeqi/code/forcingkv_private_lingbotworld_lql_d206858"),
    )
    delivery.add_argument("--output", type=Path)

    matrix = commands.add_parser("run-production-matrix")
    matrix.add_argument("--configs", type=Path, nargs="+", required=True)
    matrix.add_argument("--seeds", type=int, nargs="+", required=True)
    matrix.add_argument("--output-root", type=Path, required=True)
    matrix.add_argument("--python", type=Path, required=True)
    matrix.add_argument("--cuda-visible-devices", default="0,1,2,3")
    matrix.add_argument("--methods", nargs="*")
    matrix.add_argument("--case-ids", nargs="*")
    matrix.add_argument("--limit", type=int)
    matrix.add_argument("--continue-on-error", action="store_true")
    matrix_mode = matrix.add_mutually_exclusive_group()
    matrix_mode.add_argument(
        "--execute", action="store_true",
        help="explicitly start GPU generation after writing the immutable plan",
    )
    matrix_mode.add_argument("--plan-only", action="store_true", help=argparse.SUPPRESS)

    benchmark_dino = commands.add_parser("audit-direct-bbox-dino")
    benchmark_dino.add_argument("--annotations", type=Path, required=True)
    benchmark_dino.add_argument("--case-id", required=True)
    benchmark_dino.add_argument("--video", type=Path, required=True)
    benchmark_dino.add_argument("--model", type=Path, required=True)
    benchmark_dino.add_argument("--frame-stride", type=int, default=4)
    benchmark_dino.add_argument("--device", default="cuda")
    benchmark_dino.add_argument("--batch-size", type=int, default=8)
    benchmark_dino.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command in {"run-official-dino", "run-official-matrix"}:
        args.metadata = args.metadata or args.benchmark_root / "metadata.jsonl"
        args.annotations = args.annotations or args.metadata
    if args.command == "prepare-benchmark":
        result = prepare_requests(args.benchmark_root, args.output, args.seeds,
                                  metadata=args.metadata, output_latent_frames=args.output_latent_frames)
    elif args.command == "snapshot-protocol":
        result = snapshot_protocol(args.requests, args.output, args.protocol)
    elif args.command == "init-run":
        config = load(args.config).normalized()
        root = Path(config["output_root"]) / args.run_id
        result = new_manifest(config, command=["cmbench", "run", str(args.config)], source_root=args.source_root)
        write_immutable_manifest(root / "manifest.json", result)
        (root / "config.resolved.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    elif args.command == "plan-legacy-run":
        config = load(args.config).normalized()
        root = args.run_dir / args.run_id
        request = root / "request.jsonl"
        select_request(Path(config["request_file"]), config["case_id"], request)
        result = remote_plan(config, args.run_id, request, NavSettings(host=args.host, master_port=args.master_port))
        write_immutable_manifest(root / "remote-plan.json", result)
    elif args.command == "nav-probe":
        result = probe_nav(NavSettings(host=args.host))
    elif args.command == "run-legacy":
        config = load(args.config).normalized()
        root = args.run_dir / args.run_id
        store = RunStore(root)
        success = store.successful_attempt()
        if success is not None:
            if args.skip_existing:
                _json({"status": "skipped", "reason": "successful attempt exists", "attempt": success.number})
                return
            raise FileExistsError(f"successful run already exists: {root}; use --skip-existing")
        request = root / "request.jsonl"
        select_request(Path(config["request_file"]), config["case_id"], request)
        plan_value = remote_plan(config, args.run_id, request, NavSettings(host=args.host, master_port=args.master_port))
        manifest_config = dict(config)
        manifest_config["source_request_file"] = manifest_config["request_file"]
        manifest_config["request_file"] = str(request)
        identity = new_manifest(manifest_config, command=plan_value["command"], source_root=Path.cwd())
        identity["remote_plan"] = plan_value
        store.ensure_identity(identity)
        attempt = store.start_attempt({"schema_version": 1, "remote_plan": plan_value, "cuda_visible_devices": args.cuda_ids})
        result = execute_legacy(plan_value, cuda_ids=args.cuda_ids, log_path=attempt.root / "worker.log")
        if args.sync and result["status"] == "completed":
            result["sync"] = sync_legacy_output(plan_value, attempt.root / "remote-output")
        store.finish_attempt(attempt, result)
    elif args.command == "compare":
        result = compare(args.expected, args.actual, args.output)
    elif args.command == "collect-legacy-summary":
        result = collect_legacy_summary(args.summary, case_id=args.case_id, output=args.output)
    elif args.command == "validate-golden":
        if args.golden.is_dir():
            raise ValueError("validate-golden expects one candidate/qualified JSON file, not a directory")
        result = validate_golden(args.golden)
    elif args.command == "audit-seqpr":
        result = audit_seqpr(args.path)
    elif args.command == "audit-mask-accounting":
        result = audit_mask_accounting(args.path)
    elif args.command == "inspect-video":
        result = inspect_video(args.path)
    elif args.command == "compare-videos":
        result = compare_videos(args.expected, args.actual)
    elif args.command == "compare-dino":
        result = compare_dino(args.expected, args.actual, model_path=args.model, device=args.device, batch_size=args.batch_size)
    elif args.command == "build-artifact-bundle":
        result = build_artifact_bundle(
            args.case_dir, independent_dino=args.independent_dino, official_dino=args.official_dino
        )
    elif args.command == "run-official-dino":
        result = run_official_dino(
            case_dir=args.case_dir,
            benchmark_root=args.benchmark_root,
            metadata=args.metadata,
            annotations=args.annotations,
            evaluator=args.evaluator,
            python=args.python,
            owl_model=args.owl_model,
            sam_model=args.sam_model,
            dino_model=args.dino_model,
            eval_root=args.eval_root,
            device=args.device,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    elif args.command == "run-official-matrix":
        result = build_official_matrix_plan(
            args.generation_plan,
            benchmark_root=args.benchmark_root,
            metadata=args.metadata,
            annotations=args.annotations,
            evaluator=args.evaluator,
            python=args.python,
            owl_model=args.owl_model,
            sam_model=args.sam_model,
            dino_model=args.dino_model,
            eval_root=args.eval_root,
            cuda_visible_devices=args.cuda_visible_devices,
            device=args.device,
        )
        args.eval_root.mkdir(parents=True, exist_ok=True)
        plan_path = args.eval_root / "official-evaluation.plan.json"
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if plan_path.exists() and plan_path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"official evaluation plan identity changed: {plan_path}")
        plan_path.write_text(encoded, encoding="utf-8")
        if args.execute:
            execution = run_official_matrix(
                result,
                methods=args.methods,
                case_ids=args.case_ids,
                seeds=args.seeds,
                limit=args.limit,
                batch_size=args.batch_size,
                continue_on_error=args.continue_on_error,
            )
            result = {"plan": result, "execution": execution}
    elif args.command == "aggregate-production-matrix":
        plan = json.loads(args.evaluation_plan.read_text(encoding="utf-8"))
        result = bind_aggregate_to_official_plan(
            aggregate_official_matrix(plan), args.evaluation_plan
        )
        write_immutable_manifest(args.output, result)
    elif args.command == "audit-production-matrix":
        result = audit_generation_matrix(json.loads(args.plan.read_text(encoding="utf-8")))
    elif args.command == "audit-official-matrix":
        result = audit_official_matrix_progress(json.loads(args.plan.read_text(encoding="utf-8")))
    elif args.command == "audit-delivery":
        result = audit_delivery(
            project_root=args.project_root,
            generation_plan_path=args.generation_plan,
            official_plan_path=args.official_plan,
            aggregate_path=args.aggregate,
            protocol_snapshot=args.protocol_snapshot,
            golden_dir=args.golden_dir,
            legacy_root=args.legacy_root,
        )
    elif args.command == "run-production-matrix":
        result = build_matrix_plan(
            args.configs,
            seeds=args.seeds,
            output_root=args.output_root,
            source_root=Path.cwd(),
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        plan_path = args.output_root / "matrix.plan.json"
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if plan_path.exists() and plan_path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"matrix plan identity changed: {plan_path}")
        plan_path.write_text(encoded, encoding="utf-8")
        if args.execute:
            execution = run_matrix(
                result,
                python=args.python,
                cuda_visible_devices=args.cuda_visible_devices,
                source_root=Path.cwd(),
                limit=args.limit,
                methods=args.methods,
                case_ids=args.case_ids,
                continue_on_error=args.continue_on_error,
            )
            result = {"plan": result, "execution": execution}
    elif args.command == "audit-direct-bbox-dino":
        result = evaluate_direct_bbox_dino(
            annotation_path=args.annotations,
            case_id=args.case_id,
            generated_video=args.video,
            model_path=args.model,
            frame_stride=args.frame_stride,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        result = build(args.runs, args.output)
    if getattr(args, "output", None) is not None and args.command in {"inspect-video", "compare-videos", "compare-dino", "build-artifact-bundle", "audit-direct-bbox-dino", "audit-production-matrix", "audit-official-matrix", "audit-delivery"}:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json(result)


if __name__ == "__main__":
    main()
