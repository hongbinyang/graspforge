"""Command-line entry points for running and evaluating grasp experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from uuid import uuid4

import mujoco
import numpy as np

from graspforge.artifacts import (
    KINDS,
    clean_runs,
    discover_runs,
    list_runs,
    select_runs,
)
from graspforge.contacts import MODEL_PATH, build_index, load_model
from graspforge.controller import (
    NOMINAL_DISTURBANCE,
    ControllerConfig,
    Phase,
    print_timeline,
    run_grasp,
)
from graspforge.scenarios import SCENARIOS, run_scenario
from graspforge.video import render_video
from graspforge.video_library import clean_videos, generate_videos, list_videos
from graspforge.evaluation import (
    MIN_SUCCESSES,
    SuccessCriteria,
    evaluate_episode,
    plot_diagnostics,
    print_report,
    print_suite_table,
    render_final,
    run_suite,
    save_episode,
    write_metrics,
)


def _output_dir(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _prepare_output(path: Path | None, kind: str, filenames: tuple[str, ...]) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = _output_dir(f"artifacts/runs/{kind}/{stamp}-{uuid4().hex[:8]}")
    path.mkdir(parents=True, exist_ok=True)
    existing = [name for name in filenames if (path / name).exists()]
    if existing:
        raise FileExistsError(
            f"output files already exist in {path}: {', '.join(existing)}"
        )
    return path


def _write_manifest(
    output: Path,
    kind: str,
    config: ControllerConfig,
    criteria: SuccessCriteria,
    **parameters,
) -> None:
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "model": {
            "asset": MODEL_PATH.name,
            "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "versions": {
            "graspforge": version("graspforge"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "controller_config": asdict(config),
        "success_criteria": asdict(criteria),
        "parameters": parameters,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")


def inspect_model(_: argparse.Namespace) -> int:
    model = load_model()
    index = build_index(model)
    print(f"MuJoCo {mujoco.__version__}")
    print(f"model: {MODEL_PATH}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nsensordata={model.nsensordata}")
    print(f"timestep={model.opt.timestep:g} s")
    print(
        "actuators: "
        f"lift={index.lift_act}, left={index.left_act}, right={index.right_act}"
    )
    return 0


def run_nominal(args: argparse.Namespace) -> int:
    output = _prepare_output(
        args.output,
        "nominal",
        (
            "grasp_episode.npz",
            "grasp_diagnostics.png",
            "grasp_final.png",
            "manifest.json",
        ),
    )
    model = load_model()
    data = mujoco.MjData(model)
    index = build_index(model)
    config = ControllerConfig()
    criteria = SuccessCriteria()

    # One model/data pair owns the entire episode, including the optional final render.
    log = run_grasp(
        model,
        data,
        index,
        config,
        disturbance=NOMINAL_DISTURBANCE,
    )
    report = evaluate_episode(
        log,
        criteria,
        config.hold_time,
        NOMINAL_DISTURBANCE,
    )

    _write_manifest(
        output, "nominal", config, criteria, disturbance=asdict(NOMINAL_DISTURBANCE)
    )
    episode_path = output / "grasp_episode.npz"
    save_episode(episode_path, log, report)
    plot_diagnostics(episode_path, output / "grasp_diagnostics.png")
    if args.render:
        render_final(output / "grasp_final.png", model, data, camera=args.camera)

    print_timeline(log)
    print_report(report)
    print(f"artifacts: {output}")
    return 0 if report.success and Phase(int(log["phase"][-1])) is Phase.DONE else 1


def run_benchmark(args: argparse.Namespace) -> int:
    output = _prepare_output(
        args.output, "benchmark", ("grasp_metrics.json", "manifest.json")
    )
    config = ControllerConfig()
    criteria = SuccessCriteria()
    seeds = range(args.seed_start, args.seed_start + args.trials)

    # Every trial constructs isolated MuJoCo model, data, and controller instances.
    results = run_suite(seeds, config, criteria)
    print_suite_table(results)
    required = args.min_successes
    _write_manifest(
        output, "benchmark", config, criteria, seeds=list(seeds), min_successes=required
    )
    metrics_path = output / "grasp_metrics.json"
    write_metrics(results, config, criteria, metrics_path, min_successes=required)
    successes = sum(result.report.success for result in results)
    print(f"successes: {successes}/{args.trials}; required: {required}")
    print(f"metrics: {metrics_path}")
    return 0 if successes >= required else 1


def run_physical_scenario(args: argparse.Namespace) -> int:
    log, report, parameters = run_scenario(
        args.case,
        friction=args.friction,
        force=args.force,
        offset=args.offset,
        yaw=args.yaw,
        seed=args.seed,
    )
    output = _prepare_output(
        args.output, "scenarios", ("grasp_episode.npz", "manifest.json")
    )
    config = ControllerConfig()
    criteria = SuccessCriteria()
    _write_manifest(output, "scenario", config, criteria, **parameters)
    episode_path = output / "grasp_episode.npz"
    if report is None:
        np.savez_compressed(episode_path, **log)
        print("Squeeze and release complete")
    else:
        save_episode(episode_path, log, report)
        print_timeline(log)
        print_report(report)
    print(f"episode: {episode_path}")
    # Failure scenarios are successful experiments, even when the grasp fails.
    return 0


def replay_video(args: argparse.Namespace) -> int:
    if not args.episode.is_file():
        raise SystemExit(
            f"Episode file not found: {args.episode}\n"
            "Run 'graspforge scenario nominal' first, then use the episode path it prints."
        )
    output = args.output or args.episode.with_name("grasp_video.mp4")
    render_video(
        args.episode,
        output,
        fps=args.fps,
        camera=args.camera,
        width=args.width,
        height=args.height,
    )
    print(f"video: {output}")
    return 0


def manage_videos(args: argparse.Namespace) -> int:
    if args.action == "clean":
        try:
            clean_videos(args.directory)
        except OSError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.action == "list":
        list_videos(args.directory)
    else:
        try:
            generate_videos(
                args.directory, args.case or SCENARIOS, fps=args.fps, camera=args.camera
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    return 0


def manage_artifacts(args: argparse.Namespace) -> int:
    try:
        runs = discover_runs(
            Path.cwd() / "artifacts" / "runs",
            args.path or (),
            strict=args.action == "clean",
        )
        runs = select_runs(
            runs,
            kind=args.kind,
            older_than=getattr(args, "older_than", None),
            keep=getattr(args, "keep", 0),
        )
        if args.action == "list":
            list_runs(runs)
        else:
            clean_runs(runs, apply=args.apply)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graspforge",
        description="Run contact-rich parallel-gripper experiments in MuJoCo.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="print compiled model dimensions")
    inspect.set_defaults(handler=inspect_model)

    run = commands.add_parser("run", help="run and evaluate the nominal grasp")
    run.add_argument("--output", type=_output_dir, default=None)
    run.add_argument("--render", action="store_true", help="save the final RGB frame")
    run.add_argument(
        "--camera", default="overview", help="named MJCF camera used by --render"
    )
    run.set_defaults(handler=run_nominal)

    benchmark = commands.add_parser("benchmark", help="run seeded robustness trials")
    benchmark.add_argument("--trials", type=int, default=20)
    benchmark.add_argument("--seed-start", type=int, default=0)
    benchmark.add_argument("--min-successes", type=int, default=None)
    benchmark.add_argument("--output", type=_output_dir, default=None)
    benchmark.set_defaults(handler=run_benchmark)
    scenario = commands.add_parser(
        "scenario", help="run a physical inspection scenario"
    )
    scenario.add_argument("case", choices=SCENARIOS)
    scenario.add_argument(
        "--friction",
        type=float,
        default=0.15,
        help="cube friction for friction scenario",
    )
    scenario.add_argument(
        "--force",
        type=float,
        default=15.0,
        help="push magnitude in N for disturbance scenario",
    )
    scenario.add_argument(
        "--offset",
        type=float,
        default=0.015,
        help="cube x offset in m for misaligned scenario",
    )
    scenario.add_argument(
        "--yaw",
        type=float,
        default=5.0,
        help="cube yaw in degrees for misaligned scenario",
    )
    scenario.add_argument(
        "--seed", type=int, default=0, help="seed for seeded scenario"
    )
    scenario.add_argument("--output", type=_output_dir, default=None)
    scenario.set_defaults(handler=run_physical_scenario)

    replay = commands.add_parser("video", help="render an existing episode to MP4")
    replay.add_argument("episode", type=_output_dir)
    replay.add_argument("--output", type=_output_dir, default=None)
    replay.add_argument("--fps", type=int, default=30)
    replay.add_argument("--camera", default="overview")
    replay.add_argument("--width", type=int, default=640)
    replay.add_argument("--height", type=int, default=480)
    replay.set_defaults(handler=replay_video)
    videos = commands.add_parser(
        "videos", help="generate and manage a scenario video library"
    )
    video_actions = videos.add_subparsers(dest="action", required=True)
    for action in ("generate", "list", "clean"):
        sub = video_actions.add_parser(action)
        sub.add_argument(
            "--directory",
            type=_output_dir,
            default=_output_dir("artifacts/runs/videos"),
        )
        sub.set_defaults(handler=manage_videos)
        if action == "generate":
            sub.add_argument(
                "--case",
                action="append",
                choices=SCENARIOS,
                help="generate only this scenario; repeat to select several",
            )
            sub.add_argument("--fps", type=int, default=30)
            sub.add_argument("--camera", default="overview")
    artifacts = commands.add_parser("artifacts", help="list and clean experiment runs")
    artifact_actions = artifacts.add_subparsers(dest="action", required=True)
    for action in ("list", "clean"):
        sub = artifact_actions.add_parser(action)
        sub.add_argument(
            "--path",
            action="append",
            type=lambda p: Path(p).expanduser().absolute(),
            help="explicit run directory, including a custom output location; repeatable",
        )
        sub.add_argument("--kind", choices=KINDS)
        sub.set_defaults(handler=manage_artifacts)
        if action == "clean":
            sub.add_argument(
                "--older-than", help="only runs older than N days, e.g. 30d"
            )
            sub.add_argument(
                "--keep",
                type=int,
                default=0,
                help="preserve this many newest matching runs",
            )
            sub.add_argument(
                "--apply",
                action="store_true",
                help="delete the selected runs; default is preview",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "trials", 1) <= 0:
        raise SystemExit("--trials must be positive")
    if args.command == "benchmark":
        if args.seed_start < 0:
            raise SystemExit("--seed-start must be non-negative")
        if args.min_successes is None:
            args.min_successes = min(MIN_SUCCESSES, args.trials)
        elif not 0 <= args.min_successes <= args.trials:
            raise SystemExit("--min-successes must be between 0 and --trials")
    return args.handler(args)
