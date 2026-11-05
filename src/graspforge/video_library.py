"""Manage a reproducible set of scenario videos as a single library."""

from pathlib import Path
import hashlib
from importlib.metadata import version
import json
import platform

import mujoco
import tempfile

import numpy as np

from graspforge.artifacts import PROTECTION_MARKER, _writable
from graspforge.contacts import MODEL_PATH
from graspforge.controller import ControllerConfig
from graspforge.evaluation import SuccessCriteria, save_episode
from graspforge.scenarios import SCENARIOS, run_scenario
from graspforge.video import render_video

MARKER = ".graspforge-video-library.json"


def _load_library(directory, *, create=False):
    directory = Path(directory)
    marker = directory / MARKER
    if not marker.exists():
        if not create:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        index = {"version": 1, "cases": []}
        marker.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return index
    index = json.loads(marker.read_text(encoding="utf-8"))
    if (
        index.get("version") != 1
        or not isinstance(index.get("cases"), list)
        or any(case not in SCENARIOS for case in index["cases"])
    ):
        raise ValueError(f"Invalid video library index: {marker}")
    return index


def _write_index(directory, index):
    marker = Path(directory) / MARKER
    # Replace atomically so an interrupted run does not leave a partial index.
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, encoding="utf-8", delete=False
    ) as file:
        tmp = Path(file.name)
        json.dump(index, file, indent=2)
        file.write("\n")
    tmp.replace(marker)


def generate_videos(
    directory, cases, *, fps=30, camera="overview", width=640, height=480
):
    """Create missing videos, retaining a metadata sidecar for each scenario."""
    directory = Path(directory)
    if any(case not in SCENARIOS for case in cases):
        raise ValueError("unknown video scenario")
    settings = {"fps": fps, "camera": camera, "width": width, "height": height}
    # Validate settings before generating a potentially expensive simulation.
    if (
        not isinstance(fps, int)
        or not 1 <= fps <= 120
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
    ):
        raise ValueError(
            "fps must be 1–120 and dimensions must be positive even numbers"
        )
    fingerprint = {
        "video_settings": settings,
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": {
            "graspforge": version("graspforge"),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    index = _load_library(directory, create=True)
    for case in dict.fromkeys(cases):
        video = directory / f"{case}.mp4"
        metadata = directory / f"{case}.json"
        owned = case in index["cases"]
        if not owned and (video.exists() or metadata.exists()):
            raise FileExistsError(
                f"Refusing to overwrite existing files for {case} in {directory}"
            )
        if owned and video.is_file() and metadata.is_file():
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if all(saved.get(key) == value for key, value in fingerprint.items()):
                print(f"  kept: {video}")
                continue
        if not owned:
            index["cases"].append(case)
            _write_index(directory, index)
        log, report, params = run_scenario(case)
        info = {
            "scenario": params,
            "controller_config": vars(ControllerConfig()),
            "success_criteria": vars(SuccessCriteria()),
            "success": report.success if report else None,
            "reason": str(log["reason"]),
            **fingerprint,
        }
        # Stage both files on the same filesystem. Failed renders leave any
        # previous completed video in place and can be retried safely.
        with tempfile.TemporaryDirectory(
            prefix=".graspforge-video-", dir=directory
        ) as tmp:
            tmp = Path(tmp)
            episode = tmp / "episode.npz"
            if report is None:
                np.savez_compressed(episode, **log)
            else:
                save_episode(episode, log, report)
            staged_video = tmp / "video.mp4"
            render_video(episode, staged_video, **settings)
            staged_metadata = tmp / "metadata.json"
            staged_metadata.write_text(
                json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staged_video.replace(video)
            staged_metadata.replace(metadata)
        print(f"  created: {video}")
    return directory


def clean_videos(directory):
    """Remove only indexed videos and their sidecars; leave unrelated files alone."""
    directory = Path(directory)
    index = _load_library(directory)
    if index is None:
        print(f"No managed video library at {directory}")
        return 0
    selected = [
        directory / f"{case}{suffix}"
        for case in index["cases"]
        for suffix in (".mp4", ".json")
    ]
    errors = []
    try:
        if (
            directory.is_symlink()
            or (directory / PROTECTION_MARKER).exists()
            or (directory / PROTECTION_MARKER).is_symlink()
        ):
            errors.append(f"protected video library: {directory}")
        if not _writable(directory) or not _writable(directory.parent):
            errors.append(f"protected or non-writable directory: {directory}")
        for path in [directory / MARKER, *selected]:
            if path.is_symlink():
                errors.append(f"protected symlink: {path}")
            elif path.exists() and (not path.is_file() or not _writable(path)):
                errors.append(f"protected or non-writable file: {path}")
    except OSError as exc:
        errors.append(f"cannot inspect video library {directory}: {exc}")
    if errors:
        raise PermissionError(
            "Video cleanup blocked; no files were deleted:\n" + "\n".join(errors)
        )
    count = 0
    for path in selected:
        if path.is_file():
            path.unlink()
            count += 1
    (directory / MARKER).unlink()
    try:
        directory.rmdir()
    except OSError:
        pass  # Preserve any unrelated files and directories.
    print(f"Removed {count} managed video files from {directory}")
    return count


def list_videos(directory):
    directory = Path(directory)
    index = _load_library(directory)
    if index is None or not index["cases"]:
        print(f"No managed videos at {directory}")
        return
    for case in index["cases"]:
        path = directory / f"{case}.mp4"
        print(f"{case}: {path if path.is_file() else 'incomplete'}")
