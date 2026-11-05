"""Discover and carefully remove complete, manifest-backed experiment runs."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat

KINDS = ("nominal", "benchmark", "scenario")
MANAGED_FILES = frozenset(
    {
        "manifest.json",
        "grasp_episode.npz",
        "grasp_metrics.json",
        "grasp_diagnostics.png",
        "grasp_final.png",
        "grasp_video.mp4",
    }
)
PROTECTION_MARKER = ".graspforge-protected"


@dataclass(frozen=True)
class ArtifactRun:
    path: Path
    kind: str
    created: datetime
    size: int
    complete: bool


def _is_reference(path):
    parts = path.resolve().parts
    return any(
        parts[i : i + 2] == ("artifacts", "reference") for i in range(len(parts) - 1)
    )


def _read_run(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Not a regular run directory: {path}")
    if _is_reference(path):
        raise PermissionError(f"Protected reference artifacts: {path}")
    manifest = path / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"Missing or unsafe run manifest: {path}")
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info.get("schema_version") != 1 or info.get("kind") not in KINDS:
            raise ValueError("unrecognized manifest")
        created = datetime.fromisoformat(info["created_utc"])
        if created.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        created = created.astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run manifest: {manifest}: {exc}") from exc
    files = list(path.iterdir())
    size = sum(
        item.stat().st_size
        for item in files
        if item.is_file() and not item.is_symlink()
    )
    expected = (
        "grasp_metrics.json" if info["kind"] == "benchmark" else "grasp_episode.npz"
    )
    return ArtifactRun(path, info["kind"], created, size, (path / expected).is_file())


def discover_runs(root, paths=(), *, strict=False):
    """Find runs beneath root, or inspect explicitly selected custom run directories."""
    if paths:
        return [_read_run(path) for path in dict.fromkeys(Path(p) for p in paths)]
    root = Path(root)
    if not root.exists():
        return []
    runs = []

    def on_walk_error(exc):
        if strict:
            raise PermissionError(f"Cannot inspect artifact directory: {exc}") from exc
        print(f"Cannot inspect artifact directory: {exc}")

    for directory, dirs, files in os.walk(
        root, followlinks=False, onerror=on_walk_error
    ):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        if "manifest.json" in files:
            try:
                runs.append(_read_run(Path(directory)))
            except (ValueError, OSError) as exc:
                # A manifest-like file must never be treated as permission to delete.
                if strict:
                    raise ValueError(f"Cannot inspect run {directory}: {exc}") from exc
                print(f"Unrecognized run, skipped: {directory}: {exc}")
            dirs[:] = []
    return runs


def select_runs(runs, *, kind=None, older_than=None, keep=0, now=None):
    if keep < 0:
        raise ValueError("--keep must be non-negative")
    if older_than is not None and not re.fullmatch(r"[1-9][0-9]*d", older_than):
        raise ValueError("--older-than must be a positive number of days, e.g. 30d")
    selected = sorted(
        (r for r in runs if kind is None or r.kind == kind),
        key=lambda r: (r.created, str(r.path)),
        reverse=True,
    )
    selected = selected[keep:]
    if older_than:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(
            days=int(older_than[:-1])
        )
        selected = [r for r in selected if r.created < cutoff]
    return selected


def _writable(path):
    st = path.stat()
    immutable = (
        getattr(stat, "UF_IMMUTABLE", 0)
        | getattr(stat, "SF_IMMUTABLE", 0)
        | getattr(stat, "UF_APPEND", 0)
        | getattr(stat, "SF_APPEND", 0)
    )
    return (
        bool(st.st_mode & 0o222)
        and not (getattr(st, "st_flags", 0) & immutable)
        and os.access(path, os.W_OK)
    )


def protection_errors(run):
    """Explain anything that prevents treating a run as an owned, removable unit."""
    path = run.path
    errors = []
    try:
        if _is_reference(path) or path.is_symlink():
            errors.append(f"protected or symlinked directory: {path}")
            return errors
        if (path / PROTECTION_MARKER).exists() or (
            path / PROTECTION_MARKER
        ).is_symlink():
            errors.append(f"protected run: {path} ({PROTECTION_MARKER})")
        if not _writable(path) or not _writable(path.parent):
            errors.append(f"protected or non-writable directory: {path}")
        for item in path.iterdir():
            if item.name == PROTECTION_MARKER:
                continue
            if (
                item.is_symlink()
                or not item.is_file()
                or item.name not in MANAGED_FILES
            ):
                errors.append(f"unmanaged or protected item: {item}")
            elif not _writable(item):
                errors.append(f"protected or non-writable file: {item}")
    except OSError as exc:
        errors.append(f"cannot inspect protected run {path}: {exc}")
    return errors


def list_runs(runs):
    if not runs:
        print("No matching experiment runs.")
    for run in runs:
        status = "complete" if run.complete else "incomplete"
        protected = " protected" if protection_errors(run) else ""
        print(
            f"{run.created.isoformat()}  {run.kind:9}  {status}{protected}  {run.size} bytes  {run.path}"
        )


def clean_runs(runs, *, apply=False):
    errors = [message for run in runs for message in protection_errors(run)]
    if errors:
        raise PermissionError(
            "Cleanup blocked; no runs were deleted:\n" + "\n".join(errors)
        )
    if not runs:
        print("No matching experiment runs.")
        return
    for run in runs:
        print(
            f"{'Deleting' if apply else 'Would delete'}: {run.path} ({run.size} bytes)"
        )
    if not apply:
        print("Preview only. Add --apply to delete these runs.")
        return
    for run in runs:
        # Recheck immediately before deletion, and never recurse into unknown content.
        errors = protection_errors(run)
        if errors:
            raise PermissionError(
                "Cleanup stopped; earlier runs may already have been deleted:\n"
                + "\n".join(errors)
            )
        try:
            for item in run.path.iterdir():
                if (
                    item.name not in MANAGED_FILES
                    or item.is_symlink()
                    or not item.is_file()
                ):
                    raise OSError(f"unexpected or protected item appeared: {item}")
                if not _writable(item):
                    raise PermissionError(f"file became protected: {item}")
                item.unlink()
            run.path.rmdir()
        except OSError as exc:
            raise OSError(
                f"Cleanup stopped while deleting {run.path}; earlier files or runs may already have been deleted: {exc}"
            ) from exc
