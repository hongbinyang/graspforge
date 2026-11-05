"""Safe artifact discovery, retention and protection checks."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from graspforge.artifacts import PROTECTION_MARKER
from graspforge.cli import main


def make_run(path, *, kind="scenario", days=0, complete=True):
    path.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "created_utc": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    if complete:
        (
            path
            / ("grasp_metrics.json" if kind == "benchmark" else "grasp_episode.npz")
        ).write_bytes(b"data")
    return path


def test_listing_preview_and_retention(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts" / "runs"
    old = make_run(root / "scenario" / "old", days=40)
    recent = make_run(
        root / "nominal" / "recent", kind="nominal", days=2, complete=False
    )
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    assert str(old) in out and "incomplete" in out
    assert main(["artifacts", "clean", "--older-than", "30d"]) == 0
    assert old.exists() and recent.exists()
    assert "Preview only" in capsys.readouterr().out
    assert main(["artifacts", "clean", "--older-than", "30d", "--apply"]) == 0
    assert not old.exists() and recent.exists()


def test_keep_newest_and_kind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts" / "runs"
    old = make_run(root / "old", days=4)
    new = make_run(root / "new", days=2)
    other = make_run(root / "benchmark", kind="benchmark", days=10)
    assert (
        main(["artifacts", "clean", "--kind", "scenario", "--keep", "1", "--apply"])
        == 0
    )
    assert not old.exists() and new.exists() and other.exists()


def test_protection_blocks_all_deletion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts" / "runs"
    safe = make_run(root / "safe")
    protected = make_run(root / "protected")
    (protected / PROTECTION_MARKER).touch()
    with pytest.raises(SystemExit, match="protected run"):
        main(["artifacts", "clean", "--apply"])
    assert safe.exists() and protected.exists()
    with pytest.raises(SystemExit, match="protected run"):
        main(["artifacts", "clean"])


def test_readonly_and_unmanaged_files_block_cleanup(tmp_path):
    run = make_run(tmp_path / "custom")
    episode = run / "grasp_episode.npz"
    episode.chmod(0o444)
    with pytest.raises(SystemExit, match="protected or non-writable file"):
        main(["artifacts", "clean", "--path", str(run), "--apply"])
    episode.chmod(0o644)
    (run / "notes.txt").write_text("keep")
    with pytest.raises(SystemExit, match="unmanaged or protected item"):
        main(["artifacts", "clean", "--path", str(run), "--apply"])
    assert episode.exists() and (run / "notes.txt").exists()


def test_reference_and_symlink_are_protected(tmp_path):
    ref = make_run(tmp_path / "artifacts" / "reference" / "ref")
    with pytest.raises(SystemExit, match="Protected reference"):
        main(["artifacts", "clean", "--path", str(ref), "--apply"])
    link = tmp_path / "link"
    link.symlink_to(ref, target_is_directory=True)
    with pytest.raises(SystemExit, match="regular run directory"):
        main(["artifacts", "clean", "--path", str(link), "--apply"])
    assert ref.exists()


def test_rejects_bad_retention_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="positive number of days"):
        main(["artifacts", "clean", "--older-than", "soon"])
    with pytest.raises(SystemExit, match="non-negative"):
        main(["artifacts", "clean", "--keep", "-1"])


def test_unreadable_or_invalid_manifest_blocks_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts" / "runs"
    valid = make_run(root / "valid")
    invalid = root / "unknown"
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("invalid json")
    with pytest.raises(SystemExit, match="Cannot inspect run"):
        main(["artifacts", "clean", "--apply"])
    assert valid.exists()
