"""Lifecycle checks for managed scenario videos."""

import json
from pathlib import Path

import pytest

from graspforge.cli import main
from graspforge.video_library import MARKER


def test_generate_reuse_list_and_safe_cleanup(tmp_path, monkeypatch, capsys):
    import graspforge.video_library as library

    calls = []

    def fake_render(episode, output, **kwargs):
        assert Path(episode).is_file()
        calls.append(output)
        output.write_bytes(b"fake video")

    monkeypatch.setattr(library, "render_video", fake_render)
    folder = tmp_path / "videos"
    args = ["videos", "generate", "--case", "nominal", "--directory", str(folder)]
    assert main(args) == 0
    assert main(args) == 0
    assert len(calls) == 1
    info = json.loads((folder / "nominal.json").read_text())
    assert info["scenario"]["scenario"] == "nominal"
    assert info["success"] is True
    assert len(info["model_sha256"]) == 64
    assert main(["videos", "list", "--directory", str(folder)]) == 0
    assert "nominal.mp4" in capsys.readouterr().out
    unrelated = folder / "my-own-video.mp4"
    unrelated.write_bytes(b"keep")
    assert main(["videos", "clean", "--directory", str(folder)]) == 0
    assert unrelated.read_bytes() == b"keep"
    assert not (folder / "nominal.mp4").exists()
    assert not (folder / "nominal.json").exists()
    assert not (folder / MARKER).exists()
    assert main(["videos", "clean", "--directory", str(folder)]) == 0


def test_refuses_existing_unmanaged_files(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "nominal.mp4").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        main(["videos", "generate", "--case", "nominal", "--directory", str(folder)])
    assert (folder / "nominal.mp4").read_bytes() == b"keep"


def test_failed_render_can_be_retried_and_cleaned(tmp_path, monkeypatch):
    import graspforge.video_library as library

    folder = tmp_path / "videos"

    def fail_render(*args, **kwargs):
        raise RuntimeError("graphics unavailable")

    monkeypatch.setattr(library, "render_video", fail_render)
    with pytest.raises(SystemExit, match="graphics unavailable"):
        main(["videos", "generate", "--case", "nominal", "--directory", str(folder)])
    assert "nominal" in json.loads((folder / MARKER).read_text())["cases"]
    monkeypatch.setattr(
        library,
        "render_video",
        lambda episode, output, **kwargs: output.write_bytes(b"video"),
    )
    assert (
        main(["videos", "generate", "--case", "nominal", "--directory", str(folder)])
        == 0
    )
    assert (folder / "nominal.mp4").exists()
    assert main(["videos", "clean", "--directory", str(folder)]) == 0
    assert not folder.exists()


def test_clean_rejects_tampered_index(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / MARKER).write_text(json.dumps({"version": 1, "cases": ["../other"]}))
    with pytest.raises(ValueError, match="Invalid video library index"):
        main(["videos", "clean", "--directory", str(folder)])


def test_setting_change_regenerates_and_failure_preserves_video(tmp_path, monkeypatch):
    import graspforge.video_library as library

    folder = tmp_path / "videos"
    args = ["videos", "generate", "--case", "nominal", "--directory", str(folder)]
    monkeypatch.setattr(
        library,
        "render_video",
        lambda episode, output, **kwargs: output.write_bytes(
            str(kwargs["fps"]).encode()
        ),
    )
    assert main(args) == 0
    video = folder / "nominal.mp4"
    assert video.read_bytes() == b"30"

    def fail(*args, **kwargs):
        raise RuntimeError("graphics unavailable")

    monkeypatch.setattr(library, "render_video", fail)
    with pytest.raises(SystemExit, match="graphics unavailable"):
        main(args + ["--fps", "24"])
    assert video.read_bytes() == b"30"
    monkeypatch.setattr(
        library,
        "render_video",
        lambda episode, output, **kwargs: output.write_bytes(
            str(kwargs["fps"]).encode()
        ),
    )
    assert main(args + ["--fps", "24"]) == 0
    assert video.read_bytes() == b"24"


def test_video_clean_reports_protection_before_deleting(tmp_path, monkeypatch):
    import graspforge.video_library as library

    folder = tmp_path / "videos"
    monkeypatch.setattr(
        library,
        "render_video",
        lambda episode, output, **kwargs: output.write_bytes(b"video"),
    )
    assert (
        main(["videos", "generate", "--case", "nominal", "--directory", str(folder)])
        == 0
    )
    video = folder / "nominal.mp4"
    video.chmod(0o444)
    with pytest.raises(SystemExit, match="protected or non-writable file"):
        main(["videos", "clean", "--directory", str(folder)])
    assert video.exists() and (folder / "nominal.json").exists()
