"""Exercise real scenario outcomes and the replay/encoding boundary."""

import json

import numpy as np
import pytest

from graspforge.cli import main
from graspforge.controller import Phase
from graspforge.scenarios import run_scenario
from graspforge.video import frame_indices, render_video


@pytest.mark.parametrize(
    "case,phase,reason",
    [
        ("nominal", Phase.DONE, "hold complete"),
        ("overload", Phase.FAILED, "contact lost"),
        ("missing", Phase.FAILED, "close timeout"),
    ],
)
def test_physical_scenarios(case, phase, reason):
    log, report, params = run_scenario(case)
    assert int(log["phase"][-1]) == phase
    assert str(log["reason"]).startswith(reason)
    assert report.success == (phase == Phase.DONE)
    assert params["scenario"] == case


def test_friction_and_seeded_variation_are_recorded():
    log, report, params = run_scenario("friction", friction=0.15)
    assert params["cube_friction"] == 0.15
    assert not report.success
    assert log["phase"][-1] == Phase.FAILED
    _, _, params = run_scenario("seeded", seed=3)
    assert params["trial"]["seed"] == 3


def test_squeeze_release_has_physical_contact_then_clears():
    log, report, _ = run_scenario("squeeze-release")
    assert report is None
    assert log["touch_left"][log["phase"] == 0].max() > 0.5
    assert log["touch_right"][log["phase"] == 0].max() > 0.5
    assert log["touch_left"][-1] == 0
    assert log["touch_right"][-1] == 0


def test_scenario_cli_persists_replayable_failure(tmp_path):
    out = tmp_path / "missing"
    assert main(["scenario", "missing", "--output", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["parameters"]["scenario"] == "missing"
    with np.load(out / "grasp_episode.npz", allow_pickle=False) as episode:
        assert str(episode["reason"]) == "close timeout"
        assert not bool(episode["success"])


def test_video_frame_sampling_and_validation():
    assert frame_indices([0, 0.1, 0.2, 0.3], 10).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        frame_indices([0, 0], 30)
    with pytest.raises(ValueError):
        frame_indices([0, 1], 0)


def test_video_encodes_episode_and_protects_output(tmp_path, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    import graspforge.video as video

    episode, _, _ = run_scenario("squeeze-release")
    episode_path = tmp_path / "episode.npz"
    np.savez_compressed(episode_path, **episode)

    class FakeRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            assert camera == "overview"

        def render(self):
            return np.full(self.shape, 90, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(video.mujoco, "Renderer", FakeRenderer)
    output = tmp_path / "out.mp4"
    assert render_video(episode_path, output, fps=5, width=160, height=120) == output
    assert output.stat().st_size > 100
    assert b"ftyp" in output.read_bytes()[:16]
    with pytest.raises(FileExistsError):
        render_video(episode_path, output)


def test_video_rejects_model_mismatch_before_rendering(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    (tmp_path / "manifest.json").write_text(json.dumps({"model": {"sha256": "wrong"}}))
    with pytest.raises(ValueError, match="model hash"):
        render_video(tmp_path / "episode.npz", tmp_path / "out.mp4")


def test_video_cli_explains_missing_episode(tmp_path):
    with pytest.raises(SystemExit, match="Episode file not found") as error:
        main(["video", str(tmp_path / "missing.npz")])
    assert "graspforge scenario nominal" in str(error.value)
