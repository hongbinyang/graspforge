"""Validation of configuration, CLI boundaries, and persisted experiment identity."""

import json

import numpy as np
import pytest

from graspforge.cli import _prepare_output, main
from graspforge.contacts import build_index, load_model
from graspforge.controller import ControllerConfig, Disturbance, run_grasp


@pytest.mark.parametrize(
    "field,value",
    [
        ("lift_time", 0),
        ("close_speed", 0),
        ("hold_time", -1),
        ("kp_force", float("nan")),
        ("contact_confirm_steps", 0),
        ("contact_confirm_steps", 1.5),
    ],
)
def test_invalid_controller_config(field, value):
    with pytest.raises(ValueError, match=field):
        ControllerConfig(**{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force": (1, 2)},
        {"force": (0, 0, float("nan"))},
        {"force": (0, 0, 0), "duration": float("inf")},
    ],
)
def test_invalid_disturbance(kwargs):
    with pytest.raises(ValueError):
        Disturbance(**kwargs)


def test_disturbance_must_fit_in_hold():
    model = load_model()
    import mujoco

    with pytest.raises(ValueError, match="fit within"):
        run_grasp(
            model,
            mujoco.MjData(model),
            build_index(model),
            ControllerConfig(),
            disturbance=Disturbance(force=(0, 0, 0), hold_fraction=0.95, duration=0.2),
        )


def test_benchmark_rejects_impossible_threshold_before_creating_output(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(SystemExit, match="--min-successes"):
        main(
            [
                "benchmark",
                "--trials",
                "2",
                "--min-successes",
                "5",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_output_is_unique_and_existing_results_are_protected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = _prepare_output(None, "nominal", ("manifest.json",))
    second = _prepare_output(None, "nominal", ("manifest.json",))
    assert first != second
    (first / "manifest.json").write_text("existing")
    with pytest.raises(FileExistsError):
        _prepare_output(first, "nominal", ("manifest.json",))
    assert (first / "manifest.json").read_text() == "existing"


def test_nominal_cli_records_manifest_and_pickle_free_episode(tmp_path):
    output = tmp_path / "nominal"
    assert main(["run", "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["kind"] == "nominal"
    assert len(manifest["model"]["sha256"]) == 64
    assert len(manifest["source_sha256"]["controller.py"]) == 64
    assert manifest["controller_config"]["lift_height"] == 0.15
    assert manifest["parameters"]["disturbance"]["force"] == [0.0, 5.0, 0.0]
    with np.load(output / "grasp_episode.npz", allow_pickle=False) as episode:
        assert bool(episode["success"])
        assert all(episode[key].dtype.kind != "O" for key in episode.files)


def test_benchmark_cli_records_threshold_and_seed(tmp_path):
    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--trials",
                "1",
                "--min-successes",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "grasp_metrics.json").read_text())
    assert manifest["parameters"] == {"seeds": [0], "min_successes": 1}
    assert metrics["meta"]["min_successes_required"] == 1
