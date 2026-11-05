"""Reproducible physical scenarios for inspection and video replay."""

from dataclasses import asdict
import math

import mujoco
import numpy as np

from graspforge.contacts import (
    OPEN_FORCE,
    _set_fingers_at_contact,
    build_index,
    gravity_feedforward,
    load_model,
)
from graspforge.controller import ControllerConfig, Disturbance, run_grasp
from graspforge.evaluation import (
    SuccessCriteria,
    apply_friction,
    evaluate_episode,
    make_trial_setup,
    sample_trial,
)

SCENARIOS = (
    "nominal",
    "overload",
    "friction",
    "disturbance",
    "misaligned",
    "missing",
    "seeded",
    "squeeze-release",
)


def run_scenario(name, *, friction=0.15, force=15.0, offset=0.015, yaw=5.0, seed=0):
    """Return episode, report, and exact inputs; a failed grasp is a valid result."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    if not all(np.isfinite(v) for v in (friction, force, offset, yaw)):
        raise ValueError("scenario parameters must be finite")
    if friction <= 0 or force < 0 or seed < 0:
        raise ValueError(
            "friction must be positive; force and seed must be non-negative"
        )
    model = load_model()
    data = mujoco.MjData(model)
    ix = build_index(model)
    cfg = ControllerConfig()
    criteria = SuccessCriteria()
    if name == "squeeze-release":
        return (
            _squeeze_release(model, data, ix),
            None,
            {
                "scenario": name,
                "squeeze_force_N": 8.0,
                "squeeze_seconds": 1.0,
                "release_force_N": OPEN_FORCE,
                "release_seconds": 0.3,
            },
        )
    params = {"scenario": name}
    setup = None
    push = 40.0 if name == "overload" else force if name == "disturbance" else 5.0
    disturbance = Disturbance(force=(0.0, push, 0.0))
    params["disturbance"] = asdict(disturbance)
    if name == "friction":
        apply_friction(model, ix, friction)
        params["cube_friction"] = friction
    elif name in ("misaligned", "missing"):
        x = offset if name == "misaligned" else 0.2
        angle = yaw if name == "misaligned" else 0.0

        def setup(model, data):
            data.qpos[ix.cube_qpos][0] += x
            half = math.radians(angle) / 2
            data.qpos[ix.cube_qpos][3:7] = [math.cos(half), 0, 0, math.sin(half)]

        params.update(offset_x_m=x, yaw_deg=angle)
    elif name == "seeded":
        trial = sample_trial(seed)
        apply_friction(model, ix, trial.mu)
        setup = make_trial_setup(trial, ix)
        disturbance = trial.disturbance
        params.update(trial=trial.to_dict(), disturbance=asdict(disturbance))
    log = run_grasp(model, data, ix, cfg, disturbance=disturbance, setup=setup)
    report = evaluate_episode(log, criteria, cfg.hold_time, disturbance)
    return log, report, params


def _squeeze_release(model, data, ix):
    _set_fingers_at_contact(model, data, ix)
    dt = float(model.opt.timestep)
    squeeze_steps = round(1.0 / dt)
    total_steps = squeeze_steps + round(0.3 / dt)
    rows = []
    for k in range(total_steps + 1):
        mujoco.mj_forward(model, data)
        rows.append(
            (
                float(data.time),
                data.qpos.copy(),
                data.qvel.copy(),
                float(data.sensordata[ix.left_touch]),
                float(data.sensordata[ix.right_touch]),
                0 if k < squeeze_steps else 1,
            )
        )
        if k == total_steps:
            break
        data.ctrl[ix.lift_act] = gravity_feedforward(model, ix)
        data.ctrl[ix.left_act] = data.ctrl[ix.right_act] = (
            8.0 if k < squeeze_steps else OPEN_FORCE
        )
        mujoco.mj_step(model, data)
    return {
        "t": np.array([r[0] for r in rows]),
        "qpos": np.array([r[1] for r in rows]),
        "qvel": np.array([r[2] for r in rows]),
        "touch_left": np.array([r[3] for r in rows]),
        "touch_right": np.array([r[4] for r in rows]),
        "phase": np.array([r[5] for r in rows]),
        "phase_names": np.array(["SQUEEZE", "RELEASE"]),
        "slip": np.full(len(rows), np.nan),
        "dist_active": np.zeros(len(rows), dtype=bool),
        "dt": np.array(dt),
        "reason": np.array("squeeze and release complete"),
    }
