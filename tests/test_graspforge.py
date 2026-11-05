"""Standalone deterministic regression tests for GraspForge.

Run after an editable install: python tests/test_graspforge.py
Exit code 0 = all tests passed, 1 = at least one failed.
"""
import functools
import sys
import time
import traceback
from collections import defaultdict
import mujoco
import numpy as np
from graspforge.contacts import GripperIndex, build_index, load_model, _set_fingers_at_contact, compare_touch, extract_contacts, force_on_cube, force_to_world, friction_utilization, hold_squeeze, pair_label, gravity_feedforward, OPEN_FORCE, summarize_contacts, summarize_contacts, swapped
from graspforge.controller import NOMINAL_DISTURBANCE, ControllerConfig, run_grasp
from graspforge.evaluation import SuccessCriteria, evaluate_episode, analyze_disturbance
_TESTS: list = []

def test(fn):
    _TESTS.append(fn)
    return fn

def run(tests: list) -> int:
    """Run each test in isolation; return the number of failures."""
    failures = 0
    for fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            duration = time.perf_counter() - t0
            print(f'PASS {fn.__name__} ({duration:.3f} s)')
        except Exception:
            duration = time.perf_counter() - t0
            print(f'FAIL {fn.__name__} ({duration:.3f} s)')
            print(traceback.format_exc())
            failures += 1
    return failures

def main() -> int:
    t0 = time.perf_counter()
    failures = run(_TESTS)
    total = len(_TESTS)
    print(f'\n{total - failures}/{total} passed in {time.perf_counter() - t0:.1f} s')
    return 0 if failures == 0 else 1

def fresh() -> tuple[mujoco.MjModel, mujoco.MjData, GripperIndex]:
    """A new model, data and index: nothing shared with any other test."""
    model = load_model()
    return (model, mujoco.MjData(model), build_index(model))

@functools.lru_cache(maxsize=None)
def nominal_episode() -> dict[str, np.ndarray]:
    """The nominal disturbed episode, run once per test session and shared READ-ONLY."""
    model, data, ix = fresh()
    log = run_grasp(model, data, ix, ControllerConfig(), disturbance=NOMINAL_DISTURBANCE)
    for arr in log.values():
        arr.flags.writeable = False
    return log

@functools.lru_cache(maxsize=None)
def nominal_report():
    return evaluate_episode(nominal_episode(), SuccessCriteria(), ControllerConfig().hold_time, NOMINAL_DISTURBANCE)

@test
def test_shared_episode_is_read_only():
    log = nominal_episode()
    try:
        log['touch_left'][0] = 30.0
    except ValueError:
        pass
    else:
        raise AssertionError('the shared nominal episode is writable')
    assert nominal_episode() is log, 'the fixture must be computed once and cached'
EXPECTED_NAMES = {mujoco.mjtObj.mjOBJ_BODY: ['cube', 'carriage', 'left_finger', 'right_finger'], mujoco.mjtObj.mjOBJ_JOINT: ['cube_free', 'lift', 'left_slide', 'right_slide'], mujoco.mjtObj.mjOBJ_GEOM: ['table', 'cube_geom', 'left_pad', 'right_pad'], mujoco.mjtObj.mjOBJ_SITE: ['cube_center', 'left_tip', 'right_tip'], mujoco.mjtObj.mjOBJ_ACTUATOR: ['lift_motor', 'left_motor', 'right_motor'], mujoco.mjtObj.mjOBJ_SENSOR: ['left_touch', 'right_touch'], mujoco.mjtObj.mjOBJ_CAMERA: ['front', 'overview'], mujoco.mjtObj.mjOBJ_KEY: ['home']}
JOINT_SIZES = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3), mujoco.mjtJoint.mjJNT_SLIDE: (1, 1), mujoco.mjtJoint.mjJNT_HINGE: (1, 1)}

@test
def test_model_dimensions_and_named_elements():
    model, _, ix = fresh()
    assert (model.nq, model.nv, model.nu, model.nsensor) == (10, 9, 3, 2), f'dims (nq, nv, nu, nsensor) = {(model.nq, model.nv, model.nu, model.nsensor)}'
    missing_names = [name for type, names in EXPECTED_NAMES.items() for name in names if mujoco.mj_name2id(model, type, name) < 0]
    assert len(missing_names) == 0, f'some names are missing: {missing_names}'
    wrong = []
    for act, dof in ((ix.lift_act, ix.lift_dof), (ix.left_act, ix.left_dof), (ix.right_act, ix.right_dof)):
        assert model.actuator_trntype[act] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.jnt_dofadr[model.actuator_trnid[act, 0]] == dof, f'actuator {act} drives the wrong joint'
    for name, site in (('left_touch', ix.left_tip), ('right_touch', ix.right_tip)):
        assert model.sensor_objid[model.sensor(name).id] == site, f'{name} reads the wrong site'
    assert model.key_qpos[model.key('home').id].shape == (model.nq,)

@test
def test_cube_free_joint_contribution():
    model, data, ix = fresh()
    j = model.joint('cube_free').id
    assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_parentid[model.jnt_bodyid[j]] == 0, 'free joint must be on a world child'
    assert ix.cube_qpos.stop - ix.cube_qpos.start == 7 and ix.cube_qvel.stop - ix.cube_qvel.start == 6
    for jj in range(model.njnt):
        end_q = model.jnt_qposadr[jj + 1] if jj + 1 < model.njnt else model.nq
        end_v = model.jnt_dofadr[jj + 1] if jj + 1 < model.njnt else model.nv
        span = (end_q - model.jnt_qposadr[jj], end_v - model.jnt_dofadr[jj])
        assert span == JOINT_SIZES[model.jnt_type[jj]], f'joint {jj} spans {span}'
    gap = {jj: JOINT_SIZES[t][0] - JOINT_SIZES[t][1] for jj, t in enumerate(model.jnt_type)}
    assert sum(gap.values()) == model.nq - model.nv
    assert {jj for jj, g in gap.items() if g} == {j}, 'nq − nv must come from cube_free alone'
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    q = data.qpos[ix.cube_qpos]
    assert np.allclose(q[3:7], [1, 0, 0, 0]), f'home orientation {q[3:7]}'
    assert data.qpos.shape == (model.nq,) and data.qvel.shape == (model.nv,)

def steady_grasp():
    """Fresh model/data with both pads squeezing the resting cube at 8 N for 1 s."""
    model, data, ix = fresh()
    _set_fingers_at_contact(model, data, ix)
    hold_squeeze(model, data, ix, seconds=1.0)
    mujoco.mj_forward(model, data)
    return (model, data, ix)

@test
def test_contact_extraction_and_geom_order_independence():
    model, data, ix = steady_grasp()
    records = extract_contacts(model, data)
    s = summarize_contacts(records, ix)
    assert s.left.count > 0 and s.right.count > 0 and (s.table.count > 0) and (s.other_count == 0)
    assert [r.index for r in records] == list(range(data.ncon))
    for r in records:
        swapped_r = swapped(r)
        assert pair_label(swapped_r, ix) == pair_label(r, ix), "order of pair shouldn't matter"
        assert np.allclose(force_on_cube(swapped_r, ix), force_on_cube(r, ix), atol=1e-12), " order of force on cube shouldn't matter"
        det = np.linalg.det(swapped_r.frame)
        assert np.isclose(det, 1.0, atol=1e-12), f'Invalid rotation: determinant is {det} (expected 1.0)'
        assert np.allclose(swapped_r.f_world, -r.f_world, atol=1e-12), 'The swap fixture must change the raw f_world by flipping its sign!'
    snap = [r.f_world.copy() for r in records]
    mujoco.mj_step(model, data)
    assert all((np.array_equal(r.f_world, f) for r, f in zip(records, snap)))

@test
def test_contact_frame_world_frame_norm_preservation():
    model, data, ix = steady_grasp()
    for r in extract_contacts(model, data):
        assert np.allclose(r.frame @ r.frame.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r.frame), 1.0)
        assert np.isclose(np.linalg.norm(r.f_world), np.linalg.norm(r.f_local[:3]), rtol=1e-12, atol=1e-15)
        assert np.isclose(r.normal @ r.f_world, r.f_normal, rtol=1e-12, atol=1e-12)
    rng = np.random.default_rng(0)
    transpose_bug_visible = False
    for _ in range(20):
        frame, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(frame) < 0:
            frame[0] *= -1
        f_local = rng.normal(size=6)
        world = force_to_world(frame, f_local)
        assert np.isclose(np.linalg.norm(world), np.linalg.norm(f_local[:3]))
        assert np.isclose(frame[0] @ world, f_local[0])
        assert np.allclose(frame @ world, f_local[:3])
        if not np.allclose(frame @ f_local[:3], world):
            transpose_bug_visible = True
    assert transpose_bug_visible, 'random frames were all symmetric: the transpose bug would go unseen'

@test
def test_touch_contact_force_agreement():
    model, data, ix = steady_grasp()
    c = compare_touch(model, data, ix)
    assert c.touch_left > 0 and c.touch_right > 0
    assert c.err_left < 0.05 and c.err_right < 0.05, f'errors {c.err_left:.2e}, {c.err_right:.2e}'
    assert c.imbalance < 0.05
    ff = gravity_feedforward(model, ix)
    for _ in range(int(round(0.3 / model.opt.timestep))):
        data.ctrl[ix.lift_act] = ff
        data.ctrl[ix.left_act] = OPEN_FORCE
        data.ctrl[ix.right_act] = OPEN_FORCE
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.sensordata[ix.left_touch] == 0.0 and data.sensordata[ix.right_touch] == 0.0, f'touch after release: {data.sensordata[ix.left_touch]}, {data.sensordata[ix.right_touch]}'
    s = summarize_contacts(extract_contacts(model, data), ix)
    assert s.left.count == 0 and s.right.count == 0, 'pads still touching the cube'

@test
def test_friction_limit_calculation():
    assert friction_utilization(8.0, 5.0, 0.8) == 5 / 6.4
    assert np.isnan(friction_utilization(0.0, 1.0, 0.8))
    assert np.isnan(friction_utilization(8.0, 1.0, 0.0))
    log = nominal_episode()
    u = np.concatenate([log['u_left'], log['u_right']])
    u = u[np.isfinite(u)]
    assert u.size > 0, 'no loaded contacts in the episode'
    assert u.max() <= 1.0 + 1e-06, f'utilization {u.max():.6f} exceeds the friction cone'
    model, _, ix = fresh()
    mu = float(model.geom_friction[ix.cube_geom, 0])
    m = float(model.body_mass[ix.cube_body])
    g = -float(model.opt.gravity[2])
    r = analyze_disturbance(nominal_episode(), mu, m, g)
    rel = abs(r['u_peak'] - r['u_pred']) / r['u_pred']
    assert rel < 0.25, f"measured peak {r['u_peak']:.3f} vs predicted {r['u_pred']:.3f} ({rel:.0%} off)"
from graspforge.controller import LEGAL_TRANSITIONS, TERMINAL, FingerForceLoop, GraspController, Observation, Phase, check_transition, lift_command, FAILED_OPEN_FORCE

def make_obs(t: float, **overrides) -> Observation:
    obs = dict(t=t, touch_left=0.0, touch_right=0.0, q_left=0.0, q_right=0.0, qd_left=0.0, qd_right=0.0, lift_q=0.0, lift_qd=0.0, cube_pos=np.array([0, 0, 0.45]), tip_mid=np.array([0, 0, 0.45]), max_penetration=0.0, finite=True)
    obs.update(overrides)
    return Observation(**obs)

def drive(ctrl: GraspController, t0: float, seconds: float, **overrides):
    obs = make_obs(t0, **overrides)
    phases = []
    n = int(round(seconds / ctrl.dt))
    for k in range(n):
        obs = make_obs(t0 + k * ctrl.dt, **overrides)
        cmd = ctrl.step(obs)
        if len(phases) == 0 or phases[-1] != cmd.phase:
            phases.append(cmd.phase)
    return (cmd, t0 + n * ctrl.dt, phases)

@test
def test_legal_state_machine_transitions():
    legal_phases = LEGAL_TRANSITIONS.keys()
    all_phases = set((v for v in Phase))
    assert legal_phases == all_phases, f'not all phases are covered: {legal_phases ^ all_phases}'
    assert all([len(LEGAL_TRANSITIONS[t]) == 0 for t in TERMINAL]), 'every terminal phase has an EMPTY successor set'
    assert all([Phase.FAILED in LEGAL_TRANSITIONS[k] for k in LEGAL_TRANSITIONS.keys() - TERMINAL]), 'every non-terminal phase can reach FAILED'
    try:
        check_transition(Phase.SETTLE, Phase.HOLD)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpect error when Phase.SETTLE -> Phase.HOLD')
    try:
        check_transition(Phase.PRELOAD, Phase.SETTLE)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpect error when Phase.PRELOAD -> Phase.SETTLE')
    try:
        check_transition(Phase.DONE, Phase.FAILED)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpect error when Phase.DONE -> Phase.FAILED')
    cfg = ControllerConfig()
    model, data, ix = fresh()
    ctrl = GraspController(model, ix, cfg)
    seq = []
    t = 0.0
    for seconds, overrides in ((cfg.settle_time + 10 * ctrl.dt, {}), ((cfg.contact_confirm_steps + 5) * ctrl.dt, dict(touch_left=cfg.contact_threshold + 0.2, touch_right=cfg.contact_threshold + 0.2)), (cfg.preload_ramp_time + cfg.preload_dwell + 0.05, dict(touch_left=cfg.preload_target, touch_right=cfg.preload_target)), (cfg.lift_time + cfg.hold_time + 0.05, dict(touch_left=cfg.preload_target, touch_right=cfg.preload_target))):
        cmd, t, phases = drive(ctrl, t, seconds, **overrides)
        for p in phases:
            if len(seq) == 0 or seq[-1] != p:
                seq.append(p)
    assert Phase.LIFT in seq and Phase.HOLD in seq, 'through LIFT and HOLD'
    assert seq == [Phase.SETTLE, Phase.CLOSE, Phase.PRELOAD, Phase.LIFT, Phase.HOLD, Phase.DONE]
    for a, b in zip(seq, seq[1:]):
        assert b in LEGAL_TRANSITIONS[a], f'{a.name} to {b.name} is not legal transition'
    cmd, t, phases = drive(ctrl, t, seconds=0.1, touch_left=30.0, touch_right=30.0)
    assert cmd.phase == Phase.DONE, 'should be finished'

@test
def test_close_phase_timeout():
    cfg = ControllerConfig()
    model, data, ix = fresh()
    ctrl = GraspController(model, ix, cfg)
    deadline = cfg.settle_time + cfg.close_timeout
    margin = 5 * ctrl.dt
    cmd, t, _ = drive(ctrl=ctrl, t0=0, seconds=deadline - margin)
    assert cmd.phase.name == Phase.CLOSE.name, f'phase {cmd.phase.name} at t={t:.3f}, expected CLOSE before the deadline'
    cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=margin * 2)
    assert cmd.phase.name == Phase.FAILED.name, f'phase {cmd.phase.name} at t={t:.3f}, expected FAILED after the deadline'
    assert cmd.reason.startswith('close timeout'), f'reason {cmd.reason!r}'
    assert bool(np.all(cmd.applied[1:] == FAILED_OPEN_FORCE)), f'finger commands {cmd.applied[1:]}'

@test
def test_control_clipping_and_anti_windup():
    model, _, ix = fresh()
    cfg = ControllerConfig()
    dt = model.opt.timestep
    lo, hi = model.actuator_ctrlrange[ix.left_act]
    force_loop = FingerForceLoop(cfg, lo, hi, dt)
    for _ in range(int(round(2.0 / dt))):
        out = force_loop.step(f_ref=12.0, touch=0.0, qd=0.0)
        assert cfg.i_max_force >= abs(cfg.ki_force * force_loop.integral), f'integral term {cfg.ki_force * force_loop.integral:.3f} N exceeds the clamp'
    assert out.raw > hi and out.applied == hi and out.saturated, f'raw {out.raw}, applied {out.applied}, saturated {out.saturated}'
    steps = 0
    out = force_loop.step(f_ref=12.0, touch=14.0, qd=0.0)
    steps += 1
    while out.saturated:
        out = force_loop.step(f_ref=12.0, touch=14.0, qd=0.0)
        steps += 1
        assert steps < 1000, 'output never left saturation'
    assert steps <= 10, f'took {steps} steps to leave saturation (windup?)'
    force_loop.reset()
    out = force_loop.step(f_ref=0.0, touch=30.0, qd=0.0)
    assert out.applied == lo and out.saturated and (out.raw < lo), f'negative side: raw {out.raw:.2f}, applied {out.applied}, saturated {out.saturated}'
    l_lo, l_hi = model.actuator_ctrlrange[ix.lift_act]
    z_ref = 0.15
    z = 0.1
    zd_ref = zd = 0.0
    u_ff = 7.85
    loop_output = lift_command(cfg, z_ref, zd_ref, z, zd, u_ff, l_lo, l_hi)
    assert loop_output.applied == l_hi and loop_output.saturated and (loop_output.raw > l_hi), f'clipped: loop_output.applied({loop_output.applied}) == l_hi({l_hi}) and loop_output.saturated({loop_output.saturated}) and loop_output.raw({loop_output.raw}) > l_hi'
    z_ref = 0.05
    z = 0.1
    loop_output = lift_command(cfg, z_ref, zd_ref, z, zd, u_ff, l_lo, l_hi)
    assert loop_output.applied == l_lo and loop_output.saturated and (loop_output.raw < l_lo), f'clipped: loop_output.applied){loop_output.applied} == l_lo({l_lo}) and loop_output.saturated({loop_output.saturated}) and loop_output.raw({loop_output.raw}) < l_lo'
    log = nominal_episode()
    acts = [ix.lift_act, ix.left_act, ix.right_act]
    ranges = model.actuator_ctrlrange[acts]
    col_lo = ranges[:, 0]
    col_hi = ranges[:, 1]
    applied = log['applied']
    assert np.all(applied >= col_lo), "an applied command is below its actuator's lower limit"
    assert np.all(applied <= col_hi), "an applied command is above its actuator's upper limit"
    changed = log['raw'] != log['applied']
    assert np.array_equal(log['saturated'], changed), "saturated flags don't match raw != applied"
import copy
from graspforge.contacts import PAD_GAP_HOME
from graspforge.controller import Disturbance, EPISODE_MAX_TIME
from graspforge.evaluation import _logs_identical, replay_step, run_trial, sample_trial

def drive_to(ctrl: GraspController, until: Phase) -> float:
    t = 0.0
    count_limit = EPISODE_MAX_TIME / ctrl.dt
    count = 0
    cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=ctrl.dt)
    while count <= count_limit and cmd.phase != until:
        count += 1
        if cmd.phase == Phase.SETTLE:
            cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=ctrl.dt, touch_left=0.0, touch_right=0.0)
        elif cmd.phase == Phase.CLOSE:
            cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=ctrl.dt, touch_left=ctrl.cfg.contact_threshold + 0.2, touch_right=ctrl.cfg.contact_threshold + 0.2)
        elif cmd.phase in [Phase.PRELOAD, Phase.LIFT, Phase.HOLD]:
            cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=ctrl.dt, touch_left=ctrl.cfg.preload_target, touch_right=ctrl.cfg.preload_target)
        elif cmd.phase in [Phase.FAILED, Phase.DONE]:
            raise AssertionError(f'unexpected finished at stage {cmd.phase.name}')
    if cmd.phase == until:
        return t
    else:
        raise AssertionError(f"hasn't reached {until.name} after {count_limit} drives")

@test
def test_disturbance_application_and_clearing():
    log = nominal_episode()
    active = log['dist_active']
    dt = float(log['dt'])
    n = int(round(NOMINAL_DISTURBANCE.duration / dt))
    rows = np.flatnonzero(active)
    assert len(rows) == n, f'{len(rows)} active rows, expected {n}'
    assert np.all(np.diff(rows) == 1), 'the active rows should form ONE contiguous run'
    k_hold = int(np.flatnonzero(log['phase'] == int(Phase.HOLD))[0])
    offset = int(round(NOMINAL_DISTURBANCE.hold_fraction * ControllerConfig().hold_time / dt))
    assert rows[0] == k_hold + offset, f'window starts at row {rows[0]}, expected {k_hold + offset}'
    expected = np.zeros_like(log['xfrc'])
    expected[rows, :3] = NOMINAL_DISTURBANCE.force
    assert np.array_equal(log['xfrc'], expected), 'the logged wrench differs from the specification'
    model, data, ix = fresh()
    stale = run_grasp(model, data, ix, ControllerConfig(), disturbance=NOMINAL_DISTURBANCE, _clear_buffer=False)
    end = np.flatnonzero(stale['dist_active'])[-1] + 1
    assert np.any(stale['xfrc'][end:] != 0.0), 'a stale wrench after the window went unnoticed'

@test
def test_episode_alignment():
    log = nominal_episode()
    model, _, ix = fresh()
    dt = float(log['dt'])

    def first_row(phase: Phase) -> int:
        """First row index where the logged phase equals `phase`."""
        return int(np.flatnonzero(log['phase'] == int(phase))[0])
    probes = {'CLOSE (fingers moving)': first_row(Phase.CLOSE) + 200, 'LIFT (top speed)': first_row(Phase.LIFT) + int(round(0.75 / dt)), 'push window': int(np.flatnonzero(log['dist_active'])[50])}
    for label, k in probes.items():
        q = replay_step(model, ix, log, k)
        d_prev = np.linalg.norm(q - log['qpos'][k])
        d_match = np.linalg.norm(q - log['qpos'][k + 1])
        d_next = np.linalg.norm(q - log['qpos'][k + 2])
        assert d_match < 1e-09, f'{label}: replay misses row k+1 by {d_match:.2e}'
        assert d_match < 0.001 * min(d_prev, d_next), f'{label}: ambiguous match (k: {d_prev:.2e}, k+1: {d_match:.2e}, k+2: {d_next:.2e})'

@test
def test_nominal_grasp_success():
    success_report = nominal_report()
    assert len(success_report.criteria) == 9, 'there should be 9 criteria in the report'
    assert success_report.success and len(success_report.failures) == 0, f'every report should be passed and failures are: {[c for c in success_report.criteria if not c.passed]}'
    assert nominal_episode()['phase'][-1] == Phase.DONE, f"the final phase{nominal_episode()['phase'][-1]} should be DONE"

@test
def test_deterministic_replay():
    model, data, ix = fresh()
    log1 = run_grasp(model, data, ix, ControllerConfig(), disturbance=NOMINAL_DISTURBANCE)
    model, data, ix = fresh()
    log2 = run_grasp(model, data, ix, ControllerConfig(), disturbance=NOMINAL_DISTURBANCE)
    diff = _logs_identical(log1, log2)
    assert len(diff) == 0, f'the same inputs should give bit-identical results: difference is {diff}'
    result1 = run_trial(0)
    result2 = run_trial(0)
    assert result1.report.success == result2.report.success, f'run trial with same seed should get same result: {result1.report.success} vs {result2.report.success}'
    assert result1.report.failures == result2.report.failures, f'run trial with same seed should get same result: {result1.report.failures} vs {result2.report.failures}'
    values1 = [c.value for c in result1.report.criteria]
    values2 = [c.value for c in result2.report.criteria]
    assert np.array_equal(values1, values2, equal_nan=True), f'run trial with same seed should get same result: {values1} vs {values2}'
    trial_cfg1 = sample_trial(1)
    trial_cfg2 = sample_trial(2)
    assert trial_cfg1 != trial_cfg2, f' run trial with different should get different result: {trial_cfg1} vs {trial_cfg2}'

@test
def test_failure_detection_dropped_cube_and_excessive_force():
    cfg = ControllerConfig()
    model, _, ix = fresh()
    ctrl = GraspController(model, ix, cfg)
    cases = [{'until': Phase.HOLD, 'situation': 'contact lost', 'parameter': dict(touch_left=0.0, touch_right=0.0)}, {'until': Phase.HOLD, 'situation': 'slip', 'parameter': dict(touch_left=ctrl.cfg.preload_target, touch_right=ctrl.cfg.preload_target, cube_pos=np.array([0, 0, 0.41]))}, {'until': Phase.HOLD, 'situation': 'excessive force', 'parameter': dict(touch_left=30.0, touch_right=30.0)}]
    for case in cases:
        until, situation, parameter = (case['until'], case['situation'], case['parameter'])
        model, _, ix = fresh()
        ctrl = GraspController(model, ix, cfg)
        t = drive_to(ctrl, until)
        cmd, t, _ = drive(ctrl=ctrl, t0=t, seconds=10 * ctrl.dt, **parameter)
        assert cmd.phase == Phase.FAILED, f"phase({cmd.phase}) should be 'FAILED'"
        assert cmd.reason.startswith(situation), f"failure reason ({cmd.reason}) should be '{situation}'"
    cfg = ControllerConfig()
    crit = SuccessCriteria()
    overload = Disturbance(force=(0.0, 40.0, 0.0))
    model, data, ix = fresh()
    log = run_grasp(model, data, ix, cfg, disturbance=overload)
    rep = evaluate_episode(log, crit, cfg.hold_time, overload)
    must_fail = {'lifted_throughout_hold', 'reached_done'}
    missing = must_fail - set(rep.failures)
    assert not missing, f'overload episode: expected {sorted(missing)} to fail; failures were {rep.failures}'
    doctored = copy.deepcopy(nominal_episode())
    k = int(np.flatnonzero(doctored['phase'] == int(Phase.HOLD))[0])
    doctored['touch_left'][k] = 30.0
    rep = evaluate_episode(doctored, crit, cfg.hold_time, NOMINAL_DISTURBANCE)
    fails = [f for f in list(rep.failures)]
    assert fails == ['fingertip_force'], f'doctored 30 N sample: failures were {rep.failures}'
from graspforge.evaluation import MIN_SUCCESSES, N_TRIALS

@test
def test_robustness_threshold():
    success_count = 0
    failure_track = []
    for seed in range(N_TRIALS):
        trial_result = run_trial(seed)
        if trial_result.report.success:
            success_count += 1
        else:
            failure_track.append((seed, trial_result.report.failures))
    assert success_count >= MIN_SUCCESSES, f'{success_count}/{N_TRIALS} trials succeeded, need at least {MIN_SUCCESSES}; failing (seed, failures): {failure_track}'
if __name__ == '__main__':
    sys.exit(main())
