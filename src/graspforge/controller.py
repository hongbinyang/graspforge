"""Hybrid force-and-motion controller and deterministic MuJoCo rollout."""
from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from graspforge.contacts import GripperIndex, PAD_GAP_HOME, max_penetration
MAX_CUBE_OFFSET = 0.01
MAX_CLOSE_GAP = PAD_GAP_HOME + MAX_CUBE_OFFSET

class Phase(IntEnum):
    """Integer-valued so a phase can be logged in an npz array; .name for JSON/printing."""
    SETTLE = 0
    CLOSE = 1
    PRELOAD = 2
    LIFT = 3
    HOLD = 4
    DONE = 5
    FAILED = 6
TERMINAL = frozenset({Phase.DONE, Phase.FAILED})
# The transition table is the source of truth for controller phase changes.
LEGAL_TRANSITIONS: dict[Phase, frozenset[Phase]] = {Phase.SETTLE: frozenset({Phase.CLOSE, Phase.FAILED}), Phase.CLOSE: frozenset({Phase.PRELOAD, Phase.FAILED}), Phase.PRELOAD: frozenset({Phase.LIFT, Phase.FAILED}), Phase.LIFT: frozenset({Phase.HOLD, Phase.FAILED}), Phase.HOLD: frozenset({Phase.DONE, Phase.FAILED}), Phase.DONE: frozenset(), Phase.FAILED: frozenset()}

def check_transition(old: Phase, new: Phase) -> None:
    """Raise ValueError if old → new is not a legal transition (old == new is always fine)."""
    if old != new and new not in LEGAL_TRANSITIONS[old]:
        raise ValueError(f'illegal transition from {old.name} to {new.name}')

@dataclass(frozen=True)
class ControllerConfig:
    """Immutable gains, thresholds, timing parameters, and safety limits."""
    settle_time: float = 0.5
    close_speed: float = 0.05
    close_timeout: float = 2.0
    kv_close: float = 20.0
    contact_threshold: float = 0.2
    touch_hold_force: float = 0.5
    contact_confirm_steps: int = 5
    preload_target: float = 8.0
    preload_ramp_time: float = 0.3
    preload_tol: float = 0.05
    preload_dwell: float = 0.1
    preload_timeout: float = 1.5
    kp_force: float = 0.5
    ki_force: float = 5.0
    kd_finger: float = 2.0
    i_max_force: float = 3.0
    lift_height: float = 0.15
    lift_time: float = 1.5
    hold_time: float = 2.0
    kp_lift: float = 2000.0
    kd_lift: float = 100.0
    payload_mass: float = 0.2
    f_max_safe: float = 25.0
    pen_max: float = 0.005
    slip_fail: float = 0.03

    def __post_init__(self):
        if not 0 < self.contact_threshold < self.touch_hold_force:
            raise ValueError(f'keep contact detected: 0 < contact_threshold({self.contact_threshold}) < touch_hold_force({self.touch_hold_force})')
        if not self.touch_hold_force < self.preload_target < self.f_max_safe:
            raise ValueError(f'check range: touch_hold_force({self.touch_hold_force}) < preload_target({self.preload_target}) < f_max_safe({self.f_max_safe})')
        if not self.preload_ramp_time + self.preload_dwell < self.preload_timeout:
            raise ValueError(f'check PRELOAD timing: preload_ramp_time({self.preload_ramp_time}) + preload_dwell({self.preload_dwell}) < preload_timeout({self.preload_timeout})')
        if not MAX_CLOSE_GAP / self.close_speed < 0.5 * self.close_timeout:
            raise ValueError(f'check close timeout: MAX_CLOSE_GAP({MAX_CLOSE_GAP}) / close_speed({self.close_speed}) < 0.5 * close_timeout({self.close_timeout})')
@dataclass(frozen=True)
class Observation:
    """One consistent instant, as plain numbers. Built only by observe()."""
    t: float
    touch_left: float
    touch_right: float
    q_left: float
    q_right: float
    qd_left: float
    qd_right: float
    lift_q: float
    lift_qd: float
    cube_pos: np.ndarray
    tip_mid: np.ndarray
    max_penetration: float
    finite: bool

    @property
    def cube_offset(self) -> np.ndarray:
        """Cube centre relative to the fingertip midpoint (world axes)."""
        return self.cube_pos - self.tip_mid

def observe(model: mujoco.MjModel, data: mujoco.MjData, ix: GripperIndex) -> Observation:
    """Snapshot of what the controller may see.

    Precondition: `data` holds the result of a forward pass (mj_forward) at the
    current state, so state, sensors and contacts describe the same instant.
    """
    # The rollout calls mj_forward before this function so sensors and state share t_k.
    t = data.time
    touch_left = float(data.sensordata[ix.left_touch])
    touch_right = float(data.sensordata[ix.right_touch])
    q_left = float(data.qpos[ix.left_qpos])
    q_right = float(data.qpos[ix.right_qpos])
    qd_left = float(data.qvel[ix.left_dof])
    qd_right = float(data.qvel[ix.right_dof])
    lift_q = float(data.qpos[ix.lift_qpos])
    lift_qd = float(data.qvel[ix.lift_dof])
    cube_pos = data.site_xpos[ix.cube_site].copy()
    tip_mid = 0.5 * (data.site_xpos[ix.left_tip] + data.site_xpos[ix.right_tip])
    the_max_penetration = float(max_penetration(data))
    finite = all((np.isfinite(a).all() for a in (data.qpos, data.qvel, data.ctrl, data.sensordata)))
    return Observation(t=t, touch_left=touch_left, touch_right=touch_right, q_left=q_left, q_right=q_right, qd_left=qd_left, qd_right=qd_right, lift_q=lift_q, lift_qd=lift_qd, cube_pos=cube_pos, tip_mid=tip_mid, max_penetration=the_max_penetration, finite=finite)

def slip(obs: Observation, offset_ref: np.ndarray) -> float:
    return float(np.linalg.norm(obs.cube_offset - offset_ref))

@dataclass(frozen=True)
class LoopOutput:
    raw: float
    applied: float
    saturated: bool

class FingerForceLoop:
    """Force regulation for one finger: feedforward + PI on touch error + velocity damping.

    Sign convention (from A3/A4): positive command closes the finger and
    pushes INTO the cube, so it raises touch; positive qd is closing motion.
    """

    def __init__(self, cfg: ControllerConfig, ctrl_lo: float, ctrl_hi: float, dt: float):
        self.cfg = cfg
        self.lo, self.hi = (float(ctrl_lo), float(ctrl_hi))
        self.dt = float(dt)
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def step(self, f_ref: float, touch: float, qd: float) -> LoopOutput:
        c = self.cfg
        e = f_ref - touch
        # Clamp the integral contribution to bound recovery after actuator saturation.
        if c.ki_force > 0:
            self.integral += e * self.dt
            self.integral = float(np.clip(self.integral, -(c.i_max_force / c.ki_force), c.i_max_force / c.ki_force))
        raw = f_ref + self.cfg.kp_force * e + self.cfg.ki_force * self.integral - self.cfg.kd_finger * qd
        applied = np.clip(raw, self.lo, self.hi)
        saturated = bool(applied != raw)
        return LoopOutput(raw=float(raw), applied=float(applied), saturated=bool(saturated))

class FingerCloser:
    """CLOSE behaviour for one finger: approach at close_speed, then hold lightly once contact is confirmed."""

    def __init__(self, cfg: ControllerConfig, ctrl_lo: float, ctrl_hi: float, joint_damping: float):
        self.cfg = cfg
        self.lo, self.hi = (float(ctrl_lo), float(ctrl_hi))
        self.joint_damping = float(joint_damping)
        self.count = 0
        self.contact = False

    def reset(self) -> None:
        self.count = 0
        self.contact = False

    def step(self, touch: float, qd: float) -> LoopOutput:
        c = self.cfg
        if touch > c.contact_threshold:
            self.count += 1
            if self.count >= c.contact_confirm_steps:
                self.contact = True
        else:
            self.count = 0
        if self.contact:
            raw = c.touch_hold_force - c.kd_finger * qd
        else:
            raw = self.joint_damping * c.close_speed + c.kv_close * (c.close_speed - qd)
        applied = float(np.clip(raw, self.lo, self.hi))
        saturated = applied != raw
        return LoopOutput(raw=float(raw), applied=applied, saturated=bool(saturated))

def min_jerk(t: float, duration: float, distance: float) -> tuple[float, float, float]:
    """Minimum-jerk profile from 0 to `distance` over `duration`: (position, velocity, acceleration)."""
    s = np.clip(t / duration, 0.0, 1.0)
    position = distance * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    velocity = distance / duration * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4)
    acceleration = distance / duration ** 2 * (60 * s - 180 * s ** 2 + 120 * s ** 3)
    return (float(position), float(velocity), float(acceleration))

def lift_command(cfg: ControllerConfig, z_ref: float, zd_ref: float, z: float, zd: float, u_ff: float, ctrl_lo: float, ctrl_hi: float) -> LoopOutput:
    """PD on lift position/velocity around a feedforward force."""
    raw = u_ff + cfg.kp_lift * (z_ref - z) + cfg.kd_lift * (zd_ref - zd)
    applied = np.clip(raw, ctrl_lo, ctrl_hi)
    saturated = bool(applied != raw)
    return LoopOutput(raw=float(raw), applied=float(applied), saturated=bool(saturated))
from graspforge.contacts import gravity_feedforward
FAILED_OPEN_FORCE = -2.0

def _fixed(value: float, lo: float, hi: float) -> LoopOutput:
    """A constant command, clipped and flagged like any control-law output."""
    applied = float(np.clip(value, lo, hi))
    return LoopOutput(raw=float(value), applied=applied, saturated=applied != value)

@dataclass(frozen=True)
class Command:
    """One control step's output. Actuator order is always [lift, left, right]."""
    phase: Phase
    raw: np.ndarray
    applied: np.ndarray
    saturated: np.ndarray
    f_ref: float
    z_ref: float
    slip: float
    reason: str

class GraspController:

    def __init__(self, model: mujoco.MjModel, ix: GripperIndex, cfg: ControllerConfig):
        self.cfg = cfg
        self.dt = float(model.opt.timestep)
        self.lift_rng = model.actuator_ctrlrange[ix.lift_act].astype(float)
        self.finger_rng = model.actuator_ctrlrange[ix.left_act].astype(float)
        self.finger_damping = float(model.dof_damping[ix.left_dof])
        self.g = float(-model.opt.gravity[2])
        self.ff_gripper = gravity_feedforward(model, ix)
        self.m_lifted = self.ff_gripper / self.g + cfg.payload_mass
        self.reset()

    def reset(self) -> None:
        """Fresh episode state. Nothing survives from a previous episode."""
        lo, hi = self.finger_rng
        self.phase = Phase.SETTLE
        self.t0 = None
        self.reason = ''
        self.closers = (FingerCloser(self.cfg, lo, hi, self.finger_damping), FingerCloser(self.cfg, lo, hi, self.finger_damping))
        self.loops = (FingerForceLoop(self.cfg, lo, hi, self.dt), FingerForceLoop(self.cfg, lo, hi, self.dt))
        self.dwell = 0.0
        self.offset_ref = None
        self.z_start = 0.0
        self.z_hold = 0.0

    def step(self, obs: Observation) -> Command:
        if self.t0 is None:
            self.t0 = obs.t
        tau = obs.t - self.t0
        if self.phase not in TERMINAL:
            reason = self._safety(obs)
            if reason is not None:
                self._enter(Phase.FAILED, reason, obs)
            else:
                nxt = self._transition(obs, tau)
                if nxt is not None:
                    self._enter(*nxt, obs)
            tau = obs.t - self.t0
        f_ref, (u_left, u_right) = self._finger_commands(obs, tau)
        z_ref, u_lift = self._lift_command(obs, tau)
        outs = (u_lift, u_left, u_right)
        slip_now = slip(obs, self.offset_ref) if self.offset_ref is not None else float('nan')
        return Command(phase=self.phase, raw=np.array([o.raw for o in outs]), applied=np.array([o.applied for o in outs]), saturated=np.array([o.saturated for o in outs]), f_ref=f_ref, z_ref=z_ref, slip=slip_now, reason=self.reason)

    def _safety(self, obs: Observation) -> str | None:
        c = self.cfg
        if not obs.finite:
            return f'non-finite state: not finite({obs.finite})'
        if obs.touch_left > c.f_max_safe or obs.touch_right > c.f_max_safe:
            return f'excessive force: touch_left({obs.touch_left}) > f_max_safe({c.f_max_safe}) or touch_right({obs.touch_right}) > f_max_safe({c.f_max_safe})'
        if obs.max_penetration > c.pen_max:
            return f'unsafe penetration: max_penetration({obs.max_penetration}) > pen_max({c.pen_max})'
        if self.phase in (Phase.LIFT, Phase.HOLD):
            if obs.touch_left < c.contact_threshold or obs.touch_right < c.contact_threshold:
                return f'contact lost: touch_left({obs.touch_left}) < contact_threshold({c.contact_threshold}) or touch_right({obs.touch_right}) < contact_threshold({c.contact_threshold})'
            if slip(obs, self.offset_ref) > c.slip_fail:
                return f'slip: slip(obs, offset_ref)({slip(obs, self.offset_ref)}) > slip_fail({c.slip_fail})'
        return None

    def _transition(self, obs: Observation, tau: float) -> tuple[Phase, str] | None:
        c = self.cfg
        if self.phase == Phase.SETTLE and tau >= c.settle_time:
            return (Phase.CLOSE, 'settled')
        if self.phase == Phase.CLOSE:
            if self.closers[0].contact and self.closers[1].contact:
                return (Phase.PRELOAD, 'both in contact')
            if tau > c.close_timeout:
                return (Phase.FAILED, 'close timeout')
        if self.phase == Phase.PRELOAD:
            band = c.preload_tol * c.preload_target
            in_band = tau >= c.preload_ramp_time and abs(obs.touch_left - c.preload_target) <= band and (abs(obs.touch_right - c.preload_target) <= band)
            self.dwell = self.dwell + self.dt if in_band else 0.0
            if self.dwell >= c.preload_dwell:
                return (Phase.LIFT, 'preload settled')
            if tau >= c.preload_timeout:
                return (Phase.FAILED, 'preload timeout')
        if self.phase == Phase.LIFT and tau >= c.lift_time:
            return (Phase.HOLD, 'lift complete')
        if self.phase == Phase.HOLD and tau >= c.hold_time:
            return (Phase.DONE, 'hold complete')
        return None

    def _enter(self, new: Phase, reason: str, obs: Observation) -> None:
        check_transition(self.phase, new)
        self.phase = new
        self.t0 = obs.t
        self.reason = reason
        if new == Phase.PRELOAD:
            for loop in self.loops:
                loop.reset()
            self.dwell = 0.0
        if new == Phase.LIFT:
            self.offset_ref = obs.cube_offset.copy()
            self.z_start = obs.lift_q
        if Phase.FAILED == new:
            self.z_hold = obs.lift_q

    def _finger_commands(self, obs: Observation, tau: float) -> tuple[float, tuple[LoopOutput, LoopOutput]]:
        lo, hi = self.finger_rng
        c = self.cfg
        if self.phase == Phase.SETTLE:
            return (float('nan'), (_fixed(0.0, lo, hi), _fixed(0.0, lo, hi)))
        if self.phase == Phase.CLOSE:
            out_left = self.closers[0].step(obs.touch_left, obs.qd_left)
            out_right = self.closers[1].step(obs.touch_right, obs.qd_right)
            return (float('nan'), (out_left, out_right))
        if self.phase == Phase.PRELOAD:
            f_ref = c.touch_hold_force + (c.preload_target - c.touch_hold_force) * min(tau / c.preload_ramp_time, 1.0)
            out_left = self.loops[0].step(f_ref, obs.touch_left, obs.qd_left)
            out_right = self.loops[1].step(f_ref, obs.touch_right, obs.qd_right)
            return (f_ref, (out_left, out_right))
        if self.phase in [Phase.LIFT, Phase.HOLD, Phase.DONE]:
            f_ref = c.preload_target
            out_left = self.loops[0].step(f_ref, obs.touch_left, obs.qd_left)
            out_right = self.loops[1].step(f_ref, obs.touch_right, obs.qd_right)
            return (f_ref, (out_left, out_right))
        if self.phase == Phase.FAILED:
            return (float('nan'), (_fixed(FAILED_OPEN_FORCE, lo, hi), _fixed(FAILED_OPEN_FORCE, lo, hi)))

    def _lift_command(self, obs: Observation, tau: float) -> tuple[float, LoopOutput]:
        c = self.cfg
        if self.phase in [Phase.SETTLE, Phase.CLOSE, Phase.PRELOAD]:
            z_ref, zd_ref, u_ff = (0.0, 0.0, self.ff_gripper)
        elif self.phase == Phase.LIFT:
            z, zd, zdd = min_jerk(tau, c.lift_time, c.lift_height)
            z_ref = self.z_start + z
            zd_ref = zd
            u_ff = self.m_lifted * (self.g + zdd)
        elif self.phase in [Phase.HOLD, Phase.DONE]:
            z_ref = self.z_start + c.lift_height
            zd_ref = 0
            u_ff = self.m_lifted * self.g
        elif self.phase == Phase.FAILED:
            z_ref, zd_ref, u_ff = (self.z_hold, 0.0, self.ff_gripper)
        else:
            raise RuntimeError('unsupported case')
        out = lift_command(self.cfg, z_ref, zd_ref, obs.lift_q, obs.lift_qd, u_ff, *self.lift_rng)
        return (z_ref, out)

def _base_obs(t: float=0.0, **kw) -> Observation:
    """A home-like observation; override any field by keyword."""
    fields = dict(t=t, touch_left=0.0, touch_right=0.0, q_left=0.0, q_right=0.0, qd_left=0.0, qd_right=0.0, lift_q=0.0, lift_qd=0.0, cube_pos=np.array([0.0, 0.0, 0.45]), tip_mid=np.array([0.0, 0.0, 0.45]), max_penetration=0.0, finite=True)
    fields.update(kw)
    return Observation(**fields)

def _run(ctrl: GraspController, seconds: float, t0: float, **kw) -> tuple[Command, float]:
    """Feed the same observation (with advancing time) for `seconds`; return the last command and time."""
    n = int(round(seconds / ctrl.dt))
    cmd = None
    for k in range(n):
        t = t0 + k * ctrl.dt
        cmd = ctrl.step(_base_obs(t, **kw))
    return (cmd, t0 + n * ctrl.dt)
EPISODE_MAX_TIME = 8.0
TERMINAL_TAIL = 0.1
from graspforge.contacts import extract_contacts, summarize_contacts

@dataclass(frozen=True)
class Disturbance:
    """A constant world-frame force on the cube's centre of mass, for a fixed window inside HOLD."""
    force: tuple[float, float, float]
    hold_fraction: float = 0.5
    duration: float = 0.1

    def __post_init__(self):
        if not all(np.isfinite(self.force)):
            raise ValueError(f'force must be finite, got {self.force}')
        if not 0.0 <= self.hold_fraction < 1.0:
            raise ValueError(f'hold_fraction must be in [0, 1), got {self.hold_fraction}')
        if not self.duration > 0.0:
            raise ValueError(f'duration must be > 0, got {self.duration}')
NOMINAL_DISTURBANCE = Disturbance(force=(0.0, 5.0, 0.0))
from typing import Callable

def run_grasp(model, data, ix, cfg, disturbance=None, _clear_buffer=True, setup: Callable[[mujoco.MjModel, mujoco.MjData], None] | None=None):
    """One episode from the home keyframe. Returns the per-step log as arrays.

    Convention per step k:
        obs_k = observe(...)        state, sensors, contacts at t_k (consistent)
        cmd_k = controller(obs_k)
        ctrl  = cmd_k.applied       held over [t_k, t_{k+1})
        mj_step, then mj_forward    → data describes t_{k+1}, ready for obs_{k+1}

    disturbance:   applied to the cube for a window inside HOLD, if given.
    _clear_buffer: TEST HOOK ONLY. False skips clearing xfrc_applied outside the
                   window, to prove the evaluator catches a stale buffer (E4).
    """
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    if setup is not None:
        setup(model, data)
    mujoco.mj_forward(model, data)
    ctrl = GraspController(model, ix, cfg)
    act = [ix.lift_act, ix.left_act, ix.right_act]
    rows = []
    t_terminal = None
    n_max = int(round(EPISODE_MAX_TIME / model.opt.timestep))
    dt = model.opt.timestep
    k_hold = None
    if disturbance is not None:
        offset_steps = int(round(disturbance.hold_fraction * cfg.hold_time / dt))
        window_steps = int(round(disturbance.duration / dt))
    for k in range(n_max):
        obs = observe(model, data, ix)
        cmd = ctrl.step(obs)
        for i, a in enumerate(act):
            data.ctrl[a] = cmd.applied[i]
        if cmd.phase == Phase.HOLD and k_hold is None:
            k_hold = k
        active = disturbance is not None and k_hold is not None and (k_hold + offset_steps <= k < k_hold + offset_steps + window_steps)
        # xfrc_applied persists across steps, so the entire wrench is cleared when inactive.
        if active:
            data.xfrc_applied[ix.cube_body] = [*disturbance.force, 0, 0, 0]
        elif _clear_buffer:
            data.xfrc_applied[ix.cube_body] = list(np.zeros(6))
        s = summarize_contacts(extract_contacts(model, data), ix)
        # Each row describes state at t_k and the command held over [t_k, t_k+1).
        rows.append(dict(t=obs.t, phase=int(cmd.phase), touch_left=obs.touch_left, touch_right=obs.touch_right, q_left=obs.q_left, q_right=obs.q_right, lift_q=obs.lift_q, z_ref=cmd.z_ref, f_ref=cmd.f_ref, cube_z=float(obs.cube_pos[2]), tip_z=float(obs.tip_mid[2]), slip=cmd.slip, max_pen=obs.max_penetration, raw=cmd.raw.copy(), applied=cmd.applied.copy(), saturated=cmd.saturated.copy(), xfrc=data.xfrc_applied[ix.cube_body].copy(), dist_active=active, fn_left=s.left.normal_sum, fn_right=s.right.normal_sum, n_table=s.table.count, n_other=s.other_count, u_left=s.left.max_utilization, u_right=s.right.max_utilization, n_left=s.left.count, n_right=s.right.count, qpos=data.qpos.copy(), qvel=data.qvel.copy(), cube_pos=obs.cube_pos, cube_quat=data.qpos[ix.cube_qpos][3:7].copy(), tip_mid=obs.tip_mid, carriage_pos=data.xpos[ix.carriage_body].copy()))
        if cmd.phase in TERMINAL and t_terminal is None:
            t_terminal = obs.t
        if t_terminal is not None and obs.t - t_terminal >= TERMINAL_TAIL:
            break
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    log = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    log['reason'] = np.array(ctrl.reason)
    log['dt'] = np.array(dt)
    return log

def print_timeline(log: dict[str, np.ndarray]) -> None:
    phases = log['phase']
    for p in Phase:
        idx = np.flatnonzero(phases == int(p))
        if idx.size:
            print(f"  {p.name:<8} enters at t = {log['t'][idx[0]]:.3f} s")
    final = Phase(int(phases[-1]))
    print(f"  final: {final.name}  ({log['reason']})")
