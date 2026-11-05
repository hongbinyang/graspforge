"""Independent success evaluation, seeded trials, and artifact generation."""
from dataclasses import dataclass

import mujoco
import numpy as np

from graspforge.contacts import GripperIndex, build_index, load_model
from graspforge.controller import ControllerConfig, Disturbance, Phase, run_grasp

@dataclass(frozen=True)
class SuccessCriteria:
    """Physical task requirements evaluated independently of controller status."""
    min_rise: float = 0.12
    max_slip: float = 0.02
    max_force: float = 25.0
    max_penetration: float = 0.005
    contact_force: float = 0.2

@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    value: float
    limit: str
    detail: str = ''

@dataclass(frozen=True)
class SuccessReport:
    criteria: tuple[Criterion, ...]

    @property
    def success(self) -> bool:
        return all((c.passed for c in self.criteria))

    @property
    def failures(self) -> list[str]:
        return [c.name for c in self.criteria if not c.passed]

    def to_dict(self) -> dict:
        return {'success': self.success, 'criteria': [dict(name=c.name, passed=c.passed, value=c.value, limit=c.limit, detail=c.detail) for c in self.criteria]}
# These signals are undefined outside the phases where their physical meaning exists.
NAN_ALLOWED = frozenset({'slip', 'f_ref', 'u_left', 'u_right'})

def evaluate_episode(log: dict[str, np.ndarray], crit: SuccessCriteria, hold_time: float, disturbance: Disturbance | None) -> SuccessReport:
    dt = float(log['dt'])
    phase = log['phase']
    n = len(log['t'])
    lift_rows = np.flatnonzero(phase == int(Phase.LIFT))
    hold_rows = np.flatnonzero(phase == int(Phase.HOLD))
    hold_steps = int(round(hold_time / dt))
    # Evaluation masks are derived from recorded phases, never controller internals.
    held = np.zeros(n, dtype=bool)
    if lift_rows.size:
        held[lift_rows[0]:] = True
    hold = np.zeros(n, dtype=bool)
    hold_complete = False
    if hold_rows.size:
        k_hold = int(hold_rows[0])
        hold[k_hold:k_hold + hold_steps] = True
        hold_complete = k_hold + hold_steps <= n
    z0 = float(log['cube_z'][0])
    out = []
    left = log['touch_left'] > crit.contact_force
    right = log['touch_right'] > crit.contact_force
    both = left & right
    idx = np.flatnonzero(both)
    t_contact = float(log['t'][idx[0]]) if idx.size else float('nan')
    out.append(Criterion('contact_established', idx.size > 0, t_contact, f'both > {crit.contact_force} N'))
    final = Phase(int(phase[-1]))
    out.append(Criterion('reached_done', final == Phase.DONE, float(phase[-1]), 'final phase DONE', detail=f"{final.name}: {log['reason']}"))
    h = log['cube_z'] - z0
    rise = float(h.max())
    out.append(Criterion('rise', rise >= crit.min_rise, rise, f'≥ {crit.min_rise} m'))
    h_hold = h[hold]
    low = float(h_hold.min()) if h_hold.size else float('nan')
    out.append(Criterion('lifted_throughout_hold', hold_complete and low >= crit.min_rise, low, f'≥ {crit.min_rise} m in HOLD', detail='' if hold_complete else 'episode ended before HOLD was complete'))
    s = log['slip'][held]
    n_undefined = int((~np.isfinite(s)).sum())
    peak = float(np.nanmax(s)) if np.isfinite(s).any() else float('nan')
    out.append(Criterion('slip', s.size > 0 and n_undefined == 0 and (peak < crit.max_slip), peak, f'< {crit.max_slip} m', detail=f'{n_undefined} undefined rows in LIFT/HOLD' if n_undefined else ''))
    peak = float(max(log['touch_left'].max(), log['touch_right'].max(), log['fn_left'].max(), log['fn_right'].max()))
    out.append(Criterion('fingertip_force', peak < crit.max_force, peak, f'< {crit.max_force} N'))
    m_p = log['max_pen']
    max_m_p = float(np.max(m_p))
    out.append(Criterion('penetration', max_m_p < crit.max_penetration, max_m_p, f'< {crit.max_penetration} m'))
    bad = []
    for name, arr in log.items():
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        broken = np.isinf(arr).any() if name in NAN_ALLOWED else (~np.isfinite(arr)).any()
        if broken:
            bad.append(name)
    out.append(Criterion('finite', not bad, float(len(bad)), '0 non-finite columns', detail=', '.join(bad)))
    n = len(log['t'])
    expected = np.zeros((n, 6))
    hold_rows = np.flatnonzero(hold)
    if disturbance is not None and hold_rows.size:
        k_hold = int(hold_rows[0])
        start = k_hold + int(round(disturbance.hold_fraction * hold_time / dt))
        end = start + int(round(disturbance.duration / dt))
        expected[start:end, :3] = disturbance.force
    mismatch = np.any(log['xfrc'] != expected, axis=1)
    rows = np.flatnonzero(mismatch)
    out.append(Criterion('disturbance_buffer', rows.size == 0, float(rows.size), '0 mismatched rows', detail=f"first mismatch at t = {log['t'][rows[0]]:.3f} s" if rows.size else ''))
    return SuccessReport(tuple(out))

def print_report(rep: SuccessReport) -> None:
    for c in rep.criteria:
        mark = 'PASS' if c.passed else 'FAIL'
        print(f'  [{mark}] {c.name:<24} {c.value:>10.4g}   {c.limit:<22} {c.detail}')
    print(f"  overall: {('SUCCESS' if rep.success else 'FAILURE ' + str(rep.failures))}")

def analyze_disturbance(log: dict[str, np.ndarray], mu: float, cube_mass: float, g: float=9.81) -> dict[str, float]:
    """Measured response to the disturbance vs the quasi-static friction budget."""
    act = log['dist_active']
    idx = np.flatnonzero(act)
    if idx.size == 0:
        raise ValueError('no disturbance window in this log')
    k0, k1 = (int(idx[0]), int(idx[-1]) + 1)
    hold = log['phase'] == int(Phase.HOLD)
    k_hold_end = int(np.flatnonzero(hold)[-1])
    force = log['xfrc'][k0, :3]
    f_n = float(0.5 * (log['fn_left'][k0 - 1] + log['fn_right'][k0 - 1]))
    f_t = float(np.sqrt((cube_mass * g / 2) ** 2 + (abs(force[1]) / 2) ** 2))
    capacity = mu * f_n
    u_pred = f_t / capacity
    u_before = float(np.mean(np.fmax(log['u_left'][k0 - 50:k0], log['u_right'][k0 - 50:k0])))
    u_peak = float(np.fmax(log['u_left'][k0:k1], log['u_right'][k0:k1]).max())
    slip_start = float(log['slip'][k0 - 1])
    slip_end = float(log['slip'][k1 - 1])
    slip_hold = float(log['slip'][k_hold_end])
    tl, tr = (log['touch_left'][k0:k1], log['touch_right'][k0:k1])
    touch_span = float(max(tl.max(), tr.max()) - min(tl.min(), tr.min()))
    return dict(u_pred=u_pred, u_before=u_before, u_peak=u_peak, slip_start=slip_start, slip_end=slip_end, slip_hold=slip_hold, touch_span=touch_span, f_n=f_n)
import copy

def _doctored(log: dict[str, np.ndarray], column: str, rows, value) -> dict[str, np.ndarray]:
    """A copy of `log` with log[column][rows] = value. The original is untouched."""
    result = copy.deepcopy(log)
    result[column][rows] = value
    return result
import math
from dataclasses import asdict
from graspforge.controller import MAX_CUBE_OFFSET
OFFSET_RANGE = (-MAX_CUBE_OFFSET, MAX_CUBE_OFFSET)
YAW_RANGE_DEG = (-5.0, 5.0)
MU_RANGE = (0.6, 1.2)
DIST_MAG_RANGE = (2.0, 6.0)

@dataclass(frozen=True)
class TrialConfig:
    seed: int
    offset_x: float
    offset_y: float
    yaw_deg: float
    mu: float
    dist_angle: float
    dist_mag: float

    @property
    def disturbance(self) -> Disturbance:
        return Disturbance(force=(self.dist_mag * math.cos(self.dist_angle), self.dist_mag * math.sin(self.dist_angle), 0.0))

    def to_dict(self) -> dict:
        return asdict(self)

def sample_trial(seed: int) -> TrialConfig:
    rng = np.random.default_rng(seed)
    offset_x = float(rng.uniform(*OFFSET_RANGE))
    offset_y = float(rng.uniform(*OFFSET_RANGE))
    yaw_deg = float(rng.uniform(*YAW_RANGE_DEG))
    mu = float(rng.uniform(*MU_RANGE))
    dist_angle = float(rng.uniform(0.0, 2 * math.pi))
    dist_mag = float(rng.uniform(*DIST_MAG_RANGE))
    return TrialConfig(seed=int(seed), offset_x=offset_x, offset_y=offset_y, yaw_deg=yaw_deg, mu=mu, dist_angle=dist_angle, dist_mag=dist_mag)

def make_trial_setup(trial: TrialConfig, ix: GripperIndex):
    """Return setup(model, data) that places the cube at the trial's offset and yaw."""

    def setup(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qpos[ix.cube_qpos][0] += trial.offset_x
        data.qpos[ix.cube_qpos][1] += trial.offset_y
        half = 0.5 * math.radians(trial.yaw_deg)
        data.qpos[ix.cube_qpos][3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return setup

def apply_friction(model: mujoco.MjModel, ix: GripperIndex, mu: float) -> None:
    model.geom_friction[ix.cube_geom][0] = mu

@dataclass(frozen=True)
class TrialResult:
    trial: TrialConfig
    report: SuccessReport
    final_phase: str
    reason: str
    phase_entry: dict[str, float]
    u_peak: float
    u_pred: float
    slip_hold_end: float

    def to_dict(self) -> dict:
        return dict(trial=self.trial.to_dict(), report=self.report.to_dict(), final_phase=self.final_phase, reason=self.reason, phase_entry=dict(self.phase_entry), u_peak=self.u_peak, u_pred=self.u_pred, slip_hold_end=self.slip_hold_end)

def run_trial(seed: int, cfg: ControllerConfig=ControllerConfig(), crit: SuccessCriteria=SuccessCriteria(), return_log: bool=False):
    trial = sample_trial(seed)
    # A trial owns fresh model and data objects so sampled friction cannot leak between seeds.
    model = load_model()
    ix = build_index(model)
    apply_friction(model, ix, trial.mu)
    data = mujoco.MjData(model)
    log = run_grasp(model, data, ix, cfg, disturbance=trial.disturbance, setup=make_trial_setup(trial, ix))
    report = evaluate_episode(log, crit, cfg.hold_time, trial.disturbance)
    phase_entry = {}
    for p in Phase:
        rows = np.flatnonzero(log['phase'] == int(p))
        if rows.size:
            phase_entry[p.name] = float(log['t'][rows[0]])
    try:
        analysis = analyze_disturbance(log, trial.mu, float(model.body_mass[ix.cube_body]), -float(model.opt.gravity[2]))
        u_peak, u_pred = (analysis['u_peak'], analysis['u_pred'])
    except ValueError:
        u_peak = u_pred = float('nan')
    hold_rows = np.flatnonzero(log['phase'] == int(Phase.HOLD))
    k_end = int(hold_rows[0]) + int(round(cfg.hold_time / float(log['dt']))) - 1 if hold_rows.size else -1
    slip_hold_end = float(log['slip'][k_end]) if 0 <= k_end < len(log['slip']) else float('nan')
    result = TrialResult(trial=trial, report=report, final_phase=Phase(int(log['phase'][-1])).name, reason=str(log['reason']), phase_entry=phase_entry, u_peak=u_peak, u_pred=u_pred, slip_hold_end=slip_hold_end)
    return (result, log) if return_log else result

def _logs_identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    """Names of columns that differ between two logs (empty list = identical)."""
    a_keys, b_keys = (a.keys(), b.keys())
    commons = a_keys & b_keys
    differences = set(a_keys ^ b_keys)
    for k in commons:
        float_col = np.issubdtype(a[k].dtype, np.floating)
        if not np.array_equal(a[k], b[k], equal_nan=float_col):
            differences.add(k)
    return sorted(differences)
import json
from dataclasses import asdict
from graspforge.contacts import ARTIFACT_DIR
N_TRIALS = 20
MIN_SUCCESSES = 18
METRICS_PATH = ARTIFACT_DIR / 'grasp_metrics.json'

def _json_safe(x):
    """Convert to plain JSON types: NumPy → Python, nan/inf → None, tuples → lists."""

def _json_safe(x):
    """Convert to plain JSON types: NumPy → Python, nan/inf → None, tuples → lists."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    elif isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    elif isinstance(x, (bool, np.bool_)):
        return bool(x)
    elif isinstance(x, (int, np.integer)):
        return int(x)
    elif isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    else:
        return x

def run_suite(seeds, cfg: ControllerConfig=ControllerConfig(), crit: SuccessCriteria=SuccessCriteria()) -> list[TrialResult]:
    results = []
    for s in seeds:
        r = run_trial(s, cfg, crit)
        results.append(r)
        print(f"  trial {s:>2}: {('PASS' if r.report.success else 'FAIL ' + str(r.report.failures))}")
    return results

def write_metrics(results: list[TrialResult], cfg: ControllerConfig, crit: SuccessCriteria, path=METRICS_PATH, min_successes: int=MIN_SUCCESSES) -> dict:
    n_trials = len(results)
    if n_trials == 0:
        raise ValueError('results must contain at least one trial')
    successes = sum((r.report.success for r in results))
    seeds = [result.trial.seed for result in results]
    meta = {'mujoco_version': mujoco.__version__, 'n_trials': n_trials, 'successes': successes, 'success_rate': successes / n_trials, 'min_successes_required': min_successes, 'seeds': seeds, 'ranges': {'offset_m': OFFSET_RANGE, 'yaw_deg': YAW_RANGE_DEG, 'mu': MU_RANGE, 'dist_mag_N': DIST_MAG_RANGE}, 'controller_config': asdict(cfg), 'success_criteria': asdict(crit)}
    trials = [r.to_dict() for r in results]
    doc = {'meta': meta, 'trials': trials}
    safe = _json_safe(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    return safe

def _criterion(rep: SuccessReport, name: str) -> float:
    return next((c.value for c in rep.criteria if c.name == name))

def print_suite_table(results: list[TrialResult]) -> None:
    print(f"{'seed':>4} {'μ':>6} {'yaw°':>6} {'off_x':>6} {'push':>5} {'dir°':>5} {'final':>7} {'slip mm':>8} {'F max':>6} {'pen mm':>7} {'u_peak':>7} {'u_pred':>7}  failures")
    for r in results:
        t, rep = (r.trial, r.report)
        print(f"{t.seed:>4} {t.mu:6.3f} {t.yaw_deg:6.2f} {t.offset_x * 1000.0:6.2f} {t.dist_mag:5.2f} {math.degrees(t.dist_angle):5.0f} {r.final_phase:>7} {_criterion(rep, 'slip') * 1000.0:8.3f} {_criterion(rep, 'fingertip_force'):6.2f} {_criterion(rep, 'penetration') * 1000.0:7.3f} {r.u_peak:7.3f} {r.u_pred:7.3f}  {','.join(rep.failures)}")

def replay_step(model: mujoco.MjModel, ix: GripperIndex, log: dict[str, np.ndarray], k: int) -> np.ndarray:
    """qpos after one mj_step from row k's state with row k's inputs, on a FRESH MjData."""
    data = mujoco.MjData(model)
    act = [ix.lift_act, ix.left_act, ix.right_act]
    data.qpos[:] = log['qpos'][k]
    data.qvel[:] = log['qvel'][k]
    data.time = log['t'][k]
    for i, a in enumerate(act):
        data.ctrl[a] = log['applied'][k][i]
    data.xfrc_applied[ix.cube_body][:] = log['xfrc'][k]
    mujoco.mj_step(model, data)
    return data.qpos.copy()

def check_alignment(model, ix, log, k: int) -> dict[str, float]:
    """Distance from the replayed qpos to rows k, k+1, k+2."""
    q = replay_step(model, ix, log, k)
    return {f'row k{off:+d}': float(np.linalg.norm(q - log['qpos'][k + off])) for off in (0, 1, 2)}
EPISODE_PATH = ARTIFACT_DIR / 'grasp_episode.npz'
COLUMN_DOC = {'t': 's; time at the start of the row', 'phase': 'int; controller phase at t_k, see phase_names', 'touch_left': 'N; left touch sensor at t_k', 'touch_right': 'N; right touch sensor at t_k', 'q_left': 'm; left finger joint (positive = closed)', 'q_right': 'm; right finger joint', 'lift_q': 'm; lift joint above home', 'z_ref': 'm; lift reference', 'f_ref': 'N; finger force target (nan when not force-controlling)', 'cube_z': 'm; cube centre height', 'tip_z': 'm; fingertip midpoint height', 'slip': 'm; cube displacement relative to fingertips since grasp (nan before LIFT)', 'max_pen': 'm; deepest contact penetration at t_k', 'raw': 'N; commands over [t_k, t_k+1) before clipping, order = actuator_names', 'applied': 'N; commands over [t_k, t_k+1) after clipping, order = actuator_names', 'saturated': 'bool; raw != applied, order = actuator_names', 'xfrc': 'N, N·m; cube wrench over [t_k, t_k+1): force xyz, torque xyz (world)', 'dist_active': 'bool; disturbance window active over [t_k, t_k+1)', 'fn_left': 'N; Σ normal force, left pad–cube contacts at t_k', 'fn_right': 'N; same, right pad', 'n_table': 'int; cube–table contact points', 'n_other': 'int; unexpected contact points', 'u_left': '1; max friction utilization, left pad (nan if unloaded)', 'u_right': '1; same, right pad', 'n_left': 'int; left pad–cube contact points', 'n_right': 'int; right pad–cube contact points', 'qpos': 'mixed; full generalized positions at t_k (cube free joint first)', 'qvel': 'mixed; full generalized velocities at t_k', 'cube_pos': 'm; cube centre, world', 'cube_quat': '1; cube orientation (w, x, y, z)', 'tip_mid': 'm; fingertip midpoint, world', 'carriage_pos': 'm; carriage body origin, world'}

def save_episode(path, log: dict[str, np.ndarray], report: SuccessReport) -> dict[str, np.ndarray]:
    """Write the episode as an npz of plain arrays. Returns exactly what was saved."""
    arrays = {k: v for k, v in log.items() if k not in ('reason', 'dt')}
    undocumented = set(arrays) - set(COLUMN_DOC)
    if undocumented:
        raise ValueError(f'columns without COLUMN_DOC entries: {sorted(undocumented)}')
    arrays['dt'] = log['dt']
    arrays['reason'] = log['reason']
    arrays['phase_names'] = np.array([p.name for p in Phase], dtype=str)
    arrays['actuator_names'] = np.array(['lift_motor', 'left_motor', 'right_motor'], dtype=str)
    arrays['column_names'] = np.array(list(COLUMN_DOC), dtype=str)
    arrays['column_docs'] = np.array(list(COLUMN_DOC.values()), dtype=str)
    arrays['success'] = np.array(report.success)
    arrays['criteria_names'] = np.array([c.name for c in report.criteria], dtype=str)
    arrays['criteria_passed'] = np.array([c.passed for c in report.criteria], dtype=bool)
    arrays['criteria_values'] = np.array([c.value for c in report.criteria], dtype=float)
    arrays['criteria_limits'] = np.array([c.limit for c in report.criteria], dtype=str)
    arrays['failures'] = np.array(report.failures, dtype=str)
    # Object arrays would force consumers to enable unsafe pickle loading.
    for key, arr in arrays.items():
        if arr.dtype.kind == 'O':
            raise ValueError(f"'{key}' has dtype object; saving it would need pickle (check for None or mixed types in that column)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DIAGNOSTICS_PATH = ARTIFACT_DIR / 'grasp_diagnostics.png'

def _phase_boundaries(phase: np.ndarray, t: np.ndarray, names: np.ndarray) -> list[tuple[float, str]]:
    starts = np.concatenate(([0], np.flatnonzero(np.diff(phase)) + 1))
    return [(float(t[k]), str(names[phase[k]])) for k in starts]

def _true_spans(mask: np.ndarray, t: np.ndarray, dt: float) -> list[tuple[float, float]]:
    padded = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(padded)
    starts, ends = (np.flatnonzero(d == 1), np.flatnonzero(d == -1))
    return [(float(t[s]), float(t[e - 1]) + dt) for s, e in zip(starts, ends)]

def plot_diagnostics(npz_path, out_path=DIAGNOSTICS_PATH) -> None:
    with np.load(npz_path, allow_pickle=False) as z:
        e = {k: z[k] for k in z.files}
    t, dt = (e['t'], float(e['dt']))
    act_names = [str(a) for a in e['actuator_names']]
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    ax_h, ax_touch, ax_fn, ax_q, ax_u, ax_slip = axes
    ax_h.plot(t, e['cube_z'], label='cube centre')
    ax_h.plot(t, e['tip_z'], label='fingertip midpoint')
    ax_h.plot(t, e['z_ref'] + e['tip_z'][0], '--', lw=1, label='lift reference (+ start height)')
    ax_h.set_ylabel('height (m)')
    ax_touch.plot(t, e['touch_left'], label='touch left')
    ax_touch.plot(t, e['touch_right'], label='touch right')
    ax_touch.plot(t, e['f_ref'], 'k--', lw=1, label='force target')
    ax_touch.set_ylabel('touch (N)')
    ax_fn.plot(t, e['fn_left'], label='Σfn left (contacts)')
    ax_fn.plot(t, e['fn_right'], label='Σfn right (contacts)')
    ax_fn.set_ylabel('contact normal (N)')
    ax_q.plot(t, e['q_left'] * 1000.0, label='left finger')
    ax_q.plot(t, e['q_right'] * 1000.0, label='right finger')
    ax_q.set_ylabel('finger q (mm)')
    for i, name in enumerate(act_names):
        line, = ax_u.plot(t, e['applied'][:, i], label=f'{name} applied')
        sat = e['saturated'][:, i]
        if sat.any():
            ax_u.plot(t[sat], e['applied'][sat, i], 'x', color=line.get_color(), label=f'{name} saturated')
    ax_u.set_ylabel('control (N)')
    ax_slip.plot(t, e['slip'] * 1000.0, label='slip')
    ax_slip.set_ylabel('slip (mm)')
    ax_slip.set_xlabel('time (s)')
    bounds = _phase_boundaries(e['phase'], t, e['phase_names'])
    spans = _true_spans(e['dist_active'], t, dt)
    for ax in axes:
        for tb, _ in bounds:
            ax.axvline(tb, color='0.6', lw=0.8, ls=':')
        for t0, t1 in spans:
            ax.axvspan(t0, t1, color='tab:red', alpha=0.15)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    for tb, name in bounds:
        ax_h.annotate(name, (tb, 1.0), xycoords=('data', 'axes fraction'), xytext=(2, -10), textcoords='offset points', fontsize=8, color='0.3')
    verdict = 'SUCCESS' if bool(e['success']) else 'FAILURE: ' + ', '.join(e['failures'])
    fig.suptitle(f"Grasp episode — {verdict} ({e['reason']}); shaded: disturbance", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
FINAL_PATH = ARTIFACT_DIR / 'grasp_final.png'

def render_final(path, model: mujoco.MjModel, data: mujoco.MjData, camera: str='front', width: int=960, height: int=720, show_sites: bool=False) -> None:
    opt = mujoco.MjvOption()
    if not show_sites:
        opt.sitegroup[:] = 0
    # Renderer owns an OpenGL context and must be closed even when image writing fails.
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.update_scene(data, camera=camera, scene_option=opt)
        image = renderer.render()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, image)
    finally:
        renderer.close()

def _yaw_deg(quat: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan2(quat[:, 3], quat[:, 0]))
