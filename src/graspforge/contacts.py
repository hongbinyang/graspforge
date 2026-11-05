"""Model indexing and contact-force analysis for the GraspForge scene."""
from pathlib import Path
import mujoco
import numpy as np
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from collections import defaultdict

@dataclass(frozen=True)
class GripperIndex:
    """Named MuJoCo IDs and state addresses resolved for one compiled model."""
    carriage_body: int
    cube_body: int
    cube_geom: int
    table_geom: int
    left_pad: int
    right_pad: int
    cube_site: int
    left_tip: int
    right_tip: int
    cube_qpos: slice
    cube_qvel: slice
    lift_qpos: int
    left_qpos: int
    right_qpos: int
    lift_dof: int
    left_dof: int
    right_dof: int
    lift_act: int
    left_act: int
    right_act: int
    left_touch: int
    right_touch: int
    pair_labels: Mapping[frozenset, str] = field(compare=False, repr=False)

def build_index(model: mujoco.MjModel) -> GripperIndex:
    cube_j = model.joint('cube_free').id
    lift_j = model.joint('lift').id
    left_j = model.joint('left_slide').id
    right_j = model.joint('right_slide').id
    # A free joint occupies seven configuration entries but six tangent velocities.
    q0 = int(model.jnt_qposadr[cube_j])
    v0 = int(model.jnt_dofadr[cube_j])
    cube_qpos = slice(q0, q0 + 7)
    cube_qvel = slice(v0, v0 + 6)
    # Sensor IDs are not sensordata addresses when preceding sensors have dimension > 1.
    left_touch = int(model.sensor_adr[model.sensor('left_touch').id])
    right_touch = int(model.sensor_adr[model.sensor('right_touch').id])
    cube_g = model.geom('cube_geom').id
    table_g = model.geom('table').id
    left_g = model.geom('left_pad').id
    right_g = model.geom('right_pad').id
    pair_labels = MappingProxyType({frozenset((cube_g, table_g)): 'cube-table', frozenset((left_g, cube_g)): 'left_pad-cube', frozenset((right_g, cube_g)): 'right_pad-cube'})
    return GripperIndex(cube_body=model.body('cube').id, cube_geom=model.geom('cube_geom').id, table_geom=model.geom('table').id, left_pad=model.geom('left_pad').id, right_pad=model.geom('right_pad').id, cube_site=model.site('cube_center').id, left_tip=model.site('left_tip').id, right_tip=model.site('right_tip').id, cube_qpos=cube_qpos, cube_qvel=cube_qvel, lift_qpos=int(model.jnt_qposadr[lift_j]), left_qpos=int(model.jnt_qposadr[left_j]), right_qpos=int(model.jnt_qposadr[right_j]), lift_dof=int(model.jnt_dofadr[lift_j]), left_dof=int(model.jnt_dofadr[left_j]), right_dof=int(model.jnt_dofadr[right_j]), lift_act=model.actuator('lift_motor').id, left_act=model.actuator('left_motor').id, right_act=model.actuator('right_motor').id, left_touch=left_touch, right_touch=right_touch, carriage_body=model.body('carriage').id, pair_labels=pair_labels)

@dataclass(frozen=True)
class ContactRecord:
    """One live contact point, fully resolved. Forces follow MuJoCo's
    convention: the force exerted BY geom1 ON geom2."""
    index: int
    geom1: int
    geom2: int
    geom1_name: str
    geom2_name: str
    body1_name: str
    body2_name: str
    pos: np.ndarray
    frame: np.ndarray
    dist: float
    penetration: float
    mu: float
    f_local: np.ndarray
    f_normal: float
    f_tangential: float
    f_world: np.ndarray
    utilization: float

    @property
    def normal(self) -> np.ndarray:
        return self.frame[0]
FORCE_EPS = 1e-09

def extract_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[ContactRecord]:
    records = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = (int(c.geom1), int(c.geom2))
        # Contact buffers belong to MjData and are overwritten by the next forward pass.
        frame = c.frame.reshape(3, 3).copy()
        pos = c.pos.copy()
        f_local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f_local)
        # MuJoCo stores world-space contact axes in rows, hence the transpose.
        f_world = frame.T @ f_local[:3]
        f_normal = f_local[0]
        f_tangential = np.linalg.norm(f_local[1:3])
        mu = c.friction[0]
        utilization = f_tangential / (mu * f_normal) if f_normal > FORCE_EPS and mu > 0 else float('nan')
        penetration = max(0.0, -c.dist)
        records.append(ContactRecord(index=i, geom1=g1, geom2=g2, geom1_name=model.geom(g1).name, geom2_name=model.geom(g2).name, body1_name=model.body(model.geom_bodyid[g1]).name, body2_name=model.body(model.geom_bodyid[g2]).name, pos=pos, frame=frame, dist=float(c.dist), penetration=float(penetration), mu=float(mu), f_local=f_local, f_normal=float(f_normal), f_tangential=float(f_tangential), f_world=f_world, utilization=float(utilization)))
    return records

def print_records(records: list[ContactRecord]) -> None:
    print(f"{'i':>2} {'geom1':>10} {'geom2':<10} {'fn':>8} {'|ft|':>8} {'u':>6} {'pen':>9}  f_world")
    for r in records:
        print(f'{r.index:>2} {r.geom1_name:>10} {r.geom2_name:<10} {r.f_normal:8.4f} {r.f_tangential:8.4f} {r.utilization:6.3f} {r.penetration:9.2e}  {np.round(r.f_world, 4)}')

def pair_label(rec: ContactRecord, ix: GripperIndex) -> str:
    """Order-independent label: 'left_pad-cube', 'right_pad-cube', 'cube-table', or 'other'."""
    return ix.pair_labels.get(frozenset((rec.geom1, rec.geom2)), 'other')

def force_on_cube(rec: ContactRecord, ix: GripperIndex) -> np.ndarray:
    """World-frame force that this contact exerts ON the cube, shape (3,)."""
    if rec.geom2 == ix.cube_geom:
        return rec.f_world.copy()
    elif rec.geom1 == ix.cube_geom:
        return -rec.f_world
    else:
        raise ValueError(f'contact {rec.index} ({rec.geom1_name}–{rec.geom2_name}) does not involve the cube')

@dataclass(frozen=True)
class ContactGroup:
    """Aggregate of every contact sharing one label."""
    count: int
    normal_sum: float
    force_on_cube: np.ndarray
    max_utilization: float
    max_penetration: float

@dataclass(frozen=True)
class ContactSummary:
    left: ContactGroup
    right: ContactGroup
    table: ContactGroup
    other_count: int
    ncon: int

def _aggregate(group: list[ContactRecord], ix: GripperIndex) -> ContactGroup:
    count = len(group)
    normal_sum = 0.0
    sum_force_on_cube = np.zeros(3)
    max_utilization = -np.inf
    max_penetration = 0.0
    for cr in group:
        normal_sum += cr.f_normal
        sum_force_on_cube += force_on_cube(cr, ix)
        if np.isfinite(cr.utilization):
            max_utilization = max(max_utilization, cr.utilization)
        max_penetration = max(max_penetration, cr.penetration)
    return ContactGroup(count=count, normal_sum=normal_sum, force_on_cube=sum_force_on_cube, max_utilization=np.nan if max_utilization == -np.inf else max_utilization, max_penetration=max_penetration)

def summarize_contacts(records: list[ContactRecord], ix: GripperIndex) -> ContactSummary:
    bins = defaultdict(list)
    for rec in records:
        bins[pair_label(rec, ix)].append(rec)
    summary = ContactSummary(left=_aggregate(bins['left_pad-cube'], ix), right=_aggregate(bins['right_pad-cube'], ix), table=_aggregate(bins['cube-table'], ix), other_count=len(bins['other']), ncon=len(records))
    total = summary.left.count + summary.right.count + summary.table.count + summary.other_count
    assert total == summary.ncon == len(records), 'contacts lost or double-counted'
    return summary

def print_summary(s: ContactSummary) -> None:
    for name, g in (('left', s.left), ('right', s.right), ('table', s.table)):
        print(f'{name:>5}: n={g.count}  Σfn={g.normal_sum:8.4f} N  F_cube={np.round(g.force_on_cube, 4)}  u_max={g.max_utilization:.3f}  pen_max={g.max_penetration:.2e} m')
    print(f'other: n={s.other_count}   ncon={s.ncon}')
PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / 'assets' / 'gripper.xml'
# Library callers may override every output path; this value supports interactive use.
DEFAULT_OUTPUT_DIR = Path.cwd() / 'artifacts'
ARTIFACT_DIR = DEFAULT_OUTPUT_DIR
TABLE_TOP_Z = 0.4
CUBE_HALF = 0.05

def load_model(path: Path=MODEL_PATH) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))

def render_snapshot(model, data, path, camera='front', width=640, height=480):
    """Render one frame from a named camera to a PNG (visual sanity check)."""
    import matplotlib.pyplot as plt
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.update_scene(data, camera=camera)
        plt.imsave(path, renderer.render())
    finally:
        renderer.close()
CUBE_MASS = 0.2
SETTLE_SECONDS = 1.0

def max_penetration(data: mujoco.MjData) -> float:
    """Deepest penetration among live contacts, in metres (0 if none penetrate)."""
    if data.ncon == 0:
        return 0.0
    dists = data.contact.dist[:data.ncon]
    return float(max(0.0, -dists.min()))
PAD_GAP_HOME = 0.02
GRIPPER_MASS = 0.6
LIFT_CTRL_LIMIT = 30.0
FINGER_CTRL_LIMIT = 15.0
SQUEEZE_FORCE = 8.0
SQUEEZE_SECONDS = 2.0

def gravity_feedforward(model: mujoco.MjModel, ix: GripperIndex) -> float:
    mass = model.body_subtreemass[ix.carriage_body]
    gravity = model.opt.gravity[2]
    return -(mass * gravity)

def _set_fingers_at_contact(model: mujoco.MjModel, data: mujoco.MjData, ix: GripperIndex) -> None:
    """Home pose, but with both pad faces exactly on the cube faces."""
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    data.qpos[ix.left_qpos] = PAD_GAP_HOME
    data.qpos[ix.right_qpos] = PAD_GAP_HOME
    mujoco.mj_forward(model, data)

def hold_squeeze(model: mujoco.MjModel, data: mujoco.MjData, ix: GripperIndex, seconds: float, force: float=SQUEEZE_FORCE) -> None:
    """Hold the lift with gravity feedforward and push both fingers with `force`."""
    ff = gravity_feedforward(model, ix)
    for _ in range(int(round(seconds / model.opt.timestep))):
        data.ctrl[ix.lift_act] = ff
        data.ctrl[ix.left_act] = force
        data.ctrl[ix.right_act] = force
        mujoco.mj_step(model, data)
from dataclasses import replace

def swapped(rec: ContactRecord) -> ContactRecord:
    """The same physical contact, reported with geom1 and geom2 exchanged.

    Physics doesn't change; only the bookkeeping does:
      - the normal must point from the NEW geom1 to the NEW geom2, so n' = −n;
      - keep the frame right-handed (n × t1 = t2): with n' = −n and t1' = t1,
        that forces t2' = −t2;
      - f_world becomes the force on the NEW geom2 (old geom1), i.e. −f_world;
      - f_local re-expressed in the new frame: fn unchanged (still a push),
        ft1' = −ft1, ft2' = ft2. Check: frame'.T @ f_local'[:3] = −f_world.
    """
    n, t1, t2 = rec.frame
    frame = np.stack([-n, t1, -t2])
    f_local = rec.f_local.copy()
    f_local[1] = -f_local[1]
    return replace(rec, geom1=rec.geom2, geom2=rec.geom1, geom1_name=rec.geom2_name, geom2_name=rec.geom1_name, body1_name=rec.body2_name, body2_name=rec.body1_name, frame=frame, f_local=f_local, f_world=-rec.f_world)
NORMAL_TOL = 1e-09
CONE_TOL = 1e-06
NEWTON_TOL = 1e-08
CALIBRATION_TOL = 0.05
BALANCE_TOL = 0.05

@dataclass(frozen=True)
class TouchComparison:
    touch_left: float
    touch_right: float
    contact_left: float
    contact_right: float
    err_left: float
    err_right: float
    imbalance: float

def _rel_err(measured: float, reference: float) -> float:
    if abs(reference) <= FORCE_EPS:
        return np.nan
    return abs(measured - reference) / abs(reference)

def compare_touch(model: mujoco.MjModel, data: mujoco.MjData, ix: GripperIndex) -> TouchComparison:
    """Compare touch sensors against contact-derived normal forces.

    Precondition: data is the result of a forward pass (mj_forward, or the
    pass inside the last mj_step) and has not been modified since. Both the
    sensor values and the contacts are read from that same pass.
    """
    summary = summarize_contacts(extract_contacts(model, data), ix)
    touch_left = float(data.sensordata[ix.left_touch])
    touch_right = float(data.sensordata[ix.right_touch])
    err_left = _rel_err(touch_left, summary.left.normal_sum)
    err_right = _rel_err(touch_right, summary.right.normal_sum)
    imbalance = np.nan if (touch_left + touch_right) / 2.0 <= FORCE_EPS else abs(touch_left - touch_right) / ((touch_left + touch_right) / 2.0)
    return TouchComparison(touch_left=touch_left, touch_right=touch_right, contact_left=summary.left.normal_sum, contact_right=summary.right.normal_sum, err_left=err_left, err_right=err_right, imbalance=imbalance)

def print_comparison(c: TouchComparison) -> None:
    print(f'touch   L={c.touch_left:8.4f}  R={c.touch_right:8.4f} N   imbalance={c.imbalance:.2%}')
    print(f'contact L={c.contact_left:8.4f}  R={c.contact_right:8.4f} N')
    print(f'error   L={c.err_left:.2e}  R={c.err_right:.2e}')

def _inside_site(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, point: np.ndarray) -> bool:
    """Is a world-frame point inside a BOX site (boundary inclusive)?"""
    assert model.site_type[site_id] == mujoco.mjtGeom.mjGEOM_BOX, 'only box sites supported'
    centre = data.site_xpos[site_id]
    R = data.site_xmat[site_id].reshape(3, 3)
    half = model.site_size[site_id]
    local = R.T @ (point - centre)
    return bool(np.all(np.abs(local) <= half))

def touch_replica(model: mujoco.MjModel, data: mujoco.MjData, ix: GripperIndex, records: list[ContactRecord], side: str) -> float:
    """Recompute a touch sensor from contact records (point-in-zone rule only)."""
    site = ix.left_tip if side == 'left' else ix.right_tip
    total = 0.0
    site_body_id = model.site_bodyid[site]
    for r in records:
        if not (model.geom_bodyid[r.geom1] == site_body_id or model.geom_bodyid[r.geom2] == site_body_id):
            continue
        if not _inside_site(model, data, site, r.pos):
            continue
        if not r.f_normal > 0:
            continue
        total += r.f_normal
    return total
OPEN_FORCE = -5.0
RELEASE_SECONDS = 0.3

def _settle_index(trace: np.ndarray) -> int | None:
    """First index from which `trace` is zero for good; None if it never gets there."""
    nonzero = np.flatnonzero(trace)
    if nonzero.size == 0:
        return 0
    last = int(nonzero[-1])
    if last == trace.size - 1:
        return None
    return last + 1
SWEEP_FORCES = (2.0, 4.0, 8.0, 12.0)
SLOPE_TOL = 0.01

def _proportional_slope(x: np.ndarray, y: np.ndarray) -> float:
    multiply = 0.0
    square = 0.0
    for xi, yi in zip(x, y):
        multiply += float(xi * yi)
        square += float(xi * xi)
    return multiply / square

def force_to_world(frame: np.ndarray, f_local: np.ndarray) -> np.ndarray:
    return frame.T @ f_local[:3]

def friction_utilization(f_normal: float, f_tangential: float, mu: float) -> float:
    return float('nan') if f_normal <= FORCE_EPS or mu <= 0 else abs(f_tangential) / (mu * f_normal)
