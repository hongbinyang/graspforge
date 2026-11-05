# GraspForge

GraspForge is a standalone MuJoCo application for developing, running, and evaluating contact-rich parallel-gripper experiments.

## Why “GraspForge”?

**GraspForge** combines grasp manipulation with a forge—a place to build and refine robust controllers through repeatable simulation.

The current release provides a complete vertical grasp pipeline:

- a free-body cube, table, lift axis, mirrored finger slides, touch sensors, and named cameras;
- contact extraction with local-to-world force conversion and friction utilization;
- a guarded `SETTLE → CLOSE → PRELOAD → LIFT → HOLD` controller;
- PI fingertip-force control, minimum-jerk lift motion, actuator clipping, and anti-windup;
- controlled external disturbances through `xfrc_applied`;
- deterministic seeded trials over pose, friction, and disturbance variation;
- success evaluation, replayable NPZ logs, strict JSON metrics, plots, and rendering;
- a standalone deterministic regression suite.

## Status

Version: `0.1.0`

Current verification:

- 16/16 application regression tests pass;
- 20/20 default robustness seeds pass;
- the nominal grasp reaches `DONE` and lifts approximately 0.15 m;
- the nominal CLI run satisfies all force, penetration, slip, disturbance-buffer, and finite-state criteria;
- a two-seed CLI benchmark passes and records its configured threshold correctly;
- the project builds as a wheel with the MJCF asset included.

Verified reference artifacts are stored under `artifacts/reference/`. New runs are written under `artifacts/runs/`, which is intentionally ignored by Git.

## Project layout

```text
graspforge/
├── pyproject.toml
├── README.md
├── src/graspforge/
│   ├── assets/gripper.xml   # MuJoCo scene and named interfaces
│   ├── contacts.py          # model indexing and contact-force analysis
│   ├── controller.py        # observations, state machine, control, rollout
│   ├── evaluation.py        # criteria, trials, persistence, diagnostics
│   ├── cli.py               # application commands
│   └── __main__.py
├── tests/test_graspforge.py # standalone deterministic regression suite
└── artifacts/reference/     # verified reference outputs
```

The package is under `src/` so tests and scripts cannot accidentally import an uninstalled checkout as a different package.

## Environment and installation

The existing Conda environment can be reused:

```bash
conda activate mujoco
cd graspforge
python -m pip install -e .
```

Confirm the application and compiled model:

```bash
graspforge inspect
```

Expected dimensions are `nq=10`, `nv=9`, and `nu=3`.

## Run the application

Run the nominal disturbed grasp and write the episode plus diagnostics:

```bash
graspforge run
```

Include an offscreen final-state render:

```bash
graspforge run --render
```

Choose a different output directory or camera:

```bash
graspforge run --render --camera overview --output artifacts/runs/demo
```

Run the default 20-seed robustness benchmark:

```bash
graspforge benchmark
```

Run a smaller development benchmark:

```bash
graspforge benchmark --trials 5 --min-successes 5 --output artifacts/runs/smoke
```

Commands return a nonzero exit code when their success requirement is not met, which makes them suitable for CI or experiment automation.

## Tests

After the editable installation:

```bash
python tests/test_graspforge.py
```

The suite verifies model wiring, contact-frame mathematics, touch/contact agreement, legal state transitions, anti-windup, disturbance clearing, row alignment, deterministic replay, failure detection, and the multi-seed robustness threshold.

## Runtime architecture

`contacts.py` resolves every named MuJoCo ID and address once per model. Contact records copy MuJoCo-owned buffers so logged values remain stable after subsequent physics steps. Contact forces are returned by MuJoCo in the contact frame and converted with `frame.T @ f_local[:3]`.

`controller.py` owns the hybrid controller. Each rollout observes a forward-consistent state at time `t_k`, calculates and clips a command, records the command applied over `[t_k, t_k+1)`, advances physics, and calls `mj_forward` before the next observation. This alignment is part of the logging contract.

`evaluation.py` is deliberately separate from the controller. The controller decides whether its state machine completed; the evaluator independently decides whether the physical task met lift, hold, slip, force, penetration, disturbance-buffer, and numerical-health requirements.

`cli.py` is the application boundary. Library code does not parse arguments or assume an output directory. This keeps controller and evaluation functions reusable from notebooks, services, or future UI processes.

## Important invariants

Preserve these rules when extending the application:

1. Construct fresh `MjModel`, `MjData`, and controller instances for every seeded trial.
2. Resolve joint, sensor, body, geometry, and actuator addresses by name; never rely on coincidental integer ordering.
3. Read only `data.contact[:data.ncon]` and copy contact-backed arrays before storing them.
4. Treat MuJoCo contact-frame rows as world-space axes; contact-to-world force conversion uses the transpose.
5. Clear all six entries of `xfrc_applied` outside the disturbance window.
6. Log state and command at `t_k` before stepping, then run `mj_step` followed by `mj_forward`.
7. Keep trial sampling local to `np.random.default_rng(seed)` and preserve its draw order when reproducibility matters.
8. Never save object arrays in NPZ artifacts; files must load with `allow_pickle=False`.
9. Do not tune controller parameters per trial when reporting a robustness result.

## Data products

`grasp_episode.npz` contains aligned time-series arrays, phase and actuator name tables, column documentation, criteria values, and the terminal reason. It is compressed and pickle-free.

`grasp_metrics.json` contains MuJoCo version information, sampling ranges, controller configuration, success criteria, and one record per seeded trial. Non-finite values are converted to JSON `null`.

`grasp_diagnostics.png` plots height, touch force, contact normal force, finger positions, applied controls, slip, phase boundaries, and the disturbance window.

`grasp_final.png` is optional because rendering requires a working OpenGL context.

## Extension direction

The next phase should turn GraspForge from a single-scene experiment into a reusable manipulation platform. A practical order is:

1. Introduce an `ExperimentConfig` with YAML or TOML loading and validated controller, scene, trial, and output sections.
2. Replace the single cube-specific setup with task and object interfaces.
3. Add controller protocols so force PI, impedance control, and model-based policies can be selected at runtime.
4. Add batch execution with process isolation and aggregate reports.
5. Add replay and comparison commands for saved NPZ episodes.
6. Add video rendering and an interactive inspection UI.
7. Add CI across supported Python and MuJoCo versions.

Avoid adding concurrency before the experiment configuration and artifact identity are stable; otherwise it becomes difficult to prove which inputs produced a result.
