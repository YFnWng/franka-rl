# Industrial motion demos: M0

M0 provides a portable NumPy/YAML package, `franka_motion`, with no Isaac Lab,
Torch, Franky or robot initialization on import. Existing PPO tasks are unchanged.
No physical controller, IK, collision check, simulator runner or hardware runner
is implemented by this milestone.

## Implemented contract

- Joint reference: seven positions in rad followed by seven velocities in rad/s.
- Cartesian reference: base-frame position (3), explicit `xyzw` quaternion (4),
  linear velocity (3) and angular velocity (3), totaling 13 physical values.
- Frames: `fr3_link0` base and `fr3_flange` TCP, identity flange-to-TCP transform.
- Measured samples have host/robot clocks and optional torque signals. Missing
  torque is `None`, never fabricated as zero.
- YAML catalogs define model provenance, separate model/command/abort intervals,
  initial posture, trajectory geometry/time laws, controller candidates and suites.
  Suites expand speed × twist-feedforward × repeat cells. Unknown/duplicate keys,
  malformed references and unapproved hardware suites are rejected.

Circle and Lissajous geometry have analytic derivatives. A quintic **speed** ramp
controls phase, including ramp displacement in the requested number of turns.
Approach/exit use quintic position segments with continuous position, velocity
and acceleration at joins; their jerk is finite but can jump at joins. Tracking
phase speed/acceleration/jerk joins are continuous. No derivative clipping is used.
Disabling twist feedforward leaves the geometric path and its physical derivatives
unchanged; only the commanded twist becomes zero.

`validate_time_law()` uses analytic componentwise extrema. `validate_trajectory()`
checks sampled analytic Cartesian derivative norms and a numerical box, explicitly
not a continuous-time certificate, reachability result or collision approval.
The initial preview pose is numerical input, not home FK. All paths hold a fixed
orientation; later robot runners must reconcile initial orientation before motion.

## Commands

M0 was installed editable into the existing Isaac virtual environment. To reproduce:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -m pip install \
  --no-deps --no-build-isolation -e source/franka_motion

direnv exec . /home/chen-lab/isaac/.venv/bin/python -m pytest tests/motion_control -q

direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/audit_motion_inputs.py

direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/preview_trajectory.py \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml \
  --output-dir /tmp/franka-motion-demo-a-preview
```

The preview writes `reference.npz`, `reference.png` and `validation.json` to a
**new** explicit directory; it refuses overwrite. Use another directory for reruns.
`--trial` selects an expanded ID, e.g. `circle__cell_001__repeat_000` (see YAML
experiment name and sweep order). `--no-plot` removes the optional Matplotlib need.
No GUI/Kit process starts. Files in `/tmp` are small disposable verification
artifacts; real experiment datasets will use the supported data volume.

## Validation and remaining gate

22 tests pass. An independently built/extracted wheel loads its five packaged
YAML catalogs/suites without simulator imports. Both USD/URDF hashes and URDF
joint limits pass the local audit; the archived runtime model validation is
successful. This is not a new dynamics test.

Controller gains, reference hold and Cartesian limits are **simulation candidates**.
The installed real-time Franky Cartesian/posture source, reference interpolation,
torque processing and stop semantics still need auditing. See the M0 request in
`deployment/HANDOFF_AGENT.md`. M1 adds joint servo/PTP simulation; M2 adds full-pose
Cartesian tracking and parity checks before hardware transfer.
