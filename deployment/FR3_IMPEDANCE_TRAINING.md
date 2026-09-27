# FR3 Franky Impedance Training Environment

Task ID: `Franka-FR3v2-FrankyImpedance-Reach-v0`

This task is the deployment-oriented replacement for training through the
Panda USD's implicit joint-position drives. It uses the validated FR3v2.1
bare-flange asset and preserves the 24-observation/7-action reach interface.

## Control contract

- Episodes reset each joint uniformly within +/-0.05 rad of the deployment
  home vector, with zero joint velocity; they do not all start at the exact home pose.
- PhysX and the explicit torque controller run at 1 kHz.
- PPO runs at 50 Hz (`dt=0.001`, `decimation=20`).
- A tanh-squashed Gaussian keeps every sampled and deterministic PPO action inside
  `(-1, 1)`. Its final layer starts at the normalized deployment-home command
  and its initial latent standard deviation is 0.01.
- Each bounded normalized action maps linearly from `[-1, 1]` to the full
  reviewed soft joint-position range using its midpoint and half-range.
- The resulting joint-position target is held directly until the next policy
  update. There is no reference governor, interpolation, or motion planner.
- The explicit controller uses nominal `K=100 N m/rad`, `D=20 N m s/rad`,
  0.5 rad error clipping, Coriolis compensation, PhysX gravity compensation,
  1 N m/ms user-torque slew limiting, 100 Hz filtering, and the reviewed
  soft-limit repulsion.
- The last seven policy observations contain the normalized held command.
- Smoothness is learned with squared hinge penalties on command increments and
  1 kHz peak measured velocities above the 20% deployment envelope. Command
  increments are normalized by `v_max * 0.02 s`, so all joints share a physical
  scale. A clipping-overshoot penalty guards the interface, while valid actions
  anywhere in the full range have no magnitude penalty. The envelope remains a
  reward and qualification criterion, not a one-step termination or motion planner.
- A 1 kHz moving-link contact sensor penalizes each link whose peak contact
  force exceeds 1 N during a policy interval. In this free-space task the
  signal conservatively includes self-contact and unexpected scene contact.
- A joint-7-only L1 posture term has weight -0.01 relative to the deployment
  home pose. It remains secondary to Cartesian tracking across the reviewed range.
- The actuator armature is 0.003 kg m^2. The inherited 0.001 value produced a
  joint-7 numerical limit cycle; the observed stability boundary was between
  0.0011 and 0.0015 in the nominal plant. A 0.002 nominal value passed that
  check but its 0.0016 randomized lower bound failed under simultaneous plant
  variations. The 0.003 nominal value gives a 0.0024--0.0036 DR interval. This
  is a simulation stabilizer pending trace calibration, not a measured FR3
  motor parameter.

This model reproduces the reviewed controller equations and limits, but it is
not considered hardware-validated until simulation traces have been compared
against Franky traces from the same reference trajectory.

## Target-volume visualization

The dedicated view reads the position bounds directly from the selected task,
then draws a translucent blue bounds cuboid and an orange sample cloud in the
robot-base frame:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . python scripts/visualize_target_volume.py \
  --task Franka-FR3v2-FrankyImpedance-Reach-v0 \
  --num_samples 400 \
  --seed 42 \
  --device cuda:0 \
  --viz kit
```

## Validation

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . python -u scripts/validate_impedance_env.py \
  --task Franka-FR3v2-FrankyImpedance-Reach-v0 \
  --scenario nominal \
  --num_envs 16 \
  --steps 100 \
  --device cuda:0
```

Repeat with `--scenario fr3_impedance_gain_dr_v1` to exercise reset-sampled
controller gain and plant randomization.

## Training

Nominal:

```bash
direnv exec . python -u scripts/rsl_rl/train.py \
  --task Franka-FR3v2-FrankyImpedance-Reach-v0 \
  --scenario nominal \
  --num_envs 1024 \
  --max_iterations 300 \
  --run_name fr3_impedance_nominal_v3_bounded \
  --device cuda:0 \
  --viz none
```

Domain randomized:

```bash
direnv exec . python -u scripts/rsl_rl/train.py \
  --task Franka-FR3v2-FrankyImpedance-Reach-v0 \
  --scenario fr3_impedance_gain_dr_v1 \
  --num_envs 1024 \
  --max_iterations 300 \
  --run_name fr3_impedance_dr_v3_bounded \
  --device cuda:0 \
  --viz none
```

Begin with 256 environments if 1 kHz inverse-dynamics queries exceed GPU
memory or reduce throughput excessively, then scale based on measured usage.


## 50 Hz incremental-position task

Task ID: `Franka-FR3v2-FrankyImpedance-IncrementalReach-v0`

This is the candidate deployment task. It retains the same 1 kHz explicit
Franky torque loop, but the seven actor outputs are normalized position
increments rather than absolute positions. At 50 Hz, one action is bounded to
8.7 mrad on joints 1--4 and 10.44 mrad on joints 5--7. The integrated reference
can still traverse the complete reviewed soft joint range. Zero action holds
the current reference, including during reset and action-delay fill.

The policy observation is 31D: measured relative joint position (7), measured
joint velocity (7), flange position error (3), normalized integrated reference
(7), and the preceding normalized increment (7). Reference acceleration is
trained as a physical smoothness cost; there is no hidden interpolation or
quintic motion planner.

Finite smoke test:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u scripts/validate_impedance_env.py \
  --task Franka-FR3v2-FrankyImpedance-IncrementalReach-v0 \
  --scenario nominal \
  --num_envs 16 \
  --steps 100 \
  --device cuda:0
```

Nominal training:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u scripts/rsl_rl/train.py \
  --task Franka-FR3v2-FrankyImpedance-IncrementalReach-v0 \
  --scenario nominal \
  --num_envs 1024 \
  --max_iterations 150 \
  --run_name fr3_incremental_impedance_nominal_v1 \
  --device cuda:0 \
  --viz none
```

The existing `fr3_impedance_gain_dr_v1` scenario is compatible with this task;
its delayed-action neutral value resolves to zero increment. Train and qualify
the nominal policy before starting the DR comparison.

## Six-action precision revision

Task ID: `Franka-FR3v2-FrankyImpedance-Incremental6DReach-v0`

This is a separate task so existing seven-action checkpoints remain loadable.
The policy commands joints 1--6; joint 7 is initialized from the reset state
and then held by the impedance controller. Joint 7 remains in the measured
joint state, but its previous reference/action entries are removed, reducing
the policy interface from 31 observations/7 actions to 29 observations/6
actions. The joint-7 posture reward is disabled because the policy cannot
change it.

The original broad Cartesian tracking kernel is retained for approach. A
second narrow exponential term with `sigma=0.0025 m^2` (5 cm characteristic
distance) adds terminal-position resolution without simultaneously changing
the other reward weights.

## Circle-path evaluation

Task ID: `Franka-FR3v2-FrankyImpedance-Incremental6DCirclePath-v0`

This evaluation-only task loads the same six-action policy and low-level
controller as the random-point task, but replaces random pose commands with the
named `circle_xy` path in `source/franka_rl/franka_rl/config/paths.yaml`. It
advances to the next waypoint when the flange is within the configured
position threshold or when that waypoint's deadline expires. A complete path
is successful only if every waypoint was reached before its deadline.

The evaluation artifacts record whole-path success, waypoint reach and timeout
counts, trajectory error, and the existing safety/control metrics. Path
geometry, thresholds, timeout, catalog source, and catalog hash are copied into
the run metadata.

Visual single-path evaluation:

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
cd "$FRANKA_RL_DATA_ROOT/runs"
direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  /home/chen-lab/isaac/franka-rl/scripts/rsl_rl/evaluate.py \
  --task Franka-FR3v2-FrankyImpedance-Incremental6DCirclePath-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 1 \
  --num_episodes 1 \
  --seed 123 \
  --device cuda:0 \
  --deterministic \
  --viz kit \
  --real-time
```

Validate before training:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u scripts/validate_impedance_env.py \
  --task Franka-FR3v2-FrankyImpedance-Incremental6DReach-v0 \
  --scenario nominal \
  --num_envs 16 \
  --steps 100 \
  --device cuda:0
```

Nominal training uses its own experiment directory; the checkpoint cadence is
left at the runner default rather than adding frequent saves:

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
mkdir -p "$FRANKA_RL_DATA_ROOT/runs"
cd "$FRANKA_RL_DATA_ROOT/runs"
direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  /home/chen-lab/isaac/franka-rl/scripts/rsl_rl/train.py \
  --task Franka-FR3v2-FrankyImpedance-Incremental6DReach-v0 \
  --scenario nominal \
  --num_envs 1024 \
  --max_iterations 150 \
  --run_name fr3_incremental_6d_fine_nominal_v1 \
  --device cuda:0 \
  --viz none
```
