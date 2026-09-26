# FR3 joint-velocity policy task

## Status

The simulation-side first milestone is implemented. The registered task is:

```text
Franka-FR3v2-JointVelocity-Reach-v0
```

Its deployment contract is `fr3_reach_joint_velocity_v1`. Position-policy
checkpoints and bundles are not compatible even though both policies currently
use a 24-value observation and seven actions.

The task is suitable for simulation experiments. It is **not yet qualified for
hardware deployment**. Values marked provisional below must be replaced or
bounded using measurements from Franky's `JointVelocityMotion` on the FR3.

## Policy contract

The policy runs at 30 Hz and emits seven normalized target-velocity actions.
The action processor:

1. rejects non-finite commands and latches an episode fault;
2. clips normalized commands to `[-1, 1]`;
3. maps them to a provisional `[-0.30, 0.30] rad/s` range;
4. applies a provisional `1.0 rad/s^2` policy-rate acceleration limit;
5. applies a position-dependent stopping-distance envelope with a 0.10 rad
   joint-limit margin;
6. sends the governed velocity target to a zero-stiffness implicit drive;
7. adds PhysX gravity-compensation efforts every physics step.

The policy timestep is 1/30 s. Physics runs at 120 Hz with decimation four.
The provisional simulated velocity servo uses damping 40 for joints 1--4 and
20 for joints 5--7. These are simulation placeholders, not identified FR3
internal controller gains.

Observation order:

```text
0:7    joint positions relative to the default pose [rad]
7:14   measured joint velocities [rad/s]
14:17  flange position error in the base frame [m]
17:24  previous finite, clipped normalized velocity action
```

## Local validation

With the repository `.envrc` loaded:

```bash
python -u scripts/validate_velocity_env.py \
  --num_envs 16 \
  --device cuda:0 \
  --viz none

python -u scripts/jacobian_velocity_agent.py \
  --num_envs 256 \
  --num_steps 300 \
  --device cuda:0 \
  --viz none
```

The implementation validation on 2026-09-25 produced zero zero-action drift,
correct directed joint motion, and no resets. A 64-environment, 300-step DLS
run reported 100% ever below 3 cm, 0.0269 m final mean error, 0.0020 m best mean
error, and zero failure resets.

A five-iteration PPO integration run completed under:

```text
logs/rsl_rl/fr3_velocity_reach/
2026-09-25_22-22-12_implementation_smoke
```

This verifies rollout, reward, clipping, learning, and artifact paths only; it
is not a trained policy.

## Provisional nominal training

Do not call this result deployment-ready. It can be used to test whether PPO
learns the velocity-action formulation while hardware characterization is
pending:

```bash
cd "$FRANKA_RL_DATA_ROOT/runs"

/home/chen-lab/isaac/.venv/bin/python -u \
  /home/chen-lab/isaac/franka-rl/scripts/rsl_rl/train.py \
  --task Franka-FR3v2-JointVelocity-Reach-v0 \
  --scenario nominal \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 123 \
  --run_name provisional_velocity_nominal \
  --device cuda:0 \
  --viz none
```

## Provisional domain-randomized training

The packaged profile is `fr3_velocity_dr_train_v1`.

It resamples the following per environment at reset:

- velocity-servo damping: 0.80--1.20 times nominal;
- joint friction addition: 0.00--0.10 and armature: 0.80--1.20 times nominal;
- link inertia: 0.90--1.10 times nominal;
- flange payload: 0--1 kg, with the configured three-axis COM range;
- small joint-state and task-error observation noise;
- initial joint-position offset: plus or minus 0.125 rad;
- action delay: zero or one 30 Hz policy step;
- target-velocity response: 0.85--1.15 times nominal; and
- acceleration response: 0.75--1.25 times nominal.

It deliberately does not randomize stiffness or effort limits. Stiffness is
zero for this velocity servo, while hardware safety effort limits are fixed
constraints rather than uncertain control parameters.

Validate the sampled response factors and randomized physics with:

```bash
python -u scripts/validate_velocity_env.py \
  --scenario fr3_velocity_dr_train_v1 \
  --num_envs 64 \
  --device cuda:0 \
  --viz none
```

Train the provisional DR policy from the data volume:

```bash
cd "$FRANKA_RL_DATA_ROOT/runs"

/home/chen-lab/isaac/.venv/bin/python -u \
  /home/chen-lab/isaac/franka-rl/scripts/rsl_rl/train.py \
  --task Franka-FR3v2-JointVelocity-Reach-v0 \
  --scenario fr3_velocity_dr_train_v1 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 123 \
  --run_name provisional_velocity_dr_v1 \
  --device cuda:0 \
  --viz none
```

This profile is for simulation integration and early comparison only. Replace
its command-response and latency ranges using the Franky measurements below
before training a deployment-candidate policy.

## Hardware measurements required next

Use the pinned Franky build documented in `franky_runtime/README.md`. Start with
small, separately approved joint-velocity motions; do not run PPO on hardware.
For each approved joint and direction, record:

- requested 30 Hz target velocity;
- Franky's 1 kHz generated joint-velocity control signal;
- `q`, `dq`, `q_d`, `dq_d`, and `ddq_d`;
- measured and desired joint torques;
- callback robot time, host receipt time, and target replacement time;
- current and last motion errors;
- stop latency after a stale-command watchdog event.

Required experiments are zero velocity, a small smooth pulse, a sign reversal,
30 Hz changing targets, repeated identical targets, and one intentionally
omitted 30 Hz update. Begin around 0.03 rad/s and do not exceed reviewed limits.

The measurements must answer:

1. Does 30 Hz asynchronous replacement of `JointVelocityMotion` remain smooth?
2. What velocity, acceleration, and jerk does Franky actually emit at the
   selected relative dynamics factor?
3. Does an identical target act as a keepalive without restarting Ruckig?
4. What are the effective tracking lag, overshoot, and steady-state error?
5. Does zero velocity hold the arm under the firmware's internal joint
   impedance controller?
6. What exact action should the watchdog issue before
   `JointVelocityStopMotion`?

After these measurements, update the simulated velocity scale, acceleration and
jerk transition model, servo response, delay randomization, and watchdog model.
Train deployment-candidate nominal and DR policies only after that update.
