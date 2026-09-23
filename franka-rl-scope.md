# Franka RL Motion Control Demo — Repository Scope Handoff

## 1. Project Purpose

Build a compact reinforcement-learning motion-control demo using a Franka manipulator, primarily in Isaac Lab, with optional deployment to a real Franka arm if the existing lab control stack is already functional.

The project is intended to:

1. Build hands-on familiarity with PPO-based robot control.
2. Demonstrate understanding of:
   - observation/action design,
   - reward shaping,
   - policy training,
   - domain randomization,
   - sim-to-real considerations,
   - comparison between learning-based and model-based control.
3. Produce a small but technically defensible project that can be discussed in a robotics motion-control interview.
4. Avoid unnecessary complexity such as grasping, vision, contact-rich manipulation, or custom robot modeling.

The target completion time is approximately **5–6 development days**, leaving one day for interview review.

---

# 2. Core Research Question

The project should answer:

> Can a PPO policy learn robust end-effector reaching/tracking for a Franka manipulator using joint-position residual actions, and how does domain randomization affect robustness to model mismatch compared with both a nominal policy and a classical Jacobian controller?

The project is **not** intended to demonstrate that RL outperforms classical control.

A valid and useful outcome is:

> Classical resolved-rate control performs better under nominal conditions, while the RL project demonstrates a complete policy-training, robustness-testing, and potentially sim-to-real deployment pipeline.

That conclusion is preferable to artificially designing the experiment to make RL appear superior.

---

# 3. Scope

## In Scope

### Simulation
- Isaac Lab
- Franka Panda
- PPO
- Cartesian end-effector reaching
- Optionally Cartesian pose tracking
- Joint-position or joint-position-residual action space
- Low-level joint PD/impedance controller
- Parallel environment training
- Classical Jacobian controller baseline
- Domain randomization
- Robustness evaluation
- Training/evaluation logging

### Optional Real-Robot Deployment
Only attempt this if the lab already has:
- a working Franka bring-up,
- tested joint position/impedance control,
- functioning safety limits,
- emergency stop,
- an existing communication interface.

Real deployment should be limited to low-speed reaching inside a conservative workspace.

## Explicitly Out of Scope

Do **not** spend time on:
- grasping,
- object manipulation,
- camera observations,
- image-based RL,
- contact-rich manipulation,
- SAC unless PPO is already fully working,
- custom CUDA kernels,
- custom Franka URDF modeling if Isaac Lab already provides one,
- replacing the Franka low-level controller,
- HoloMotion integration,
- imitation learning,
- diffusion policies,
- multi-task RL,
- large-scale hyperparameter sweeps.

---

# 4. Proposed Task

## Task: Random Cartesian Reaching

At the beginning of each episode, sample a reachable target:

\[
p_d \in \mathcal{W}
\]

where \(\mathcal{W}\) is a conservative Cartesian workspace around a nominal Franka configuration.

The robot must drive its end effector to:

\[
p_{ee} \rightarrow p_d.
\]

### Initial Version

Track position only:

\[
e_p = p_d - p_{ee}.
\]

### Stretch Version

Track full pose:

\[
T_{ee} \rightarrow T_d
\]

using orientation error represented with a rotation vector or another appropriate local SO(3) representation.

Do not begin with full-pose tracking.

---

# 5. Environment Definition

Each environment should expose the standard RL structure:

\[
(o_t, a_t, r_t, d_t).
\]

## Observation

Initial observation:

\[
o_t =
[
q_t,\;
\dot q_t,\;
p_d-p_{ee},\;
a_{t-1}
].
\]

Where:

- \(q_t \in \mathbb{R}^7\): joint positions
- \(\dot q_t \in \mathbb{R}^7\): joint velocities
- \(p_d-p_{ee} \in \mathbb{R}^3\): Cartesian position error
- \(a_{t-1}\): previous policy action

Normalize observations where appropriate.

### Optional Additions

Only add these after the baseline trains successfully:

- orientation error,
- target relative to robot base,
- end-effector velocity.

Avoid giving the policy unnecessary absolute world coordinates.

---

# 6. Action Space

Preferred action:

\[
a_t \in [-1,1]^7
\]

mapped to joint-position residuals:

\[
q_d =
q_{\text{nominal}}
+
s_a a_t
\]

or

\[
q_d(t)
=
q(t)+s_a a_t.
\]

The low-level controller computes approximately:

\[
\tau =
K_p(q_d-q)
-
K_d\dot q.
\]

This architecture is preferred over direct torque actions because:

1. it is safer for eventual real deployment,
2. it leverages the existing low-level impedance controller,
3. it provides local stabilization,
4. it is analogous to common locomotion RL architectures where the policy outputs desired joint positions.

### Action Limits

Choose conservative residual limits initially.

For example, policy actions should not be able to request large instantaneous joint changes.

Apply:
- joint position limits,
- joint velocity limits,
- action clipping.

---

# 7. Reward Function

Start with the minimum reward necessary.

## Position Tracking

\[
r_p
=
\exp
\left(
-\frac{\|p_d-p_{ee}\|^2}{\sigma_p}
\right).
\]

## Action Magnitude Penalty

\[
r_a
=
-\lambda_a\|a_t\|^2.
\]

## Action-Rate Penalty

\[
r_{\Delta a}
=
-\lambda_{\Delta a}
\|a_t-a_{t-1}\|^2.
\]

Total:

\[
r =
w_p r_p
+
r_a
+
r_{\Delta a}.
\]

### Optional Penalties

Add only if necessary:

\[
r_{\dot q}
=
-\lambda_v\|\dot q\|^2
\]

and joint-limit penalties.

Do not begin with a large reward function containing many terms.

The project should explicitly document why each reward term was introduced.

---

# 8. Episode Definition

Episode begins with:

- robot near nominal configuration,
- randomized reachable target.

Episode ends when:

### Success

\[
\|p_{ee}-p_d\| < \epsilon
\]

for a specified number of consecutive policy steps.

### Failure

- joint limit violation,
- unsafe Cartesian workspace violation,
- excessive velocity,
- simulation instability.

### Timeout

Maximum episode horizon exceeded.

Record:
- success rate,
- time-to-target,
- final error.

---

# 9. PPO Training

Use the standard PPO implementation already supported by Isaac Lab where possible.

Do not implement PPO from scratch unless needed for learning purposes.

The project owner should nevertheless understand:

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\]

and

\[
L_{\text{clip}}
=
\mathbb{E}
\left[
\min
\left(
r_t A_t,\;
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

Also understand:
- critic/value loss,
- entropy bonus,
- GAE,
- rollout length,
- minibatches,
- number of epochs,
- policy update frequency.

Avoid hyperparameter tuning until the basic pipeline works.

---

# 10. Classical Baseline

Implement a Jacobian resolved-rate controller.

Given:

\[
e = p_d-p_{ee},
\]

command Cartesian velocity:

\[
\dot x_d = K_p e.
\]

Then:

\[
\dot q
=
J^\dagger \dot x_d.
\]

Prefer damped least squares:

\[
J^\# =
J^T
(JJ^T+\lambda^2 I)^{-1}.
\]

Integrate to generate joint targets:

\[
q_d(k+1)
=
q_d(k)+\dot q\Delta t.
\]

Use the same:
- target distribution,
- joint controller,
- safety limits,
- evaluation metrics

as the PPO policy.

The goal is a fair comparison, not to force PPO to win.

---

# 11. Domain Randomization

Train two PPO policies:

## Policy A — Nominal

Uses nominal simulation parameters.

## Policy B — Domain Randomized

Randomize a modest set of physically meaningful parameters.

Suggested randomization:

### Robot Parameters
- joint damping,
- joint friction,
- actuator strength/gain,
- optionally link mass/inertia within a small range.

### Observation
- joint-position noise,
- joint-velocity noise.

### Control
- action delay,
- small control gain variation.

Avoid excessively wide randomization ranges.

The intended lesson is:

\[
\text{plausible uncertainty}
\neq
\text{arbitrary random physics}.
\]

---

# 12. Robustness Evaluation

Evaluate all controllers under conditions outside nominal training:

1. nominal model,
2. reduced actuator strength,
3. increased damping,
4. modified friction,
5. observation noise,
6. 1-step action delay,
7. 2-step action delay.

Compare:

- Jacobian baseline,
- nominal PPO,
- randomized PPO.

Suggested metrics:

\[
\text{Success Rate}
\]

\[
E_p =
\frac{1}{T}
\sum_t
\|p_{ee}(t)-p_d\|
\]

\[
T_{\text{reach}}
\]

\[
J_{\Delta a}
=
\frac1T
\sum_t
\|a_t-a_{t-1}\|^2.
\]

Also record:
- max joint velocity,
- max joint acceleration if available.

---

# 13. Required Ablation

Perform at least one simple ablation.

Preferred ablation:

### Action-Rate Penalty

Train or evaluate:

\[
\lambda_{\Delta a}=0
\]

versus

\[
\lambda_{\Delta a}>0.
\]

Compare:
- trajectory smoothness,
- reaching time,
- action variation.

The expected lesson is the tradeoff:

\[
\text{smoothness}
\leftrightarrow
\text{responsiveness}.
\]

This directly connects to general motion-control design.

---

# 14. Optional Real Franka Deployment

## Gate Condition

Do not begin real deployment until:

- simulation policy is stable,
- the classical baseline works,
- action limits are verified,
- the existing Franka interface is understood.

## Deployment Architecture

Preferred:

```text
Franka state
    |
    v
Observation builder
    |
    v
PPO inference
    |
    v
Action scaling / safety clamp
    |
    v
Desired joint position
    |
    v
Existing Franka impedance / position controller
    |
    v
Robot