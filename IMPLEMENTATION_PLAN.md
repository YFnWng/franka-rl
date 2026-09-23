# Franka RL Demo — Implementation Plan

## 1. Outcome and design choices

Build this as a small **external Isaac Lab project** using the **manager-based single-agent workflow**, PhysX, and RSL-RL PPO. The implementation should be an adaptation of the installed Isaac Lab Franka reach environment, not a fork or edit of `IsaacLab/source/isaaclab_tasks`.

The minimum defensible result is:

1. a position-only Franka reaching environment;
2. a nominal PPO policy;
3. a modestly domain-randomized PPO policy;
4. a batched damped-least-squares Jacobian baseline that uses the same environment action path;
5. a fixed robustness matrix and an action-rate-penalty ablation;
6. reproducible metrics, plots, configuration snapshots, and short videos.

The first implementation will use these decisions:

- **End effector:** `panda_hand`.
- **Controlled joints:** `panda_joint1` through `panda_joint7`; the fingers stay at their asset defaults.
- **Action semantics:** a normalized 7-vector maps to a target centered on the robot's default pose,
  `q_des = q_default + action_scale * clip(action, -1, 1)`. This matches Isaac Lab's existing Franka reach action configuration (`use_default_offset=True`) and avoids the drift and ambiguous Markov state introduced by accumulating `q(t) + delta_q` targets.
- **Action scale:** start at `0.25 rad`; raise only if workspace coverage tests show it is necessary. The upstream example uses `0.5 rad`, which is a useful ceiling rather than the initial setting.
- **Frames:** targets and Cartesian errors are expressed in the Franka root frame. Never use cloned-environment world coordinates as policy input.
- **Observation:** exactly 24 values in a fixed order: 7 relative joint positions, 7 joint velocities, 3 target-minus-hand position error, and 7 previous actions.
- **Control rate:** 60 Hz physics with decimation 2, hence a 30 Hz policy/controller step, matching the checked-out reach task's starting point.
- **Target lifetime:** one target per episode. Do not inherit the upstream task's four-second target resampling.
- **Episode horizon:** 6 seconds initially. Success requires error below `0.03 m` for 5 consecutive policy steps. Timeout remains separate from failure.
- **Reward:** only exponential position tracking, action magnitude, and action rate at first. Do not copy the upstream orientation term or curriculum.
- **PPO:** RSL-RL with the upstream Franka reach PPO configuration as the initial hyperparameter baseline, changing only experiment name and observation normalization if validation shows it is needed.
- **Randomization:** separate nominal and randomized environment configurations; no hidden randomization in the nominal task.
- **Evaluation:** deterministic target sets and reset seeds shared by all three controllers. Robustness conditions are fixed test configurations, not random mixtures.
- **Real robot:** no implementation in the core five-day plan. Export and a hardware adapter interface are allowed only after the simulation gates pass.

## 2. Upstream references and compatibility target

The local compatibility target is the current checkout:

- Isaac Lab commit `ffff603eafc6b74264a5261cc0183d6a65390d78`
- tag/description `v3.0.0-beta2.patch1`
- Python environment supplied by `/home/chen-lab/isaac/.venv` through `franka-rl/.envrc`

Use these local files as the primary implementation references:

- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py`
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py`
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py`
- `IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py`
- `IsaacLab/tools/template/templates/tasks/manager-based_single-agent/`
- `IsaacLab/tools/template/templates/extension/`

Relevant official online references:

- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments) confirms that `Isaac-Reach-Franka-v0` is the upstream reference task.
- [RL scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html) gives the supported RSL-RL train/play entry points.
- [External project template](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/developer-guide/template.html) recommends keeping a custom project outside the Isaac Lab checkout.
- [Differential IK tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html) shows batched Jacobian retrieval, base-frame transforms, and joint-target application.
- [Differential IK API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.controllers.html) documents damped least squares and the other supported inverse methods.
- [Domain-randomization migration guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_omniisaacgymenvs.html#domain-randomization) documents event-based gain/material randomization and observation/action noise.
- [DelayBuffer](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/utils/buffers/delay_buffer.html) documents batched, per-environment discrete delays.

Pinning the checkout matters because Isaac Lab 3.x changes actuator ownership and parameter-writing APIs. Do not implement from an older blog post without checking it against this checkout.

## 3. Proposed repository layout

```text
franka-rl/
├── .envrc
├── .gitignore
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml                       # workspace tooling: pytest, lint config
├── franka-rl-scope.md                   # original handoff; retain as source requirements
├── hello_franka.py                      # keep as a standalone Isaac Sim smoke example
├── configs/
│   ├── evaluation/
│   │   ├── targets.yaml                 # fixed target box/grid and seed list
│   │   └── robustness.yaml              # seven named perturbation scenarios
│   └── experiments/
│       ├── nominal.yaml                 # run name, seed set, iteration budget
│       ├── randomized.yaml
│       └── no_action_rate.yaml
├── docs/
│   ├── ARCHITECTURE.md                  # frames, tensor shapes, timing, action semantics
│   ├── EXPERIMENTS.md                   # hypotheses and immutable comparison protocol
│   ├── REWARD_DESIGN.md                 # each term, units, weight, and motivation
│   ├── SIM_TO_REAL.md                   # risk register and deployment gate only
│   └── RESULTS.md                       # generated-summary template; no binary artifacts
├── scripts/
│   ├── _common.sh                       # resolve repo/Isaac paths and validate environment
│   ├── bootstrap.sh                     # editable install only; no global installs
│   ├── check_environment.py             # versions, CUDA, task registration, data volume
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   ├── train.sh                         # thin upstream RSL-RL CLI wrapper
│   ├── play.sh                          # thin upstream play/export wrapper
│   ├── evaluate.py                      # PPO and Jacobian evaluation driver
│   ├── run_matrix.py                    # scenario/controller/seed orchestration
│   └── plot_results.py                  # reads tables and emits aggregate figures
├── source/
│   └── franka_rl/
│       ├── pyproject.toml
│       ├── setup.py
│       ├── config/
│       │   └── extension.toml
│       └── franka_rl/
│           ├── __init__.py              # imports tasks, causing Gym registration
│           ├── cli.py                   # external-callback registration hook
│           ├── controllers/
│           │   ├── __init__.py
│           │   └── jacobian_dls.py      # batched 3x7 DLS resolved-rate controller
│           ├── evaluation/
│           │   ├── __init__.py
│           │   ├── metrics.py           # GPU episode accumulators and reductions
│           │   ├── scenarios.py         # typed evaluation overrides
│           │   └── writers.py           # batched end-of-run CSV/JSON output
│           ├── tasks/
│           │   ├── __init__.py
│           │   └── reach/
│           │       ├── __init__.py      # four Gym registrations
│           │       ├── reach_env_cfg.py # shared, nominal, DR, and play configs
│           │       ├── agents/
│           │       │   ├── __init__.py
│           │       │   └── rsl_rl_ppo_cfg.py
│           │       └── mdp/
│           │           ├── __init__.py  # re-export common Isaac Lab MDP terms
│           │           ├── actions.py   # policy-step delayed joint-position action
│           │           ├── actions_cfg.py
│           │           ├── events.py    # any custom strength/friction fixed overrides
│           │           ├── observations.py
│           │           ├── rewards.py
│           │           └── terminations.py
│           └── utils/
│               ├── artifacts.py         # validated FRANKA_RL_DATA_ROOT paths
│               └── reproducibility.py   # version/revision/seed manifest
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_action_mapping.py
    │   ├── test_jacobian_dls.py
    │   ├── test_metrics.py
    │   └── test_reward_math.py
    └── integration/
        ├── test_task_registration.py
        ├── test_env_shapes.py
        ├── test_selective_reset.py
        ├── test_action_delay.py
        └── test_jacobian_reaches.py
```

Do not copy Isaac Lab's full training scripts into this repository. The shell wrappers should call the checked-out `isaaclab.sh train/play` commands with `--rl_library rsl_rl` and an external registration callback. This keeps task code local while avoiding a stale fork of runner code.

## 4. Environment contract

### 4.1 Registered task IDs

Register only four public IDs:

| ID | Purpose |
|---|---|
| `FrankaRL-Reach-v0` | nominal training/evaluation configuration |
| `FrankaRL-Reach-DR-v0` | modest domain-randomized training configuration |
| `FrankaRL-Reach-Play-v0` | small visible nominal scene, observation corruption off |
| `FrankaRL-Reach-DR-Play-v0` | small visible DR-policy playback, randomization disabled by default |

The action-rate ablation should be a Hydra/config override of the nominal task, not another permanent Gym ID. Robustness scenarios should also be evaluation overrides rather than seven additional task registrations.

### 4.2 Scene

Use `FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`, a ground plane, a simple table or pedestal, dome light, and a target marker. Avoid objects, cameras, contact sensors, and custom USD assets. Start with 4096 environments and lower via CLI when memory or throughput requires it.

Resolve joint/body IDs once during manager initialization with `SceneEntityCfg`; never look them up or loop over environments inside a step.

### 4.3 Command and workspace

Reuse `UniformPoseCommandCfg` initially because it already provides batched sampling and debug visualization, but fix orientation and expose only the three-dimensional position error to the policy and reward. Configure it to resample only at reset.

Start with the conservative root-frame box:

```text
x: [0.35, 0.60] m
y: [-0.20, 0.20] m
z: [0.20, 0.50] m
```

Before training, sample at least 10,000 targets and verify reachability using DLS from randomized reset poses. Shrink the box if the classical controller cannot reach at least 99% without violating safety limits. Do not let PPO compensate for unreachable commands.

### 4.4 Observation

The policy group concatenates, in this exact order:

```text
joint_pos_rel[7] | joint_vel[7] | ee_position_error_b[3] | last_arm_action[7]
```

Acceptance checks:

- tensor shape is `(num_envs, 24)`;
- dtype/device match the environment;
- position error is invariant to environment clone origin;
- gripper joints/actions are absent;
- `last_arm_action` resets to zero for only the reset environments;
- nominal config has no observation corruption;
- DR config uses small, separately specified noise for joint position and velocity.

Use physically interpretable scaling/normalization. Either apply explicit term scales in the observation configuration or enable RSL-RL's running normalization, but do not silently use both.

### 4.5 Action and delay

Implement a small custom action term derived from the standard joint-position action term:

1. clip normalized action to `[-1, 1]`;
2. push it through a batched `DelayBuffer` once per **policy step**;
3. map the delayed value to `q_default + scale * action`;
4. clamp to soft joint-position limits with a small margin;
5. send the target through the normal Isaac Lab actuator/PD path.

With zero configured lag, its output must be numerically identical to `JointPositionActionCfg` for the same input. Delay state must reset selectively. The randomized training task may sample a lag of 0–1 steps per environment; evaluation uses exact lags 0, 1, or 2.

### 4.6 Reward

Implement and log the three terms separately:

```text
position_tracking = exp(-squared_distance / sigma_p)
action_l2         = sum(action^2)
action_rate_l2    = sum((action - previous_action)^2)
```

Initial parameters:

```text
sigma_p:             0.01 m^2
position weight:     +1.0
action weight:       -1.0e-4
action-rate weight:  -1.0e-3
```

These are starting values, not claims of optimality. First verify signs, ranges, reset behavior, and learning signal. Add velocity or joint-limit penalties only when a recorded failure mode justifies them. The ablation changes only the action-rate weight to zero.

### 4.7 Termination and success

Implement GPU-batched termination terms for:

- timeout;
- sustained success: position error below `0.03 m` for 5 consecutive policy steps;
- any controlled joint outside the configured safety margin;
- excessive joint velocity;
- end effector outside a broad safety workspace or non-finite simulation state.

The sustained-success counter must reset to zero whenever an environment leaves the threshold and reset only the requested environment IDs at episode reset. Report timeout separately from unsafe failure.

## 5. Jacobian baseline

`jacobian_dls.py` should implement a batched, position-only controller rather than wrapping the full-pose policy action:

```text
v_des = clamp(Kp * (p_target - p_ee), max_cartesian_speed)
q_dot = J_pos^T (J_pos J_pos^T + damping^2 I_3)^-1 v_des
q_target = clamp(q + q_dot * control_dt, soft_joint_limits)
normalized_action = clamp((q_target - q_default) / action_scale, -1, 1)
```

Feed `normalized_action` back through the environment's action term. This guarantees that PPO and DLS share action scaling, action delay, joint clamps, actuator gains, PD control, physics, resets, and metrics.

Add optional null-space centering only after the plain DLS baseline passes. Keep `Kp`, damping, maximum Cartesian speed, and any null-space gain in the evaluation manifest. Unit-test against `torch.linalg.pinv` away from singularities and explicitly test finite output near a singular Jacobian.

## 6. Domain randomization and robustness scenarios

Keep ranges modest and measurable. A reasonable first DR configuration is:

| Quantity | Nominal training | DR training proposal | Test points |
|---|---:|---:|---:|
| actuator stiffness | 1.0x | uniform 0.9–1.1x | 0.8x |
| actuator damping | 1.0x | uniform 0.9–1.1x | 1.25x |
| joint friction | asset default | small additive/scale range validated from API | elevated fixed value |
| link mass | 1.0x | uniform 0.95–1.05x | optional 1.1x |
| joint-position noise | 0 | ±0.002 rad | ±0.005 rad |
| joint-velocity noise | 0 | ±0.01 rad/s | ±0.03 rad/s |
| action delay | 0 steps | integer 0–1 | exactly 1 and 2 |

Use Isaac Lab event terms such as `randomize_actuator_gains`, `randomize_joint_parameters`, and `randomize_rigid_body_mass` where supported by the checked-out PhysX backend. Treat reduced actuator **strength** as an effort-limit perturbation, not as a synonym for lower stiffness. If runtime effort-limit mutation is not reliable in this checkout, create the fixed evaluation scenario by replacing the Franka actuator configuration before scene creation.

The seven required robustness rows are:

1. nominal;
2. 0.8x effort limit;
3. 1.25x damping;
4. elevated joint friction;
5. observation noise;
6. one policy-step action delay;
7. two policy-step action delay.

Only the randomized PPO is trained on a mixture. All controllers are evaluated on the same fixed rows.

## 7. Evaluation and artifacts

### 7.1 Protocol

Use at least 5 seeds and 200 episodes per controller/scenario pair for final results, with a smaller smoke matrix during development. Store the actual target tensor or target-generation seed and replay it across controllers. Evaluation must disable exploratory action noise.

For each episode record:

- controller, checkpoint, scenario, seed, and episode ID;
- target position and reset joint position;
- success, timeout, and unsafe-failure flags;
- first sustained-success time;
- final and mean position error;
- mean squared action rate;
- maximum absolute joint velocity;
- maximum absolute finite-difference joint acceleration;
- return and episode length.

Write one tidy CSV or Parquet table per completed evaluation run and one JSON manifest. Aggregate plots should show confidence intervals or seed dispersion, not only a single mean.

### 7.2 Artifact paths

All checkpoints, runs, videos, and evaluation tables go under:

```text
${FRANKA_RL_DATA_ROOT:-/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data}/
├── checkpoints/
├── runs/
├── evaluations/
├── videos/
└── benchmarks/
```

Before a long run, `check_environment.py` must verify that the volume exists, is mounted, is writable, and has adequate free space. It must fail clearly rather than fall back to the repository or home directory. Buffer per-step metrics in memory and write at episode/run boundaries.

Each run manifest records:

- Isaac Lab and project Git revisions, dirty flags, and package versions;
- task ID and fully resolved environment/agent configuration;
- seeds, target split, scenario name, and controller parameters;
- GPU, driver, CUDA, Isaac Sim, PyTorch, and RSL-RL versions;
- checkpoint path and hash.

## 8. Verification gates

Do not advance merely because a script launches.

### Gate A — installation and registration

- editable install succeeds in `/home/chen-lab/isaac/.venv`;
- all four task IDs appear exactly once;
- zero/random agents launch with 1 and 64 environments;
- no source file under `IsaacLab/` is modified.

### Gate B — MDP correctness

- observation and action shapes are `(N, 24)` and `(N, 7)`;
- frame-invariance test passes across cloned origins;
- reward is maximal at zero error and decreases monotonically;
- partial resets change only selected environments;
- 0/1/2-step delay tests match exact expected action histories;
- all tensors stay on the simulation device in the hot path;
- no per-environment Python loop, `.item()`, CPU conversion, allocation-heavy logging, or file I/O occurs in `step()`.

### Gate C — classical controller

- reaches at least 99% of the chosen target set in nominal simulation;
- no NaNs near the tested singular configurations;
- obeys action, joint, velocity, and workspace limits;
- produces the exact same metrics schema as learned policies.

### Gate D — learning smoke test

- 256–512 environments, fixed seed, short run;
- return improves and median final error falls below the random-policy result;
- visual playback confirms target-directed motion rather than limit exploitation;
- resume and policy export work.

### Gate E — nominal experiment

- train at least 3 nominal seeds;
- choose checkpoints using a declared rule, not the robustness test result;
- freeze configs and target split before DR comparison.

### Gate F — randomized experiment and ablation

- train at least 3 DR seeds with the same PPO budget;
- run the complete controller × scenario matrix;
- train/evaluate `action_rate.weight=0` with all other settings fixed;
- generate summary tables, learning curves, robustness plots, and representative videos.

## 9. Five-to-six-day execution order

### Day 1 — scaffold and validate semantics

- Create the external package and editable install.
- Register nominal/play tasks.
- Implement the 24-D observation, 7-D action, position reward, basic reset, and timeout.
- Run unit tests plus zero/random-agent checks.
- Write `ARCHITECTURE.md` while frame and timing decisions are fresh.

**Exit:** Gate A and the shape/frame portions of Gate B pass.

### Day 2 — safety, success, and classical baseline

- Add selective-reset-safe success/failure terms.
- Implement policy-step delay and its tests.
- Implement batched position-only DLS and workspace reachability sweep.
- Finalize the conservative target box.

**Exit:** Gates B and C pass.

### Day 3 — PPO nominal

- Add RSL-RL config and artifact manifests.
- Run a short smoke train, inspect playback, then start nominal seeds.
- Make only evidence-driven reward/action-scale changes and record them.

**Exit:** Gate D passes and the nominal configuration is frozen.

### Day 4 — domain randomization

- Implement and verify each randomization independently.
- Register the DR task and start matched-seed DR training.
- Build fixed evaluation scenario overrides.

**Exit:** parameter introspection confirms sampled values and no nominal leakage.

### Day 5 — evaluation and ablation

- Run the full robustness matrix for DLS, nominal PPO, and DR PPO.
- Run the no-action-rate experiment.
- Generate result tables, plots, and short videos.

**Exit:** Gate F data is complete or gaps are explicitly identified.

### Day 6 — interpretation and interview package

- Write `RESULTS.md` around evidence, including where DLS wins.
- Summarize PPO, GAE, action design, reward tradeoffs, DR ranges, failure cases, and sim-to-real risks.
- Verify every documented command from a clean shell.
- Only then decide whether real-arm work is justified.

## 10. Explicit non-goals and deferrals

- Do not turn `hello_franka.py` into the RL environment; it uses an Isaac Sim experimental manipulator example and is useful only as a simulator sanity check.
- Do not add pose tracking until position reaching and the comparison matrix are complete.
- Do not add grasping, cameras, custom Franka assets, SAC, imitation learning, or a custom PPO implementation.
- Do not add a UI extension; the Python external package is sufficient.
- Do not promise real-robot deployment. The optional adapter must be a separate package/module with an explicit safety review, rate limits, watchdog, and existing lab controller interface.
- Do not commit checkpoints, videos, generated evaluation tables, caches, or large logs.

## 11. First implementation slice

The first coding slice should be deliberately narrow:

1. scaffold `source/franka_rl` from the installed external manager-based template;
2. register `FrankaRL-Reach-v0` and `FrankaRL-Reach-Play-v0`;
3. adapt `FRANKA_PANDA_CFG`, `UniformPoseCommandCfg`, and `JointPositionActionCfg` from the upstream Franka reach task;
4. remove orientation from observation/reward and restrict joint observations to the seven arm joints;
5. add the root-frame 3-D position-error observation and exponential reward;
6. verify one target per episode and exact tensor shapes;
7. run zero/random agents and the focused integration tests;
8. implement DLS before starting PPO training.

This slice proves the environment contract and comparison path before domain randomization, experiment orchestration, or presentation work expands the surface area.
