# Explicit velocity-reference impedance route

This route is a separate deployment contract for policies trained to emit joint
velocity. It does not adapt an incremental-position policy at runtime.

At policy tick `k`, the observation contains the reference that was applied over
the preceding control interval, `q_ref[k]`, and the preceding normalized velocity
action. The actor then produces `a[k]`. The runtime maps it to

```text
dq_ref = a * dq_max
```

and publishes `dq_ref` to a persistent C++ Franky motion. In the 1 kHz FCI loop,
using libfranka's measured time step `dt`, that motion evaluates

```text
q_ref <- project_soft_limits(q_ref + dq_ref * dt)
tau = K (q_ref - q) + D (dq_ref - dq) + c(q, dq) + tau_limit
```

followed by Franky's configured torque-rate limit. An outward velocity is set to
zero when the integrated reference reaches a soft bound. Joint 7 always receives
zero velocity. The 50 Hz Python process never performs the integration and no
Python callback runs in the real-time reference path.

`q_ref` is therefore controller state, rather than a hypothetical projection.
The backend reads the last reference actually evaluated by the C++ controller for
the next observation and for logging. At reset it is seeded from the same measured
joint sample used by preflight.

The policy contract is intentionally distinct:

- runtime: `franky_joint_velocity_impedance_tracking_v1`
- contract: `fr3_joint_velocity_impedance_29d_v1`
- action: `normalized_joint_velocity`, six joints at 50 Hz
- integration: `controller_1khz_forward_euler`
- observation tail: current normalized `q_ref`, then previous normalized velocity
- torque controller: the same explicit K/D, Coriolis compensation, soft-limit
  torque, error clip, and torque slew fields used by the position route

Existing `fr3_incremental_position_29d_v1` bundles are rejected. A policy must be
trained and exported with this actuator and observation contract before hardware
execution.

## Franky extension

The exact-source patch is `franky_velocity_impedance.patch`. It applies to Franky
commit `f88f0e9b`, the source commit in the currently pinned
`2.0.1.dev58+gf88f0e9b.libfranka.0.21.2` wheel. It adds
`JointVelocityImpedanceTrackingMotion`, its Python binding, wait-free velocity
publication, 1 kHz integration, applied-reference publication, and soft-bound
projection.

Apply and compile it in a clean checkout:

```bash
git clone https://github.com/TimSchneider42/franky.git ~/yifan/franky-velocity-impedance
git -C ~/yifan/franky-velocity-impedance checkout f88f0e9b
git -C ~/yifan/franky-velocity-impedance apply \
  ~/yifan/franka-rl/deployment/franky_runtime/franky_velocity_impedance.patch
```

Build against libfranka 0.21.2 and Ruckig 0.17.3. Package it with `auditwheel`, as
Franky's official wheel build does, so libfranka and Ruckig remain private to the
wheel. Install the repaired wheel into a new virtual environment; do not replace
the validated `franky-server10` environment. Before connecting to FCI, verify:

```bash
python - <<'PY'
import franky
assert hasattr(franky, "JointVelocityImpedanceTrackingMotion")
print(franky.__version__)
PY
```

The complete C++ core and Python module were built against libfranka 0.21.2,
Ruckig 0.17.3, pybind11 3.0.4, and Python 3.12. An offline binding smoke test
constructed the new motion and exercised its initial-reference, velocity, and
applied-reference API without opening FCI. Hardware motion remains pending a
velocity-policy bundle, a separate environment, and a fake/shadow review.
