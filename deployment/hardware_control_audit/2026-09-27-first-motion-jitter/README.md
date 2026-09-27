# First-motion jitter audit

Session 2026092705 completed all 24 circle_yz waypoints with no timeouts, no
sample loss, no robot errors, and a clean stop. The operator observed heavy
jitter during the beginning.

The trace confirms that the transient is concentrated in the 1.48 s approach
from home to waypoint 0. J5/J6 reached 0.97/1.07 rad/s there, versus 0.21/0.19
rad/s after waypoint 0. Their 99th-percentile initial acceleration was about
45-48 rad/s2 at K=200, compared with 7-8 rad/s2 in the recorded K=100 approach.

The runtime applies the 50 Hz incremental policy output as a held joint-position
reference and supplies desired joint velocity zero. K=200 therefore reacts
sharply to the reference staircase. The policy also emits large increments
while closing the initial 0.19 m Cartesian gap. The later circle is much calmer.

For the next presentation run, use the policy's K=100/D=20 training nominal.
Keep J1/J3/J4 at 0.08 rad and J7 at 0.04 rad. Raise only J2/J5/J6 to 0.12 rad,
above their approximately 0.082 rad peaks in the earlier K=100 trace. This
changes the fault gate rather than policy output or controller behavior.

A longer-term improvement should be trained and evaluated with an action-rate
penalty or a deployment-identical interpolated reference. Adding an untrained
filter only on hardware would change the policy's action dynamics.

## Completed K=100 comparison

Session 2026092706 also completed 24/24 waypoints without robot errors, sample
loss, timeout, or stop failure. At waypoint 0, K=100 reduced J5/J6 peak velocity
from 0.967/1.070 to 0.436/0.496 rad/s and their 99th-percentile acceleration
from 60.9/63.6 to 7.09/7.26 rad/s2. Tracking error increased from 0.072/0.078
to 0.095/0.108 rad. This confirms a command-discontinuity tradeoff rather than
a path-planning failure.

The next-generation design is documented in
deployment/CONTINUOUS_REFERENCE_REDESIGN.md.

