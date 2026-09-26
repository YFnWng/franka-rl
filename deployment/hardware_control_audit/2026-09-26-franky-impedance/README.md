# Franky impedance controller calibration handoff

The RT host completed seven-joint response tests for the uniform K=50, K=100,
and K=200 controller family with D=2*sqrt(K). Every session completed without
robot errors, invalid commands, dropped samples, or torque-slew activation.
`comparison.json` is the portable training input. K=100 is nominal; K=50 and
K=200 are 0.5x/2x gain-DR endpoints.

These measurements define and validate the controller cases but do not make the
current Isaac task deployment-faithful. Read
`deployment/DEPLOYMENT_TRAINING_READINESS.md` before training. Raw 1 kHz traces
remain in the RT-host hardware inventory.
