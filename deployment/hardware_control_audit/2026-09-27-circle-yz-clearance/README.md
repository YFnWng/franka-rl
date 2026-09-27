# `circle_yz` physical-clearance record

The operator confirmed that the exact checked-in `circle_yz` geometry is
physically clear. The reviewed envelope is x=0.475 m, y `[-0.15,0.15]` m, and z
`[0.20,0.50]` m, with no attached end effector. The source catalog and its hash
are recorded in `clearance.json`.

This closes the physical-clearance gate for this exact geometry. Any path or
catalog change requires another review. This record does not select a policy or
authorize robot motion.

The operator subsequently reviewed the updated catalog whose only relevant
change is a 2.0 s timeout for waypoint 0; later waypoints remain at 1.0 s. The
geometry is unchanged. The active catalog SHA-256 is
`39455c82dfec508cdac0d26befda5a701501006c3758d1c115adc868a1ff6c69`, and the
operator re-acknowledged this timing/hash while requesting preparation of the
hardware run.
