# Offline hardware-control return package

**Later live update:** [LIVE_INSPECTION.md](LIVE_INSPECTION.md) records successful
read-only FCI access, robot-returned fr3v2.1 mass/COM differences and the decision
to preserve default internal settings. It supersedes the earlier statements
below that no live model had been captured. Offline-test scope remains unchanged.

Start with [FINDINGS.md](FINDINGS.md). [control_contract.json](control_contract.json)
is versioned audit data with explicit unknowns; it is not deployable configuration.
[evidence.json](evidence.json) records installed versions, source hashes, source
links and binary dependency resolution. [limits.json](limits.json) separates
manufacturer/library data from unresolved lab limits. Existing robot/model
inventory is referenced in `../../model_audit/2026-09-24/`, not duplicated.

No hardware connection or Isaac installation is needed to check the vectors.
The C++ oracle creates only JointVelocityLimitsConfig, never a Robot object.
Its optional helper outputs are not an enabled or complete deployment governor.
The Python reference has equivalent scalar limiter math for simulation reuse.

From this directory, with Python 3.10+ and NumPy installed:

```bash
python -I -B verify_vectors.py
sha256sum -c SHA256SUMS
```

Verified here with CPython 3.12.14 and NumPy 2.5.3:
26 observation/action cases (zero, 24 sentinels, existing state snapshot) match
C++ exactly in float32; six limiter/filter cases agree within 1e-12 absolute.
No new policy inference, closed-loop simulation, temporal governor, state-age
watchdog or hardware controller was tested. Synthetic targets in inputs.json
include out-of-limit requests intentionally; they are not motion commands.

To regenerate the native oracle output, install/build libfranka 0.19.0 and
nlohmann-json headers, then adapt only the library prefix below:

```bash
LIBFRANKA_PREFIX=/home/chen-lab/local/libfranka-0.19.0
g++ -std=c++17 -O2 -ffp-contract=off \
  -I"$LIBFRANKA_PREFIX/include" offline_reference.cpp \
  -L"$LIBFRANKA_PREFIX/lib" -Wl,-rpath,"$LIBFRANKA_PREFIX/lib" \
  -lfranka -o /tmp/fr3_control_audit_reference
/tmp/fr3_control_audit_reference \
  ../../model_audit/2026-09-24/fr3v2_no_hand_fake.urdf inputs.json \
  > /tmp/fr3_control_audit_vectors.json
cmp vectors.json /tmp/fr3_control_audit_vectors.json
```

Deprecation warnings for the old velocity helpers are expected: those helpers
are deliberately included because the installed ROS wrapper uses them if its
optional limiter is enabled. Compiler/platform floating-point differences may
require numerical comparison rather than byte comparison; the committed vectors
are the output from this host's local library. No downloaded USD or large traces
are included. Absolute paths in evidence.json are provenance on the real-time
host; the fixed upstream links identify versioned source where available.
