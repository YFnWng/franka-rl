# Generated deployment releases

Artifact root: `/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/deployment_bundles`

Transfer the entire selected directory. Use the separately recorded manifest hashes below.

## dr_v2_model_999_release_1

Manifest SHA-256: `80c697b2b07b50b7ab5cdbcae86c3e4e1a68e965102d348f37ca4094a677952f`

Native RSL-RL / ONNX maximum absolute error: `3.814697265625e-06`; 1,024 cases.

## nominal_model_299_release_1

Manifest SHA-256: `69829e1878f7a19ae9f477e53861ef4b831dd39de72a5c693559095af28798e4`

Native RSL-RL / ONNX maximum absolute error: `4.76837158203125e-06`; 1,024 cases.

The portable verifier passed in a fresh process without importing Isaac, ROS, PyTorch, or RSL-RL.

Tests: six offline tests passed. These releases contain synthetic verification vectors; no recorded observations were supplied. C++ parity and hardware commissioning remain pending.

Earlier `dr_v2_model_999_v1`, `dr_v2_model_999_v2`, and `nominal_model_299_v1` directories are intermediate exports retained on the data volume; use the `release_1` directories listed here.
