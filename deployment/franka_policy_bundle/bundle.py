"""Export and independently verify immutable, CPU-only policy bundles."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .contract import TrainingLoader, action_mapping, extract_contract, observation, require, validate_contract


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git_state(path):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return dict(revision=run("rev-parse", "HEAD"), status=run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return dict(revision=None, status="unavailable")


def session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def verify(bundle, expected_manifest_sha256=None):
    """No PyTorch, RSL-RL, checkpoint, Isaac, or ROS dependency."""
    import onnx

    root = Path(bundle).resolve(strict=True)
    if expected_manifest_sha256:
        require(sha256(root / "manifest.json") == expected_manifest_sha256, "Manifest trust anchor mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema_version"] == 1, "Unsupported manifest")
    required = {
        "policy.onnx",
        "policy_contract.yaml",
        "test_vectors.npz",
        "verification.json",
        "provenance/agent.yaml",
        "provenance/env.yaml",
        "schemas/policy_contract.schema.json",
    }
    require(required <= manifest["files"].keys(), "Missing required file hashes")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"Invalid bundle path: {name}")
        require(sha256(path) == digest, f"Hash mismatch: {name}")
    c = yaml.safe_load((root / "policy_contract.yaml").read_text())
    import jsonschema

    schema = json.loads((root / "schemas" / "policy_contract.schema.json").read_text())
    jsonschema.validate(c, schema)
    validate_contract(c)
    saved_env = yaml.load((root / "provenance/env.yaml").read_text(), Loader=TrainingLoader)
    saved_agent = yaml.load((root / "provenance/agent.yaml").read_text(), Loader=TrainingLoader)
    require(extract_contract(saved_env, saved_agent) == c, "Contract differs from saved training configuration")
    onnx.checker.check_model(str(root / "policy.onnx"))
    model = session(root / "policy.onnx")
    inputs, outputs = model.get_inputs(), model.get_outputs()
    observation_size = c["observation"]["size"]
    action_size = c["action"]["size"]
    require(len(inputs) == len(outputs) == 1, "Unexpected tensor count")
    require(
        inputs[0].name == "observation"
        and inputs[0].type == "tensor(float)"
        and inputs[0].shape == ["batch", observation_size],
        "Unexpected ONNX input",
    )
    require(
        outputs[0].name == "action"
        and outputs[0].type == "tensor(float)"
        and outputs[0].shape == ["batch", action_size],
        "Unexpected ONNX output",
    )
    tolerance = manifest["verification_tolerance_abs"]
    require(np.isfinite(tolerance) and 0 < tolerance <= 1e-5, "Invalid tolerance")
    with np.load(root / "test_vectors.npz", allow_pickle=False) as v:
        obs = v["observation"]
        require(
            obs.dtype == np.float32
            and obs.ndim == 2
            and obs.shape[1] == observation_size
            and len(obs) >= observation_size + 1
            and np.isfinite(obs).all(),
            "Invalid test observations",
        )
        reference = v["reference"] if "reference" in v.files else None
        assembled = observation(
            v["q"], v["dq"], v["target"], v["hand"], v["previous_action"], c, reference=reference
        )
        require(np.allclose(assembled, obs, atol=1e-6, rtol=0), "Observation assembly mismatch")
        action = model.run(["action"], {"observation": obs})[0]
        expected = v["action"]
        require(
            action.shape == expected.shape == (len(obs), action_size)
            and np.isfinite(action).all()
            and np.isfinite(expected).all(),
            "Invalid actor output",
        )
        error = float(np.max(np.abs(action - expected)))
        require(error <= tolerance, f"Actor mismatch: {error}")
        single = model.run(["action"], {"observation": obs[:1]})[0]
        require(np.allclose(single, expected[:1], atol=tolerance, rtol=0), "Single-state mismatch")
        require(
            np.allclose(action_mapping(expected, c, reference=reference), v["q_policy"], atol=1e-6, rtol=0),
            "Action mapping mismatch",
        )
    return dict(
        passed=True,
        vectors=len(obs),
        max_abs_error=error,
        manifest_sha256=sha256(root / "manifest.json"),
        provider="CPUExecutionProvider",
        hardware_validated=False,
    )


def export(checkpoint, output, recorded_observations=None, seed=123, count=1024):
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists(), f"Refusing to overwrite {output}")
    params = checkpoint.parent / "params"
    env = yaml.load((params / "env.yaml").read_text(), Loader=TrainingLoader)
    agent = yaml.load((params / "agent.yaml").read_text(), Loader=TrainingLoader)
    c = extract_contract(env, agent)
    observation_size = c["observation"]["size"]
    action_size = c["action"]["size"]
    sentinel_count = observation_size + 1
    require(count >= sentinel_count, f"At least {sentinel_count} test vectors required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = copy.deepcopy(agent["actor"])
    cfg.pop("class_name")
    actor = (
        MLPModel(
            TensorDict({"policy": torch.zeros(1, observation_size)}, batch_size=[1]),
            {"actor": ["policy"]},
            "actor",
            action_size,
            **cfg,
        )
        .cpu()
        .eval()
    )
    actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    require(all(torch.isfinite(v).all() for v in actor.state_dict().values()), "Nonfinite checkpoint")
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, (count, observation_size)).astype(np.float32)
    obs[0] = 0
    obs[1:sentinel_count] = np.eye(observation_size, dtype=np.float32)
    if recorded_observations:
        with np.load(recorded_observations, allow_pickle=False) as recorded:
            extra = np.asarray(recorded["observation"], dtype=np.float32)
        require(
            extra.ndim == 2 and extra.shape[1] == observation_size and np.isfinite(extra).all(),
            "Invalid recorded observations",
        )
        obs = np.concatenate((obs, extra))
    q = obs[:, :7] + np.asarray(c["default_joint_position_rad"], dtype=np.float32)
    dq = obs[:, 7:14]
    target = obs[:, 14:17]
    hand = np.zeros((len(obs), 3), np.float32)
    reference = None
    if c["schema_version"] == 1:
        previous = obs[:, 17:24]
    else:
        lower = np.asarray(c["action"]["soft_lower_rad"], dtype=np.float32)
        upper = np.asarray(c["action"]["soft_upper_rad"], dtype=np.float32)
        reference = np.repeat(q[:, 6:7], 7, axis=1)
        reference[:, :6] = lower[:6] + 0.5 * (obs[:, 17:23] + 1.0) * (upper[:6] - lower[:6])
        previous = obs[:, 23:29]
    obs = observation(q, dq, target, hand, previous, c, reference=reference)
    with torch.inference_mode():
        expected = actor(
            TensorDict({"policy": torch.from_numpy(obs)}, batch_size=[len(obs)]), stochastic_output=False
        ).numpy()
    # Native MLPModel deterministic output is validated against its exported MLP below.
    output.parent.mkdir(parents=True, exist_ok=True)
    require(
        shutil.disk_usage(output.parent).free > checkpoint.stat().st_size * 3 + 256 * 1024**2, "Insufficient free space"
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        repo = Path(__file__).parents[2]
        export_actor = actor.as_onnx(verbose=False).cpu().eval()
        torch.onnx.export(
            export_actor,
            torch.zeros(1, observation_size),
            str(staging / "policy.onnx"),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        (staging / "policy_contract.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        vectors = dict(
            observation=obs,
            action=expected,
            q=q,
            dq=dq,
            target=target,
            hand=hand,
            previous_action=previous,
            q_policy=action_mapping(expected, c, reference=reference),
        )
        if reference is not None:
            vectors["reference"] = reference
        np.savez(staging / "test_vectors.npz", **vectors)
        # JSON fixtures make parity checks possible without an NPZ reader on the C++ host.
        write_json(
            staging / "test_vectors.json",
            {
                k: v.tolist()
                for k, v in dict(
                    observation=obs[:sentinel_count],
                    action=expected[:sentinel_count],
                    q=q[:sentinel_count],
                    dq=dq[:sentinel_count],
                    target=target[:sentinel_count],
                    hand=hand[:sentinel_count],
                    previous_action=previous[:sentinel_count],
                    q_policy=action_mapping(
                        expected[:sentinel_count], c,
                        reference=reference[:sentinel_count] if reference is not None else None,
                    ),
                    **({"reference": reference[:sentinel_count]} if reference is not None else {}),
                ).items()
            },
        )
        (staging / "provenance").mkdir()
        for name in ["env.yaml", "agent.yaml", "scenario.yaml"]:
            if (params / name).exists():
                shutil.copyfile(params / name, staging / "provenance" / name)
        source_snapshot = [
            "source/franka_rl/franka_rl/tasks/manager_based/franka_incremental_impedance/incremental_impedance_env_cfg.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_incremental_impedance/rsl_rl_incremental_impedance_ppo_cfg.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/impedance_actions.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/impedance_actions_cfg.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/impedance_metrics.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/observations.py",
            "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/rewards.py",
            "source/franka_rl/franka_rl/utils/rsl_rl_distributions.py",
        ]
        if c.get("contract_id") == "fr3_joint_velocity_impedance_29d_v1":
            source_snapshot.extend(
                [
                    "source/franka_rl/franka_rl/tasks/manager_based/franka_velocity_impedance/velocity_impedance_env_cfg.py",
                    "source/franka_rl/franka_rl/tasks/manager_based/franka_velocity_impedance/rsl_rl_velocity_impedance_ppo_cfg.py",
                    "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/impedance_controller.py",
                ]
            )
        for relative_name in source_snapshot:
            source = repo / relative_name
            require(source.is_file(), f"Source snapshot file missing: {relative_name}")
            destination = staging / "provenance" / "source_snapshot" / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copytree(Path(__file__).parents[1] / "schemas", staging / "schemas")
        shutil.copytree(
            Path(__file__).parent, staging / "franka_policy_bundle", ignore=shutil.ignore_patterns("__pycache__")
        )
        (staging / "verify.py").write_text("from franka_policy_bundle.bundle import verify_main\nverify_main()\n")
        shutil.copyfile(Path(__file__).parents[1] / "requirements.txt", staging / "requirements.txt")
        (staging / "README.md").write_text(
            "# Franka policy bundle\n\n"
            "Verify locally: `python verify.py . --manifest-sha256 <trusted hash>`\n"
            "Install requirements.txt in a separate environment. No Isaac, ROS, or PyTorch needed for verification.\n"
            f"ONNX input observation: float32 [batch,{observation_size}]; "
            f"output action: float32 [batch,{action_size}].\n"
            "Use policy_contract.yaml for frames, units, ordering, timing and action mapping.\n"
            "Numerical verification is not hardware commissioning. No controller or safety governor is included.\n"
            "JSON vectors contain zero and per-input sentinel cases; NPZ additionally contains seeded random cases.\n"
        )
        actual = session(staging / "policy.onnx").run(["action"], {"observation": obs})[0]
        error = float(np.max(np.abs(actual - expected)))
        require(np.isfinite(actual).all() and error <= 1e-5, f"Native RSL-RL/ONNX mismatch: {error}")
        write_json(
            staging / "verification.json",
            dict(
                passed=True,
                reference="native RSL-RL MLPModel deterministic output",
                max_abs_error=error,
                vectors=len(obs),
                seed=seed,
                recorded_vectors=len(obs) - count,
                hardware_validated=False,
                cpp_parity_validated=False,
            ),
        )
        manifest = dict(
            schema_version=1,
            bundle_id=output.name,
            created_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint), iteration=ckpt.get("iter")),
            task=env.get("control_contract", "Template-Franka-Rl-v0"),
            policy_contract_id=c.get("contract_id", "panda_joint_position_24d_v1"),
            verification_tolerance_abs=1e-5,
            git=dict(project=git_state(repo), isaaclab=git_state(repo.parent / "IsaacLab")),
            versions={
                name: importlib.metadata.version(name)
                for name in ["torch", "rsl-rl-lib", "onnx", "onnxruntime", "numpy", "PyYAML"]
            },
            python=sys.version,
            exporter_argv=sys.argv,
            source_snapshot=source_snapshot,
            recorded_observations_sha256=sha256(recorded_observations) if recorded_observations else None,
            files={str(f.relative_to(staging)): sha256(f) for f in sorted(staging.rglob("*")) if f.is_file()},
        )
        write_json(staging / "manifest.json", manifest)
        report = verify(staging)
        staging.rename(output)
        return report
    except BaseException:
        # Preserve failed staging artifacts for diagnosis; never advertise a partial final bundle.
        print(f"Export failed; staging preserved at {staging}", file=sys.stderr)
        raise


def export_main():
    p = argparse.ArgumentParser(description="Export a verified offline Franka policy bundle")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle-id", required=True)
    p.add_argument(
        "--data-root",
        default=os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"),
    )
    p.add_argument("--recorded-observations")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-vectors", type=int, default=1024)
    a = p.parse_args()
    require(a.bundle_id not in {".", ".."} and Path(a.bundle_id).name == a.bundle_id, "Invalid bundle ID")
    root = Path(a.data_root).resolve(strict=True)
    require(root.is_dir() and os.access(root, os.W_OK), "Data root is not writable")
    if str(root).startswith("/media/"):
        require(
            any(os.path.ismount(x) for x in [root, *root.parents] if str(x) not in {"/", "/media"}),
            "Data volume unmounted",
        )
    output = root / "deployment_bundles" / a.bundle_id
    print(json.dumps(export(a.checkpoint, output, a.recorded_observations, a.seed, a.num_vectors), indent=2))
    print(f"Bundle: {output}")


def verify_main():
    p = argparse.ArgumentParser(description="Verify a bundle without Isaac or PyTorch")
    p.add_argument("bundle")
    p.add_argument("--manifest-sha256", help="Trusted hash supplied separately from the bundle")
    a = p.parse_args()
    print(json.dumps(verify(a.bundle, a.manifest_sha256), indent=2))
