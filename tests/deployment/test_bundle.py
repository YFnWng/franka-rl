import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment"))
from franka_policy_bundle.bundle import export, sha256, verify
from franka_policy_bundle.contract import TrainingLoader, action_mapping, extract_contract, observation
from franka_policy_bundle.suite import resolve_suite


@pytest.fixture
def configs():
    terms = {}
    for name, func in [
        ("joint_pos_rel", "joint_pos_rel"),
        ("joint_vel_rel", "joint_vel_rel"),
        ("ee_position_error", "ee_position_error_b"),
        ("previous_action", "last_action"),
    ]:
        terms[name] = dict(
            func="test:" + func,
            params=dict(
                asset_cfg=dict(name="robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]),
                command_name="ee_pose",
            ),
        )
    env = dict(
        observations=dict(policy=dict(concatenate_terms=True, **terms)),
        actions=dict(
            arm_action=dict(
                class_type="test:JointPositionAction",
                use_default_offset=True,
                joint_names=["panda_joint.*"],
                clip=None,
                scale=0.5,
            )
        ),
        scene=dict(
            robot=dict(init_state=dict(joint_pos={f"panda_joint{i}": i / 10 for i in range(1, 8)}, joint_vel={".*": 0}))
        ),
        commands=dict(ee_pose=dict(ranges=dict(pos_x=[0.35, 0.6], pos_y=[-0.2, 0.2], pos_z=[0.2, 0.5]))),
        sim=dict(dt=1 / 60),
        decimation=2,
    )
    agent = dict(
        actor=dict(
            class_name="MLPModel",
            hidden_dims=[8, 8],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=dict(class_name="GaussianDistribution", init_std=1.0),
        )
    )
    return env, agent


def test_sentinel_and_action(configs):
    c = extract_contract(*configs)
    q = np.asarray(c["default_joint_position_rad"]) + np.arange(7)
    obs = observation(q, np.arange(7) + 7, np.arange(3) + 15, np.ones(3), np.arange(7) + 17, c)
    np.testing.assert_allclose(obs, np.arange(24), atol=1e-6)
    np.testing.assert_allclose(action_mapping(np.ones(7), c), np.asarray(c["default_joint_position_rad"]) + 0.5)
    with pytest.raises(ValueError):
        observation(q, np.zeros(7), np.zeros(3), np.zeros(3), np.full(7, np.nan), c)


def test_reject_changed_contract(configs):
    env, agent = copy.deepcopy(configs)
    agent["actor"]["obs_normalization"] = True
    with pytest.raises(ValueError, match="Normalized"):
        extract_contract(env, agent)
    env, agent = copy.deepcopy(configs)
    env["observations"]["policy"]["joint_pos_rel"]["scale"] = 2
    with pytest.raises(ValueError, match="Transformed"):
        extract_contract(env, agent)


def test_yaml_does_not_execute():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load("!!python/object/apply:os.system [echo forbidden]", Loader=TrainingLoader)


@pytest.fixture
def bundle(tmp_path, configs):
    env, agent = configs
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
    (params / "agent.yaml").write_text(yaml.safe_dump(agent))
    cfg = copy.deepcopy(agent["actor"])
    cfg.pop("class_name")
    actor = MLPModel(
        TensorDict({"policy": torch.zeros(1, 24)}, batch_size=[1]), {"actor": ["policy"]}, "actor", 7, **cfg
    )
    ckpt = tmp_path / "model.pt"
    torch.save(dict(actor_state_dict=actor.state_dict(), iter=1), ckpt)
    target = tmp_path / "bundle"
    export(ckpt, target, count=64)
    return target


def test_export_verify_tamper(bundle):
    assert verify(bundle)["passed"]
    with pytest.raises(ValueError, match="trust anchor"):
        verify(bundle, "0" * 64)
    path = bundle / "policy_contract.yaml"
    path.write_text(path.read_text() + "# tampering\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify(bundle)


def test_no_overwrite(bundle):
    with pytest.raises(ValueError, match="overwrite"):
        export(bundle.parent / "model.pt", bundle)


def test_suite(bundle):
    root = bundle.parent
    targets = dict(
        schema_version=1, name="test", frame_id="panda_link0", targets=[dict(id="center", position_m=[0.45, 0, 0.35])]
    )
    tp = root / "targets.yaml"
    tp.write_text(yaml.safe_dump(targets))
    template = Path(__file__).resolve().parents[2] / "deployment/suites/shadow_smoke.template.yaml"
    s = yaml.safe_load(template.read_text())
    s["policy_bundle"] = dict(path="bundle", sha256=sha256(bundle / "manifest.json"))
    s["target_set"] = dict(path="targets.yaml", sha256=sha256(tp))
    sp = root / "suite.yaml"
    sp.write_text(yaml.safe_dump(s))
    assert len(resolve_suite(sp)["trials"]) == 3
    targets["targets"][0]["position_m"][0] = 3
    tp.write_text(yaml.safe_dump(targets))
    s["target_set"]["sha256"] = sha256(tp)
    sp.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError, match="outside policy workspace"):
        resolve_suite(sp)
