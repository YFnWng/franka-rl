"""Versioned, simulator-independent contracts for Franka policy actors."""

from __future__ import annotations

import numpy as np
import yaml


PANDA_OBSERVATION_LAYOUT = [
    {"start": 0, "stop": 7, "expression": "q_measured - q_default", "unit": "rad"},
    {"start": 7, "stop": 14, "expression": "dq_measured", "unit": "rad/s"},
    {"start": 14, "stop": 17, "expression": "target_base - panda_hand_base", "unit": "m"},
    {"start": 17, "stop": 24, "expression": "previous_raw_actor_output", "unit": "unitless"},
]

FR3_INCREMENTAL_POSITION_OBSERVATION_LAYOUT = [
    {"start": 0, "stop": 7, "expression": "q_measured - q_default", "unit": "rad"},
    {"start": 7, "stop": 14, "expression": "dq_measured", "unit": "rad/s"},
    {"start": 14, "stop": 17, "expression": "target_base - fr3_flange_base", "unit": "m"},
    {"start": 17, "stop": 23, "expression": "q_reference_normalized", "unit": "unitless"},
    {
        "start": 23,
        "stop": 29,
        "expression": "previous_normalized_position_increment",
        "unit": "unitless",
    },
]

# Backward-compatible public name used by older tests/importers.
OBSERVATION_LAYOUT = PANDA_OBSERVATION_LAYOUT

FR3_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_SIM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FR3_CONTRACT_ID = "fr3_incremental_position_29d_v1"


class TrainingLoader(yaml.SafeLoader):
    """Read Isaac metadata without constructing Python objects."""


TrainingLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: l.construct_sequence(n))
TrainingLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", lambda l, n: l.construct_sequence(n)
)
TrainingLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", lambda l, t, n: l.construct_mapping(n))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_actor(agent, *, squashed: bool) -> None:
    actor = agent["actor"]
    require(actor["class_name"] == "MLPModel" and actor["activation"] == "elu", "Unsupported actor")
    require(actor["obs_normalization"] is False, "Normalized actors unsupported")
    require(agent.get("clip_actions") is None and not agent.get("obs_groups"), "Unsupported runner transforms")
    if squashed:
        distribution = actor.get("distribution_cfg") or {}
        require(
            str(distribution.get("class_name", "")).endswith(":SquashedGaussianDistribution"),
            "FR3 incremental actor must use SquashedGaussianDistribution",
        )
        require(
            list(distribution.get("initial_action", ())) == [0.0] * 6,
            "FR3 incremental actor initial action must be six zeros",
        )


def _observation_terms(policy):
    return [name for name, value in policy.items() if isinstance(value, dict) and "func" in value]


def _validate_term(term, expected_function: str, name: str) -> None:
    require(term["func"].split(":")[-1] == expected_function, f"Unsupported observation: {name}")
    require(
        not term.get("modifiers")
        and term.get("clip") is None
        and term.get("scale") is None
        and not term.get("history_length"),
        f"Transformed term: {name}",
    )


def extract_contract(env, agent):
    """Extract a strict deployment contract from saved training configurations."""

    if env.get("robot_model", "panda") == "fr3v2_bare_flange":
        contract = _extract_fr3_incremental_contract(env, agent)
    else:
        contract = _extract_panda_contract(env, agent)
    validate_contract(contract)
    return contract


def _extract_panda_contract(env, agent):
    require(env.get("robot_model", "panda") == "panda", "Unsupported robot model")
    _validate_actor(agent, squashed=False)
    names = [f"panda_joint{i}" for i in range(1, 8)]
    policy = env["observations"]["policy"]
    terms = _observation_terms(policy)
    expected = ["joint_pos_rel", "joint_vel_rel", "ee_position_error", "previous_action"]
    require(terms == expected, f"Unexpected observation ordering: {terms}")
    require(policy["concatenate_terms"] and not policy.get("history_length"), "Unsupported history")
    for name, function in zip(
        terms, ["joint_pos_rel", "joint_vel_rel", "ee_position_error_b", "last_action"], strict=True
    ):
        _validate_term(policy[name], function, name)
    for name in expected[:2]:
        asset = policy[name]["params"]["asset_cfg"]
        require(asset["name"] == "robot" and asset["joint_names"] == ["panda_joint.*"], "Joint selection differs")
    error = policy["ee_position_error"]["params"]
    require(
        error["command_name"] == "ee_pose" and error["asset_cfg"]["body_names"] == ["panda_hand"],
        "Tracked body differs",
    )
    require(list(env["actions"]) == ["arm_action"], "Additional actions unsupported")
    action = env["actions"]["arm_action"]
    require(
        action["class_type"].endswith(":JointPositionAction")
        and action["use_default_offset"]
        and action["joint_names"] == ["panda_joint.*"]
        and action["clip"] is None,
        "Unsupported action mapping",
    )
    init = env["scene"]["robot"]["init_state"]
    require(all(value == 0 for value in init["joint_vel"].values()), "Nonzero default velocity")
    ranges = env["commands"]["ee_pose"]["ranges"]
    return dict(
        schema_version=1,
        robot_model="panda",
        joint_names=names,
        default_joint_position_rad=[init["joint_pos"][name] for name in names],
        default_joint_velocity_rad_s=[0.0] * 7,
        policy_period_s=env["sim"]["dt"] * env["decimation"],
        observation_layout=PANDA_OBSERVATION_LAYOUT,
        observation=dict(
            size=24,
            dtype="float32",
            terms=expected,
            normalization=False,
            previous_action="raw_actor_output",
            reset_previous_action=[0.0] * 7,
        ),
        action=dict(
            size=7,
            dtype="float32",
            inference="deterministic_mean",
            mapping="default_plus_scaled",
            scale_rad=action["scale"],
            clip=None,
        ),
        frames=dict(base="panda_link0", tracked_body="panda_hand"),
        target_workspace_m={axis: ranges[f"pos_{axis}"] for axis in "xyz"},
        deployment_observation_noise=False,
    )


def _extract_fr3_incremental_contract(env, agent):
    _validate_actor(agent, squashed=True)
    require(
        env.get("control_contract") == "fr3_franky_incremental_6d_joint_impedance_v1",
        "Unsupported FR3 control contract",
    )
    require(env["sim"]["dt"] == 0.001 and env["decimation"] == 20, "FR3 policy must use 1 kHz/50 Hz timing")
    policy = env["observations"]["policy"]
    terms = _observation_terms(policy)
    expected = [
        "joint_pos_rel",
        "joint_vel_rel",
        "ee_position_error",
        "position_reference",
        "previous_increment_action",
    ]
    require(terms == expected, f"Unexpected FR3 observation ordering: {terms}")
    require(policy["concatenate_terms"] and not policy.get("history_length"), "Unsupported FR3 observation history")
    functions = [
        "joint_pos_rel",
        "joint_vel_rel",
        "ee_position_error_b",
        "normalized_position_reference",
        "incremental_position_action",
    ]
    for name, function in zip(terms, functions, strict=True):
        _validate_term(policy[name], function, name)
    for name in expected[:2]:
        asset = policy[name]["params"]["asset_cfg"]
        require(asset["name"] == "robot" and asset["joint_names"] == ["panda_joint.*"], "FR3 joint selection differs")
    error = policy["ee_position_error"]["params"]
    require(
        error["command_name"] == "ee_pose" and error["asset_cfg"]["body_names"] == ["fr3_flange"],
        "FR3 tracked body differs",
    )
    for name in expected[3:]:
        require(policy[name]["params"] == {"action_name": "arm_action"}, f"Unexpected {name} source")

    require(list(env["actions"]) == ["arm_action"], "Additional FR3 actions unsupported")
    action = env["actions"]["arm_action"]
    require(
        action["class_type"].endswith(":FrankyIncremental6DImpedanceAction")
        and action["joint_names"] == FR3_SIM_JOINT_NAMES
        and action["preserve_order"] is True
        and action["clip"] is None,
        "Unsupported FR3 incremental action mapping",
    )
    require(
        action["nominal_stiffness"] == 100.0
        and action["position_error_clip"] == 0.5
        and action["torque_slew_rate"] == 1000.0
        and action["command_filter_cutoff_hz"] == 100.0
        and action["compensate_coriolis"] is True
        and action["compensate_gravity"] is True,
        "Unsupported FR3 impedance controller",
    )
    init = env["scene"]["robot"]["init_state"]
    require(all(value == 0 for value in init["joint_vel"].values()), "Nonzero FR3 default velocity")
    ranges = env["commands"]["ee_pose"]["ranges"]
    max_velocity = [float(value) for value in action["max_reference_velocity"]]
    max_increment = [value * 0.02 for value in max_velocity]
    return dict(
        schema_version=2,
        contract_id=FR3_CONTRACT_ID,
        robot_model="fr3",
        robot_description="fr3v2.1_bare_flange",
        joint_names=FR3_JOINT_NAMES,
        default_joint_position_rad=[init["joint_pos"][name] for name in FR3_SIM_JOINT_NAMES],
        default_joint_velocity_rad_s=[0.0] * 7,
        policy_period_s=0.02,
        observation_layout=FR3_INCREMENTAL_POSITION_OBSERVATION_LAYOUT,
        observation=dict(
            size=29,
            dtype="float32",
            terms=expected,
            normalization=False,
            previous_action="bounded_normalized_position_increment",
            reset_previous_action=[0.0] * 6,
            reference_initialization="measured_start_position",
        ),
        action=dict(
            type="normalized_position_increment",
            size=6,
            dtype="float32",
            inference="deterministic_tanh",
            normalized_bounds=[-1.0, 1.0],
            integration="forward_euler",
            initial_reference="measured_start_position",
            controlled_joint_names=FR3_JOINT_NAMES[:6],
            held_joint_names=FR3_JOINT_NAMES[6:],
            max_reference_velocity_rad_s=max_velocity,
            max_position_increment_rad=max_increment,
            soft_limit_projection=True,
            soft_lower_rad=[float(value) for value in action["soft_lower"]],
            soft_upper_rad=[float(value) for value in action["soft_upper"]],
        ),
        frames=dict(base="fr3_link0", tracked_body="fr3_flange"),
        target_workspace_m={axis: ranges[f"pos_{axis}"] for axis in "xyz"},
        deployment_observation_noise=False,
        controller=dict(
            nominal_stiffness_nm_rad=float(action["nominal_stiffness"]),
            nominal_damping_nms_rad=20.0,
            position_error_clip_rad=float(action["position_error_clip"]),
            torque_slew_rate_nm_s=float(action["torque_slew_rate"]),
            torque_filter_cutoff_hz=float(action["command_filter_cutoff_hz"]),
            coriolis_compensation=True,
            hardware_gravity_convention="libfranka_external_to_user_torque",
        ),
    )


def validate_contract(contract):
    if contract.get("schema_version") == 1:
        _validate_panda_contract(contract)
    elif contract.get("schema_version") == 2:
        _validate_fr3_incremental_contract(contract)
    else:
        raise ValueError("Unsupported contract schema version")


def _validate_common(contract, *, observation_size: int, action_size: int) -> None:
    q = np.asarray(contract["default_joint_position_rad"], dtype=float)
    require(q.shape == (7,) and np.isfinite(q).all(), "Invalid default positions")
    require(contract["default_joint_velocity_rad_s"] == [0.0] * 7, "Invalid default velocities")
    require(
        np.isfinite(contract["policy_period_s"]) and contract["policy_period_s"] > 0,
        "Invalid policy period",
    )
    require(contract["observation"]["size"] == observation_size, "Invalid observation size")
    require(contract["action"]["size"] == action_size, "Invalid action size")
    require(
        contract["observation"]["dtype"] == contract["action"]["dtype"] == "float32",
        "Invalid tensor dtype",
    )
    require(contract["deployment_observation_noise"] is False, "Deployment observation noise unsupported")
    for axis in "xyz":
        bounds = np.asarray(contract["target_workspace_m"][axis], dtype=float)
        require(bounds.shape == (2,) and np.isfinite(bounds).all() and bounds[0] < bounds[1], "Invalid workspace")


def _validate_panda_contract(contract):
    require(contract["observation_layout"] == PANDA_OBSERVATION_LAYOUT, "Invalid Panda observation layout")
    require(contract["robot_model"] == "panda", "Unsupported Panda contract")
    require(contract["joint_names"] == [f"panda_joint{i}" for i in range(1, 8)], "Invalid Panda joint order")
    require(
        contract["observation"]
        == dict(
            size=24,
            dtype="float32",
            terms=["joint_pos_rel", "joint_vel_rel", "ee_position_error", "previous_action"],
            normalization=False,
            previous_action="raw_actor_output",
            reset_previous_action=[0.0] * 7,
        ),
        "Invalid Panda observation contract",
    )
    require(
        contract["action"]["mapping"] == "default_plus_scaled"
        and contract["action"]["inference"] == "deterministic_mean"
        and contract["action"]["clip"] is None,
        "Invalid Panda action contract",
    )
    require(contract["frames"] == dict(base="panda_link0", tracked_body="panda_hand"), "Invalid Panda frames")
    require(np.isfinite(contract["action"]["scale_rad"]) and contract["action"]["scale_rad"] > 0, "Invalid scale")
    _validate_common(contract, observation_size=24, action_size=7)


def _validate_fr3_incremental_contract(contract):
    require(contract.get("contract_id") == FR3_CONTRACT_ID, "Unsupported FR3 contract ID")
    require(contract.get("robot_model") == "fr3", "Unsupported FR3 robot model")
    require(contract.get("robot_description") == "fr3v2.1_bare_flange", "Unsupported FR3 description")
    require(contract["joint_names"] == FR3_JOINT_NAMES, "Invalid FR3 joint order")
    require(contract["observation_layout"] == FR3_INCREMENTAL_POSITION_OBSERVATION_LAYOUT, "Invalid FR3 observation layout")
    require(contract["frames"] == dict(base="fr3_link0", tracked_body="fr3_flange"), "Invalid FR3 frames")
    observation_contract = contract["observation"]
    require(
        observation_contract["terms"]
        == ["joint_pos_rel", "joint_vel_rel", "ee_position_error", "position_reference", "previous_increment_action"]
        and observation_contract["normalization"] is False
        and observation_contract["previous_action"] == "bounded_normalized_position_increment"
        and observation_contract["reset_previous_action"] == [0.0] * 6
        and observation_contract["reference_initialization"] == "measured_start_position",
        "Invalid FR3 observation contract",
    )
    action = contract["action"]
    require(
        action["type"] == "normalized_position_increment"
        and action["inference"] == "deterministic_tanh"
        and action["normalized_bounds"] == [-1.0, 1.0]
        and action["integration"] == "forward_euler"
        and action["initial_reference"] == "measured_start_position"
        and action["controlled_joint_names"] == FR3_JOINT_NAMES[:6]
        and action["held_joint_names"] == FR3_JOINT_NAMES[6:]
        and action["soft_limit_projection"] is True,
        "Invalid FR3 action contract",
    )
    vmax = np.asarray(action["max_reference_velocity_rad_s"], dtype=float)
    increment = np.asarray(action["max_position_increment_rad"], dtype=float)
    lower = np.asarray(action["soft_lower_rad"], dtype=float)
    upper = np.asarray(action["soft_upper_rad"], dtype=float)
    require(vmax.shape == increment.shape == (6,) and np.isfinite(vmax).all() and (vmax > 0).all(), "Invalid FR3 velocity")
    require(np.allclose(increment, vmax * contract["policy_period_s"], atol=1e-12, rtol=0), "Invalid FR3 increment")
    require(lower.shape == upper.shape == (7,) and np.isfinite(lower).all() and np.all(lower < upper), "Invalid FR3 soft limits")
    controller = contract["controller"]
    require(
        controller["nominal_stiffness_nm_rad"] == 100.0
        and controller["nominal_damping_nms_rad"] == 20.0
        and controller["position_error_clip_rad"] == 0.5
        and controller["torque_slew_rate_nm_s"] == 1000.0
        and controller["torque_filter_cutoff_hz"] == 100.0
        and controller["coriolis_compensation"] is True,
        "Invalid FR3 controller contract",
    )
    _validate_common(contract, observation_size=29, action_size=6)
    require(abs(contract["policy_period_s"] - 0.02) < 1e-12, "FR3 policy period must be 20 ms")


def observation(q, dq, target, hand, previous_action, contract, *, reference=None, z_axis_error=None):
    """Assemble a deployment observation from physical state arrays."""

    base = [np.asarray(value, dtype=np.float32) for value in (q, dq, target, hand)]
    for value, width in zip(base, [7, 7, 3, 3], strict=True):
        require(value.ndim >= 1 and value.shape[-1] == width and np.isfinite(value).all(), "Invalid state")
        require(value.shape[:-1] == base[0].shape[:-1], "Batch dimensions differ")
    q, dq, target, hand = base
    previous = np.asarray(previous_action, dtype=np.float32)
    default = np.asarray(contract["default_joint_position_rad"], dtype=np.float32)
    if contract["schema_version"] == 1:
        require(previous.shape == (*q.shape[:-1], 7) and np.isfinite(previous).all(), "Invalid previous action")
        return np.concatenate((q - default, dq, target - hand, previous), axis=-1)

    reference_array = np.asarray(reference, dtype=np.float32)
    require(reference_array.shape == q.shape and np.isfinite(reference_array).all(), "Invalid position reference")
    require(previous.shape == (*q.shape[:-1], 6) and np.isfinite(previous).all(), "Invalid previous increment")
    lower = np.asarray(contract["action"]["soft_lower_rad"], dtype=np.float32)[:6]
    upper = np.asarray(contract["action"]["soft_upper_rad"], dtype=np.float32)[:6]
    normalized_reference = 2.0 * (reference_array[..., :6] - lower) / (upper - lower) - 1.0
    result = np.concatenate((q - default, dq, target - hand, normalized_reference, previous), axis=-1)
    if z_axis_error is not None:
        z_error = np.asarray(z_axis_error, dtype=np.float32)
        require(z_error.shape == (*q.shape[:-1], 2) and np.isfinite(z_error).all(), "Invalid z-axis error")
        result = np.concatenate((result, z_error), axis=-1)
    return result


def action_mapping(raw, contract, *, reference=None):
    """Map deterministic actor output to the next physical position reference."""

    raw = np.asarray(raw, dtype=np.float32)
    require(raw.ndim >= 1 and np.isfinite(raw).all(), "Invalid action")
    if contract["schema_version"] == 1:
        require(raw.shape[-1] == 7, "Invalid Panda action")
        return np.asarray(contract["default_joint_position_rad"], dtype=np.float32) + contract["action"]["scale_rad"] * raw

    require(raw.shape[-1] == 6 and np.all(raw >= -1.0) and np.all(raw <= 1.0), "Invalid FR3 action")
    reference_array = np.asarray(reference, dtype=np.float32)
    require(reference_array.shape == (*raw.shape[:-1], 7) and np.isfinite(reference_array).all(), "Invalid FR3 reference")
    increment = np.asarray(contract["action"]["max_position_increment_rad"], dtype=np.float32)
    lower = np.asarray(contract["action"]["soft_lower_rad"], dtype=np.float32)
    upper = np.asarray(contract["action"]["soft_upper_rad"], dtype=np.float32)
    target = reference_array.copy()
    target[..., :6] = np.clip(reference_array[..., :6] + raw * increment, lower[:6], upper[:6])
    return target
