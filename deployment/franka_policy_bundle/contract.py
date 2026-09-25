"""Versioned, simulator-independent contract for the Franka reach actor."""

import numpy as np
import yaml

OBSERVATION_LAYOUT = [
    {"start": 0, "stop": 7, "expression": "q_measured - q_default", "unit": "rad"},
    {"start": 7, "stop": 14, "expression": "dq_measured", "unit": "rad/s"},
    {"start": 14, "stop": 17, "expression": "target_base - panda_hand_base", "unit": "m"},
    {"start": 17, "stop": 24, "expression": "previous_raw_actor_output", "unit": "unitless"},
]


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


def extract_contract(env, agent):
    require(env.get("robot_model", "panda") == "panda", "FR3 export requires a separately versioned hardware contract")
    actor = agent["actor"]
    require(actor["class_name"] == "MLPModel" and actor["activation"] == "elu", "Unsupported actor")
    require(actor["obs_normalization"] is False, "Normalized actors unsupported")
    require(agent.get("clip_actions") is None and not agent.get("obs_groups"), "Unsupported runner transforms")
    names = [f"panda_joint{i}" for i in range(1, 8)]
    policy = env["observations"]["policy"]
    terms = [k for k, v in policy.items() if isinstance(v, dict) and "func" in v]
    expected = ["joint_pos_rel", "joint_vel_rel", "ee_position_error", "previous_action"]
    require(terms == expected, f"Unexpected observation ordering: {terms}")
    require(policy["concatenate_terms"] and not policy.get("history_length"), "Unsupported history")
    for name, function in zip(
        terms, ["joint_pos_rel", "joint_vel_rel", "ee_position_error_b", "last_action"], strict=True
    ):
        term = policy[name]
        require(term["func"].split(":")[-1] == function, f"Unsupported observation: {name}")
        require(
            not term.get("modifiers")
            and term.get("clip") is None
            and term.get("scale") is None
            and not term.get("history_length"),
            f"Transformed term: {name}",
        )
    for name in expected[:2]:
        a = policy[name]["params"]["asset_cfg"]
        require(a["name"] == "robot" and a["joint_names"] == ["panda_joint.*"], "Joint selection differs")
    error = policy["ee_position_error"]["params"]
    require(
        error["command_name"] == "ee_pose" and error["asset_cfg"]["body_names"] == ["panda_hand"],
        "Tracked body differs",
    )
    require(list(env["actions"]) == ["arm_action"], "Additional actions unsupported")
    a = env["actions"]["arm_action"]
    require(
        a["class_type"].endswith(":JointPositionAction")
        and a["use_default_offset"]
        and a["joint_names"] == ["panda_joint.*"]
        and a["clip"] is None,
        "Unsupported action mapping",
    )
    init = env["scene"]["robot"]["init_state"]
    require(all(v == 0 for v in init["joint_vel"].values()), "Nonzero default velocity")
    ranges = env["commands"]["ee_pose"]["ranges"]
    c = dict(
        schema_version=1,
        robot_model="panda",
        joint_names=names,
        default_joint_position_rad=[init["joint_pos"][n] for n in names],
        default_joint_velocity_rad_s=[0.0] * 7,
        policy_period_s=env["sim"]["dt"] * env["decimation"],
        observation_layout=OBSERVATION_LAYOUT,
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
            scale_rad=a["scale"],
            clip=None,
        ),
        frames=dict(base="panda_link0", tracked_body="panda_hand"),
        target_workspace_m={a: ranges[f"pos_{a}"] for a in "xyz"},
        deployment_observation_noise=False,
    )
    validate_contract(c)
    return c


def validate_contract(c):
    require(c["observation_layout"] == OBSERVATION_LAYOUT, "Invalid observation slices/units")
    require(c["schema_version"] == 1 and c["robot_model"] == "panda", "Unsupported contract")
    require(c["joint_names"] == [f"panda_joint{i}" for i in range(1, 8)], "Invalid joint order")
    require(
        c["observation"]
        == dict(
            size=24,
            dtype="float32",
            terms=["joint_pos_rel", "joint_vel_rel", "ee_position_error", "previous_action"],
            normalization=False,
            previous_action="raw_actor_output",
            reset_previous_action=[0.0] * 7,
        ),
        "Invalid observation contract",
    )
    require(
        c["action"]["mapping"] == "default_plus_scaled"
        and c["action"]["inference"] == "deterministic_mean"
        and c["action"]["size"] == 7
        and c["action"]["dtype"] == "float32"
        and c["action"]["clip"] is None,
        "Invalid action contract",
    )
    require(c["frames"] == dict(base="panda_link0", tracked_body="panda_hand"), "Invalid frames")
    q = np.asarray(c["default_joint_position_rad"])
    require(q.shape == (7,) and np.isfinite(q).all(), "Invalid default positions")
    require(c["default_joint_velocity_rad_s"] == [0.0] * 7, "Invalid default velocities")
    for v in [c["policy_period_s"], c["action"]["scale_rad"]]:
        require(np.isscalar(v) and np.isfinite(v) and v > 0, "Invalid period/scale")
    for a in "xyz":
        b = np.asarray(c["target_workspace_m"][a])
        require(b.shape == (2,) and np.isfinite(b).all() and b[0] < b[1], "Invalid workspace")


def observation(q, dq, target, hand, previous_action, contract):
    arrays = [np.asarray(x, dtype=np.float32) for x in (q, dq, target, hand, previous_action)]
    for x, width in zip(arrays, [7, 7, 3, 3, 7], strict=True):
        require(x.ndim >= 1 and x.shape[-1] == width and np.isfinite(x).all(), "Invalid state")
        require(x.shape[:-1] == arrays[0].shape[:-1], "Batch dimensions differ")
    q, dq, target, hand, previous_action = arrays
    return np.concatenate(
        (q - np.asarray(contract["default_joint_position_rad"], dtype=np.float32), dq, target - hand, previous_action),
        axis=-1,
    )


def action_mapping(raw, contract):
    raw = np.asarray(raw, dtype=np.float32)
    require(raw.ndim >= 1 and raw.shape[-1] == 7 and np.isfinite(raw).all(), "Invalid action")
    return np.asarray(contract["default_joint_position_rad"], dtype=np.float32) + contract["action"]["scale_rad"] * raw
