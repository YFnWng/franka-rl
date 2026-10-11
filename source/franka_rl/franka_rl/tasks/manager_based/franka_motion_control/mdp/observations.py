import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms


def measured_joint_state(env):
    robot = env.scene["robot"]
    return torch.cat((robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=1)


def applied_motion_reference(env):
    return env.motion_action.processed_actions


def tcp_pose_base(env):
    robot = env.scene["robot"]
    i = env.motion_action.flange_id
    p, q = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, i], robot.data.body_quat_w.torch[:, i])
    return torch.cat((p, q), dim=1)


def tcp_twist_base(env):
    robot = env.scene["robot"]
    i = env.motion_action.flange_id
    base_q = robot.data.root_quat_w.torch
    # Base is fixed; no moving-base twist subtraction is needed in this task.
    # body_lin_vel_w is a COM shorthand in this pinned checkout, NOT the
    # flange-origin velocity. Use the actor/link twist consistent with pose/J.
    twist = robot.data.body_link_vel_w.torch[:, i]
    return torch.cat((quat_apply_inverse(base_q, twist[:, :3]),
        quat_apply_inverse(base_q, twist[:, 3:])), dim=1)


def motion_trial_status(env):
    return env.motion_action.fault_code[:, None].float()


def contact_force(env):
    forces = env.scene["arm_contacts"].data.net_forces_w.torch
    return torch.linalg.vector_norm(forces, dim=-1).amax(1, keepdim=True)
