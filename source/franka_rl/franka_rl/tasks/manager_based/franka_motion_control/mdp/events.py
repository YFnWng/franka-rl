import torch


def reset_motion_state(env, env_ids):
    """Exact configured home; no randomization and no inherited terminal joints."""
    robot = env.scene["robot"]
    q = robot.data.default_joint_pos.torch[env_ids].clone()
    dq = torch.zeros_like(q)
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)
