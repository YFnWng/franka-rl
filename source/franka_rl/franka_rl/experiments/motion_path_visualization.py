"""Static, non-colliding scene overlay for Cartesian tracking geometry."""
import numpy as np


def tracking_curve_points(trajectory, count=512):
    """Sample the track only, excluding home approach, settle and return."""
    path = trajectory.path
    return np.asarray([path.evaluate(float(t)).reference.position_m
                       for t in np.linspace(0., path.duration_s, count)])


class MotionPathVisualization:
    """One marker group reused across trials; never updated in the servo loop."""
    def __init__(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/MotionControl/TargetPath",
            markers={"path": sim_utils.SphereCfg(
                radius=.0015,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(.05, .65, 1.), emissive_color=(.02, .15, .25)),
            )},
        ))

    def show(self, trajectory, robot):
        import torch
        from isaaclab.utils.math import quat_apply
        position = robot.data.root_pos_w.torch
        rotation = robot.data.root_quat_w.torch
        points = torch.as_tensor(tracking_curve_points(trajectory),
                                 dtype=position.dtype, device=position.device)
        count = len(points)
        local = points.repeat(len(position), 1)
        world = quat_apply(rotation.repeat_interleave(count, dim=0), local)
        world += position.repeat_interleave(count, dim=0)
        self.markers.visualize(translations=world)
        self.markers.set_visibility(True)
