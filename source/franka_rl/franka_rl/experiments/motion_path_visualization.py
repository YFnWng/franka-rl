"""Static, non-colliding scene overlay for Cartesian tracking geometry."""
import numpy as np


def path_visualization_enabled(sim):
    """Kit visualizers can run with legacy has_gui=False in Isaac Lab 3."""
    return bool(sim.has_gui or any(
        viz.supports_markers() and viz.cfg.enable_markers for viz in sim.visualizers))


def tracking_curve_points(trajectory, count=512):
    """Sample the track only, excluding home approach, settle and return."""
    path = trajectory.path
    return np.asarray([path.evaluate(float(t)).reference.position_m
                       for t in np.linspace(0., path.duration_s, count)])


class MotionPathVisualization:
    """Static target curve and measured TCP axes, outside the 1 kHz servo loop."""
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
        self.markers.set_visibility(False)
        self.frame = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/MotionControl/MeasuredTCPFrame",
            markers={name: sim_utils.CylinderCfg(radius=.0015,height=.06,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color,emissive_color=color))
                for name,color in (("x",(1.,0.,0.)),("y",(0.,1.,0.)),("z",(0.,0.,1.)))},
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
        self.markers.set_visibility(True)
        self.markers.visualize(translations=world)

    def update_frame(self, robot, flange_id):
        """Actual controlled flange frame; identity TCP offset in current tasks."""
        position = robot.data.body_link_pos_w.torch[:,flange_id]
        rotation = robot.data.body_link_quat_w.torch[:,flange_id]
        centers,orientations,indices = frame_marker_poses(position,rotation)
        self.frame.visualize(translations=centers,orientations=orientations,marker_indices=indices)


def frame_marker_poses(position,rotation,length=.06):
    """Cylinder centers/orientations for positive XYZ axes; quaternion order xyzw."""
    import torch
    from isaaclab.utils.math import quat_apply,quat_mul
    offsets = torch.eye(3,device=position.device,dtype=position.dtype)*(length/2)
    s = 2**-.5
    local_q = torch.tensor([[0.,s,0.,s],[-s,0.,0.,s],[0.,0.,0.,1.]],
                           device=rotation.device,dtype=rotation.dtype)
    q = rotation.repeat_interleave(3,dim=0)
    centers = position.repeat_interleave(3,dim=0)+quat_apply(q,offsets.repeat(len(position),1))
    orientations = quat_mul(q,local_q.repeat(len(position),1))
    indices = torch.arange(3,device=position.device,dtype=torch.int32).repeat(len(position))
    return centers,orientations,indices
