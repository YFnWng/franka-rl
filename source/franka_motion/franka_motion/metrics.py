"""Portable offline geometric metrics; tracking error is not contouring error."""
import numpy as np
from .contracts import finite_vector,positive


def circle_contour_error(position_m,center_m,radius_m,plane_rotation):
    """Exact distance to a 3D circle, independent of phase/time or closest waypoint."""
    center = np.array(finite_vector(center_m,3,"center_m"))
    radius = positive(radius_m,"radius_m")
    position,rotation = np.asarray(position_m),np.asarray(plane_rotation)
    if position.shape[-1] != 3 or not np.isfinite(position).all() or rotation.shape != (3,3):
        raise ValueError("Expected finite XYZ samples and a 3x3 plane rotation")
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rotation),1.):
        raise ValueError("Plane rotation must be proper orthonormal")
    local = (position-center)@rotation
    return np.sqrt((np.linalg.norm(local[...,:2],axis=-1)-radius)**2+local[...,2]**2)


def sampled_curve_contour_error(position_m,curve_points_m):
    """Nearest sampled path distance, an upper approximation, not exact contouring."""
    positions,curve = np.asarray(position_m),np.asarray(curve_points_m)
    if positions.shape[-1] != 3 or curve.ndim != 2 or curve.shape[1] != 3 or not len(curve):
        raise ValueError("Expected XYZ positions and a nonempty sampled XYZ curve")
    if not np.isfinite(positions).all() or not np.isfinite(curve).all():
        raise ValueError("Curve samples must be finite")
    flat = positions.reshape(-1,3)
    result = np.empty(len(flat))
    for i in range(0,len(flat),64):
        distances = np.sum((flat[i:i+64,None,:]-curve[None,:,:])**2,axis=-1)
        result[i:i+64] = np.sqrt(distances.min(axis=1))
    return result.reshape(positions.shape[:-1])
