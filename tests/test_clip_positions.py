"""The kinematics, facing and descriptor entry points read the clip's positions.

They take no positions from the caller (#127): each reads the node positions
the clip holds as derived state, and positions the caller holds take the
array path (``bvh.index`` then ``pybvh.geometry``).
"""

import pytest
from synthetic_bvh import make_pos_y_up_rotating_bvh

from pybvh import analysis

# Each entry point called on a clip, by the name a user calls it with.
ENTRY_POINTS = {
    "analysis.node_velocities": lambda bvh, **kw: analysis.node_velocities(bvh, **kw),
    "analysis.joint_velocities": lambda bvh, **kw: analysis.joint_velocities(bvh, **kw),
    "analysis.node_accelerations": lambda bvh, **kw: analysis.node_accelerations(bvh, **kw),
    "analysis.joint_accelerations": lambda bvh, **kw: analysis.joint_accelerations(bvh, **kw),
    "analysis.node_speed_derivative": lambda bvh, **kw: analysis.node_speed_derivative(bvh, **kw),
    "analysis.joint_speed_derivative": lambda bvh, **kw: analysis.joint_speed_derivative(bvh, **kw),
    "analysis.node_jerk": lambda bvh, **kw: analysis.node_jerk(bvh, **kw),
    "analysis.joint_jerk": lambda bvh, **kw: analysis.joint_jerk(bvh, **kw),
    "analysis.facing_frame": lambda bvh, **kw: analysis.facing_frame(bvh, **kw),
    "Bvh.node_velocities": lambda bvh, **kw: bvh.node_velocities(**kw),
    "Bvh.joint_velocities": lambda bvh, **kw: bvh.joint_velocities(**kw),
    "Bvh.node_accelerations": lambda bvh, **kw: bvh.node_accelerations(**kw),
    "Bvh.joint_accelerations": lambda bvh, **kw: bvh.joint_accelerations(**kw),
    "Bvh.node_speed_derivative": lambda bvh, **kw: bvh.node_speed_derivative(**kw),
    "Bvh.joint_speed_derivative": lambda bvh, **kw: bvh.joint_speed_derivative(**kw),
    "Bvh.node_jerk": lambda bvh, **kw: bvh.node_jerk(**kw),
    "Bvh.joint_jerk": lambda bvh, **kw: bvh.joint_jerk(**kw),
    "Bvh.facing_frame": lambda bvh, **kw: bvh.facing_frame(**kw),
    "Bvh.forward_at": lambda bvh, **kw: bvh.forward_at(3, **kw),
    "Bvh.left_at": lambda bvh, **kw: bvh.left_at(3, **kw),
    "Bvh.curvature": lambda bvh, **kw: bvh.curvature("LeftFoot", **kw),
    "Bvh.torsion": lambda bvh, **kw: bvh.torsion("LeftFoot", **kw),
    "Bvh.movement_phase": lambda bvh, **kw: bvh.movement_phase("LeftFoot", **kw),
    "Bvh.path_length": lambda bvh, **kw: bvh.path_length("LeftFoot", **kw),
    "Bvh.directness": lambda bvh, **kw: bvh.directness("LeftFoot", **kw),
    "Bvh.ground_path": lambda bvh, **kw: bvh.ground_path("LeftFoot", **kw),
    "Bvh.inter_joint_distance": lambda bvh, **kw: bvh.inter_joint_distance(
        [("LeftFoot", "RightFoot")], **kw
    ),
    "Bvh.joint_angle": lambda bvh, **kw: bvh.joint_angle("Hips", "LeftLeg", "LeftFoot", **kw),
    "Bvh.triangle_area": lambda bvh, **kw: bvh.triangle_area("Head", "LeftFoot", "RightFoot", **kw),
    "Bvh.segment_axis_angle": lambda bvh, **kw: bvh.segment_axis_angle("LeftLeg", "LeftFoot", **kw),
    "Bvh.bounding_box": lambda bvh, **kw: bvh.bounding_box(**kw),
    "Bvh.bounding_sphere": lambda bvh, **kw: bvh.bounding_sphere(**kw),
    "Bvh.bounding_ellipsoid": lambda bvh, **kw: bvh.bounding_ellipsoid(**kw),
    "Bvh.center_of_mass": lambda bvh, **kw: bvh.center_of_mass(**kw),
    "Bvh.com_displacement": lambda bvh, **kw: bvh.com_displacement(**kw),
    "Bvh.verticality": lambda bvh, **kw: bvh.verticality(**kw),
    "Bvh.smoothness": lambda bvh, **kw: bvh.smoothness("LeftFoot", **kw),
    "Bvh.velocity_reductions": lambda bvh, **kw: bvh.velocity_reductions("LeftFoot", **kw),
}


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_passing_coords_raises_type_error(name):
    bvh = make_pos_y_up_rotating_bvh()
    with pytest.raises(TypeError, match="coords"):
        ENTRY_POINTS[name](bvh, coords=bvh.node_positions())
