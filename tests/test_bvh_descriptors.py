"""Tests for the v0.8.0 Bvh motion-descriptor method wrappers (step 6).

Each wrapper is a thin delegation, so the core check is that it equals the
module-level kernel applied to the data it extracts — plus the node-space
resolution contract (end sites are first-class) and a few analytic values.
"""
import numpy as np
import pytest

from pybvh import geometry, analysis
from synthetic_bvh import make_pos_y_up_bvh, make_pos_y_up_rotating_bvh
from pybvh.tools import _axis_to_vector


def _bvh():
    return make_pos_y_up_bvh()


# ----------------------------------------------------------------
#  Trajectory wrappers == kernel on the extracted joint trajectory
# ----------------------------------------------------------------

def test_curvature_torsion_path_directness_match_kernel():
    bvh = _bvh()
    idx = bvh.index("Spine", axis="node")
    traj = bvh.node_positions()[:, idx, :]
    np.testing.assert_allclose(bvh.curvature("Spine"),
                               geometry.curvature(traj, bvh.frame_time))
    np.testing.assert_allclose(bvh.torsion("Spine"),
                               geometry.torsion(traj, bvh.frame_time))
    np.testing.assert_allclose(bvh.path_length("Spine"), geometry.path_length(traj))
    assert np.isnan(bvh.directness("Spine")) == np.isnan(geometry.directness(traj))


def test_ground_path_matches_kernel():
    bvh = _bvh()
    idx = bvh.index("Head", axis="node")
    traj = bvh.node_positions()[:, idx, :]
    up = _axis_to_vector(bvh.world_up)
    got = bvh.ground_path("Head")
    ref = geometry.ground_path(traj, up)
    np.testing.assert_allclose(got.distance, ref.distance)
    np.testing.assert_allclose(got.area, ref.area)


# ----------------------------------------------------------------
#  Relational wrappers resolve names in NODE space (end sites included)
# ----------------------------------------------------------------

def test_inter_joint_distance_by_name_matches_kernel():
    bvh = _bvh()
    pos = bvh.node_positions()
    pairs = [("Hips", "Head"), ("LeftFoot", "RightFoot")]
    idx_pairs = [[bvh.index(a, axis="node"), bvh.index(b, axis="node")] for a, b in pairs]
    np.testing.assert_allclose(bvh.inter_joint_distance(pairs),
                               geometry.inter_joint_distance(pos, idx_pairs))


def test_joint_angle_and_triangle_area_by_name():
    bvh = _bvh()
    pos = bvh.node_positions()
    i = lambda n: bvh.index(n, axis="node")
    np.testing.assert_allclose(
        bvh.joint_angle("LeftFoot", "Hips", "RightFoot"),
        geometry.joint_angle(pos[:, i("LeftFoot")], pos[:, i("Hips")], pos[:, i("RightFoot")]))
    np.testing.assert_allclose(
        bvh.triangle_area("LeftFoot", "Hips", "RightFoot"),
        geometry.triangle_area(pos[:, i("LeftFoot")], pos[:, i("Hips")], pos[:, i("RightFoot")]))


def test_segment_axis_angle_matches_kernel():
    bvh = _bvh()
    pos = bvh.node_positions()
    seg = pos[:, bvh.index("Head", axis="node")] - pos[:, bvh.index("Hips", axis="node")]
    np.testing.assert_allclose(
        bvh.segment_axis_angle("Hips", "Head"),
        geometry.segment_axis_angle(seg, _axis_to_vector(bvh.world_up)))


def test_relational_accepts_int_node_index():
    bvh = _bvh()
    # integer index path must agree with the name path
    np.testing.assert_allclose(bvh.path_length(1), bvh.path_length(bvh.nodes[1].name))


# ----------------------------------------------------------------
#  Bounding / center-of-mass wrappers
# ----------------------------------------------------------------

def test_bounding_and_center_of_mass_wrappers_match_kernel():
    bvh = _bvh()
    pos = bvh.node_positions()
    np.testing.assert_allclose(bvh.bounding_box().volume, geometry.bounding_box(pos).volume)
    np.testing.assert_allclose(bvh.bounding_sphere().radius, geometry.bounding_sphere(pos).radius)
    np.testing.assert_allclose(bvh.center_of_mass(), geometry.center_of_mass(pos))
    np.testing.assert_allclose(bvh.verticality(),
                               geometry.verticality(pos, _axis_to_vector(bvh.world_up)))


def test_com_displacement_defaults_to_first_frame_com():
    bvh = _bvh()
    com = geometry.center_of_mass(bvh.node_positions())
    # default reference is the first-frame CoM (same world frame) -> travel
    np.testing.assert_allclose(bvh.com_displacement(),
                               geometry.com_displacement(com, com[0]))
    assert bvh.com_displacement()[0] == 0.0  # zero displacement at the start


# ----------------------------------------------------------------
#  Analysis wrappers
# ----------------------------------------------------------------

def test_jerk_wrappers_match_module():
    bvh = _bvh()
    np.testing.assert_allclose(bvh.node_jerk(), analysis.node_jerk(bvh))
    np.testing.assert_allclose(bvh.joint_jerk(), analysis.joint_jerk(bvh))


def test_smoothness_wrapper_uses_joint_speed():
    bvh = make_pos_y_up_rotating_bvh()
    idx = bvh.index("LeftFoot", axis="node")
    speed = np.linalg.norm(bvh.node_velocities()[:, idx, :], axis=-1)
    fs = 1.0 / bvh.frame_time
    np.testing.assert_allclose(bvh.smoothness("LeftFoot", metric="number_of_peaks"),
                               analysis.smoothness(speed, fs, metric="number_of_peaks"))


def test_kinetic_energy_and_walking_pace_match_module():
    bvh = _bvh()
    np.testing.assert_allclose(bvh.kinetic_energy(), analysis.kinetic_energy(bvh))
    np.testing.assert_allclose(bvh.walking_pace(), analysis.walking_pace(bvh))


def test_gait_wrappers_pass_foot_joints_through():
    bvh = _bvh()
    feet = ["LeftFoot", "RightFoot"]
    assert bvh.cadence(foot_joints=feet) == analysis.cadence(bvh, foot_joints=feet)
    w, m = bvh.stride_length(foot_joints=feet), analysis.stride_length(bvh, foot_joints=feet)
    assert (np.isnan(w) and np.isnan(m)) or np.isclose(w, m)
    g_method = bvh.gait_parameters(foot_joints=feet)
    g_module = analysis.gait_parameters(bvh, foot_joints=feet)
    assert np.allclose(np.array(g_method), np.array(g_module), equal_nan=True)


def test_gait_standalone_scalars_match_bundle():
    # the unified definition: each standalone scalar equals the bundle field
    bvh = _bvh()
    feet = ["LeftFoot", "RightFoot"]
    g = bvh.gait_parameters(foot_joints=feet)
    np.testing.assert_allclose(bvh.cadence(foot_joints=feet), g.cadence)
    np.testing.assert_allclose(bvh.walking_pace(), g.walking_pace)
    s = bvh.stride_length(foot_joints=feet)
    assert (np.isnan(s) and np.isnan(g.stride_length)) or np.isclose(s, g.stride_length)


def test_gait_wrappers_accept_precomputed_contacts():
    bvh = _bvh()
    feet = ["LeftFoot", "RightFoot"]
    contacts = np.asarray(bvh.foot_contacts(foot_joints=feet))
    np.testing.assert_allclose(
        bvh.cadence(foot_joints=feet, contacts=contacts),
        bvh.cadence(foot_joints=feet))
    s_pre = bvh.stride_length(foot_joints=feet, contacts=contacts)
    s = bvh.stride_length(foot_joints=feet)
    assert (np.isnan(s_pre) and np.isnan(s)) or np.isclose(s_pre, s)


def test_range_of_motion_wrapper_matches_kernel():
    bvh = make_pos_y_up_rotating_bvh()
    jidx = bvh.index("Hips", axis="joint")
    np.testing.assert_allclose(
        bvh.range_of_motion("Hips"),
        analysis.range_of_motion(bvh.joint_angles[:, jidx, :], axis=0))
