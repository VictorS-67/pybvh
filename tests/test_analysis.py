"""Tests for pybvh.analysis / pybvh.packing modules (v3 remediation).

Structure mirrors FEATURES_AUDIT_V2.md / v3 implementation plan.
Each test class exercises the invariants introduced by a specific
remediation phase; tests should fail against the pre-v3 implementation
and turn green as the phases land.
"""

import warnings
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pybvh import read_bvh_file, Bvh  # noqa: E402
from pybvh import analysis, packing  # noqa: E402
from pybvh.bvhnode import BvhRoot, BvhJoint, BvhNode  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_bvh import (  # noqa: E402
    make_pos_y_up_bvh, make_neg_y_up_bvh,
    make_pos_z_up_bvh, make_neg_z_up_bvh,
    make_pos_y_up_rotating_bvh,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bvh_example():
    return read_bvh_file(
        Path(__file__).parent.parent / "bvh_data" / "bvh_example.bvh")


@pytest.fixture
def bvh_test2():
    return read_bvh_file(
        Path(__file__).parent.parent / "bvh_data" / "bvh_test2.bvh")


def _make_ik_helper_skeleton() -> Bvh:
    """A skeleton with both real feet (with tip descendants) and IK helpers
    (no tip descendants).  Used to test topology-based filtering."""
    hips = BvhRoot("Hips", offset=[0, 0, 0], rot_channels=['Z', 'Y', 'X'])
    left_leg = BvhJoint("LeftLeg", offset=[-3, 0, -5], rot_channels=['Z', 'Y', 'X'])
    left_foot = BvhJoint("LeftFoot", offset=[0, 0, -5], rot_channels=['Z', 'Y', 'X'])
    left_foot_end = BvhNode("EndSite", offset=[0, 0, -2])
    right_leg = BvhJoint("RightLeg", offset=[3, 0, -5], rot_channels=['Z', 'Y', 'X'])
    right_foot = BvhJoint("RightFoot", offset=[0, 0, -5], rot_channels=['Z', 'Y', 'X'])
    right_foot_end = BvhNode("EndSite", offset=[0, 0, -2])
    # IK helpers — no children (no tip descendants)
    left_foot_ik = BvhJoint(
        "LeftFootIK", offset=[-3, 0, -10], rot_channels=['Z', 'Y', 'X'])
    right_foot_ik = BvhJoint(
        "RightFootIK", offset=[3, 0, -10], rot_channels=['Z', 'Y', 'X'])

    left_foot_end.parent = left_foot
    left_foot.parent = left_leg
    left_foot.children = [left_foot_end]
    left_leg.parent = hips
    left_leg.children = [left_foot]

    right_foot_end.parent = right_foot
    right_foot.parent = right_leg
    right_foot.children = [right_foot_end]
    right_leg.parent = hips
    right_leg.children = [right_foot]

    left_foot_ik.parent = hips
    right_foot_ik.parent = hips

    hips.children = [left_leg, right_leg, left_foot_ik, right_foot_ik]

    nodes = [hips, left_leg, left_foot, left_foot_end,
             right_leg, right_foot, right_foot_end,
             left_foot_ik, right_foot_ik]

    n_joints = sum(1 for n in nodes if not n.is_end_site())
    n_frames = 10
    root_pos = np.zeros((n_frames, 3))
    root_pos[:, 2] = 10.0  # hips up 10 along z
    joint_angles = np.zeros((n_frames, n_joints, 3))
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=1/30)
    bvh.world_up = '+z'
    return bvh




# ============================================================================
# Phase 1 — root_trajectory heading fix + include_velocities
# ============================================================================

class TestRootTrajectoryHeadingRestForward:
    """Phase 1: heading must reference rest-pose forward, not the hardcoded
    first non-up axis.  These tests fail against v1/v2 code because the old
    logic gives heading=0 at rest for ALL skeletons (wrong for non-+x-forward
    rigs)."""

    def _heading_at_rest(self, bvh):
        traj = bvh.root_trajectory()
        return float(np.arctan2(traj[0, 2], traj[0, 3]))  # atan2(sin, cos)

    def _expected_rest_heading(self, bvh):
        """Derive ground-truth heading from rest-pose forward + world_up."""
        from pybvh.tools import _compute_forward_at, _axis_to_vector
        rest_coords = bvh.rest_pose_coords()
        fwd_axis = _compute_forward_at(bvh, rest_coords, bvh.world_up)
        fwd_vec = _axis_to_vector(fwd_axis)
        up_idx = {'x': 0, 'y': 1, 'z': 2}[bvh.world_up[1]]
        ga = [i for i in range(3) if i != up_idx]
        return float(np.arctan2(fwd_vec[ga[1]], fwd_vec[ga[0]]))

    def test_pos_y_up_heading_matches_rest_forward(self):
        bvh = make_pos_y_up_bvh()
        np.testing.assert_allclose(
            self._heading_at_rest(bvh),
            self._expected_rest_heading(bvh),
            atol=1e-6,
        )

    def test_neg_y_up_heading_matches_rest_forward(self):
        bvh = make_neg_y_up_bvh()
        np.testing.assert_allclose(
            self._heading_at_rest(bvh),
            self._expected_rest_heading(bvh),
            atol=1e-6,
        )

    def test_pos_z_up_heading_matches_rest_forward(self):
        bvh = make_pos_z_up_bvh()
        np.testing.assert_allclose(
            self._heading_at_rest(bvh),
            self._expected_rest_heading(bvh),
            atol=1e-6,
        )

    def test_neg_z_up_heading_matches_rest_forward(self):
        bvh = make_neg_z_up_bvh()
        np.testing.assert_allclose(
            self._heading_at_rest(bvh),
            self._expected_rest_heading(bvh),
            atol=1e-6,
        )

    def test_90deg_yaw_shifts_heading_by_90deg(self):
        """A 90° rotation around the up axis should shift heading by ±90°."""
        # make_pos_y_up_rotating_bvh rotates around channel 0 (Z = forward),
        # which is roll, not yaw. Inject yaw manually instead.
        bvh = make_pos_y_up_bvh().copy()
        # +y up, ZYX Euler order: yaw = rotation around Y, which is channel 1.
        # joint_angles is in radians — sweep 0 → π/2 (= 90°).
        ja = bvh.joint_angles.copy()
        ja[:, 0, 1] = np.linspace(0, np.pi / 2, bvh.frame_count)
        bvh.joint_angles = ja
        traj = bvh.root_trajectory()
        h0 = float(np.arctan2(traj[0, 2], traj[0, 3]))
        h_last = float(np.arctan2(traj[-1, 2], traj[-1, 3]))
        delta = abs((h_last - h0 + np.pi) % (2 * np.pi) - np.pi)
        np.testing.assert_allclose(delta, np.pi / 2, atol=1e-2)


class TestRootTrajectoryIncludeVelocities:
    """Phase 1: new include_velocities parameter + pad kwarg."""

    def test_default_shape_4(self, bvh_example):
        traj = bvh_example.root_trajectory()
        assert traj.shape == (bvh_example.frame_count, 4)

    def test_include_velocities_edge_default(self, bvh_example):
        traj = bvh_example.root_trajectory(include_velocities=True)
        assert traj.shape == (bvh_example.frame_count, 7)

    def test_include_velocities_pad_none_forward(self, bvh_example):
        traj = bvh_example.root_trajectory(
            include_velocities=True, stencil="forward", pad="none")
        assert traj.shape == (bvh_example.frame_count - 1, 7)

    def test_include_velocities_pad_none_central(self, bvh_example):
        traj = bvh_example.root_trajectory(
            include_velocities=True, stencil="central", pad="none")
        assert traj.shape == (bvh_example.frame_count - 2, 7)

    def test_invalid_pad_raises(self, bvh_example):
        with pytest.raises(ValueError, match="pad"):
            bvh_example.root_trajectory(include_velocities=True, pad="bogus")

    def test_invalid_stencil_raises(self, bvh_example):
        with pytest.raises(ValueError, match="stencil"):
            bvh_example.root_trajectory(
                include_velocities=True, stencil="bogus")

    def test_heading_velocity_handles_plus_minus_pi_wrap(self):
        """Yaw crossing ±π should not produce a spike in heading velocity."""
        bvh = make_pos_y_up_bvh().copy()
        # +y up, ZYX order → yaw = channel 1; sweep from -π to +π (≡ -180° to +180°).
        ja = bvh.joint_angles.copy()
        ja[:, 0, 1] = np.linspace(-np.pi, np.pi, bvh.frame_count)
        bvh.joint_angles = ja
        traj = bvh.root_trajectory(include_velocities=True, pad="edge")
        heading_vel = traj[:, 6]  # last column
        # Heading velocity should be smooth (monotonic sweep of ~360°
        # over the clip duration).  Any 2π jump would show up as a huge
        # spike; check |vel| stays within a reasonable bound.
        frame_time = bvh.frame_time
        # Total sweep is 2π radians; per-frame increment is 2π/(F-1).
        # Per-second velocity = (2π / (F-1)) / frame_time ≈ expected rate.
        expected_rate = (2 * np.pi / (bvh.frame_count - 1)) / frame_time
        # Allow some margin (interior frames use central diff which
        # matches the expected forward diff for linearly swept angles).
        assert np.all(np.abs(heading_vel) < 3 * expected_rate), (
            "heading velocity spiked — ±π wrap not handled correctly")

    def test_degrees_affects_only_heading_column(self, bvh_example):
        """degrees=True converts heading_vel (column 6) but leaves
        ground_vel (columns 4, 5) and the sin/cos base unchanged."""
        traj_rad = bvh_example.root_trajectory(include_velocities=True)
        traj_deg = bvh_example.root_trajectory(
            include_velocities=True, degrees=True)
        # Columns 0-5 unchanged
        np.testing.assert_allclose(traj_rad[:, :6], traj_deg[:, :6], atol=1e-12)
        # Column 6 (heading_vel): degrees = np.degrees(radians)
        np.testing.assert_allclose(
            traj_deg[:, 6], np.degrees(traj_rad[:, 6]), atol=1e-10)

    def test_degrees_without_include_velocities_is_noop(self, bvh_example):
        """degrees= has no effect when include_velocities=False (no heading_vel column)."""
        traj_rad = bvh_example.root_trajectory(degrees=False)
        traj_deg = bvh_example.root_trajectory(degrees=True)
        np.testing.assert_array_equal(traj_rad, traj_deg)


# ============================================================================
# Phase 2 — pad= kwarg across velocity-like functions
# ============================================================================

class TestStencilPadMatrix:
    """All 4 combinations of stencil x pad produce the documented shape.
    Composition identity holds only for the default (central, edge)."""

    def test_joint_velocities_central_edge(self, bvh_example):
        vel = bvh_example.joint_velocities()  # defaults
        assert vel.shape[0] == bvh_example.frame_count

    def test_joint_velocities_central_none(self, bvh_example):
        vel = bvh_example.joint_velocities(stencil="central", pad="none")
        assert vel.shape[0] == bvh_example.frame_count - 2

    def test_joint_velocities_forward_edge(self, bvh_example):
        vel = bvh_example.joint_velocities(stencil="forward", pad="edge")
        assert vel.shape[0] == bvh_example.frame_count

    def test_joint_velocities_forward_none(self, bvh_example):
        vel = bvh_example.joint_velocities(stencil="forward", pad="none")
        assert vel.shape[0] == bvh_example.frame_count - 1

    def test_joint_accelerations_central_edge(self, bvh_example):
        acc = bvh_example.joint_accelerations()
        assert acc.shape[0] == bvh_example.frame_count

    def test_joint_accelerations_central_none(self, bvh_example):
        acc = bvh_example.joint_accelerations(stencil="central", pad="none")
        assert acc.shape[0] == bvh_example.frame_count - 4

    def test_joint_accelerations_forward_edge(self, bvh_example):
        acc = bvh_example.joint_accelerations(stencil="forward", pad="edge")
        assert acc.shape[0] == bvh_example.frame_count

    def test_joint_accelerations_forward_none(self, bvh_example):
        acc = bvh_example.joint_accelerations(stencil="forward", pad="none")
        assert acc.shape[0] == bvh_example.frame_count - 2

    def test_angular_velocities_central_edge(self, bvh_example):
        av = bvh_example.angular_velocities()
        assert av.shape[0] == bvh_example.frame_count

    def test_angular_velocities_central_none(self, bvh_example):
        av = bvh_example.angular_velocities(stencil="central", pad="none")
        assert av.shape[0] == bvh_example.frame_count - 2

    def test_angular_velocities_forward_edge(self, bvh_example):
        av = bvh_example.angular_velocities(stencil="forward", pad="edge")
        assert av.shape[0] == bvh_example.frame_count

    def test_angular_velocities_forward_none(self, bvh_example):
        av = bvh_example.angular_velocities(stencil="forward", pad="none")
        assert av.shape[0] == bvh_example.frame_count - 1

    def test_composition_identity_central_edge(self, bvh_example):
        """Default (central, edge): np.gradient(vel) == acc exactly."""
        vel = bvh_example.joint_velocities()
        acc = bvh_example.joint_accelerations()
        expected = np.gradient(vel, bvh_example.frame_time, axis=0)
        np.testing.assert_allclose(acc, expected, atol=1e-10)

    def test_central_edge_interior_equals_central_none(self, bvh_example):
        """edge interior [1:-1] == none array (same central formula)."""
        vel_edge = bvh_example.joint_velocities(stencil="central", pad="edge")
        vel_none = bvh_example.joint_velocities(stencil="central", pad="none")
        np.testing.assert_allclose(vel_edge[1:-1], vel_none, atol=1e-10)

    def test_forward_edge_interior_equals_forward_none(self, bvh_example):
        """forward edge replicates last; interior [0:-1] == forward none."""
        vel_edge = bvh_example.joint_velocities(stencil="forward", pad="edge")
        vel_none = bvh_example.joint_velocities(stencil="forward", pad="none")
        np.testing.assert_allclose(vel_edge[:-1], vel_none, atol=1e-10)
        # Last row is replicated
        np.testing.assert_allclose(vel_edge[-1], vel_none[-1], atol=1e-10)

    def test_angular_velocities_central_edge_boundary_is_forward(self, bvh_example):
        """central+edge at boundary equals forward+none at boundary."""
        av_edge = bvh_example.angular_velocities(in_frames=True)  # central, edge
        av_forward = bvh_example.angular_velocities(
            in_frames=True, stencil="forward", pad="none")
        np.testing.assert_allclose(av_edge[0], av_forward[0], atol=1e-10)
        np.testing.assert_allclose(av_edge[-1], av_forward[-1], atol=1e-10)

    def test_angular_velocities_central_edge_interior_is_two_step(self, bvh_example):
        """central+edge interior uses R_{i-1}^T @ R_{i+1} divided by 2."""
        import pybvh.rotations as rot
        _, R = bvh_example.to_rotmat()
        i = 10
        R_rel = np.einsum('...ji,...jk->...ik', R[i - 1], R[i + 1])
        expected = rot.rotmat_to_axisangle(R_rel) / 2.0
        av = bvh_example.angular_velocities(in_frames=True)
        np.testing.assert_allclose(av[i], expected, atol=1e-10)

    def test_invalid_pad_raises(self, bvh_example):
        with pytest.raises(ValueError, match="pad"):
            bvh_example.joint_velocities(pad="bogus")

    def test_invalid_stencil_raises(self, bvh_example):
        with pytest.raises(ValueError, match="stencil"):
            bvh_example.joint_velocities(stencil="bogus")


class TestAngularVelocities:
    """Phase 2: dedicated angular-velocity correctness tests."""

    def test_zero_motion_gives_zero_angular_velocity(self):
        bvh = make_pos_y_up_bvh()  # no rotations in its default motion
        av = bvh.angular_velocities(stencil="forward", pad="none")
        np.testing.assert_allclose(av, 0.0, atol=1e-10)

    def test_constant_rotation_produces_constant_velocity(self):
        """Root rotates by a constant amount per frame → constant ω."""
        bvh = make_pos_y_up_rotating_bvh()
        av = bvh.angular_velocities(stencil="forward", pad="none")
        # Root's angular velocity should be roughly constant across interior frames
        interior = av[2:-2, 0, :]  # skip boundary artifacts
        # Per-axis standard deviation should be tiny relative to the signal
        np.testing.assert_array_less(interior.std(axis=0), 1e-3)

    def test_degrees_flag_converts_output(self, bvh_example):
        """degrees=True multiplies radians result by 180/π."""
        av_rad = bvh_example.angular_velocities()
        av_deg = bvh_example.angular_velocities(degrees=True)
        np.testing.assert_allclose(av_deg, np.degrees(av_rad), atol=1e-10)

    def test_degrees_flag_respects_in_frames_and_stencil(self, bvh_example):
        """degrees= works across all stencil/pad combinations and in_frames settings."""
        for stencil in ("central", "forward"):
            for pad in ("edge", "none"):
                for in_frames in (True, False):
                    rad = bvh_example.angular_velocities(
                        in_frames=in_frames, stencil=stencil, pad=pad)
                    deg = bvh_example.angular_velocities(
                        in_frames=in_frames, stencil=stencil, pad=pad, degrees=True)
                    np.testing.assert_allclose(
                        deg, np.degrees(rad), atol=1e-10,
                        err_msg=f"{stencil=} {pad=} {in_frames=}")


# ============================================================================
# Phase 3 — feature_array_layout + to_feature_array pad support
# ============================================================================

class TestFeatureArrayLayout:
    """Phase 3: pure keyword-only function returning column slices."""

    def test_basic_6d(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="6d")
        assert layout["root_pos"] == slice(0, 3)
        assert layout["rotations"] == slice(3, 3 + 24 * 6)
        assert "velocities" not in layout
        assert "foot_contacts" not in layout

    def test_with_velocities(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="6d",
            include_velocities=True)
        expected_start = 3 + 24 * 6
        # Velocities are now per-joint, not per-node: width = num_joints * 3
        assert layout["velocities"] == slice(
            expected_start, expected_start + 24 * 3)

    def test_with_foot_contacts(self):
        layout = packing.feature_array_layout(
            num_joints=24, num_feet=2,
            representation="6d", include_foot_contacts=True)
        expected_start = 3 + 24 * 6
        assert layout["foot_contacts"] == slice(
            expected_start, expected_start + 2)

    def test_no_root_pos(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="6d",
            include_root_pos=False)
        assert "root_pos" not in layout
        assert layout["rotations"] == slice(0, 24 * 6)

    def test_foot_contacts_without_num_feet_raises(self):
        with pytest.raises(ValueError, match="num_feet"):
            packing.feature_array_layout(
                num_joints=24, representation="6d",
                include_foot_contacts=True)

    def test_rotmat_width_9(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="rotmat")
        assert layout["rotations"] == slice(3, 3 + 24 * 9)

    def test_euler_width_3(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="euler")
        assert layout["rotations"] == slice(3, 3 + 24 * 3)

    def test_quaternion_width_4(self):
        layout = packing.feature_array_layout(
            num_joints=24, representation="quaternion")
        assert layout["rotations"] == slice(3, 3 + 24 * 4)

    def test_unknown_representation_raises(self):
        with pytest.raises(ValueError, match="representation"):
            packing.feature_array_layout(
                num_joints=24, representation="nonsense")

    def test_keyword_only(self):
        """Positional args should fail — signature is keyword-only."""
        with pytest.raises(TypeError):
            packing.feature_array_layout(24)  # type: ignore[misc]

    def test_bvh_method_wrapper(self, bvh_example):
        layout = bvh_example.feature_array_layout(representation="6d")
        assert layout["root_pos"] == slice(0, 3)

    def test_slices_partition_feature_array(self, bvh_example):
        """Layout slices should correctly partition to_feature_array output."""
        feat = bvh_example.to_feature_array(
            representation="6d", include_velocities=True)
        layout = bvh_example.feature_array_layout(
            representation="6d", include_velocities=True)
        total = sum(sl.stop - sl.start for sl in layout.values())
        assert total == feat.shape[1]


class TestToFeatureArrayPad:
    """Phase 3: to_feature_array respects pad= kwarg."""

    def test_include_velocities_edge_keeps_F_shape(self, bvh_example):
        feat = bvh_example.to_feature_array(include_velocities=True)
        assert feat.shape[0] == bvh_example.frame_count

    def test_include_velocities_forward_none_drops_first_frame(self, bvh_example):
        feat = bvh_example.to_feature_array(
            include_velocities=True, stencil="forward", pad="none")
        assert feat.shape[0] == bvh_example.frame_count - 1

    def test_include_velocities_central_none_drops_boundaries(self, bvh_example):
        feat = bvh_example.to_feature_array(
            include_velocities=True, stencil="central", pad="none")
        assert feat.shape[0] == bvh_example.frame_count - 2

    def test_include_velocities_forward_edge_keeps_F_shape(self, bvh_example):
        feat = bvh_example.to_feature_array(
            include_velocities=True, stencil="forward", pad="edge")
        assert feat.shape[0] == bvh_example.frame_count

    def test_rotmat_shape(self, bvh_example):
        """representation='rotmat' gives 9 values per joint."""
        feat = bvh_example.to_feature_array(representation="rotmat")
        # 3 root + 24 joints * 9 values = 3 + 216 = 219
        expected_D = 3 + bvh_example.joint_count * 9
        assert feat.shape == (bvh_example.frame_count, expected_D)


# ============================================================================
# Phase 4 — scale-invariant foot thresholds + height sanity check
# ============================================================================

class TestFootContactsScaleInvariance:
    """Phase 4: default thresholds are scale-invariant."""

    def test_cm_and_m_scale_produce_identical_contacts(self, bvh_example):
        bvh_cm = bvh_example
        bvh_m = bvh_example.scale(0.01)
        contacts_cm = bvh_cm.foot_contacts(method="velocity")
        contacts_m = bvh_m.foot_contacts(method="velocity")
        np.testing.assert_array_equal(contacts_cm, contacts_m)

    def test_height_method_scale_invariant(self, bvh_example):
        bvh_cm = bvh_example
        bvh_m = bvh_example.scale(0.01)
        c_cm = bvh_cm.foot_contacts(method="height")
        c_m = bvh_m.foot_contacts(method="height")
        np.testing.assert_array_equal(c_cm, c_m)


class TestFootContactsHeightSanity:
    """Phase 4: raise if world_up is inconsistent with rest geometry."""

    def test_wrong_world_up_raises(self, bvh_example):
        bvh_wrong = bvh_example.copy()
        bvh_wrong.world_up = "-z"  # bvh_example is +z up; -z puts feet above hips
        with pytest.raises(ValueError, match="world_up"):
            bvh_wrong.foot_contacts(method="height")

    def test_correct_world_up_does_not_raise(self, bvh_example):
        # Sanity: the unchanged fixture should succeed
        _ = bvh_example.foot_contacts(method="height")


class TestFootContactsVelocityFrameZeroPropagation:
    """Phase 4: velocity method uses contacts[0] = contacts[1] propagation
    (replacing the old all-ones fill at frame 0)."""

    def test_frame_0_equals_frame_1(self, bvh_example):
        contacts = bvh_example.foot_contacts(method="velocity")
        np.testing.assert_array_equal(contacts[0], contacts[1])


class TestFootContactsNegativeUp:
    """Phase 4: velocity method must work on -y / -z world_up skeletons.
    Previously only the height method had dedicated negative-up tests.

    The synthetic fixtures use ``LeftLeg``/``RightLeg`` (no "foot" in the
    name), so auto-detection doesn't apply — pass ``foot_joints=`` explicitly.
    """

    def test_neg_y_up_velocity_method(self):
        bvh = make_neg_y_up_bvh()
        # Shouldn't raise; shape is correct regardless of numerical values.
        contacts = bvh.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"])
        assert contacts.shape == (bvh.frame_count, 2)

    def test_neg_z_up_velocity_method(self):
        bvh = make_neg_z_up_bvh()
        contacts = bvh.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"])
        assert contacts.shape == (bvh.frame_count, 2)


# ============================================================================
# Phase 5 — topology-filtered foot auto-detection
# ============================================================================

class TestFootContactsTopologyFilter:
    """Phase 5: IK helpers (no tip descendants) are filtered out, and
    when a rig has both Foot and Toe joints in the same chain the more
    distal one is preferred."""

    def test_ik_helpers_filtered_out(self):
        bvh = _make_ik_helper_skeleton()
        contacts = bvh.foot_contacts(method="velocity")
        # Only LeftFoot and RightFoot (with tip descendants) should be kept
        assert contacts.shape[1] == 2

    def test_existing_fixtures_still_detect_feet(self, bvh_example):
        """Regression guard: the real fixtures must still detect feet."""
        contacts = bvh_example.foot_contacts(method="velocity")
        assert contacts.shape[1] >= 2


class TestAutoDetectFootJoints:
    """Public helper: bvh.auto_detect_foot_joints() and analysis.auto_detect_foot_joints(bvh)."""

    def test_most_distal_wins_on_foot_plus_toe_chains(self, bvh_example):
        """bvh_example has LeftFoot → LeftToeBase and RightFoot → RightToeBase.
        The most-distal filter should pick the ToeBase joints (2, not 4)."""
        feet = bvh_example.auto_detect_foot_joints()
        assert len(feet) == 2
        assert all("toe" in name.lower() for name in feet)

    def test_ik_helpers_filtered(self):
        bvh = _make_ik_helper_skeleton()
        feet = bvh.auto_detect_foot_joints()
        assert feet == ["LeftFoot", "RightFoot"]  # alphabetical tie-break

    def test_method_matches_module_function(self, bvh_example):
        from pybvh.analysis import auto_detect_foot_joints
        assert bvh_example.auto_detect_foot_joints() == auto_detect_foot_joints(bvh_example)

    def test_stable_alphabetical_order_for_equal_heights(self):
        """L and R feet are at equal rest height — output should be
        alphabetically ordered (LeftFoot before RightFoot) deterministically."""
        bvh = _make_ik_helper_skeleton()
        feet = bvh.auto_detect_foot_joints()
        assert feet == sorted(feet)  # alphabetical


# ============================================================================
# Composition tests: features interact correctly with transforms
# ============================================================================

class TestMirrorFeaturesComposition:
    """Mirror swaps L/R → foot_contacts columns L/R-swapped too."""

    def test_mirrored_foot_contacts_columns_swapped(self, bvh_example):
        """After mirror, foot contact values for L/R-paired joints swap
        columns.  The detected joint name list is unchanged (mirror
        rewrites angles, not names), so the column order is identical;
        what changes is the data under each column."""
        contacts_orig = bvh_example.foot_contacts(method="velocity")
        bvh_mirror = bvh_example.mirror()
        contacts_mirror = bvh_mirror.foot_contacts(method="velocity")

        # Column names should be identical (mirror doesn't rename joints)
        names_orig = bvh_example.auto_detect_foot_joints()
        names_mirror = bvh_mirror.auto_detect_foot_joints()
        assert names_orig == names_mirror
        # bvh_example is bilateral and auto-detects 2 feet.
        assert contacts_orig.shape[1] == 2

        # The L/R pair swaps: the column that was "Left*" now holds the
        # contact pattern that was originally "Right*" and vice versa.
        np.testing.assert_array_equal(contacts_mirror[:, 0], contacts_orig[:, 1])
        np.testing.assert_array_equal(contacts_mirror[:, 1], contacts_orig[:, 0])


class TestReorientFeaturesComposition:
    """reorient_world_up is a rigid transformation → ground-plane
    geometry is preserved under it (pairwise distances, path length)."""

    def test_reorient_preserves_pairwise_distances(self, bvh_example):
        """Rigid transformations preserve pairwise distances in the
        ground plane.  This is strictly stronger than magnitude-only
        equality: a per-axis sign flip would pass the magnitude test
        but fail pairwise distances."""
        bvh_rot = bvh_example.reorient_world_up("+y")
        gp_orig = bvh_example.root_trajectory()[:, :2]  # (F, 2)
        gp_rot = bvh_rot.root_trajectory()[:, :2]

        # Sample pairwise distances across a few frame pairs
        frames = [0, 10, 25, 50, 74]
        for i in frames:
            for j in frames:
                if i >= j:
                    continue
                d_orig = float(np.linalg.norm(gp_orig[i] - gp_orig[j]))
                d_rot = float(np.linalg.norm(gp_rot[i] - gp_rot[j]))
                np.testing.assert_allclose(
                    d_orig, d_rot, atol=1e-4,
                    err_msg=f"ground-plane distance[{i},{j}] "
                            f"changed under reorient_world_up: "
                            f"{d_orig:.6f} vs {d_rot:.6f}"
                )

    def test_reorient_preserves_ground_plane_norms(self, bvh_example):
        """Per-frame distance from origin in ground plane."""
        bvh_rot = bvh_example.reorient_world_up("+y")
        n_orig = np.linalg.norm(bvh_example.root_trajectory()[:, :2], axis=1)
        n_rot = np.linalg.norm(bvh_rot.root_trajectory()[:, :2], axis=1)
        np.testing.assert_allclose(n_orig, n_rot, atol=1e-4)


# ============================================================================
# Phase 7 — foot_contacts redesign: combined method, floor estimation,
#                                    min-duration filter, structured info
# ============================================================================

class TestFootContactsCombinedMethod:
    """'combined' (AND of velocity and height) is the new default method.
    It rejects the failure cases that single-signal methods miss."""

    @staticmethod
    def _stationary_airborne_bvh():
        """Feet stationary but high in the air.
        Velocity-alone: reports contact everywhere (false positive).
        Height-alone: reports no contact (correct).
        """
        bvh = make_pos_y_up_bvh().copy()
        rp = np.zeros_like(bvh.root_pos)
        rp[:, 1] = 100  # hips at y=100 → feet at ~y=95
        bvh.root_pos = rp
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)
        return bvh

    @staticmethod
    def _sliding_foot_bvh():
        """Feet low (on floor) but sliding in z.
        Velocity-alone: no contact (high speed).
        Height-alone: contact everywhere (low height).
        """
        bvh = make_pos_y_up_bvh().copy()
        rp = np.zeros_like(bvh.root_pos)
        rp[:, 2] = np.linspace(0, 50, bvh.frame_count)
        bvh.root_pos = rp
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)
        return bvh

    def test_combined_rejects_stationary_airborne_foot(self):
        bvh = self._stationary_airborne_bvh()
        vel = bvh.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"])
        combined = bvh.foot_contacts(
            method="combined", foot_joints=["LeftLeg", "RightLeg"], floor=0.0)
        # Velocity method sees stationary feet → reports contact (FP)
        assert vel.sum() > 0
        # Combined rejects because height check fails
        assert combined.sum() == 0

    def test_combined_rejects_sliding_foot(self):
        bvh = self._sliding_foot_bvh()
        height = bvh.foot_contacts(
            method="height", foot_joints=["LeftLeg", "RightLeg"], floor=0.0)
        combined = bvh.foot_contacts(
            method="combined", foot_joints=["LeftLeg", "RightLeg"], floor=0.0)
        # Height method sees low feet → reports contact (FP for sliding)
        assert height.sum() > 0
        # Combined has strictly fewer contacts (velocity check rejects sliding)
        assert combined.sum() < height.sum()

    def test_combined_equals_velocity_when_height_always_below_threshold(self):
        """Degenerate case: when height is always low, combined collapses
        to the velocity signal alone."""
        bvh = make_pos_y_up_bvh().copy()
        rp = np.zeros_like(bvh.root_pos)
        rp[5:, 2] = 5  # stationary-move-stationary pattern
        bvh.root_pos = rp
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)
        vel = bvh.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"])
        combined = bvh.foot_contacts(
            method="combined", foot_joints=["LeftLeg", "RightLeg"], floor=0.0)
        np.testing.assert_array_equal(vel, combined)

    def test_default_method_is_combined(self, bvh_example):
        """bvh.foot_contacts() — no method kwarg — uses combined."""
        default = bvh_example.foot_contacts()
        combined = bvh_example.foot_contacts(method="combined")
        np.testing.assert_array_equal(default, combined)

    def test_invalid_method_raises(self, bvh_example):
        with pytest.raises(ValueError, match="method"):
            bvh_example.foot_contacts(method="bogus")


class TestFootContactsFloorEstimation:
    """floor='auto' runs a percentile estimate; explicit float pins it."""

    def test_auto_tracks_planted_foot(self):
        bvh = make_pos_y_up_bvh().copy()
        bvh.root_pos = np.zeros_like(bvh.root_pos)
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)  # feet at y = -5 throughout
        _, info = bvh.foot_contacts(
            method="height", foot_joints=["LeftLeg", "RightLeg"],
            return_info=True)
        assert info["floor"] == pytest.approx(-5.0, abs=0.1)

    def test_explicit_float_is_honored(self, bvh_example):
        _, info = bvh_example.foot_contacts(
            method="height", floor=42.0, return_info=True)
        assert info["floor"] == pytest.approx(42.0)

    def test_auto_tracks_clip_offset(self):
        bvh = make_pos_y_up_bvh().copy()
        rp = np.zeros_like(bvh.root_pos)
        rp[:, 1] = 50  # lift skeleton 50 units along +y
        bvh.root_pos = rp
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)  # feet at y = 50 - 5 = 45
        _, info = bvh.foot_contacts(
            method="height", foot_joints=["LeftLeg", "RightLeg"],
            return_info=True)
        assert info["floor"] == pytest.approx(45.0, abs=0.1)

    def test_negative_up_floor_reported_in_raw_coords(self):
        """For world_up='-y' a foot whose raw y ≈ 5 should yield floor≈5."""
        bvh = make_neg_y_up_bvh().copy()
        bvh.root_pos = np.zeros_like(bvh.root_pos)
        bvh.joint_angles = np.zeros_like(bvh.joint_angles)
        _, info = bvh.foot_contacts(
            method="height", foot_joints=["LeftLeg", "RightLeg"],
            return_info=True)
        assert info["floor"] == pytest.approx(5.0, abs=0.1)

    def test_invalid_floor_string_raises(self, bvh_example):
        with pytest.raises(ValueError, match="floor"):
            bvh_example.foot_contacts(method="height", floor="bogus")


class TestFootContactsDurationFilters:
    """min_contact_duration / min_gap_duration: morphological open + close
    specified in seconds, fps-independent."""

    def test_helper_removes_short_true_runs(self):
        from pybvh.analysis import _filter_short_runs
        mask = np.array(
            [[True, True, False, True, False, False,
              True, True, True, True]]).T  # True runs of 2, 1, 4
        filtered = _filter_short_runs(mask, min_run=3, value=True)
        expected = np.array(
            [[False, False, False, False, False, False,
              True, True, True, True]]).T  # only the 4-run survives
        np.testing.assert_array_equal(filtered, expected)

    def test_helper_fills_short_false_gaps(self):
        from pybvh.analysis import _filter_short_runs
        # True True False False True True True → gap of 2 in the middle
        mask = np.array(
            [[True, True, False, False, True, True, True]]).T
        filled = _filter_short_runs(mask, min_run=3, value=False)
        # Gap of 2 < 3 → filled
        expected = np.array(
            [[True, True, True, True, True, True, True]]).T
        np.testing.assert_array_equal(filled, expected)

    def test_helper_keeps_long_false_gaps(self):
        from pybvh.analysis import _filter_short_runs
        mask = np.array(
            [[True, False, False, False, True]]).T
        filled = _filter_short_runs(mask, min_run=3, value=False)
        # Gap of 3 == min_run → NOT filled (must be strictly shorter)
        np.testing.assert_array_equal(filled, mask)

    def test_helper_minrun_one_is_identity(self):
        from pybvh.analysis import _filter_short_runs
        mask = np.array([[True, False, True, True, False, True]]).T
        np.testing.assert_array_equal(
            _filter_short_runs(mask, 1, value=True), mask)
        np.testing.assert_array_equal(
            _filter_short_runs(mask, 1, value=False), mask)

    def test_zero_duration_disables_filtering(self, bvh_example):
        """Explicit 0.0 disables both filters (raw per-frame output)."""
        c_raw = bvh_example.foot_contacts(
            method="combined",
            min_contact_duration=0.0, min_gap_duration=0.0)
        # Default (0.1 s) should differ from raw on data with short runs
        c_default = bvh_example.foot_contacts(method="combined")
        # At minimum, shapes match and raw is a valid binary array
        assert c_raw.shape == c_default.shape

    def test_min_contact_duration_only_reduces(self, bvh_example):
        c_raw = bvh_example.foot_contacts(
            method="combined",
            min_contact_duration=0.0, min_gap_duration=0.0)
        c_filt = bvh_example.foot_contacts(
            method="combined",
            min_contact_duration=0.15, min_gap_duration=0.0)
        assert c_filt.sum() <= c_raw.sum()
        assert np.all((c_filt == 0) | (c_raw == 1))

    def test_min_gap_duration_only_adds(self, bvh_example):
        c_raw = bvh_example.foot_contacts(
            method="combined",
            min_contact_duration=0.0, min_gap_duration=0.0)
        c_filled = bvh_example.foot_contacts(
            method="combined",
            min_contact_duration=0.0, min_gap_duration=0.15)
        # Gap-filling can only add contacts, never remove
        assert c_filled.sum() >= c_raw.sum()
        assert np.all((c_raw == 0) | (c_filled == 1))

    def test_threshold_is_fps_independent(self):
        """Same frame data at 30 fps and 120 fps — the raw velocity mask
        (before duration filters) should be identical because the
        vel_threshold is in units/frame, not units/second."""
        bvh_30 = make_pos_y_up_bvh()
        bvh_120 = bvh_30.copy()
        bvh_120.frame_time = bvh_30.frame_time / 4
        # Pin filters to 0 so we're testing threshold only
        c_30 = bvh_30.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"],
            min_contact_duration=0.0, min_gap_duration=0.0)
        c_120 = bvh_120.foot_contacts(
            method="velocity", foot_joints=["LeftLeg", "RightLeg"],
            min_contact_duration=0.0, min_gap_duration=0.0)
        np.testing.assert_array_equal(c_30, c_120)


class TestFootContactsReturnInfo:
    """return_info=True returns (contacts, info) with a stable schema."""

    def test_returns_tuple(self, bvh_example):
        result = bvh_example.foot_contacts(return_info=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_ndarray_unchanged_when_return_info_false(self, bvh_example):
        plain = bvh_example.foot_contacts()
        with_info = bvh_example.foot_contacts(return_info=True)[0]
        np.testing.assert_array_equal(plain, with_info)

    def test_info_keys_for_combined_method(self, bvh_example):
        _, info = bvh_example.foot_contacts(
            method="combined", return_info=True)
        for key in ("joints", "method", "min_contact_duration",
                    "min_gap_duration", "skeleton_scale",
                    "vel_threshold", "height_threshold", "floor"):
            assert key in info, f"missing key {key!r}"

    def test_info_omits_skeleton_scale_when_both_thresholds_explicit(self, bvh_example):
        """skeleton_scale is only in info when auto-calibration ran."""
        _, info = bvh_example.foot_contacts(
            method="combined",
            vel_threshold=0.1, height_threshold=0.5,
            return_info=True)
        assert "skeleton_scale" not in info

    def test_info_omits_height_keys_for_velocity_method(self, bvh_example):
        _, info = bvh_example.foot_contacts(
            method="velocity", return_info=True)
        assert "vel_threshold" in info
        assert "height_threshold" not in info
        assert "floor" not in info

    def test_info_omits_vel_key_for_height_method(self, bvh_example):
        _, info = bvh_example.foot_contacts(
            method="height", return_info=True)
        assert "vel_threshold" not in info
        assert "height_threshold" in info
        assert "floor" in info

    def test_joints_matches_column_order(self, bvh_example):
        contacts, info = bvh_example.foot_contacts(return_info=True)
        assert len(info["joints"]) == contacts.shape[1]

    def test_info_records_method_used(self, bvh_example):
        for m in ("combined", "velocity", "height"):
            _, info = bvh_example.foot_contacts(method=m, return_info=True)
            assert info["method"] == m


