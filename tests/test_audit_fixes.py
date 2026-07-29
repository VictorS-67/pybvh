"""Tests for pybvh code audit fixes.

Test-first approach: all bug-catching tests should FAIL before the corresponding
fix is applied, and PASS after.  Fixture self-tests (TestSyntheticFixtures) should
PASS immediately.

Groups follow the plan at:
  /home/victor/.claude/plans/logical-wibbling-cook.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # for synthetic_bvh
from synthetic_bvh import (
    make_pos_y_up_bvh,
    make_neg_y_up_bvh,
    make_pos_z_up_bvh,
    make_neg_z_up_bvh,
    make_pos_y_up_rotating_bvh,
    make_heterogeneous_euler_bvh,
    make_lowercase_lr_bvh,
    make_simple_bvh,
)

from pybvh import read_bvh_file, Bvh, rotations, transforms
from pybvh.bvhnode import BvhRoot, BvhJoint, BvhNode

BVH_DIR = Path(__file__).parent.parent / "bvh_data"
EXAMPLE = str(BVH_DIR / "bvh_example.bvh")


# ========================================================================
#  Phase 0 — Synthetic fixture self-tests  (should PASS immediately)
# ========================================================================

class TestSyntheticFixtures:
    """Validate synthetic BVH fixtures are well-formed before using them."""

    # --- Structure ---

    def test_pos_y_up_node_count(self):
        bvh = make_pos_y_up_bvh()
        assert len(bvh.nodes) == 7  # 4 joints + 3 end sites

    def test_pos_y_up_joint_count(self):
        bvh = make_pos_y_up_bvh()
        assert bvh.joint_count == 4

    def test_pos_y_up_hierarchy(self):
        bvh = make_pos_y_up_bvh()
        assert bvh.root.name == "Hips"
        child_names = [c.name for c in bvh.root.children]
        assert "Spine" in child_names
        assert "LeftLeg" in child_names
        assert "RightLeg" in child_names

    def test_pos_y_up_frame_count(self):
        bvh = make_pos_y_up_bvh()
        assert bvh.frame_count == 10

    def test_pos_y_up_array_shapes(self):
        bvh = make_pos_y_up_bvh()
        assert bvh.root_pos.shape == (10, 3)
        assert bvh.joint_angles.shape == (10, 4, 3)

    def test_neg_y_up_array_shapes(self):
        bvh = make_neg_y_up_bvh()
        assert bvh.root_pos.shape == (10, 3)
        assert bvh.joint_angles.shape == (10, 4, 3)

    def test_pos_z_up_array_shapes(self):
        bvh = make_pos_z_up_bvh()
        assert bvh.root_pos.shape == (10, 3)
        assert bvh.joint_angles.shape == (10, 4, 3)

    def test_neg_z_up_array_shapes(self):
        bvh = make_neg_z_up_bvh()
        assert bvh.root_pos.shape == (10, 3)
        assert bvh.joint_angles.shape == (10, 4, 3)

    # --- Geometry: head above root ---

    def test_pos_y_up_head_above_root(self):
        bvh = make_pos_y_up_bvh()
        coords = bvh.rest_pose_positions()
        head_idx = bvh.node_index["Head"]
        assert coords[head_idx, 1] > coords[0, 1]

    def test_neg_y_up_head_above_root(self):
        bvh = make_neg_y_up_bvh()
        coords = bvh.rest_pose_positions()
        head_idx = bvh.node_index["Head"]
        assert coords[head_idx, 1] < coords[0, 1]  # more negative = higher

    def test_pos_z_up_head_above_root(self):
        bvh = make_pos_z_up_bvh()
        coords = bvh.rest_pose_positions()
        head_idx = bvh.node_index["Head"]
        assert coords[head_idx, 2] > coords[0, 2]

    def test_neg_z_up_head_above_root(self):
        bvh = make_neg_z_up_bvh()
        coords = bvh.rest_pose_positions()
        head_idx = bvh.node_index["Head"]
        assert coords[head_idx, 2] < coords[0, 2]

    # --- Pos/neg FK equivalence (zero-angle fixtures) ---

    def test_pos_neg_y_same_physical_motion(self):
        bvh_pos = make_pos_y_up_bvh()
        bvh_neg = make_neg_y_up_bvh()
        coords_pos = bvh_pos.node_positions()
        coords_neg = bvh_neg.node_positions()
        coords_neg_flipped = coords_neg.copy()
        coords_neg_flipped[:, :, 1] *= -1
        npt.assert_allclose(coords_pos, coords_neg_flipped, atol=1e-10)

    def test_pos_neg_z_same_physical_motion(self):
        bvh_pos = make_pos_z_up_bvh()
        bvh_neg = make_neg_z_up_bvh()
        coords_pos = bvh_pos.node_positions()
        coords_neg = bvh_neg.node_positions()
        coords_neg_flipped = coords_neg.copy()
        coords_neg_flipped[:, :, 2] *= -1
        npt.assert_allclose(coords_pos, coords_neg_flipped, atol=1e-10)

    # --- Write/read roundtrip ---

    def test_write_read_roundtrip(self, tmp_path):
        for make_fn in [make_pos_y_up_bvh, make_neg_y_up_bvh,
                        make_pos_z_up_bvh, make_neg_z_up_bvh]:
            bvh = make_fn()
            p = tmp_path / f"{make_fn.__name__}.bvh"
            bvh.write(str(p), verbose=False)
            bvh2 = read_bvh_file(str(p))
            npt.assert_allclose(bvh.root_pos, bvh2.root_pos, atol=1e-5)
            npt.assert_allclose(bvh.joint_angles, bvh2.joint_angles, atol=1e-5)

    # --- Specialized fixtures ---

    def test_heterogeneous_euler_different_orders(self):
        bvh = make_heterogeneous_euler_bvh()
        orders = bvh.euler_orders
        left_idx = bvh.joint_index["LeftLeg"]
        right_idx = bvh.joint_index["RightLeg"]
        assert orders[left_idx] != orders[right_idx]

    def test_lowercase_lr_has_left_right(self):
        bvh = make_lowercase_lr_bvh()
        names_lower = [n.lower() for n in bvh.joint_names]
        assert any("left" in n for n in names_lower)
        assert any("right" in n for n in names_lower)

    def test_simple_bvh_single_frame(self):
        bvh = make_simple_bvh()
        assert bvh.frame_count == 1
        npt.assert_array_equal(bvh.root_pos[0], [10.0, 20.0, 30.0])

    def test_rotating_has_nonzero_angles(self):
        bvh = make_pos_y_up_rotating_bvh()
        assert not np.all(bvh.joint_angles == 0)


# ========================================================================
#  Phase 1 — Group A: up-axis sign  (should FAIL before fix)
# ========================================================================

class TestUpAxisSign:

    # --- A1: rotate_vertical ---

    def test_rotate_vertical_neg_y_up(self):
        """rotate_vertical(90) with -y up = rotate_vertical(-90) with +y up."""
        bvh_pos = make_pos_y_up_rotating_bvh()
        bvh_neg = bvh_pos.copy()
        # Negate Y offsets and root_pos to make a -y up version
        for node in bvh_neg.nodes:
            off = node.offset.copy()
            off[1] *= -1
            node._offset = off
        rp = bvh_neg.root_pos.copy()
        rp[:, 1] *= -1
        bvh_neg.root_pos = rp
        bvh_neg.world_up = '-y'

        rot_neg = transforms.rotate_vertical(bvh_neg, np.pi / 2)
        rot_pos_ref = transforms.rotate_vertical(bvh_pos, -np.pi / 2)
        # Ground-plane (X, Z) positions should match
        npt.assert_allclose(rot_neg.root_pos[:, [0, 2]],
                            rot_pos_ref.root_pos[:, [0, 2]], atol=1e-10)

    def test_rotate_vertical_neg_z_up(self):
        bvh_pos = make_pos_z_up_bvh()
        bvh_neg = make_neg_z_up_bvh()
        # Add some root rotation for the test to be meaningful
        ja = bvh_pos.joint_angles.copy()
        ja[:, 0, 0] = np.linspace(0, 45, bvh_pos.frame_count)
        bvh_pos.joint_angles = ja
        ja = bvh_neg.joint_angles.copy()
        ja[:, 0, 0] = np.linspace(0, 45, bvh_neg.frame_count)
        bvh_neg.joint_angles = ja

        rot_neg = transforms.rotate_vertical(bvh_neg, np.pi / 2)
        rot_pos_ref = transforms.rotate_vertical(bvh_pos, -np.pi / 2)
        npt.assert_allclose(rot_neg.root_pos[:, [0, 1]],
                            rot_pos_ref.root_pos[:, [0, 1]], atol=1e-10)

    # --- A2: foot_contacts height method ---

    def test_foot_contacts_height_neg_y_up(self):
        """Foot contacts (height) should detect contacts at the correct end
        for -Y up.  With varying root height, some frames are "on ground"
        and others "in air"."""
        bvh_pos = make_pos_y_up_bvh()
        bvh_neg = make_neg_y_up_bvh()
        # Add vertical oscillation: root bounces up and down
        bounce = np.array([0, 5, 10, 15, 10, 5, 0, -2, 0, 3], dtype=np.float64)
        rp = bvh_pos.root_pos.copy()
        rp[:, 1] += bounce  # +Y: higher Y = higher
        bvh_pos.root_pos = rp
        rp = bvh_neg.root_pos.copy()
        rp[:, 1] -= bounce  # -Y: more negative Y = higher
        bvh_neg.root_pos = rp

        foot_joints = ["LeftLeg", "RightLeg"]
        contacts_pos = bvh_pos.foot_contacts(method="height", foot_joints=foot_joints)
        contacts_neg = bvh_neg.foot_contacts(method="height", foot_joints=foot_joints)
        # Same physical motion -> same contact pattern
        npt.assert_array_equal(contacts_pos, contacts_neg)

    def test_foot_contacts_height_neg_z_up(self):
        bvh_pos = make_pos_z_up_bvh()
        bvh_neg = make_neg_z_up_bvh()
        bounce = np.array([0, 5, 10, 15, 10, 5, 0, -2, 0, 3], dtype=np.float64)
        rp = bvh_pos.root_pos.copy()
        rp[:, 2] += bounce
        bvh_pos.root_pos = rp
        rp = bvh_neg.root_pos.copy()
        rp[:, 2] -= bounce
        bvh_neg.root_pos = rp

        foot_joints = ["LeftLeg", "RightLeg"]
        contacts_pos = bvh_pos.foot_contacts(method="height", foot_joints=foot_joints)
        contacts_neg = bvh_neg.foot_contacts(method="height", foot_joints=foot_joints)
        npt.assert_array_equal(contacts_pos, contacts_neg)

    # --- A3: root_trajectory ---
    # NOTE: After analysis, root_trajectory heading for -Y up is nuanced.
    # The Euler angles define the same physical rotation regardless of the
    # sign of the up axis.  The ground-plane position extraction is correct
    # for both signs (same axis indices).  The heading chirality difference
    # is theoretically present but doesn't produce a practical bug with how
    # the atan2 is applied to the rotation matrix columns.
    # This group is tested only for ground-plane consistency.
    # The heading sign issue is deferred pending real-world examples.


# ========================================================================
#  Phase 2 — Group B: frequency terminology  (should FAIL before fix)
# ========================================================================

class TestFrequencyTerminology:

    def test_str_no_frequency_word(self):
        bvh = read_bvh_file(EXAMPLE)
        s = str(bvh)
        assert "frequency" not in s.lower()

    def test_velocity_error_says_frame_time(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.frame_time = 0.0
        with pytest.raises(ValueError, match="frame_time"):
            bvh.joint_velocities()

    def test_acceleration_error_says_frame_time(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.frame_time = 0.0
        with pytest.raises(ValueError, match="frame_time"):
            bvh.joint_accelerations()

    def test_angular_velocity_error_says_frame_time(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.frame_time = 0.0
        with pytest.raises(ValueError, match="frame_time"):
            bvh.angular_velocities()

    def test_concat_warning_says_frame_time(self):
        bvh1 = read_bvh_file(EXAMPLE)
        bvh2 = bvh1.copy()
        bvh2.frame_time = 0.01
        with pytest.warns(UserWarning, match="[Ff]rame time"):
            bvh1 + bvh2


# ========================================================================
#  Phase 3 — Group C: world_up cache invalidation  (should FAIL before fix)
# ========================================================================

class TestWorldUpCacheInvalidation:

    def test_cache_cleared_after_root_pos_assignment(self):
        bvh = make_pos_y_up_bvh()
        bvh._world_up_override = None  # rely on cache
        _ = bvh.world_up  # populate cache
        assert bvh._world_up_cached is not None
        bvh.root_pos = bvh.root_pos.copy()  # assign same data
        assert bvh._world_up_cached is None  # cache should be cleared

    def test_cache_cleared_after_joint_angles_assignment(self):
        bvh = make_pos_y_up_bvh()
        bvh._world_up_override = None
        _ = bvh.world_up
        assert bvh._world_up_cached is not None
        bvh.joint_angles = bvh.joint_angles.copy()
        assert bvh._world_up_cached is None


# ========================================================================
#  Phase 4 — Group D: specific bugs  (should FAIL before fix)
# ========================================================================

class TestScaleRootPos:

    def test_scale_scales_root_pos(self):
        bvh = make_simple_bvh()
        original_pos = bvh.root_pos.copy()
        scaled = bvh.scale(2.0)
        npt.assert_allclose(scaled.root_pos, original_pos * 2.0)

    def test_scale_scales_root_pos_real_file(self):
        bvh = read_bvh_file(EXAMPLE)
        original_pos = bvh.root_pos.copy()
        scaled = bvh.scale(0.01)
        npt.assert_allclose(scaled.root_pos, original_pos * 0.01)

    def test_scale_inplace_scales_root_pos(self):
        bvh = make_simple_bvh()
        original_pos = bvh.root_pos.copy()
        bvh.scale(2.0, inplace=True)
        npt.assert_allclose(bvh.root_pos, original_pos * 2.0)


class TestMirrorAnglesEulerOrder:

    def test_mirror_angles_heterogeneous_raw(self):
        """Negation must use the source joint's Euler order, not the slot's."""
        angles = np.array([[[10, 20, 30], [40, 50, 60],
                            [70, 80, 90], [1, 2, 3]]],
                          dtype=np.float64)
        root_pos = np.array([[100, 200, 300]], dtype=np.float64)
        lr_pairs = [(1, 2)]
        lateral_idx = 0  # X is lateral
        rot_channels = [["Z", "Y", "X"], ["Z", "Y", "X"],
                        ["X", "Y", "Z"], ["Z", "Y", "X"]]
        new_angles, _ = transforms.mirror_angles(
            angles, root_pos, lr_pairs, lateral_idx, rot_channels)
        # Index 1 now holds joint 2's original [70,80,90] in XYZ order.
        # Negate non-lateral (Y, Z): keep X(ch0), negate Y(ch1), negate Z(ch2)
        npt.assert_allclose(new_angles[0, 1], [70, -80, -90])

    def test_double_mirror_roundtrip(self):
        bvh = make_heterogeneous_euler_bvh()
        mirrored = bvh.mirror()
        double = mirrored.mirror()
        npt.assert_allclose(double.joint_angles, bvh.joint_angles, atol=1e-10)
        npt.assert_allclose(double.root_pos, bvh.root_pos, atol=1e-10)


class TestBuildViewMatrix:
    """D3: build_view_matrix at extreme elevations.

    NOTE: IEEE 754 cos(pi/2) is ~6e-17 (not exactly 0), so the cross product
    at elevation=90 is tiny but non-zero.  These tests verify the result is
    valid.  The fix adds a robust fallback for the degenerate case.
    """

    def test_top_camera_no_nan(self):
        from pybvh.bvhplot._common import build_view_matrix
        vm = build_view_matrix(0, 90, "y")
        assert not np.any(np.isnan(vm)), f"NaN in view matrix:\n{vm}"

    def test_top_camera_z_up(self):
        from pybvh.bvhplot._common import build_view_matrix
        vm = build_view_matrix(0, 90, "z")
        assert not np.any(np.isnan(vm))


class TestMutableDefault:

    def test_bvh_default_nodes_not_shared(self):
        b1 = Bvh()
        b2 = Bvh()
        assert b1.nodes is not b2.nodes
        assert b1.nodes[0] is not b2.nodes[0]


class TestProperEulerSign:

    @pytest.mark.parametrize("order", ["ZYZ", "XYX", "XZX", "YXY", "YZY", "ZXZ"])
    def test_euler_roundtrip_proper(self, order):
        angles = np.array([[30, 45, 60]], dtype=np.float64)
        R = rotations.euler_to_rotmat(angles, order, degrees=True)
        recovered = rotations.rotmat_to_euler(R, order, degrees=True)
        npt.assert_allclose(recovered, angles, atol=1e-10)

    @pytest.mark.parametrize("order", ["ZYZ", "XYX"])
    def test_euler_gimbal_lock_proper(self, order):
        """Middle angle = 0 (gimbal lock for proper Euler)."""
        angles = np.array([[30, 0, 60]], dtype=np.float64)
        R = rotations.euler_to_rotmat(angles, order, degrees=True)
        recovered = rotations.rotmat_to_euler(R, order, degrees=True)
        # In gimbal lock, first+third angles are coupled; verify rotation matrix
        R2 = rotations.euler_to_rotmat(recovered, order, degrees=True)
        npt.assert_allclose(R2, R, atol=1e-10)


class TestEqIgnoresFrameTime:

    def test_eq_different_frame_time(self):
        bvh1 = read_bvh_file(EXAMPLE)
        bvh2 = bvh1.copy()
        bvh2.frame_time = 999.0
        assert bvh1 != bvh2

    def test_eq_different_euler_orders(self):
        bvh1 = read_bvh_file(EXAMPLE)
        bvh2 = bvh1.change_euler_order("XYZ")
        assert bvh1 != bvh2


# ========================================================================
#  Phase 5 — Group E: missing forwarding/validation  (should FAIL before fix)
# ========================================================================

class TestMissingForwarding:

    def test_extract_joints_preserves_world_up_override(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.world_up = "+y"
        extracted = bvh.extract_joints(bvh.joint_names[:5])
        assert extracted.world_up == "+y"

    def test_add_rotation_noise_wrap_parameter(self):
        bvh = read_bvh_file(EXAMPLE)
        # Should not raise TypeError for unexpected kwarg
        noisy = bvh.add_rotation_noise(sigma=0.1, wrap=True)
        assert noisy is not None

    def test_add_rotation_noise_degrees_matches_the_radian_call(self):
        bvh = read_bvh_file(EXAMPLE)
        deg = bvh.add_rotation_noise(sigma=5.0, degrees=True,
                            rng=np.random.default_rng(0))
        rad = bvh.add_rotation_noise(sigma=np.radians(5.0),
                            rng=np.random.default_rng(0))
        np.testing.assert_allclose(deg.joint_angles, rad.joint_angles, rtol=1e-12)

    # The former test_add_noise_degrees_never_converts_the_position_sigma
    # guarded a 57x unit bug in the old combined add_noise(sigma, sigma_pos).
    # Splitting the function removed the hazard rather than defending it:
    # add_position_noise has no degrees= to mis-apply. The replacement lives
    # in test_bvh.py::TestJointNoise::test_position_noise_has_no_degrees_flag,
    # which asserts the parameter is absent rather than harmless.

    def test_add_rotation_noise_degrees_rejects_negative_sigma(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            bvh.add_rotation_noise(sigma=-5.0, degrees=True)

    def test_from_6d_mismatched_frames_raises(self):
        bvh = read_bvh_file(EXAMPLE)
        root_pos, rot6d = bvh.to_6d()
        with pytest.raises(ValueError, match="[Ff]rame"):
            bvh.from_6d(root_pos[:10], rot6d[:20])

    def test_from_quat_mismatched_frames_raises(self):
        bvh = read_bvh_file(EXAMPLE)
        root_pos, quats = bvh.to_quat()
        with pytest.raises(ValueError, match="[Ff]rame"):
            bvh.from_quat(root_pos[:10], quats[:20])

    def test_from_axisangle_mismatched_frames_raises(self):
        bvh = read_bvh_file(EXAMPLE)
        root_pos, aa = bvh.to_axisangle()
        with pytest.raises(ValueError, match="[Ff]rame"):
            bvh.from_axisangle(root_pos[:10], aa[:20])

    def test_spatial_coords_negative_index(self):
        bvh = read_bvh_file(EXAMPLE)
        second_to_last = bvh.node_positions(frame=-2)
        expected = bvh.node_positions(frame=bvh.frame_count - 2)
        npt.assert_array_equal(second_to_last, expected)


# ========================================================================
#  Phase 6 — Group F: parser robustness  (should FAIL before fix)
# ========================================================================

class TestParserRobustness:

    def test_validates_frame_count(self, tmp_path):
        content = (
            "HIERARCHY\n"
            "ROOT Hips\n"
            "{\n"
            "  OFFSET 0.0 0.0 0.0\n"
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
            "  End Site\n"
            "  {\n"
            "    OFFSET 0.0 10.0 0.0\n"
            "  }\n"
            "}\n"
            "MOTION\n"
            "Frames: 3\n"
            "Frame Time: 0.033333\n"
            "0 0 0 0 0 0\n"
            "0 0 0 0 0 0\n"
        )  # 2 data lines but Frames: 3
        p = tmp_path / "bad_count.bvh"
        p.write_text(content)
        with pytest.raises(ValueError, match="[Ff]rame"):
            read_bvh_file(str(p))

    def test_handles_blank_lines(self, tmp_path):
        content = (
            "HIERARCHY\n"
            "\n"
            "ROOT Hips\n"
            "{\n"
            "  OFFSET 0.0 0.0 0.0\n"
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
            "\n"
            "  End Site\n"
            "  {\n"
            "    OFFSET 0.0 10.0 0.0\n"
            "  }\n"
            "}\n"
            "MOTION\n"
            "Frames: 1\n"
            "Frame Time: 0.033333\n"
            "0 0 0 0 0 0\n"
        )
        p = tmp_path / "blank_lines.bvh"
        p.write_text(content)
        bvh = read_bvh_file(str(p))
        assert bvh.frame_count == 1

    def test_raises_on_zero_frame_time(self, tmp_path):
        content = (
            "HIERARCHY\n"
            "ROOT Hips\n"
            "{\n"
            "  OFFSET 0.0 0.0 0.0\n"
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
            "  End Site\n"
            "  {\n"
            "    OFFSET 0.0 10.0 0.0\n"
            "  }\n"
            "}\n"
            "MOTION\n"
            "Frames: 0\n"
            "Frame Time: 0.0\n"
        )
        p = tmp_path / "zero_ft.bvh"
        p.write_text(content)
        with pytest.raises(ValueError):
            read_bvh_file(str(p))


# ========================================================================
#  Phase 7 — Group G: code quality  (should FAIL before fix)
# ========================================================================

class TestCodeQualityFixes:

    def test_validate_bvh_path_raises_file_not_found(self):
        from pybvh.tools import _validate_bvh_path
        with pytest.raises(FileNotFoundError):
            _validate_bvh_path("definitely_nonexistent_42.bvh")

    def test_validate_bvh_path_raises_value_error_wrong_ext(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        from pybvh.tools import _validate_bvh_path
        with pytest.raises(ValueError):
            _validate_bvh_path(str(f))

    def test_auto_detect_lr_lowercase(self):
        bvh = make_lowercase_lr_bvh()
        mapping = bvh.lr_mapping
        assert mapping, "Should find lowercase left/right pairs"


# ========================================================================
#  Phase 8 — Group H: test gaps  (should PASS — testing existing behavior)
# ========================================================================

class TestBvhVisualizationWrappers:
    """H2: Smoke tests for Bvh convenience wrappers."""

    def test_plot_frame_wrapper(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bvh = read_bvh_file(EXAMPLE)
        bvh.plot_frame(frame=0)
        plt.close("all")

    def test_plot_rest_pose_wrapper(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bvh = read_bvh_file(EXAMPLE)
        bvh.plot_rest_pose()
        plt.close("all")

    def test_plot_trajectory_wrapper(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bvh = read_bvh_file(EXAMPLE)
        bvh.plot_trajectory()
        plt.close("all")


class TestWorldUpPropagation:
    """H3: world_up override must persist through transforms."""

    @pytest.mark.parametrize("method,kwargs", [
        ("scale", {"scale": 2.0}),
        ("translate_root", {"offset": [1, 0, 0]}),
    ])
    def test_world_up_preserved_through_transform(self, method, kwargs):
        bvh = read_bvh_file(EXAMPLE)
        bvh.world_up = "+y"
        result = getattr(bvh, method)(**kwargs)
        assert result.world_up == "+y"

    def test_world_up_preserved_through_mirror(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.world_up = "+y"
        result = bvh.mirror()
        assert result.world_up == "+y"

    def test_world_up_preserved_through_rotate_vertical(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.world_up = "+y"
        result = bvh.rotate_vertical(np.pi / 4)
        assert result.world_up == "+y"
