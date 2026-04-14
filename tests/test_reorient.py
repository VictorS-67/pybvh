"""Tests for world_up features: load-time parameter, property validation,
reorient_world_up, reorient_rest_up, reorient_rest_forward, warning toggle.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_bvh import (
    make_pos_y_up_bvh, make_neg_y_up_bvh,
    make_pos_z_up_bvh, make_neg_z_up_bvh,
    make_pos_y_up_rotating_bvh, make_simple_bvh,
    make_disagreement_bvh,
)

from pybvh import read_bvh_file, Bvh, transforms
from pybvh.tools import (
    _axis_aligned_rotation, _axis_to_vector,
    _rest_upward, _compute_forward_at,
)

BVH_DIR = Path(__file__).parent.parent / "bvh_data"
EXAMPLE = str(BVH_DIR / "bvh_example.bvh")
TEST3 = str(BVH_DIR / "bvh_test3.bvh")  # disagreement file


# ========================================================================
#  Feature 2: Property shape validation
# ========================================================================

class TestPropertyValidation:

    def test_root_pos_rejects_1d(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="root_pos"):
            bvh.root_pos = np.zeros(10)

    def test_root_pos_rejects_wrong_columns(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="root_pos"):
            bvh.root_pos = np.zeros((10, 4))

    def test_joint_angles_rejects_2d(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="joint_angles"):
            bvh.joint_angles = np.zeros((10, 3))

    def test_joint_angles_rejects_wrong_last_dim(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="joint_angles"):
            bvh.joint_angles = np.zeros((10, 24, 2))

    def test_valid_shapes_accepted(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.root_pos = np.zeros((5, 3))
        bvh.joint_angles = np.zeros((5, 24, 3))
        assert bvh.frame_count == 5

    def test_empty_shapes_accepted(self):
        bvh = Bvh()
        assert bvh.root_pos.shape == (0, 3)
        assert bvh.joint_angles.shape == (0, 0, 3)


# ========================================================================
#  Features 1+6: world_up parameter + warning toggle
# ========================================================================

class TestWorldUpParameter:

    def test_read_bvh_file_with_world_up(self):
        bvh = read_bvh_file(EXAMPLE, world_up="+y")
        assert bvh.world_up == "+y"

    def test_read_bvh_file_auto_default(self):
        bvh = read_bvh_file(EXAMPLE)
        assert bvh.world_up in {"+x", "-x", "+y", "-y", "+z", "-z"}

    def test_read_bvh_file_invalid_raises(self):
        with pytest.raises(ValueError):
            read_bvh_file(EXAMPLE, world_up="bad")

    def test_read_bvh_file_override_suppresses_warning(self):
        """Loading bvh_test3 with explicit world_up should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh = read_bvh_file(TEST3, world_up="+z")
            user_warns = [x for x in w if issubclass(x.category, UserWarning)
                          and "world up" in str(x.message).lower()]
            assert len(user_warns) == 0

    def test_read_bvh_file_auto_warns_on_disagreement(self):
        """Loading bvh_test3 without override should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh = read_bvh_file(TEST3)
            user_warns = [x for x in w if issubclass(x.category, UserWarning)
                          and "world up" in str(x.message).lower()]
            assert len(user_warns) > 0

    def test_warn_toggle_suppresses(self):
        """warn_on_world_up_disagreement=False suppresses the warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh = read_bvh_file(TEST3, warn_on_world_up_disagreement=False)
            user_warns = [x for x in w if issubclass(x.category, UserWarning)
                          and "world up" in str(x.message).lower()]
            assert len(user_warns) == 0

    def test_bvh_init_world_up_param(self):
        bvh = Bvh(world_up="+y")
        assert bvh.world_up == "+y"

    def test_read_bvh_directory_world_up(self):
        from pybvh import read_bvh_directory
        clips = read_bvh_directory(BVH_DIR, world_up="+y")
        assert all(c.world_up == "+y" for c in clips)


# ========================================================================
#  Helper: _axis_aligned_rotation
# ========================================================================

class TestAxisAlignedRotation:

    @pytest.mark.parametrize("from_ax,to_ax", [
        ("+y", "+z"), ("+z", "+y"), ("+y", "-y"), ("+z", "-z"),
        ("+y", "+x"), ("-z", "+y"), ("+x", "-x"), ("-y", "+z"),
        ("+x", "+z"), ("-x", "-y"),
    ])
    def test_maps_axis_correctly(self, from_ax, to_ax):
        R = _axis_aligned_rotation(from_ax, to_ax)
        result = R @ _axis_to_vector(from_ax)
        expected = _axis_to_vector(to_ax)
        npt.assert_allclose(result, expected, atol=1e-14)

    @pytest.mark.parametrize("ax", ["+x", "+y", "+z", "-x", "-y", "-z"])
    def test_identity_for_same_axis(self, ax):
        R = _axis_aligned_rotation(ax, ax)
        npt.assert_allclose(R, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("from_ax,to_ax", [
        ("+y", "+z"), ("+y", "-y"), ("+x", "+z"),
    ])
    def test_det_is_one(self, from_ax, to_ax):
        R = _axis_aligned_rotation(from_ax, to_ax)
        assert abs(np.linalg.det(R) - 1.0) < 1e-14

    @pytest.mark.parametrize("from_ax,to_ax", [
        ("+y", "+z"), ("+y", "-y"), ("-z", "+x"),
    ])
    def test_entries_are_exact_integers(self, from_ax, to_ax):
        R = _axis_aligned_rotation(from_ax, to_ax)
        npt.assert_array_equal(R, R.astype(int))


# ========================================================================
#  Feature 3: reorient_world_up
# ========================================================================

class TestReorientWorldUp:

    def test_noop_same_axis(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_world_up(bvh, "+y")
        npt.assert_allclose(result.root_pos, bvh.root_pos)
        npt.assert_allclose(result.joint_angles, bvh.joint_angles)

    def test_metadata_updated(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_world_up(bvh, "+z")
        assert result.world_up == "+z"

    def test_fk_positions_rotated(self):
        """FK positions in new system = R @ FK positions in old system."""
        bvh = make_pos_y_up_bvh()
        R = _axis_aligned_rotation("+y", "+z")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_fk_positions_rotated_with_animation(self):
        """Same test with rotating BVH (non-zero root angles)."""
        bvh = make_pos_y_up_rotating_bvh()
        R = _axis_aligned_rotation("+y", "+z")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_roundtrip_fk_exact(self):
        """Y->Z->Y round-trip: FK positions must match original."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.spatial_coords()
        bvh_z = transforms.reorient_world_up(bvh, "+z")
        bvh_back = transforms.reorient_world_up(bvh_z, "+y")
        coords_after = bvh_back.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_180_degree_rotation(self):
        """+Y -> -Y: FK Y and X should be negated."""
        bvh = make_pos_y_up_bvh()
        R = _axis_aligned_rotation("+y", "-y")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "-y")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_negative_axes(self):
        """-Z -> +Y reorientation."""
        bvh = make_neg_z_up_bvh()
        R = _axis_aligned_rotation("-z", "+y")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "+y")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_inplace(self):
        bvh = make_pos_y_up_bvh()
        coords_before = bvh.spatial_coords()
        R = _axis_aligned_rotation("+y", "+z")
        result = transforms.reorient_world_up(bvh, "+z", inplace=True)
        assert result is None
        assert bvh.world_up == "+z"
        coords_after = bvh.spatial_coords()
        for f in range(coords_before.shape[0]):
            expected = (R @ coords_before[f].T).T
            npt.assert_allclose(coords_after[f], expected, atol=1e-10)

    def test_write_read_roundtrip(self, tmp_path):
        """Reorient, write, read back, compare FK."""
        bvh = make_pos_y_up_rotating_bvh()
        reoriented = transforms.reorient_world_up(bvh, "+z")
        coords_expected = reoriented.spatial_coords()
        p = tmp_path / "reoriented.bvh"
        reoriented.write(str(p), verbose=False)
        bvh2 = read_bvh_file(str(p), world_up="+z")
        coords_loaded = bvh2.spatial_coords()
        npt.assert_allclose(coords_loaded, coords_expected, atol=1e-4)

    def test_real_file(self):
        """Reorient a real BVH file (+z -> +y)."""
        bvh = read_bvh_file(EXAMPLE)
        R = _axis_aligned_rotation(bvh.world_up, "+y")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "+y")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_y_up_bvh()
        result = bvh.reorient_world_up("+z")
        assert result.world_up == "+z"


# ========================================================================
#  Feature 4: reorient_rest_up
# ========================================================================

class TestReorientRestUp:

    def test_fk_invariance(self):
        """FK positions must be identical before and after."""
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_up(bvh, "+y")
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_fk_invariance_with_rotation(self):
        """Same with rotating BVH."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_up(bvh, "+z")
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_rest_up_changes(self):
        bvh = make_pos_z_up_bvh()
        result = transforms.reorient_rest_up(bvh, "+y")
        assert _rest_upward(result) == "+y"

    def test_root_pos_unchanged(self):
        bvh = make_pos_z_up_bvh()
        original_pos = bvh.root_pos.copy()
        result = transforms.reorient_rest_up(bvh, "+y")
        npt.assert_allclose(result.root_pos, original_pos)

    def test_noop_same_axis(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_rest_up(bvh, "+y")
        npt.assert_allclose(result.joint_angles, bvh.joint_angles)

    def test_roundtrip_fk(self):
        """Z->Y->Z round-trip: FK positions must match."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.spatial_coords()
        bvh2 = transforms.reorient_rest_up(bvh, "+z")
        bvh3 = transforms.reorient_rest_up(bvh2, "+y")
        coords_after = bvh3.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_real_file(self):
        """Reorient a real BVH file's rest pose."""
        bvh = read_bvh_file(EXAMPLE)
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_up(bvh, "+y")
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)
        assert _rest_upward(result) == "+y"

    def test_inplace(self):
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_up(bvh, "+y", inplace=True)
        assert result is None
        coords_after = bvh.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.spatial_coords()
        result = bvh.reorient_rest_up("+y")
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Feature 5: reorient_rest_forward
# ========================================================================

class TestReorientRestForward:

    def test_fk_invariance(self):
        bvh = make_pos_y_up_bvh()
        coords_before = bvh.spatial_coords()
        # Get current forward, pick a different one
        rest = bvh.rest_pose_coords()
        current_fwd = _compute_forward_at(bvh, rest, bvh.world_up)
        # Pick target that's perpendicular to up and different from current
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd
                  and c[1] != bvh.world_up[1]][0]
        result = transforms.reorient_rest_forward(bvh, target)
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_fk_invariance_with_rotation(self):
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.spatial_coords()
        rest = bvh.rest_pose_coords()
        current_fwd = _compute_forward_at(bvh, rest, bvh.world_up)
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd
                  and c[1] != bvh.world_up[1]][0]
        result = transforms.reorient_rest_forward(bvh, target)
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_parallel_axis_raises(self):
        bvh = make_pos_y_up_bvh()
        with pytest.raises(ValueError, match="parallel"):
            transforms.reorient_rest_forward(bvh, "+y")

    def test_inplace(self):
        bvh = make_pos_y_up_bvh()
        rest = bvh.rest_pose_coords()
        current_fwd = _compute_forward_at(bvh, rest, bvh.world_up)
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd
                  and c[1] != bvh.world_up[1]][0]
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_forward(bvh, target, inplace=True)
        assert result is None
        coords_after = bvh.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_y_up_bvh()
        rest = bvh.rest_pose_coords()
        current_fwd = _compute_forward_at(bvh, rest, bvh.world_up)
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd
                  and c[1] != bvh.world_up[1]][0]
        coords_before = bvh.spatial_coords()
        result = bvh.reorient_rest_forward(target)
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Heterogeneous Euler order tests
# ========================================================================

class TestReorientHeterogeneousEulerOrders:
    """Verify reorient functions work when joints have different Euler orders."""

    def test_reorient_world_up_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh
        bvh = make_heterogeneous_euler_bvh()
        R = _axis_aligned_rotation(bvh.world_up, "+z")
        coords_old = bvh.spatial_coords()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.spatial_coords()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_reorient_rest_up_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh
        bvh = make_heterogeneous_euler_bvh()
        coords_before = bvh.spatial_coords()
        result = transforms.reorient_rest_up(bvh, "+z")
        coords_after = result.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_reorient_world_up_roundtrip_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh
        bvh = make_heterogeneous_euler_bvh()
        coords_before = bvh.spatial_coords()
        bvh2 = transforms.reorient_world_up(bvh, "+z")
        bvh3 = transforms.reorient_world_up(bvh2, "+y")
        coords_after = bvh3.spatial_coords()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Disagreement fixture tests
# ========================================================================

class TestDisagreementFixture:

    def test_disagreement_bvh_constructed(self):
        bvh = make_disagreement_bvh()
        assert bvh.frame_count > 0
        # Rest-pose topology should be +Z up (offsets along Z)
        rest_up = _rest_upward(bvh)
        assert rest_up[1] == "z"
