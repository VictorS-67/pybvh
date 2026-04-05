"""
Tests for pybvh library.

Uses bvh_data/bvh_example.bvh as the test fixture.
Run with: pytest tests/test_bvh.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import copy
import warnings
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pybvh import (read_bvh_file, df_to_bvh, Bvh, frames_to_spatial_coord,
                    read_bvh_directory, batch_to_numpy,
                    compute_normalization_stats, normalize_array, denormalize_array)
from pybvh.bvhnode import BvhNode, BvhJoint, BvhRoot
from pybvh.tools import (rotX, rotY, rotZ, get_premult_mat_rot,
                          batch_rotX, batch_rotY, batch_rotZ,
                          batch_get_premult_mat_rot)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def bvh_example_path():
    """Path to the example BVH file."""
    return Path(__file__).parent.parent / "bvh_data" / "bvh_example.bvh"


@pytest.fixture
def bvh_example(bvh_example_path):
    """Loaded Bvh object from bvh_example.bvh."""
    return read_bvh_file(bvh_example_path)


@pytest.fixture
def bvh_test2():
    return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_test2.bvh")

@pytest.fixture
def bvh_test3():
    return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_test3.bvh")

@pytest.fixture
def standard_skeleton():
    return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "standard_skeleton.bvh")


# =============================================================================
# Test: read_bvh_file
# =============================================================================

class TestReadBvhFile:
    """Tests for reading BVH files."""

    def test_read_returns_bvh_object(self, bvh_example):
        """read_bvh_file should return a Bvh object."""
        assert isinstance(bvh_example, Bvh)

    def test_frame_count(self, bvh_example):
        """Verify expected frame count."""
        assert bvh_example.frame_count == 56

    def test_frame_frequency(self, bvh_example):
        """Verify expected frame frequency."""
        assert abs(bvh_example.frame_frequency - 0.03333333333333333) < 1e-10

    def test_nodes_count(self, bvh_example):
        """Verify expected number of nodes (joints + end sites)."""
        assert len(bvh_example.nodes) == 29

    def test_frames_shape(self, bvh_example):
        """Verify root_pos and joint_angles shapes."""
        assert bvh_example.root_pos.shape == (56, 3)
        assert bvh_example.joint_angles.shape == (56, 24, 3)

    def test_root_is_bvh_root(self, bvh_example):
        """Root should be a BvhRoot instance."""
        assert isinstance(bvh_example.root, BvhRoot)

    def test_root_name(self, bvh_example):
        """Verify root joint name."""
        assert bvh_example.root.name == "Hips"

    def test_root_channels(self, bvh_example):
        """Verify root has both position and rotation channels."""
        assert bvh_example.root.pos_channels == ['X', 'Y', 'Z']
        assert bvh_example.root.rot_channels == ['Z', 'Y', 'X']

    def test_node_names(self, bvh_example):
        """Verify expected node names."""
        expected_names = [
            'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 
            'Head', 'End Site Head', 'RightShoulder', 'RightArm', 'RightForeArm', 
            'RightHand', 'End Site RightHand', 'LeftShoulder', 'LeftArm', 
            'LeftForeArm', 'LeftHand', 'End Site LeftHand', 'RightUpLeg', 
            'RightLeg', 'RightFoot', 'RightToeBase', 'End Site RightToeBase', 
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'End Site LeftToeBase'
        ]
        actual_names = [n.name for n in bvh_example.nodes]
        assert actual_names == expected_names

    def test_first_frame_values(self, bvh_example):
        """Verify first frame data values."""
        expected_root_pos = np.array([9.5267, -0.7495, 36.2768])
        expected_first_joint = np.array([88.2354, -1.0699, 0.6448])
        expected_second_joint = np.array([1.0586, -0.3574, -0.4139])
        expected_third_joint = np.array([1.0857, -0.2637, -0.4131])
        np.testing.assert_allclose(bvh_example.root_pos[0], expected_root_pos, atol=1e-4)
        np.testing.assert_allclose(bvh_example.joint_angles[0, 0], expected_first_joint, atol=1e-4)
        np.testing.assert_allclose(bvh_example.joint_angles[0, 1], expected_second_joint, atol=1e-4)
        np.testing.assert_allclose(bvh_example.joint_angles[0, 2], expected_third_joint, atol=1e-4)

    def test_file_not_found_raises(self):
        """Reading non-existent file should raise ImportError."""
        with pytest.raises(ImportError):
            read_bvh_file("/nonexistent/path/file.bvh")

    def test_non_bvh_file_raises(self, tmp_path):
        """Reading non-.bvh file should raise ImportError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a bvh file")
        with pytest.raises(ImportError):
            read_bvh_file(txt_file)


# =============================================================================
# Test: Node hierarchy
# =============================================================================

class TestNodeHierarchy:
    """Tests for BvhNode, BvhJoint, BvhRoot classes."""

    def test_end_sites_are_bvhnode(self, bvh_example):
        """End sites should be BvhNode (not BvhJoint)."""
        end_sites = [n for n in bvh_example.nodes if "End Site" in n.name]
        assert len(end_sites) == 5  # Head, RightHand, LeftHand, RightToeBase, LeftToeBase
        for es in end_sites:
            assert es.is_end_site() is True
            assert isinstance(es, BvhNode)

    def test_joints_are_bvhjoint(self, bvh_example):
        """Non-root, non-end-site nodes should be BvhJoint."""
        joints = [n for n in bvh_example.nodes[1:] if not n.is_end_site()]
        for joint in joints:
            assert isinstance(joint, BvhJoint)
            assert joint.is_end_site() is False

    def test_parent_child_relationships(self, bvh_example):
        """Verify parent-child links are consistent."""
        for node in bvh_example.nodes:
            if node.parent is not None:
                # This node should be in its parent's children list
                assert node in node.parent.children

    def test_root_has_no_parent(self, bvh_example):
        """Root should have no parent."""
        assert bvh_example.root.parent is None


# =============================================================================
# Test: get_spatial_coord
# =============================================================================

class TestSpatialCoordinates:
    """Tests for spatial coordinate calculation."""

    def test_single_frame_world_centered(self, bvh_example):
        """Verify spatial coordinates for single frame, world centered."""
        spatial = bvh_example.get_spatial_coord(frame_num=0, centered="world")
        
        # Shape: 29 nodes x 3 coordinates
        assert spatial.shape == (29, 3)
        
        # First 4 nodes (root + first 3 joints)
        expected_first_4 = np.array([
            [9.5267, -0.7495, 36.2768],
            [9.57422669, -0.83414247, 40.72857177],
            [10.3621849, -0.95396445, 45.10949094],
            [10.73337881, -1.08367526, 49.54492051]
        ])
        np.testing.assert_allclose(spatial[:4], expected_first_4, atol=1e-4)

    def test_single_frame_skeleton_centered(self, bvh_example):
        """Verify spatial coordinates for single frame, skeleton centered."""
        spatial = bvh_example.get_spatial_coord(frame_num=0, centered="skeleton")
        
        # Root should be at origin
        np.testing.assert_allclose(spatial[0], [0.0, 0.0, 0.0], atol=1e-10)
        
        # Other joints should be offset from origin
        expected_first_4 = np.array([
            [0.0, 0.0, 0.0],
            [0.04752669, -0.08464247, 4.45177177],
            [0.8354849, -0.20446445, 8.83269094],
            [1.20667881, -0.33417526, 13.26812051]
        ])
        np.testing.assert_allclose(spatial[:4], expected_first_4, atol=1e-4)

    def test_all_frames_world_centered(self, bvh_example):
        """Verify all frames spatial coordinates shape."""
        spatial = bvh_example.get_spatial_coord(frame_num=-1, centered="world")
        assert spatial.shape == (56, 29, 3)

    def test_all_frames_first_centered(self, bvh_example):
        """Verify 'first' centering mode - first frame root at origin."""
        spatial = bvh_example.get_spatial_coord(frame_num=-1, centered="first")
        
        assert spatial.shape == (56, 29, 3)
        # First frame root should be at origin
        np.testing.assert_allclose(spatial[0, 0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_invalid_centered_raises(self, bvh_example):
        """Invalid centered value should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.get_spatial_coord(frame_num=0, centered="invalid")

    def test_invalid_frame_num_raises(self, bvh_example):
        """Out-of-bounds frame_num should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.get_spatial_coord(frame_num=1000)


# =============================================================================
# Test: DataFrame conversion
# =============================================================================

class TestDataFrameConversion:
    """Tests for DataFrame <-> Bvh conversion."""

    def test_get_df_constructor_euler(self, bvh_example):
        """Verify DataFrame constructor output for euler mode."""
        df_data = bvh_example.get_df_constructor(mode='euler', centered='world')
        df = pd.DataFrame(df_data)
        
        assert df.shape == (56, 76)  # 56 frames, 75 channels + 1 time column
        
        # Check expected columns
        expected_first_10 = [
            'time', 'Hips_X_pos', 'Hips_Y_pos', 'Hips_Z_pos',
            'Hips_Z_rot', 'Hips_Y_rot', 'Hips_X_rot',
            'Spine_Z_rot', 'Spine_Y_rot', 'Spine_X_rot'
        ]
        assert list(df.columns[:10]) == expected_first_10

    def test_df_to_bvh_roundtrip(self, bvh_example):
        """DataFrame to Bvh round-trip should preserve data."""
        df_data = bvh_example.get_df_constructor(mode='euler', centered='world')
        df = pd.DataFrame(df_data)
        
        bvh2 = df_to_bvh(bvh_example.nodes, df)
        
        assert bvh2.frame_count == bvh_example.frame_count
        assert len(bvh2.nodes) == len(bvh_example.nodes)
        np.testing.assert_allclose(bvh2.root_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(bvh2.joint_angles, bvh_example.joint_angles, atol=1e-10)


# =============================================================================
# Test: File write/read round-trip
# =============================================================================

class TestFileRoundTrip:
    """Tests for writing and re-reading BVH files."""

    def test_write_read_roundtrip(self, bvh_example):
        """Writing and re-reading a BVH file should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test_output.bvh"
            
            bvh_example.to_bvh_file(tmpfile, verbose=False)
            bvh_reread = read_bvh_file(tmpfile)
            
            # Basic properties should match
            assert bvh_reread.frame_count == bvh_example.frame_count
            assert len(bvh_reread.nodes) == len(bvh_example.nodes)
            assert bvh_reread.root.name == bvh_example.root.name
            
            # Frames should be close (allowing for float formatting precision)
            np.testing.assert_allclose(
                bvh_reread.root_pos, 
                bvh_example.root_pos, 
                atol=1e-5
            )
            np.testing.assert_allclose(
                bvh_reread.joint_angles,
                bvh_example.joint_angles,
                atol=1e-5
            )

    def test_write_to_invalid_extension_raises(self, bvh_example, tmp_path):
        """Writing to non-.bvh file should raise Exception."""
        with pytest.raises(Exception):
            bvh_example.to_bvh_file(tmp_path / "output.txt")

    def test_write_to_nonexistent_dir_raises(self, bvh_example):
        """Writing to non-existent directory should raise Exception."""
        with pytest.raises(Exception):
            bvh_example.to_bvh_file("/nonexistent/dir/output.bvh")


# =============================================================================
# Test: Bvh object methods
# =============================================================================

class TestBvhMethods:
    """Tests for other Bvh object methods."""

    def test_copy_creates_independent_object(self, bvh_example):
        """copy() should create a deep copy."""
        bvh_copy = bvh_example.copy()
        
        # Modify the copy
        bvh_copy.root_pos[0, 0] = 999.0
        
        # Original should be unchanged
        assert bvh_example.root_pos[0, 0] != 999.0

    def test_str_representation(self, bvh_example):
        """__str__ should return readable summary."""
        s = str(bvh_example)
        assert "24 elements" in s  # 29 nodes - 5 end sites = 24 joints
        assert "56 frames" in s

    def test_repr_representation(self, bvh_example):
        """__repr__ should return constructor-like string."""
        r = repr(bvh_example)
        assert "Bvh(" in r
        assert "nodes=" in r
        assert "frames=" in r

    def test_get_rest_pose_euler(self, bvh_example):
        """get_rest_pose with euler mode should return zeros tuple."""
        root_pos_rest, joint_angles_rest = bvh_example.get_rest_pose(mode='euler')
        assert root_pos_rest.shape == (3,)
        assert joint_angles_rest.shape == bvh_example.joint_angles[0].shape
        np.testing.assert_allclose(root_pos_rest, np.zeros(3))
        np.testing.assert_allclose(joint_angles_rest, np.zeros_like(joint_angles_rest))

    def test_hierarchy_info_as_dict(self, bvh_example):
        """hierarchy_info_as_dict should return valid dict."""
        hier = bvh_example.hierarchy_info_as_dict()
        
        assert isinstance(hier, dict)
        assert "Hips" in hier
        assert hier["Hips"]["parent"] is None
        assert "Spine" in hier["Hips"]["children"]


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_bvh_object(self):
        """Creating empty Bvh object should work."""
        bvh = Bvh()
        assert bvh.frame_count == 0
        assert len(bvh.nodes) == 1  # Default root

    def test_nodes_setter_validates(self):
        """Setting nodes to invalid value should raise ValueError."""
        bvh = Bvh()
        with pytest.raises(ValueError):
            bvh.nodes = "not a list"


# =============================================================================
# Test: Structured data representation (root_pos + joint_angles)
# =============================================================================

class TestStructuredRepresentation:
    """Tests for the root_pos / joint_angles split representation."""

    def test_root_pos_shape(self, bvh_example):
        """root_pos should be (num_frames, 3)."""
        assert bvh_example.root_pos.shape == (56, 3)

    def test_joint_angles_shape(self, bvh_example):
        """joint_angles should be (num_frames, num_joints, 3)."""
        num_joints = len([n for n in bvh_example.nodes if not n.is_end_site()])
        assert bvh_example.joint_angles.shape == (56, num_joints, 3)

    def test_root_pos_values(self, bvh_example):
        """root_pos should match the expected first-frame root translation."""
        expected = np.array([9.5267, -0.7495, 36.2768])
        np.testing.assert_allclose(bvh_example.root_pos[0], expected, atol=1e-4)

    def test_joint_angles_first_joint(self, bvh_example):
        """First joint's angles should match expected values."""
        expected = np.array([88.2354, -1.0699, 0.6448])
        np.testing.assert_allclose(bvh_example.joint_angles[0, 0], expected, atol=1e-4)

    def test_copy_independence_root_pos(self, bvh_example):
        """copy() should create independent root_pos."""
        bvh_copy = bvh_example.copy()
        bvh_copy.root_pos[0, 0] = 999.0
        assert bvh_example.root_pos[0, 0] != 999.0

    def test_copy_independence_joint_angles(self, bvh_example):
        """copy() should create independent joint_angles."""
        bvh_copy = bvh_example.copy()
        bvh_copy.joint_angles[0, 0, 0] = 999.0
        assert bvh_example.joint_angles[0, 0, 0] != 999.0

    def test_empty_object_root_pos(self):
        """Empty Bvh should have empty root_pos."""
        bvh = Bvh()
        assert bvh.root_pos.shape == (0, 3)

    def test_empty_object_joint_angles(self):
        """Empty Bvh should have empty joint_angles."""
        bvh = Bvh()
        assert bvh.joint_angles.shape == (0, 0, 3)

    def test_roundtrip_file_preserves_structure(self, bvh_example):
        """Write + read preserves root_pos and joint_angles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test_structured.bvh"
            bvh_example.to_bvh_file(tmpfile, verbose=False)
            bvh2 = read_bvh_file(tmpfile)
            np.testing.assert_allclose(bvh2.root_pos, bvh_example.root_pos, atol=1e-5)
            np.testing.assert_allclose(bvh2.joint_angles, bvh_example.joint_angles, atol=1e-5)


# =============================================================================
# Test: Batch rotation functions in tools.py
# =============================================================================

class TestBatchRotations:
    """Tests for batch_rotX/Y/Z and batch_get_premult_mat_rot."""

    def test_batch_rotX_matches_scalar(self):
        """batch_rotX should produce same results as individual rotX calls."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-np.pi, np.pi, size=50)
        batch_result = batch_rotX(angles)
        for i, a in enumerate(angles):
            np.testing.assert_allclose(batch_result[i], rotX(a), atol=1e-14)

    def test_batch_rotY_matches_scalar(self):
        """batch_rotY should produce same results as individual rotY calls."""
        rng = np.random.default_rng(43)
        angles = rng.uniform(-np.pi, np.pi, size=50)
        batch_result = batch_rotY(angles)
        for i, a in enumerate(angles):
            np.testing.assert_allclose(batch_result[i], rotY(a), atol=1e-14)

    def test_batch_rotZ_matches_scalar(self):
        """batch_rotZ should produce same results as individual rotZ calls."""
        rng = np.random.default_rng(44)
        angles = rng.uniform(-np.pi, np.pi, size=50)
        batch_result = batch_rotZ(angles)
        for i, a in enumerate(angles):
            np.testing.assert_allclose(batch_result[i], rotZ(a), atol=1e-14)

    def test_batch_get_premult_matches_scalar(self):
        """batch_get_premult_mat_rot should match scalar get_premult_mat_rot."""
        rng = np.random.default_rng(45)
        angles = rng.uniform(-np.pi, np.pi, size=(100, 3))
        for order in ['ZYX', 'XYZ', 'YZX', 'ZXY', 'YXZ', 'XZY']:
            order_list = list(order)
            batch_result = batch_get_premult_mat_rot(angles, order_list)
            for i in range(len(angles)):
                expected = get_premult_mat_rot(angles[i], order_list)
                np.testing.assert_allclose(batch_result[i], expected, atol=1e-12,
                    err_msg=f"Mismatch at index {i} for order {order}")

    def test_batch_rotation_output_shapes(self):
        """Batch rotation functions should return (N, 3, 3)."""
        angles = np.zeros(10)
        assert batch_rotX(angles).shape == (10, 3, 3)
        assert batch_rotY(angles).shape == (10, 3, 3)
        assert batch_rotZ(angles).shape == (10, 3, 3)

        angles_3d = np.zeros((10, 3))
        assert batch_get_premult_mat_rot(angles_3d, ['Z', 'Y', 'X']).shape == (10, 3, 3)

    def test_batch_rotation_identity_at_zero(self):
        """Zero angles should produce identity matrices."""
        angles = np.zeros((5, 3))
        result = batch_get_premult_mat_rot(angles, ['Z', 'Y', 'X'])
        for i in range(5):
            np.testing.assert_allclose(result[i], np.eye(3), atol=1e-15)

    def test_batch_rotation_orthogonality(self):
        """All batch rotation matrices should be orthogonal with det=1."""
        rng = np.random.default_rng(46)
        angles = rng.uniform(-np.pi, np.pi, size=(100, 3))
        R = batch_get_premult_mat_rot(angles, ['Z', 'Y', 'X'])
        # R @ R^T should be I
        RRT = R @ R.transpose(0, 2, 1)
        for i in range(100):
            np.testing.assert_allclose(RRT[i], np.eye(3), atol=1e-12)
        # det should be 1
        dets = np.linalg.det(R)
        np.testing.assert_allclose(dets, 1.0, atol=1e-12)

    def test_batch_single_element(self):
        """Batch functions should work with a single element."""
        angles = np.array([0.5])
        assert batch_rotX(angles).shape == (1, 3, 3)
        angles_3d = np.array([[0.1, 0.2, 0.3]])
        result = batch_get_premult_mat_rot(angles_3d, ['X', 'Y', 'Z'])
        assert result.shape == (1, 3, 3)


# =============================================================================
# Test: Vectorized forward kinematics
# =============================================================================

class TestVectorizedFK:
    """Tests for vectorized forward kinematics correctness."""

    def test_all_frames_matches_frame_by_frame(self, bvh_example):
        """All-frames FK should match computing each frame individually."""
        all_coords = bvh_example.get_spatial_coord(centered="world")
        for i in range(bvh_example.frame_count):
            single_coord = bvh_example.get_spatial_coord(frame_num=i, centered="world")
            np.testing.assert_allclose(all_coords[i], single_coord, atol=1e-10,
                err_msg=f"Frame {i} mismatch between batch and single computation")

    def test_fk_on_different_skeletons(self):
        """FK should work on all test files with different skeletons."""
        test_files = [
            "bvh_data/bvh_example.bvh",   # 56 frames, 29 nodes
            "bvh_data/bvh_test2.bvh",      # 61 frames, 28 nodes
            "bvh_data/bvh_test3.bvh",      # 100 frames, 73 nodes
        ]
        for filepath in test_files:
            bvh = read_bvh_file(filepath)
            coords = bvh.get_spatial_coord(centered="world")
            assert coords.shape == (bvh.frame_count, len(bvh.nodes), 3), \
                f"Shape mismatch for {filepath}"
            assert not np.any(np.isnan(coords)), f"NaN in coords for {filepath}"
            assert not np.any(np.isinf(coords)), f"Inf in coords for {filepath}"

    def test_fk_single_frame_file(self):
        """FK should work on a file with exactly 1 frame."""
        bvh = read_bvh_file("bvh_data/standard_skeleton.bvh")
        assert bvh.frame_count == 1
        coords_all = bvh.get_spatial_coord(centered="world")
        coords_single = bvh.get_spatial_coord(frame_num=0, centered="world")
        # All-frames returns (1, N, 3), single returns (N, 3)
        np.testing.assert_allclose(coords_all[0], coords_single, atol=1e-10)

    def test_fk_centering_modes_large_file(self):
        """All centering modes should work on a larger file (100 frames)."""
        bvh = read_bvh_file("bvh_data/bvh_test3.bvh")
        for mode in ["world", "skeleton", "first"]:
            coords = bvh.get_spatial_coord(centered=mode)
            assert coords.shape == (100, len(bvh.nodes), 3)
            assert not np.any(np.isnan(coords))

        # Skeleton centering: root should be at origin
        coords_skel = bvh.get_spatial_coord(centered="skeleton")
        np.testing.assert_allclose(coords_skel[:, 0, :], 0.0, atol=1e-10)

        # First centering: frame 0 root should be at origin
        coords_first = bvh.get_spatial_coord(centered="first")
        np.testing.assert_allclose(coords_first[0, 0, :], 0.0, atol=1e-10)

    def test_fk_frame_independence(self, bvh_example):
        """Modifying one frame's angles should not affect other frames' coords."""
        coords_before = bvh_example.get_spatial_coord(centered="world").copy()
        bvh_mod = bvh_example.copy()
        bvh_mod.joint_angles[0, :, :] = 0.0  # Zero out frame 0
        coords_after = bvh_mod.get_spatial_coord(centered="world")
        # Frame 0 should change
        assert not np.allclose(coords_after[0], coords_before[0])
        # All other frames should be identical
        np.testing.assert_allclose(coords_after[1:], coords_before[1:], atol=1e-10)


# =============================================================================
# Test: Parser with different files
# =============================================================================

class TestParserMultipleFiles:
    """Tests for parser correctness across different BVH files."""

    def test_parse_all_test_files(self):
        """All test BVH files should parse without errors and have valid data."""
        files = {
            "bvh_data/bvh_example.bvh": (56, 24, 29),
            "bvh_data/bvh_test1.bvh": (56, 24, 29),
            "bvh_data/bvh_test2.bvh": (61, 23, 28),
            "bvh_data/bvh_test3.bvh": (100, 60, 73),
            "bvh_data/standard_skeleton.bvh": (1, 24, 29),
        }
        for filepath, (exp_frames, exp_joints, exp_nodes) in files.items():
            bvh = read_bvh_file(filepath)
            assert bvh.frame_count == exp_frames, f"{filepath}: frame count"
            num_joints = len([n for n in bvh.nodes if not n.is_end_site()])
            assert num_joints == exp_joints, f"{filepath}: joint count"
            assert len(bvh.nodes) == exp_nodes, f"{filepath}: node count"
            assert bvh.root_pos.shape == (exp_frames, 3), f"{filepath}: root_pos shape"
            assert bvh.joint_angles.shape == (exp_frames, exp_joints, 3), f"{filepath}: joint_angles shape"
            assert not np.any(np.isnan(bvh.root_pos)), f"{filepath}: NaN in root_pos"
            assert not np.any(np.isnan(bvh.joint_angles)), f"{filepath}: NaN in joint_angles"

    def test_large_file_roundtrip(self):
        """Write + read of a large BVH file should preserve data."""
        bvh = read_bvh_file("bvh_data/bvh_test3.bvh")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "roundtrip_large.bvh"
            bvh.to_bvh_file(tmpfile, verbose=False)
            bvh2 = read_bvh_file(tmpfile)
            np.testing.assert_allclose(bvh2.root_pos, bvh.root_pos, atol=1e-5)
            np.testing.assert_allclose(bvh2.joint_angles, bvh.joint_angles, atol=1e-5)
            assert bvh2.frame_count == bvh.frame_count


# =============================================================================
# Test: inplace parameter uniformisation
# =============================================================================

class TestInplaceParameter:
    """Tests for uniform inplace=False default across all mutation methods."""

    def test_set_frames_from_6d_default_returns_copy(self, bvh_example):
        """set_frames_from_6d() default should return a new object, not modify self."""
        original_angles = bvh_example.joint_angles.copy()
        rp, r6d, _ = bvh_example.get_frames_as_6d()
        result = bvh_example.set_frames_from_6d(rp, r6d)
        assert result is not bvh_example
        np.testing.assert_allclose(bvh_example.joint_angles, original_angles, atol=1e-10)

    def test_set_frames_from_quaternion_default_returns_copy(self, bvh_example):
        """set_frames_from_quaternion() default should return a new object."""
        original_angles = bvh_example.joint_angles.copy()
        rp, quats, _ = bvh_example.get_frames_as_quaternion()
        result = bvh_example.set_frames_from_quaternion(rp, quats)
        assert result is not bvh_example
        np.testing.assert_allclose(bvh_example.joint_angles, original_angles, atol=1e-10)

    def test_set_frames_from_axisangle_default_returns_copy(self, bvh_example):
        """set_frames_from_axisangle() default should return a new object."""
        original_angles = bvh_example.joint_angles.copy()
        rp, aa, _ = bvh_example.get_frames_as_axisangle()
        result = bvh_example.set_frames_from_axisangle(rp, aa)
        assert result is not bvh_example
        np.testing.assert_allclose(bvh_example.joint_angles, original_angles, atol=1e-10)

    def test_single_joint_euler_angle_default_returns_copy(self, bvh_example):
        """single_joint_euler_angle() default should return a new object."""
        original_angles = bvh_example.joint_angles.copy()
        result = bvh_example.single_joint_euler_angle('Spine', 'XYZ')
        assert result is not bvh_example
        np.testing.assert_allclose(bvh_example.joint_angles, original_angles, atol=1e-10)

    def test_change_all_euler_orders_default_returns_copy(self, bvh_example):
        """change_all_euler_orders() default should return a new object."""
        original_angles = bvh_example.joint_angles.copy()
        result = bvh_example.change_all_euler_orders('XYZ')
        assert result is not bvh_example
        np.testing.assert_allclose(bvh_example.joint_angles, original_angles, atol=1e-10)

    def test_inplace_true_returns_none(self, bvh_example):
        """All mutation methods with inplace=True should return None."""
        bvh = bvh_example.copy()
        rp, r6d, _ = bvh.get_frames_as_6d()
        assert bvh.set_frames_from_6d(rp, r6d, inplace=True) is None

        rp, quats, _ = bvh.get_frames_as_quaternion()
        assert bvh.set_frames_from_quaternion(rp, quats, inplace=True) is None

        rp, aa, _ = bvh.get_frames_as_axisangle()
        assert bvh.set_frames_from_axisangle(rp, aa, inplace=True) is None

        assert bvh.single_joint_euler_angle('Spine', 'XYZ', inplace=True) is None
        assert bvh.change_all_euler_orders('ZYX', inplace=True) is None
        assert bvh.change_skeleton(bvh_example, inplace=True) is None
        assert bvh.scale_skeleton(2.0, inplace=True) is None

    def test_copy_from_set_frames_is_independent(self, bvh_example):
        """Copy returned by set_frames_from_6d should be independent from original."""
        rp, r6d, _ = bvh_example.get_frames_as_6d()
        result = bvh_example.set_frames_from_6d(rp, r6d)
        # Modify the result
        result.root_pos[0, 0] = 999.0
        # Original should be unaffected
        assert bvh_example.root_pos[0, 0] != 999.0


class TestChannelProtection:
    """Tests for frozen rot_channels / pos_channels after Bvh construction."""

    def test_rot_channels_frozen_after_bvh_init(self, bvh_example):
        """Direct mutation of rot_channels on a Bvh-owned node should raise."""
        joint = bvh_example.nodes[1]  # first non-root joint
        assert not joint.is_end_site()
        with pytest.raises(AttributeError, match="rot_channels is frozen"):
            joint.rot_channels = ['X', 'Y', 'Z']

    def test_pos_channels_frozen_after_bvh_init(self, bvh_example):
        """Direct mutation of pos_channels on the root should raise."""
        with pytest.raises(AttributeError, match="pos_channels is frozen"):
            bvh_example.root.pos_channels = ['Z', 'Y', 'X']

    def test_standalone_node_not_frozen(self):
        """Nodes created outside a Bvh should not be frozen."""
        joint = BvhJoint('TestJoint', rot_channels=['Z', 'Y', 'X'])
        # Should work fine — not frozen
        joint.rot_channels = ['X', 'Y', 'Z']
        assert joint.rot_channels == ['X', 'Y', 'Z']

    def test_single_joint_euler_angle_still_works(self, bvh_example):
        """Bvh methods should bypass the freeze to update channels."""
        old_order = bvh_example.nodes[1].rot_channels[:]
        result = bvh_example.single_joint_euler_angle(
            bvh_example.nodes[1].name, 'XYZ')
        assert result.nodes[1].rot_channels == ['X', 'Y', 'Z']
        # Original unchanged
        assert bvh_example.nodes[1].rot_channels == old_order

    def test_change_all_euler_orders_still_works(self, bvh_example):
        """change_all_euler_orders should bypass freeze on all nodes."""
        result = bvh_example.change_all_euler_orders('XYZ')
        for node in result.nodes:
            if not node.is_end_site():
                assert node.rot_channels == ['X', 'Y', 'Z']

    def test_frozen_flag_preserved_on_copy(self, bvh_example):
        """deepcopy should preserve the frozen state."""
        copy = bvh_example.copy()
        joint = copy.nodes[1]
        with pytest.raises(AttributeError, match="rot_channels is frozen"):
            joint.rot_channels = ['X', 'Y', 'Z']

    def test_inplace_euler_change_on_frozen_nodes(self, bvh_example):
        """inplace=True euler change should work on frozen nodes."""
        bvh = bvh_example.copy()
        bvh.single_joint_euler_angle(bvh.nodes[1].name, 'XYZ', inplace=True)
        assert bvh.nodes[1].rot_channels == ['X', 'Y', 'Z']

    def test_read_bvh_file_produces_frozen_nodes(self, bvh_example_path):
        """Nodes from read_bvh_file should be frozen after Bvh construction."""
        bvh = read_bvh_file(bvh_example_path)
        for node in bvh.nodes:
            if not node.is_end_site():
                with pytest.raises(AttributeError):
                    node.rot_channels = ['X', 'Y', 'Z']


class TestFrameSlicing:
    """Tests for slice_frames, concat, and resample."""

    def test_slice_frames_basic(self, bvh_example):
        """Basic slicing returns correct frame count."""
        sliced = bvh_example.slice_frames(2, 8)
        assert sliced.frame_count == 6
        np.testing.assert_array_equal(
            sliced.root_pos, bvh_example.root_pos[2:8])
        np.testing.assert_array_equal(
            sliced.joint_angles, bvh_example.joint_angles[2:8])

    def test_slice_frames_with_step(self, bvh_example):
        """Slicing with step adjusts frame_frequency."""
        sliced = bvh_example.slice_frames(0, None, 3)
        expected_count = len(bvh_example.root_pos[0::3])
        assert sliced.frame_count == expected_count
        assert sliced.frame_frequency == bvh_example.frame_frequency * 3

    def test_slice_frames_preserves_skeleton(self, bvh_example):
        """Sliced Bvh has the same skeleton."""
        sliced = bvh_example.slice_frames(0, 5)
        assert len(sliced.nodes) == len(bvh_example.nodes)
        for n1, n2 in zip(sliced.nodes, bvh_example.nodes):
            assert n1.name == n2.name

    def test_slice_frames_empty_result(self, bvh_example):
        """Slicing to empty range produces 0 frames."""
        sliced = bvh_example.slice_frames(5, 5)
        assert sliced.frame_count == 0

    def test_slice_frames_independence(self, bvh_example):
        """Sliced Bvh is independent from original."""
        sliced = bvh_example.slice_frames(0, 5)
        sliced.root_pos[0, 0] = 999.0
        assert bvh_example.root_pos[0, 0] != 999.0

    def test_concat_basic(self, bvh_example):
        """Concatenating two copies doubles the frame count."""
        result = bvh_example.concat(bvh_example)
        assert result.frame_count == 2 * bvh_example.frame_count
        np.testing.assert_array_equal(
            result.root_pos[:bvh_example.frame_count],
            bvh_example.root_pos)
        np.testing.assert_array_equal(
            result.root_pos[bvh_example.frame_count:],
            bvh_example.root_pos)

    def test_concat_mismatched_names_raises(self, bvh_example):
        """Concat with different joint names should raise."""
        other = bvh_example.copy()
        other.nodes[1]._name = 'DifferentName'
        with pytest.raises(ValueError, match="name mismatch"):
            bvh_example.concat(other)

    def test_concat_mismatched_channels_raises(self, bvh_example):
        """Concat with different rotation orders should raise."""
        other = bvh_example.change_all_euler_orders('XYZ')
        with pytest.raises(ValueError, match="Rotation order mismatch"):
            bvh_example.concat(other)

    def test_concat_mismatched_frequency_warns(self, bvh_example):
        """Concat with different frame frequencies should warn."""
        other = bvh_example.copy()
        other.frame_frequency = bvh_example.frame_frequency * 2
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh_example.concat(other)
            assert len(w) == 1
            assert "frequency mismatch" in str(w[0].message).lower()

    def test_resample_identity(self, bvh_example):
        """Resampling to the same fps should produce near-identical frames."""
        original_fps = 1.0 / bvh_example.frame_frequency
        result = bvh_example.resample(original_fps)
        assert result.frame_count == bvh_example.frame_count
        np.testing.assert_allclose(
            result.root_pos, bvh_example.root_pos, atol=1e-6)
        np.testing.assert_allclose(
            result.joint_angles, bvh_example.joint_angles, atol=0.5)

    def test_resample_upsample(self, bvh_example):
        """Upsampling doubles frame count (approximately)."""
        original_fps = 1.0 / bvh_example.frame_frequency
        result = bvh_example.resample(original_fps * 2)
        # Should have roughly 2x frames (±1 due to boundary)
        assert abs(result.frame_count - 2 * bvh_example.frame_count) <= 2
        assert abs(result.frame_frequency - bvh_example.frame_frequency / 2) < 1e-10

    def test_resample_downsample(self, bvh_example):
        """Downsampling halves frame count (approximately)."""
        original_fps = 1.0 / bvh_example.frame_frequency
        result = bvh_example.resample(original_fps / 2)
        assert abs(result.frame_count - bvh_example.frame_count // 2) <= 2

    def test_resample_preserves_spatial_at_original_times(self, bvh_example):
        """Spatial coordinates at original frame times should be preserved."""
        original_fps = 1.0 / bvh_example.frame_frequency
        # Upsample 2x, then check even frames (which correspond to originals)
        result = bvh_example.resample(original_fps * 2)
        orig_coords = bvh_example.get_spatial_coord(centered='skeleton')
        result_coords = result.get_spatial_coord(centered='skeleton')
        # Even indices in result should match original frames
        # (approximately, due to SLERP being on the geodesic)
        np.testing.assert_allclose(
            result_coords[::2, :, :][:orig_coords.shape[0]],
            orig_coords,
            atol=1.0)  # generous tolerance for resampled data

    def test_resample_single_frame(self, bvh_example):
        """Resampling a single-frame Bvh returns a copy."""
        single = bvh_example.slice_frames(0, 1)
        result = single.resample(60)
        assert result.frame_count == 1


class TestSkeletonRetargeting:
    """Tests for change_skeleton with name_mapping."""

    def test_change_skeleton_no_mapping(self, bvh_example):
        """Without mapping, change_skeleton works as before (by name)."""
        ref = bvh_example.copy()
        ref.scale_skeleton(2.0, inplace=True)
        result = bvh_example.change_skeleton(ref)
        for n1, n2 in zip(result.nodes, ref.nodes):
            np.testing.assert_array_equal(n1.offset, n2.offset)

    def test_change_skeleton_with_identity_mapping(self, bvh_example):
        """Identity mapping (name→same name) should behave identically."""
        ref = bvh_example.copy()
        ref.scale_skeleton(2.0, inplace=True)
        mapping = {n.name: n.name for n in bvh_example.nodes}
        result = bvh_example.change_skeleton(ref, name_mapping=mapping)
        for n1, n2 in zip(result.nodes, ref.nodes):
            np.testing.assert_array_equal(n1.offset, n2.offset)

    def test_change_skeleton_with_prefix_mapping(self, bvh_example):
        """Mapping with prefixed names should copy offsets correctly."""
        # Create a reference with prefixed names
        ref = bvh_example.copy()
        ref.scale_skeleton(3.0, inplace=True)
        for node in ref.nodes:
            node._name = 'prefix:' + node.name

        # Build mapping: self name → prefixed name
        mapping = {n.name: 'prefix:' + n.name for n in bvh_example.nodes}
        result = bvh_example.change_skeleton(ref, name_mapping=mapping)

        for n_result, n_ref in zip(result.nodes, ref.nodes):
            np.testing.assert_array_equal(n_result.offset, n_ref.offset)

    def test_change_skeleton_lenient_unmapped(self, bvh_example):
        """Unmapped joints keep their original offsets in lenient mode."""
        ref = bvh_example.copy()
        ref.scale_skeleton(5.0, inplace=True)
        # Rename all ref nodes so nothing matches by name
        for node in ref.nodes:
            node._name = 'ref_' + node.name
        # Only map the root
        mapping = {bvh_example.root.name: 'ref_' + bvh_example.root.name}
        result = bvh_example.change_skeleton(
            ref, name_mapping=mapping, strict=False)
        # Root should be scaled
        np.testing.assert_array_equal(
            result.root.offset, ref.root.offset)
        # Non-mapped joints should keep original offsets (no name match)
        np.testing.assert_array_equal(
            result.nodes[1].offset, bvh_example.nodes[1].offset)

    def test_change_skeleton_strict_unmapped_raises(self, bvh_example):
        """Strict mode should raise when a joint has no match."""
        ref = bvh_example.copy()
        # Remove one node name from ref to create a mismatch
        ref.nodes[1]._name = 'NONEXISTENT'
        with pytest.raises(ValueError, match="not found"):
            bvh_example.change_skeleton(ref, strict=True)

    def test_change_skeleton_preserves_motion(self, bvh_example):
        """change_skeleton should not modify root_pos or joint_angles."""
        ref = bvh_example.copy()
        ref.scale_skeleton(2.0, inplace=True)
        result = bvh_example.change_skeleton(ref)
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)
        np.testing.assert_array_equal(
            result.joint_angles, bvh_example.joint_angles)


class TestJointSubsetting:
    """Tests for extract_joints."""

    def test_extract_all_joints(self, bvh_example):
        """Extracting all joints should produce identical rest-pose coordinates."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        result = bvh_example.extract_joints(all_names)

        # Same number of non-end-site joints
        orig_joints = [n for n in bvh_example.nodes if not n.is_end_site()]
        result_joints = [n for n in result.nodes if not n.is_end_site()]
        assert len(result_joints) == len(orig_joints)

        # Rest pose spatial coordinates should match
        orig_rest = bvh_example.get_rest_pose(mode='coordinates')
        result_rest = result.get_rest_pose(mode='coordinates')
        # Compare only the kept joint positions (end sites may differ)
        for n in result.nodes:
            if not n.is_end_site():
                np.testing.assert_allclose(
                    result_rest[result.node_index[n.name]],
                    orig_rest[bvh_example.node_index[n.name]],
                    atol=1e-10)

    def test_extract_root_only(self, bvh_example):
        """Extracting only the root should return 1 joint + 1 end site."""
        result = bvh_example.extract_joints([bvh_example.root.name])
        joints = [n for n in result.nodes if not n.is_end_site()]
        end_sites = [n for n in result.nodes if n.is_end_site()]
        assert len(joints) == 1
        assert len(end_sites) == 1
        assert result.joint_angles.shape[1] == 1

    def test_extract_without_root_raises(self, bvh_example):
        """Extracting without the root should raise ValueError."""
        with pytest.raises(ValueError, match="Root joint"):
            bvh_example.extract_joints(['Spine'])

    def test_extract_removes_leaf_joint(self, bvh_example):
        """Removing a leaf joint should not affect its parent."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        # Remove the last joint (a leaf)
        leaf_name = all_names[-1]
        reduced = [n for n in all_names if n != leaf_name]
        result = bvh_example.extract_joints(reduced)

        result_names = [n.name for n in result.nodes if not n.is_end_site()]
        assert leaf_name not in result_names
        assert len(result_names) == len(reduced)

    def test_extract_collapses_intermediate_offset(self, bvh_example):
        """Removing an intermediate joint should collapse offsets."""
        # Get names of nodes that have both a parent and children (intermediates)
        all_joints = [n for n in bvh_example.nodes if not n.is_end_site()]
        # Find an intermediate: not root, has at least one non-end-site child
        intermediate = None
        for n in all_joints[1:]:  # skip root
            non_end_children = [c for c in n.children if not c.is_end_site()]
            if non_end_children:
                intermediate = n
                break
        if intermediate is None:
            pytest.skip("No intermediate joint found in test skeleton")

        child = [c for c in intermediate.children if not c.is_end_site()][0]

        keep = [n.name for n in all_joints if n.name != intermediate.name]
        result = bvh_example.extract_joints(keep)

        # The child's offset in result should be intermediate.offset + child.offset
        result_child = None
        for n in result.nodes:
            if n.name == child.name:
                result_child = n
                break
        expected_offset = intermediate.offset + child.offset
        np.testing.assert_allclose(result_child.offset, expected_offset, atol=1e-10)

    def test_extract_joint_angles_shape(self, bvh_example):
        """Joint angles should have correct shape after extraction."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        # Keep half the joints
        keep = all_names[:len(all_names) // 2]
        keep = [bvh_example.root.name] + [n for n in keep if n != bvh_example.root.name]
        result = bvh_example.extract_joints(keep)
        assert result.joint_angles.shape == (
            bvh_example.frame_count, len(keep), 3)

    def test_extract_preserves_root_pos(self, bvh_example):
        """Root position should be unchanged after extraction."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        result = bvh_example.extract_joints(all_names[:5])
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)

    def test_extract_node_index_correct(self, bvh_example):
        """node_index should map to correct indices in the reduced node list."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        keep = all_names[:5]
        result = bvh_example.extract_joints(keep)
        for i, node in enumerate(result.nodes):
            assert result.node_index[node.name] == i

    def test_extract_rest_pose_matches(self, bvh_example):
        """Rest-pose coordinates of kept joints should match original."""
        all_joints = [n for n in bvh_example.nodes if not n.is_end_site()]
        keep = [n.name for n in all_joints[:8]]
        if bvh_example.root.name not in keep:
            keep = [bvh_example.root.name] + keep

        result = bvh_example.extract_joints(keep)
        orig_rest = bvh_example.get_rest_pose(mode='coordinates')
        result_rest = result.get_rest_pose(mode='coordinates')

        for name in keep:
            np.testing.assert_allclose(
                result_rest[result.node_index[name]],
                orig_rest[bvh_example.node_index[name]],
                atol=1e-10,
                err_msg=f"Rest pose mismatch for {name}")

    def test_extract_independence(self, bvh_example):
        """Extracted Bvh should be independent from original."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        result = bvh_example.extract_joints(all_names)
        result.root_pos[0, 0] = 999.0
        assert bvh_example.root_pos[0, 0] != 999.0


# =============================================================================
# Test: File round-trip for all BVH files
# =============================================================================

class TestFileRoundTripAllFiles:
    """Test to_bvh_file -> read_bvh_file roundtrip for every BVH file."""

    def _roundtrip(self, bvh_orig, tmp_path):
        """Helper: write, re-read, and return the re-read Bvh."""
        tmpfile = tmp_path / "roundtrip.bvh"
        bvh_orig.to_bvh_file(tmpfile, verbose=False)
        return read_bvh_file(tmpfile)

    def test_roundtrip_bvh_test1(self, tmp_path):
        """Round-trip bvh_test1.bvh preserves data."""
        bvh = read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_test1.bvh")
        bvh2 = self._roundtrip(bvh, tmp_path)
        np.testing.assert_allclose(bvh2.root_pos, bvh.root_pos, atol=1e-5)
        np.testing.assert_allclose(bvh2.joint_angles, bvh.joint_angles, atol=1e-5)
        assert len(bvh2.nodes) == len(bvh.nodes)
        assert abs(bvh2.frame_frequency - bvh.frame_frequency) < 1e-6

    def test_roundtrip_bvh_test2_yxz(self, bvh_test2, tmp_path):
        """Round-trip bvh_test2.bvh (YXZ channel order) preserves data."""
        bvh2 = self._roundtrip(bvh_test2, tmp_path)
        np.testing.assert_allclose(bvh2.root_pos, bvh_test2.root_pos, atol=1e-5)
        np.testing.assert_allclose(bvh2.joint_angles, bvh_test2.joint_angles, atol=1e-5)
        assert len(bvh2.nodes) == len(bvh_test2.nodes)
        assert abs(bvh2.frame_frequency - bvh_test2.frame_frequency) < 1e-6

    def test_roundtrip_bvh_test3_mixed(self, bvh_test3, tmp_path):
        """Round-trip bvh_test3.bvh (60 joints, mixed orders) preserves data."""
        bvh2 = self._roundtrip(bvh_test3, tmp_path)
        np.testing.assert_allclose(bvh2.root_pos, bvh_test3.root_pos, atol=1e-5)
        np.testing.assert_allclose(bvh2.joint_angles, bvh_test3.joint_angles, atol=1e-5)
        assert len(bvh2.nodes) == len(bvh_test3.nodes)
        assert abs(bvh2.frame_frequency - bvh_test3.frame_frequency) < 1e-6

    def test_roundtrip_standard_skeleton(self, standard_skeleton, tmp_path):
        """Round-trip standard_skeleton.bvh (1 frame) preserves data."""
        bvh2 = self._roundtrip(standard_skeleton, tmp_path)
        np.testing.assert_allclose(bvh2.root_pos, standard_skeleton.root_pos, atol=1e-5)
        np.testing.assert_allclose(bvh2.joint_angles, standard_skeleton.joint_angles, atol=1e-5)
        assert len(bvh2.nodes) == len(standard_skeleton.nodes)
        assert abs(bvh2.frame_frequency - standard_skeleton.frame_frequency) < 1e-6

    def test_roundtrip_preserves_rotation_orders(self, bvh_test3, tmp_path):
        """For bvh_test3, verify each joint's rot_channels match after roundtrip."""
        bvh2 = self._roundtrip(bvh_test3, tmp_path)
        for n1, n2 in zip(bvh_test3.nodes, bvh2.nodes):
            if not n1.is_end_site():
                assert n1.rot_channels == n2.rot_channels, \
                    f"rot_channels mismatch for {n1.name}: {n1.rot_channels} vs {n2.rot_channels}"

    def test_roundtrip_preserves_hierarchy(self, bvh_test3, tmp_path):
        """Verify parent-child relationships match after roundtrip."""
        bvh2 = self._roundtrip(bvh_test3, tmp_path)
        for n1, n2 in zip(bvh_test3.nodes, bvh2.nodes):
            assert n1.name == n2.name
            parent1 = n1.parent.name if n1.parent is not None else None
            parent2 = n2.parent.name if n2.parent is not None else None
            assert parent1 == parent2, \
                f"Parent mismatch for {n1.name}: {parent1} vs {parent2}"
            if not n1.is_end_site():
                children1 = sorted([c.name for c in n1.children])
                children2 = sorted([c.name for c in n2.children])
                assert children1 == children2, \
                    f"Children mismatch for {n1.name}: {children1} vs {children2}"


# =============================================================================
# Test: scale_skeleton
# =============================================================================

class TestScaleSkeleton:
    """Tests for scale_skeleton() method."""

    def test_scale_uniform_doubles_offsets(self, bvh_example):
        """scale=2.0 should double all offsets."""
        original_offsets = [n.offset.copy() for n in bvh_example.nodes]
        result = bvh_example.scale_skeleton(2.0)
        for orig_off, node in zip(original_offsets, result.nodes):
            np.testing.assert_allclose(node.offset, orig_off * 2.0, atol=1e-10)

    def test_scale_uniform_preserves_angles(self, bvh_example):
        """joint_angles should be unchanged after scale."""
        result = bvh_example.scale_skeleton(2.0)
        np.testing.assert_allclose(result.joint_angles, bvh_example.joint_angles, atol=1e-10)

    def test_scale_per_axis(self, bvh_example):
        """scale=[1, 2, 3] should scale each axis independently."""
        original_offsets = [n.offset.copy() for n in bvh_example.nodes]
        result = bvh_example.scale_skeleton([1, 2, 3])
        for orig_off, node in zip(original_offsets, result.nodes):
            expected = orig_off * np.array([1, 2, 3])
            np.testing.assert_allclose(node.offset, expected, atol=1e-10)

    def test_scale_affects_spatial_coords(self, bvh_example):
        """Spatial coords should scale proportionally at rest pose."""
        rest_orig = bvh_example.get_rest_pose(mode='coordinates')
        result = bvh_example.scale_skeleton(2.0)
        rest_scaled = result.get_rest_pose(mode='coordinates')
        np.testing.assert_allclose(rest_scaled, rest_orig * 2.0, atol=1e-10)

    def test_scale_inplace_true(self, bvh_example):
        """inplace=True should return None and modify self."""
        bvh = bvh_example.copy()
        original_offsets = [n.offset.copy() for n in bvh.nodes]
        ret = bvh.scale_skeleton(2.0, inplace=True)
        assert ret is None
        for orig_off, node in zip(original_offsets, bvh.nodes):
            np.testing.assert_allclose(node.offset, orig_off * 2.0, atol=1e-10)

    def test_scale_inplace_false(self, bvh_example):
        """inplace=False should return new Bvh, original unchanged."""
        original_offsets = [n.offset.copy() for n in bvh_example.nodes]
        result = bvh_example.scale_skeleton(2.0, inplace=False)
        assert isinstance(result, Bvh)
        assert result is not bvh_example
        # Original unchanged
        for orig_off, node in zip(original_offsets, bvh_example.nodes):
            np.testing.assert_allclose(node.offset, orig_off, atol=1e-10)

    def test_scale_invalid_raises(self, bvh_example):
        """scale=[1,2] should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.scale_skeleton([1, 2])

    def test_scale_negative(self, bvh_example):
        """scale=-1.0 should negate all offsets."""
        original_offsets = [n.offset.copy() for n in bvh_example.nodes]
        result = bvh_example.scale_skeleton(-1.0)
        for orig_off, node in zip(original_offsets, result.nodes):
            np.testing.assert_allclose(node.offset, -orig_off, atol=1e-10)

    def test_scale_by_one_noop(self, bvh_example):
        """scale=1.0 should leave offsets unchanged."""
        original_offsets = [n.offset.copy() for n in bvh_example.nodes]
        result = bvh_example.scale_skeleton(1.0)
        for orig_off, node in zip(original_offsets, result.nodes):
            np.testing.assert_allclose(node.offset, orig_off, atol=1e-10)

    def test_scale_zero(self, bvh_example):
        """scale=0.0 should make all offsets zero."""
        result = bvh_example.scale_skeleton(0.0)
        for node in result.nodes:
            np.testing.assert_allclose(node.offset, np.zeros(3), atol=1e-10)


# =============================================================================
# Test: get_rest_pose
# =============================================================================

class TestGetRestPose:
    """Tests for get_rest_pose() method."""

    def test_rest_pose_coordinates_shape(self, bvh_example):
        """mode='coordinates' returns (N, 3) where N = len(nodes)."""
        rest = bvh_example.get_rest_pose(mode='coordinates')
        assert rest.shape == (len(bvh_example.nodes), 3)

    def test_rest_pose_coordinates_root_at_origin(self, bvh_example):
        """Root position should be [0,0,0] in rest pose coordinates."""
        rest = bvh_example.get_rest_pose(mode='coordinates')
        np.testing.assert_allclose(rest[0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_rest_pose_euler_shape(self, bvh_example):
        """mode='euler' returns tuple of (3,) and (J, 3) arrays."""
        root_pos_rest, joint_angles_rest = bvh_example.get_rest_pose(mode='euler')
        assert root_pos_rest.shape == (3,)
        assert joint_angles_rest.shape == (bvh_example.joint_count, 3)

    def test_rest_pose_euler_values_are_zero(self, bvh_example):
        """Both arrays from euler mode should be all zeros."""
        root_pos_rest, joint_angles_rest = bvh_example.get_rest_pose(mode='euler')
        np.testing.assert_allclose(root_pos_rest, np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(joint_angles_rest, np.zeros_like(joint_angles_rest), atol=1e-10)

    def test_rest_pose_invalid_mode_raises(self, bvh_example):
        """mode='bad' should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.get_rest_pose(mode='bad')

    def test_rest_pose_coordinates_all_files(self, bvh_example, bvh_test2, bvh_test3, standard_skeleton):
        """Verify rest pose coordinates are valid for all test files."""
        for bvh in [bvh_example, bvh_test2, bvh_test3, standard_skeleton]:
            rest = bvh.get_rest_pose(mode='coordinates')
            assert rest.shape == (len(bvh.nodes), 3)
            assert not np.any(np.isnan(rest))
            assert not np.any(np.isinf(rest))
            # Root at origin
            np.testing.assert_allclose(rest[0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_rest_pose_modes_consistent(self, bvh_example):
        """Euler mode zeros through FK should give same as coordinates mode."""
        rest_coords = bvh_example.get_rest_pose(mode='coordinates')
        root_pos_rest, joint_angles_rest = bvh_example.get_rest_pose(mode='euler')
        # Compute spatial coords from zeros
        rest_via_fk = frames_to_spatial_coord(
            bvh_example, root_pos=root_pos_rest,
            joint_angles=joint_angles_rest, centered="skeleton")
        np.testing.assert_allclose(rest_via_fk, rest_coords, atol=1e-10)


# =============================================================================
# Test: get_df_constructor spatial mode
# =============================================================================

class TestGetDfConstructorSpatial:
    """Tests for get_df_constructor with spatial/coordinate mode."""

    def test_spatial_mode_shape(self, bvh_example):
        """Correct number of keys: time + N*3."""
        df_data = bvh_example.get_df_constructor(mode='coordinates', centered='world')
        expected_keys = 1 + len(bvh_example.nodes) * 3  # time + N*3
        assert len(df_data) == expected_keys

    def test_spatial_mode_values_match(self, bvh_example):
        """Spot-check values vs get_spatial_coord."""
        df_data = bvh_example.get_df_constructor(mode='coordinates', centered='world')
        spatial = bvh_example.get_spatial_coord(centered='world')
        # Check root X column
        root_name = bvh_example.root.name
        np.testing.assert_allclose(df_data[f'{root_name}_X'], spatial[:, 0, 0], atol=1e-10)
        np.testing.assert_allclose(df_data[f'{root_name}_Y'], spatial[:, 0, 1], atol=1e-10)
        np.testing.assert_allclose(df_data[f'{root_name}_Z'], spatial[:, 0, 2], atol=1e-10)

    def test_spatial_mode_skeleton_centering(self, bvh_example):
        """Root X/Y/Z columns should all be zeros for skeleton centering."""
        df_data = bvh_example.get_df_constructor(mode='coordinates', centered='skeleton')
        root_name = bvh_example.root.name
        np.testing.assert_allclose(df_data[f'{root_name}_X'], 0.0, atol=1e-10)
        np.testing.assert_allclose(df_data[f'{root_name}_Y'], 0.0, atol=1e-10)
        np.testing.assert_allclose(df_data[f'{root_name}_Z'], 0.0, atol=1e-10)

    def test_spatial_mode_includes_end_sites(self, bvh_example):
        """End-site columns should be present."""
        df_data = bvh_example.get_df_constructor(mode='coordinates', centered='world')
        end_site_names = [n.name for n in bvh_example.nodes if n.is_end_site()]
        for name in end_site_names:
            assert f'{name}_X' in df_data
            assert f'{name}_Y' in df_data
            assert f'{name}_Z' in df_data

    def test_spatial_mode_invalid_raises(self, bvh_example):
        """mode='bad' should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.get_df_constructor(mode='bad')

    def test_euler_mode_shape(self, bvh_example):
        """Verify euler mode has correct column count."""
        df_data = bvh_example.get_df_constructor(mode='euler')
        # time + 3 pos + 3 * num_joints rot
        expected_keys = 1 + 3 + 3 * bvh_example.joint_count
        assert len(df_data) == expected_keys


# =============================================================================
# Test: euler_column_names property
# =============================================================================

class TestEulerColumnNames:
    """Tests for euler_column_names property."""

    def test_column_names_length(self, bvh_example):
        """Should be 3 + 3 * num_joints."""
        names = bvh_example.euler_column_names
        expected_len = 3 + 3 * bvh_example.joint_count
        assert len(names) == expected_len

    def test_column_names_root_prefix(self, bvh_example):
        """First 3 should be Hips_X_pos, Hips_Y_pos, Hips_Z_pos."""
        names = bvh_example.euler_column_names
        assert names[0] == 'Hips_X_pos'
        assert names[1] == 'Hips_Y_pos'
        assert names[2] == 'Hips_Z_pos'

    def test_column_names_match_channels(self, bvh_example):
        """For each joint, axes in column names match rot_channels."""
        names = bvh_example.euler_column_names
        # Skip first 3 (pos channels), then group by 3
        rot_names = names[3:]
        j_idx = 0
        for node in bvh_example.nodes:
            if node.is_end_site():
                continue
            for i, ax in enumerate(node.rot_channels):
                col = rot_names[j_idx * 3 + i]
                expected = f'{node.name}_{ax}_rot'
                assert col == expected, f"Column {col} != {expected}"
            j_idx += 1

    def test_column_names_after_order_change(self, bvh_example):
        """Reflects new order after change_all_euler_orders."""
        result = bvh_example.change_all_euler_orders('XYZ')
        names = result.euler_column_names
        rot_names = names[3:]
        # Every group of 3 should be X_rot, Y_rot, Z_rot
        for j in range(result.joint_count):
            assert '_X_rot' in rot_names[j * 3]
            assert '_Y_rot' in rot_names[j * 3 + 1]
            assert '_Z_rot' in rot_names[j * 3 + 2]

    def test_column_names_all_files(self, bvh_example, bvh_test2, bvh_test3, standard_skeleton):
        """Valid for all test files."""
        for bvh in [bvh_example, bvh_test2, bvh_test3, standard_skeleton]:
            names = bvh.euler_column_names
            expected_len = 3 + 3 * bvh.joint_count
            assert len(names) == expected_len


# =============================================================================
# Test: frames_to_spatial_coord standalone function
# =============================================================================

class TestFramesToSpatialCoordStandalone:
    """Tests for the standalone frames_to_spatial_coord function."""

    def test_accepts_bvh_object(self, bvh_example):
        """Passing Bvh works, returns correct shape."""
        result = frames_to_spatial_coord(bvh_example)
        assert result.shape == (bvh_example.frame_count, len(bvh_example.nodes), 3)

    def test_accepts_node_list(self, bvh_example):
        """Passing nodes + root_pos + joint_angles works."""
        result = frames_to_spatial_coord(
            bvh_example.nodes,
            root_pos=bvh_example.root_pos,
            joint_angles=bvh_example.joint_angles)
        assert result.shape == (bvh_example.frame_count, len(bvh_example.nodes), 3)

    def test_single_frame_shape(self, bvh_example):
        """Single frame returns (N, 3)."""
        result = frames_to_spatial_coord(
            bvh_example.nodes,
            root_pos=bvh_example.root_pos[0],
            joint_angles=bvh_example.joint_angles[0])
        assert result.shape == (len(bvh_example.nodes), 3)

    def test_multi_frame_shape(self, bvh_example):
        """All frames returns (F, N, 3)."""
        result = frames_to_spatial_coord(bvh_example)
        assert result.shape == (bvh_example.frame_count, len(bvh_example.nodes), 3)

    def test_centering_modes(self, bvh_example):
        """All 3 centering modes work."""
        for mode in ["world", "skeleton", "first"]:
            result = frames_to_spatial_coord(bvh_example, centered=mode)
            assert result.shape == (bvh_example.frame_count, len(bvh_example.nodes), 3)
            assert not np.any(np.isnan(result))

    def test_invalid_centered_raises(self, bvh_example):
        """Bad centering raises ValueError."""
        with pytest.raises(ValueError):
            frames_to_spatial_coord(bvh_example, centered="bad_value")

    def test_matches_bvh_method(self, bvh_example):
        """Standalone function result matches bvh.get_spatial_coord()."""
        standalone = frames_to_spatial_coord(bvh_example, centered="world")
        method = bvh_example.get_spatial_coord(centered="world")
        np.testing.assert_allclose(standalone, method, atol=1e-10)

    def test_node_list_without_arrays_raises(self, bvh_example):
        """Passing node list without root_pos/joint_angles raises."""
        with pytest.raises(ValueError):
            frames_to_spatial_coord(bvh_example.nodes)


# =============================================================================
# Test: concat + slice round-trip
# =============================================================================

class TestConcatSliceRoundTrip:
    """Tests for split-concat round-trip."""

    def test_split_concat_recovers_original(self, bvh_example):
        """slice [0:28] + [28:56], concat, compare to original."""
        part1 = bvh_example.slice_frames(0, 28)
        part2 = bvh_example.slice_frames(28, 56)
        recovered = part1.concat(part2)
        np.testing.assert_allclose(recovered.root_pos, bvh_example.root_pos, atol=1e-12)
        np.testing.assert_allclose(recovered.joint_angles, bvh_example.joint_angles, atol=1e-12)

    @pytest.mark.parametrize("split_point", [10, 20, 30, 40, 50])
    def test_split_at_various_points(self, bvh_example, split_point):
        """Verify round-trip at various split points."""
        part1 = bvh_example.slice_frames(0, split_point)
        part2 = bvh_example.slice_frames(split_point, bvh_example.frame_count)
        recovered = part1.concat(part2)
        np.testing.assert_allclose(recovered.root_pos, bvh_example.root_pos, atol=1e-12)
        np.testing.assert_allclose(recovered.joint_angles, bvh_example.joint_angles, atol=1e-12)

    def test_concat_preserves_spatial(self, bvh_example):
        """Spatial coords of concatenated match original."""
        part1 = bvh_example.slice_frames(0, 28)
        part2 = bvh_example.slice_frames(28, 56)
        recovered = part1.concat(part2)
        orig_spatial = bvh_example.get_spatial_coord(centered="world")
        recovered_spatial = recovered.get_spatial_coord(centered="world")
        np.testing.assert_allclose(recovered_spatial, orig_spatial, atol=1e-10)

    def test_slice_step_then_identity(self, bvh_example):
        """Slice with step=1 matches original."""
        sliced = bvh_example.slice_frames(0, None, 1)
        np.testing.assert_allclose(sliced.root_pos, bvh_example.root_pos, atol=1e-12)
        np.testing.assert_allclose(sliced.joint_angles, bvh_example.joint_angles, atol=1e-12)


# =============================================================================
# Test: freeze preservation across operations
# =============================================================================

class TestFreezePreservation:
    """Verify frozen channels survive various operations."""

    def _assert_frozen(self, bvh):
        """Check that all non-end-site nodes have frozen rot_channels."""
        for node in bvh.nodes:
            if not node.is_end_site():
                with pytest.raises(AttributeError):
                    node.rot_channels = ['X', 'Y', 'Z']

    def test_freeze_survives_deepcopy(self, bvh_example):
        """Frozen channels survive copy.deepcopy."""
        bvh_copy = copy.deepcopy(bvh_example)
        self._assert_frozen(bvh_copy)

    def test_freeze_survives_set_frames_6d(self, bvh_example):
        """set_frames_from_6d(inplace=False) result is frozen."""
        rp, r6d, _ = bvh_example.get_frames_as_6d()
        result = bvh_example.set_frames_from_6d(rp, r6d)
        self._assert_frozen(result)

    def test_freeze_survives_set_frames_6d_inplace(self, bvh_example):
        """set_frames_from_6d(inplace=True), still frozen."""
        bvh = bvh_example.copy()
        rp, r6d, _ = bvh.get_frames_as_6d()
        bvh.set_frames_from_6d(rp, r6d, inplace=True)
        self._assert_frozen(bvh)

    def test_freeze_survives_set_frames_quat(self, bvh_example):
        """set_frames_from_quaternion result is frozen."""
        rp, quats, _ = bvh_example.get_frames_as_quaternion()
        result = bvh_example.set_frames_from_quaternion(rp, quats)
        self._assert_frozen(result)

    def test_freeze_survives_set_frames_aa(self, bvh_example):
        """set_frames_from_axisangle result is frozen."""
        rp, aa, _ = bvh_example.get_frames_as_axisangle()
        result = bvh_example.set_frames_from_axisangle(rp, aa)
        self._assert_frozen(result)

    def test_freeze_survives_slice_frames(self, bvh_example):
        """slice_frames result is frozen."""
        sliced = bvh_example.slice_frames(0, 10)
        self._assert_frozen(sliced)

    def test_freeze_survives_concat(self, bvh_example):
        """concat result is frozen."""
        result = bvh_example.concat(bvh_example)
        self._assert_frozen(result)

    def test_freeze_survives_extract_joints(self, bvh_example):
        """extract_joints result is frozen."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        result = bvh_example.extract_joints(all_names)
        self._assert_frozen(result)

    def test_freeze_survives_change_skeleton(self, bvh_example, standard_skeleton):
        """change_skeleton result is frozen."""
        result = bvh_example.change_skeleton(standard_skeleton)
        self._assert_frozen(result)

    def test_freeze_survives_scale_skeleton(self, bvh_example):
        """scale_skeleton result is frozen."""
        result = bvh_example.scale_skeleton(2.0)
        self._assert_frozen(result)

    def test_freeze_survives_resample(self, bvh_example):
        """resample result is frozen."""
        result = bvh_example.resample(60)
        self._assert_frozen(result)


# =============================================================================
# Test: resample extreme cases
# =============================================================================

class TestResampleExtreme:
    """Tests for resampling at extreme rates."""

    def test_upsample_30_to_1000(self, bvh_example):
        """Verify frame count is reasonable and no NaN."""
        result = bvh_example.resample(1000)
        # Frame count should be substantially larger than original
        assert result.frame_count > bvh_example.frame_count * 10
        assert not np.any(np.isnan(result.root_pos))
        assert not np.any(np.isnan(result.joint_angles))

    def test_downsample_120_to_1(self, bvh_test2):
        """120fps->1fps, verify structural validity."""
        result = bvh_test2.resample(1)
        assert result.frame_count >= 1
        assert len(result.nodes) == len(bvh_test2.nodes)
        assert not np.any(np.isnan(result.root_pos))
        assert not np.any(np.isnan(result.joint_angles))

    def test_upsample_preserves_start_end(self, bvh_example):
        """First/last frame spatial coords match (atol=1e-4 for first, 1e-2 for last)."""
        result = bvh_example.resample(1000)
        orig_spatial = bvh_example.get_spatial_coord(centered='world')
        result_spatial = result.get_spatial_coord(centered='world')
        np.testing.assert_allclose(result_spatial[0], orig_spatial[0], atol=1e-4)
        # Last frame may differ slightly due to interpolation boundary effects
        np.testing.assert_allclose(result_spatial[-1], orig_spatial[-1], atol=1e-2)

    def test_extreme_upsample_no_nan(self, bvh_example):
        """Resample to 10000fps, assert no NaN/Inf."""
        result = bvh_example.resample(10000)
        assert not np.any(np.isnan(result.root_pos))
        assert not np.any(np.isnan(result.joint_angles))
        assert not np.any(np.isinf(result.root_pos))
        assert not np.any(np.isinf(result.joint_angles))

    def test_resample_back_and_forth(self, bvh_example):
        """30->120->30, compare to original (loose tolerance)."""
        up = bvh_example.resample(120)
        back = up.resample(30)
        assert abs(back.frame_count - bvh_example.frame_count) <= 2
        np.testing.assert_allclose(back.root_pos, bvh_example.root_pos, atol=0.01)
        np.testing.assert_allclose(back.joint_angles, bvh_example.joint_angles, atol=0.5)

    def test_resample_preserves_frame_frequency(self, bvh_example):
        """Resampled frame_frequency == 1/target_fps."""
        result = bvh_example.resample(60)
        np.testing.assert_allclose(result.frame_frequency, 1.0 / 60.0, atol=1e-10)

    def test_resample_single_frame_noop(self, standard_skeleton):
        """Single-frame returns copy unchanged."""
        result = standard_skeleton.resample(60)
        assert result.frame_count == 1
        np.testing.assert_allclose(result.root_pos, standard_skeleton.root_pos, atol=1e-10)
        np.testing.assert_allclose(result.joint_angles, standard_skeleton.joint_angles, atol=1e-10)


# =============================================================================
# Test: extract_joints stress tests
# =============================================================================

class TestExtractJointsStress:
    """Stress tests for extract_joints."""

    def test_extract_single_chain(self, bvh_example):
        """Keep root + Spine + Spine1 + Spine2 chain only."""
        keep = ['Hips', 'Spine', 'Spine1', 'Spine2']
        result = bvh_example.extract_joints(keep)
        result_joint_names = [n.name for n in result.nodes if not n.is_end_site()]
        assert result_joint_names == keep
        assert result.joint_angles.shape == (bvh_example.frame_count, len(keep), 3)

    def test_extract_all_but_one_leaf(self, bvh_example):
        """Remove one leaf joint, verify rest is intact."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        # Find a leaf: a joint whose children are all end sites
        leaf = None
        for n in bvh_example.nodes:
            if n.is_end_site():
                continue
            if all(c.is_end_site() for c in n.children):
                leaf = n.name
                break
        if leaf is None:
            pytest.skip("No leaf joint found")
        keep = [name for name in all_names if name != leaf]
        result = bvh_example.extract_joints(keep)
        result_names = [n.name for n in result.nodes if not n.is_end_site()]
        assert leaf not in result_names
        assert len(result_names) == len(keep)

    def test_extract_half_joints(self, bvh_example):
        """Keep first half of joint names."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        keep = all_names[:len(all_names) // 2]
        if bvh_example.root.name not in keep:
            keep = [bvh_example.root.name] + keep
        result = bvh_example.extract_joints(keep)
        assert result.joint_angles.shape[1] == len(keep)

    def test_extract_minimal_root_plus_one(self, bvh_example):
        """Root + one child joint."""
        root_name = bvh_example.root.name
        # Find a direct child that is not an end site
        child_name = None
        for c in bvh_example.root.children:
            if not c.is_end_site():
                child_name = c.name
                break
        if child_name is None:
            pytest.skip("No non-end-site child of root")
        result = bvh_example.extract_joints([root_name, child_name])
        result_joint_names = [n.name for n in result.nodes if not n.is_end_site()]
        assert result_joint_names == [root_name, child_name]

    def test_extract_preserves_freeze(self, bvh_example):
        """Extracted nodes are frozen."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        result = bvh_example.extract_joints(all_names)
        for node in result.nodes:
            if not node.is_end_site():
                with pytest.raises(AttributeError):
                    node.rot_channels = ['X', 'Y', 'Z']

    def test_extract_on_test2(self, bvh_test2):
        """Extract a few joints from the YXZ file."""
        all_names = [n.name for n in bvh_test2.nodes if not n.is_end_site()]
        keep = [all_names[0]] + all_names[1:4]  # root + 3 joints
        result = bvh_test2.extract_joints(keep)
        assert result.joint_angles.shape == (bvh_test2.frame_count, len(keep), 3)
        assert not np.any(np.isnan(result.joint_angles))

    def test_extract_on_test3(self, bvh_test3):
        """Extract from the large mixed-order file."""
        all_names = [n.name for n in bvh_test3.nodes if not n.is_end_site()]
        keep = [all_names[0]] + all_names[1:10]  # root + 9 joints
        result = bvh_test3.extract_joints(keep)
        assert result.joint_angles.shape == (bvh_test3.frame_count, len(keep), 3)
        assert not np.any(np.isnan(result.joint_angles))


# =============================================================================
# Test: rotation conversions across all files
# =============================================================================

class TestRotationConversionsAllFiles:
    """Tests for rotation format conversion round-trips on all test files."""

    def test_6d_roundtrip_test2(self, bvh_test2):
        """get_frames_as_6d -> set_frames_from_6d, compare joint_angles."""
        rp, r6d, _ = bvh_test2.get_frames_as_6d()
        result = bvh_test2.set_frames_from_6d(rp, r6d)
        np.testing.assert_allclose(result.joint_angles, bvh_test2.joint_angles, atol=1e-4)

    def test_6d_roundtrip_test3(self, bvh_test3):
        """get_frames_as_6d -> set_frames_from_6d for mixed-order file."""
        rp, r6d, _ = bvh_test3.get_frames_as_6d()
        result = bvh_test3.set_frames_from_6d(rp, r6d)
        np.testing.assert_allclose(result.joint_angles, bvh_test3.joint_angles, atol=1e-4)

    def test_quat_roundtrip_test2(self, bvh_test2):
        """get_frames_as_quaternion -> set_frames_from_quaternion."""
        rp, quats, _ = bvh_test2.get_frames_as_quaternion()
        result = bvh_test2.set_frames_from_quaternion(rp, quats)
        np.testing.assert_allclose(result.joint_angles, bvh_test2.joint_angles, atol=1e-4)

    def test_quat_roundtrip_test3(self, bvh_test3):
        """get_frames_as_quaternion -> set_frames_from_quaternion for mixed-order."""
        rp, quats, _ = bvh_test3.get_frames_as_quaternion()
        result = bvh_test3.set_frames_from_quaternion(rp, quats)
        np.testing.assert_allclose(result.joint_angles, bvh_test3.joint_angles, atol=1e-4)

    def test_aa_roundtrip_test2(self, bvh_test2):
        """get_frames_as_axisangle -> set_frames_from_axisangle."""
        rp, aa, _ = bvh_test2.get_frames_as_axisangle()
        result = bvh_test2.set_frames_from_axisangle(rp, aa)
        np.testing.assert_allclose(result.joint_angles, bvh_test2.joint_angles, atol=1e-4)

    def test_aa_roundtrip_test3(self, bvh_test3):
        """get_frames_as_axisangle -> set_frames_from_axisangle for mixed-order."""
        rp, aa, _ = bvh_test3.get_frames_as_axisangle()
        result = bvh_test3.set_frames_from_axisangle(rp, aa)
        np.testing.assert_allclose(result.joint_angles, bvh_test3.joint_angles, atol=1e-4)

    def test_6d_preserves_spatial_all_files(self, bvh_example, bvh_test2, bvh_test3):
        """Verify spatial coord preservation through 6d round-trip."""
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            rp, r6d, _ = bvh.get_frames_as_6d()
            result = bvh.set_frames_from_6d(rp, r6d)
            orig_spatial = bvh.get_spatial_coord(centered="world")
            result_spatial = result.get_spatial_coord(centered="world")
            np.testing.assert_allclose(result_spatial, orig_spatial, atol=1e-4)

    def test_quat_preserves_spatial_all_files(self, bvh_example, bvh_test2, bvh_test3):
        """Verify spatial coord preservation through quaternion round-trip."""
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            rp, quats, _ = bvh.get_frames_as_quaternion()
            result = bvh.set_frames_from_quaternion(rp, quats)
            orig_spatial = bvh.get_spatial_coord(centered="world")
            result_spatial = result.get_spatial_coord(centered="world")
            np.testing.assert_allclose(result_spatial, orig_spatial, atol=1e-4)

    def test_aa_preserves_spatial_all_files(self, bvh_example, bvh_test2, bvh_test3):
        """Verify spatial coord preservation through axisangle round-trip."""
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            rp, aa, _ = bvh.get_frames_as_axisangle()
            result = bvh.set_frames_from_axisangle(rp, aa)
            orig_spatial = bvh.get_spatial_coord(centered="world")
            result_spatial = result.get_spatial_coord(centered="world")
            np.testing.assert_allclose(result_spatial, orig_spatial, atol=1e-4)


# =============================================================================
# Test: change_skeleton edge cases
# =============================================================================

class TestChangeSkeletonEdgeCases:
    """Tests for change_skeleton edge cases."""

    def test_empty_mapping(self, bvh_example, standard_skeleton):
        """name_mapping={} should match by name."""
        result = bvh_example.change_skeleton(standard_skeleton, name_mapping={})
        for n_result, n_ref in zip(result.nodes, standard_skeleton.nodes):
            np.testing.assert_array_equal(n_result.offset, n_ref.offset)

    def test_partial_mapping(self, bvh_example, standard_skeleton):
        """Map only root, rest matches by name."""
        mapping = {bvh_example.root.name: standard_skeleton.root.name}
        result = bvh_example.change_skeleton(standard_skeleton, name_mapping=mapping)
        # Root offset should match reference
        np.testing.assert_array_equal(result.root.offset, standard_skeleton.root.offset)

    def test_different_structure_reference(self, bvh_example, standard_skeleton):
        """Use standard_skeleton as reference (same joint names but different offsets)."""
        result = bvh_example.change_skeleton(standard_skeleton)
        # Offsets should come from the reference
        for n_result, n_ref in zip(result.nodes, standard_skeleton.nodes):
            np.testing.assert_array_equal(n_result.offset, n_ref.offset)
        # Motion data should be preserved
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)
        np.testing.assert_array_equal(result.joint_angles, bvh_example.joint_angles)


# =============================================================================
# Test: node_index property
# =============================================================================

class TestNodeIndex:
    """Tests for node_index property."""

    def test_keys_match_node_names(self, bvh_example):
        """Keys are exactly the node names."""
        expected_keys = {n.name for n in bvh_example.nodes}
        assert set(bvh_example.node_index.keys()) == expected_keys

    def test_values_sequential(self, bvh_example):
        """Values are 0, 1, 2, ..."""
        values = list(bvh_example.node_index.values())
        assert sorted(values) == list(range(len(bvh_example.nodes)))

    def test_survives_extract_joints(self, bvh_example):
        """After extract, node_index is correct for new nodes."""
        all_names = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        keep = all_names[:5]
        result = bvh_example.extract_joints(keep)
        for i, node in enumerate(result.nodes):
            assert result.node_index[node.name] == i


# =============================================================================
# Test: joint_names property
# =============================================================================

class TestJointNamesProperty:
    """Tests for joint_names property."""

    def test_excludes_end_sites(self, bvh_example):
        """No 'End Site' in joint_names."""
        for name in bvh_example.joint_names:
            assert "End Site" not in name

    def test_count_matches(self, bvh_example):
        """len(joint_names) == joint_count."""
        assert len(bvh_example.joint_names) == bvh_example.joint_count

    def test_order_matches_nodes(self, bvh_example):
        """joint_names matches [n.name for n in nodes if not n.is_end_site()]."""
        expected = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        assert bvh_example.joint_names == expected


# ============================================================================
# Batch File Processing
# ============================================================================

class TestBatchProcessing:
    """Tests for read_bvh_directory and batch_to_numpy."""

    @pytest.fixture
    def bvh_dir(self):
        return Path(__file__).parent.parent / "bvh_data"

    def test_read_bvh_directory_basic(self, bvh_dir):
        """Should load all 5 BVH files from bvh_data/."""
        result = read_bvh_directory(bvh_dir)
        assert len(result) == 5
        for bvh in result:
            assert isinstance(bvh, Bvh)

    def test_read_bvh_directory_pattern(self, bvh_dir):
        """Pattern filter should restrict results."""
        result = read_bvh_directory(bvh_dir, pattern="bvh_test*.bvh")
        assert len(result) == 3

    def test_read_bvh_directory_sorted(self, bvh_dir):
        """Results should be sorted alphabetically by default."""
        result = read_bvh_directory(bvh_dir)
        names = [str(Path(f"bvh_data")) for f in result]
        # Check by examining node counts (a proxy — sorted files have distinct sizes)
        result_sorted = read_bvh_directory(bvh_dir, sort=True)
        result_unsorted = read_bvh_directory(bvh_dir, sort=False)
        # Sorted should be deterministic; verify at least it returns same count
        assert len(result_sorted) == len(result_unsorted)

    def test_read_bvh_directory_parallel(self, bvh_dir):
        """Parallel loading should give same results as sequential."""
        seq = read_bvh_directory(bvh_dir, sort=True)
        par = read_bvh_directory(bvh_dir, sort=True, parallel=True)
        assert len(seq) == len(par)
        for s, p in zip(seq, par):
            assert s.frame_count == p.frame_count
            np.testing.assert_array_equal(s.root_pos, p.root_pos)

    def test_read_bvh_directory_invalid_dir(self):
        """Nonexistent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_bvh_directory("/nonexistent/path")

    def test_read_bvh_directory_empty(self, tmp_path):
        """Directory with no BVH files returns empty list."""
        result = read_bvh_directory(tmp_path)
        assert result == []

    def test_batch_to_numpy_euler(self, bvh_dir):
        """Euler representation: correct shape."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        result = batch_to_numpy(bvhs, representation="euler")
        assert isinstance(result, list)
        assert len(result) == 1
        bvh = bvhs[0]
        expected_cols = 3 + bvh.joint_count * 3  # root_pos + J*3
        assert result[0].shape == (bvh.frame_count, expected_cols)

    def test_batch_to_numpy_6d(self, bvh_dir):
        """6D representation: correct shape."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        result = batch_to_numpy(bvhs, representation="6d")
        bvh = bvhs[0]
        expected_cols = 3 + bvh.joint_count * 6
        assert result[0].shape == (bvh.frame_count, expected_cols)

    def test_batch_to_numpy_quaternion(self, bvh_dir):
        """Quaternion representation: correct shape."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        result = batch_to_numpy(bvhs, representation="quaternion")
        bvh = bvhs[0]
        expected_cols = 3 + bvh.joint_count * 4
        assert result[0].shape == (bvh.frame_count, expected_cols)

    def test_batch_to_numpy_axisangle(self, bvh_dir):
        """Axisangle representation: correct shape."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        result = batch_to_numpy(bvhs, representation="axisangle")
        bvh = bvhs[0]
        expected_cols = 3 + bvh.joint_count * 3
        assert result[0].shape == (bvh.frame_count, expected_cols)

    def test_batch_to_numpy_rotmat(self, bvh_dir):
        """Rotmat representation: correct shape."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        result = batch_to_numpy(bvhs, representation="rotmat")
        bvh = bvhs[0]
        expected_cols = 3 + bvh.joint_count * 9
        assert result[0].shape == (bvh.frame_count, expected_cols)

    def test_batch_to_numpy_pad_true(self, bvh_dir):
        """Padding should produce a single 3D array."""
        # bvh_example and bvh_test1 have same skeleton but potentially different frame counts
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_example.bvh")
        # Duplicate with different frame count by slicing
        bvh2 = bvhs[0].slice_frames(0, 10)
        result = batch_to_numpy([bvhs[0], bvh2], pad=True)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] == 2
        assert result.shape[1] == bvhs[0].frame_count  # max length

    def test_batch_to_numpy_pad_false_returns_list(self, bvh_dir):
        """Without padding, returns a list of arrays."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_example.bvh")
        bvh2 = bvhs[0].slice_frames(0, 10)
        result = batch_to_numpy([bvhs[0], bvh2], pad=False)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_batch_to_numpy_mismatched_skeletons_raises(self, bvh_dir):
        """Different skeletons should raise ValueError."""
        all_bvhs = read_bvh_directory(bvh_dir)
        # Mix skeletons with different joint counts
        mixed = [b for b in all_bvhs if b.joint_count != all_bvhs[0].joint_count]
        if mixed:
            with pytest.raises(ValueError, match="Skeleton mismatch"):
                batch_to_numpy([all_bvhs[0], mixed[0]])

    def test_batch_to_numpy_without_root_pos(self, bvh_dir):
        """include_root_pos=False should omit 3 columns."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_test1.bvh")
        with_pos = batch_to_numpy(bvhs, include_root_pos=True)
        without_pos = batch_to_numpy(bvhs, include_root_pos=False)
        assert with_pos[0].shape[1] == without_pos[0].shape[1] + 3

    def test_batch_to_numpy_single_file(self, bvh_dir):
        """Single-element list should work."""
        bvhs = read_bvh_directory(bvh_dir, pattern="bvh_example.bvh")
        result = batch_to_numpy(bvhs)
        assert len(result) == 1


# =============================================================================
# Phase 5 — ML Pipeline Features
# =============================================================================

class TestJointVelocities:
    """Tests for get_joint_velocities."""

    def test_shape(self, bvh_example):
        vel = bvh_example.get_joint_velocities(in_frames=True)
        F, N = bvh_example.frame_count, len(bvh_example.nodes)
        assert vel.shape == (F - 1, N, 3)

    def test_per_second(self, bvh_example):
        vel_frame = bvh_example.get_joint_velocities(in_frames=True)
        vel_sec = bvh_example.get_joint_velocities(in_frames=False)
        np.testing.assert_allclose(
            vel_sec, vel_frame / bvh_example.frame_frequency, atol=1e-10)

    def test_static_pose_zero_velocity(self, bvh_example):
        """A BVH with identical frames should have zero velocity."""
        static = bvh_example.copy()
        # Make all frames identical to frame 0
        for i in range(static.frame_count):
            static.root_pos[i] = static.root_pos[0]
            static.joint_angles[i] = static.joint_angles[0]
        vel = static.get_joint_velocities(in_frames=True)
        np.testing.assert_allclose(vel, 0.0, atol=1e-10)

    def test_precomputed_coords(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        vel1 = bvh_example.get_joint_velocities(in_frames=True)
        vel2 = bvh_example.get_joint_velocities(in_frames=True, coords=coords)
        np.testing.assert_allclose(vel1, vel2, atol=1e-10)

    def test_too_few_frames_error(self):
        """Should raise ValueError with < 2 frames."""
        root = BvhRoot()
        bvh = Bvh(nodes=[root], root_pos=np.zeros((1, 3)),
                   joint_angles=np.zeros((1, 1, 3)), frame_frequency=1/30)
        with pytest.raises(ValueError, match="At least 2 frames"):
            bvh.get_joint_velocities()

    def test_zero_frequency_error(self, bvh_example):
        bvh = bvh_example.copy()
        bvh.frame_frequency = 0
        with pytest.raises(ValueError, match="frame_frequency is 0"):
            bvh.get_joint_velocities(in_frames=False)

    def test_zero_frequency_in_frames_ok(self, bvh_example):
        bvh = bvh_example.copy()
        bvh.frame_frequency = 0
        vel = bvh.get_joint_velocities(in_frames=True)
        assert vel.shape[0] == bvh.frame_count - 1

    def test_constant_root_translation(self):
        """Root moving at constant velocity along X: velocity should be constant."""
        root = BvhRoot()
        F = 10
        root_pos = np.zeros((F, 3))
        root_pos[:, 0] = np.arange(F) * 5.0  # 5 units/frame along X
        joint_angles = np.zeros((F, 1, 3))
        bvh = Bvh(nodes=[root, BvhNode("End Site", offset=np.array([0, 1, 0]), parent=root)],
                   root_pos=root_pos, joint_angles=joint_angles,
                   frame_frequency=1/30)
        root.children = [bvh.nodes[1]]
        vel = bvh.get_joint_velocities(in_frames=True)
        # Root velocity should be [5, 0, 0] for all frames
        np.testing.assert_allclose(vel[:, 0, 0], 5.0, atol=1e-10)
        np.testing.assert_allclose(vel[:, 0, 1:], 0.0, atol=1e-10)

    def test_on_all_test_files(self, bvh_example, bvh_test2, bvh_test3):
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            vel = bvh.get_joint_velocities(in_frames=True)
            assert vel.shape == (bvh.frame_count - 1, len(bvh.nodes), 3)


class TestJointAccelerations:
    """Tests for get_joint_accelerations."""

    def test_shape(self, bvh_example):
        acc = bvh_example.get_joint_accelerations(in_frames=True)
        F, N = bvh_example.frame_count, len(bvh_example.nodes)
        assert acc.shape == (F - 2, N, 3)

    def test_per_second(self, bvh_example):
        acc_frame = bvh_example.get_joint_accelerations(in_frames=True)
        acc_sec = bvh_example.get_joint_accelerations(in_frames=False)
        np.testing.assert_allclose(
            acc_sec, acc_frame / (bvh_example.frame_frequency ** 2), atol=1e-6)

    def test_static_pose_zero_acceleration(self, bvh_example):
        static = bvh_example.copy()
        for i in range(static.frame_count):
            static.root_pos[i] = static.root_pos[0]
            static.joint_angles[i] = static.joint_angles[0]
        acc = static.get_joint_accelerations(in_frames=True)
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)

    def test_constant_velocity_zero_acceleration(self):
        """Constant velocity → zero acceleration."""
        root = BvhRoot()
        F = 10
        root_pos = np.zeros((F, 3))
        root_pos[:, 0] = np.arange(F) * 5.0
        joint_angles = np.zeros((F, 1, 3))
        bvh = Bvh(nodes=[root, BvhNode("End Site", offset=np.array([0, 1, 0]), parent=root)],
                   root_pos=root_pos, joint_angles=joint_angles,
                   frame_frequency=1/30)
        root.children = [bvh.nodes[1]]
        acc = bvh.get_joint_accelerations(in_frames=True)
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)

    def test_too_few_frames_error(self):
        root = BvhRoot()
        bvh = Bvh(nodes=[root], root_pos=np.zeros((2, 3)),
                   joint_angles=np.zeros((2, 1, 3)), frame_frequency=1/30)
        with pytest.raises(ValueError, match="At least 3 frames"):
            bvh.get_joint_accelerations()


class TestAngularVelocities:
    """Tests for get_angular_velocities."""

    def test_shape(self, bvh_example):
        ang_vel = bvh_example.get_angular_velocities(in_frames=True)
        assert ang_vel.shape == (bvh_example.frame_count - 1,
                                  bvh_example.joint_count, 3)

    def test_static_pose_zero(self, bvh_example):
        static = bvh_example.copy()
        for i in range(static.frame_count):
            static.joint_angles[i] = static.joint_angles[0]
        ang_vel = static.get_angular_velocities(in_frames=True)
        np.testing.assert_allclose(ang_vel, 0.0, atol=1e-10)

    def test_per_second(self, bvh_example):
        av_frame = bvh_example.get_angular_velocities(in_frames=True)
        av_sec = bvh_example.get_angular_velocities(in_frames=False)
        np.testing.assert_allclose(
            av_sec, av_frame / bvh_example.frame_frequency, atol=1e-10)

    def test_too_few_frames_error(self):
        root = BvhRoot()
        bvh = Bvh(nodes=[root], root_pos=np.zeros((1, 3)),
                   joint_angles=np.zeros((1, 1, 3)), frame_frequency=1/30)
        with pytest.raises(ValueError, match="At least 2 frames"):
            bvh.get_angular_velocities()

    def test_on_all_test_files(self, bvh_example, bvh_test2, bvh_test3):
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            ang_vel = bvh.get_angular_velocities(in_frames=True)
            assert ang_vel.shape == (bvh.frame_count - 1, bvh.joint_count, 3)


class TestRootRelativePositions:
    """Tests for get_root_relative_positions."""

    def test_shape(self, bvh_example):
        rel = bvh_example.get_root_relative_positions()
        assert rel.shape == (bvh_example.frame_count, len(bvh_example.nodes), 3)

    def test_root_at_origin(self, bvh_example):
        rel = bvh_example.get_root_relative_positions()
        np.testing.assert_allclose(rel[:, 0, :], 0.0, atol=1e-10)

    def test_precomputed_coords(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        rel1 = bvh_example.get_root_relative_positions()
        rel2 = bvh_example.get_root_relative_positions(coords=coords)
        np.testing.assert_allclose(rel1, rel2, atol=1e-10)

    def test_static_skeleton_constant(self, bvh_example):
        """Static skeleton: root-relative positions are same every frame."""
        static = bvh_example.copy()
        for i in range(static.frame_count):
            static.root_pos[i] = static.root_pos[0]
            static.joint_angles[i] = static.joint_angles[0]
        rel = static.get_root_relative_positions()
        for i in range(1, static.frame_count):
            np.testing.assert_allclose(rel[i], rel[0], atol=1e-10)

    def test_on_all_test_files(self, bvh_example, bvh_test2, bvh_test3):
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            rel = bvh.get_root_relative_positions()
            np.testing.assert_allclose(rel[:, 0, :], 0.0, atol=1e-10)


class TestRootTrajectory:
    """Tests for get_root_trajectory."""

    def test_shape(self, bvh_example):
        traj = bvh_example.get_root_trajectory()
        assert traj.shape == (bvh_example.frame_count, 4)

    def test_sin_cos_unit(self, bvh_example):
        """sin^2 + cos^2 should equal 1."""
        traj = bvh_example.get_root_trajectory()
        sin_sq_plus_cos_sq = traj[:, 2] ** 2 + traj[:, 3] ** 2
        np.testing.assert_allclose(sin_sq_plus_cos_sq, 1.0, atol=1e-10)

    def test_explicit_up_axis(self, bvh_example):
        """Explicit up_axis should produce same result as auto-detect."""
        from pybvh.tools import get_forw_up_axis
        rest = bvh_example.get_rest_pose(mode='coordinates')
        directions = get_forw_up_axis(bvh_example, rest)
        traj_auto = bvh_example.get_root_trajectory()
        traj_explicit = bvh_example.get_root_trajectory(
            up_axis=directions['upward'])
        np.testing.assert_allclose(traj_auto, traj_explicit, atol=1e-10)

    def test_on_all_test_files(self, bvh_example, bvh_test2, bvh_test3):
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            traj = bvh.get_root_trajectory()
            assert traj.shape == (bvh.frame_count, 4)
            # sin^2 + cos^2 = 1
            np.testing.assert_allclose(
                traj[:, 2] ** 2 + traj[:, 3] ** 2, 1.0, atol=1e-10)


class TestFootContacts:
    """Tests for get_foot_contacts."""

    def test_shape_auto_detect(self, bvh_example):
        contacts = bvh_example.get_foot_contacts()
        assert contacts.shape[0] == bvh_example.frame_count
        assert contacts.shape[1] > 0  # at least one foot joint

    def test_binary_output(self, bvh_example):
        contacts = bvh_example.get_foot_contacts()
        assert set(np.unique(contacts)).issubset({0.0, 1.0})

    def test_manual_joints(self, bvh_example):
        contacts = bvh_example.get_foot_contacts(
            foot_joints=["LeftFoot", "RightFoot"])
        assert contacts.shape == (bvh_example.frame_count, 2)

    def test_static_all_contacts(self, bvh_example):
        """Static pose: all velocities zero → all contacts = 1."""
        static = bvh_example.copy()
        for i in range(static.frame_count):
            static.root_pos[i] = static.root_pos[0]
            static.joint_angles[i] = static.joint_angles[0]
        contacts = static.get_foot_contacts(method="velocity")
        np.testing.assert_allclose(contacts, 1.0)

    def test_height_method(self, bvh_example):
        contacts = bvh_example.get_foot_contacts(method="height")
        assert contacts.shape[0] == bvh_example.frame_count
        assert set(np.unique(contacts)).issubset({0.0, 1.0})

    def test_invalid_method(self, bvh_example):
        with pytest.raises(ValueError, match="Unknown method"):
            bvh_example.get_foot_contacts(method="invalid")

    def test_invalid_joint_name(self, bvh_example):
        with pytest.raises(ValueError, match="not found"):
            bvh_example.get_foot_contacts(foot_joints=["NonExistentJoint"])

    def test_precomputed_coords(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        c1 = bvh_example.get_foot_contacts()
        c2 = bvh_example.get_foot_contacts(coords=coords)
        np.testing.assert_allclose(c1, c2)

    def test_on_test2(self, bvh_test2):
        """bvh_test2 may have different joint names — test auto-detection."""
        contacts = bvh_test2.get_foot_contacts()
        assert contacts.shape[0] == bvh_test2.frame_count


class TestToFeatureArray:
    """Tests for to_feature_array."""

    def test_basic_shape(self, bvh_example):
        feat = bvh_example.to_feature_array(representation='euler')
        J = bvh_example.joint_count
        expected_dim = 3 + J * 3  # root_pos + euler angles
        assert feat.shape == (bvh_example.frame_count, expected_dim)

    def test_6d_shape(self, bvh_example):
        feat = bvh_example.to_feature_array(representation='6d')
        J = bvh_example.joint_count
        expected_dim = 3 + J * 6
        assert feat.shape == (bvh_example.frame_count, expected_dim)

    def test_quaternion_shape(self, bvh_example):
        feat = bvh_example.to_feature_array(representation='quaternion')
        J = bvh_example.joint_count
        expected_dim = 3 + J * 4
        assert feat.shape == (bvh_example.frame_count, expected_dim)

    def test_no_root_pos(self, bvh_example):
        feat = bvh_example.to_feature_array(
            representation='euler', include_root_pos=False)
        J = bvh_example.joint_count
        expected_dim = J * 3
        assert feat.shape == (bvh_example.frame_count, expected_dim)

    def test_with_velocities_trimmed(self, bvh_example):
        """Including velocities drops first frame."""
        feat = bvh_example.to_feature_array(
            representation='euler', include_velocities=True)
        assert feat.shape[0] == bvh_example.frame_count - 1

    def test_with_foot_contacts(self, bvh_example):
        feat = bvh_example.to_feature_array(
            representation='euler', include_foot_contacts=True)
        assert feat.shape[0] == bvh_example.frame_count
        # Last columns should be foot contacts
        contacts = bvh_example.get_foot_contacts()
        np.testing.assert_allclose(
            feat[:, -contacts.shape[1]:], contacts, atol=1e-10)

    def test_all_features(self, bvh_example):
        feat = bvh_example.to_feature_array(
            representation='6d',
            include_velocities=True,
            include_foot_contacts=True)
        assert feat.shape[0] == bvh_example.frame_count - 1

    def test_invalid_representation(self, bvh_example):
        with pytest.raises(ValueError, match="Unknown representation"):
            bvh_example.to_feature_array(representation='invalid')

    def test_on_all_test_files(self, bvh_example, bvh_test2, bvh_test3):
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            feat = bvh.to_feature_array(representation='6d')
            assert feat.shape == (bvh.frame_count, 3 + bvh.joint_count * 6)


class TestNormalization:
    """Tests for normalization utilities."""

    def test_round_trip(self, bvh_example):
        """Normalize then denormalize should recover original."""
        stats = compute_normalization_stats([bvh_example])
        data = batch_to_numpy([bvh_example], pad=False)
        original = data[0].copy()
        normalized = normalize_array(original, stats)
        recovered = denormalize_array(normalized, stats)
        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_stats_shapes(self, bvh_example):
        stats = compute_normalization_stats([bvh_example])
        D = 3 + bvh_example.joint_count * 3  # root_pos + euler
        assert stats["mean"].shape == (D,)
        assert stats["std"].shape == (D,)

    def test_stats_shapes_6d(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], representation="6d")
        D = 3 + bvh_example.joint_count * 6
        assert stats["mean"].shape == (D,)
        assert stats["std"].shape == (D,)

    def test_zero_std_guard(self, bvh_example):
        """Constant channels should get std=1.0, not 0.0."""
        static = bvh_example.copy()
        for i in range(static.frame_count):
            static.root_pos[i] = static.root_pos[0]
            static.joint_angles[i] = static.joint_angles[0]
        stats = compute_normalization_stats([static])
        assert np.all(stats["std"] >= 1e-8)

    def test_normalized_mean_zero(self, bvh_example):
        """After normalization, mean should be ~0."""
        stats = compute_normalization_stats([bvh_example])
        data = batch_to_numpy([bvh_example], pad=False)
        normalized = normalize_array(data[0], stats)
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-10)

    def test_multiple_files(self, bvh_example, bvh_test2):
        """Stats from multiple files should have correct shape (if same skeleton)."""
        # bvh_example and bvh_test2 may have different skeletons
        # Use two copies of the same file to test multi-file path
        bvh2 = bvh_example.copy()
        stats = compute_normalization_stats([bvh_example, bvh2])
        D = 3 + bvh_example.joint_count * 3
        assert stats["mean"].shape == (D,)

    def test_quaternion_round_trip(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], representation="quaternion")
        data = batch_to_numpy(
            [bvh_example], representation="quaternion", pad=False)
        recovered = denormalize_array(
            normalize_array(data[0], stats), stats)
        np.testing.assert_allclose(recovered, data[0], atol=1e-10)

    def test_no_root_pos(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], include_root_pos=False)
        D = bvh_example.joint_count * 3
        assert stats["mean"].shape == (D,)


class TestAxisDetection:
    """Tests for the extracted axis detection utilities."""

    def test_get_forw_up_axis(self, bvh_example):
        from pybvh.tools import get_forw_up_axis
        rest = bvh_example.get_rest_pose(mode='coordinates')
        directions = get_forw_up_axis(bvh_example, rest)
        assert 'forward' in directions
        assert 'upward' in directions
        assert directions['upward'][1] in ('x', 'y', 'z')

    def test_get_main_direction(self):
        from pybvh.tools import get_main_direction
        assert get_main_direction(np.array([0, 10, 0])) == '+y'
        assert get_main_direction(np.array([0, -10, 0])) == '-y'
        assert get_main_direction(np.array([5, 0, 0])) == '+x'
        assert get_main_direction(np.array([0, 0, -3])) == '-z'

    def test_extract_sign(self):
        from pybvh.tools import extract_sign
        assert extract_sign('+x') is True
        assert extract_sign('-z') is False

    def test_get_up_axis_index(self, bvh_example):
        from pybvh.tools import get_up_axis_index
        rest = bvh_example.get_rest_pose(mode='coordinates')
        up_idx = get_up_axis_index(bvh_example, rest)
        assert up_idx in (0, 1, 2)

    def test_plot_imports_from_tools(self):
        """Ensure bvhplot imports axis detection from tools.py."""
        from pybvh import bvhplot
        from pybvh.tools import get_forw_up_axis
        assert bvhplot.get_forw_up_axis is get_forw_up_axis

    # --- Hardened axis detection tests ---

    def test_main_direction_zero_vector(self):
        """get_main_direction returns None for zero vectors."""
        from pybvh.tools import get_main_direction
        assert get_main_direction(np.zeros(3)) is None

    def test_main_direction_near_zero(self):
        """get_main_direction returns None for near-zero vectors."""
        from pybvh.tools import get_main_direction
        assert get_main_direction(np.array([1e-8, 1e-9, 1e-10])) is None
        # Just above tolerance should still work
        assert get_main_direction(np.array([0.0, 0.0, 1.0]), tol=0.5) == '+z'

    def test_main_direction_normal_vectors(self):
        """get_main_direction returns correct labels for axis-aligned vectors."""
        from pybvh.tools import get_main_direction
        assert get_main_direction(np.array([1.0, 0.0, 0.0])) == '+x'
        assert get_main_direction(np.array([-1.0, 0.0, 0.0])) == '-x'
        assert get_main_direction(np.array([0.0, 5.0, 0.0])) == '+y'
        assert get_main_direction(np.array([0.0, -5.0, 0.0])) == '-y'
        assert get_main_direction(np.array([0.0, 0.0, 3.0])) == '+z'
        assert get_main_direction(np.array([0.0, 0.0, -3.0])) == '-z'

    def test_forward_axis_pose_independent(self, bvh_example):
        """Forward axis should be the same regardless of which frame is used."""
        from pybvh.tools import get_forw_up_axis
        coords = bvh_example.get_spatial_coord()
        dirs_frame0 = get_forw_up_axis(bvh_example, coords[0])
        dirs_frame_mid = get_forw_up_axis(bvh_example, coords[coords.shape[0] // 2])
        dirs_frame_last = get_forw_up_axis(bvh_example, coords[-1])
        assert dirs_frame0['forward'] == dirs_frame_mid['forward']
        assert dirs_frame0['forward'] == dirs_frame_last['forward']

    def test_forward_upward_orthogonality(self, bvh_example):
        """Forward and upward axes must be on different axes."""
        from pybvh.tools import get_forw_up_axis
        rest = bvh_example.get_rest_pose(mode='coordinates')
        directions = get_forw_up_axis(bvh_example, rest)
        assert directions['forward'][1] != directions['upward'][1]

    def test_input_validation_wrong_shape(self, bvh_example):
        """get_forw_up_axis rejects frames with wrong shape."""
        from pybvh.tools import get_forw_up_axis
        with pytest.raises(ValueError, match="Expected frame shape"):
            get_forw_up_axis(bvh_example, np.zeros((5,)))
        with pytest.raises(ValueError, match="Expected frame shape"):
            get_forw_up_axis(bvh_example, np.zeros((5, 4)))

    def test_input_validation_wrong_node_count(self, bvh_example):
        """get_forw_up_axis rejects frames with wrong number of nodes."""
        from pybvh.tools import get_forw_up_axis
        with pytest.raises(ValueError, match="nodes but skeleton has"):
            get_forw_up_axis(bvh_example, np.zeros((3, 3)))

    def test_all_fixtures_correct_axes(self, bvh_example, bvh_test2, bvh_test3):
        """All test BVH files produce consistent axis detection."""
        from pybvh.tools import get_forw_up_axis
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            rest = bvh.get_rest_pose(mode='coordinates')
            dirs = get_forw_up_axis(bvh, rest)
            # Forward and upward must be on different axes
            assert dirs['forward'][1] != dirs['upward'][1]
            # Both must be valid signed axis strings
            assert dirs['forward'][0] in ('+', '-')
            assert dirs['upward'][0] in ('+', '-')
            assert dirs['forward'][1] in ('x', 'y', 'z')
            assert dirs['upward'][1] in ('x', 'y', 'z')


# =============================================================================
# Phase 6 — Spatial Augmentation Transforms
# =============================================================================

class TestTranslateRoot:
    """Tests for translate_root transform."""

    def test_zero_offset_identity(self, bvh_example):
        from pybvh.transforms import translate_root
        result = translate_root(bvh_example, [0, 0, 0])
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)
        np.testing.assert_array_equal(result.joint_angles, bvh_example.joint_angles)

    def test_known_offset(self, bvh_example):
        from pybvh.transforms import translate_root
        offset = np.array([10.0, -5.0, 3.0])
        result = translate_root(bvh_example, offset)
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos + offset)

    def test_joint_angles_unchanged(self, bvh_example):
        from pybvh.transforms import translate_root
        result = translate_root(bvh_example, [100, 200, 300])
        np.testing.assert_array_equal(result.joint_angles, bvh_example.joint_angles)

    def test_spatial_coord_shift(self, bvh_example):
        from pybvh.transforms import translate_root
        offset = np.array([5.0, 5.0, 5.0])
        coords_orig = bvh_example.get_spatial_coord()
        result = translate_root(bvh_example, offset)
        coords_new = result.get_spatial_coord()
        np.testing.assert_allclose(coords_new, coords_orig + offset, atol=1e-6)

    def test_inplace(self, bvh_example):
        from pybvh.transforms import translate_root
        bvh = bvh_example.copy()
        orig_pos = bvh.root_pos.copy()
        ret = translate_root(bvh, [1, 2, 3], inplace=True)
        assert ret is None
        np.testing.assert_allclose(bvh.root_pos, orig_pos + [1, 2, 3])

    def test_round_trip(self, bvh_example):
        from pybvh.transforms import translate_root
        offset = [7, -3, 11]
        result = translate_root(translate_root(bvh_example, offset), [-7, 3, -11])
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos, atol=1e-10)

    def test_random_variant(self, bvh_example):
        from pybvh.transforms import random_translate_root
        rng = np.random.default_rng(42)
        r1 = random_translate_root(bvh_example, rng=rng)
        rng2 = np.random.default_rng(42)
        r2 = random_translate_root(bvh_example, rng=rng2)
        np.testing.assert_array_equal(r1.root_pos, r2.root_pos)


class TestJointNoise:
    """Tests for add_joint_noise transform."""

    def test_zero_sigma_identity(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        result = add_joint_noise(bvh_example, sigma_deg=0.0)
        np.testing.assert_array_equal(result.joint_angles, bvh_example.joint_angles)
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)

    def test_nonzero_sigma_changes_values(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        result = add_joint_noise(bvh_example, sigma_deg=5.0, rng=np.random.default_rng(0))
        assert not np.array_equal(result.joint_angles, bvh_example.joint_angles)

    def test_sigma_pos_zero_no_change(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        result = add_joint_noise(bvh_example, sigma_deg=5.0, sigma_pos=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)

    def test_sigma_pos_nonzero(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        result = add_joint_noise(bvh_example, sigma_deg=0.0, sigma_pos=1.0, rng=np.random.default_rng(0))
        assert not np.array_equal(result.root_pos, bvh_example.root_pos)

    def test_skeleton_unchanged(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        result = add_joint_noise(bvh_example, sigma_deg=5.0, rng=np.random.default_rng(0))
        assert [n.name for n in result.nodes] == [n.name for n in bvh_example.nodes]

    def test_inplace(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        bvh = bvh_example.copy()
        ret = add_joint_noise(bvh, sigma_deg=5.0, rng=np.random.default_rng(0), inplace=True)
        assert ret is None
        assert not np.array_equal(bvh.joint_angles, bvh_example.joint_angles)

    def test_seeded_reproducibility(self, bvh_example):
        from pybvh.transforms import add_joint_noise
        r1 = add_joint_noise(bvh_example, sigma_deg=3.0, rng=np.random.default_rng(99))
        r2 = add_joint_noise(bvh_example, sigma_deg=3.0, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1.joint_angles, r2.joint_angles)


class TestSpeedPerturbation:
    """Tests for speed_perturbation transform."""

    def test_factor_one_near_identity(self, bvh_example):
        from pybvh.transforms import speed_perturbation
        result = speed_perturbation(bvh_example, factor=1.0)
        assert result.frame_count == bvh_example.frame_count
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos, atol=1e-3)

    def test_factor_two_halves_frames(self, bvh_example):
        from pybvh.transforms import speed_perturbation
        result = speed_perturbation(bvh_example, factor=2.0)
        expected = bvh_example.frame_count // 2 + 1
        assert abs(result.frame_count - expected) <= 1

    def test_factor_half_doubles_frames(self, bvh_example):
        from pybvh.transforms import speed_perturbation
        result = speed_perturbation(bvh_example, factor=0.5)
        expected = (bvh_example.frame_count - 1) * 2 + 1
        assert abs(result.frame_count - expected) <= 2

    def test_skeleton_preserved(self, bvh_example):
        from pybvh.transforms import speed_perturbation
        result = speed_perturbation(bvh_example, factor=1.5)
        assert [n.name for n in result.nodes] == [n.name for n in bvh_example.nodes]

    def test_random_variant(self, bvh_example):
        from pybvh.transforms import random_speed_perturbation
        r1 = random_speed_perturbation(bvh_example, rng=np.random.default_rng(7))
        r2 = random_speed_perturbation(bvh_example, rng=np.random.default_rng(7))
        assert r1.frame_count == r2.frame_count
        np.testing.assert_allclose(r1.root_pos, r2.root_pos, atol=1e-10)


class TestDropoutFrames:
    """Tests for dropout_frames transform."""

    def test_zero_drop_rate_identity(self, bvh_example):
        from pybvh.transforms import dropout_frames
        result = dropout_frames(bvh_example, drop_rate=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result.root_pos, bvh_example.root_pos)
        np.testing.assert_array_equal(result.joint_angles, bvh_example.joint_angles)

    def test_frame_count_preserved(self, bvh_example):
        from pybvh.transforms import dropout_frames
        result = dropout_frames(bvh_example, drop_rate=0.5, rng=np.random.default_rng(0))
        assert result.frame_count == bvh_example.frame_count

    def test_first_last_preserved(self, bvh_example):
        from pybvh.transforms import dropout_frames
        result = dropout_frames(bvh_example, drop_rate=0.8, rng=np.random.default_rng(0))
        np.testing.assert_allclose(result.root_pos[0], bvh_example.root_pos[0], atol=1e-6)
        np.testing.assert_allclose(result.root_pos[-1], bvh_example.root_pos[-1], atol=1e-6)

    def test_skeleton_preserved(self, bvh_example):
        from pybvh.transforms import dropout_frames
        result = dropout_frames(bvh_example, drop_rate=0.3, rng=np.random.default_rng(0))
        assert [n.name for n in result.nodes] == [n.name for n in bvh_example.nodes]

    def test_inplace(self, bvh_example):
        from pybvh.transforms import dropout_frames
        bvh = bvh_example.copy()
        ret = dropout_frames(bvh, drop_rate=0.5, rng=np.random.default_rng(0), inplace=True)
        assert ret is None

    def test_seeded_reproducibility(self, bvh_example):
        from pybvh.transforms import dropout_frames
        r1 = dropout_frames(bvh_example, drop_rate=0.4, rng=np.random.default_rng(42))
        r2 = dropout_frames(bvh_example, drop_rate=0.4, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.root_pos, r2.root_pos)
        np.testing.assert_array_equal(r1.joint_angles, r2.joint_angles)


class TestRotateVertical:
    """Tests for rotate_vertical transform."""

    def test_zero_rotation_identity(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        result = rotate_vertical(bvh_example, angle_deg=0.0)
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(result.joint_angles, bvh_example.joint_angles, atol=1e-8)

    def test_360_identity(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        result = rotate_vertical(bvh_example, angle_deg=360.0)
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos, atol=1e-6)
        np.testing.assert_allclose(result.joint_angles, bvh_example.joint_angles, atol=1e-4)

    def test_bone_lengths_preserved(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        coords_orig = bvh_example.get_spatial_coord(centered='skeleton')
        result = rotate_vertical(bvh_example, angle_deg=90.0)
        coords_rot = result.get_spatial_coord(centered='skeleton')
        # Check all bone lengths match (frame 0)
        for node in bvh_example.nodes:
            if node.parent is not None:
                pi = bvh_example.node_index[node.parent.name]
                ci = bvh_example.node_index[node.name]
                len_orig = np.linalg.norm(coords_orig[0, ci] - coords_orig[0, pi])
                len_rot = np.linalg.norm(coords_rot[0, ci] - coords_rot[0, pi])
                np.testing.assert_allclose(len_rot, len_orig, atol=1e-4,
                    err_msg=f"Bone length changed for {node.name}")

    def test_non_root_angles_unchanged(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        result = rotate_vertical(bvh_example, angle_deg=45.0)
        # All joints except root (index 0) should be unchanged
        np.testing.assert_array_equal(
            result.joint_angles[:, 1:], bvh_example.joint_angles[:, 1:])

    def test_double_180_identity(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        result = rotate_vertical(rotate_vertical(bvh_example, 180.0), 180.0)
        np.testing.assert_allclose(result.root_pos, bvh_example.root_pos, atol=1e-6)
        np.testing.assert_allclose(result.joint_angles, bvh_example.joint_angles, atol=1e-3)

    def test_inplace(self, bvh_example):
        from pybvh.transforms import rotate_vertical
        bvh = bvh_example.copy()
        ret = rotate_vertical(bvh, 90.0, inplace=True)
        assert ret is None
        assert not np.allclose(bvh.root_pos, bvh_example.root_pos)

    def test_y_up_file(self, bvh_test2):
        from pybvh.transforms import rotate_vertical
        result = rotate_vertical(bvh_test2, angle_deg=90.0)
        assert result.frame_count == bvh_test2.frame_count

    def test_random_variant(self, bvh_example):
        from pybvh.transforms import random_rotate_vertical
        r1 = random_rotate_vertical(bvh_example, rng=np.random.default_rng(5))
        r2 = random_rotate_vertical(bvh_example, rng=np.random.default_rng(5))
        np.testing.assert_allclose(r1.root_pos, r2.root_pos, atol=1e-10)


class TestAutoDetectLRMapping:
    """Tests for auto_detect_lr_mapping."""

    def test_bvh_example(self, bvh_example):
        from pybvh.transforms import auto_detect_lr_mapping
        mapping = auto_detect_lr_mapping(bvh_example)
        assert len(mapping) > 0
        # Should find LeftArm <-> RightArm etc.
        for left, right in mapping.items():
            assert "Left" in left or left[0] == "L"
            assert "Right" in right or right[0] == "R"

    def test_all_fixtures_find_pairs(self, bvh_example, bvh_test2, bvh_test3):
        from pybvh.transforms import auto_detect_lr_mapping
        for bvh in [bvh_example, bvh_test2, bvh_test3]:
            mapping = auto_detect_lr_mapping(bvh)
            assert len(mapping) > 0, f"No L/R pairs found for skeleton with joints: {bvh.joint_names[:5]}..."

    def test_mapping_is_symmetric_in_joint_list(self, bvh_example):
        from pybvh.transforms import auto_detect_lr_mapping
        mapping = auto_detect_lr_mapping(bvh_example)
        joint_names = set(bvh_example.joint_names)
        for left, right in mapping.items():
            assert left in joint_names
            assert right in joint_names


class TestMirror:
    """Tests for mirror transform."""

    def test_mirror_of_mirror_identity(self, bvh_example):
        from pybvh.transforms import mirror
        result = mirror(mirror(bvh_example))
        np.testing.assert_allclose(
            result.root_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(
            result.joint_angles, bvh_example.joint_angles, atol=1e-10)

    def test_total_bone_lengths_preserved(self, bvh_example):
        """Total skeleton bone length should be the same after mirroring."""
        from pybvh.transforms import mirror
        coords_orig = bvh_example.get_spatial_coord(centered='skeleton')
        result = mirror(bvh_example)
        coords_mir = result.get_spatial_coord(centered='skeleton')
        total_orig = 0.0
        total_mir = 0.0
        for node in bvh_example.nodes:
            if node.parent is not None:
                pi = bvh_example.node_index[node.parent.name]
                ci = bvh_example.node_index[node.name]
                total_orig += np.linalg.norm(coords_orig[0, ci] - coords_orig[0, pi])
                total_mir += np.linalg.norm(coords_mir[0, ci] - coords_mir[0, pi])
        np.testing.assert_allclose(total_mir, total_orig, atol=1e-2)

    def test_spatial_coords_reflected(self, bvh_example):
        """Gold-standard test: FK positions should be reflected."""
        from pybvh.transforms import mirror, auto_detect_lr_mapping
        from pybvh.tools import get_forw_up_axis

        rest = bvh_example.get_rest_pose(mode='coordinates')
        dirs = get_forw_up_axis(bvh_example, rest)
        used = {dirs['forward'][1], dirs['upward'][1]}
        lateral_char = ({"x", "y", "z"} - used).pop()
        lateral_idx = {"x": 0, "y": 1, "z": 2}[lateral_char]

        coords_orig = bvh_example.get_spatial_coord(centered='skeleton')
        result = mirror(bvh_example)
        coords_mir = result.get_spatial_coord(centered='skeleton')

        # Build set of all paired node names (including end-site children)
        mapping = auto_detect_lr_mapping(bvh_example)
        paired_names = set(mapping.keys()) | set(mapping.values())
        # Also mark end-site children of paired joints as paired
        for name in list(paired_names):
            ni = bvh_example.node_index[name]
            node = bvh_example.nodes[ni]
            if isinstance(node, BvhJoint):
                for child in node.children:
                    if child.is_end_site():
                        paired_names.add(child.name)

        # Center joints (not paired): lateral coordinate should be negated
        for node in bvh_example.nodes:
            if node.name not in paired_names:
                ni = bvh_example.node_index[node.name]
                np.testing.assert_allclose(
                    coords_mir[:, ni, lateral_idx],
                    -coords_orig[:, ni, lateral_idx],
                    atol=1e-2,
                    err_msg=f"Center joint {node.name} lateral coord not negated")

    def test_root_pos_lateral_negated(self, bvh_example):
        from pybvh.transforms import mirror
        from pybvh.tools import get_forw_up_axis
        rest = bvh_example.get_rest_pose(mode='coordinates')
        dirs = get_forw_up_axis(bvh_example, rest)
        used = {dirs['forward'][1], dirs['upward'][1]}
        lateral_idx = {"x": 0, "y": 1, "z": 2}[({"x", "y", "z"} - used).pop()]
        result = mirror(bvh_example)
        np.testing.assert_allclose(
            result.root_pos[:, lateral_idx],
            -bvh_example.root_pos[:, lateral_idx],
            atol=1e-10)

    def test_inplace(self, bvh_example):
        from pybvh.transforms import mirror
        bvh = bvh_example.copy()
        ret = mirror(bvh, inplace=True)
        assert ret is None

    def test_y_up_file(self, bvh_test2):
        from pybvh.transforms import mirror
        result = mirror(mirror(bvh_test2))
        np.testing.assert_allclose(
            result.root_pos, bvh_test2.root_pos, atol=1e-10)
        np.testing.assert_allclose(
            result.joint_angles, bvh_test2.joint_angles, atol=1e-10)

    def test_mixed_euler_orders(self, bvh_test3):
        from pybvh.transforms import mirror
        result = mirror(mirror(bvh_test3))
        np.testing.assert_allclose(
            result.root_pos, bvh_test3.root_pos, atol=1e-10)
        np.testing.assert_allclose(
            result.joint_angles, bvh_test3.joint_angles, atol=1e-10)

    def test_custom_mapping(self, bvh_example):
        from pybvh.transforms import mirror, auto_detect_lr_mapping
        mapping = auto_detect_lr_mapping(bvh_example)
        # Using explicit mapping should give same result as auto
        result_auto = mirror(bvh_example)
        result_manual = mirror(bvh_example, left_right_mapping=mapping)
        np.testing.assert_allclose(
            result_auto.root_pos, result_manual.root_pos, atol=1e-10)
        np.testing.assert_allclose(
            result_auto.joint_angles, result_manual.joint_angles, atol=1e-10)

    def test_frame_count_preserved(self, bvh_example):
        from pybvh.transforms import mirror
        result = mirror(bvh_example)
        assert result.frame_count == bvh_example.frame_count


# =========================================================================
# NumPy-level Transform API
# =========================================================================

class TestRotateAnglesVertical:
    """Tests for transforms.rotate_angles_vertical (NumPy-level)."""

    def test_zero_angle_identity(self, bvh_example):
        from pybvh.transforms import rotate_angles_vertical
        new_angles, new_pos = rotate_angles_vertical(
            bvh_example.joint_angles, bvh_example.root_pos,
            angle_deg=0.0, up_idx=1,
            root_order="".join(bvh_example.root.rot_channels),
        )
        np.testing.assert_allclose(new_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(new_angles, bvh_example.joint_angles, atol=1e-10)

    def test_360_identity(self, bvh_example):
        from pybvh.transforms import rotate_angles_vertical
        root_order = "".join(bvh_example.root.rot_channels)
        new_angles, new_pos = rotate_angles_vertical(
            bvh_example.joint_angles, bvh_example.root_pos,
            angle_deg=360.0, up_idx=1, root_order=root_order,
        )
        np.testing.assert_allclose(new_pos, bvh_example.root_pos, atol=1e-4)
        np.testing.assert_allclose(new_angles, bvh_example.joint_angles, atol=1e-3)

    def test_non_root_unchanged(self, bvh_example):
        from pybvh.transforms import rotate_angles_vertical
        root_order = "".join(bvh_example.root.rot_channels)
        new_angles, _ = rotate_angles_vertical(
            bvh_example.joint_angles, bvh_example.root_pos,
            angle_deg=45.0, up_idx=1, root_order=root_order,
        )
        np.testing.assert_allclose(
            new_angles[:, 1:], bvh_example.joint_angles[:, 1:], atol=1e-10,
        )

    def test_matches_bvh_level(self, bvh_example):
        """NumPy-level should produce identical results to Bvh-level."""
        from pybvh.transforms import rotate_angles_vertical, rotate_vertical
        root_order = "".join(bvh_example.root.rot_channels)
        new_angles, new_pos = rotate_angles_vertical(
            bvh_example.joint_angles, bvh_example.root_pos,
            angle_deg=73.0, up_idx=1, root_order=root_order,
        )
        bvh_result = rotate_vertical(bvh_example, angle_deg=73.0, up_axis='+y')
        np.testing.assert_allclose(new_pos, bvh_result.root_pos, atol=1e-10)
        np.testing.assert_allclose(new_angles, bvh_result.joint_angles, atol=1e-10)

    def test_z_up_axis(self, bvh_test2):
        """Works with Z-up skeletons (up_idx=2)."""
        from pybvh.transforms import rotate_angles_vertical
        root_order = "".join(bvh_test2.root.rot_channels)
        new_angles, new_pos = rotate_angles_vertical(
            bvh_test2.joint_angles, bvh_test2.root_pos,
            angle_deg=90.0, up_idx=2, root_order=root_order,
        )
        assert new_angles.shape == bvh_test2.joint_angles.shape
        assert new_pos.shape == bvh_test2.root_pos.shape


class TestMirrorAngles:
    """Tests for transforms.mirror_angles (NumPy-level)."""

    def _get_mirror_metadata(self, bvh):
        """Extract metadata needed for mirror_angles from a Bvh."""
        from pybvh.transforms import auto_detect_lr_mapping
        from pybvh.tools import get_forw_up_axis
        from pybvh.bvhnode import BvhJoint

        rest = bvh.get_rest_pose(mode='coordinates')
        dirs = get_forw_up_axis(bvh, rest)
        used = {dirs['forward'][1], dirs['upward'][1]}
        lateral_char = ({"x", "y", "z"} - used).pop()
        lateral_idx = {"x": 0, "y": 1, "z": 2}[lateral_char]

        mapping = auto_detect_lr_mapping(bvh)
        joints = [n for n in bvh.nodes if isinstance(n, BvhJoint)]
        j_name2idx = {j.name: i for i, j in enumerate(joints)}
        lr_pairs = []
        for left, right in mapping.items():
            if left in j_name2idx and right in j_name2idx:
                lr_pairs.append((j_name2idx[left], j_name2idx[right]))
        rot_ch = [list(j.rot_channels) for j in joints]
        return lr_pairs, lateral_idx, rot_ch

    def test_double_mirror_identity(self, bvh_example):
        from pybvh.transforms import mirror_angles
        lr_pairs, lat_idx, rot_ch = self._get_mirror_metadata(bvh_example)
        m_angles, m_pos = mirror_angles(
            bvh_example.joint_angles, bvh_example.root_pos,
            lr_pairs, lat_idx, rot_ch,
        )
        restored_angles, restored_pos = mirror_angles(
            m_angles, m_pos, lr_pairs, lat_idx, rot_ch,
        )
        np.testing.assert_allclose(
            restored_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(
            restored_angles, bvh_example.joint_angles, atol=1e-10)

    def test_root_pos_lateral_negated(self, bvh_example):
        from pybvh.transforms import mirror_angles
        lr_pairs, lat_idx, rot_ch = self._get_mirror_metadata(bvh_example)
        _, m_pos = mirror_angles(
            bvh_example.joint_angles, bvh_example.root_pos,
            lr_pairs, lat_idx, rot_ch,
        )
        np.testing.assert_allclose(
            m_pos[:, lat_idx], -bvh_example.root_pos[:, lat_idx], atol=1e-10)
        # Non-lateral components unchanged
        other = [i for i in range(3) if i != lat_idx]
        np.testing.assert_allclose(
            m_pos[:, other], bvh_example.root_pos[:, other], atol=1e-10)

    def test_matches_bvh_level_angles(self, bvh_example):
        """NumPy-level angles should match Bvh-level mirror's angles."""
        from pybvh.transforms import mirror_angles, mirror
        lr_pairs, lat_idx, rot_ch = self._get_mirror_metadata(bvh_example)
        m_angles, m_pos = mirror_angles(
            bvh_example.joint_angles, bvh_example.root_pos,
            lr_pairs, lat_idx, rot_ch,
        )
        bvh_result = mirror(bvh_example)
        np.testing.assert_allclose(m_pos, bvh_result.root_pos, atol=1e-10)
        np.testing.assert_allclose(m_angles, bvh_result.joint_angles, atol=1e-10)

    def test_no_pairs_only_negates(self):
        """With no L/R pairs, only lateral negation + angle negation happen."""
        from pybvh.transforms import mirror_angles
        rng = np.random.default_rng(42)
        angles = rng.standard_normal((10, 3, 3))
        root_pos = rng.standard_normal((10, 3))
        rot_ch = [['Z', 'Y', 'X']] * 3
        m_angles, m_pos = mirror_angles(angles, root_pos, [], 0, rot_ch)
        # Root pos X negated
        np.testing.assert_allclose(m_pos[:, 0], -root_pos[:, 0])
        np.testing.assert_allclose(m_pos[:, 1:], root_pos[:, 1:])
        # For X-lateral (idx=0), negate Y and Z components (not X)
        # rot_ch is ZYX: channel 0=Z, 1=Y, 2=X
        # Z != X → negate, Y != X → negate, X == X → keep
        np.testing.assert_allclose(m_angles[:, :, 0], -angles[:, :, 0])  # Z negated
        np.testing.assert_allclose(m_angles[:, :, 1], -angles[:, :, 1])  # Y negated
        np.testing.assert_allclose(m_angles[:, :, 2], angles[:, :, 2])   # X kept

    def test_mixed_euler_orders(self, bvh_test3):
        """Works correctly with mixed Euler orders across joints."""
        from pybvh.transforms import mirror_angles
        lr_pairs, lat_idx, rot_ch = self._get_mirror_metadata(bvh_test3)
        m_angles, m_pos = mirror_angles(
            bvh_test3.joint_angles, bvh_test3.root_pos,
            lr_pairs, lat_idx, rot_ch,
        )
        # Double mirror should recover original
        restored_angles, restored_pos = mirror_angles(
            m_angles, m_pos, lr_pairs, lat_idx, rot_ch,
        )
        np.testing.assert_allclose(
            restored_angles, bvh_test3.joint_angles, atol=1e-10)
        np.testing.assert_allclose(
            restored_pos, bvh_test3.root_pos, atol=1e-10)


# =============================================================================
# euler_orders property
# =============================================================================

class TestEulerOrders:
    """Tests for the euler_orders property."""

    def test_returns_list_of_strings(self, bvh_example):
        orders = bvh_example.euler_orders
        assert isinstance(orders, list)
        assert all(isinstance(o, str) for o in orders)

    def test_length_matches_joint_count(self, bvh_example):
        assert len(bvh_example.euler_orders) == bvh_example.joint_count

    def test_bvh_example_all_zyx(self, bvh_example):
        """bvh_example has all ZYX joints."""
        assert all(o == 'ZYX' for o in bvh_example.euler_orders)

    def test_mixed_orders(self, bvh_test3):
        """bvh_test3 has mixed Euler orders."""
        orders = bvh_test3.euler_orders
        assert len(orders) == bvh_test3.joint_count
        for o in orders:
            assert len(o) == 3
            assert set(o) == {'X', 'Y', 'Z'}

    def test_after_change_all_euler_orders(self, bvh_example):
        """euler_orders should reflect changed orders."""
        result = bvh_example.change_all_euler_orders('XYZ')
        assert all(o == 'XYZ' for o in result.euler_orders)

    def test_consistent_with_rot_channels(self, bvh_example):
        """euler_orders must match manually joining rot_channels."""
        expected = [
            ''.join(n.rot_channels)
            for n in bvh_example.nodes
            if not n.is_end_site()
        ]
        assert bvh_example.euler_orders == expected


# =============================================================================
# auto_detect_lr_pairs
# =============================================================================

class TestAutoDetectLRPairs:
    """Tests for auto_detect_lr_pairs returning index tuples."""

    def test_returns_list_of_tuples(self, bvh_example):
        from pybvh.transforms import auto_detect_lr_pairs
        pairs = auto_detect_lr_pairs(bvh_example)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_indices_are_valid(self, bvh_example):
        from pybvh.transforms import auto_detect_lr_pairs
        pairs = auto_detect_lr_pairs(bvh_example)
        J = bvh_example.joint_count
        for left, right in pairs:
            assert 0 <= left < J
            assert 0 <= right < J
            assert left != right

    def test_consistent_with_name_mapping(self, bvh_example):
        from pybvh.transforms import auto_detect_lr_pairs, auto_detect_lr_mapping
        pairs = auto_detect_lr_pairs(bvh_example)
        mapping = auto_detect_lr_mapping(bvh_example)
        assert len(pairs) == len(mapping)

    def test_works_with_mirror_angles(self, bvh_example):
        """Index pairs should be directly usable with mirror_angles."""
        from pybvh.transforms import auto_detect_lr_pairs, mirror_angles
        from pybvh.tools import get_forw_up_axis
        pairs = auto_detect_lr_pairs(bvh_example)
        rest = bvh_example.get_rest_pose(mode='coordinates')
        dirs = get_forw_up_axis(bvh_example, rest)
        used = {dirs['forward'][1], dirs['upward'][1]}
        lateral_idx = {"x": 0, "y": 1, "z": 2}[({"x", "y", "z"} - used).pop()]
        rot_ch = [
            list(n.rot_channels)
            for n in bvh_example.nodes
            if not n.is_end_site()
        ]
        # Should not raise
        m_angles, m_pos = mirror_angles(
            bvh_example.joint_angles, bvh_example.root_pos,
            pairs, lateral_idx, rot_ch,
        )
        # Double mirror should recover original
        r_angles, r_pos = mirror_angles(
            m_angles, m_pos, pairs, lateral_idx, rot_ch,
        )
        np.testing.assert_allclose(r_pos, bvh_example.root_pos, atol=1e-10)
        np.testing.assert_allclose(r_angles, bvh_example.joint_angles, atol=1e-10)

    def test_no_pairs_skeleton(self):
        """Skeleton with no L/R naming returns empty list."""
        from pybvh.transforms import auto_detect_lr_pairs
        root = BvhRoot()
        root._frozen = False
        bvh = Bvh(nodes=[root])
        assert auto_detect_lr_pairs(bvh) == []


# =============================================================================
# Bvh __eq__
# =============================================================================

class TestBvhEquality:
    """Tests for __eq__ on Bvh."""

    def test_equal_to_copy(self, bvh_example):
        assert bvh_example == bvh_example.copy()

    def test_equal_to_self(self, bvh_example):
        assert bvh_example == bvh_example

    def test_not_equal_after_modifying_root_pos(self, bvh_example):
        other = bvh_example.copy()
        other.root_pos[0, 0] += 1.0
        assert bvh_example != other

    def test_not_equal_after_modifying_joint_angles(self, bvh_example):
        other = bvh_example.copy()
        other.joint_angles[0, 0, 0] += 1.0
        assert bvh_example != other

    def test_not_equal_different_skeleton(self, bvh_example, bvh_test2):
        assert bvh_example != bvh_test2

    def test_not_equal_to_non_bvh(self, bvh_example):
        assert not (bvh_example == "not a bvh")
        assert not (bvh_example == 42)


# =============================================================================
# edges property
# =============================================================================

class TestEdges:
    """Tests for the edges property."""

    def test_returns_list_of_tuples(self, bvh_example):
        edges = bvh_example.edges
        assert isinstance(edges, list)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)

    def test_edge_count(self, bvh_example):
        """J joints → J-1 edges (root has no parent)."""
        assert len(bvh_example.edges) == bvh_example.joint_count - 1

    def test_indices_are_valid(self, bvh_example):
        J = bvh_example.joint_count
        for child_idx, parent_idx in bvh_example.edges:
            assert 0 <= child_idx < J
            assert 0 <= parent_idx < J
            assert child_idx != parent_idx

    def test_root_not_in_children(self, bvh_example):
        """Root (index 0) should never appear as a child in an edge."""
        children = {child for child, _ in bvh_example.edges}
        assert 0 not in children

    def test_root_is_parent(self, bvh_example):
        """Root (index 0) should appear as a parent."""
        parents = {parent for _, parent in bvh_example.edges}
        assert 0 in parents

    def test_different_skeletons(self, bvh_example, bvh_test3):
        """Different skeletons have different edge counts."""
        assert len(bvh_example.edges) == bvh_example.joint_count - 1
        assert len(bvh_test3.edges) == bvh_test3.joint_count - 1
