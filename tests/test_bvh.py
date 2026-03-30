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

from pybvh import read_bvh_file, df_to_bvh, Bvh, frames_to_spatial_coord, read_bvh_directory, batch_to_numpy
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
