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
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pybvh import read_bvh_file, df_to_bvh, Bvh
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
