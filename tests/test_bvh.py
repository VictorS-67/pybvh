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
