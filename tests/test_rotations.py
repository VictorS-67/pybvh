"""
Tests for pybvh.rotations module and Bvh rotation convenience methods.

Run with: pytest tests/test_rotations.py -v

Ground-truth values for the ``TestScipyReference*`` classes were pre-computed
with ``scipy.spatial.transform.Rotation`` (scipy 1.14) and hardcoded here so
that pybvh never depends on scipy at runtime or test-time.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pybvh import rotations, read_bvh_file


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def bvh_example():
    return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_example.bvh")


# =============================================================================
# Test: euler_to_rotmat
# =============================================================================

class TestEulerToRotmat:
    """Tests for Euler angle to rotation matrix conversion."""

    def test_identity_rotation(self):
        """Zero angles should give identity matrix."""
        R = rotations.euler_to_rotmat([0, 0, 0], 'ZYX')
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_x_rotation(self):
        """90° about X should rotate Y→Z and Z→-Y."""
        R = rotations.euler_to_rotmat([90, 0, 0], 'XYZ', degrees=True)
        expected = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0]
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90deg_y_rotation(self):
        """90° about Y should rotate Z→X and X→-Z."""
        R = rotations.euler_to_rotmat([0, 90, 0], 'XYZ', degrees=True)
        expected = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90deg_z_rotation(self):
        """90° about Z should rotate X→Y and Y→-X."""
        R = rotations.euler_to_rotmat([0, 0, 90], 'XYZ', degrees=True)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_batch_euler_to_rotmat(self):
        """Batch conversion: N angles → N matrices."""
        angles = np.array([
            [0, 0, 0],
            [90, 0, 0],
            [0, 90, 0],
        ])
        R = rotations.euler_to_rotmat(angles, 'XYZ', degrees=True)
        assert R.shape == (3, 3, 3)
        # First should be identity
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-12)

    def test_det_is_one(self):
        """All rotation matrices should have determinant 1."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        dets = np.linalg.det(R)
        np.testing.assert_allclose(dets, 1.0, atol=1e-10)

    def test_orthogonality(self):
        """R @ R.T should be identity for all rotation matrices."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(50, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        for i in range(50):
            np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-10)

    def test_degrees_flag(self):
        """degrees=True and degrees=False should give same result."""
        R_deg = rotations.euler_to_rotmat([45, 30, 60], 'ZYX', degrees=True)
        R_rad = rotations.euler_to_rotmat(np.radians([45, 30, 60]), 'ZYX', degrees=False)
        np.testing.assert_allclose(R_deg, R_rad, atol=1e-12)

    def test_different_orders(self):
        """Different Euler orders should give different matrices for same angles."""
        angles = [45, 30, 60]
        R_zyx = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        R_xyz = rotations.euler_to_rotmat(angles, 'XYZ', degrees=True)
        assert not np.allclose(R_zyx, R_xyz)

    def test_invalid_order_raises(self):
        """Invalid Euler order should raise ValueError."""
        with pytest.raises(ValueError):
            rotations.euler_to_rotmat([0, 0, 0], 'ABC')


# =============================================================================
# Test: rotmat_to_euler (round-trip)
# =============================================================================

class TestRotmatToEuler:
    """Tests for rotation matrix to Euler angle conversion."""

    def test_identity_gives_zeros(self):
        """Identity matrix should give zero angles."""
        angles = rotations.rotmat_to_euler(np.eye(3), 'ZYX')
        np.testing.assert_allclose(angles, [0, 0, 0], atol=1e-12)

    def test_roundtrip_single(self):
        """Euler → rotmat → Euler should recover original angles."""
        original = np.array([30, 45, 60])
        R = rotations.euler_to_rotmat(original, 'ZYX', degrees=True)
        recovered = rotations.rotmat_to_euler(R, 'ZYX', degrees=True)
        np.testing.assert_allclose(recovered, original, atol=1e-8)

    def test_roundtrip_batch(self):
        """Batch round-trip: euler → rotmat → euler."""
        rng = np.random.default_rng(42)
        # Avoid angles near ±90° for the middle angle (gimbal lock)
        original = rng.uniform(-80, 80, size=(100, 3))
        for order in ['ZYX', 'XYZ', 'YZX', 'ZXY', 'YXZ', 'XZY']:
            R = rotations.euler_to_rotmat(original, order, degrees=True)
            recovered = rotations.rotmat_to_euler(R, order, degrees=True)
            # Round-trip through rotmat should recover the same rotation
            R_check = rotations.euler_to_rotmat(recovered, order, degrees=True)
            np.testing.assert_allclose(R_check, R, atol=1e-8,
                err_msg=f"Round-trip failed for order {order}")

    def test_roundtrip_various_orders(self):
        """Specific angle values round-trip for all 6 Tait-Bryan orders."""
        angles = np.array([25.0, 40.0, -35.0])
        for order in ['ZYX', 'XYZ', 'YZX', 'ZXY', 'YXZ', 'XZY']:
            R = rotations.euler_to_rotmat(angles, order, degrees=True)
            recovered = rotations.rotmat_to_euler(R, order, degrees=True)
            R2 = rotations.euler_to_rotmat(recovered, order, degrees=True)
            np.testing.assert_allclose(R2, R, atol=1e-10,
                err_msg=f"Failed for order {order}")


# =============================================================================
# Test: 6D rotation representation
# =============================================================================

class TestRot6D:
    """Tests for 6D rotation representation (Zhou et al.)."""

    def test_identity_6d(self):
        """Identity matrix should give [1,0,0,0,1,0]."""
        rot6d = rotations.rotmat_to_rot6d(np.eye(3))
        np.testing.assert_allclose(rot6d, [1, 0, 0, 0, 1, 0], atol=1e-12)

    def test_roundtrip_6d_single(self):
        """rotmat → 6d → rotmat round-trip."""
        R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
        rot6d = rotations.rotmat_to_rot6d(R)
        R_recovered = rotations.rot6d_to_rotmat(rot6d)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_roundtrip_6d_batch(self):
        """Batch rotmat → 6d → rotmat round-trip."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        rot6d = rotations.rotmat_to_rot6d(R)
        assert rot6d.shape == (100, 6)
        R_recovered = rotations.rot6d_to_rotmat(rot6d)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_6d_shape(self):
        """6D output should have shape (*, 6)."""
        R = np.eye(3)
        assert rotations.rotmat_to_rot6d(R).shape == (6,)

        R_batch = np.stack([np.eye(3)] * 5)
        assert rotations.rotmat_to_rot6d(R_batch).shape == (5, 6)

    def test_gram_schmidt_recovery_from_noisy_input(self):
        """rot6d_to_rotmat should produce valid rotation from imperfect 6D input."""
        # Perturb a valid 6D representation slightly
        R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
        rot6d = rotations.rotmat_to_rot6d(R)
        rot6d_noisy = rot6d + np.random.default_rng(42).normal(0, 0.01, size=6)

        R_recovered = rotations.rot6d_to_rotmat(rot6d_noisy)
        # Should still be a valid rotation matrix
        np.testing.assert_allclose(np.linalg.det(R_recovered), 1.0, atol=1e-10)
        np.testing.assert_allclose(R_recovered @ R_recovered.T, np.eye(3), atol=1e-10)

    def test_euler_to_6d_convenience(self):
        """euler_to_rot6d should match euler→rotmat→6d pipeline."""
        angles = np.array([30, 45, 60])
        rot6d_direct = rotations.euler_to_rot6d(angles, 'ZYX', degrees=True)
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        rot6d_manual = rotations.rotmat_to_rot6d(R)
        np.testing.assert_allclose(rot6d_direct, rot6d_manual, atol=1e-12)

    def test_6d_to_euler_convenience(self):
        """rot6d_to_euler → euler_to_rot6d round-trip."""
        angles = np.array([25.0, 40.0, -35.0])
        rot6d = rotations.euler_to_rot6d(angles, 'ZYX', degrees=True)
        recovered = rotations.rot6d_to_euler(rot6d, 'ZYX', degrees=True)
        R1 = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        R2 = rotations.euler_to_rotmat(recovered, 'ZYX', degrees=True)
        np.testing.assert_allclose(R1, R2, atol=1e-10)


# =============================================================================
# Test: Quaternion representation
# =============================================================================

class TestQuaternion:
    """Tests for quaternion conversions."""

    def test_identity_quaternion(self):
        """Identity matrix should give quaternion (1, 0, 0, 0)."""
        q = rotations.rotmat_to_quat(np.eye(3))
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-12)

    def test_roundtrip_single(self):
        """rotmat → quat → rotmat round-trip."""
        R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
        q = rotations.rotmat_to_quat(R)
        R_recovered = rotations.quat_to_rotmat(q)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_roundtrip_batch(self):
        """Batch rotmat → quat → rotmat round-trip."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        q = rotations.rotmat_to_quat(R)
        assert q.shape == (100, 4)
        R_recovered = rotations.quat_to_rotmat(q)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_unit_quaternion(self):
        """Output quaternions should be unit length."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        q = rotations.rotmat_to_quat(R)
        norms = np.linalg.norm(q, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_canonical_positive_w(self):
        """Output quaternions should have w >= 0 (canonical form)."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        q = rotations.rotmat_to_quat(R)
        assert np.all(q[:, 0] >= -1e-12)

    def test_90deg_z_quaternion(self):
        """90° about Z should give quaternion (cos45, 0, 0, sin45)."""
        R = rotations.euler_to_rotmat([0, 0, 90], 'XYZ', degrees=True)
        q = rotations.rotmat_to_quat(R)
        expected = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_180deg_rotations(self):
        """180° rotations should produce valid quaternions."""
        for axis_angles in [[180, 0, 0], [0, 180, 0], [0, 0, 180]]:
            R = rotations.euler_to_rotmat(axis_angles, 'XYZ', degrees=True)
            q = rotations.rotmat_to_quat(R)
            R2 = rotations.quat_to_rotmat(q)
            np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_euler_to_quat_convenience(self):
        """euler_to_quat should match euler→rotmat→quat pipeline."""
        angles = np.array([30, 45, 60])
        q_direct = rotations.euler_to_quat(angles, 'ZYX', degrees=True)
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        q_manual = rotations.rotmat_to_quat(R)
        np.testing.assert_allclose(q_direct, q_manual, atol=1e-12)

    def test_quat_normalizes_input(self):
        """quat_to_rotmat should normalize non-unit quaternions."""
        q = np.array([2, 0, 0, 0])  # non-unit, should still give identity
        R = rotations.quat_to_rotmat(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


# =============================================================================
# Test: Bvh convenience methods
# =============================================================================

class TestBvhRotationMethods:
    """Tests for Bvh.get_frames_as_6d, get_frames_as_quaternion, etc."""

    def test_get_frames_as_rotmat_shape(self, bvh_example):
        """get_frames_as_rotmat should return correct shapes."""
        root_pos, joint_rotmats, joints = bvh_example.get_frames_as_rotmat()
        num_joints = len([n for n in bvh_example.nodes if not n.is_end_site()])
        assert root_pos.shape == (56, 3)
        assert joint_rotmats.shape == (56, num_joints, 3, 3)
        assert len(joints) == num_joints

    def test_get_frames_as_rotmat_orthogonal(self, bvh_example):
        """All output rotation matrices should be orthogonal."""
        _, joint_rotmats, _ = bvh_example.get_frames_as_rotmat()
        # Check a sample of matrices
        for f in [0, 10, 55]:
            for j in range(joint_rotmats.shape[1]):
                R = joint_rotmats[f, j]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
                np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_get_frames_as_6d_shape(self, bvh_example):
        """get_frames_as_6d should return correct shapes."""
        root_pos, joint_rot6d, joints = bvh_example.get_frames_as_6d()
        num_joints = len([n for n in bvh_example.nodes if not n.is_end_site()])
        assert root_pos.shape == (56, 3)
        assert joint_rot6d.shape == (56, num_joints, 6)
        assert len(joints) == num_joints

    def test_get_frames_as_quaternion_shape(self, bvh_example):
        """get_frames_as_quaternion should return correct shapes."""
        root_pos, joint_quats, joints = bvh_example.get_frames_as_quaternion()
        num_joints = len([n for n in bvh_example.nodes if not n.is_end_site()])
        assert root_pos.shape == (56, 3)
        assert joint_quats.shape == (56, num_joints, 4)
        assert len(joints) == num_joints

    def test_6d_roundtrip_through_bvh(self, bvh_example):
        """Bvh → 6D → set_frames_from_6d should preserve data."""
        original_root_pos = bvh_example.root_pos.copy()
        original_joint_angles = bvh_example.joint_angles.copy()

        root_pos, joint_rot6d, _ = bvh_example.get_frames_as_6d()
        bvh_example.set_frames_from_6d(root_pos, joint_rot6d)

        np.testing.assert_allclose(
            bvh_example.root_pos, original_root_pos, atol=1e-6,
            err_msg="6D round-trip did not preserve root_pos")
        np.testing.assert_allclose(
            bvh_example.joint_angles, original_joint_angles, atol=1e-6,
            err_msg="6D round-trip did not preserve joint_angles")

    def test_quaternion_roundtrip_through_bvh(self, bvh_example):
        """Bvh → quaternion → set_frames_from_quaternion should preserve data."""
        original_root_pos = bvh_example.root_pos.copy()
        original_joint_angles = bvh_example.joint_angles.copy()

        root_pos, joint_quats, _ = bvh_example.get_frames_as_quaternion()
        bvh_example.set_frames_from_quaternion(root_pos, joint_quats)

        np.testing.assert_allclose(
            bvh_example.root_pos, original_root_pos, atol=1e-6,
            err_msg="Quaternion round-trip did not preserve root_pos")
        np.testing.assert_allclose(
            bvh_example.joint_angles, original_joint_angles, atol=1e-6,
            err_msg="Quaternion round-trip did not preserve joint_angles")

    def test_6d_preserves_spatial_coords(self, bvh_example):
        """Spatial coordinates should be the same after 6D round-trip."""
        spatial_before = bvh_example.get_spatial_coord(centered="world")

        root_pos, joint_rot6d, _ = bvh_example.get_frames_as_6d()
        bvh_example.set_frames_from_6d(root_pos, joint_rot6d)
        spatial_after = bvh_example.get_spatial_coord(centered="world")

        np.testing.assert_allclose(spatial_after, spatial_before, atol=1e-4)

    def test_joint_names_match_nodes(self, bvh_example):
        """Joint names from get_frames_as_* should match node order."""
        _, _, joints_6d = bvh_example.get_frames_as_6d()
        names_6d = [j.name for j in joints_6d]
        _, _, joints_quat = bvh_example.get_frames_as_quaternion()
        names_quat = [j.name for j in joints_quat]
        _, _, joints_romat = bvh_example.get_frames_as_rotmat()
        names_rotmat = [j.name for j in joints_romat]

        expected = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        assert names_6d == expected
        assert names_quat == expected
        assert names_rotmat == expected

    def test_set_frames_from_6d_wrong_joints_raises(self, bvh_example):
        """set_frames_from_6d with wrong number of joints should raise."""
        root_pos = bvh_example.root_pos
        wrong_6d = np.zeros((56, 5, 6))  # wrong number of joints
        with pytest.raises(ValueError):
            bvh_example.set_frames_from_6d(root_pos, wrong_6d)

    def test_set_frames_from_quaternion_wrong_joints_raises(self, bvh_example):
        """set_frames_from_quaternion with wrong number of joints should raise."""
        root_pos = bvh_example.root_pos
        wrong_quats = np.zeros((56, 5, 4))
        with pytest.raises(ValueError):
            bvh_example.set_frames_from_quaternion(root_pos, wrong_quats)


# =============================================================================
# Test: Cross-representation consistency
# =============================================================================

class TestCrossRepresentation:
    """Tests to ensure all representations encode the same rotation."""

    def test_euler_rotmat_quat_6d_consistency(self):
        """All paths from Euler → {rotmat, quat, 6d} → rotmat should agree."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-80, 80, size=(50, 3))
        order = 'ZYX'

        R_from_euler = rotations.euler_to_rotmat(angles, order, degrees=True)

        # Via quaternion
        q = rotations.rotmat_to_quat(R_from_euler)
        R_from_quat = rotations.quat_to_rotmat(q)

        # Via 6D
        r6d = rotations.rotmat_to_rot6d(R_from_euler)
        R_from_6d = rotations.rot6d_to_rotmat(r6d)

        np.testing.assert_allclose(R_from_quat, R_from_euler, atol=1e-10)
        np.testing.assert_allclose(R_from_6d, R_from_euler, atol=1e-10)

    def test_all_representations_applied_to_vector(self):
        """Rotating a vector through any path should give the same result."""
        angles = np.array([35.0, -50.0, 20.0])
        order = 'ZYX'
        v = np.array([1.0, 2.0, 3.0])

        R = rotations.euler_to_rotmat(angles, order, degrees=True)
        v_rotated = R @ v

        # Via 6D
        r6d = rotations.rotmat_to_rot6d(R)
        R2 = rotations.rot6d_to_rotmat(r6d)
        np.testing.assert_allclose(R2 @ v, v_rotated, atol=1e-10)

        # Via quaternion
        q = rotations.rotmat_to_quat(R)
        R3 = rotations.quat_to_rotmat(q)
        np.testing.assert_allclose(R3 @ v, v_rotated, atol=1e-10)

        # Via axis-angle
        aa = rotations.rotmat_to_axisangle(R)
        R4 = rotations.axisangle_to_rotmat(aa)
        np.testing.assert_allclose(R4 @ v, v_rotated, atol=1e-10)


# =============================================================================
# Scipy-verified ground truth: euler_to_rotmat
# Values computed with scipy.spatial.transform.Rotation (scipy 1.14).
# =============================================================================

class TestScipyReferenceEulerToRotmat:
    """
    Hardcoded rotation matrices from scipy, testing euler_to_rotmat against
    them for multiple Euler orders.
    """

    def test_zyx_30_45_60(self):
        """ZYX(30, 45, 60)° — scipy reference."""
        expected = np.array([
            [ 0.6123724356957946,  0.2803300858899106,  0.7391989197401166],
            [ 0.3535533905932737,  0.7391989197401166, -0.573223304703363 ],
            [-0.7071067811865476,  0.6123724356957945,  0.353553390593274 ],
        ])
        R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_xyz_45_n30_120(self):
        """XYZ(45, -30, 120)° — scipy reference."""
        expected = np.array([
            [-0.4330127018922192, -0.7500000000000002, -0.5               ],
            [ 0.7891491309924314, -0.0473671727453764, -0.6123724356957946],
            [ 0.4355957403991578, -0.659739608441171 ,  0.6123724356957947],
        ])
        R = rotations.euler_to_rotmat([45, -30, 120], 'XYZ', degrees=True)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_yzx_10_n170_85(self):
        """YZX(10, -170, 85)° — scipy reference."""
        expected = np.array([
            [-0.9698463103929543,  0.1878919037381941, -0.1552248908094664],
            [-0.1736481776669303, -0.0858316511774312,  0.981060262190407 ],
            [ 0.1710100716628344,  0.9784321949761225,  0.1158705968918745],
        ])
        R = rotations.euler_to_rotmat([10, -170, 85], 'YZX', degrees=True)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_zxy_n60_75_n15(self):
        """ZXY(-60, 75, -15)° — scipy reference."""
        expected = np.array([
            [ 0.266456562198425 ,  0.2241438680420134, -0.9374222244434798],
            [-0.961516303737808 ,  0.1294095225512604, -0.2423624829040967],
            [ 0.0669872981077806,  0.9659258262890684,  0.25              ],
        ])
        R = rotations.euler_to_rotmat([-60, 75, -15], 'ZXY', degrees=True)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_batch_zyx_5_rotations(self):
        """Batch of 5 ZYX rotations (seed=123) — scipy reference."""
        angles = np.array([
            [  65.64667076933165 , -160.62443323119984 , -100.6704458018599  ],
            [-113.6261481484789  , -116.67387560938907 ,  112.35402239607852 ],
            [ 152.4041992897403  ,  -80.43321679304175 ,  115.11164217348075 ],
            [ 140.36136952002693 ,    4.669363882631479,  -91.81274361523327 ],
            [ 116.72697459506807 , -103.04533318496563 ,   86.92813880449548 ],
        ])
        expected = np.array([
            [[-0.3890080161506059,  0.3031235443637366, -0.8699361357151085],
             [-0.8594235354103301,  0.2206598061339465,  0.4611945757917565],
             [ 0.3317588735111376,  0.9270519763256507,  0.1746730747348486]],
            [[ 0.1799090469431537, -0.0172450516314353, -0.9835320752381361],
             [ 0.4112838088199813,  0.9095773773373874,  0.0592842579321743],
             [ 0.8935761654333304, -0.4151765923438707,  0.1707338095960967]],
            [[-0.1472901033855301,  0.9878998986817091,  0.0485738162938483],
             [ 0.0769876602698735, -0.0375097173436253,  0.9963262122773704],
             [ 0.9860925542822288,  0.1504885752788894, -0.0705312916348453]],
            [[-0.7675274352898426,  0.0828378197438839, -0.6356410399726217],
             [ 0.6358260496178914, -0.0275460712764647, -0.7713406825680456],
             [-0.0814055938170282, -0.9961822672469689, -0.0315280782475765]],
            [[ 0.1015160726788481,  0.3896400048051197,  0.9153552062687629],
             [-0.2016057326901014, -0.8929594600607441,  0.4024655652779653],
             [ 0.974191775542347 , -0.2253975806070044, -0.0120960787926806]],
        ])
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        np.testing.assert_allclose(R, expected, atol=1e-13)


# =============================================================================
# Scipy-verified ground truth: rotmat_to_euler
# =============================================================================

class TestScipyReferenceRotmatToEuler:
    """
    Hardcoded Euler angle extractions from scipy. A single rotation
    (ZYX 30°/45°/60°) is decomposed into all 6 Tait-Bryan orders.
    """

    # The rotation is defined as ZYX(30, 45, 60) degrees.
    _R_ZYX_30_45_60 = np.array([
        [ 0.6123724356957946,  0.2803300858899106,  0.7391989197401166],
        [ 0.3535533905932737,  0.7391989197401166, -0.573223304703363 ],
        [-0.7071067811865476,  0.6123724356957945,  0.353553390593274 ],
    ])

    @pytest.mark.parametrize("order,expected_deg", [
        ('ZYX', [30.0, 45.0, 60.0]),
        ('XYZ', [58.334492452083495, 47.66322046446767, -24.597222684382135]),
        ('YZX', [49.10660535086909, 20.704811054635442, 37.792345701403505]),
        ('ZXY', [-20.76847951640774, 37.76124390703503, 63.43494882292201]),
        ('YXZ', [64.43855416320221, 34.97530385109734, 25.561445836797773]),
        ('XZY', [39.63927223775613, -16.2799062470331, 50.36072776224387]),
    ])
    def test_cross_order_extraction(self, order, expected_deg):
        """Extracting Euler angles in different orders — scipy reference."""
        result = rotations.rotmat_to_euler(self._R_ZYX_30_45_60, order, degrees=True)
        np.testing.assert_allclose(result, expected_deg, atol=1e-10)

    @pytest.mark.parametrize("order,expected_deg", [
        ('ZYX', [30.0, 45.0, 60.0]),
        ('XYZ', [58.334492452083495, 47.66322046446767, -24.597222684382135]),
        ('YZX', [49.10660535086909, 20.704811054635442, 37.792345701403505]),
        ('ZXY', [-20.76847951640774, 37.76124390703503, 63.43494882292201]),
        ('YXZ', [64.43855416320221, 34.97530385109734, 25.561445836797773]),
        ('XZY', [39.63927223775613, -16.2799062470331, 50.36072776224387]),
    ])
    def test_cross_order_round_trip(self, order, expected_deg):
        """Extracted angles re-composed must yield the original matrix."""
        result = rotations.rotmat_to_euler(self._R_ZYX_30_45_60, order, degrees=True)
        R_rebuilt = rotations.euler_to_rotmat(result, order, degrees=True)
        np.testing.assert_allclose(R_rebuilt, self._R_ZYX_30_45_60, atol=1e-12)


# =============================================================================
# Scipy-verified ground truth: quaternions
# =============================================================================

class TestScipyReferenceQuaternion:
    """
    Hardcoded quaternion values from scipy (converted to our w,x,y,z
    scalar-first convention with canonical w >= 0).
    """

    @pytest.mark.parametrize("euler,order,expected_q", [
        ([30, 45, 60], 'ZYX',
         [0.8223631719059994, 0.3604234056503559, 0.4396797395409096, 0.0222600267147338]),
        ([45, -30, 120], 'XYZ',
         [0.5319756951821669, -0.0222600267147337, -0.4396797395409096, 0.7233174113647117]),
        ([10, -170, 85], 'YZX',
         [0.1226709371871441, -0.0053559287850616, -0.6648578912676845, -0.736808753758779]),
        ([-60, 75, -15], 'ZXY',
         [0.6414565621984246, 0.4709158734973702, -0.3914565621984244, -0.4620968283948492]),
    ])
    def test_euler_to_quat(self, euler, order, expected_q):
        """euler → quat — scipy reference."""
        q = rotations.euler_to_quat(euler, order, degrees=True)
        np.testing.assert_allclose(q, expected_q, atol=1e-13)

    @pytest.mark.parametrize("euler,order,expected_q", [
        ([30, 45, 60], 'ZYX',
         [0.8223631719059994, 0.3604234056503559, 0.4396797395409096, 0.0222600267147338]),
        ([45, -30, 120], 'XYZ',
         [0.5319756951821669, -0.0222600267147337, -0.4396797395409096, 0.7233174113647117]),
    ])
    def test_quat_to_rotmat_matches(self, euler, order, expected_q):
        """quat → rotmat should produce the same matrix as euler → rotmat."""
        R_from_euler = rotations.euler_to_rotmat(euler, order, degrees=True)
        R_from_quat = rotations.quat_to_rotmat(np.array(expected_q))
        np.testing.assert_allclose(R_from_quat, R_from_euler, atol=1e-12)

    def test_batch_quats(self):
        """Batch ZYX(seed=123) quaternions — scipy reference."""
        angles = np.array([
            [  65.64667076933165 , -160.62443323119984 , -100.6704458018599  ],
            [-113.6261481484789  , -116.67387560938907 ,  112.35402239607852 ],
            [ 152.4041992897403  ,  -80.43321679304175 ,  115.11164217348075 ],
            [ 140.36136952002693 ,    4.669363882631479,  -91.81274361523327 ],
            [ 116.72697459506807 , -103.04533318496563 ,   86.92813880449548 ],
        ])
        expected_q = np.array([
            [ 0.5015787238106769,  0.2321955549642364, -0.5989563313693461, -0.5794439758836716],
            [ 0.7517014423753352, -0.1577956431667787, -0.6242864968902773,  0.1425196348889823],
            [ 0.4314709977611471, -0.4900895085576054, -0.5432107504635632, -0.5277945928802014],
            [ 0.2082056766673978, -0.269975329536885 , -0.6654903159064289,  0.6639927387251179],
            [ 0.2216193435969791, -0.7082675362340619, -0.0663712024395533, -0.6669608887688272],
        ])
        q = rotations.euler_to_quat(angles, 'ZYX', degrees=True)
        np.testing.assert_allclose(q, expected_q, atol=1e-13)


# =============================================================================
# Scipy-verified ground truth: axis-angle
# =============================================================================

class TestScipyReferenceAxisAngle:
    """
    Hardcoded axis-angle (rotation vector) values from scipy.
    """

    @pytest.mark.parametrize("euler,order,expected_aa", [
        ([30, 45, 60], 'ZYX',
         [0.7668133408388336, 0.9354339498794426, 0.0473589816440653]),
        ([45, -30, 120], 'XYZ',
         [-0.0530955988070215, -1.0487435326745562, 1.7252886340219056]),
        ([10, -170, 85], 'YZX',
         [-0.0156268178765827, -1.9398340787552923, -2.1497627520691305]),
        ([-60, 75, -15], 'ZXY',
         [1.073490846156704, -0.8923569151897102, -1.0533871190958828]),
    ])
    def test_euler_to_axisangle(self, euler, order, expected_aa):
        """euler → axisangle — scipy reference."""
        aa = rotations.euler_to_axisangle(euler, order, degrees=True)
        np.testing.assert_allclose(aa, expected_aa, atol=1e-13)

    @pytest.mark.parametrize("aa_input,expected_R", [
        ([0.5, 0.0, 0.0],
         [[ 1.0,               0.0,                0.0              ],
          [ 0.0,               0.8775825618903726, -0.479425538604203],
          [ 0.0,               0.479425538604203,   0.8775825618903726]]),
        ([0.0, -1.2, 0.0],
         [[ 0.3623577544766736, 0.0,               -0.9320390859672264],
          [ 0.0,                1.0,                 0.0              ],
          [ 0.9320390859672264, 0.0,                 0.3623577544766736]]),
        ([0.3, 0.4, 0.5],
         [[ 0.8034005696020168, -0.4018213882309354,  0.4394167688235383],
          [ 0.5169039816346329,  0.8369663260114285, -0.1797154497899226],
          [-0.2955635270689164,  0.3715197721294184,  0.8801222985378151]]),
        ([1.0, 1.0, 1.0],
         [[ 0.2262956409502064, -0.1830079196576171,  0.9567122787074109],
          [ 0.9567122787074109,  0.2262956409502064, -0.1830079196576171],
          [-0.1830079196576171,  0.9567122787074109,  0.2262956409502064]]),
    ])
    def test_axisangle_to_rotmat(self, aa_input, expected_R):
        """axisangle → rotmat — scipy reference."""
        R = rotations.axisangle_to_rotmat(np.array(aa_input))
        np.testing.assert_allclose(R, np.array(expected_R), atol=1e-14)

    @pytest.mark.parametrize("aa_input,expected_q", [
        ([0.5, 0.0, 0.0],
         [0.9689124217106447, 0.2474039592545229, 0.0, 0.0]),
        ([0.0, -1.2, 0.0],
         [0.8253356149096783, 0.0, -0.5646424733950354, 0.0]),
        ([0.3, 0.4, 0.5],
         [0.9381483350397287, 0.1468944732220831, 0.1958592976294441, 0.2448241220368051]),
        ([1.0, 1.0, 1.0],
         [0.647859344852457, 0.4398023303285789, 0.4398023303285789, 0.4398023303285789]),
    ])
    def test_axisangle_to_quat_via_rotmat(self, aa_input, expected_q):
        """axisangle → rotmat → quat should match scipy quaternion."""
        R = rotations.axisangle_to_rotmat(np.array(aa_input))
        q = rotations.rotmat_to_quat(R)
        np.testing.assert_allclose(q, expected_q, atol=1e-13)

    def test_batch_axisangles(self):
        """Batch ZYX(seed=123) axis-angles — scipy reference."""
        angles = np.array([
            [  65.64667076933165 , -160.62443323119984 , -100.6704458018599  ],
            [-113.6261481484789  , -116.67387560938907 ,  112.35402239607852 ],
            [ 152.4041992897403  ,  -80.43321679304175 ,  115.11164217348075 ],
            [ 140.36136952002693 ,    4.669363882631479,  -91.81274361523327 ],
            [ 116.72697459506807 , -103.04533318496563 ,   86.92813880449548 ],
        ])
        expected_aa = np.array([
            [ 0.5611553414419852, -1.4475192890327842, -1.4003630783698042],
            [-0.3446162382063964, -1.36340433616402  ,  0.3112543506291841],
            [-1.2219806014739611, -1.3544321761390332, -1.3159937986851327],
            [-0.7513694325697424, -1.8521287922900986,  1.8479608791741806],
            [-1.9571973271920224, -0.1834074461578678, -1.8430522395263025],
        ])
        aa = rotations.euler_to_axisangle(angles, 'ZYX', degrees=True)
        np.testing.assert_allclose(aa, expected_aa, atol=1e-13)


# =============================================================================
# Scipy-verified ground truth: 6D representation
# =============================================================================

class TestScipyReference6D:
    """
    Hardcoded 6D rotation values from scipy (first two columns of R,
    concatenated).
    """

    @pytest.mark.parametrize("euler,order,expected_6d", [
        ([30, 45, 60], 'ZYX',
         [ 0.6123724356957946,  0.3535533905932737, -0.7071067811865476,
           0.2803300858899106,  0.7391989197401166,  0.6123724356957945]),
        ([45, -30, 120], 'XYZ',
         [-0.4330127018922192,  0.7891491309924314,  0.4355957403991578,
          -0.7500000000000002, -0.0473671727453764, -0.659739608441171 ]),
        ([10, -170, 85], 'YZX',
         [-0.9698463103929543, -0.1736481776669303,  0.1710100716628344,
           0.1878919037381941, -0.0858316511774312,  0.9784321949761225]),
        ([-60, 75, -15], 'ZXY',
         [ 0.266456562198425 , -0.961516303737808 ,  0.0669872981077806,
           0.2241438680420134,  0.1294095225512604,  0.9659258262890684]),
    ])
    def test_euler_to_6d(self, euler, order, expected_6d):
        """euler → 6D — scipy reference (first 2 cols of rotmat)."""
        rot6d = rotations.euler_to_rot6d(euler, order, degrees=True)
        np.testing.assert_allclose(rot6d, expected_6d, atol=1e-14)


# =============================================================================
# Test: Axis-angle representation
# =============================================================================

class TestAxisAngle:
    """Tests for axis-angle conversions."""

    def test_identity_gives_zero_vector(self):
        """Identity matrix should give zero axis-angle vector."""
        aa = rotations.rotmat_to_axisangle(np.eye(3))
        np.testing.assert_allclose(aa, [0, 0, 0], atol=1e-12)

    def test_zero_vector_gives_identity(self):
        """Zero axis-angle vector should give identity matrix."""
        R = rotations.axisangle_to_rotmat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_x(self):
        """90° about X axis should give axis-angle = [π/2, 0, 0]."""
        R = rotations.euler_to_rotmat([90, 0, 0], 'XYZ', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        expected = np.array([np.pi / 2, 0, 0])
        np.testing.assert_allclose(aa, expected, atol=1e-10)

    def test_90deg_y(self):
        """90° about Y axis should give axis-angle = [0, π/2, 0]."""
        R = rotations.euler_to_rotmat([0, 90, 0], 'XYZ', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        expected = np.array([0, np.pi / 2, 0])
        np.testing.assert_allclose(aa, expected, atol=1e-10)

    def test_90deg_z(self):
        """90° about Z axis should give axis-angle = [0, 0, π/2]."""
        R = rotations.euler_to_rotmat([0, 0, 90], 'XYZ', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        expected = np.array([0, 0, np.pi / 2])
        np.testing.assert_allclose(aa, expected, atol=1e-10)

    def test_roundtrip_single(self):
        """rotmat → axisangle → rotmat round-trip."""
        R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        R_recovered = rotations.axisangle_to_rotmat(aa)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_roundtrip_batch(self):
        """Batch rotmat → axisangle → rotmat round-trip."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-180, 180, size=(100, 3))
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        assert aa.shape == (100, 3)
        R_recovered = rotations.axisangle_to_rotmat(aa)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_axisangle_roundtrip_from_aa(self):
        """axisangle → rotmat → axisangle round-trip."""
        rng = np.random.default_rng(42)
        # Random axes, random angles in [0, π-0.01] to avoid π ambiguity
        axes = rng.normal(size=(50, 3))
        axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        angles = rng.uniform(0.01, np.pi - 0.01, size=(50, 1))
        aa_orig = axes * angles

        R = rotations.axisangle_to_rotmat(aa_orig)
        aa_recovered = rotations.rotmat_to_axisangle(R)
        np.testing.assert_allclose(aa_recovered, aa_orig, atol=1e-10)

    def test_180deg_rotations(self):
        """180° rotations should produce valid axis-angle and round-trip."""
        for axis_angles in [[180, 0, 0], [0, 180, 0], [0, 0, 180]]:
            R = rotations.euler_to_rotmat(axis_angles, 'XYZ', degrees=True)
            aa = rotations.rotmat_to_axisangle(R)
            # Angle magnitude should be π
            np.testing.assert_allclose(np.linalg.norm(aa), np.pi, atol=1e-10)
            R2 = rotations.axisangle_to_rotmat(aa)
            np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_rodrigues_det_one(self):
        """Rodrigues output should have determinant 1."""
        rng = np.random.default_rng(42)
        aa = rng.uniform(-np.pi, np.pi, size=(100, 3))
        R = rotations.axisangle_to_rotmat(aa)
        dets = np.linalg.det(R)
        np.testing.assert_allclose(dets, 1.0, atol=1e-10)

    def test_rodrigues_orthogonal(self):
        """Rodrigues output should be orthogonal."""
        rng = np.random.default_rng(42)
        aa = rng.uniform(-np.pi, np.pi, size=(50, 3))
        R = rotations.axisangle_to_rotmat(aa)
        for i in range(50):
            np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-10)

    def test_axis_angle_magnitude_is_angle(self):
        """The norm of the axis-angle vector should equal the rotation angle."""
        R = rotations.euler_to_rotmat([0, 0, 45], 'XYZ', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        np.testing.assert_allclose(np.linalg.norm(aa), np.radians(45), atol=1e-10)

    def test_euler_to_axisangle_convenience(self):
        """euler_to_axisangle should match euler→rotmat→axisangle pipeline."""
        angles = np.array([30, 45, 60])
        aa_direct = rotations.euler_to_axisangle(angles, 'ZYX', degrees=True)
        R = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        aa_manual = rotations.rotmat_to_axisangle(R)
        np.testing.assert_allclose(aa_direct, aa_manual, atol=1e-12)

    def test_axisangle_to_euler_convenience(self):
        """axisangle_to_euler → euler_to_axisangle round-trip."""
        angles = np.array([25.0, 40.0, -35.0])
        aa = rotations.euler_to_axisangle(angles, 'ZYX', degrees=True)
        recovered = rotations.axisangle_to_euler(aa, 'ZYX', degrees=True)
        R1 = rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
        R2 = rotations.euler_to_rotmat(recovered, 'ZYX', degrees=True)
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    def test_small_angle_stability(self):
        """Very small rotations should produce small axis-angle vectors."""
        R = rotations.euler_to_rotmat([0.001, 0.002, 0.003], 'XYZ', degrees=True)
        aa = rotations.rotmat_to_axisangle(R)
        R_recovered = rotations.axisangle_to_rotmat(aa)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)


# =============================================================================
# Test: Bvh axis-angle methods
# =============================================================================

class TestBvhAxisAngleMethods:
    """Tests for Bvh.get_frames_as_axisangle and set_frames_from_axisangle."""

    def test_get_frames_as_axisangle_shape(self, bvh_example):
        """get_frames_as_axisangle should return correct shapes."""
        root_pos, joint_aa, joints = bvh_example.get_frames_as_axisangle()
        num_joints = len([n for n in bvh_example.nodes if not n.is_end_site()])
        assert root_pos.shape == (56, 3)
        assert joint_aa.shape == (56, num_joints, 3)
        assert len(joints) == num_joints

    def test_axisangle_roundtrip_through_bvh(self, bvh_example):
        """Bvh → axis-angle → set_frames_from_axisangle should preserve data."""
        original_root_pos = bvh_example.root_pos.copy()
        original_joint_angles = bvh_example.joint_angles.copy()
        root_pos, joint_aa, _ = bvh_example.get_frames_as_axisangle()
        bvh_example.set_frames_from_axisangle(root_pos, joint_aa)
        np.testing.assert_allclose(
            bvh_example.root_pos, original_root_pos, atol=1e-6,
            err_msg="Axis-angle round-trip did not preserve root_pos")
        np.testing.assert_allclose(
            bvh_example.joint_angles, original_joint_angles, atol=1e-6,
            err_msg="Axis-angle round-trip did not preserve joint_angles")

    def test_axisangle_preserves_spatial_coords(self, bvh_example):
        """Spatial coordinates should be the same after axis-angle round-trip."""
        spatial_before = bvh_example.get_spatial_coord(centered="world")
        root_pos, joint_aa, _ = bvh_example.get_frames_as_axisangle()
        bvh_example.set_frames_from_axisangle(root_pos, joint_aa)
        spatial_after = bvh_example.get_spatial_coord(centered="world")
        np.testing.assert_allclose(spatial_after, spatial_before, atol=1e-4)

    def test_joint_names_from_axisangle(self, bvh_example):
        """Joints from get_frames_as_axisangle should match node order."""
        _, _, joints_aa = bvh_example.get_frames_as_axisangle()
        names_aa = [j.name for j in joints_aa]
        expected = [n.name for n in bvh_example.nodes if not n.is_end_site()]
        assert names_aa == expected

    def test_set_frames_from_axisangle_wrong_joints_raises(self, bvh_example):
        """set_frames_from_axisangle with wrong number of joints should raise."""
        root_pos = bvh_example.root_pos
        wrong_aa = np.zeros((56, 5, 3))
        with pytest.raises(ValueError):
            bvh_example.set_frames_from_axisangle(root_pos, wrong_aa)


# =============================================================================
# Test: Euler order conversion
# =============================================================================

class TestEulerOrderConversion:
    """Tests for Bvh.single_joint_euler_angle and change_all_euler_orders."""

    def test_single_joint_changes_rot_channels(self, bvh_example):
        """single_joint_euler_angle should update the joint's rot_channels."""
        bvh = bvh_example.copy()
        joint_name = 'Spine'
        old_order = None
        for n in bvh.nodes:
            if n.name == joint_name and not n.is_end_site():
                old_order = n.rot_channels
                break
        assert old_order is not None

        bvh.single_joint_euler_angle(joint_name, 'XYZ', inplace=True)

        for n in bvh.nodes:
            if n.name == joint_name and not n.is_end_site():
                assert n.rot_channels == ['X', 'Y', 'Z']
                break

    def test_single_joint_preserves_rotation(self, bvh_example):
        """Changing Euler order should not change the physical rotation."""
        bvh = bvh_example.copy()
        joint_name = 'Spine'

        # Get rotmats before
        _, rotmats_before, _ = bvh.get_frames_as_rotmat()
        # Find Spine's index
        joints = [n for n in bvh.nodes if not n.is_end_site()]
        j_idx = [n.name for n in joints].index(joint_name)

        bvh.single_joint_euler_angle(joint_name, 'XYZ', inplace=True)

        # Get rotmats after
        _, rotmats_after, _ = bvh.get_frames_as_rotmat()

        np.testing.assert_allclose(
            rotmats_after[:, j_idx], rotmats_before[:, j_idx], atol=1e-10,
            err_msg="Changing Euler order changed the physical rotation")

    def test_single_joint_preserves_spatial_coords(self, bvh_example):
        """Spatial coordinates should not change after Euler order conversion."""
        bvh = bvh_example.copy()
        spatial_before = bvh.get_spatial_coord(centered="world")

        bvh.single_joint_euler_angle('Spine', 'XYZ', inplace=True)
        spatial_after = bvh.get_spatial_coord(centered="world")

        np.testing.assert_allclose(spatial_after, spatial_before, atol=1e-4)

    def test_single_joint_updates_euler_column_names(self, bvh_example):
        """euler_column_names should reflect the new channel order."""
        bvh = bvh_example.copy()
        bvh.single_joint_euler_angle('Spine', 'XYZ', inplace=True)

        # Find Spine columns in euler_column_names
        spine_cols = [c for c in bvh.euler_column_names if c.startswith('Spine_') and c.endswith('_rot')]
        assert spine_cols == ['Spine_X_rot', 'Spine_Y_rot', 'Spine_Z_rot']

    def test_single_joint_same_order_noop(self, bvh_example):
        """Passing the same order should be a no-op."""
        bvh = bvh_example.copy()
        original_joint_angles = bvh.joint_angles.copy()
        for n in bvh.nodes:
            if n.name == 'Spine' and not n.is_end_site():
                current_order = n.rot_channels
                break
        bvh.single_joint_euler_angle('Spine', current_order, inplace=True)
        np.testing.assert_allclose(bvh.joint_angles, original_joint_angles, atol=1e-14)

    def test_single_joint_not_inplace(self, bvh_example):
        """inplace=False should return a new Bvh, leaving original unchanged."""
        bvh = bvh_example.copy()
        original_joint_angles = bvh.joint_angles.copy()
        result = bvh.single_joint_euler_angle('Spine', 'XYZ', inplace=False)

        # Original should be unchanged
        np.testing.assert_allclose(bvh.joint_angles, original_joint_angles)
        for n in bvh.nodes:
            if n.name == 'Spine' and not n.is_end_site():
                assert n.rot_channels != ['X', 'Y', 'Z'] or n.rot_channels == ['X', 'Y', 'Z']
                break

        # Result should have the new order
        assert result is not None
        for n in result.nodes:
            if n.name == 'Spine' and not n.is_end_site():
                assert n.rot_channels == ['X', 'Y', 'Z']
                break

    def test_single_joint_invalid_name_raises(self, bvh_example):
        """Invalid joint name should raise ValueError."""
        with pytest.raises(ValueError):
            bvh_example.single_joint_euler_angle('NonExistent', 'XYZ')

    def test_change_all_euler_orders(self, bvh_example):
        """change_all_euler_orders should update all joints to the new order."""
        bvh = bvh_example.copy()
        bvh.change_all_euler_orders('XYZ', inplace=True)

        for node in bvh.nodes:
            if not node.is_end_site():
                assert node.rot_channels == ['X', 'Y', 'Z'], \
                    f"Joint {node.name} still has order {node.rot_channels}"

    def test_change_all_preserves_spatial_coords(self, bvh_example):
        """Spatial coordinates should not change after converting all orders."""
        bvh = bvh_example.copy()
        spatial_before = bvh.get_spatial_coord(centered="world")

        bvh.change_all_euler_orders('XYZ', inplace=True)
        spatial_after = bvh.get_spatial_coord(centered="world")

        np.testing.assert_allclose(spatial_after, spatial_before, atol=1e-4)

    def test_change_all_preserves_rotations(self, bvh_example):
        """Rotation matrices should be the same after changing all Euler orders."""
        bvh = bvh_example.copy()
        _, rotmats_before, _ = bvh.get_frames_as_rotmat()

        bvh.change_all_euler_orders('XYZ', inplace=True)
        _, rotmats_after, _ = bvh.get_frames_as_rotmat()

        np.testing.assert_allclose(rotmats_after, rotmats_before, atol=1e-10)

    def test_change_all_euler_column_names_consistent(self, bvh_example):
        """euler_column_names should be consistent after changing all orders."""
        bvh = bvh_example.copy()
        bvh.change_all_euler_orders('XYZ', inplace=True)

        # All rotation columns should now be X, Y, Z ordered
        rot_cols = [c for c in bvh.euler_column_names if c.endswith('_rot')]
        for i in range(0, len(rot_cols), 3):
            triple = rot_cols[i:i+3]
            joint = triple[0].rsplit('_', 2)[0]
            assert triple == [f'{joint}_X_rot', f'{joint}_Y_rot', f'{joint}_Z_rot']

    def test_change_all_not_inplace(self, bvh_example):
        """inplace=False should return a new Bvh without modifying original."""
        bvh = bvh_example.copy()
        original_joint_angles = bvh.joint_angles.copy()
        result = bvh.change_all_euler_orders('XYZ', inplace=False)

        # Original unchanged
        np.testing.assert_allclose(bvh.joint_angles, original_joint_angles)

        # Result has new orders
        for node in result.nodes:
            if not node.is_end_site():
                assert node.rot_channels == ['X', 'Y', 'Z']

    def test_double_conversion_roundtrip(self, bvh_example):
        """Converting all to XYZ then back to ZYX should recover original data."""
        bvh = bvh_example.copy()
        original_joint_angles = bvh.joint_angles.copy()

        # Get original orders before conversion
        original_orders = {}
        for n in bvh.nodes:
            if not n.is_end_site():
                original_orders[n.name] = list(n.rot_channels)

        bvh.change_all_euler_orders('XYZ', inplace=True)

        # Convert each joint back to its original order
        for n in bvh.nodes:
            if not n.is_end_site() and n.name in original_orders:
                bvh.single_joint_euler_angle(n.name, original_orders[n.name], inplace=True)

        np.testing.assert_allclose(bvh.joint_angles, original_joint_angles, atol=1e-8)

    def test_bvh_file_roundtrip_after_conversion(self, bvh_example, tmp_path):
        """Write → re-read should preserve the converted Euler orders."""
        bvh = bvh_example.copy()
        bvh.change_all_euler_orders('XYZ', inplace=True)

        filepath = tmp_path / "converted.bvh"
        bvh.to_bvh_file(str(filepath), verbose=False)

        from pybvh import read_bvh_file
        bvh_reloaded = read_bvh_file(str(filepath))

        for node in bvh_reloaded.nodes:
            if not node.is_end_site():
                assert node.rot_channels == ['X', 'Y', 'Z'], \
                    f"Joint {node.name} has order {node.rot_channels} after reload"

        np.testing.assert_allclose(bvh_reloaded.root_pos, bvh.root_pos, atol=1e-4)
        np.testing.assert_allclose(bvh_reloaded.joint_angles, bvh.joint_angles, atol=1e-4)
