"""Analytic SE(3) tests that have no external golden oracle.

The frozen differential tests (vs pytransform3d / scipy) live in
test_se3_golden.py. Here: relative_transform (the geometry→SE(3) bridge) via
its algebraic invariants, plus single-vs-batched shape handling and a few
exp/log invariants that pin behavior independent of any reference library.
"""
import numpy as np
import pytest

from pybvh import rotations as rot


def _assert_is_se3(T):
    R = T[..., :3, :3]
    eye = np.broadcast_to(np.eye(3), R.shape)
    np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), eye, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), np.ones(R.shape[:-2]), atol=1e-10)
    np.testing.assert_allclose(T[..., 3, :], np.broadcast_to([0, 0, 0, 1.0], T[..., 3, :].shape))


# ----------------------------------------------------------------
#  relative_transform — geometry -> SE(3) bridge (algebraic invariants)
# ----------------------------------------------------------------

def test_relative_transform_self_is_identity():
    seg = np.array([[1.0, 2, 3], [4, 0, -1]])  # (2, 3): start, end
    np.testing.assert_allclose(rot.relative_transform(seg, seg), np.eye(4), atol=1e-12)


def test_relative_transform_is_valid_se3():
    rng = np.random.default_rng(0)
    seg_m = rng.normal(size=(2, 3))
    seg_n = rng.normal(size=(2, 3))
    _assert_is_se3(rot.relative_transform(seg_m, seg_n))


def test_relative_transform_transitive():
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(size=(3, 2, 3))
    lhs = rot.relative_transform(a, b) @ rot.relative_transform(b, c)
    np.testing.assert_allclose(lhs, rot.relative_transform(a, c), atol=1e-10)


def test_relative_transform_inverse():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(2, 2, 3))
    fwd = rot.relative_transform(a, b)
    bwd = rot.relative_transform(b, a)
    np.testing.assert_allclose(fwd @ bwd, np.eye(4), atol=1e-10)


def test_relative_transform_zero_length_segment_is_nan_no_warning():
    # a coincident-endpoint segment has no frame -> nan, and must not warn
    degenerate = np.array([[1.0, 1, 1], [1, 1, 1]])
    other = np.array([[0.0, 0, 0], [1, 0, 0]])
    out = rot.relative_transform(degenerate, other)
    assert np.isnan(out).any()


def test_relative_transform_batched():
    rng = np.random.default_rng(3)
    seg_m = rng.normal(size=(7, 2, 3))
    seg_n = rng.normal(size=(7, 2, 3))
    out = rot.relative_transform(seg_m, seg_n)
    assert out.shape == (7, 4, 4)
    _assert_is_se3(out)
    np.testing.assert_allclose(out[4], rot.relative_transform(seg_m[4], seg_n[4]))


# ----------------------------------------------------------------
#  exp / log single-vs-batched shapes
# ----------------------------------------------------------------

def test_se3_exp_log_single_and_batch_shapes():
    twist = np.array([0.3, -0.2, 0.5, 1.0, 2.0, -1.0])
    assert rot.se3_exp(twist).shape == (4, 4)
    assert rot.se3_log(rot.se3_exp(twist)).shape == (6,)
    batch = np.stack([twist, -twist, twist * 0.1])
    assert rot.se3_exp(batch).shape == (3, 4, 4)
    assert rot.se3_log(rot.se3_exp(batch)).shape == (3, 6)
    np.testing.assert_allclose(rot.se3_log(rot.se3_exp(twist)), twist, atol=1e-10)


def test_se3_exp_pure_rotation_has_zero_translation():
    # [ω, 0] -> rotation only, d = V·0 = 0
    twist = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0])
    T = rot.se3_exp(twist)
    np.testing.assert_allclose(T[:3, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], rot.axisangle_to_rotmat(twist[:3]), atol=1e-12)


def test_rotation_geodesic_distance_basic():
    # identity vs a 90° rotation about z -> π/2
    R0 = np.eye(3)
    Rz = rot.axisangle_to_rotmat(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(rot.rotation_geodesic_distance(R0, Rz), np.pi / 2, atol=1e-12)
    # self-distance is 0
    np.testing.assert_allclose(rot.rotation_geodesic_distance(Rz, Rz), 0.0, atol=1e-12)
