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


# ----------------------------------------------------------------
#  se3_inverse — closed-form rigid inverse
# ----------------------------------------------------------------

def test_se3_inverse_is_inverse():
    rng = np.random.default_rng(4)
    T = rot.se3_exp(rng.normal(size=(11, 6)))
    eye = np.broadcast_to(np.eye(4), T.shape)
    np.testing.assert_allclose(rot.se3_inverse(T) @ T, eye, atol=1e-12)
    np.testing.assert_allclose(T @ rot.se3_inverse(T), eye, atol=1e-12)


def test_se3_inverse_matches_generic_inverse():
    rng = np.random.default_rng(5)
    T = rot.se3_exp(rng.normal(size=(6, 6)))
    np.testing.assert_allclose(rot.se3_inverse(T), np.linalg.inv(T), atol=1e-12)


def test_se3_inverse_single_and_batch_shapes():
    T = rot.se3_exp(np.array([0.3, -0.2, 0.5, 1.0, 2.0, -1.0]))
    inv = rot.se3_inverse(T)
    assert inv.shape == (4, 4)
    _assert_is_se3(inv)
    batch = rot.se3_exp(np.zeros((3, 2, 6)))
    assert rot.se3_inverse(batch).shape == (3, 2, 4, 4)


# ----------------------------------------------------------------
#  screw_interpolate — array t broadcasting
# ----------------------------------------------------------------

def test_screw_interpolate_array_t():
    rng = np.random.default_rng(6)
    T0, T1 = rot.se3_exp(rng.normal(size=(2, 6)))
    t = np.linspace(0.0, 1.0, 5)
    out = rot.screw_interpolate(T0, T1, t)
    assert out.shape == (5, 4, 4)
    _assert_is_se3(out)
    np.testing.assert_allclose(out[0], T0, atol=1e-10)
    np.testing.assert_allclose(out[-1], T1, atol=1e-9)
    # each entry matches the scalar-t call
    for i, ti in enumerate(t):
        np.testing.assert_allclose(out[i], rot.screw_interpolate(T0, T1, ti),
                                   atol=1e-12)


def test_screw_interpolate_batchwise_t():
    rng = np.random.default_rng(7)
    T0 = rot.se3_exp(rng.normal(size=(4, 6)))
    T1 = rot.se3_exp(rng.normal(size=(4, 6)))
    t = np.array([0.0, 0.25, 0.75, 1.0])
    out = rot.screw_interpolate(T0, T1, t)
    assert out.shape == (4, 4, 4)
    for i, ti in enumerate(t):
        np.testing.assert_allclose(out[i], rot.screw_interpolate(T0[i], T1[i], ti),
                                   atol=1e-12)


# ----------------------------------------------------------------
#  _segment_frame — fixed reference axis (temporal continuity)
# ----------------------------------------------------------------

def test_segment_frame_vertical_segment_has_valid_frame():
    # x parallel to the default +y reference -> perpendicular fallback
    seg = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    T = rot._segment_frame(seg)
    _assert_is_se3(T[np.newaxis])
    np.testing.assert_allclose(T[:3, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_segment_frame_is_temporally_continuous():
    """A slowly rotating segment must produce a slowly rotating frame.

    The old per-entry reference (world axis least aligned with x) jumped
    whenever the segment crossed an axis bisector; the fixed default
    reference keeps consecutive frames within an angle bound proportional
    to the segment's own rotation step.
    """
    steps = 200
    angles = np.linspace(0.0, np.pi / 2, steps)
    # sweep x->y in the tilted plane z=0.5: |x_x| and |x_y| swap dominance
    # at the bisector, exactly where the old per-entry reference jumped
    dirs = np.stack([np.cos(angles), np.sin(angles),
                     np.full(steps, 0.5)], axis=-1)
    seg = np.stack([np.zeros((steps, 3)), dirs], axis=-2)  # (steps, 2, 3)
    T = rot._segment_frame(seg)
    step_angle = rot.rotation_geodesic_distance(T[:-1, :3, :3], T[1:, :3, :3])
    # the segment itself rotates ~ (π/2)/steps per frame; the frame must
    # not jump by orders of magnitude more than that anywhere
    assert step_angle.max() < 5 * (np.pi / 2) / (steps - 1)


def test_segment_frame_explicit_ref_axis():
    seg = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    T = rot._segment_frame(seg, ref=np.array([0.0, 0.0, 1.0]))
    # y = ref × x = z × x = +y
    np.testing.assert_allclose(T[:3, 1], [0.0, 1.0, 0.0], atol=1e-12)
    _assert_is_se3(T[np.newaxis])


# ----------------------------------------------------------------
#  mean_rotation — chordal (Frobenius) mean on SO(3)
# ----------------------------------------------------------------

def _random_rotations(rng, n):
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(0.0, np.pi, size=(n, 1))
    return rot.axisangle_to_rotmat(axes * angles)


def test_mean_rotation_identical_inputs():
    rng = np.random.default_rng(10)
    R = _random_rotations(rng, 1)[0]
    stack = np.broadcast_to(R, (5, 3, 3)).copy()
    np.testing.assert_allclose(rot.mean_rotation(stack), R, atol=1e-12)


def test_mean_rotation_same_axis_pair_bisects():
    # mean of Rz(a), Rz(b) is exactly Rz((a+b)/2): the matrix mean is a
    # scaled rotation in the z-block and the SVD projection rescales it
    Ra = rot.axisangle_to_rotmat(np.array([0.0, 0.0, 0.3]))
    Rb = rot.axisangle_to_rotmat(np.array([0.0, 0.0, 0.7]))
    mid = rot.axisangle_to_rotmat(np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(
        rot.mean_rotation(np.stack([Ra, Rb])), mid, atol=1e-12)


def test_mean_rotation_random_batch_is_rotation():
    rng = np.random.default_rng(11)
    m = rot.mean_rotation(_random_rotations(rng, 20))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_mean_rotation_batch_matches_loop():
    rng = np.random.default_rng(12)
    R = _random_rotations(rng, 4 * 7).reshape(4, 7, 3, 3)
    batched = rot.mean_rotation(R)
    assert batched.shape == (4, 3, 3)
    for b in range(4):
        np.testing.assert_allclose(batched[b], rot.mean_rotation(R[b]),
                                   atol=1e-14)


def test_mean_rotation_antipodal_pair_stays_right_handed():
    # 180 degrees apart: the arithmetic mean loses rank, so the mean's
    # orientation in the collapsed plane is documented-arbitrary — but the
    # result must still be a proper (right-handed) rotation, never a
    # reflection.  Validity only; the specific choice is not pinned.
    pair = np.stack([
        np.eye(3), rot.axisangle_to_rotmat(np.array([0.0, 0.0, np.pi]))])
    m = rot.mean_rotation(pair)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_mean_rotation_small_spread_matches_quaternion_mean():
    # For tight clusters the chordal mean agrees with normalized
    # sign-aligned quaternion averaging to first order
    rng = np.random.default_rng(13)
    base = _random_rotations(rng, 1)[0]
    perturbations = rot.axisangle_to_rotmat(
        rng.normal(scale=0.02, size=(30, 3)))
    R = base @ perturbations
    q = rot.rotmat_to_quat(R)
    q = np.where((q @ q[0])[:, None] < 0.0, -q, q)   # sign-align to the first
    q_mean = q.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    np.testing.assert_allclose(
        rot.mean_rotation(R), rot.quat_to_rotmat(q_mean), atol=1e-6)


def test_mean_rotation_empty_and_bad_shapes_raise():
    with pytest.raises(ValueError, match="empty"):
        rot.mean_rotation(np.zeros((0, 3, 3)))
    with pytest.raises(ValueError, match="shape"):
        rot.mean_rotation(np.eye(3))            # no N axis
    with pytest.raises(ValueError, match="shape"):
        rot.mean_rotation(np.zeros((4, 3, 4)))  # trailing shape not (3, 3)
