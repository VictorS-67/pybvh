"""Analytic tests for pybvh.geometry — closed-form known answers.

Every assertion here has a hand-derivable oracle (circle curvature = 1/r,
straight-line directness = 1, unit right-triangle area = 1/2, ...), so these
never go stale and pin pybvh's conventions independently of any reference lib.
Also covers the structural invariants from the implementation plan: shape
sweeps, vectorization over the frame axis (no Python frame loop), the shared
finite-difference convention, and the nan zero-denominator policy.
"""
import numpy as np
import pytest

from pybvh import geometry as geo
from pybvh import tools


# ----------------------------------------------------------------
#  Inter-point relations
# ----------------------------------------------------------------

def test_inter_joint_distance_known():
    pos = np.array([[0.0, 0, 0], [3, 4, 0], [0, 0, 12]])
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = geo.inter_joint_distance(pos, pairs)
    np.testing.assert_allclose(out, [5.0, 12.0, 13.0])


def test_inter_joint_distance_vectorizes_over_frames():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(6, 9, 3))  # (F, N, 3)
    pairs = [(0, 3), (1, 8)]
    out = geo.inter_joint_distance(pos, pairs)
    assert out.shape == (6, 2)
    np.testing.assert_allclose(out[4], geo.inter_joint_distance(pos[4], pairs))


def test_joint_angle_right_angle_and_straight():
    o = np.zeros(3)
    np.testing.assert_allclose(
        geo.joint_angle([1, 0, 0], o, [0, 1, 0]), np.pi / 2)
    np.testing.assert_allclose(
        geo.joint_angle([1, 0, 0], o, [-1, 0, 0]), np.pi)
    np.testing.assert_allclose(
        geo.joint_angle([1, 0, 0], o, [0, 1, 0], degrees=True), 90.0)


def test_joint_angle_symmetric_in_outer_points():
    rng = np.random.default_rng(2)
    a, v, b = rng.normal(size=(3, 5, 3))
    np.testing.assert_allclose(
        geo.joint_angle(a, v, b), geo.joint_angle(b, v, a))


def test_joint_angle_shape_sweep():
    o = np.zeros(3)
    assert np.ndim(geo.joint_angle([1, 0, 0], o, [0, 1, 0])) == 0      # scalar
    assert geo.joint_angle(np.ones((5, 3)), np.zeros((5, 3)),
                           np.ones((5, 3))).shape == (5,)
    assert geo.joint_angle(np.ones((4, 5, 3)), np.zeros((4, 5, 3)),
                           np.ones((4, 5, 3))).shape == (4, 5)


def test_segment_axis_angle_known():
    up = np.array([0.0, 1, 0])
    np.testing.assert_allclose(geo.segment_axis_angle([0, 1, 0], up), 0.0)
    np.testing.assert_allclose(geo.segment_axis_angle([1, 0, 0], up), np.pi / 2)
    np.testing.assert_allclose(geo.segment_axis_angle([0, -1, 0], up), np.pi)


def test_triangle_area_unit_and_equilateral():
    np.testing.assert_allclose(
        geo.triangle_area([0, 0, 0], [1, 0, 0], [0, 1, 0]), 0.5)
    s = 2.0
    eq = geo.triangle_area([0, 0, 0], [s, 0, 0], [s / 2, s * np.sqrt(3) / 2, 0])
    np.testing.assert_allclose(eq, np.sqrt(3) / 4 * s ** 2)


def test_point_to_plane_distance_signed_and_abs():
    pp, n = np.zeros(3), np.array([0.0, 0, 1])
    np.testing.assert_allclose(geo.point_to_plane_distance([0, 0, 5], pp, n), 5.0)
    np.testing.assert_allclose(geo.point_to_plane_distance([0, 0, -3], pp, n), -3.0)
    np.testing.assert_allclose(
        geo.point_to_plane_distance([0, 0, -3], pp, n, signed=False), 3.0)


def test_point_to_segment_distance_interior_and_clamped():
    a, b = np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])
    np.testing.assert_allclose(geo.point_to_segment_distance([0, 1, 0], a, b), 1.0)
    # beyond the endpoint -> distance to the endpoint, not the infinite line
    np.testing.assert_allclose(geo.point_to_segment_distance([2, 0, 0], a, b), 1.0)
    # degenerate segment -> point-to-point
    np.testing.assert_allclose(
        geo.point_to_segment_distance([3, 4, 0], a, a), np.linalg.norm([4, 4, 0]))


# ----------------------------------------------------------------
#  Bounding volumes & center of mass
# ----------------------------------------------------------------

def _cube_corners():
    return np.array([[x, y, z] for x in (0.0, 1) for y in (0.0, 1) for z in (0.0, 1)])


def test_bounding_box_unit_cube():
    bb = geo.bounding_box(_cube_corners())
    np.testing.assert_allclose(bb.min, [0, 0, 0])
    np.testing.assert_allclose(bb.max, [1, 1, 1])
    np.testing.assert_allclose(bb.extent, [1, 1, 1])
    np.testing.assert_allclose(bb.volume, 1.0)


def test_bounding_sphere_axis_points():
    c, r = np.array([2.0, -1, 3]), 1.5
    pts = c + r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                            [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    sph = geo.bounding_sphere(pts)
    np.testing.assert_allclose(sph.center, c, atol=1e-9)
    np.testing.assert_allclose(sph.radius, r, atol=1e-9)


def test_bounding_sphere_encloses_all_points():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(50, 3))
    sph = geo.bounding_sphere(pts)
    d = np.linalg.norm(pts - sph.center, axis=-1)
    assert np.all(d <= sph.radius + 1e-9)  # enclosure guarantee


def test_bounding_ellipsoid_axis_aligned_box():
    half = np.array([1.0, 2.0, 3.0])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-2, 2)
                        for z in (-3, 3.0)])
    ell = geo.bounding_ellipsoid(corners)
    np.testing.assert_allclose(ell.center, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.sort(ell.radii), np.sort(half))


def test_center_of_mass_uniform_and_weighted():
    pts = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    np.testing.assert_allclose(geo.center_of_mass(pts), [0.5, 0.5, 0.5])
    # all weight on the last point
    np.testing.assert_allclose(
        geo.center_of_mass(pts, weights=[0, 0, 0, 1.0]), [0, 0, 2])


def test_com_displacement_known():
    np.testing.assert_allclose(
        geo.com_displacement([3.0, 4, 0], [0, 0, 0]), 5.0)


def test_verticality_known_rectangle():
    up = np.array([0.0, 1, 0])
    # 1 wide (x in {0,1}), 2 tall (y in {0,2})
    pts = np.array([[0.0, 0, 0], [0, 2, 0], [1, 0, 0], [1, 2, 0]])
    np.testing.assert_allclose(geo.verticality(pts, up), 2.0)


# ----------------------------------------------------------------
#  Trajectory descriptors (closed-form)
# ----------------------------------------------------------------

def test_path_length_straight_and_square():
    line = np.linspace([0, 0, 0], [10, 0, 0], 11)
    np.testing.assert_allclose(geo.path_length(line), 10.0)
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0.0]])
    np.testing.assert_allclose(geo.path_length(square), 4.0)


def test_directness_line_loop_and_L():
    line = np.linspace([0, 0, 0], [5, 0, 0], 20)
    np.testing.assert_allclose(geo.directness(line), 1.0)
    out_and_back = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0.0]])
    np.testing.assert_allclose(geo.directness(out_and_back), 0.0)
    L = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.0]])
    np.testing.assert_allclose(geo.directness(L), np.sqrt(2) / 2)


def _circle(n, r):
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    traj = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n)], axis=1)
    return traj, theta[1] - theta[0]


def test_curvature_circle_is_one_over_r():
    traj, dt = _circle(720, 2.0)
    kappa = geo.curvature(traj, dt)
    np.testing.assert_allclose(kappa[3:-3], 0.5, rtol=1e-3)


def test_curvature_straight_line_is_zero():
    line = np.linspace([0, 0, 0], [5, 0, 0], 50)
    kappa = geo.curvature(line, 0.1)
    np.testing.assert_allclose(kappa[2:-2], 0.0, atol=1e-9)


def _helix(n, r, b, turns):
    theta = np.linspace(0.0, turns * 2 * np.pi, n)
    traj = np.stack([r * np.cos(theta), r * np.sin(theta), b * theta], axis=1)
    return traj, theta[1] - theta[0]


def test_torsion_planar_circle_is_zero():
    traj, dt = _circle(720, 2.0)
    tau = geo.torsion(traj, dt)
    np.testing.assert_allclose(tau[5:-5], 0.0, atol=1e-6)


def test_torsion_helix_matches_closed_form():
    r, b = 1.0, 0.5
    traj, dt = _helix(2000, r, b, turns=4)
    tau = geo.torsion(traj, dt)
    expected = b / (r ** 2 + b ** 2)  # = 0.4
    np.testing.assert_allclose(tau[20:-20], expected, rtol=1e-2)


def test_movement_phase_circle_is_speed_times_curvature():
    r = 2.0
    traj, dt = _circle(720, r)
    phase = geo.movement_phase(traj, dt)
    # speed = r, curvature = 1/r  ->  product = 1
    np.testing.assert_allclose(phase[3:-3], 1.0, rtol=1e-3)


def test_ground_path_square():
    up = np.array([0.0, 1, 0])
    s = 3.0
    traj = np.array([[0, 0, 0], [s, 0, 0], [s, 0, s], [0, 0, s]])
    gp = geo.ground_path(traj, up)
    np.testing.assert_allclose(gp.area, s ** 2)          # shoelace closes the loop
    np.testing.assert_allclose(gp.distance, 3 * s)        # open path, 3 segments


# ----------------------------------------------------------------
#  Pose-level ops
# ----------------------------------------------------------------

def test_pose_distance_is_euclidean():
    a = np.zeros((4, 3))
    b = a.copy()
    b[2] = [3, 4, 0]
    # true (sqrt) distance since v0.8.0: sum of squares is 25 -> distance 5
    np.testing.assert_allclose(geo.pose_distance(a, b), 5.0)


def test_pose_distance_vectorizes_over_frames():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 8, 6, 3))
    out = geo.pose_distance(a, b)
    assert out.shape == (8,)
    np.testing.assert_allclose(out[5], geo.pose_distance(a[5], b[5]))


def test_mean_pose_subtract_removes_temporal_mean():
    rng = np.random.default_rng(5)
    seq = rng.normal(size=(30, 7, 3)) + np.array([1.0, 2, 3])
    centered = geo.mean_pose_subtract(seq)
    np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-12)
    assert centered.shape == seq.shape


# ----------------------------------------------------------------
#  Vectorization over the frame axis (no Python frame loop)
# ----------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["bounding_box", "bounding_sphere",
                                    "bounding_ellipsoid", "center_of_mass"])
def test_pointset_kernels_vectorize_over_frames(kernel):
    rng = np.random.default_rng(6)
    pts = rng.normal(size=(11, 14, 3))  # (F, P, 3)
    fn = getattr(geo, kernel)
    batched = fn(pts)
    per_frame = fn(pts[7])
    first = batched[0] if isinstance(batched, tuple) else batched
    assert first.shape[0] == 11  # leading frame axis preserved
    if isinstance(batched, tuple):
        for field_batched, field_single in zip(batched, per_frame):
            np.testing.assert_allclose(field_batched[7], field_single)
    else:
        np.testing.assert_allclose(batched[7], per_frame)


# ----------------------------------------------------------------
#  Shared finite-difference convention
# ----------------------------------------------------------------

def test_finite_difference_central_matches_np_gradient():
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(20, 4, 3))
    dt = 0.0333
    np.testing.assert_allclose(
        tools.finite_difference(arr, dt, stencil="central", pad="edge"),
        np.gradient(arr, dt, axis=0))


def test_finite_difference_forward_formula_and_shapes():
    rng = np.random.default_rng(8)
    arr = rng.normal(size=(10, 3))
    dt = 0.5
    fwd_none = tools.finite_difference(arr, dt, stencil="forward", pad="none")
    np.testing.assert_allclose(fwd_none, (arr[1:] - arr[:-1]) / dt)
    assert fwd_none.shape == (9, 3)
    assert tools.finite_difference(arr, dt, stencil="forward", pad="edge").shape == (10, 3)
    assert tools.finite_difference(arr, dt, stencil="central", pad="none").shape == (8, 3)


def test_finite_difference_rejects_bad_args():
    arr = np.zeros((5, 3))
    with pytest.raises(ValueError):
        tools.finite_difference(arr, 0.1, stencil="bogus")
    with pytest.raises(ValueError):
        tools.finite_difference(arr, 0.1, pad="bogus")


def test_trajectory_derivative_kernels_reject_bad_args():
    # a typo'd pad used to silently mean "edge" in the derivative kernels
    traj = np.zeros((10, 3))
    with pytest.raises(ValueError, match="stencil"):
        geo.curvature(traj, 0.1, stencil="bogus")
    with pytest.raises(ValueError, match="pad"):
        geo.movement_phase(traj, 0.1, pad="bogus")
    with pytest.raises(ValueError, match="pad"):
        geo.torsion(traj, 0.1, pad="bogus")


# ----------------------------------------------------------------
#  Zero-denominator policy -> nan, consistently
# ----------------------------------------------------------------

def test_nan_sentinels_on_degenerate_input():
    stationary = np.zeros((10, 3))
    assert np.all(np.isnan(geo.curvature(stationary, 0.1)))
    assert np.isnan(geo.directness(stationary))
    # a perfectly vertical point set has zero horizontal width
    vertical = np.array([[0.0, 0, 0], [0, 1, 0], [0, 2, 0]])
    assert np.isnan(geo.verticality(vertical, np.array([0.0, 1, 0])))
