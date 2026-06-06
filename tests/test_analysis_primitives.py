"""Analytic tests for the v0.8.0 pybvh.analysis primitives.

Covers jerk (composition identity), the smoothness dispatcher + simple
kernels (SPARC/DLJ/LDLJ exactness is in test_smoothness_golden.py), signal
reductions, kinetic energy, gait, range of motion, and the covariance
descriptors. Every assertion has a hand-derivable oracle.
"""
import numpy as np
import pytest

from pybvh import analysis
from synthetic_bvh import make_pos_y_up_bvh


# ----------------------------------------------------------------
#  Jerk
# ----------------------------------------------------------------

def test_jerk_composition_identity_central_edge():
    bvh = make_pos_y_up_bvh()  # 10 frames, frame_time 1/30
    acc = analysis.node_accelerations(bvh)            # central/edge defaults
    jerk = analysis.node_jerk(bvh)
    np.testing.assert_allclose(
        np.gradient(acc, bvh.frame_time, axis=0), jerk, atol=1e-9)


def test_joint_jerk_is_node_jerk_on_joints():
    bvh = make_pos_y_up_bvh()
    nj = analysis.node_jerk(bvh)
    jj = analysis.joint_jerk(bvh)
    joint_idx = [i for i, n in enumerate(bvh.nodes) if not n.is_end_site()]
    np.testing.assert_allclose(jj, nj[:, joint_idx, :])


def test_jerk_shapes_per_stencil_pad():
    bvh = make_pos_y_up_bvh()  # F = 10
    n = len(bvh.nodes)
    assert analysis.node_jerk(bvh, stencil="central", pad="edge").shape == (10, n, 3)
    assert analysis.node_jerk(bvh, stencil="central", pad="none").shape == (4, n, 3)
    assert analysis.node_jerk(bvh, stencil="forward", pad="edge").shape == (10, n, 3)
    assert analysis.node_jerk(bvh, stencil="forward", pad="none").shape == (7, n, 3)


def test_jerk_too_short_raises():
    bvh = make_pos_y_up_bvh()
    short = bvh.slice_frames(0, 5)  # 5 frames < 7 needed for central+none
    with pytest.raises(ValueError):
        analysis.node_jerk(short, stencil="central", pad="none")


# ----------------------------------------------------------------
#  Smoothness dispatcher + simple kernels
# ----------------------------------------------------------------

def test_number_of_peaks_counts_local_maxima():
    assert analysis.number_of_peaks([0, 1, 0, 1, 0]) == 2
    assert analysis.number_of_peaks([0, 1, 2, 1, 0]) == 1
    assert analysis.number_of_peaks([0, 1, 2, 3, 4]) == 0  # monotone


def test_speed_metric_flat_profile_is_one():
    np.testing.assert_allclose(analysis.speed_metric(np.ones(20)), 1.0)


def test_smoothness_degenerate_inputs_are_graceful():
    # a perfectly still joint (zero speed) -> degenerate spectrum -> nan, no crash
    assert np.isnan(analysis.sparc(np.zeros(64), 100.0))
    # constant speed -> zero jerk -> perfectly smooth -> +inf (no leaked warning)
    assert analysis.log_dimensionless_jerk(np.full(50, 2.0), 100.0) == np.inf


def test_smoothness_dispatcher_matches_kernels_and_rejects_unknown():
    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    np.testing.assert_allclose(
        analysis.smoothness(s, fs, metric="sparc"), analysis.sparc(s, fs))
    np.testing.assert_allclose(
        analysis.smoothness(s, fs, metric="number_of_peaks"),
        analysis.number_of_peaks(s))
    with pytest.raises(ValueError):
        analysis.smoothness(s, fs, metric="not_a_metric")


# ----------------------------------------------------------------
#  Signal reductions
# ----------------------------------------------------------------

def test_velocity_reductions_known_profile():
    speed = np.array([0.0, 2.0, 4.0, 1.0])
    vr = analysis.velocity_reductions(speed, fs=1.0)
    assert vr.peak == 4.0
    np.testing.assert_allclose(vr.mean, 1.75)
    np.testing.assert_allclose(vr.peak_to_mean, 4.0 / 1.75)
    # largest decrease is 4 -> 1 (rate -3); peak_deceleration is +3
    np.testing.assert_allclose(vr.peak_deceleration, 3.0)


def test_zero_crossings_count_and_axis():
    np.testing.assert_array_equal(analysis.zero_crossings([1, -1, 1, -1]), 3)
    np.testing.assert_array_equal(analysis.zero_crossings([1, 0, -1]), 0)  # exact 0 not a crossing
    sig = np.array([[1, 1], [-1, 1], [1, 1]])  # col0 crosses twice, col1 never
    np.testing.assert_array_equal(analysis.zero_crossings(sig, axis=0), [2, 0])


def test_active_segments_and_duration():
    speed = np.array([0.0, 2, 3, 0, 5])
    np.testing.assert_array_equal(
        analysis.active_segments(speed, threshold=1.0),
        [False, True, True, False, True])
    np.testing.assert_allclose(
        analysis.active_duration(speed, threshold=1.0, frame_time=0.5), 1.5)


# ----------------------------------------------------------------
#  Kinetic energy
# ----------------------------------------------------------------

def test_kinetic_energy_unit_mass_matches_velocity_sum():
    bvh = make_pos_y_up_bvh()
    vel = analysis.joint_velocities(bvh)               # (F, J, 3)
    expected = np.sum(vel ** 2, axis=(1, 2))           # Σ‖v‖² per frame
    np.testing.assert_allclose(analysis.kinetic_energy(bvh), expected)


def test_kinetic_energy_with_masses():
    bvh = make_pos_y_up_bvh()
    vel = analysis.joint_velocities(bvh)
    masses = np.arange(1, vel.shape[1] + 1, dtype=float)
    expected = 0.5 * np.sum(masses * np.sum(vel ** 2, axis=-1), axis=-1)
    np.testing.assert_allclose(analysis.kinetic_energy(bvh, masses=masses), expected)


# ----------------------------------------------------------------
#  Gait
# ----------------------------------------------------------------

def test_walking_pace_translating_root():
    bvh = make_pos_y_up_bvh()
    # root moves z: 0 -> 50 over (10-1)/30 s; up is +y so horizontal dist = 50
    duration = (bvh.frame_count - 1) * bvh.frame_time
    np.testing.assert_allclose(analysis.walking_pace(bvh), 50.0 / duration)


def test_cadence_and_stride_from_known_contacts(monkeypatch):
    bvh = make_pos_y_up_bvh()
    contacts = np.zeros((bvh.frame_count, 2))
    contacts[2:4, 0] = 1   # foot 0: one onset at frame 2
    contacts[6:8, 0] = 1   # foot 0: another onset at frame 6
    contacts[4:6, 1] = 1   # foot 1: one onset at frame 4
    monkeypatch.setattr(analysis, "foot_contacts", lambda *a, **k: contacts)

    duration = (bvh.frame_count - 1) * bvh.frame_time
    np.testing.assert_allclose(analysis.cadence(bvh), 3 / duration)   # 3 onsets
    np.testing.assert_allclose(analysis.stride_length(bvh), 50.0 / (3 / 2))


def test_stride_length_nan_without_contacts(monkeypatch):
    bvh = make_pos_y_up_bvh()
    monkeypatch.setattr(analysis, "foot_contacts",
                        lambda *a, **k: np.zeros((bvh.frame_count, 2)))
    assert np.isnan(analysis.stride_length(bvh))
    assert analysis.cadence(bvh) == 0.0


# ----------------------------------------------------------------
#  Range of motion
# ----------------------------------------------------------------

def test_range_of_motion_peak_to_peak():
    sig = np.array([[1.0, 2], [3, 0], [2, 5]])
    np.testing.assert_allclose(analysis.range_of_motion(sig, axis=0), [2.0, 5.0])


# ----------------------------------------------------------------
#  Covariance descriptors
# ----------------------------------------------------------------

def test_cov3dj_shape_symmetry_and_value():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(40, 5, 3))  # (F, N, 3)
    cov = analysis.cov3dj(pos)
    assert cov.shape == (15, 15)
    np.testing.assert_allclose(cov, cov.T)
    flat = pos.reshape(40, -1)
    centered = flat - flat.mean(0)
    np.testing.assert_allclose(cov, centered.T @ centered / 40)


def test_lagged_correlation_lag0_and_bounds():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(30, 4))
    np.testing.assert_allclose(analysis.lagged_correlation(v, 0), v.T @ v / 30)
    m1 = analysis.lagged_correlation(v, 1)
    np.testing.assert_allclose(m1, v[1:].T @ v[:-1] / 29)
    assert m1.shape == (4, 4)
    with pytest.raises(ValueError):
        analysis.lagged_correlation(v, 30)
    with pytest.raises(ValueError):
        analysis.lagged_correlation(v, -1)
