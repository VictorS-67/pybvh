"""Analytic tests for the v0.8.0 pybvh.analysis primitives.

Covers jerk (composition identity), the smoothness dispatcher + simple
kernels (SPARC/DLJ/LDLJ exactness is in test_smoothness_golden.py), signal
reductions, kinetic energy, gait, range of motion, and the covariance
descriptors. Every assertion has a hand-derivable oracle.
"""
import numpy as np
import pytest

from pybvh import analysis
from synthetic_bvh import (make_clip_bvh, make_pos_y_up_bvh,
                           make_neg_y_up_bvh, make_pos_y_up_rotating_bvh)


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
    short = bvh[0:5]  # 5 frames < 7 needed for central+none
    with pytest.raises(ValueError):
        analysis.node_jerk(short, stencil="central", pad="none")


# ----------------------------------------------------------------
#  Speed derivative (tangential acceleration d‖v‖/dt)
# ----------------------------------------------------------------

def _sculpted_coords(bvh, traj):
    """Node-shaped ``(F, N, 3)`` coords with every node following ``traj``."""
    F, N = bvh.frame_count, len(bvh.nodes)
    return np.broadcast_to(traj[:, None, :], (F, N, 3)).copy()


def test_speed_derivative_zero_on_uniform_circular_motion():
    # constant speed, changing direction: d‖v‖/dt ~ 0 everywhere while the
    # (centripetal) acceleration vector is large — the delta-of-norm is NOT
    # recoverable from the norm-of-delta (node_accelerations).
    bvh = make_clip_bvh(n_frames=60, frame_time=1.0 / 30.0)
    t = np.arange(bvh.frame_count) * bvh.frame_time
    omega, radius = 2.0 * np.pi, 5.0
    circle = np.stack([radius * np.cos(omega * t),
                       np.zeros_like(t),
                       radius * np.sin(omega * t)], axis=-1)
    coords = _sculpted_coords(bvh, circle)
    sd = analysis.node_speed_derivative(bvh, coords=coords,
                                        stencil="central", pad="none")
    acc = analysis.node_accelerations(bvh, coords=coords,
                                      stencil="central", pad="none")
    np.testing.assert_allclose(sd, 0.0, atol=1e-9)
    # centripetal magnitude ~ omega^2 * radius ~ 197 units/s^2
    assert np.linalg.norm(acc, axis=-1).min() > 100.0


def test_speed_derivative_matches_analytic_on_accelerating_line():
    # p(t) = 1/2 a t^2 x̂ -> speed = a t -> d‖v‖/dt = a; the central stencil
    # is exact on quadratics, so pad="none" recovers a exactly.
    bvh = make_clip_bvh(n_frames=20, frame_time=1.0 / 30.0)
    t = np.arange(bvh.frame_count) * bvh.frame_time
    a = 3.0
    line = np.zeros((bvh.frame_count, 3))
    line[:, 0] = 0.5 * a * t ** 2
    coords = _sculpted_coords(bvh, line)
    sd = analysis.node_speed_derivative(bvh, coords=coords,
                                        stencil="central", pad="none")
    np.testing.assert_allclose(sd, a, atol=1e-9)


def test_joint_speed_derivative_is_node_variant_on_joints():
    bvh = make_pos_y_up_rotating_bvh()
    nsd = analysis.node_speed_derivative(bvh)
    jsd = analysis.joint_speed_derivative(bvh)
    joint_idx = [i for i, n in enumerate(bvh.nodes) if not n.is_end_site()]
    np.testing.assert_allclose(jsd, nsd[:, joint_idx])


def test_speed_derivative_shapes_per_stencil_pad():
    bvh = make_pos_y_up_bvh()  # F = 10
    n = len(bvh.nodes)
    assert analysis.node_speed_derivative(bvh, stencil="central", pad="edge").shape == (10, n)
    assert analysis.node_speed_derivative(bvh, stencil="central", pad="none").shape == (6, n)
    assert analysis.node_speed_derivative(bvh, stencil="forward", pad="edge").shape == (10, n)
    assert analysis.node_speed_derivative(bvh, stencil="forward", pad="none").shape == (8, n)


def test_speed_derivative_too_short_raises():
    bvh = make_pos_y_up_bvh()
    short = bvh[0:4]  # 4 frames < 5 needed for central+none (two applications)
    with pytest.raises(ValueError):
        analysis.node_speed_derivative(short, stencil="central", pad="none")


def test_joint_speed_derivative_rejects_joint_shaped_coords():
    bvh = make_pos_y_up_bvh()
    joint_coords = bvh.node_positions()[:, :bvh.joint_count, :]
    with pytest.raises(ValueError, match="node-shaped"):
        analysis.joint_speed_derivative(bvh, coords=joint_coords)


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
    # zero speed -> zero peak -> DLJ normalization undefined -> nan, no warning
    assert np.isnan(analysis.dimensionless_jerk(np.zeros(50), 100.0))
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
#  Trailing-axis (batched) reduction kernels
# ----------------------------------------------------------------

_ALL_SMOOTHNESS_METRICS = [
    "sparc", "dimensionless_jerk", "log_dimensionless_jerk",
    "integrated_squared_jerk", "mean_squared_jerk", "rms_squared_jerk",
    "number_of_peaks", "speed_metric",
]


@pytest.mark.parametrize("metric", _ALL_SMOOTHNESS_METRICS)
def test_smoothness_kernels_batch_equals_python_loop(metric):
    rng = np.random.default_rng(3)
    speeds = np.abs(rng.normal(size=(64, 5))) + 0.1
    fs = 100.0
    batched = analysis.smoothness(speeds, fs, metric=metric)
    looped = np.array([analysis.smoothness(speeds[:, k], fs, metric=metric)
                       for k in range(speeds.shape[1])])
    assert batched.shape == (5,)
    # last-ulp tolerance: axis-0 reductions may sum in a different order
    # than the contiguous 1-D column reduction
    np.testing.assert_allclose(batched, looped, rtol=1e-12)


def test_velocity_reductions_batch_equals_python_loop():
    rng = np.random.default_rng(4)
    speeds = np.abs(rng.normal(size=(50, 5)))
    fs = 30.0
    batched = analysis.velocity_reductions(speeds, fs)
    for k in range(speeds.shape[1]):
        single = analysis.velocity_reductions(speeds[:, k], fs)
        for field_batched, field_single in zip(batched, single):
            assert field_batched.shape == (5,)
            np.testing.assert_allclose(field_batched[k], field_single,
                                       rtol=1e-12)


def test_active_duration_batch_equals_python_loop():
    rng = np.random.default_rng(5)
    speeds = np.abs(rng.normal(size=(40, 5)))
    batched = analysis.active_duration(speeds, threshold=0.5, fs=2.0)
    looped = np.array([analysis.active_duration(speeds[:, k], 0.5, 2.0)
                       for k in range(speeds.shape[1])])
    assert batched.shape == (5,)
    np.testing.assert_array_equal(batched, looped)


def test_batched_kernels_per_column_nan_isolation():
    # a degenerate column goes nan on its own; its neighbors are unaffected
    rng = np.random.default_rng(6)
    speeds = np.abs(rng.normal(size=(64, 3))) + 0.1
    speeds[:, 1] = 0.0  # zero peak -> nan (sparc / DLJ / speed_metric)
    for metric in ("sparc", "dimensionless_jerk", "speed_metric"):
        out = analysis.smoothness(speeds, 100.0, metric=metric)
        assert np.isnan(out[1])
        assert np.isfinite(out[[0, 2]]).all()
    # LDLJ: a constant-speed column is perfectly smooth -> +inf, per column
    speeds[:, 1] = 2.0
    ldlj = analysis.smoothness(speeds, 100.0, metric="log_dimensionless_jerk")
    assert ldlj[1] == np.inf
    assert np.isfinite(ldlj[[0, 2]]).all()


def test_velocity_reductions_batch_per_column_nan_and_clamp():
    speeds = np.zeros((10, 2))
    speeds[:, 1] = np.linspace(0.0, 1.0, 10)  # monotonically rising
    vr = analysis.velocity_reductions(speeds, fs=1.0)
    assert np.isnan(vr.peak_to_mean[0])       # zero-mean column -> nan
    np.testing.assert_allclose(vr.peak_to_mean[1], 2.0)  # 1.0 / 0.5
    assert vr.peak_deceleration[1] == 0.0     # never falls -> clamped at 0
    assert vr.peak_acceleration[0] == 0.0     # never rises -> clamped at 0


def test_reduction_kernels_reject_higher_rank_input():
    bad = np.ones((4, 3, 2))
    with pytest.raises(ValueError, match=r"\(T,\)"):
        analysis.velocity_reductions(bad, fs=1.0)
    with pytest.raises(ValueError, match=r"\(T, K\)"):
        analysis.active_duration(bad, threshold=0.5, fs=1.0)
    for metric in _ALL_SMOOTHNESS_METRICS:
        with pytest.raises(ValueError, match=r"\(T, K\)"):
            analysis.smoothness(bad, 100.0, metric=metric)


# ----------------------------------------------------------------
#  Signal reductions
# ----------------------------------------------------------------

def test_velocity_reductions_known_profile():
    speed = np.array([0.0, 2.0, 4.0, 1.0])
    vr = analysis.velocity_reductions(speed, fs=1.0)
    assert vr.peak == 4.0
    np.testing.assert_allclose(vr.mean, 1.75)
    np.testing.assert_allclose(vr.peak_to_mean, 4.0 / 1.75)
    # rates are [+2, +2, -3]: largest increase is +2, largest decrease is +3
    np.testing.assert_allclose(vr.peak_acceleration, 2.0)
    np.testing.assert_allclose(vr.peak_deceleration, 3.0)


def test_velocity_reductions_directional_rates_clamp_at_zero():
    # a monotonically rising profile never decelerates (and vice versa);
    # both directional rates are >= 0 per the contract.
    rising = np.array([0.0, 1.0, 2.0, 3.0])
    vr = analysis.velocity_reductions(rising, fs=1.0)
    np.testing.assert_allclose(vr.peak_acceleration, 1.0)
    assert vr.peak_deceleration == 0.0
    falling = rising[::-1]
    vr = analysis.velocity_reductions(falling, fs=1.0)
    assert vr.peak_acceleration == 0.0
    np.testing.assert_allclose(vr.peak_deceleration, 1.0)


def test_velocity_reductions_extrema_are_speed_derivative_extrema():
    # peak_acceleration / peak_deceleration are the clamped positive/negative
    # extrema of the per-frame d‖v‖/dt series: velocity_reductions on the
    # forward/none speed profile must agree with node_speed_derivative under
    # the same convention. Asserted so the two can never drift apart.
    bvh = make_pos_y_up_rotating_bvh()
    fs = 1.0 / bvh.frame_time
    speed = np.linalg.norm(
        analysis.node_velocities(bvh, stencil="forward", pad="none"), axis=-1)
    sd = analysis.node_speed_derivative(bvh, stencil="forward", pad="none")
    vr = analysis.velocity_reductions(speed, fs)
    assert (sd > 0).any() and (sd < 0).any()  # non-degenerate motion
    np.testing.assert_allclose(
        vr.peak_acceleration, np.maximum(sd.max(axis=0), 0.0), rtol=1e-12)
    np.testing.assert_allclose(
        vr.peak_deceleration, np.maximum(-sd.min(axis=0), 0.0), rtol=1e-12)


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
    # 3 active samples at 2 Hz -> 1.5 s
    np.testing.assert_allclose(
        analysis.active_duration(speed, threshold=1.0, fs=2.0), 1.5)


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


def test_kinetic_energy_masses_dict_matches_ordered_array():
    bvh = make_pos_y_up_bvh()
    names = bvh.joint_names
    masses_arr = np.arange(1, len(names) + 1, dtype=float)
    masses_dict = {n: float(m) for n, m in zip(names, masses_arr)}
    # name-keyed dict is order-independent: shuffle it, result is unchanged
    shuffled = dict(reversed(list(masses_dict.items())))
    np.testing.assert_allclose(
        analysis.kinetic_energy(bvh, masses=shuffled),
        analysis.kinetic_energy(bvh, masses=masses_arr))


def test_kinetic_energy_masses_validation():
    bvh = make_pos_y_up_bvh()
    names = bvh.joint_names
    full = {n: 1.0 for n in names}
    with pytest.raises(ValueError, match="missing masses"):
        incomplete = {n: 1.0 for n in names[:-1]}
        analysis.kinetic_energy(bvh, masses=incomplete)
    with pytest.raises(ValueError, match="unknown joint"):
        analysis.kinetic_energy(bvh, masses={**full, "NotAJoint": 1.0})
    with pytest.raises(ValueError, match="shape"):
        analysis.kinetic_energy(bvh, masses=np.ones(len(names) - 1))


def test_kinetic_energy_nonpositive_mass_total_raises():
    # an all-zero (or negative-total) mass vector would silently zero the
    # energy — both the array and the dict form must reject it
    bvh = make_pos_y_up_bvh()
    names = bvh.joint_names
    with pytest.raises(ValueError, match="positive total"):
        analysis.kinetic_energy(bvh, masses=np.zeros(len(names)))
    with pytest.raises(ValueError, match="positive total"):
        analysis.kinetic_energy(bvh, masses={n: 0.0 for n in names})
    negative_total = np.zeros(len(names))
    negative_total[0] = -1.0
    with pytest.raises(ValueError, match="positive total"):
        analysis.kinetic_energy(bvh, masses=negative_total)
    # the unit-mass masses=None default path is untouched
    assert np.all(np.isfinite(analysis.kinetic_energy(bvh)))


# ----------------------------------------------------------------
#  Gait
# ----------------------------------------------------------------

def test_walking_pace_translating_root():
    bvh = make_pos_y_up_bvh()
    # root moves z: 0 -> 50 over (10-1)/30 s; up is +y so horizontal dist = 50
    duration = (bvh.frame_count - 1) * bvh.frame_time
    np.testing.assert_allclose(analysis.walking_pace(bvh), 50.0 / duration)


def test_cadence_from_known_contacts():
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]              # end-site feet, passed explicitly
    contacts = np.zeros((bvh.frame_count, 2))
    contacts[2:4, 0] = 1   # foot 0: onset at frame 2
    contacts[6:8, 0] = 1   # foot 0: onset at frame 6
    contacts[4:6, 1] = 1   # foot 1: onset at frame 4

    duration = (bvh.frame_count - 1) * bvh.frame_time
    np.testing.assert_allclose(
        analysis.cadence(bvh, foot_joints=feet, contacts=contacts),
        3 / duration)                              # 3 onsets
    # foot-measured stride_length is covered by the _compute_gait_parameters tests


def test_stride_length_nan_without_contacts():
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]              # end-site feet, passed explicitly
    no_contacts = np.zeros((bvh.frame_count, 2))
    assert np.isnan(analysis.stride_length(bvh, foot_joints=feet,
                                           contacts=no_contacts))
    assert analysis.cadence(bvh, foot_joints=feet, contacts=no_contacts) == 0.0


def _two_foot_walk():
    """Clean periodic 2-foot walk with known landings.

    Left onsets at frames 1,5,9 (x = 0,2,4); Right onsets at 3,7,11 (x = 1,3,5);
    each contact lasts 2 frames; feet never overlap.
    """
    F = 13
    contacts = np.zeros((F, 2))
    for s, e in [(1, 3), (5, 7), (9, 11)]:
        contacts[s:e, 0] = 1
    for s, e in [(3, 5), (7, 9), (11, 13)]:
        contacts[s:e, 1] = 1
    foot_h = np.zeros((F, 2, 3))
    foot_h[[1, 5, 9], 0, 0] = [0.0, 2.0, 4.0]
    foot_h[[3, 7, 11], 1, 0] = [1.0, 3.0, 5.0]
    return contacts, foot_h


def test_gait_core_symmetric_walk():
    contacts, foot_h = _two_foot_walk()
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 11.0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.stride_length, 2.0)             # 0→2→4, 1→3→5
    np.testing.assert_allclose(g.stride_cv, 0.0)                 # all strides equal
    np.testing.assert_allclose(g.step_length, 1.0)              # landings 0,1,2,3,4,5
    np.testing.assert_allclose(g.stance_fraction, 0.5)          # 2-frame stance, 4 cycle
    np.testing.assert_allclose(g.double_support_fraction, 0.0)  # feet never overlap
    np.testing.assert_allclose(g.asymmetry, 0.0)               # L and R equal
    np.testing.assert_allclose(g.cadence, 6 / 12)              # 6 onsets / 12 s
    np.testing.assert_allclose(g.walking_pace, 11.0 / 12)


def test_gait_core_asymmetry_and_cv():
    contacts, foot_h = _two_foot_walk()
    foot_h[[3, 7, 11], 1, 0] = [0.0, 1.0, 2.0]    # Right strides 1,1 (Left stays 2,2)
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    # strides: Left [2,2], Right [1,1] -> mean 1.5
    np.testing.assert_allclose(g.stride_length, 1.5)
    # each foot is internally constant, so within-foot variability is 0;
    # the left/right difference shows up in asymmetry, not stride_cv
    np.testing.assert_allclose(g.stride_cv, 0.0)
    np.testing.assert_allclose(g.asymmetry, abs(2 - 1) / 1.5)


def test_gait_core_stride_cv_is_within_foot():
    F = 13
    contacts = np.zeros((F, 1))
    for s, e in [(1, 3), (5, 7), (9, 11)]:
        contacts[s:e, 0] = 1                       # onsets 1,5,9
    foot_h = np.zeros((F, 1, 3))
    foot_h[[1, 5, 9], 0, 0] = [0.0, 2.0, 6.0]      # strides 2, 4 (mean 3)
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.stride_length, 3.0)
    np.testing.assert_allclose(g.stride_cv, np.std([2.0, 4.0]) / 3.0)


def test_gait_core_stride_cv_nan_with_single_stride_per_foot():
    # two feet, one stride each -> stride_length defined, but no within-foot
    # variability is measurable, so stride_cv is nan (not a misleading 0)
    F = 9
    contacts = np.zeros((F, 2))
    for s, e in [(1, 3), (5, 7)]:
        contacts[s:e, 0] = 1                       # Left onsets 1,5  -> 1 stride
    for s, e in [(3, 5), (7, 9)]:
        contacts[s:e, 1] = 1                       # Right onsets 3,7 -> 1 stride
    foot_h = np.zeros((F, 2, 3))
    foot_h[[1, 5], 0, 0] = [0.0, 2.0]
    foot_h[[3, 7], 1, 0] = [1.0, 3.0]
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    assert not np.isnan(g.stride_length)           # 2 strides pooled
    assert np.isnan(g.stride_cv)                   # no foot has >= 2 strides


def test_gait_core_double_support():
    contacts = np.zeros((4, 2))
    contacts[0:3, 0] = 1     # foot 0 planted 0,1,2
    contacts[1:4, 1] = 1     # foot 1 planted 1,2,3  -> overlap 1,2
    g = analysis._compute_gait_parameters(
        contacts, np.zeros((4, 2, 3)), ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.double_support_fraction, 2 / 4)


def test_gait_core_nan_edges():
    contacts = np.zeros((6, 1))
    contacts[1:3, 0] = 1     # a single foot, one onset only
    g = analysis._compute_gait_parameters(
        contacts, np.zeros((6, 1, 3)), ["LeftFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    assert np.isnan(g.stride_length)                # <2 contacts
    assert np.isnan(g.asymmetry)                    # no left/right pair
    assert np.isnan(g.double_support_fraction)      # only one foot


def test_gait_core_step_length_is_forward_only():
    # opposite feet sit far apart laterally (y) but advance little forward (x);
    # step_length must measure the forward advance, not the lateral step width.
    F = 8
    contacts = np.zeros((F, 2))
    contacts[1:3, 0] = 1; contacts[5:7, 0] = 1     # left onsets 1,5
    contacts[3:5, 1] = 1                            # right onset 3
    foot_h = np.zeros((F, 2, 3))
    foot_h[[1, 5], 0, :] = [[0.0, 5.0, 0.0], [2.0, 5.0, 0.0]]   # left forward 0->2, y=+5
    foot_h[3, 1, :] = [1.0, -5.0, 0.0]                          # right x=1, y=-5
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    # landings by frame L@1(x0), R@3(x1), L@5(x2): forward advances |1-0|,|2-1| = 1
    np.testing.assert_allclose(g.step_length, 1.0)   # Euclidean would be ~10 (y-gap)


def test_gait_core_step_length_nan_without_net_travel():
    contacts, foot_h = _two_foot_walk()
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [0.0, 0.0, 0.0])
    assert np.isnan(g.step_length)


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


def test_lagged_covariance_centers_lag0_and_bounds():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(30, 4)) + 7.0     # constant offset must contribute nothing
    c = v - v.mean(axis=0)                 # centered on the temporal mean
    np.testing.assert_allclose(analysis.lagged_covariance(v, 0), c.T @ c / 30)
    m1 = analysis.lagged_covariance(v, 1)
    np.testing.assert_allclose(m1, c[1:].T @ c[:-1] / 29)
    assert m1.shape == (4, 4)
    # lag 0 on centered input == population covariance == cov3dj convention
    np.testing.assert_allclose(
        analysis.lagged_covariance(v, 0), np.cov(v.T, bias=True))
    with pytest.raises(ValueError):
        analysis.lagged_covariance(v, 30)
    with pytest.raises(ValueError):
        analysis.lagged_covariance(v, -1)


# ----------------------------------------------------------------
#  foot_contacts detection core (hysteresis / Otsu / confidence /
#  timing refinement / diagnostics) — pure-signal oracles
# ----------------------------------------------------------------

def test_hysteresis_mask_equals_single_threshold_at_zero_band():
    sig = np.array([[5.0, 0.5], [1.5, 3.0], [0.2, 0.1]])
    np.testing.assert_array_equal(analysis._hysteresis_mask(sig, 1.0, 1.0), sig < 1.0)


def test_hysteresis_keeps_strong_runs_and_drops_weak_only():
    # contact = LOW signal; low=1, high=2
    sig = np.array([
        [5.0, 5.0],
        [1.5, 1.5],   # weak (between 1 and 2)
        [0.5, 5.0],   # col0 strong here; col1 back high
        [1.5, 5.0],   # weak
        [5.0, 5.0],
    ])
    got = analysis._hysteresis_mask(sig, 1.0, 2.0)
    expected = np.array([
        [False, False],
        [True,  False],   # col0 run rows1-3 contains a strong frame -> kept
        [True,  False],
        [True,  False],
        [False, False],   # col1 isolated weak-only frame -> dropped
    ])
    np.testing.assert_array_equal(got, expected)


def test_otsu_finds_valley_on_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0.0, 0.05, 200), rng.normal(2.0, 0.05, 200)])
    thr, strength = analysis._otsu_threshold(v)
    assert thr is not None and 0.3 < thr < 1.7 and strength > 0.6


def test_otsu_fallback_on_unimodal_and_edges():
    rng = np.random.default_rng(1)
    assert analysis._otsu_threshold(rng.normal(1.0, 0.1, 300))[0] is None  # unimodal
    assert analysis._otsu_threshold(np.zeros(50))[0] is None               # zero variance
    assert analysis._otsu_threshold(np.array([1.0, 2.0, 3.0]))[0] is None  # < 8 samples


def test_resolve_adaptive_per_foot_fallback_and_clamp():
    rng = np.random.default_rng(2)
    sig = np.empty((400, 2))
    sig[:, 0] = np.concatenate([rng.normal(0, 0.02, 200), rng.normal(1.0, 0.02, 200)])
    sig[:, 1] = rng.normal(0.5, 0.1, 400)   # unimodal -> fallback
    base = 0.05
    thr, used = analysis._resolve_adaptive(sig, base)
    assert used[0] and not used[1]
    np.testing.assert_allclose(thr[1], base)               # fallback
    assert 0.25 * base <= thr[0] <= 4.0 * base             # clamped


def test_detect_contacts_recovers_plant_lift_two_feet():
    F = 10
    speed = np.full((F, 2), 5.0); clearance = np.full((F, 2), 5.0)
    speed[3:7, 0] = 0.1; clearance[3:7, 0] = 0.1
    speed[1:4, 1] = 0.1; clearance[1:4, 1] = 0.1
    mask, _ = analysis._detect_contacts(
        speed, clearance, method="combined",
        vel_threshold=1.0, height_threshold=1.0, hysteresis=0.0)
    expected = np.zeros((F, 2), bool); expected[3:7, 0] = True; expected[1:4, 1] = True
    np.testing.assert_array_equal(mask, expected)


def test_contact_confidence_combined_formula():
    speed = np.array([[5.0], [0.5], [0.5], [5.0]])
    clearance = speed.copy()
    _, conf = analysis._detect_contacts(
        speed, clearance, method="combined",
        vel_threshold=1.0, height_threshold=1.0, hysteresis=0.0)
    # contact rows 1,2: margin=(1-0.5)/1=0.5 per signal; masks identical -> agreement 1
    np.testing.assert_allclose(conf, np.sqrt(0.5 * 1.0))


def test_foot_contacts_detects_bouncing_root_dwell():
    # Feet dwell low (contact) on known frames and lift between -> the height
    # method must recover the dwells. Drive the foot trajectory via `coords`.
    bvh = make_pos_y_up_bvh()
    up = 1  # +y
    coords = bvh.node_positions().copy()
    for f in [bvh.index(n, space="node") for n in ["LeftFoot", "RightFoot"]]:
        coords[4:7, f, up] += 40.0         # lift frames 4-6; dwell low elsewhere
    c = bvh.foot_contacts(foot_joints=["LeftFoot", "RightFoot"],
                          method="height", coords=coords, hysteresis=0.0)
    assert c[0:4].all() and c[7:10].all()  # contact during the low dwells
    assert not c[4:7].any()                # airborne during the lift


# --- boundary-aware hysteresis release (open runs trimmed to raw support) ---

def test_release_open_runs_trims_truncated_swing():
    # filled run reaches the last frame, but raw support ends earlier: the
    # trailing band is speculative (a swing cut off by the clip) -> dropped.
    F = 8
    mask = np.zeros((F, 1), bool); mask[2:8, 0] = True   # filled run [2,8) touches end
    raw = np.zeros((F, 1), bool); raw[2:5, 0] = True     # raw lift at frame 5
    out = analysis._release_open_runs(mask, raw)
    expected = np.zeros((F, 1), bool); expected[2:5, 0] = True
    np.testing.assert_array_equal(out, expected)


def test_release_open_runs_trims_leading_band():
    # symmetric at the start: a run touching frame 0 trims to where raw begins.
    F = 8
    mask = np.zeros((F, 1), bool); mask[0:6, 0] = True
    raw = np.zeros((F, 1), bool); raw[3:6, 0] = True
    out = analysis._release_open_runs(mask, raw)
    expected = np.zeros((F, 1), bool); expected[3:6, 0] = True
    np.testing.assert_array_equal(out, expected)


def test_release_open_runs_leaves_closed_and_supported_runs():
    # col0: interior run (touches neither edge) -> untouched even with no raw.
    # col1: boundary run raw-supported to the edge -> untouched.
    # col2: planted the whole clip (raw all true) -> untouched.
    F = 10
    mask = np.zeros((F, 3), bool); raw = np.zeros((F, 3), bool)
    mask[3:6, 0] = True
    mask[7:10, 1] = True; raw[7:10, 1] = True
    mask[:, 2] = True; raw[:, 2] = True
    out = analysis._release_open_runs(mask, raw)
    np.testing.assert_array_equal(out, mask)


def test_detect_contacts_releases_open_swing_under_hysteresis():
    # thr=1, band +-0.5 -> low=0.5 high=1.5 raw=<1. A deep contact rising into
    # the band at the clip end: the latch must release at the raw lift, not hold.
    F = 8
    speed = np.array([5, 5, 0.1, 0.1, 0.1, 0.8, 1.2, 1.4]).reshape(F, 1)
    mask, _ = analysis._detect_contacts(
        speed, None, method="velocity",
        vel_threshold=1.0, height_threshold=None, hysteresis=0.5)
    expected = np.zeros((F, 1), bool); expected[2:6, 0] = True   # released at frame 6
    np.testing.assert_array_equal(mask, expected)


def test_detect_contacts_keeps_genuine_boundary_contact():
    # control: signal stays in deep contact to the last frame -> raw supports the
    # edge, nothing is trimmed.
    F = 8
    speed = np.array([5, 5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(F, 1)
    mask, _ = analysis._detect_contacts(
        speed, None, method="velocity",
        vel_threshold=1.0, height_threshold=None, hysteresis=0.5)
    expected = np.zeros((F, 1), bool); expected[2:8, 0] = True
    np.testing.assert_array_equal(mask, expected)


def test_foot_contacts_releases_foot_lifting_at_clip_end():
    # end-to-end: a foot planted then lifting in the final frames, clip cut
    # mid-lift. With default hysteresis the latch must not hold it to the edge.
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]
    fis = [bvh.index(n, space="node") for n in feet]
    coords = bvh.node_positions().copy()
    coords[:, fis, 1] = 0.0          # planted at the floor...
    coords[7:10, fis, 1] = 1.1       # ...then in the band (above raw thr=1, below high=1.25)
    c = np.asarray(bvh.foot_contacts(
        foot_joints=feet, method="height", coords=coords, floor=0.0,
        height_threshold=1.0, min_contact_duration=0.0, min_gap_duration=0.0))
    assert c[:7].all()               # genuine plant kept
    assert not c[7:].any()           # open-end lift released, not held to frame 9


def test_contact_diagnostics_skate_airborne_height():
    F = 6
    mask = np.zeros((F, 2), bool); mask[1:4, 0] = True; mask[2:5, 1] = True
    foot_coords = np.zeros((F, 2, 3))
    foot_coords[2:5, 1, 0] = [0.0, 1.5, 3.0]    # foot1 slides 3 in x during contact
    clearance = np.zeros((F, 2))                # all on the floor
    diag = analysis._contact_diagnostics(mask, foot_coords, clearance, up_idx=2, scale=10.0)
    np.testing.assert_allclose(diag["foot_skate"]["max"][0], 0.0)
    np.testing.assert_allclose(diag["foot_skate"]["max"][1], 3.0 / 10.0)
    np.testing.assert_allclose(diag["airborne_fraction"], 2 / 6)   # rows 0,5 have no contact
    np.testing.assert_allclose(diag["height_at_contact"], [0.0, 0.0])


# ----------------------------------------------------------------
#  Velocity-informed height threshold + floor_height
# ----------------------------------------------------------------

def test_velocity_informed_height_reduces_to_margin_on_floor():
    F, margin = 20, 1.0
    clearance = np.zeros((F, 1)); clearance[10:, 0] = 10.0     # stance on floor, swing high
    speed = np.zeros((F, 1)); speed[10:, 0] = 5.0             # slow then fast
    thr = analysis._velocity_informed_height(clearance, speed, 1.0, margin)
    np.testing.assert_allclose(thr, [margin])                # contact_h≈0 -> margin


def test_velocity_informed_height_calibrates_to_hover():
    F, margin, h0 = 20, 1.0, 5.0
    clearance = np.full((F, 1), h0); clearance[10:, 0] = h0 + 5 * margin
    speed = np.zeros((F, 1)); speed[10:, 0] = 5.0
    thr = analysis._velocity_informed_height(clearance, speed, 1.0, margin)
    np.testing.assert_allclose(thr, [h0 + margin])           # lifts to the hover level


def test_velocity_informed_height_guard_rejects_held_airborne():
    F, margin = 20, 1.0
    clearance = np.full((F, 1), 8.0)                          # flat high, no swing
    speed = np.zeros((F, 1))                                  # all slow (held still)
    thr = analysis._velocity_informed_height(clearance, speed, 1.0, margin)
    np.testing.assert_allclose(thr, [margin])                # guard -> fixed -> rejected


def test_velocity_informed_height_per_foot_and_no_slow():
    F, margin = 20, 1.0
    clearance = np.zeros((F, 2))
    clearance[:, 0] = 5.0; clearance[10:, 0] = 10.0           # foot0 hovers at 5
    clearance[10:, 1] = 10.0                                  # foot1 reaches floor
    speed = np.zeros((F, 2)); speed[10:, :] = 5.0
    thr = analysis._velocity_informed_height(clearance, speed, 1.0, margin)
    np.testing.assert_allclose(thr, [5 + margin, 0 + margin])  # distinct per foot
    # never slow -> fixed margin
    thr2 = analysis._velocity_informed_height(np.zeros((F, 1)), np.full((F, 1), 5.0), 1.0, margin)
    np.testing.assert_allclose(thr2, [margin])


def test_floor_height_cache_invalidates_on_mutation():
    bvh = make_pos_y_up_bvh()
    f0 = bvh.floor_height
    bvh.root_pos = bvh.root_pos + np.array([0.0, 7.0, 0.0])   # +y shift
    np.testing.assert_allclose(bvh.floor_height, f0 + 7.0)    # recomputed, not stale


def test_floor_height_footless_fallback_to_all_nodes():
    bvh = make_pos_y_up_bvh()
    assert analysis.auto_detect_foot_joints(bvh) == []        # feet are end sites
    coords = bvh.node_positions()
    expected = float(np.percentile(coords[:, :, 1].min(axis=1), 2.0))  # all-nodes min, +y
    np.testing.assert_allclose(bvh.floor_height, expected)


def test_floor_height_negative_up_raw_coords_and_copy():
    bvh = make_neg_y_up_bvh()
    assert bvh.floor_height == bvh.copy().floor_height        # survives copy
    # raw coordinate (not sign-corrected): for -y up the floor is a high +y value
    coords = bvh.node_positions()
    expected = float(np.percentile((coords[:, :, 1] * -1).min(axis=1), 2.0) * -1)
    np.testing.assert_allclose(bvh.floor_height, expected)


def _hover_clip(hover_frac, swing_frac=0.3):
    """make_pos_y_up_bvh + coords where each foot is fully stationary during stance
    (at floor+hover) then swings high and moving. base = the floor (rest foot height)."""
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]
    fidx = [bvh.index(n, space="node") for n in feet]
    scale = analysis._skeleton_scale(bvh.rest_pose_positions(), fidx)
    coords = bvh.node_positions().copy()
    F = bvh.frame_count; half = F // 2
    for f in fidx:
        rest = coords[0, f, :].copy()
        stance = rest.copy(); stance[1] = rest[1] + hover_frac * scale
        coords[:half, f, :] = stance                                   # fully still (slow)
        coords[half:, f, :] = stance
        coords[half:, f, 1] = rest[1] + swing_frac * scale             # swing high
        coords[half:, f, 0] = stance[0] + np.linspace(0.1 * scale, scale, F - half)  # moving -> fast
    base = float(coords[0, fidx[0], 1] - hover_frac * scale)           # the floor (rest height)
    return bvh, feet, coords, base


def test_height_reference_recovers_hovering_stance():
    # hysteresis=0 isolates the height-threshold logic from the band machinery.
    bvh, feet, coords, base = _hover_clip(hover_frac=0.05)
    # explicit low floor (below the hover) mimics the retargeting artifact
    cv = bvh.foot_contacts(foot_joints=feet, coords=coords, floor=float(base),
                           height_reference="velocity", hysteresis=0.0)
    cf = bvh.foot_contacts(foot_joints=feet, coords=coords, floor=float(base),
                           height_reference="floor", hysteresis=0.0)
    assert cv.sum() > cf.sum()   # velocity recovers stance the floor-anchored threshold rejects


def test_height_reference_clean_rig_identity():
    # feet reach the floor during stance -> velocity and floor anchoring are
    # identical (same threshold), regardless of hysteresis.
    bvh, feet, coords, base = _hover_clip(hover_frac=0.0)
    cv = bvh.foot_contacts(foot_joints=feet, coords=coords, height_reference="velocity")
    cf = bvh.foot_contacts(foot_joints=feet, coords=coords, height_reference="floor")
    np.testing.assert_array_equal(cv, cf)


def test_height_reference_rejects_held_airborne_foot():
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]
    fidx = [bvh.index(n, space="node") for n in feet]
    scale = analysis._skeleton_scale(bvh.rest_pose_positions(), fidx)
    coords = bvh.node_positions().copy()
    F = bvh.frame_count; base = coords[0, fidx[0], 1]; half = F // 2
    coords[:, fidx[1], :] = coords[0, fidx[1], :]             # foot1 held still...
    coords[:, fidx[1], 1] = base + 0.3 * scale                # ...and high (airborne)
    coords[:half, fidx[0], :] = coords[0, fidx[0], :]; coords[:half, fidx[0], 1] = base  # foot0 stance on floor
    coords[half:, fidx[0], 1] = base + 0.3 * scale; coords[half:, fidx[0], 0] = np.linspace(0, scale, F - half)
    cv = bvh.foot_contacts(foot_joints=feet, coords=coords, height_reference="velocity", hysteresis=0.0)
    assert cv[:, 1].sum() == 0                                # held-airborne foot rejected
