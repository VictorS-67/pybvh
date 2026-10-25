"""Analytic tests for the v0.8.0 pybvh.analysis primitives.

Covers jerk (composition identity), the smoothness dispatcher + simple
kernels (SPARC/DLJ/LDLJ exactness is in test_smoothness_golden.py), signal
reductions, kinetic energy, gait, range of motion, and the covariance
descriptors. Every assertion has a hand-derivable oracle.
"""

import inspect
import re

import numpy as np
import pytest
from synthetic_bvh import (
    CONTACT_FEET,
    contact_coords,
    make_clip_bvh,
    make_contact_rig,
    make_neg_y_up_bvh,
    make_pos_y_up_bvh,
    make_pos_y_up_rotating_bvh,
)

from pybvh import analysis
from pybvh.bvh import Bvh

# ----------------------------------------------------------------
#  Jerk
# ----------------------------------------------------------------


def test_jerk_composition_identity_central_edge():
    bvh = make_pos_y_up_bvh()  # 10 frames, frame_time 1/30
    acc = analysis.node_accelerations(bvh)  # central/edge defaults
    jerk = analysis.node_jerk(bvh)
    np.testing.assert_allclose(np.gradient(acc, bvh.frame_time, axis=0), jerk, atol=1e-9)


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
    circle = np.stack(
        [radius * np.cos(omega * t), np.zeros_like(t), radius * np.sin(omega * t)], axis=-1
    )
    coords = _sculpted_coords(bvh, circle)
    sd = analysis.node_speed_derivative(bvh, coords=coords, stencil="central", pad="none")
    acc = analysis.node_accelerations(bvh, coords=coords, stencil="central", pad="none")
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
    line[:, 0] = 0.5 * a * t**2
    coords = _sculpted_coords(bvh, line)
    sd = analysis.node_speed_derivative(bvh, coords=coords, stencil="central", pad="none")
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
    joint_coords = bvh.node_positions()[:, : bvh.joint_count, :]
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


def test_speed_metric_stays_in_range_for_a_signed_velocity():
    """A signed velocity must not produce a ratio outside the documented [0, 1]."""
    signed = np.concatenate([np.ones(25), -np.ones(25)])  # raw mean is 0
    np.testing.assert_allclose(analysis.speed_metric(signed), 1.0)
    # unchanged for a genuine (non-negative) speed profile
    rng = np.random.default_rng(20)
    s = np.abs(rng.normal(size=64)) + 0.1
    np.testing.assert_allclose(analysis.speed_metric(s), s.mean() / s.max(), rtol=1e-12)


def test_speed_metric_is_the_exact_dlj_normalizer_ratio_for_signed_input():
    """One magnitude convention across both, so the identity survives sign."""
    signed = np.concatenate([np.ones(25), -np.ones(25)])
    fs = 100.0
    np.testing.assert_allclose(
        analysis.dimensionless_jerk(signed, fs, normalize="mean_speed"),
        analysis.dimensionless_jerk(signed, fs) / analysis.speed_metric(signed) ** 2,
        rtol=1e-12,
    )


def test_signed_scalar_velocity_is_a_supported_input():
    """Hogan & Sternad define these on x(t), so ẋ is signed by construction.

    The pinned golden fixture's fourth signal genuinely goes negative, so
    rejecting sign here would invalidate the reference test.
    """
    fs = 100.0
    t = np.linspace(0.0, 1.0, 200)
    minjerk = (t**2) * (1.0 - t) ** 2
    signed = minjerk / minjerk.max() + 0.10 * np.sin(8 * np.pi * t)
    assert (signed < 0).any()
    for metric in _ALL_SMOOTHNESS_METRICS:
        out = analysis.smoothness(signed, fs, metric=metric)
        assert np.isfinite(out), metric


def test_number_of_peaks_min_height_filters_small_maxima():
    # three maxima at heights 1.0, 0.2, 1.0
    profile = np.array([0.0, 1.0, 0.0, 0.2, 0.0, 1.0, 0.0])
    assert analysis.number_of_peaks(profile) == 3
    assert analysis.number_of_peaks(profile, min_height=0.5) == 2
    assert analysis.number_of_peaks(profile, min_height=2.0) == 0
    # default is exactly the unthresholded count, including for signed input
    signed = np.array([-3.0, -1.0, -3.0])
    assert analysis.number_of_peaks(signed) == 1
    assert analysis.number_of_peaks(signed, min_height=0.0) == 0


def test_number_of_peaks_flat_maximum_counts_zero():
    """Documented consequence of strict comparison — a tie is not a peak."""
    assert analysis.number_of_peaks([0.0, 1.0, 1.0, 0.0]) == 0
    assert analysis.number_of_peaks([0.0, 1.0, 0.9, 1.0, 0.0]) == 2


def test_number_of_peaks_min_height_batches_per_column():
    speeds = np.array([[0.0, 0.0], [1.0, 0.2], [0.0, 0.0]])
    np.testing.assert_array_equal(analysis.number_of_peaks(speeds, min_height=0.5), [1, 0])


def test_smoothness_degenerate_inputs_are_graceful():
    # a perfectly still joint (zero speed) -> degenerate spectrum -> nan, no crash
    assert np.isnan(analysis.sparc(np.zeros(64), 100.0))
    # zero speed -> zero peak -> DLJ normalization undefined -> nan, no warning
    assert np.isnan(analysis.dimensionless_jerk(np.zeros(50), 100.0))
    # constant speed -> zero jerk -> perfectly smooth -> +inf (no leaked warning)
    assert analysis.log_dimensionless_jerk(np.full(50, 2.0), 100.0) == np.inf


def test_dlj_mean_speed_variant_is_peak_variant_over_speed_metric_squared():
    # the two genuinely distinct normalizers differ by exactly
    # (v_mean / v_peak)**2, which is what speed_metric returns
    rng = np.random.default_rng(11)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    np.testing.assert_allclose(
        analysis.dimensionless_jerk(s, fs, normalize="mean_speed"),
        analysis.dimensionless_jerk(s, fs) / analysis.speed_metric(s) ** 2,
        rtol=1e-12,
    )


def test_dlj_mean_speed_equals_amplitude_at_the_implied_arc_length():
    # "mean_speed" IS "amplitude" evaluated at A = ∫|v| dt, so passing that
    # arc length explicitly must reproduce it to float precision
    rng = np.random.default_rng(12)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    dt = 1.0 / fs
    arc_length = s.sum() * dt
    np.testing.assert_allclose(
        analysis.dimensionless_jerk(s, fs, normalize="amplitude", amplitude=arc_length),
        analysis.dimensionless_jerk(s, fs, normalize="mean_speed"),
        rtol=1e-12,
    )


def test_dlj_default_normalizer_is_unchanged_and_variants_share_the_sign():
    rng = np.random.default_rng(13)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    duration = s.shape[0] / fs
    peak_form = -(duration**3 / s.max() ** 2) * analysis.integrated_squared_jerk(s, fs)
    np.testing.assert_allclose(analysis.dimensionless_jerk(s, fs), peak_form, rtol=1e-12)
    for normalize, amplitude in (("peak_speed", None), ("mean_speed", None), ("amplitude", 2.5)):
        assert analysis.dimensionless_jerk(s, fs, normalize=normalize, amplitude=amplitude) < 0


def test_dlj_amplitude_batches_per_column():
    rng = np.random.default_rng(14)
    speeds = np.abs(rng.normal(size=(64, 3))) + 0.1
    fs = 100.0
    extents = np.array([1.0, 2.0, 4.0])
    batched = analysis.dimensionless_jerk(speeds, fs, normalize="amplitude", amplitude=extents)
    looped = np.array(
        [
            analysis.dimensionless_jerk(
                speeds[:, k], fs, normalize="amplitude", amplitude=extents[k]
            )
            for k in range(speeds.shape[1])
        ]
    )
    assert batched.shape == (3,)
    np.testing.assert_allclose(batched, looped, rtol=1e-12)


def test_dlj_rejects_mismatched_normalize_and_amplitude():
    s = np.abs(np.random.default_rng(15).normal(size=32)) + 0.1
    fs = 100.0
    with pytest.raises(ValueError, match="Unknown normalize"):
        analysis.dimensionless_jerk(s, fs, normalize="path_length")
    with pytest.raises(ValueError, match="needs the movement extent"):
        analysis.dimensionless_jerk(s, fs, normalize="amplitude")
    # a silently-ignored amplitude= is exactly how a convention mismatch
    # becomes an unnoticed wrong number
    with pytest.raises(ValueError, match="applies only to"):
        analysis.dimensionless_jerk(s, fs, normalize="peak_speed", amplitude=1.0)
    with pytest.raises(ValueError, match="applies only to"):
        analysis.log_dimensionless_jerk(s, fs, normalize="mean_speed", amplitude=1.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        analysis.dimensionless_jerk(s, fs, normalize="amplitude", amplitude=-1.0)
    with pytest.raises(ValueError, match="must be a scalar"):
        analysis.dimensionless_jerk(s, fs, normalize="amplitude", amplitude=np.ones(3))


def test_dlj_variants_degenerate_only_on_a_genuinely_zero_extent():
    fs = 100.0
    dt = 1.0 / fs
    # all three normalizers go nan on the same input the peak form always
    # has: a profile with no movement in it
    assert np.isnan(analysis.dimensionless_jerk(np.zeros(50), fs))
    assert np.isnan(analysis.dimensionless_jerk(np.zeros(50), fs, normalize="mean_speed"))
    assert np.isnan(
        analysis.dimensionless_jerk(np.ones(50), fs, normalize="amplitude", amplitude=0.0)
    )
    # the mean normalizer takes |v|, so a sign-flipping profile — whose raw
    # mean is 0 but which covers real distance — stays well defined, and
    # still matches the arc length it implies
    signed = np.concatenate([np.ones(25), -np.ones(25)])
    assert signed.mean() == 0.0
    dlj_mean = analysis.dimensionless_jerk(signed, fs, normalize="mean_speed")
    assert np.isfinite(dlj_mean)
    np.testing.assert_allclose(
        dlj_mean,
        analysis.dimensionless_jerk(
            signed, fs, normalize="amplitude", amplitude=np.abs(signed).sum() * dt
        ),
        rtol=1e-12,
    )


def test_dlj_normalize_reaches_the_kernel_through_both_wrappers():
    rng = np.random.default_rng(16)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    np.testing.assert_allclose(
        analysis.smoothness(s, fs, metric="dimensionless_jerk", normalize="mean_speed"),
        analysis.dimensionless_jerk(s, fs, normalize="mean_speed"),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        analysis.log_dimensionless_jerk(s, fs, normalize="mean_speed"),
        -np.log(np.abs(analysis.dimensionless_jerk(s, fs, normalize="mean_speed"))),
        rtol=1e-12,
    )


def test_smoothness_dispatcher_matches_kernels_and_rejects_unknown():
    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(size=64)) + 0.1
    fs = 100.0
    np.testing.assert_allclose(analysis.smoothness(s, fs, metric="sparc"), analysis.sparc(s, fs))
    np.testing.assert_allclose(
        analysis.smoothness(s, fs, metric="number_of_peaks"), analysis.number_of_peaks(s)
    )
    with pytest.raises(ValueError):
        analysis.smoothness(s, fs, metric="not_a_metric")


# ----------------------------------------------------------------
#  Trailing-axis (batched) reduction kernels
# ----------------------------------------------------------------

_ALL_SMOOTHNESS_METRICS = sorted(
    {**analysis._SMOOTHNESS_FS_METRICS, **analysis._SMOOTHNESS_PLAIN_METRICS}
)


def _metrics_named_in(docstring):
    """The metric names a `... one of ``"a"``, ``"b"``.` sentence lists."""
    lowered = docstring.lower()
    start = lowered.index("one of")
    sentence = docstring[start : docstring.index(".", start)]
    return set(re.findall(r'``"([a-z_0-9]+)"``', sentence))


@pytest.mark.parametrize(
    "documented",
    [
        pytest.param(analysis.smoothness.__doc__, id="analysis.smoothness"),
        pytest.param(Bvh.smoothness.__doc__, id="Bvh.smoothness"),
    ],
)
def test_smoothness_docstrings_list_exactly_the_registry(documented):
    """The metric names live in four places; this keeps three in step with one.

    Same shape as tests/test_docs_api_coverage.py: two-way, so a metric
    added without a docs update fails, and so does a stale name left
    behind after a removal.
    """
    assert _metrics_named_in(documented) == set(_ALL_SMOOTHNESS_METRICS)


@pytest.mark.parametrize(
    "documented",
    [
        pytest.param(analysis.smoothness.__doc__, id="analysis.smoothness"),
        pytest.param(Bvh.smoothness.__doc__, id="Bvh.smoothness"),
    ],
)
def test_smoothness_docstrings_name_every_metric_option(documented):
    """Every kernel option must be named at both dispatcher docstrings.

    One-way on purpose (an option name can legitimately appear in prose
    even after a rename elsewhere), but exhaustive: a kwarg added to any
    registered kernel without a dispatcher-docs update fails here. This
    is the guard that was missing when ``number_of_peaks`` gained
    ``min_height`` and both dispatcher docstrings kept claiming the
    non-DLJ metrics "take none".
    """
    registry = {**analysis._SMOOTHNESS_FS_METRICS, **analysis._SMOOTHNESS_PLAIN_METRICS}
    for metric, kernel in registry.items():
        options = [
            name for name in inspect.signature(kernel).parameters if name not in ("speed", "fs")
        ]
        for option in options:
            assert f"``{option}``" in documented, (
                f"{metric}'s option {option!r} is not named in this dispatcher docstring"
            )


def test_smoothness_dispatcher_accepts_every_registered_metric():
    """The registry is the source of truth, so nothing in it may be unreachable."""
    rng = np.random.default_rng(31)
    s = np.abs(rng.normal(size=64)) + 0.1
    for metric in _ALL_SMOOTHNESS_METRICS:
        assert np.isfinite(analysis.smoothness(s, 100.0, metric=metric)), metric


@pytest.mark.parametrize("metric", _ALL_SMOOTHNESS_METRICS)
def test_smoothness_kernels_batch_equals_python_loop(metric):
    rng = np.random.default_rng(3)
    speeds = np.abs(rng.normal(size=(64, 5))) + 0.1
    fs = 100.0
    batched = analysis.smoothness(speeds, fs, metric=metric)
    looped = np.array(
        [analysis.smoothness(speeds[:, k], fs, metric=metric) for k in range(speeds.shape[1])]
    )
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
        for field_batched, field_single in zip(batched, single, strict=True):
            assert field_batched.shape == (5,)
            np.testing.assert_allclose(field_batched[k], field_single, rtol=1e-12)


def test_active_duration_batch_equals_python_loop():
    rng = np.random.default_rng(5)
    speeds = np.abs(rng.normal(size=(40, 5)))
    batched = analysis.active_duration(speeds, threshold=0.5, fs=2.0)
    looped = np.array(
        [analysis.active_duration(speeds[:, k], 0.5, 2.0) for k in range(speeds.shape[1])]
    )
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
    assert np.isnan(vr.peak_to_mean[0])  # zero-mean column -> nan
    np.testing.assert_allclose(vr.peak_to_mean[1], 2.0)  # 1.0 / 0.5
    assert vr.peak_deceleration[1] == 0.0  # never falls -> clamped at 0
    assert vr.peak_acceleration[0] == 0.0  # never rises -> clamped at 0


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
    speed = np.linalg.norm(analysis.node_velocities(bvh, stencil="forward", pad="none"), axis=-1)
    sd = analysis.node_speed_derivative(bvh, stencil="forward", pad="none")
    vr = analysis.velocity_reductions(speed, fs)
    assert (sd > 0).any() and (sd < 0).any()  # non-degenerate motion
    np.testing.assert_allclose(vr.peak_acceleration, np.maximum(sd.max(axis=0), 0.0), rtol=1e-12)
    np.testing.assert_allclose(vr.peak_deceleration, np.maximum(-sd.min(axis=0), 0.0), rtol=1e-12)


def test_zero_crossings_count_and_axis():
    np.testing.assert_array_equal(analysis.zero_crossings([1, -1, 1, -1]), 3)
    np.testing.assert_array_equal(analysis.zero_crossings([1, 0, -1]), 0)  # exact 0 not a crossing
    sig = np.array([[1, 1], [-1, 1], [1, 1]])  # col0 crosses twice, col1 never
    np.testing.assert_array_equal(analysis.zero_crossings(sig, axis=0), [2, 0])


def test_active_segments_and_duration():
    speed = np.array([0.0, 2, 3, 0, 5])
    np.testing.assert_array_equal(
        analysis.active_segments(speed, threshold=1.0), [False, True, True, False, True]
    )
    # 3 active samples at 2 Hz -> 1.5 s
    np.testing.assert_allclose(analysis.active_duration(speed, threshold=1.0, fs=2.0), 1.5)


# ----------------------------------------------------------------
#  Kinetic energy
# ----------------------------------------------------------------


def test_kinetic_energy_unit_mass_matches_velocity_sum():
    bvh = make_pos_y_up_bvh()
    vel = analysis.joint_velocities(bvh)  # (F, J, 3)
    expected = np.sum(vel**2, axis=(1, 2))  # Σ‖v‖² per frame
    np.testing.assert_allclose(analysis.kinetic_energy(bvh), expected)


def test_kinetic_energy_with_masses():
    bvh = make_pos_y_up_bvh()
    vel = analysis.joint_velocities(bvh)
    masses = np.arange(1, vel.shape[1] + 1, dtype=float)
    expected = 0.5 * np.sum(masses * np.sum(vel**2, axis=-1), axis=-1)
    np.testing.assert_allclose(analysis.kinetic_energy(bvh, masses=masses), expected)


def test_kinetic_energy_masses_dict_matches_ordered_array():
    bvh = make_pos_y_up_bvh()
    names = bvh.joint_names
    masses_arr = np.arange(1, len(names) + 1, dtype=float)
    masses_dict = {n: float(m) for n, m in zip(names, masses_arr, strict=True)}
    # name-keyed dict is order-independent: shuffle it, result is unchanged
    shuffled = dict(reversed(list(masses_dict.items())))
    np.testing.assert_allclose(
        analysis.kinetic_energy(bvh, masses=shuffled),
        analysis.kinetic_energy(bvh, masses=masses_arr),
    )


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
    feet = ["LeftFoot", "RightFoot"]  # end-site feet, passed explicitly
    contacts = np.zeros((bvh.frame_count, 2))
    contacts[2:4, 0] = 1  # foot 0: onset at frame 2
    contacts[6:8, 0] = 1  # foot 0: onset at frame 6
    contacts[4:6, 1] = 1  # foot 1: onset at frame 4

    duration = (bvh.frame_count - 1) * bvh.frame_time
    np.testing.assert_allclose(
        analysis.cadence(bvh, foot_joints=feet, contacts=contacts), 3 / duration
    )  # 3 onsets
    # foot-measured stride_length is covered by the _compute_gait_parameters tests


def test_stride_length_nan_without_contacts():
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]  # end-site feet, passed explicitly
    no_contacts = np.zeros((bvh.frame_count, 2))
    assert np.isnan(analysis.stride_length(bvh, foot_joints=feet, contacts=no_contacts))
    # zero onsets over a *positive* duration is a measurement: rate 0.0
    assert analysis.cadence(bvh, foot_joints=feet, contacts=no_contacts) == 0.0


def test_zero_duration_rates_are_nan_not_zero():
    """A zero-duration clip leaves cadence/walking_pace *undefined* — nan,
    matching the other six GaitParameters fields ("not stepping" would be
    a measurement, and 0.0 claims it)."""
    one = make_pos_y_up_bvh()[0:1]  # single frame: no duration
    assert np.isnan(analysis.walking_pace(one))
    feet = ["LeftFoot", "RightFoot"]
    g = analysis.gait_parameters(one, foot_joints=feet, contacts=np.zeros((1, 2)))
    assert np.isnan(g.cadence)
    assert np.isnan(g.walking_pace)


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
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 11.0, [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(g.stride_length, 2.0)  # 0→2→4, 1→3→5
    np.testing.assert_allclose(g.stride_cv, 0.0)  # all strides equal
    np.testing.assert_allclose(g.step_length, 1.0)  # landings 0,1,2,3,4,5
    np.testing.assert_allclose(g.stance_fraction, 0.5)  # 2-frame stance, 4 cycle
    np.testing.assert_allclose(g.double_support_fraction, 0.0)  # feet never overlap
    np.testing.assert_allclose(g.asymmetry, 0.0)  # L and R equal
    np.testing.assert_allclose(g.cadence, 6 / 12)  # 6 onsets / 12 s
    np.testing.assert_allclose(g.walking_pace, 11.0 / 12)


def test_gait_core_asymmetry_and_cv():
    contacts, foot_h = _two_foot_walk()
    foot_h[[3, 7, 11], 1, 0] = [0.0, 1.0, 2.0]  # Right strides 1,1 (Left stays 2,2)
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0]
    )
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
        contacts[s:e, 0] = 1  # onsets 1,5,9
    foot_h = np.zeros((F, 1, 3))
    foot_h[[1, 5, 9], 0, 0] = [0.0, 2.0, 6.0]  # strides 2, 4 (mean 3)
    g = analysis._compute_gait_parameters(contacts, foot_h, ["LeftFoot"], 1.0, 0.0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.stride_length, 3.0)
    np.testing.assert_allclose(g.stride_cv, np.std([2.0, 4.0]) / 3.0)


def test_gait_core_stride_cv_nan_with_single_stride_per_foot():
    # two feet, one stride each -> stride_length defined, but no within-foot
    # variability is measurable, so stride_cv is nan (not a misleading 0)
    F = 9
    contacts = np.zeros((F, 2))
    for s, e in [(1, 3), (5, 7)]:
        contacts[s:e, 0] = 1  # Left onsets 1,5  -> 1 stride
    for s, e in [(3, 5), (7, 9)]:
        contacts[s:e, 1] = 1  # Right onsets 3,7 -> 1 stride
    foot_h = np.zeros((F, 2, 3))
    foot_h[[1, 5], 0, 0] = [0.0, 2.0]
    foot_h[[3, 7], 1, 0] = [1.0, 3.0]
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0]
    )
    assert not np.isnan(g.stride_length)  # 2 strides pooled
    assert np.isnan(g.stride_cv)  # no foot has >= 2 strides


def test_gait_core_double_support():
    contacts = np.zeros((4, 2))
    contacts[0:3, 0] = 1  # foot 0 planted 0,1,2
    contacts[1:4, 1] = 1  # foot 1 planted 1,2,3  -> overlap 1,2
    g = analysis._compute_gait_parameters(
        contacts, np.zeros((4, 2, 3)), ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(g.double_support_fraction, 2 / 4)


def test_gait_core_nan_edges():
    contacts = np.zeros((6, 1))
    contacts[1:3, 0] = 1  # a single foot, one onset only
    g = analysis._compute_gait_parameters(
        contacts, np.zeros((6, 1, 3)), ["LeftFoot"], 1.0, 0.0, [1.0, 0.0, 0.0]
    )
    assert np.isnan(g.stride_length)  # <2 contacts
    assert np.isnan(g.asymmetry)  # no left/right pair
    assert np.isnan(g.double_support_fraction)  # only one foot


def test_gait_core_step_length_is_forward_only():
    # opposite feet sit far apart laterally (y) but advance little forward (x);
    # step_length must measure the forward advance, not the lateral step width.
    F = 8
    contacts = np.zeros((F, 2))
    contacts[1:3, 0] = 1  # left onsets 1,5
    contacts[5:7, 0] = 1
    contacts[3:5, 1] = 1  # right onset 3
    foot_h = np.zeros((F, 2, 3))
    foot_h[[1, 5], 0, :] = [[0.0, 5.0, 0.0], [2.0, 5.0, 0.0]]  # left forward 0->2, y=+5
    foot_h[3, 1, :] = [1.0, -5.0, 0.0]  # right x=1, y=-5
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [1.0, 0.0, 0.0]
    )
    # landings by frame L@1(x0), R@3(x1), L@5(x2): forward advances |1-0|,|2-1| = 1
    np.testing.assert_allclose(g.step_length, 1.0)  # Euclidean would be ~10 (y-gap)


def test_gait_core_step_length_nan_without_net_travel():
    contacts, foot_h = _two_foot_walk()
    g = analysis._compute_gait_parameters(
        contacts, foot_h, ["LeftFoot", "RightFoot"], 1.0, 0.0, [0.0, 0.0, 0.0]
    )
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
    v = rng.normal(size=(30, 4)) + 7.0  # constant offset must contribute nothing
    c = v - v.mean(axis=0)  # centered on the temporal mean
    np.testing.assert_allclose(analysis.lagged_covariance(v, 0), c.T @ c / 30)
    m1 = analysis.lagged_covariance(v, 1)
    np.testing.assert_allclose(m1, c[1:].T @ c[:-1] / 29)
    assert m1.shape == (4, 4)
    # lag 0 on centered input == population covariance == cov3dj convention
    np.testing.assert_allclose(analysis.lagged_covariance(v, 0), np.cov(v.T, bias=True))
    with pytest.raises(ValueError):
        analysis.lagged_covariance(v, 30)
    with pytest.raises(ValueError):
        analysis.lagged_covariance(v, -1)


# ----------------------------------------------------------------
#  foot_contacts detection (hysteresis / Otsu / confidence / timing
#  refinement / diagnostics), signals driven through coords=
# ----------------------------------------------------------------


def _height_contacts(height, **settings):
    """Contacts of ``make_contact_rig``'s feet when only their height signal decides.

    ``height`` is ``(F,)`` for LeftFoot or ``(F, k)`` for the first ``k``
    feet. The floor is pinned at 0, so the clearance is the height itself;
    the threshold is 1 and the duration filters are off unless ``settings``
    says otherwise.
    """
    height = np.asarray(height, dtype=np.float64)
    bvh = make_contact_rig(height.shape[0])
    k = 1 if height.ndim == 1 else height.shape[1]
    kwargs = {
        "method": "height",
        "floor": 0.0,
        "height_threshold": 1.0,
        "min_contact_duration": 0.0,
        "min_gap_duration": 0.0,
        **settings,
    }
    return bvh.foot_contacts(
        foot_joints=list(CONTACT_FEET[:k]), coords=contact_coords(bvh, height), **kwargs
    )


def test_zero_hysteresis_is_the_single_threshold():
    height = np.array([3.0, 0.5, 1.5, 0.2, 0.9, 1.1, 3.0, 3.0])
    got = _height_contacts(height, hysteresis=0.0)
    np.testing.assert_array_equal(got[:, 0], [0, 1, 0, 1, 1, 0, 0, 0])  # height < 1


def test_hysteresis_holds_a_contact_through_the_band_and_drops_a_shallow_dip():
    # threshold 1, band 0.5: a frame is weak below 1.5 and strong below 0.5. A
    # weak run is a contact only if it holds a strong frame.
    height = np.array([3.0, 0.2, 0.2, 1.2, 0.2, 3.0, 3.0, 0.8, 3.0, 3.0])
    got = _height_contacts(height, hysteresis=0.5)
    # Frame 3 rises above the threshold but stays in the band, inside a run
    # with strong frames: still a contact. Frame 7 dips below the threshold but
    # never below the band: not one.
    np.testing.assert_array_equal(got[:, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0, 0])


# The default height threshold on make_contact_rig: 0.013 x its skeleton scale, 10.
RIG_HEIGHT_MARGIN = 0.013 * 10.0


def _adaptive_height_contacts(height):
    return _height_contacts(height, height_threshold=None, adaptive=True, hysteresis=0.0)


def test_adaptive_splits_a_bimodal_signal_and_keeps_the_default_on_a_unimodal_one():
    # One call, two feet. LeftFoot stands at 1.5 margins and swings at 6: the
    # split between the two clusters labels its whole stance. RightFoot hovers
    # at 1.5 margins with no swing: no convincing split, so it keeps the default
    # threshold, and only its frames below one margin are contacts.
    rng = np.random.default_rng(0)
    m = RIG_HEIGHT_MARGIN
    height = np.empty((400, 2))
    height[:200, 0] = rng.normal(1.5 * m, 0.02 * m, 200)
    height[200:, 0] = rng.normal(6.0 * m, 0.02 * m, 200)
    height[:, 1] = rng.normal(1.5 * m, 0.2 * m, 400)
    got = _adaptive_height_contacts(height)
    np.testing.assert_array_equal(got[:, 0], np.arange(400) < 200)
    np.testing.assert_array_equal(got[:, 1], height[:, 1] < m)
    assert 0 < got[:, 1].sum() < 20  # the default does label a few frames


def test_adaptive_keeps_the_default_on_a_flat_signal():
    # A foot held still: one value, no split to find.
    flat = np.full(100, 1.5 * RIG_HEIGHT_MARGIN)
    bvh = make_contact_rig(flat.size)
    c, report = bvh.foot_contacts(
        foot_joints=["LeftFoot"],
        method="height",
        coords=contact_coords(bvh, flat),
        floor=0.0,
        adaptive=True,
        return_info=True,
    )
    assert not c.any()
    np.testing.assert_allclose(report.height_threshold, [RIG_HEIGHT_MARGIN])


def test_adaptive_split_is_clamped_to_four_times_the_default():
    # Stance spread from 3 to 5 margins, swing at 100: the split between the
    # clusters lies near 50 margins, and is clamped to 4. Only the stance frames
    # below 4 margins are contacts: a rejected split (1 margin) would label none,
    # an unclamped one all of the stance.
    m = RIG_HEIGHT_MARGIN
    rng = np.random.default_rng(1)
    height = np.concatenate(
        [np.linspace(3.0 * m, 5.0 * m, 200), rng.normal(100 * m, 0.02 * m, 200)]
    )
    got = _adaptive_height_contacts(height)
    np.testing.assert_array_equal(got[:, 0], height < 4.0 * m)
    assert 50 < got[:, 0].sum() < 150


def test_confidence_is_the_margin_below_the_thresholds_times_the_agreement():
    # One frame a second, so the speed is the step along x: 5, 5, 0.5, 0.5, 5, 5
    # (frame 0 takes frame 1's), against a threshold of 2. The foot stays at
    # height 0.5, below the height threshold of 1 throughout, so the contacts
    # are frames 2 and 3.
    bvh = make_contact_rig(6, frame_time=1.0)
    coords = contact_coords(bvh, np.full(6, 0.5), x=np.array([0.0, 5.0, 5.5, 6.0, 11.0, 16.0]))
    unfiltered = {"hysteresis": 0.0, "min_contact_duration": 0.0, "min_gap_duration": 0.0}
    c, report = bvh.foot_contacts(
        foot_joints=["LeftFoot"],
        coords=coords,
        vel_threshold=2.0,
        height_threshold=1.0,
        floor=0.0,
        return_info=True,
        **unfiltered,
    )
    np.testing.assert_array_equal(c[:, 0], [0, 0, 1, 1, 0, 0])
    # On those frames the speed sits 3/4 of its threshold below it and the
    # height 1/2 of its own: margin 5/8, their mean. The velocity mask agrees
    # with the height mask (true everywhere) on 2 frames of 6.
    np.testing.assert_allclose(report.confidence, [np.sqrt(5 / 8 * 2 / 6)])
    # The velocity method alone has no agreement term: its own margin.
    _, report = bvh.foot_contacts(
        foot_joints=["LeftFoot"],
        method="velocity",
        coords=coords,
        vel_threshold=2.0,
        return_info=True,
        **unfiltered,
    )
    np.testing.assert_allclose(report.confidence, [0.75])


def test_foot_contacts_detects_bouncing_root_dwell():
    # Feet dwell low (contact) on known frames and lift between -> the height
    # method must recover the dwells. Drive the foot trajectory via `coords`.
    bvh = make_pos_y_up_bvh()
    up = 1  # +y
    coords = bvh.node_positions().copy()
    for f in [bvh.index(n, space="node") for n in ["LeftFoot", "RightFoot"]]:
        coords[4:7, f, up] += 40.0  # lift frames 4-6; dwell low elsewhere
    c = bvh.foot_contacts(
        foot_joints=["LeftFoot", "RightFoot"], method="height", coords=coords, hysteresis=0.0
    )
    assert c[0:4].all() and c[7:10].all()  # contact during the low dwells
    assert not c[4:7].any()  # airborne during the lift


# --- boundary-aware hysteresis release (open runs trimmed to raw support) ---


@pytest.mark.parametrize("edge", ["end", "start"])
def test_hysteresis_releases_a_run_open_at_the_clip_edge_at_the_raw_lift(edge):
    # threshold 1, band 0.5. A contact run that rises into the band at the clip
    # edge never leaves the band, so the band alone would hold it to the edge;
    # it ends instead at the last frame below the threshold itself (frame 5).
    height = np.array([3.0, 3.0, 0.1, 0.1, 0.1, 0.8, 1.2, 1.4])
    expected = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    if edge == "start":
        height, expected = height[::-1], expected[::-1]
    got = _height_contacts(height, hysteresis=0.5)
    np.testing.assert_array_equal(got[:, 0], expected)


def test_hysteresis_keeps_open_runs_the_raw_threshold_supports():
    # Below the threshold to the last frame, or for the whole clip: nothing to release.
    height = np.full((8, 2), 0.1)
    height[:2, 0] = 3.0
    got = _height_contacts(height, hysteresis=0.5)
    np.testing.assert_array_equal(got[:, 0], [0, 0, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(got[:, 1], np.ones(8))


def test_foot_contacts_releases_foot_lifting_at_clip_end():
    # end-to-end: a foot planted then lifting in the final frames, clip cut
    # mid-lift. With default hysteresis the latch must not hold it to the edge.
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]
    fis = [bvh.index(n, space="node") for n in feet]
    coords = bvh.node_positions().copy()
    coords[:, fis, 1] = 0.0  # planted at the floor...
    coords[7:10, fis, 1] = 1.1  # ...then in the band (above raw thr=1, below high=1.25)
    c = np.asarray(
        bvh.foot_contacts(
            foot_joints=feet,
            method="height",
            coords=coords,
            floor=0.0,
            height_threshold=1.0,
            min_contact_duration=0.0,
            min_gap_duration=0.0,
        )
    )
    assert c[:7].all()  # genuine plant kept
    assert not c[7:].any()  # open-end lift released, not held to frame 9


def test_contact_diagnostics_skate_airborne_height():
    # LeftFoot is planted on frames 1-3 at height 0.4; RightFoot on frames 2-4
    # at height 0, sliding 3 units along x as it stands.
    height = np.array([[5, 5], [0.4, 5], [0.4, 0], [0.4, 0], [5, 0], [5, 5]], dtype=float)
    x = np.zeros((6, 2))
    x[3:, 1] = [1.5, 3.0, 3.0]
    bvh = make_contact_rig(6)
    c, report = bvh.foot_contacts(
        foot_joints=list(CONTACT_FEET),
        method="height",
        coords=contact_coords(bvh, height, x=x),
        floor=0.0,
        height_threshold=1.0,
        hysteresis=0.0,
        min_contact_duration=0.0,
        min_gap_duration=0.0,
        return_info=True,
    )
    np.testing.assert_array_equal(c, height < 1.0)
    # The skate is the drift over a contact run, over the skeleton scale (10).
    np.testing.assert_allclose(report.foot_skate_max, [0.0, 3.0 / 10.0])
    np.testing.assert_allclose(report.foot_skate_mean, [0.0, 3.0 / 10.0])
    np.testing.assert_allclose(report.airborne_fraction, 2 / 6)  # frames 0 and 5
    np.testing.assert_allclose(report.height_at_contact, [0.4, 0.0])


# ----------------------------------------------------------------
#  Stance calibration of the height threshold + floor_height
# ----------------------------------------------------------------


def _stance_and_swing(stance_level):
    """Height and x of a foot that stands for 20 frames, then swings for 20.

    At one frame a second, so that a step of a few height margins is still
    slow against the default speed threshold (1.2 u/s on the contact rig).
    The stance sits at ``stance_level`` above the floor, except two frames
    0.6 margins higher, two 1.4 margins higher and three lifted 3 margins
    (slow, but out of contact under either reading of the threshold). The
    swing is 10 margins up and travels 5 units a frame. Returns
    ``(height, x, in_stance)``, ``in_stance`` the frames within one margin
    of the stance level.
    """
    m = RIG_HEIGHT_MARGIN
    above = np.zeros(40)
    above[[5, 6]] = 0.6 * m
    above[[12, 13]] = 1.4 * m
    above[[16, 17, 18]] = 3.0 * m
    above[20:] = 10.0 * m
    x = np.where(np.arange(40) < 20, 0.0, 5.0 * (np.arange(40) - 19))
    return stance_level + above, x, above < m


def test_stance_calibration_sets_each_foot_threshold_one_margin_above_its_stance():
    # LeftFoot stands 5 margins above the floor (a retargeted foot that
    # hovers), RightFoot on it. Each foot's threshold is set one margin above
    # its own stance level, the median of its height over its slow frames.
    m = RIG_HEIGHT_MARGIN
    left, x, in_stance = _stance_and_swing(5.0 * m)
    right, _, _ = _stance_and_swing(0.0)
    bvh = make_contact_rig(40, frame_time=1.0)
    coords = contact_coords(bvh, np.column_stack([left, right]), x=np.column_stack([x, x]))
    kwargs = {"foot_joints": list(CONTACT_FEET), "coords": coords, "floor": 0.0, "hysteresis": 0.0}
    got = bvh.foot_contacts(height_reference="velocity", **kwargs)
    np.testing.assert_array_equal(got[:, 0], in_stance)
    np.testing.assert_array_equal(got[:, 1], in_stance)
    # Anchored to the floor instead, the hovering foot never reads as low enough.
    fixed = bvh.foot_contacts(height_reference="floor", **kwargs)
    assert not fixed[:, 0].any()
    np.testing.assert_array_equal(fixed[:, 1], in_stance)


def test_stance_calibration_falls_back_to_the_margin_without_a_slow_frame():
    # A foot that never slows down has no stance to calibrate against.
    m = RIG_HEIGHT_MARGIN
    height = np.where(np.arange(40) < 20, 5.0 * m, 15.0 * m)
    bvh = make_contact_rig(40, frame_time=1.0)
    _, report = bvh.foot_contacts(
        foot_joints=["LeftFoot"],
        coords=contact_coords(bvh, height, x=5.0 * np.arange(40)),
        floor=0.0,
        return_info=True,
    )
    np.testing.assert_allclose(report.height_threshold, [m])


def test_floor_height_cache_invalidates_on_mutation():
    bvh = make_pos_y_up_bvh()
    f0 = bvh.floor_height
    bvh.root_pos = bvh.root_pos + np.array([0.0, 7.0, 0.0])  # +y shift
    np.testing.assert_allclose(bvh.floor_height, f0 + 7.0)  # recomputed, not stale


def test_floor_height_is_over_all_nodes():
    bvh = make_pos_y_up_bvh()
    coords = bvh.node_positions()
    expected = float(np.percentile(coords[:, :, 1].min(axis=1), 2.0))  # all-nodes min, +y
    np.testing.assert_allclose(bvh.floor_height, expected)


def test_default_thresholds_scale_with_the_mean_rest_distance_to_the_joints():
    # Joints 10 and 20 from the root at rest: the skeleton scale is their mean,
    # 15, so the default speed threshold is 0.12 x 15 = 1.8 u/s and the default
    # height threshold 0.013 x 15 = 0.195.
    bvh = make_contact_rig(6, frame_time=1.0, foot_offsets=((-6.0, -8.0, 0.0), (0.0, -20.0, 0.0)))
    joints = list(CONTACT_FEET)
    unfiltered = {"hysteresis": 0.0, "min_contact_duration": 0.0, "min_gap_duration": 0.0}
    # At one frame a second the speed is the step: 1.7, 1.7, 1.9, 1.9, 1.7
    # (frame 0 takes frame 1's).
    x = np.cumsum([0.0, 1.7, 1.7, 1.9, 1.9, 1.7])
    xs = np.column_stack([x, x])
    c = bvh.ground_contacts(
        joints, method="velocity", coords=contact_coords(bvh, np.zeros((6, 2)), x=xs), **unfiltered
    )
    np.testing.assert_array_equal(c, np.column_stack([[1, 1, 1, 0, 0, 1]] * 2))
    h = np.array([0.18, 0.21, 0.18, 0.21, 0.18, 0.21])
    c = bvh.ground_contacts(
        joints,
        method="height",
        coords=contact_coords(bvh, np.column_stack([h, h])),
        floor=0.0,
        **unfiltered,
    )
    np.testing.assert_array_equal(c, np.column_stack([[1, 0, 1, 0, 1, 0]] * 2))


def _make_toed_skeleton(toe_tip_drop=2.0, hand_height=None, n_frames=10) -> Bvh:
    """+z-up rig with detectable toe joints and an end site below them.

    ``toe_tip_drop`` sinks each toe end site that far below its toe joint —
    the geometry that makes a foot-JOINT floor and a NODE floor differ.
    ``hand_height`` (when given) is the world height of a hand end site, for
    the case where a non-foot node is the lowest thing in the clip.
    """
    from pybvh.bvh import BvhEndSite, BvhJoint, BvhRoot

    hips = BvhRoot("Hips", offset=[0, 0, 0], rot_channels=["Z", "Y", "X"])
    nodes = [hips]
    children = []
    for side, lat in (("Left", -3.0), ("Right", 3.0)):
        leg = BvhJoint(f"{side}Leg", offset=[lat, 0, -5], rot_channels=["Z", "Y", "X"])
        toe = BvhJoint(f"{side}ToeBase", offset=[0, 0, -5], rot_channels=["Z", "Y", "X"])
        tip = BvhEndSite("EndSite", offset=[0, 0, -toe_tip_drop])
        tip.parent = toe
        toe.parent, toe.children = leg, [tip]
        leg.parent, leg.children = hips, [toe]
        children.append(leg)
        nodes += [leg, toe, tip]
    if hand_height is not None:
        # Root sits at z=10, so the arm offset places the hand tip exactly
        # at `hand_height` in world coordinates.
        arm = BvhJoint("LeftArm", offset=[-6, 0, 0], rot_channels=["Z", "Y", "X"])
        hand = BvhEndSite("EndSite", offset=[0, 0, hand_height - 10.0])
        hand.parent = arm
        arm.parent, arm.children = hips, [hand]
        children.append(arm)
        nodes += [arm, hand]
    hips.children = children

    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos = np.zeros((n_frames, 3))
    root_pos[:, 2] = 10.0
    bvh = Bvh(
        nodes=nodes,
        root_pos=root_pos,
        joint_angles=np.zeros((n_frames, n_joints, 3)),
        frame_time=1 / 30,
    )
    bvh.world_up = "+z"
    return bvh


def test_floor_height_follows_the_toe_tip_not_the_toe_joint():
    """The end site is the part that touches the ground; the toe JOINT sits
    inside the foot. Estimating from detected feet (joints only) reads high
    by exactly the tip drop."""
    bvh = _make_toed_skeleton(toe_tip_drop=2.0)
    assert bvh.auto_detect_foot_joints() == ["LeftToeBase", "RightToeBase"]
    coords = bvh.node_positions()
    toe_joint = coords[:, bvh.node_index["LeftToeBase"], 2].min()
    np.testing.assert_allclose(bvh.floor_height, toe_joint - 2.0)


def test_floor_height_follows_a_hand_when_the_hand_is_lowest():
    """The floor is not about the feet: whatever body part reaches lowest
    defines the ground."""
    toed = _make_toed_skeleton(toe_tip_drop=2.0)
    feet_level = toed.floor_height
    crawling = _make_toed_skeleton(toe_tip_drop=2.0, hand_height=feet_level - 3.0)
    np.testing.assert_allclose(crawling.floor_height, feet_level - 3.0)


def test_floor_height_ignores_one_glitched_frame():
    """2nd percentile, not the true minimum: a single spurious low frame
    must not drag the ground down with it.

    Needs a clip long enough for one frame to be a small share of the
    distribution — at 100 frames the 2nd percentile lands on the third
    lowest per-frame minimum, so one outlier is absorbed entirely.
    """
    bvh = _make_toed_skeleton(n_frames=100)
    clean = bvh.floor_height
    root_pos = bvh.root_pos.copy()
    root_pos[4, 2] -= 500.0  # the whole body drops 500 units for one frame
    bvh.root_pos = root_pos
    assert bvh.floor_height == pytest.approx(clean, abs=1e-9)
    # ... while the true minimum follows it all the way down.
    assert bvh.node_positions()[:, :, 2].min() < clean - 400.0


def test_floor_height_negative_up_raw_coords_and_copy():
    bvh = make_neg_y_up_bvh()
    assert bvh.floor_height == bvh.copy().floor_height  # survives copy
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
    scale = analysis.skeleton_size(bvh, feet)
    coords = bvh.node_positions().copy()
    F = bvh.frame_count
    half = F // 2
    for f in fidx:
        rest = coords[0, f, :].copy()
        stance = rest.copy()
        stance[1] = rest[1] + hover_frac * scale
        coords[:half, f, :] = stance  # fully still (slow)
        coords[half:, f, :] = stance
        coords[half:, f, 1] = rest[1] + swing_frac * scale  # swing high
        coords[half:, f, 0] = stance[0] + np.linspace(
            0.1 * scale, scale, F - half
        )  # moving -> fast
    base = float(coords[0, fidx[0], 1] - hover_frac * scale)  # the floor (rest height)
    return bvh, feet, coords, base


def test_height_reference_recovers_hovering_stance():
    # hysteresis=0 isolates the height-threshold logic from the band machinery.
    bvh, feet, coords, base = _hover_clip(hover_frac=0.05)
    # explicit low floor (below the hover) mimics the retargeting artifact
    cv = bvh.foot_contacts(
        foot_joints=feet,
        coords=coords,
        floor=float(base),
        height_reference="velocity",
        hysteresis=0.0,
    )
    cf = bvh.foot_contacts(
        foot_joints=feet, coords=coords, floor=float(base), height_reference="floor", hysteresis=0.0
    )
    assert cv.sum() > cf.sum()  # velocity recovers stance the floor-anchored threshold rejects


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
    scale = analysis.skeleton_size(bvh, feet)
    coords = bvh.node_positions().copy()
    F = bvh.frame_count
    base = coords[0, fidx[0], 1]
    half = F // 2
    coords[:, fidx[1], :] = coords[0, fidx[1], :]  # foot1 held still...
    coords[:, fidx[1], 1] = base + 0.3 * scale  # ...and high (airborne)
    coords[:half, fidx[0], :] = coords[0, fidx[0], :]
    coords[:half, fidx[0], 1] = base  # foot0 stance on floor
    coords[half:, fidx[0], 1] = base + 0.3 * scale
    coords[half:, fidx[0], 0] = np.linspace(0, scale, F - half)
    cv = bvh.foot_contacts(
        foot_joints=feet, coords=coords, height_reference="velocity", hysteresis=0.0
    )
    assert cv[:, 1].sum() == 0  # held-airborne foot rejected
