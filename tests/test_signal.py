"""Tests for the pybvh.signal utilities + the analysis scale functions."""

import numpy as np
import pytest
from synthetic_bvh import make_pos_y_up_bvh

from pybvh import analysis, signal

# ----------------------------------------------------------------
#  finite_difference at higher orders
# ----------------------------------------------------------------


def test_finite_difference_forward_edge_tail_repeats_last_valid_value():
    # t**3 at dt=1: first differences 1, 7, 19, 37, 61; second 6, 12, 18, 24.
    cube = np.arange(6.0) ** 3
    second = signal.finite_difference(cube, 1.0, order=2, stencil="forward", pad="edge")
    # Padded once, at the end: the last valid value fills the last two
    # samples, where padding after each pass would leave 0 there.
    np.testing.assert_array_equal(second, [6.0, 12.0, 18.0, 24.0, 24.0, 24.0])


@pytest.mark.parametrize("order", [1, 2, 3])
def test_finite_difference_orders_match_their_definitions(order):
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(15, 4))
    dt = 0.25

    repeated_gradient = arr
    for _ in range(order):
        repeated_gradient = np.gradient(repeated_gradient, dt, axis=0)
    np.testing.assert_allclose(
        signal.finite_difference(arr, dt, order=order, stencil="central", pad="edge"),
        repeated_gradient,
    )
    np.testing.assert_allclose(
        signal.finite_difference(arr, dt, order=order, stencil="central", pad="none"),
        repeated_gradient[order:-order],
    )

    forward = np.diff(arr, n=order, axis=0) / dt**order
    np.testing.assert_allclose(
        signal.finite_difference(arr, dt, order=order, stencil="forward", pad="none"),
        forward,
    )
    forward_edge = signal.finite_difference(arr, dt, order=order, stencil="forward", pad="edge")
    assert forward_edge.shape == arr.shape
    np.testing.assert_allclose(forward_edge[: len(forward)], forward)
    np.testing.assert_allclose(forward_edge[len(forward) :], np.repeat(forward[-1:], order, axis=0))


@pytest.mark.parametrize("stencil", ["central", "forward"])
@pytest.mark.parametrize("pad", ["edge", "none"])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("axis", [1, -1])
def test_finite_difference_higher_orders_follow_axis(stencil, pad, order, axis):
    rng = np.random.default_rng(3)
    along_first = rng.normal(size=(12, 2, 3))
    moved = np.moveaxis(along_first, 0, axis)
    expected = signal.finite_difference(along_first, 0.1, order=order, stencil=stencil, pad=pad)
    out = signal.finite_difference(moved, 0.1, order=order, stencil=stencil, pad=pad, axis=axis)
    np.testing.assert_array_equal(out, np.moveaxis(expected, 0, axis))


def test_finite_difference_order_must_be_a_positive_int():
    arr = np.zeros((10, 3))
    with pytest.raises(ValueError, match="order must be at least 1, got 0"):
        signal.finite_difference(arr, 0.1, order=0)
    with pytest.raises(TypeError, match="order must be an int, got bool True"):
        signal.finite_difference(arr, 0.1, order=True)
    with pytest.raises(TypeError, match="order must be an int, got float 2.0"):
        signal.finite_difference(arr, 0.1, order=2.0)
    np.testing.assert_array_equal(
        signal.finite_difference(arr, 0.1, order=np.int64(2)),
        signal.finite_difference(arr, 0.1, order=2),
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64])
def test_finite_difference_accepts_unsigned_numpy_orders(dtype):
    # t**2 at dt=1: central first derivative 1, 2, 4, 6, 7; second 1, 1.5, 2, 1.5, 1.
    square = np.arange(5.0) ** 2
    out = signal.finite_difference(square, 1.0, order=dtype(2), pad="none")
    np.testing.assert_array_equal(out, [2.0])


@pytest.mark.parametrize(
    "stencil, pad, order, min_samples, shape_at_min",
    [
        ("central", "edge", 1, 2, (2,)),
        ("central", "none", 1, 2, (0,)),
        ("central", "edge", 3, 2, (2,)),
        ("central", "none", 3, 2, (0,)),
        ("forward", "edge", 1, 2, (2,)),
        ("forward", "edge", 3, 4, (4,)),
    ],
)
def test_finite_difference_minimum_samples(stencil, pad, order, min_samples, shape_at_min):
    arr = np.arange(min_samples, dtype=float) ** 2
    out = signal.finite_difference(arr, 1.0, order=order, stencil=stencil, pad=pad)
    assert out.shape == shape_at_min
    with pytest.raises(
        ValueError,
        match=f"requires at least {min_samples} samples along axis 0 \\(have {min_samples - 1}\\)",
    ):
        signal.finite_difference(arr[:-1], 1.0, order=order, stencil=stencil, pad=pad)


@pytest.mark.parametrize("order", [1, 3])
def test_finite_difference_forward_none_too_short_is_empty(order):
    pair = np.array([1.0, 4.0])
    for arr in (pair, pair[:1]):
        out = signal.finite_difference(arr, 1.0, order=order, stencil="forward", pad="none")
        assert out.shape == (max(len(arr) - order, 0),)


def test_finite_difference_short_inputs_keep_their_order_one_results():
    pair = np.array([1.0, 4.0])
    # Two samples under the central stencil: both ends one-sided.
    np.testing.assert_array_equal(signal.finite_difference(pair, 0.5), [6.0, 6.0])
    assert signal.finite_difference(pair, 0.5, pad="none").shape == (0,)
    single = pair[:1]
    assert signal.finite_difference(single, 0.5, stencil="forward", pad="none").shape == (0,)


# ----------------------------------------------------------------
#  temporal_stats
# ----------------------------------------------------------------


def test_temporal_stats_basic_moments():
    sig = np.array([1.0, 2, 3, 4, 5])
    s = signal.temporal_stats(sig)
    np.testing.assert_allclose(s.mean, 3.0)
    np.testing.assert_allclose(s.std, np.std(sig))
    assert s.min == 1.0 and s.max == 5.0
    np.testing.assert_allclose(s.skewness, 0.0, atol=1e-12)  # symmetric


def test_temporal_stats_constant_signal_is_nan_skew_kurt():
    s = signal.temporal_stats(np.full(10, 7.0))
    assert np.isnan(s.skewness) and np.isnan(s.kurtosis)


def test_temporal_stats_axis_and_kurtosis_sign():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(5000, 2))
    s = signal.temporal_stats(sig, axis=0)
    assert s.mean.shape == (2,)
    np.testing.assert_allclose(s.kurtosis, 0.0, atol=0.2)  # gaussian excess ~0


# ----------------------------------------------------------------
#  box_filter_smooth
# ----------------------------------------------------------------


def _ref_smooth(x, w):
    p = np.pad(x, ((w - 1) // 2, w // 2), mode="edge")
    return np.convolve(p, np.ones(w) / w, "valid")


def test_box_filter_matches_edge_padded_convolution():
    x = np.array([1.0, 2, 3, 4, 5, 4, 3, 2, 1])
    for w in (2, 3, 4, 5):
        out = signal.box_filter_smooth(x, w)
        assert out.shape == x.shape
        np.testing.assert_allclose(out, _ref_smooth(x, w))


def test_box_filter_constant_and_noop_and_validation():
    np.testing.assert_allclose(signal.box_filter_smooth(np.full(8, 3.0), 4), 3.0)
    x = np.arange(6.0)
    np.testing.assert_allclose(signal.box_filter_smooth(x, 1), x)
    with pytest.raises(ValueError):
        signal.box_filter_smooth(x, 0)


def test_box_filter_vectorizes_over_axis():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(20, 3))
    out = signal.box_filter_smooth(sig, 5, axis=0)
    for c in range(3):
        np.testing.assert_allclose(out[:, c], signal.box_filter_smooth(sig[:, c], 5))


# ----------------------------------------------------------------
#  fft_magnitude / dominant_frequency
# ----------------------------------------------------------------


def test_dominant_frequency_pure_sinusoid():
    fs, f0, T = 100.0, 5.0, 200
    t = np.arange(T) / fs
    sig = 3.0 + np.sin(2 * np.pi * f0 * t)  # offset must not win (DC excluded)
    np.testing.assert_allclose(signal.dominant_frequency(sig, fs), f0)


def test_fft_magnitude_shapes_and_peak():
    fs, T = 64.0, 128  # resolution fs/T = 0.5 Hz -> 8 Hz is an exact bin
    t = np.arange(T) / fs
    sig = np.cos(2 * np.pi * 8.0 * t)
    freqs, mag = signal.fft_magnitude(sig, fs)
    assert freqs.shape == (T // 2 + 1,) and mag.shape == (T // 2 + 1,)
    np.testing.assert_allclose(freqs[np.argmax(mag)], 8.0)


def test_fft_magnitude_norm_amplitude_reads_physical_amplitude():
    """norm="amplitude": an exact-bin sine of amplitude A peaks at A, and
    the un-doubled DC / Nyquist bins read their own value."""
    fs, T, A = 64.0, 128, 3.0
    t = np.arange(T) / fs
    sig = A * np.cos(2 * np.pi * 8.0 * t)
    _, mag = signal.fft_magnitude(sig, fs, norm="amplitude")
    np.testing.assert_allclose(mag[np.argmax(mag)], A, atol=1e-12)

    _, mag_dc = signal.fft_magnitude(np.full(T, 2.5), fs, norm="amplitude")
    np.testing.assert_allclose(mag_dc[0], 2.5, atol=1e-12)

    nyquist_tone = A * np.cos(np.pi * np.arange(T))  # +A, -A, +A, ...
    _, mag_nyq = signal.fft_magnitude(nyquist_tone, fs, norm="amplitude")
    np.testing.assert_allclose(mag_nyq[-1], A, atol=1e-12)


def test_fft_magnitude_norm_scalings_match_numpy_vocabulary():
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(100)
    _, raw = signal.fft_magnitude(sig)  # "backward"
    _, ortho = signal.fft_magnitude(sig, norm="ortho")
    _, forward = signal.fft_magnitude(sig, norm="forward")
    np.testing.assert_allclose(ortho, raw / np.sqrt(100))
    np.testing.assert_allclose(forward, raw / 100)
    with pytest.raises(ValueError, match="norm"):
        signal.fft_magnitude(sig, norm="nope")


def test_fft_magnitude_amplitude_multidim_axis():
    """The DC/Nyquist un-doubling must land on the transform axis for
    multi-dimensional input."""
    fs, T = 64.0, 128
    t = np.arange(T) / fs
    rows = np.stack(
        [np.cos(2 * np.pi * 8.0 * t), 2.0 * np.cos(2 * np.pi * 4.0 * t) + 1.0]
    )  # (2, T)
    _, mag = signal.fft_magnitude(rows, fs, axis=1, norm="amplitude")
    for i in range(2):
        _, expected = signal.fft_magnitude(rows[i], fs, norm="amplitude")
        np.testing.assert_allclose(mag[i], expected)


# ----------------------------------------------------------------
#  ramer_douglas_peucker
# ----------------------------------------------------------------


def test_rdp_collinear_reduces_to_endpoints():
    line = np.stack([np.linspace(0, 10, 11), np.zeros(11)], axis=1)
    out = signal.ramer_douglas_peucker(line, eps=1e-6)
    np.testing.assert_allclose(out, [[0, 0], [10, 0]])


def test_rdp_keeps_significant_corner():
    curve = np.array([[0.0, 0], [1, 0.01], [2, 0], [2, 1], [2, 2]])
    out = signal.ramer_douglas_peucker(curve, eps=0.1)
    # the (2,0) corner is kept; the near-collinear (1,0.01) is dropped
    assert any(np.allclose(p, [2, 0]) for p in out)
    assert not any(np.allclose(p, [1, 0.01]) for p in out)


def test_rdp_no_recursion_overflow_on_zigzag():
    # a pathological alternating curve drives RDP to maximum split depth;
    # the explicit-stack implementation must not raise RecursionError
    n = 3000
    zig = np.stack([np.arange(n, dtype=float), (np.arange(n) % 2) * 1.0], axis=1)
    out = signal.ramer_douglas_peucker(zig, 0.1)
    assert out.shape[1] == 2 and len(out) >= 2


def test_rdp_3d_and_large_eps():
    rng = np.random.default_rng(2)
    curve = rng.normal(size=(30, 3))
    out = signal.ramer_douglas_peucker(curve, eps=1e9)  # everything within eps
    np.testing.assert_allclose(out, curve[[0, -1]])


# ----------------------------------------------------------------
#  Scale functions
# ----------------------------------------------------------------


def test_skeleton_size_positive_and_scale_linear():
    bvh = make_pos_y_up_bvh()
    feet = ["LeftFoot", "RightFoot"]
    size = analysis.skeleton_size(bvh, foot_joints=feet)
    assert size > 0
    bigger = bvh.scale(2.0)
    np.testing.assert_allclose(analysis.skeleton_size(bigger, foot_joints=feet), 2.0 * size)


def test_skeleton_size_unknown_explicit_joint_raises():
    bvh = make_pos_y_up_bvh()
    with pytest.raises(ValueError, match="not found"):
        analysis.skeleton_size(bvh, foot_joints=["NotAJoint"])


def test_skeleton_size_unmeasurable_raises_not_fabricates():
    """A rig whose size cannot be measured raises rather than returning a
    substitute — 1.0 reads as a plausible metre-scale humanoid."""
    bvh = make_pos_y_up_bvh()
    # feet are end sites here, so auto-detection finds none
    with pytest.raises(ValueError, match="no foot joints"):
        analysis.skeleton_size(bvh)
    # explicit feet that coincide with the root: no measurable size either
    with pytest.raises(ValueError, match="no measurable size"):
        analysis.skeleton_size(bvh, foot_joints=["Hips"])


def test_relative_scale_factor_recovers_known_scale():
    # relative_scale_factor lives in pybvh.analysis (next to skeleton_size)
    # and is re-exported at package level.
    import pybvh

    rng = np.random.default_rng(3)
    target = rng.normal(size=(20, 3))
    np.testing.assert_allclose(analysis.relative_scale_factor(2.5 * target, target), 2.5)
    np.testing.assert_allclose(analysis.relative_scale_factor(target, target), 1.0)
    assert pybvh.relative_scale_factor is analysis.relative_scale_factor


def test_relative_scale_factor_edges():
    assert np.isnan(analysis.relative_scale_factor(np.ones((4, 3)), np.zeros((4, 3))))
    with pytest.raises(ValueError):
        analysis.relative_scale_factor(np.ones((4, 3)), np.ones((5, 3)))


def test_relative_scale_factor_centered_ignores_translation():
    """centered=True recovers the scale of a translated pair; the default
    origin fit is corrupted in proportion to the centroid offset."""
    rng = np.random.default_rng(11)
    target = rng.normal(size=(20, 3))
    reference = 2.5 * target + np.array([4.0, -1.0, 2.0])
    shifted_target = target + np.array([10.0, -3.0, 7.0])

    s_origin = analysis.relative_scale_factor(reference, shifted_target)
    assert not np.isclose(s_origin, 2.5)
    np.testing.assert_allclose(
        analysis.relative_scale_factor(reference, shifted_target, centered=True), 2.5
    )
    # centered fit of a constant target is degenerate -> nan
    assert np.isnan(
        analysis.relative_scale_factor(np.ones((4, 3)), np.full((4, 3), 9.0), centered=True)
    )
