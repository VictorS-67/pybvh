"""Tests for the v0.8.0 pybvh.tools signal utilities + scale functions."""
import numpy as np
import pytest

from pybvh import tools, analysis
from synthetic_bvh import make_pos_y_up_bvh


# ----------------------------------------------------------------
#  temporal_stats
# ----------------------------------------------------------------

def test_temporal_stats_basic_moments():
    sig = np.array([1.0, 2, 3, 4, 5])
    s = tools.temporal_stats(sig)
    np.testing.assert_allclose(s.mean, 3.0)
    np.testing.assert_allclose(s.std, np.std(sig))
    assert s.min == 1.0 and s.max == 5.0
    np.testing.assert_allclose(s.skewness, 0.0, atol=1e-12)  # symmetric


def test_temporal_stats_constant_signal_is_nan_skew_kurt():
    s = tools.temporal_stats(np.full(10, 7.0))
    assert np.isnan(s.skewness) and np.isnan(s.kurtosis)


def test_temporal_stats_axis_and_kurtosis_sign():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(5000, 2))
    s = tools.temporal_stats(sig, axis=0)
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
        out = tools.box_filter_smooth(x, w)
        assert out.shape == x.shape
        np.testing.assert_allclose(out, _ref_smooth(x, w))


def test_box_filter_constant_and_noop_and_validation():
    np.testing.assert_allclose(tools.box_filter_smooth(np.full(8, 3.0), 4), 3.0)
    x = np.arange(6.0)
    np.testing.assert_allclose(tools.box_filter_smooth(x, 1), x)
    with pytest.raises(ValueError):
        tools.box_filter_smooth(x, 0)


def test_box_filter_vectorizes_over_axis():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(20, 3))
    out = tools.box_filter_smooth(sig, 5, axis=0)
    for c in range(3):
        np.testing.assert_allclose(out[:, c], tools.box_filter_smooth(sig[:, c], 5))


# ----------------------------------------------------------------
#  fft_magnitude / dominant_frequency
# ----------------------------------------------------------------

def test_dominant_frequency_pure_sinusoid():
    fs, f0, T = 100.0, 5.0, 200
    t = np.arange(T) / fs
    sig = 3.0 + np.sin(2 * np.pi * f0 * t)  # offset must not win (DC excluded)
    np.testing.assert_allclose(tools.dominant_frequency(sig, fs), f0)


def test_fft_magnitude_shapes_and_peak():
    fs, T = 64.0, 128  # resolution fs/T = 0.5 Hz -> 8 Hz is an exact bin
    t = np.arange(T) / fs
    sig = np.cos(2 * np.pi * 8.0 * t)
    freqs, mag = tools.fft_magnitude(sig, fs)
    assert freqs.shape == (T // 2 + 1,) and mag.shape == (T // 2 + 1,)
    np.testing.assert_allclose(freqs[np.argmax(mag)], 8.0)


# ----------------------------------------------------------------
#  ramer_douglas_peucker
# ----------------------------------------------------------------

def test_rdp_collinear_reduces_to_endpoints():
    line = np.stack([np.linspace(0, 10, 11), np.zeros(11)], axis=1)
    out = tools.ramer_douglas_peucker(line, eps=1e-6)
    np.testing.assert_allclose(out, [[0, 0], [10, 0]])


def test_rdp_keeps_significant_corner():
    curve = np.array([[0.0, 0], [1, 0.01], [2, 0], [2, 1], [2, 2]])
    out = tools.ramer_douglas_peucker(curve, eps=0.1)
    # the (2,0) corner is kept; the near-collinear (1,0.01) is dropped
    assert any(np.allclose(p, [2, 0]) for p in out)
    assert not any(np.allclose(p, [1, 0.01]) for p in out)


def test_rdp_no_recursion_overflow_on_zigzag():
    # a pathological alternating curve drives RDP to maximum split depth;
    # the explicit-stack implementation must not raise RecursionError
    n = 3000
    zig = np.stack([np.arange(n, dtype=float), (np.arange(n) % 2) * 1.0], axis=1)
    out = tools.ramer_douglas_peucker(zig, 0.1)
    assert out.shape[1] == 2 and len(out) >= 2


def test_rdp_3d_and_large_eps():
    rng = np.random.default_rng(2)
    curve = rng.normal(size=(30, 3))
    out = tools.ramer_douglas_peucker(curve, eps=1e9)  # everything within eps
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
    np.testing.assert_allclose(analysis.skeleton_size(bigger, foot_joints=feet),
                               2.0 * size)


def test_skeleton_size_unknown_explicit_joint_raises():
    bvh = make_pos_y_up_bvh()
    with pytest.raises(ValueError, match="not found"):
        analysis.skeleton_size(bvh, foot_joints=["NotAJoint"])
    # the 1.0 fallback remains for footless *auto-detection* only
    assert analysis.skeleton_size(bvh) == 1.0        # feet are end sites here


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
