"""Golden SE(3) / geodesic tests vs frozen pytransform3d / scipy references.

These target `rotations.py` functions that DO NOT EXIST YET — they SKIP until
implemented, then validate against the frozen oracle. The convention is locked:
se(3) twist = ``[omega(3), v(3)]`` rotation-first, V-left-Jacobian-coupled
(matches pytransform3d and Vemulapalli 2014). See each fixture's `meta`.

Edge cases are over-covered on purpose — theta->0 (V Taylor series) and
theta->pi (log branch) are where SE(3) exp/log implementations break.
"""
import os

import numpy as np
import pytest

import pybvh.rotations as rot

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    p = os.path.join(FX, name + ".npz")
    if not os.path.exists(p):
        pytest.skip(f"missing fixture {name}.npz "
                    "(run tests/fixtures/generate_fixtures.py in pybvh_test)")
    return np.load(p)


def _require(*names):
    missing = [n for n in names if not hasattr(rot, n)]
    if missing:
        pytest.skip(f"rotations.{'/'.join(missing)} not implemented yet")


# pytransform3d's V-coupling (the SE(3) left Jacobian) is computed from
# ``1 - cos θ``, which underflows to 0 in float64 for θ ≲ 1e-4. With a large
# translation (the fixture stresses ‖v‖ up to 100) the lost coupling term is
# ~5e-8, so the frozen (twist, transform) PAIR is only internally consistent
# to ~5e-8 in that regime. pybvh uses the Taylor series there and is accurate
# to ~1e-12 (pinned independently in test_se3_exp_small_angle_vs_analytic_series).
# So the vs-pt comparisons are tight where pt is exact (θ ≥ 1e-2) and relaxed
# to 1e-7 in pt's underflow regime — NOT convention slack: a wrong [ω,v] order
# or V coupling is O(1), caught at either tolerance.
_PT_UNDERFLOW = 1e-2


def _v_series(omega):
    """Left Jacobian V via the Taylor series (no 1-cosθ underflow)."""
    theta2 = float(omega @ omega)
    K = rot._skew(omega)
    b = 0.5 - theta2 / 24.0 + theta2 ** 2 / 720.0
    c = 1.0 / 6.0 - theta2 / 120.0 + theta2 ** 2 / 5040.0
    return np.eye(3) + b * K + c * (K @ K)


# ---------- se3_exp: single-valued -> clean everywhere, edges included ----------
def test_se3_exp_vs_pytransform3d():
    _require("se3_exp")
    d = _load("se3_exp_log")
    mine = rot.se3_exp(d["twist"])
    exact = d["theta"] >= _PT_UNDERFLOW
    np.testing.assert_allclose(mine[exact], d["transform"][exact], atol=1e-9)
    np.testing.assert_allclose(mine[~exact], d["transform"][~exact], atol=1e-7)


# ---------- se3_exp small-angle V coupling: independent analytic oracle ----------
def test_se3_exp_small_angle_vs_analytic_series():
    """Pin the V left-Jacobian coupling at θ→0 against the closed-form series —
    independent of pytransform3d, which underflows here. This is where SE(3)
    implementations tend to break, so it is verified tightly (1e-11)."""
    _require("se3_exp")
    d = _load("se3_exp_log")
    twist = d["twist"]
    small = d["theta"] < _PT_UNDERFLOW
    assert small.sum() >= 10
    expected_d = np.array([_v_series(twist[i, :3]) @ twist[i, 3:]
                           for i in np.nonzero(small)[0]])
    got_d = rot.se3_exp(twist[small])[:, :3, 3]
    np.testing.assert_allclose(got_d, expected_d, atol=1e-11)


# ---------- se3_log: compare to ref twist only where unambiguous ----------
def test_se3_log_vs_pytransform3d_unambiguous():
    _require("se3_log")
    d = _load("se3_exp_log")
    unambiguous = d["theta"] < (np.pi - 1e-3)
    exact = unambiguous & (d["theta"] >= _PT_UNDERFLOW)
    loose = unambiguous & (d["theta"] < _PT_UNDERFLOW)
    np.testing.assert_allclose(rot.se3_log(d["transform"][exact]),
                               d["twist"][exact], atol=1e-8)
    np.testing.assert_allclose(rot.se3_log(d["transform"][loose]),
                               d["twist"][loose], atol=1e-7)


# ---------- round-trip: branch-agnostic; MUST hold at every edge case ----------
def test_se3_roundtrip_all():
    _require("se3_exp", "se3_log")
    T = _load("se3_exp_log")["transform"]
    np.testing.assert_allclose(rot.se3_exp(rot.se3_log(T)), T, atol=1e-9)


def test_se3_small_angle_V_jacobian():
    """theta -> 0 (incl. with large translation): V Taylor must stay stable."""
    _require("se3_exp", "se3_log")
    d = _load("se3_exp_log")
    mask = d["theta"] < 1e-2
    assert mask.sum() >= 10                      # ensure the regime is exercised
    T = d["transform"][mask]
    # round-trip is self-consistent -> tight everywhere
    np.testing.assert_allclose(rot.se3_exp(rot.se3_log(T)), T, atol=1e-9)
    # vs the frozen pt pair: relaxed (pt underflows the coupling here, see top)
    np.testing.assert_allclose(rot.se3_exp(d["twist"][mask]), T, atol=1e-7)


def test_se3_near_pi_log_branch():
    """theta -> pi: log is multivalued; round-trip must still reconstruct T."""
    _require("se3_exp", "se3_log")
    d = _load("se3_exp_log")
    mask = d["theta"] > (np.pi - 1e-3)
    assert mask.sum() >= 6
    T = d["transform"][mask]
    np.testing.assert_allclose(rot.se3_exp(rot.se3_log(T)), T, atol=1e-7)


def test_se3_pure_translation():
    """theta == 0: V == I, so v is the raw translation; log recovers [0, v]."""
    _require("se3_log")
    d = _load("se3_exp_log")
    mask = d["theta"] == 0.0
    assert mask.sum() >= 3
    np.testing.assert_allclose(rot.se3_log(d["transform"][mask]),
                               d["twist"][mask], atol=1e-12)


# ---------- screw interpolation ----------
def test_screw_interpolate_vs_pytransform3d():
    _require("screw_interpolate")
    d = _load("se3_screw_interp")
    out = np.array([rot.screw_interpolate(a, b, t)
                    for a, b, t in zip(d["T0"], d["T1"], d["t"])])
    np.testing.assert_allclose(out, d["interp"], atol=1e-9)


def test_screw_interpolate_endpoints():
    _require("screw_interpolate")
    d = _load("se3_screw_interp")
    for a, b, t in zip(d["T0"], d["T1"], d["t"]):
        if t == 0.0:
            np.testing.assert_allclose(rot.screw_interpolate(a, b, 0.0), a, atol=1e-9)
        elif t == 1.0:
            np.testing.assert_allclose(rot.screw_interpolate(a, b, 1.0), b, atol=1e-9)


# ---------- rotation geodesic distance ----------
def test_rotation_geodesic_distance_vs_scipy():
    _require("rotation_geodesic_distance")
    d = _load("rotation_geodesic")
    out = rot.rotation_geodesic_distance(d["rotmat_a"], d["rotmat_b"])
    np.testing.assert_allclose(out, d["angle"], atol=1e-9)
