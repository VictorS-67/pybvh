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


# ---------- se3_exp: single-valued -> clean everywhere, edges included ----------
def test_se3_exp_vs_pytransform3d():
    _require("se3_exp")
    d = _load("se3_exp_log")
    np.testing.assert_allclose(rot.se3_exp(d["twist"]), d["transform"], atol=1e-9)


# ---------- se3_log: compare to ref twist only where unambiguous ----------
def test_se3_log_vs_pytransform3d_unambiguous():
    _require("se3_log")
    d = _load("se3_exp_log")
    mask = d["theta"] < (np.pi - 1e-3)
    np.testing.assert_allclose(rot.se3_log(d["transform"][mask]),
                               d["twist"][mask], atol=1e-8)


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
    np.testing.assert_allclose(rot.se3_exp(rot.se3_log(T)), T, atol=1e-9)
    np.testing.assert_allclose(rot.se3_exp(d["twist"][mask]), T, atol=1e-9)


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
