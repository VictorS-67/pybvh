"""Differential ('golden') tests: pybvh.rotations vs frozen scipy references.

The reference values in ``tests/fixtures/*.npz`` were generated once, offline,
by ``tests/fixtures/generate_fixtures.py`` running in the ``pybvh_test`` env
(which has scipy). This test runs in the numpy-only ``pybvh`` env — it just
loads the frozen arrays, so scipy is NOT a runtime or CI dependency.

If one of these fails, suspect a CONVENTION mismatch first (see each fixture's
``meta`` JSON), not necessarily a bug — that is the documented failure mode of
differential testing.
"""
import json
import os

import numpy as np
import pytest

import pybvh.rotations as rot

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    path = os.path.join(FX, name + ".npz")
    if not os.path.exists(path):
        pytest.skip(f"missing golden fixture {name}.npz "
                    "(run tests/fixtures/generate_fixtures.py in pybvh_test)")
    return np.load(path)


def test_euler_to_rotmat_vs_scipy():
    d = _load("euler_zyx_to_rotmat")
    out = rot.euler_to_rotmat(d["euler"], "ZYX")          # radians, intrinsic
    np.testing.assert_allclose(out, d["rotmat"], atol=1e-10)


def test_rotmat_to_quat_vs_scipy():
    d = _load("rotmat_to_quat")
    out = rot.rotmat_to_quat(d["rotmat"])                 # (w,x,y,z)
    ref = d["quat_wxyz"]
    # quaternion double-cover: q and -q are the same rotation -> compare |dot|.
    dots = np.abs(np.sum(out * ref, axis=-1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-8)


def test_rotmat_to_axisangle_vs_scipy():
    d = _load("rotmat_to_axisangle")
    out = rot.rotmat_to_axisangle(d["rotmat"])            # rotvec = axis*angle
    # angles are bounded < 0.85*pi in the fixture, so the axis/sign is
    # unambiguous and an elementwise comparison is clean.
    np.testing.assert_allclose(out, d["rotvec"], atol=1e-7)


def test_fixture_meta_present():
    """Conventions must be documented alongside the numbers (anti-drift)."""
    for name in ("euler_zyx_to_rotmat", "rotmat_to_quat", "rotmat_to_axisangle"):
        meta = json.loads(str(_load(name)["meta"]))
        assert "ref" in meta and "seed" in meta
