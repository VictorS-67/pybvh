"""Golden smoothness tests vs the frozen Balasubramanian SPARC reference.

The reference (siva82kb/SPARC, ISC license) was run offline at a pinned commit
by tests/fixtures/generate_fixtures.py; only its NUMBERS are committed (its code
is not). These target `analysis.sparc` / `…log_dimensionless_jerk` /
`…dimensionless_jerk`, which don't exist yet — they SKIP until implemented.

When implemented, our defaults must match the reference's pinned convention
(SPARC: padlevel=4, fc=10 Hz, amp_th=0.05 — see the fixture `meta`).
"""
import os

import numpy as np
import pytest

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    p = os.path.join(FX, name + ".npz")
    if not os.path.exists(p):
        pytest.skip(f"missing fixture {name}.npz "
                    "(run tests/fixtures/generate_fixtures.py in pybvh_test)")
    return np.load(p)


def _fn(name):
    """The smoothness kernel `name`, wherever it ends up living, or None."""
    for mod in ("analysis", "kinematics", "features"):
        try:
            m = __import__(f"pybvh.{mod}", fromlist=[name])
        except Exception:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    return None


def _check(name, ref_key):
    fn = _fn(name)
    if fn is None:
        pytest.skip(f"analysis.{name} not implemented yet")
    d = _load("smoothness")
    fs = float(d["fs"])
    out = np.array([fn(s, fs) for s in d["signals"]])
    np.testing.assert_allclose(out, d[ref_key], atol=1e-6)


def test_sparc_vs_reference():
    _check("sparc", "sparc")


def test_log_dimensionless_jerk_vs_reference():
    _check("log_dimensionless_jerk", "ldlj")


def test_dimensionless_jerk_vs_reference():
    _check("dimensionless_jerk", "dlj")
