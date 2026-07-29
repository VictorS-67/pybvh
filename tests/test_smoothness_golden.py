"""Golden smoothness tests vs the frozen Balasubramanian SPARC reference.

The reference (siva82kb/SPARC, ISC license) was run offline at a pinned commit
by tests/fixtures/generate_fixtures.py; only its NUMBERS are committed (its code
is not). These target `analysis.sparc` / `…log_dimensionless_jerk` /
`…dimensionless_jerk`, which don't exist yet — they SKIP until implemented.

When implemented, our defaults must match the reference's pinned convention
(SPARC: padlevel=4, fc=10 Hz, amp_th=0.05 — see the fixture `meta`).
"""
import json
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


# The reference was only ever consulted at pybvh's defaults, so `padlevel`
# / `fc` / `amp_th` — the entire reason sparc takes parameters — had no
# external check. These pin the non-default paths too.

def _sparc_param_sets():
    d = _load("smoothness")
    meta = json.loads(str(d["meta"]))
    return meta.get("sparc_param_sets", [])


@pytest.mark.parametrize("index", range(4))
def test_sparc_non_default_parameters_vs_reference(index):
    fn = _fn("sparc")
    if fn is None:
        pytest.skip("analysis.sparc not implemented yet")
    d = _load("smoothness")
    key = f"sparc{index}"
    if key not in d:
        pytest.skip(f"fixture predates {key} (regenerate generate_fixtures.py)")
    params = _sparc_param_sets()[index]
    fs = float(d["fs"])
    out = np.array([fn(s, fs, **params) for s in d["signals"]])
    np.testing.assert_allclose(out, d[key], atol=1e-6)


def test_the_pinned_parameter_sets_are_not_all_the_same_answer():
    """A parameter row that never changes the result would not test its plumbing."""
    d = _load("smoothness")
    if "sparc1" not in d:
        pytest.skip("fixture predates the parameter sweep")
    rows = [d[f"sparc{i}"] for i in range(4)]
    for i in range(1, 4):
        assert not np.allclose(rows[i], rows[0]), (
            f"sparc{i} matches the default row, so it cannot detect "
            f"{_sparc_param_sets()[i]} being ignored")


def test_signed_input_is_part_of_the_pinned_contract():
    """One fixture signal goes negative — deliberately, and it must stay that way.

    These measures are defined on x(t), "any scalar coordinate", so the
    velocity they consume carries a sign. A future change that rejected
    negative input would silently drop this coverage.
    """
    d = _load("smoothness")
    assert any((s < 0).any() for s in d["signals"])
