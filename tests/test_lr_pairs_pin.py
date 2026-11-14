"""Behavior pin of every value the L/R pairs feed, on every bundled clip.

``tests/fixtures/lr_pairs_pinned.npz`` froze, before the name pairs went
through one resolver (#216), ``lr_pairs``, ``node_lr_pairs``, the body
anchor, the fields of the orientation report that the pairs feed, the
facing and the result of ``mirror`` (with the clip's own mapping, and with
it passed explicitly) on each ``.bvh`` of ``bvh_data/`` and
``bvh_data/synthetic/``. Each test reads a clip again and compares; a value
that raised is pinned as its message.

The report was first pinned as its repr. The repr is not documented
output, so each field is now its own key; those keys were made on ``main``
and checked equal to this tree's values and to the repr they replace.

When the pin was made (#216), every value was checked equal bit for bit
against ``main``. The test compares floats with ``atol=1e-12`` all the
same, so that the last bits may differ: CI runs other numpy builds.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from generate_fixtures import lr_pin_clip_paths, lr_pin_values  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
PIN_PATH = Path(__file__).parent / "fixtures" / "lr_pairs_pinned.npz"


@pytest.fixture(scope="module")
def pinned():
    return np.load(PIN_PATH)


@pytest.mark.parametrize("path", lr_pin_clip_paths(str(REPO_ROOT)))
def test_every_value_the_pairs_feed_matches_the_pin(pinned, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bvh = read_bvh_file(REPO_ROOT / path)
    values = lr_pin_values(bvh)
    pinned_keys = {key[len(path) + 1 :] for key in pinned.files if key.startswith(path + "/")}
    assert set(values) == pinned_keys
    for key, value in values.items():
        expected = pinned[f"{path}/{key}"]
        if value.dtype.kind == "f":
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12, err_msg=key)
        else:
            np.testing.assert_array_equal(value, expected, err_msg=key)
