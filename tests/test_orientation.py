"""The orientation module's interface, on plain data (ADR 0004).

Each test builds the arrays it needs (names, rest positions, index pairs)
and reads the answer the module gives, with no clip in hand. Tests of the
same facts through ``Bvh`` live with the members that read them.
"""

from __future__ import annotations

import numpy as np

from pybvh._orientation import _infer_rest_up, _up_marker_index

# A small T-pose standing along +y, its arms along x, facing +z: the root at
# the origin, a spine and a head above it, a left and a right hand.
_NAMES = ["Hips", "Spine", "Head", "LeftHand", "RightHand"]
_REST = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 10.0, 0.0],
        [6.0, 8.0, 0.0],
        [-6.0, 8.0, 0.0],
    ]
)


class TestUpMarker:
    """The one body-part lookup for "up", shared by rest up and world up."""

    def test_head_comes_before_the_spine(self):
        assert _up_marker_index(_NAMES, _REST) == 2

    def test_the_first_node_of_a_repeated_name(self):
        names = ["Hips", "Head", "Head"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 9.0, 0.0]])
        assert _up_marker_index(names, rest) == 1

    def test_a_node_at_the_root_is_skipped(self):
        names = ["Hips", "Head", "Neck"]
        rest = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 9.0, 3.0]])
        assert _up_marker_index(names, rest) == 2

    def test_names_match_whole_and_ignore_case(self):
        names = ["Hips", "mixamorig:Head", "SPINE"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, -3.0]])
        assert _up_marker_index(names, rest) == 2

    def test_none_without_a_named_part(self):
        assert _up_marker_index(["Root", "A", "B"], _REST[:3]) is None


class TestRestUp:
    def test_read_from_the_named_part(self):
        assert _infer_rest_up(_NAMES, _REST) == "+y"

    def test_read_from_the_first_of_two_heads(self):
        names = ["Hips", "Head", "Head"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 9.0, 0.0]])
        assert _infer_rest_up(names, rest) == "-z"

    def test_largest_spread_without_a_named_part(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        assert _infer_rest_up(["Root", "A", "B"], rest) == "+z"

    def test_none_on_a_degenerate_rest_pose(self):
        assert _infer_rest_up(["Root"], np.zeros((1, 3))) is None
        assert _infer_rest_up(["Root", "Head"], np.ones((2, 3))) is None
