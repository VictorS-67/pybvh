"""The orientation module's interface, on plain data (ADR 0004).

Each test builds the arrays it needs (names, rest positions, index pairs)
and reads the answer the module gives, with no clip in hand. Tests of the
same facts through ``Bvh`` live with the members that read them.
"""

from __future__ import annotations

import numpy as np

from pybvh._orientation import (
    _detect_lr_mapping_by_names,
    _infer_rest_forward,
    _infer_rest_left,
    _infer_rest_up,
    _resolve_lr_pairs,
    _resolve_node_lr_pairs,
    _root_heading_units,
    _up_marker_index,
)

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


class TestLrMappingFromNames:
    def test_pairs_by_each_naming_rule(self):
        names = ["Hips", "LeftArm", "RightArm", "leg.L", "leg.R", "LFoot", "RFoot", "Spine"]
        assert _detect_lr_mapping_by_names(names) == {
            "LeftArm": "RightArm",
            "leg.L": "leg.R",
            "LFoot": "RFoot",
        }

    def test_a_side_without_its_partner_is_not_paired(self):
        assert _detect_lr_mapping_by_names(["Hips", "LeftLeg", "OtherLeg"]) == {}


class TestLrPairs:
    # Hips, then a left and a right arm, the left with one end site and
    # the right with one, then a left and a right hand whose end-site
    # counts differ (two on the left, one on the right).
    _NODES = [
        ("Hips", -1, False),
        ("LeftArm", 0, False),
        ("EndSiteLeftArm", 1, True),
        ("RightArm", 0, False),
        ("EndSiteRightArm", 3, True),
        ("LeftHand", 0, False),
        ("EndSiteLeftHand", 5, True),
        ("EndSiteLeftHand", 5, True),
        ("RightHand", 0, False),
        ("EndSiteRightHand", 8, True),
    ]
    _MAPPING = {"LeftArm": "RightArm", "LeftHand": "RightHand"}

    def _resolve(self, mapping):
        names = [name for name, _, _ in self._NODES]
        parent_idx = np.array([parent for _, parent, _ in self._NODES])
        is_end_site = np.array([end for _, _, end in self._NODES])
        return _resolve_node_lr_pairs(mapping, names, parent_idx, is_end_site)

    def test_joint_space_pairs_follow_the_mapping(self):
        joint_index = {"Hips": 0, "LeftArm": 1, "RightArm": 2, "LeftHand": 3, "RightHand": 4}
        assert _resolve_lr_pairs(self._MAPPING, joint_index) == [(1, 2), (3, 4)]

    def test_node_space_pairs_joints_first_then_their_end_sites(self):
        pairs, _ = self._resolve(self._MAPPING)
        assert pairs == [(1, 3), (5, 8), (2, 4)]

    def test_an_end_site_mismatch_is_reported_not_decided(self):
        _, mismatches = self._resolve(self._MAPPING)
        assert mismatches == [("LeftHand", 2, "RightHand", 1)]

    def test_a_symmetric_mapping_gives_each_pair_once(self):
        symmetric = {**self._MAPPING, **{r: left for left, r in self._MAPPING.items()}}
        assert self._resolve(symmetric) == self._resolve(self._MAPPING)

    def test_no_mapping_no_pairs(self):
        assert self._resolve(None) == ([], [])


class TestRestLeft:
    """The skeleton's own left: the L/R geometry measured against rest up."""

    _PAIRS = np.array([[3, 4]])  # LeftHand, RightHand

    def test_from_the_pairs_at_rest(self):
        assert _infer_rest_left(_REST, self._PAIRS, "+y") == "+x"

    def test_measured_against_rest_up_not_any_other_axis(self):
        # The left hand sits much higher than the right: the up component
        # is removed before the axis is read.
        rest = _REST.copy()
        rest[3] = [1.0, 20.0, 0.0]
        assert _infer_rest_left(rest, self._PAIRS, "+y") == "+x"

    def test_none_when_the_pairs_lie_along_rest_up(self):
        rest = _REST.copy()
        rest[3], rest[4] = [0.0, 9.0, 0.0], [0.0, 1.0, 0.0]
        assert _infer_rest_left(rest, self._PAIRS, "+y") is None

    def test_none_without_pairs_or_without_rest_up(self):
        assert _infer_rest_left(_REST, np.empty((0, 2), dtype=np.intp), "+y") is None
        assert _infer_rest_left(_REST, self._PAIRS, None) is None


class TestRestForward:
    """Rest forward = rest left x rest up, never read against world up."""

    def test_left_cross_up(self):
        assert _infer_rest_forward("+x", "+y") == "+z"
        assert _infer_rest_forward("-x", "+z") == "+y"
        assert _infer_rest_forward("+z", "+y") == "-x"

    def test_the_house_default_is_keyed_on_rest_up(self):
        assert _infer_rest_forward(None, "+y") == "+z"
        assert _infer_rest_forward(None, "-z") == "+x"
        assert _infer_rest_forward(None, "+x") == "+y"

    def test_the_house_default_without_rest_up_reads_the_house_up(self):
        assert _infer_rest_forward(None, None) == "+z"


def _about(axis, degrees):
    """The rotation matrix turning *degrees* about the unit *axis*."""
    axis = np.asarray(axis, dtype=np.float64)
    theta = np.radians(degrees)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * cross @ cross


class TestRootHeading:
    """The heading: the root rotation applied to rest forward, on the ground."""

    def test_identity_root_reads_rest_forward(self):
        heading = _root_heading_units(np.eye(3)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_a_turn_about_up_turns_the_heading(self):
        # A quarter turn about +y carries +z onto +x.
        heading = _root_heading_units(_about([0, 1, 0], 90)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_a_tilted_root_is_read_on_the_ground(self):
        # 30 degrees forward pitch about +x: +z goes to (0, -0.5, 0.866),
        # whose ground direction is still +z.
        heading = _root_heading_units(_about([1, 0, 0], 30)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_nan_where_the_forward_stands_along_world_up(self):
        rotmats = np.stack([np.eye(3), _about([1, 0, 0], -90), np.eye(3)])
        heading = _root_heading_units(rotmats, "+z", "+y")
        assert np.isnan(heading[1]).all()
        np.testing.assert_allclose(heading[[0, 2]], [[0, 0, 1], [0, 0, 1]], atol=1e-12)

    def test_read_against_the_world_up_given(self):
        # The same root under a z-up world: rest forward +y stays +y.
        heading = _root_heading_units(np.eye(3)[np.newaxis], "+y", "+z")
        np.testing.assert_allclose(heading, [[0.0, 1.0, 0.0]], atol=1e-12)
