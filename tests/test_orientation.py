"""The orientation module's interface, on plain data (ADR 0004).

Each test builds the arrays it needs (names, rest positions, index pairs)
and reads the answer the module gives, with no clip in hand. Tests of the
same facts through ``Bvh`` live with the members that read them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pybvh._orientation import (
    OrientationReport,
    _build_orientation_report,
    _detect_lr_mapping_by_names,
    _facing_basis,
    _facing_labels,
    _infer_rest_forward,
    _infer_rest_left,
    _infer_rest_up,
    _infer_world_up,
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


class TestWorldUpGuess:
    """World up guessed from frame 0's up marker above the root, else from
    rest up, else the house default, each with its source; never a
    warning, which is the reader's to give."""

    @pytest.fixture(autouse=True)
    def _silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    def test_read_from_the_first_frame(self):
        # The T-pose lying on its back: the head along +z from the root.
        frame0 = _REST[:, [0, 2, 1]]
        assert _infer_world_up(_NAMES, _REST, frame0) == ("+z", "first_frame")

    def test_a_first_frame_in_any_convention_wins_over_rest_up(self):
        frame0 = _REST.copy()
        frame0[:, 1] *= -1  # upside down
        assert _infer_world_up(_NAMES, _REST, frame0) == ("-y", "first_frame")

    def test_an_ambiguous_first_frame_defers_to_rest_up(self):
        # The head 45 degrees off the vertical: no axis dominates twice over.
        frame0 = _REST.copy()
        frame0[2] = [5.0, 5.0, 0.0]
        assert _infer_world_up(_NAMES, _REST, frame0) == ("+y", "rest_up")

    def test_without_frames_rest_up(self):
        assert _infer_world_up(_NAMES, _REST, None) == ("+y", "rest_up")

    def test_without_a_named_part_rest_up_from_the_largest_spread(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        frame0 = rest[:, [2, 1, 0]]
        assert _infer_world_up(["Root", "A", "B"], rest, frame0) == ("+z", "rest_up")

    def test_the_house_default_when_nothing_says_which_way_is_up(self):
        assert _infer_world_up(["Root", "Tail"], np.zeros((2, 3)), np.zeros((2, 3))) == (
            "+y",
            "default",
        )


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


class TestFacingChain:
    """One chain per frame: the frame's leftward, else the root's heading at
    that frame, else the house default; forward is always left x up."""

    _PAIRS = np.array([[3, 4]])  # LeftHand, RightHand

    def test_a_pose_with_a_known_left_up_and_forward(self):
        forward, left, up, valid, defaulted = _facing_basis(_REST[np.newaxis], self._PAIRS, "+y")
        np.testing.assert_allclose(left, [[1.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(forward, [[0.0, 0.0, 1.0]], atol=1e-12)
        np.testing.assert_array_equal(up, [[0.0, 1.0, 0.0]])
        assert valid.tolist() == [True]
        assert defaulted.tolist() == [False]

    def test_a_vertical_left_right_axis_falls_back_to_the_root_heading(self):
        # World up along the arms: the frame's left/right axis is vertical,
        # like a body lying on its side. The root, turned a quarter turn
        # about +y, carries rest forward +x onto -z.
        quarter_turn = _about([0, 1, 0], 90)[np.newaxis]
        forward, left, up, valid, defaulted = _facing_basis(
            _REST[np.newaxis], self._PAIRS, "+x", quarter_turn, "+x"
        )
        np.testing.assert_allclose(forward, [[0.0, 0.0, -1.0]], atol=1e-12)
        np.testing.assert_allclose(left, [[0.0, 1.0, 0.0]], atol=1e-12)  # +x cross -z
        assert valid.tolist() == [False]
        assert defaulted.tolist() == [False]

    def test_then_the_house_default_when_the_heading_is_unreadable_too(self):
        # Rest forward +x under an identity root stays along world up +x:
        # no heading to read. Without the inputs the stage is skipped.
        for heading_inputs in ((np.eye(3)[np.newaxis], "+x"), (None, None)):
            forward, left, up, valid, defaulted = _facing_basis(
                _REST[np.newaxis], self._PAIRS, "+x", *heading_inputs
            )
            # The house default for an x-up world faces +y; left = up x forward.
            np.testing.assert_array_equal(forward, [[0.0, 1.0, 0.0]])
            np.testing.assert_array_equal(left, [[0.0, 0.0, 1.0]])
            assert valid.tolist() == [False]
            assert defaulted.tolist() == [True]

    def test_left_is_never_world_up(self):
        # #114 reproduction 1, on arrays: the left/right axis along world up.
        _, left, up, _, _ = _facing_basis(_REST[np.newaxis], self._PAIRS, "-x")
        assert abs(float(left[0] @ up[0])) < 1e-12

    def test_no_pairs_every_row_defaults(self):
        coords = np.stack([_REST, _REST])
        forward, left, _, valid, defaulted = _facing_basis(
            coords, np.empty((0, 2), dtype=np.intp), "+y"
        )
        np.testing.assert_array_equal(forward, np.tile([0.0, 0.0, 1.0], (2, 1)))
        np.testing.assert_array_equal(left, np.tile([1.0, 0.0, 0.0], (2, 1)))
        assert not valid.any() and defaulted.all()

    def test_forward_cross_left_is_up_on_every_row(self):
        # One row of each stage, measured, from the heading and defaulted,
        # under each of the six world ups.
        turned = _REST @ _about([0, 1, 0], 30).T
        for world_up in ("+x", "-x", "+y", "-y", "+z", "-z"):
            up_idx = "xyz".index(world_up[1])
            tipped = _REST.copy()  # the hands one above the other along up
            tipped[3], tipped[4] = np.eye(3)[up_idx] * 9.0, np.eye(3)[up_idx]
            coords = np.stack([turned, tipped, tipped])
            # Rest forward is the axis after up: row 1's root turns it 40
            # degrees about up, row 2's stands it along up.
            ahead, side = np.eye(3)[(up_idx + 1) % 3], np.eye(3)[(up_idx + 2) % 3]
            turn = _about(np.eye(3)[up_idx], 40)
            rotmats = np.stack([np.eye(3), turn, _about(side, 90)])
            forward, left, up, valid, defaulted = _facing_basis(
                coords, self._PAIRS, world_up, rotmats, "+" + "xyz"[(up_idx + 1) % 3]
            )
            assert valid.tolist() == [True, False, False]
            assert defaulted.tolist() == [False, False, True]
            np.testing.assert_allclose(forward[1], turn @ ahead, atol=1e-12)
            np.testing.assert_allclose(np.cross(forward, left), up, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(forward, axis=1), 1.0, atol=1e-12)


class TestFacingLabels:
    """The snapped labels of one facing row: forward, and left = up x forward."""

    def test_a_row_snaps_to_its_nearest_axes(self):
        assert _facing_labels(np.array([0.2, 0.0, 0.98]), "+y") == ("+z", "+x")
        assert _facing_labels(np.array([-0.9, 0.1, 0.0]), "+z") == ("-x", "-y")

    def test_a_diagonal_row_gives_a_right_handed_pair(self):
        forward, left = _facing_labels(np.array([1.0, 0.0, 1.0]) / np.sqrt(2), "+y")
        up = np.array([0.0, 1.0, 0.0])
        vectors = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}
        np.testing.assert_array_equal(np.cross(vectors[forward], vectors[left]), up)


class TestReport:
    """The skeleton's own frame read from the rest pose, beside a world up
    the caller settled, each fact with its source."""

    _PAIRS = np.array([[3, 4]])  # LeftHand, RightHand

    def test_a_measured_skeleton(self):
        report = _build_orientation_report(
            _NAMES, _REST, self._PAIRS, "+z", "first_frame", False, "names"
        )
        assert report == OrientationReport(
            rest_up="+y",
            rest_up_source="named_joint",
            rest_left="+x",
            rest_forward="+z",
            rest_forward_source="lr_geometry",
            world_up="+z",
            world_up_source="first_frame",
            world_up_inherited=False,
            lr_mapping_source="names",
        )

    def test_rest_up_from_the_largest_spread(self):
        names = ["Root", "A", "LeftHand", "RightHand"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [2.0, 4.0, 0.0], [-2.0, 4.0, 0.0]])
        report = _build_orientation_report(names, rest, [[2, 3]], "+y", "set", False, "set")
        assert (report.rest_up, report.rest_up_source) == ("+y", "largest_spread")
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "lr_geometry")

    def test_rest_forward_defaulted_without_pairs(self):
        report = _build_orientation_report(
            _NAMES, _REST, np.empty((0, 2), dtype=np.intp), "+y", "rest_up", True, None
        )
        assert report.rest_left is None
        # The house default for rest up +y.
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "default")

    def test_nothing_measured_on_a_degenerate_rest_pose(self):
        report = _build_orientation_report(
            ["Root"],
            np.zeros((1, 3)),
            np.empty((0, 2), dtype=np.intp),
            "+y",
            "default",
            False,
            None,
        )
        assert (report.rest_up, report.rest_up_source, report.rest_left) == (None, None, None)
        # The house default keyed on the house up '+y'.
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "default")
