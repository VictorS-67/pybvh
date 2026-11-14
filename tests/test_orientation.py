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
    _BodyAnchor,
    _build_orientation_report,
    _detect_lr_mapping_by_names,
    _facing_basis,
    _facing_labels,
    _find_body,
    _find_body_anchor,
    _heading_units,
    _infer_rest_forward,
    _infer_rest_left,
    _infer_rest_up,
    _infer_rest_up_and_left,
    _infer_world_up,
    _resolve_lr_node_pairs,
    _RestSkeleton,
    _RestUp,
    _size_reference,
    _up_marker_index,
    _warn_facing_defaulted_at_frames,
    _WorldUpGuess,
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


def _skeleton(names, rest, lr_pairs=(), end_sites=(), parents=None, static_root=False):
    """The rest skeleton the module reads, from names, positions, pairs, the
    indices of the end sites, each node's parent (by default, every node
    a child of the root) and whether the root is static."""
    is_end_site = np.zeros(len(names), dtype=bool)
    is_end_site[list(end_sites)] = True
    if parents is None:
        parents = [-1] + [0] * (len(names) - 1)
    return _RestSkeleton(
        names,
        np.asarray(rest, dtype=float),
        np.asarray(parents, dtype=np.intp),
        is_end_site,
        np.asarray(lr_pairs, dtype=np.intp).reshape(-1, 2),
        static_root,
    )


# A T-pose wider than tall, standing along +y with its hips at the root and
# nothing named for a body part: arms 20 across, 17 from feet to crown.
_NAMELESS_NAMES = ["Root", "Waist", "Crown", "LeftArm", "RightArm"]
_NAMELESS_NAMES += ["LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
_NAMELESS_REST = [
    [0.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 8.0, 0.0],
    [10.0, 6.0, 0.0],
    [-10.0, 6.0, 0.0],
    [2.0, -1.0, 0.0],
    [2.0, -9.0, 1.0],
    [-2.0, -1.0, 0.0],
    [-2.0, -9.0, 1.0],
]
_NAMELESS_PARENTS = [-1, 0, 1, 1, 1, 0, 5, 0, 7]
_NAMELESS_PAIRS = [[3, 4], [5, 7]]  # the arms, then the legs


class TestUpMarker:
    """The one body-part lookup for "up", shared by rest up and world up."""

    def test_head_comes_before_the_spine(self):
        assert _up_marker_index(_skeleton(_NAMES, _REST), 0) == 2

    def test_the_first_node_of_a_repeated_name(self):
        names = ["Hips", "Head", "Head"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 9.0, 0.0]])
        assert _up_marker_index(_skeleton(names, rest), 0) == 1

    def test_a_node_at_the_root_is_skipped(self):
        names = ["Hips", "Head", "Neck"]
        rest = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 9.0, 3.0]])
        assert _up_marker_index(_skeleton(names, rest), 0) == 2

    @pytest.mark.parametrize(
        "head",
        ["mixamorig:Head", "Bip01 Head", "CC_Base_Head", "J_Bip_C_Head", "Head.001", "HEAD"],
    )
    def test_a_head_named_in_a_rig_convention_comes_before_the_spine(self, head):
        names = ["Hips", "Spine", head]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -3.0], [0.0, 9.0, 0.0]])
        assert _up_marker_index(_skeleton(names, rest), 0) == 2

    def test_a_part_matches_as_a_whole_word_of_the_name(self):
        # "Forehead" holds "head" but is another word; "SPINE" is the
        # spine in capitals.
        names = ["Hips", "Forehead", "SPINE"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, -3.0]])
        assert _up_marker_index(_skeleton(names, rest), 0) == 2

    def test_a_joint_above_the_head_is_a_head(self):
        # Mixamo's HeadTop_End is a joint at the top of the head.
        names = ["Hips", "Spine", "HeadTop_End"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -3.0], [0.0, 9.0, 0.0]])
        assert _up_marker_index(_skeleton(names, rest), 0) == 2

    def test_an_end_site_is_never_the_marker(self):
        # An end site's name is generated from its parent's ('EndSite' +
        # the parent's name), so it names no body part of its own.
        names = ["Hips", "Spine", "EndSiteHead"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -3.0], [0.0, 9.0, 0.0]])
        assert _up_marker_index(_skeleton(names, rest, end_sites=[2]), 0) == 1

    def test_none_without_a_named_part(self):
        assert _up_marker_index(_skeleton(["Root", "A", "B"], _REST[:3]), 0) is None


class TestRestUp:
    """Rest up read from the rest pose alone, with what it was read from."""

    def test_read_from_the_named_part(self):
        assert _infer_rest_up(_skeleton(_NAMES, _REST), 0) == _RestUp("+y", "named_joint")

    def test_read_from_the_first_of_two_heads(self):
        names = ["Hips", "Head", "Head"]
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 9.0, 0.0]])
        assert _infer_rest_up(_skeleton(names, rest), 0) == _RestUp("-z", "named_joint")

    def test_opposite_the_legs_without_a_named_part(self):
        # The largest spread would say x, the arm span; the legs hang from
        # the root (the arms from the waist) and their far ends, the feet,
        # lie along -y from it.
        skeleton = _skeleton(
            _NAMELESS_NAMES, _NAMELESS_REST, _NAMELESS_PAIRS, parents=_NAMELESS_PARENTS
        )
        assert _infer_rest_up(skeleton, 0) == _RestUp("+y", "leg_pair")

    def test_the_legs_below_a_root_on_the_floor(self):
        # Root -> Hips -> legs, the root on the floor under the hips: the
        # legs meet at the hips, the body anchor, and are read from them,
        # since the feet lie level with the root (a little forward, along
        # +z).
        names = ["Root", "Hips", "Waist", "LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
        rest = [
            [0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 14.0, 0.0],
            [2.0, 8.0, 0.0],
            [2.0, 0.0, 1.0],
            [-2.0, 8.0, 0.0],
            [-2.0, 0.0, 1.0],
        ]
        skeleton = _skeleton(names, rest, [[3, 5]], parents=[-1, 0, 1, 1, 3, 1, 5])
        assert _infer_rest_up(skeleton, 1) == _RestUp("+y", "leg_pair")

    def test_a_mapping_of_the_arms_alone_names_no_legs(self):
        # Both arms hang from the waist, one branch of the hips: no pair
        # splits across the hips' branches, so the largest spread answers
        # (the arm span, x), labelled as such.
        skeleton = _skeleton(_NAMELESS_NAMES, _NAMELESS_REST, [[3, 4]], parents=_NAMELESS_PARENTS)
        assert _infer_rest_up(skeleton, 0) == _RestUp("+x", "largest_spread")

    def test_the_legs_whichever_pair_comes_first(self):
        # The arms first, then the feet, both pairs two bones from the root:
        # the feet split across the hips' branches, the arms do not.
        skeleton = _skeleton(
            _NAMELESS_NAMES, _NAMELESS_REST, [[3, 4], [6, 8]], parents=_NAMELESS_PARENTS
        )
        assert _infer_rest_up(skeleton, 0) == _RestUp("+y", "leg_pair")

    def test_largest_spread_without_a_named_part_or_pairs(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        assert _infer_rest_up(_skeleton(["Root", "A", "B"], rest), 0) == _RestUp(
            "+z", "largest_spread"
        )

    def test_none_on_a_degenerate_rest_pose(self):
        assert _infer_rest_up(_skeleton(["Root"], np.zeros((1, 3))), 0) is None
        assert _infer_rest_up(_skeleton(["Root", "Head"], np.ones((2, 3))), 0) is None


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
        assert _infer_world_up(_skeleton(_NAMES, _REST), 0, frame0) == ("+z", "first_frame")

    def test_a_first_frame_in_any_convention_wins_over_rest_up(self):
        frame0 = _REST.copy()
        frame0[:, 1] *= -1  # upside down
        assert _infer_world_up(_skeleton(_NAMES, _REST), 0, frame0) == ("-y", "first_frame")

    def test_an_ambiguous_first_frame_defers_to_rest_up(self):
        # The head 45 degrees off the vertical: no axis dominates twice over.
        frame0 = _REST.copy()
        frame0[2] = [5.0, 5.0, 0.0]
        assert _infer_world_up(_skeleton(_NAMES, _REST), 0, frame0) == ("+y", "rest_up")

    def test_without_frames_rest_up(self):
        assert _infer_world_up(_skeleton(_NAMES, _REST), 0, None) == ("+y", "rest_up")

    def test_without_a_named_part_rest_up_from_the_leg_pair(self):
        skeleton = _skeleton(
            _NAMELESS_NAMES, _NAMELESS_REST, _NAMELESS_PAIRS, parents=_NAMELESS_PARENTS
        )
        frame0 = np.asarray(_NAMELESS_REST)[:, [0, 2, 1]]
        assert _infer_world_up(skeleton, 0, frame0) == ("+y", "rest_up")

    def test_without_a_named_part_or_pairs_rest_up_from_the_largest_spread(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        frame0 = rest[:, [2, 1, 0]]
        assert _infer_world_up(_skeleton(["Root", "A", "B"], rest), 0, frame0) == ("+z", "rest_up")

    def test_the_house_default_when_nothing_says_which_way_is_up(self):
        assert _infer_world_up(
            _skeleton(["Root", "Tail"], np.zeros((2, 3))), 0, np.zeros((2, 3))
        ) == (
            "+y",
            "default",
        )


# A body under a reference root that lies far from it, along -z, and carries
# a prop that lies farther still, along +x: the body stands along +y from
# its hips (node 1), its legs hang from them, and a head joint lies on them.
_REFERENCE_NAMES = ["Reference", "Hips", "Head", "Neck", "Waist", "Prop"]
_REFERENCE_NAMES += ["LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
_REFERENCE_REST = np.array(
    [
        [0.0, 0.0, -100.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 5.0, 0.0],
        [300.0, 0.0, -100.0],
        [2.0, -1.0, 0.0],
        [2.0, -9.0, 1.0],
        [-2.0, -1.0, 0.0],
        [-2.0, -9.0, 1.0],
    ]
)
_REFERENCE_PARENTS = [-1, 0, 1, 2, 1, 0, 1, 6, 1, 8]
_REFERENCE_LEGS = [[6, 8]]
_HIPS = 1


class TestUpFromTheBodyAnchor:
    """Up read from the body anchor, not from the root: the head joint's
    direction, the skip of a head joint lying on it, the legs that meet at
    it, and frame 0's guess."""

    def test_a_head_joint_lying_on_the_anchor_is_skipped(self):
        # The head lies on the hips, the neck above them; from the root,
        # far below along -z, the head would have a direction.
        skeleton = _skeleton(_REFERENCE_NAMES, _REFERENCE_REST, parents=_REFERENCE_PARENTS)
        assert _up_marker_index(skeleton, _HIPS) == 3

    def test_rest_up_from_the_head_joint_measured_from_the_anchor(self):
        # From the root the neck lies along +z (100 against 8).
        skeleton = _skeleton(_REFERENCE_NAMES, _REFERENCE_REST, parents=_REFERENCE_PARENTS)
        assert _infer_rest_up(skeleton, _HIPS) == _RestUp("+y", "named_joint")

    def test_rest_up_from_the_legs_that_meet_at_the_anchor(self):
        # No head name; the root carries a prop as well as the hips, so a
        # walk down from the root would stop at the root, where the legs
        # do not split, and the prop's spread along x would answer.
        names = ["Reference", "Hips", "Crown", "Brow", "Waist", "Prop"]
        names += ["LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
        skeleton = _skeleton(names, _REFERENCE_REST, _REFERENCE_LEGS, parents=_REFERENCE_PARENTS)
        assert _infer_rest_up(skeleton, _HIPS) == _RestUp("+y", "leg_pair")

    def test_a_pair_that_does_not_meet_at_the_anchor_is_not_read_as_the_legs(self):
        # Both feet below one leg of the hips: the pair meets below the hips.
        names = ["Reference", "Hips", "Crown", "Brow", "Waist", "Prop"]
        names += ["LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
        parents = [-1, 0, 1, 2, 1, 0, 1, 6, 6, 8]
        skeleton = _skeleton(names, _REFERENCE_REST, [[7, 9]], parents=parents)
        assert _infer_rest_up(skeleton, _HIPS).source == "largest_spread"

    def test_the_largest_spread_reads_the_body_alone(self):
        # No head name, no pairs: the body is the anchor and the nodes below
        # it, which stand along +y; the prop beside it, under the root,
        # spreads farther along +x.
        names = ["Scene", "Body", "Torso", "Skull", "Prop"]
        rest = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]
        rest.append([100.0, 0.0, 0.0])
        skeleton = _skeleton(names, rest, parents=[-1, 0, 1, 2, 0])
        assert _infer_rest_up(skeleton, 1) == _RestUp("+y", "largest_spread")
        assert _infer_world_up(skeleton, 1, None) == ("+y", "rest_up")

    def test_world_up_at_the_first_frame_from_the_anchor(self):
        # Frame 0 is the rest pose: from the root, the neck lies along +z.
        skeleton = _skeleton(_REFERENCE_NAMES, _REFERENCE_REST, parents=_REFERENCE_PARENTS)
        assert _infer_world_up(skeleton, _HIPS, _REFERENCE_REST) == ("+y", "first_frame")


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
        return _resolve_lr_node_pairs(mapping, names, parent_idx, is_end_site)

    def test_joint_pairs_follow_the_mapping(self):
        assert self._resolve(self._MAPPING).joint_pairs == [(1, 3), (5, 8)]

    def test_end_site_pairs_follow_their_joint_pairs(self):
        assert self._resolve(self._MAPPING).end_site_pairs == [(2, 4)]

    def test_an_end_site_mismatch_is_reported_not_decided(self):
        resolved = self._resolve(self._MAPPING)
        assert resolved.end_site_mismatches == [("LeftHand", 2, "RightHand", 1)]

    def test_an_unknown_name_is_reported_and_its_pair_skipped(self):
        # An end site's generated name is not a joint's name.
        resolved = self._resolve({"LeftArm": "RightArm", "LeftFoot": "EndSiteRightHand"})
        assert resolved.joint_pairs == [(1, 3)]
        assert resolved.unknown_names == ["EndSiteRightHand", "LeftFoot"]

    def test_a_symmetric_mapping_gives_each_pair_once(self):
        symmetric = {**self._MAPPING, **{r: left for left, r in self._MAPPING.items()}}
        assert self._resolve(symmetric) == self._resolve(self._MAPPING)

    def test_no_mapping_no_pairs(self):
        assert self._resolve(None) == ([], [], [], [])


def _resolve_tree(nodes, mapping):
    """The pairs of *mapping* on a skeleton of ``(name, parent)`` joints."""
    names = [name for name, _ in nodes]
    parent_idx = np.array([parent for _, parent in nodes])
    return _resolve_lr_node_pairs(mapping, names, parent_idx, np.zeros(len(nodes), dtype=bool))


class TestLrPairsOfRepeatedNames:
    """A name pair whose names repeat binds by the tree (#216)."""

    def test_each_left_joint_takes_the_right_joint_it_meets_lowest(self):
        # Two figures under one root, named alike, the second listed with
        # its right leg first.
        nodes = [
            ("Root", -1),
            ("Hips", 0),
            ("LeftUpLeg", 1),
            ("RightUpLeg", 1),
            ("Hips", 0),
            ("RightUpLeg", 4),
            ("LeftUpLeg", 4),
        ]
        resolved = _resolve_tree(nodes, {"LeftUpLeg": "RightUpLeg"})
        assert resolved.joint_pairs == [(2, 3), (6, 5)]

    def test_a_left_joint_with_two_right_joints_equally_close_has_no_pair(self):
        nodes = [("Hips", -1), ("LeftArm", 0), ("RightArm", 0), ("RightArm", 0)]
        assert _resolve_tree(nodes, {"LeftArm": "RightArm"}).joint_pairs == []

    def test_a_figure_without_the_right_joint_lends_none_from_the_other(self):
        # The second figure has no right foot: its left foot meets the
        # first figure's right foot only at the root, which is farther
        # than that right foot's own left foot.
        nodes = [
            ("Root", -1),
            ("Hips", 0),
            ("LeftFoot", 1),
            ("RightFoot", 1),
            ("Hips", 0),
            ("LeftFoot", 4),
        ]
        assert _resolve_tree(nodes, {"LeftFoot": "RightFoot"}).joint_pairs == [(2, 3)]

    def test_a_figure_without_the_left_joint_lends_none_from_the_other(self):
        nodes = [
            ("Root", -1),
            ("Hips", 0),
            ("LeftFoot", 1),
            ("RightFoot", 1),
            ("Hips", 0),
            ("RightFoot", 4),
        ]
        assert _resolve_tree(nodes, {"LeftFoot": "RightFoot"}).joint_pairs == [(2, 3)]


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


class TestAnchorHeading:
    """The heading: the body anchor's world rotation applied to rest forward, on the ground."""

    def test_identity_anchor_reads_rest_forward(self):
        heading = _heading_units(np.eye(3)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_a_turn_about_up_turns_the_heading(self):
        # A quarter turn about +y carries +z onto +x.
        heading = _heading_units(_about([0, 1, 0], 90)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_a_tilted_root_is_read_on_the_ground(self):
        # 30 degrees forward pitch about +x: +z goes to (0, -0.5, 0.866),
        # whose ground direction is still +z.
        heading = _heading_units(_about([1, 0, 0], 30)[np.newaxis], "+z", "+y")
        np.testing.assert_allclose(heading, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_nan_where_the_forward_stands_along_world_up(self):
        rotmats = np.stack([np.eye(3), _about([1, 0, 0], -90), np.eye(3)])
        heading = _heading_units(rotmats, "+z", "+y")
        assert np.isnan(heading[1]).all()
        np.testing.assert_allclose(heading[[0, 2]], [[0, 0, 1], [0, 0, 1]], atol=1e-12)

    def test_read_against_the_world_up_given(self):
        # The same root under a z-up world: rest forward +y stays +y.
        heading = _heading_units(np.eye(3)[np.newaxis], "+y", "+z")
        np.testing.assert_allclose(heading, [[0.0, 1.0, 0.0]], atol=1e-12)


class TestFacingChain:
    """One chain per frame: the frame's leftward, else the anchor's heading at
    that frame, else the house default; forward is always left x up."""

    _PAIRS = np.array([[3, 4]])  # LeftHand, RightHand

    def test_a_pose_with_a_known_left_up_and_forward(self):
        forward, left, up, valid, defaulted = _facing_basis(_REST[np.newaxis], self._PAIRS, "+y")
        np.testing.assert_allclose(left, [[1.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(forward, [[0.0, 0.0, 1.0]], atol=1e-12)
        np.testing.assert_array_equal(up, [[0.0, 1.0, 0.0]])
        assert valid.tolist() == [True]
        assert defaulted.tolist() == [False]

    def test_a_vertical_left_right_axis_falls_back_to_the_anchor_heading(self):
        # World up along the arms: the frame's left/right axis is vertical,
        # like a body lying on its side. The anchor, turned a quarter turn
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

    def test_the_frame_warning_names_every_cause_of_a_default(self):
        # A frame defaults when its pairs give no ground direction, whether
        # they line up with world up or coincide (a rig whose left and right
        # joints sit on the midline), and when the heading cannot stand in,
        # whether it lies along world up or rest forward is itself a default.
        with pytest.warns(UserWarning) as record:
            _warn_facing_defaulted_at_frames("+y", [0, 1])
        message = str(record[0].message)
        assert "At frames 0, 1" in message
        for cause in ("line up with world_up", "coincide", "rest_forward is itself a default"):
            assert cause in message


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


def _tree(nodes, lr_pairs=(), end_sites=(), static_root=False):
    """The rest skeleton of a tree given as ``(name, parent name)`` rows in
    node order, the root's parent ``None``; ``lr_pairs`` and ``end_sites``
    by name. The rest pose is all zeros: the body anchor reads none of it."""
    names = [name for name, _ in nodes]
    parents = [-1 if parent is None else names.index(parent) for _, parent in nodes]
    return _skeleton(
        names,
        np.zeros((len(names), 3)),
        [[names.index(left), names.index(right)] for left, right in lr_pairs],
        [names.index(name) for name in end_sites],
        parents,
        static_root,
    )


# A body whose root is its hips: two legs to their feet, a spine to a chest
# that holds two arms and a head, each chain ending in an end site.
_BODY = [
    ("LeftUpLeg", "Hips"),
    ("LeftFoot", "LeftUpLeg"),
    ("LeftFoot_End", "LeftFoot"),
    ("RightUpLeg", "Hips"),
    ("RightFoot", "RightUpLeg"),
    ("RightFoot_End", "RightFoot"),
    ("Spine", "Hips"),
    ("Chest", "Spine"),
    ("LeftArm", "Chest"),
    ("LeftArm_End", "LeftArm"),
    ("RightArm", "Chest"),
    ("RightArm_End", "RightArm"),
    ("Head", "Chest"),
    ("Head_End", "Head"),
]
_BODY_END_SITES = [name for name, _ in _BODY if name.endswith("_End")]
_LEGS = ("LeftUpLeg", "RightUpLeg")
_FEET = ("LeftFoot", "RightFoot")
_ARMS = ("LeftArm", "RightArm")


def _anchor_of(nodes, lr_pairs, static_root=False):
    """The body anchor the rule finds, as names: the anchor, its source and
    the candidates."""
    end_sites = [n for n, _ in nodes if n.endswith("_End")]
    skeleton = _tree(nodes, lr_pairs, end_sites, static_root)
    found = _find_body_anchor(skeleton)
    names = skeleton.node_names
    return (
        None if found.node is None else names[found.node],
        found.source,
        [names[index] for index in found.candidates],
    )


class TestBodyAnchor:
    """The joint the body's measures start from: where the L/R pairs meet,
    at the shallowest depth; the root without a pair."""

    def test_the_root_is_the_hips(self):
        nodes = [("Hips", None), *_BODY]
        # The arms meet lower, at the chest: the legs' meeting joint is
        # the shallowest.
        assert _anchor_of(nodes, [_LEGS, _FEET, _ARMS]) == ("Hips", "lr_pairs", ["Hips"])

    def test_a_root_above_the_hips(self):
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _anchor_of(nodes, [_LEGS, _ARMS]) == ("Hips", "lr_pairs", ["Hips"])

    def test_a_reference_root_that_carries_props(self):
        nodes = [
            ("Reference", None),
            ("Hips", "Reference"),
            *_BODY,
            ("Sword", "Reference"),
            ("SwordTip", "Sword"),
            ("SwordTip_End", "SwordTip"),
            ("Marker", "Reference"),
            ("Marker_End", "Marker"),
        ]
        assert _anchor_of(nodes, [_LEGS, _ARMS]) == ("Hips", "lr_pairs", ["Hips"])

    def test_a_helper_leaf_beside_the_body(self):
        nodes = [("Bip01", None), ("Footsteps", "Bip01"), ("Hips", "Bip01"), *_BODY]
        assert _anchor_of(nodes, [_LEGS, _ARMS]) == ("Hips", "lr_pairs", ["Hips"])

    def test_a_mapping_of_the_arms_only(self):
        nodes = [("Hips", None), *_BODY]
        assert _anchor_of(nodes, [_ARMS]) == ("Chest", "lr_pairs", ["Chest"])

    def test_no_pair_gives_the_root(self):
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _anchor_of(nodes, []) == ("Reference", "default", [])

    def test_of_candidates_at_one_depth_the_one_with_more_joints_below(self):
        # Two pairs that meet at depth 1: the hips, with 10 joints counted
        # from them, and a helper with 3; end sites are not counted.
        nodes = [
            ("Root", None),
            ("Helper", "Root"),
            ("LeftTarget", "Helper"),
            ("LeftTarget_End", "LeftTarget"),
            ("RightTarget", "Helper"),
            ("RightTarget_End", "RightTarget"),
            ("Hips", "Root"),
            *_BODY,
        ]
        assert _anchor_of(nodes, [("LeftTarget", "RightTarget"), _LEGS]) == (
            "Hips",
            "lr_pairs",
            ["Hips", "Helper"],
        )

    @pytest.mark.parametrize(
        ("sizes", "expected"),
        [
            # Each other candidate fewer than half as many joints: the larger.
            ((20, 9), ("A", "lr_pairs", ["A", "B"])),
            # At least half is the tie: no body.
            ((20, 10), (None, "ambiguous", ["A", "B"])),
            ((10, 10), (None, "ambiguous", ["A", "B"])),
            # The larger one listed first, wherever it lies in node order.
            ((9, 20), ("B", "lr_pairs", ["B", "A"])),
            # Three candidates: every other one must be under half.
            ((20, 9, 9), ("A", "lr_pairs", ["A", "B", "C"])),
            ((20, 9, 10), (None, "ambiguous", ["A", "C", "B"])),
        ],
        ids=["20-9", "20-10", "10-10", "9-20", "20-9-9", "20-9-10"],
    )
    def test_of_several_candidates_the_larger_one_or_no_body(self, sizes, expected):
        # Candidates A, B, C under one root, each of the given size.
        nodes, pairs = [("Root", None)], []
        for name, size in zip("ABC"[: len(sizes)], sizes, strict=True):
            candidate_nodes, pair = _candidate(name, size)
            nodes += candidate_nodes
            pairs.append(pair)
        assert _anchor_of(nodes, pairs) == expected

    def test_two_figures_under_a_root_give_no_body(self):
        # The second figure numbered before the end sites' suffix.
        second = [(_numbered(name), _numbered(parent)) for name, parent in _BODY]
        nodes = [("Root", None), ("Hips", "Root"), *_BODY, ("Hips2", "Root"), *second]
        pairs = [_LEGS, ("LeftUpLeg2", "RightUpLeg2")]
        assert _anchor_of(nodes, pairs) == (None, "ambiguous", ["Hips", "Hips2"])


class TestBodyAnchorUnderAStaticRoot:
    """Under a static root (``CHANNELS 0``), each child of the root heads a
    tree: the largest tree is kept, with the pairs inside it, unless a
    second tree has at least half its joints."""

    def test_ik_targets_beside_the_body_give_the_body_s_anchor(self):
        # Blender's __0 above two IK targets and the hips. The targets'
        # pair meets at the static root, at depth 0, but it lies across
        # two trees and is not counted.
        nodes = [
            ("__0", None),
            ("LeftFootIK", "__0"),
            ("LeftFootIK_End", "LeftFootIK"),
            ("RightFootIK", "__0"),
            ("RightFootIK_End", "RightFootIK"),
            ("Hips", "__0"),
            *_BODY,
        ]
        pairs = [("LeftFootIK", "RightFootIK"), _LEGS, _ARMS]
        assert _anchor_of(nodes, pairs, static_root=True) == ("Hips", "lr_pairs", ["Hips"])

    @pytest.mark.parametrize(
        ("sizes", "expected"),
        [
            # A second tree of fewer than half the joints: the largest tree.
            ((20, 9), ("A", "lr_pairs", ["A"])),
            # At least half is the tie: no body, the candidate of each tree
            # compared listed, the largest tree first.
            ((20, 10), (None, "ambiguous", ["A", "B"])),
            ((10, 20), (None, "ambiguous", ["B", "A"])),
            # A third tree under half is not one of the figures compared.
            ((20, 12, 3), (None, "ambiguous", ["A", "B"])),
        ],
        ids=["20-9", "20-10", "10-20", "20-12-3"],
    )
    def test_a_comparable_second_tree_gives_no_body(self, sizes, expected):
        # Trees A, B, C under a static root, each of the given size, each
        # with a pair that meets at its head.
        nodes, pairs = [("Root", None)], []
        for name, size in zip("ABC"[: len(sizes)], sizes, strict=True):
            candidate_nodes, pair = _candidate(name, size)
            nodes += candidate_nodes
            pairs.append(pair)
        assert _anchor_of(nodes, pairs, static_root=True) == expected

    def test_the_candidate_of_a_tree_is_where_its_pairs_meet(self):
        # Two figures, each under a bone above its hips: the candidates are
        # the hips, not the trees' heads.
        second = [(_numbered(name), _numbered(parent)) for name, parent in _BODY]
        nodes = [
            ("__0", None),
            ("Armature", "__0"),
            ("Hips", "Armature"),
            *_BODY,
            ("Armature2", "__0"),
            ("Hips2", "Armature2"),
            *second,
        ]
        pairs = [_LEGS, ("LeftUpLeg2", "RightUpLeg2")]
        assert _anchor_of(nodes, pairs, static_root=True) == (None, "ambiguous", ["Hips", "Hips2"])

    def test_a_pair_across_two_trees_is_not_counted(self):
        # The only pair has a joint in each tree: the kept tree holds no
        # pair, so the anchor is the root, by default.
        nodes = [
            ("__0", None),
            ("LeftFootIK", "__0"),
            ("LeftFootIK_End", "LeftFootIK"),
            ("RightFootIK", "__0"),
            ("RightFootIK_End", "RightFootIK"),
            ("Hips", "__0"),
            *_BODY,
        ]
        pairs = [("LeftFootIK", "RightFootIK")]
        assert _anchor_of(nodes, pairs, static_root=True) == ("__0", "default", [])

    def test_the_same_children_under_a_root_with_channels_go_through_step_6(self):
        # Tree A holds 20 joints, its pair meeting below its head, at depth
        # 2; tree B holds 8, its pair meeting at its head, at depth 1.
        # Under a static root A is kept; under a root with channels the
        # shallowest meeting joint is B's.
        a_rows, _ = _candidate("A", 20)
        a_rows += [("ALimbs", "A"), ("LeftLimb", "ALimbs"), ("LeftLimb_End", "LeftLimb")]
        a_rows += [("RightLimb", "ALimbs"), ("RightLimb_End", "RightLimb")]
        b_rows, b_pair = _candidate("B", 8)
        nodes = [("Root", None), *a_rows, *b_rows]
        pairs = [("LeftLimb", "RightLimb"), b_pair]
        assert _anchor_of(nodes, pairs, static_root=True) == ("ALimbs", "lr_pairs", ["ALimbs"])
        assert _anchor_of(nodes, pairs) == ("B", "lr_pairs", ["B"])

    def test_two_comparable_trees_under_a_root_with_channels_go_through_step_6(self):
        # The two figures' pairs meet at their hips, at one depth: no body
        # by step 6, as under a static root by step 3.
        nodes, pairs = [("Root", None)], []
        for name in "AB":
            candidate_nodes, pair = _candidate(name, 20)
            nodes += candidate_nodes
            pairs.append(pair)
        assert _anchor_of(nodes, pairs) == (None, "ambiguous", ["A", "B"])


def _numbered(name):
    """*name* with a 2 after its joint's name: ``LeftFoot_End`` gives
    ``LeftFoot2_End``."""
    base, end, suffix = name.partition("_End")
    return f"{base}2{end}{suffix}"


def _candidate(name, size):
    """Rows of a joint *name* under ``Root`` holding a left/right pair of
    leaves and a chain, *size* joints in all (*name* included, the end
    sites not), and the pair."""
    left, right = f"Left{name}", f"Right{name}"
    rows = [
        (name, "Root"),
        (left, name),
        (f"{left}_End", left),
        (right, name),
        (f"{right}_End", right),
    ]
    parent = name
    for k in range(size - 3):
        rows.append((f"{name}{k}", parent))
        parent = f"{name}{k}"
    rows.append((f"{parent}_End", parent))
    return rows, (left, right)


def _body_of(nodes, anchor):
    """The body the anchor named *anchor* gives, as the names of its nodes
    in node order; end sites are the names ending in ``_End``."""
    skeleton = _tree(nodes, end_sites=[n for n, _ in nodes if n.endswith("_End")])
    names = skeleton.node_names
    in_body = _find_body(skeleton, names.index(anchor))
    return [name for name, inside in zip(names, in_body, strict=True) if inside]


# A torso whose chest holds two arms and a head, and two legs, each to be
# hung under a joint of the test's choosing.
_TORSO = [
    ("Spine", "Root"),
    ("Chest", "Spine"),
    ("LeftArm", "Chest"),
    ("RightArm", "Chest"),
    ("Neck", "Chest"),
    ("Head", "Neck"),
    ("Head_End", "Head"),
]
_LEG_ROWS = [
    ("LeftUpLeg", "Hips"),
    ("LeftFoot", "LeftUpLeg"),
    ("LeftFoot_End", "LeftFoot"),
    ("RightUpLeg", "Hips"),
    ("RightFoot", "RightUpLeg"),
    ("RightFoot_End", "RightFoot"),
]


class TestBody:
    """The body: the anchor and every node below it, started higher when
    a head name or a foot name is missing below the anchor."""

    def test_the_anchor_and_every_node_below_it(self):
        # A root above the hips stays outside the body.
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _body_of(nodes, "Hips") == ["Hips", *(name for name, _ in _BODY)]

    def test_a_missing_head_name_starts_the_body_where_one_is_below(self):
        nodes = [("Prop", None), ("Root", "Prop"), *_TORSO, ("Hips", "Root"), *_LEG_ROWS]
        assert _body_of(nodes, "Hips") == [name for name, _ in nodes[1:]]

    def test_a_missing_foot_name_starts_the_body_where_one_is_below(self):
        # The arms meet at the chest: the anchor holds the head, not the feet.
        nodes = [("Prop", None), ("Root", "Prop"), *_TORSO, ("Hips", "Root"), *_LEG_ROWS]
        assert _body_of(nodes, "Chest") == [name for name, _ in nodes[1:]]

    def test_both_missing_start_the_body_at_the_higher_joint(self):
        # The head name is one joint above the anchor, the foot name two.
        nodes = [
            ("Prop", None),
            ("Top", "Prop"),
            ("Base", "Top"),
            ("LeftToe", "Base"),
            ("RightToe", "Base"),
            ("Middle", "Top"),
            ("Neck", "Middle"),
            ("Arms", "Middle"),
            ("LeftHand", "Arms"),
            ("RightHand", "Arms"),
        ]
        assert _body_of(nodes, "Arms") == [name for name, _ in nodes[1:]]

    def test_the_closest_joint_above_with_the_name_below(self):
        # The neck is below the middle joint, which the climb stops at.
        nodes = [
            ("Top", None),
            ("Spine", "Top"),
            ("Middle", "Top"),
            ("Neck", "Middle"),
            ("Arms", "Middle"),
            ("LeftFoot", "Arms"),
            ("RightFoot", "Arms"),
        ]
        assert _body_of(nodes, "Arms") == ["Middle", "Neck", "Arms", "LeftFoot", "RightFoot"]

    def test_no_joint_above_with_the_missing_name(self):
        nodes = [("Reference", None), ("Prop", "Reference"), ("Hips", "Reference"), *_LEG_ROWS]
        assert _body_of(nodes, "Hips") == ["Hips", *(name for name, _ in _LEG_ROWS)]

    def test_names_are_read_as_words(self):
        # "Footsteps" holds "foot" but is another word, so the hips, which
        # hold "LeftLeg", are not left for the root.
        nodes = [
            ("Bip01", None),
            ("Footsteps", "Bip01"),
            ("Hips", "Bip01"),
            ("Spine", "Hips"),
            ("LeftLeg", "Hips"),
            ("RightLeg", "Hips"),
        ]
        assert _body_of(nodes, "Hips") == ["Hips", "Spine", "LeftLeg", "RightLeg"]

    def test_the_names_of_end_sites_are_not_read(self):
        nodes = [("Root", None), ("Head", "Root"), ("Hips", "Root"), ("LeftFoot", "Hips")]
        nodes += [("Spine_End", "Hips")]
        assert _body_of(nodes, "Hips") == ["Root", "Head", "Hips", "LeftFoot", "Spine_End"]


def _size_reference_of(nodes, feet, anchor):
    """The size reference of the feet named *feet*, as a name, on the tree
    of *nodes* whose body anchor is the joint named *anchor*."""
    skeleton = _tree(nodes)
    names = skeleton.node_names
    found = _size_reference(
        skeleton.parent_idx, [names.index(foot) for foot in feet], names.index(anchor)
    )
    return names[found]


class TestSizeReference:
    """The joint the sizes are read from: the lowest joint that has the
    feet in two or more of its branches, else the body anchor."""

    def test_two_feet_give_the_joint_where_the_legs_meet(self):
        # A root above the hips is passed over.
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _size_reference_of(nodes, _FEET, "Hips") == "Hips"

    def test_the_four_feet_of_a_quadruped_give_where_the_hind_and_front_chains_part(self):
        nodes = [
            ("Root", None),
            ("Pelvis", "Root"),
            ("LeftHindFoot", "Pelvis"),
            ("RightHindFoot", "Pelvis"),
            ("Spine", "Pelvis"),
            ("Chest", "Spine"),
            ("LeftFrontFoot", "Chest"),
            ("RightFrontFoot", "Chest"),
        ]
        feet = ["LeftHindFoot", "RightHindFoot", "LeftFrontFoot", "RightFrontFoot"]
        assert _size_reference_of(nodes, feet, "Chest") == "Pelvis"

    def test_one_foot_gives_the_body_anchor(self):
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _size_reference_of(nodes, ["LeftFoot"], "Hips") == "Hips"

    def test_a_foot_and_its_toe_on_one_leg_give_the_body_anchor(self):
        nodes = [("Reference", None), ("Hips", "Reference"), *_BODY]
        assert _size_reference_of(nodes, ["LeftUpLeg", "LeftFoot"], "Hips") == "Hips"

    def test_a_foot_and_a_helper_beside_the_body_give_their_meeting_joint(self):
        # Given by hand, as asked: the helper hangs from the root.
        nodes = [("Bip01", None), ("Footsteps", "Bip01"), ("Hips", "Bip01"), *_BODY]
        assert _size_reference_of(nodes, ["Footsteps", "LeftFoot"], "Hips") == "Bip01"


class TestReadInTheBody:
    """The head joint and the L/R pairs of rest left are read in the body
    alone: a prop beside it is passed over."""

    def test_a_head_name_outside_the_body_is_not_the_head_joint(self):
        # The prop hangs from the reference root, beside the hips; the
        # body's head lies on the hips, so its neck is the head joint.
        names = list(_REFERENCE_NAMES)
        names[5] = "HeadLamp"
        skeleton = _skeleton(names, _REFERENCE_REST, parents=_REFERENCE_PARENTS)
        assert _up_marker_index(skeleton, _HIPS) == 3

    def test_rest_left_reads_the_pairs_with_both_joints_in_the_body(self):
        # A pair of props beside the body lies far along -x, the legs
        # along +x.
        names = ["Reference", "Hips", "Head", "LeftLeg", "RightLeg", "LeftProp", "RightProp"]
        rest = [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 19.0, 0.0],
            [2.0, 5.0, 0.0],
            [-2.0, 5.0, 0.0],
            [-50.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ]
        parents = [-1, 0, 1, 1, 1, 0, 0]
        skeleton = _skeleton(names, rest, [[3, 4], [5, 6]], parents=parents)
        assert _infer_rest_up_and_left(skeleton, 1) == (_RestUp("+y", "named_joint"), "+x")


# The body anchor of a skeleton without pairs: its root.
_NO_ANCHOR = _BodyAnchor(0, "default", ())


class TestReport:
    """The skeleton's own frame read from the rest pose, beside a world up
    the caller settled, each fact with its source."""

    _PAIRS = np.array([[3, 4]])  # LeftHand, RightHand

    def test_a_measured_skeleton(self):
        # The hands meet at the root. Names stand for the clip's nodes.
        report = _build_orientation_report(
            _skeleton(_NAMES, _REST, self._PAIRS),
            _WorldUpGuess("+z", "first_frame"),
            False,
            "names",
            _BodyAnchor(0, "lr_pairs", (0,)),
            _NAMES,
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
            body_anchor="Hips",
            body_anchor_source="lr_pairs",
            body_anchor_candidates=("Hips",),
        )

    def test_a_skeleton_with_no_body(self):
        report = _build_orientation_report(
            _skeleton(_NAMES, _REST, self._PAIRS),
            "+y",
            False,
            "names",
            _BodyAnchor(None, "ambiguous", (2, 1)),
            _NAMES,
        )
        assert (
            report.body_anchor,
            report.body_anchor_source,
            report.body_anchor_candidates,
        ) == (None, "ambiguous", (_NAMES[2], _NAMES[1]))

    def test_an_axis_the_user_set(self):
        report = _build_orientation_report(
            _skeleton(_NAMES, _REST, self._PAIRS), "-z", False, "set", _NO_ANCHOR, _NAMES
        )
        assert (report.world_up, report.world_up_source, report.world_up_inherited) == (
            "-z",
            "set",
            False,
        )

    def test_rest_up_from_the_leg_pair(self):
        skeleton = _skeleton(
            _NAMELESS_NAMES, _NAMELESS_REST, _NAMELESS_PAIRS, parents=_NAMELESS_PARENTS
        )
        report = _build_orientation_report(
            skeleton, "+y", False, "set", _BodyAnchor(0, "lr_pairs", (0,)), _NAMELESS_NAMES
        )
        assert (report.rest_up, report.rest_up_source) == ("+y", "leg_pair")
        # Left +x (both pairs' left sides lie along +x) x up +y.
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "lr_geometry")

    def test_rest_forward_defaulted_without_pairs(self):
        report = _build_orientation_report(
            _skeleton(_NAMES, _REST),
            _WorldUpGuess("+y", "rest_up"),
            True,
            None,
            _NO_ANCHOR,
            _NAMES,
        )
        assert report.rest_left is None
        # The house default for rest up +y.
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "default")

    def test_nothing_measured_on_a_degenerate_rest_pose(self):
        report = _build_orientation_report(
            _skeleton(["Root"], np.zeros((1, 3))),
            _WorldUpGuess("+y", "default"),
            False,
            None,
            _NO_ANCHOR,
            ["Root"],
        )
        assert (report.rest_up, report.rest_up_source, report.rest_left) == (None, None, None)
        # The house default keyed on the house up '+y'.
        assert (report.rest_forward, report.rest_forward_source) == ("+z", "default")
