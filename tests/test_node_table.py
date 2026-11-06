"""Tests for the node table: `nodes_to_table` and `nodes_from_table`.

The table is the positional twin of `FkTopology`: one entry per node in
`nodes` order, parents referenced by index. Position is identity, so the
rigs of `test_name_collisions.py` (duplicate joint names, two end sites
under one joint) must come back exactly. The builder is where a node
tree is validated once; every rule it enforces has a test here, and each
message names the entry it rejects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pybvh import (
    Bvh,
    FkTopology,
    nodes_from_table,
    nodes_to_table,
    read_bvh_file,
)
from pybvh.bvhnode import BvhEndSite, BvhJoint, BvhRoot

BVH_DATA = Path(__file__).parent.parent / "bvh_data"
FIXTURES = Path(__file__).parent / "fixtures"
BUNDLED_FILES = sorted(BVH_DATA.glob("*.bvh")) + [
    FIXTURES / "rotation_first_root.bvh",
    FIXTURES / "full_precision_frame_time.bvh",
]


@pytest.fixture
def bvh_example():
    return read_bvh_file(BVH_DATA / "bvh_test1.bvh")


def _attach(parent, child):
    parent.children = parent.children + [child]
    child.parent = parent
    return child


def _two_end_sites_rig():
    """The #16 rig: one joint carrying two end sites with one display name."""
    root = BvhRoot("Hips", [0, 0, 0], "XYZ", "ZYX", [])
    hand = _attach(root, BvhJoint("Hand", [0, 1, 0], "ZYX", []))
    _attach(hand, BvhEndSite("EndSiteHand", [1, 0, 0]))
    _attach(hand, BvhEndSite("EndSiteHand", [0, 0, 2]))
    return [root, hand, hand.children[0], hand.children[1]]


def _collision_rig():
    """A joint named like another joint's generated end site (see
    `test_name_collisions.py`)."""
    root = BvhRoot("Hips", [0, 0, 0], "XYZ", "ZYX", [])
    shadowed = _attach(root, BvhJoint("EndSiteHips", [1, 0, 0], "ZYX", []))
    child = _attach(shadowed, BvhJoint("Child", [0, 1, 0], "ZYX", []))
    _attach(child, BvhEndSite("EndSiteChild", [0, 1, 0]))
    _attach(root, BvhEndSite("EndSiteHips", [0, 0, 1]))
    return [root, shadowed, child, child.children[0], root.children[1]]


def _duplicate_joint_rig():
    """Two sibling joints sharing the name 'Arm'."""
    root = BvhRoot("Hips", [0, 0, 0], "XYZ", "ZYX", [])
    first = _attach(root, BvhJoint("Arm", [1, 0, 0], "ZYX", []))
    grandchild = _attach(first, BvhJoint("ArmChild", [0, 1, 0], "ZYX", []))
    _attach(grandchild, BvhEndSite("EndSiteArmChild", [0, 1, 0]))
    second = _attach(root, BvhJoint("Arm", [-1, 0, 0], "XYZ", []))
    _attach(second, BvhEndSite("EndSiteArm", [0, 1, 0]))
    return [root, first, grandchild, grandchild.children[0], second, second.children[0]]


def _assert_same_topology(actual: FkTopology, expected: FkTopology):
    np.testing.assert_array_equal(actual.offsets, expected.offsets)
    np.testing.assert_array_equal(actual.parent_idx, expected.parent_idx)
    np.testing.assert_array_equal(actual.joint_idx, expected.joint_idx)
    assert actual.euler_orders == expected.euler_orders


def _as_bvh(nodes):
    joint_count = sum(1 for n in nodes if not n.is_end_site())
    return Bvh(nodes, np.zeros((2, 3)), np.zeros((2, joint_count, 3)), 1 / 30)


def _two_end_sites_table():
    return [
        {
            "name": "Hips",
            "parent": None,
            "offset": np.array([0.0, 0.0, 0.0]),
            "pos_channels": ["X", "Y", "Z"],
            "rot_channels": ["Z", "Y", "X"],
        },
        {
            "name": "Hand",
            "parent": 0,
            "offset": np.array([0.0, 1.0, 0.0]),
            "rot_channels": ["Z", "Y", "X"],
        },
        {"name": "EndSiteHand", "parent": 1, "offset": np.array([1.0, 0.0, 0.0])},
        {"name": "EndSiteHand", "parent": 1, "offset": np.array([0.0, 0.0, 2.0])},
    ]


# =============================================================================
# nodes_to_table: the format
# =============================================================================


class TestNodesToTable:
    def test_two_end_sites_rig_matches_the_documented_table(self):
        table = nodes_to_table(_two_end_sites_rig())
        expected = _two_end_sites_table()
        assert len(table) == len(expected)
        for entry, want in zip(table, expected, strict=True):
            assert list(entry) == list(want)
            for key in want:
                if key == "offset":
                    np.testing.assert_array_equal(entry[key], want[key])
                else:
                    assert entry[key] == want[key]

    def test_offsets_are_copies(self):
        nodes = _two_end_sites_rig()
        table = nodes_to_table(nodes)
        table[1]["offset"][0] = 99.0
        table[1]["rot_channels"][0] = "Q"
        assert nodes[1].offset[0] == 0.0
        assert nodes[1].rot_channels[0] == "Z"

    def test_bvh_example_entries(self, bvh_example):
        table = nodes_to_table(bvh_example.nodes)
        assert len(table) == len(bvh_example.nodes)
        assert table[0]["parent"] is None
        assert "pos_channels" in table[0]
        for entry, node in zip(table[1:], bvh_example.nodes[1:], strict=True):
            assert entry["parent"] == bvh_example.nodes.index(node.parent)
            assert "pos_channels" not in entry
            assert ("rot_channels" in entry) == (not node.is_end_site())

    def test_parent_outside_the_list_raises(self, bvh_example):
        with pytest.raises(ValueError, match="not in the node list"):
            nodes_to_table(bvh_example.nodes[1:])


# =============================================================================
# Round trip
# =============================================================================


class TestRoundTrip:
    """`nodes_from_table(nodes_to_table(nodes))` rebuilds the tree."""

    @pytest.fixture(
        params=[
            pytest.param(_two_end_sites_rig, id="two_end_sites"),
            pytest.param(_collision_rig, id="collision"),
            pytest.param(_duplicate_joint_rig, id="duplicate_joint"),
        ]
    )
    def rig(self, request):
        return _as_bvh(request.param())

    @pytest.mark.parametrize("path", BUNDLED_FILES, ids=lambda p: p.name)
    def test_every_bundled_file(self, path):
        """What the reader builds, the table carries and the builder rebuilds."""
        bvh = read_bvh_file(path)
        rebuilt = nodes_from_table(nodes_to_table(bvh.nodes))
        rebuilt_bvh = Bvh(rebuilt, bvh.root_pos, bvh.joint_angles, bvh.frame_time)
        assert [n.name for n in rebuilt] == [n.name for n in bvh.nodes]
        assert rebuilt_bvh.matches_hierarchy(bvh, atol=0)
        assert rebuilt_bvh.matches_channels(bvh)
        _assert_same_topology(FkTopology.from_nodes(rebuilt), bvh.fk_topology)

    def test_rigs_with_repeated_names(self, rig):
        rebuilt = nodes_from_table(nodes_to_table(rig.nodes))
        rebuilt_bvh = Bvh(rebuilt, rig.root_pos, rig.joint_angles, rig.frame_time)
        assert rebuilt_bvh.matches_hierarchy(rig)
        assert rebuilt_bvh.matches_channels(rig)
        _assert_same_topology(FkTopology.from_nodes(rebuilt), rig.fk_topology)

    def test_result_shares_no_node_with_the_input(self, bvh_example):
        rebuilt = nodes_from_table(nodes_to_table(bvh_example.nodes))
        originals = {id(node) for node in bvh_example.nodes}
        assert not any(id(node) in originals for node in rebuilt)

    def test_children_are_wired_in_table_order(self):
        nodes = nodes_from_table(_two_end_sites_table())
        root, hand, first_end, second_end = nodes
        assert root.children == [hand]
        assert hand.children == [first_end, second_end]
        assert first_end.parent is hand and second_end.parent is hand
        assert root.parent is None

    def test_node_kinds(self):
        nodes = nodes_from_table(_two_end_sites_table())
        assert type(nodes[0]) is BvhRoot
        assert type(nodes[1]) is BvhJoint
        assert type(nodes[2]) is BvhEndSite
        assert nodes[0].pos_channels == ("X", "Y", "Z")
        assert nodes[1].rot_channels == ("Z", "Y", "X")

    def test_table_is_not_shared_with_the_result(self):
        table = _two_end_sites_table()
        nodes = nodes_from_table(table)
        table[1]["offset"][1] = 99.0
        table[1]["rot_channels"][0] = "Q"
        assert nodes[1].offset[1] == 1.0
        assert nodes[1].rot_channels == ("Z", "Y", "X")


# =============================================================================
# Validation: every rule names the entry it rejects
# =============================================================================


class TestNodesFromTableRejects:
    def test_root_with_a_parent(self):
        table = _two_end_sites_table()
        table[0]["parent"] = 1
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*parent"):
            nodes_from_table(table)

    def test_second_root(self):
        table = _two_end_sites_table()
        table[1]["parent"] = None
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*one root"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_parent", [2, 3, 7, -1])
    def test_parent_index_outside_the_entries_before(self, bad_parent):
        table = _two_end_sites_table()
        table[2]["parent"] = bad_parent
        with pytest.raises(ValueError, match=r"entry 2 \('EndSiteHand'\).*\[0, 2\)"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_parent", ["Hips", 1.0, True])
    def test_parent_that_is_not_an_index(self, bad_parent):
        table = _two_end_sites_table()
        table[1]["parent"] = bad_parent
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*not an int"):
            nodes_from_table(table)

    def test_parent_that_is_an_end_site(self):
        table = _two_end_sites_table()
        table[3]["parent"] = 2
        with pytest.raises(ValueError, match=r"entry 3 \('EndSiteHand'\).*entry 2.*end site"):
            nodes_from_table(table)

    def test_root_without_rot_channels(self):
        table = _two_end_sites_table()
        del table[0]["rot_channels"]
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*rot_channels"):
            nodes_from_table(table)

    def test_breadth_first_order(self):
        """Hips, Arm, Leg, EndSiteArm, EndSiteLeg: valid parents, wrong order."""
        table = [
            {"name": "Hips", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
            {"name": "Arm", "parent": 0, "offset": [1, 0, 0], "rot_channels": "ZYX"},
            {"name": "Leg", "parent": 0, "offset": [0, -1, 0], "rot_channels": "ZYX"},
            {"name": "EndSiteArm", "parent": 1, "offset": [1, 0, 0]},
            {"name": "EndSiteLeg", "parent": 2, "offset": [0, -1, 0]},
        ]
        with pytest.raises(
            ValueError, match=r"entry 2 \('Leg'\).*depth-first.*entry 3 \('EndSiteArm'\)"
        ):
            nodes_from_table(table)

    @pytest.mark.parametrize(
        "bad_offset", [[1.0, 2.0], "abc", None, [None, 0.0, 0.0], [np.nan, 0.0, 0.0]]
    )
    def test_malformed_offset(self, bad_offset):
        """A JSON null component is a None that NumPy would turn into nan."""
        table = _two_end_sites_table()
        table[1]["offset"] = bad_offset
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*offset"):
            nodes_from_table(table)

    def test_missing_offset(self):
        table = _two_end_sites_table()
        del table[1]["offset"]
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*offset"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_channels", ["ZYQ", ["Z", "Y"], "XXYZ", 3, ["XY", "Z"], ["XYZ"]])
    def test_malformed_rot_channels(self, bad_channels):
        table = _two_end_sites_table()
        table[1]["rot_channels"] = bad_channels
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*channels"):
            nodes_from_table(table)

    def test_malformed_pos_channels(self):
        table = _two_end_sites_table()
        table[0]["pos_channels"] = "XYQ"
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*channels"):
            nodes_from_table(table)

    @pytest.mark.parametrize(
        "key, index, name", [("rot_channels", 1, "Hand"), ("pos_channels", 0, "Hips")]
    )
    def test_channels_none_is_not_a_default(self, key, index, name):
        """A JSON null must not fall back to the node constructors' ZYX / XYZ."""
        table = _two_end_sites_table()
        table[index][key] = None
        with pytest.raises(ValueError, match=rf"entry {index} \('{name}'\).*{key} None"):
            nodes_from_table(table)

    @pytest.mark.parametrize("empty", [[], (), ""])
    def test_empty_pos_channels_on_a_joint(self, empty):
        """One spelling per state: a joint that only rotates omits the key."""
        table = _two_end_sites_table()
        table[1]["pos_channels"] = empty
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\) has pos_channels \[\].*omits"):
            nodes_from_table(table)

    def test_pos_channels_on_an_end_site(self):
        table = _two_end_sites_table()
        table[2]["pos_channels"] = ["X", "Y", "Z"]
        with pytest.raises(ValueError, match=r"entry 2 \('EndSiteHand'\).*pos_channels.*end site"):
            nodes_from_table(table)

    def test_joint_without_a_name(self):
        table = _two_end_sites_table()
        del table[1]["name"]
        with pytest.raises(ValueError, match=r"entry 1\b.*name"):
            nodes_from_table(table)

    def test_name_that_is_not_a_string(self):
        table = _two_end_sites_table()
        table[1]["name"] = 7
        with pytest.raises(ValueError, match=r"entry 1\b.*name.*string"):
            nodes_from_table(table)

    def test_unknown_key(self):
        """A misspelt `rot_channels` must not silently make an end site."""
        table = _two_end_sites_table()
        table[1]["rot_channel"] = table[1].pop("rot_channels")
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*'rot_channel'"):
            nodes_from_table(table)

    def test_children_key_from_the_hierarchy_dict(self):
        table = _two_end_sites_table()
        table[1]["children"] = ["EndSiteHand"]
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*'children'"):
            nodes_from_table(table)

    def test_entry_that_is_not_a_dict(self):
        table = _two_end_sites_table()
        table[2] = ["EndSiteHand", 1, [1, 0, 0]]
        with pytest.raises(ValueError, match=r"entry 2\b.*dict"):
            nodes_from_table(table)

    def test_empty_table(self):
        with pytest.raises(ValueError, match="at least one entry"):
            nodes_from_table([])


# =============================================================================
# Leniency: what a hand-written table may leave out
# =============================================================================


class TestNodesFromTableLeniency:
    def test_root_pos_channels_default_to_xyz(self):
        table = _two_end_sites_table()
        del table[0]["pos_channels"]
        nodes = nodes_from_table(table)
        assert nodes[0].pos_channels == ("X", "Y", "Z")

    def test_end_site_name_defaults_to_the_parser_rule(self):
        table = _two_end_sites_table()
        del table[2]["name"]
        del table[3]["name"]
        nodes = nodes_from_table(table)
        assert [node.name for node in nodes[2:]] == ["EndSiteHand", "EndSiteHand"]

    def test_channel_strings_are_accepted(self):
        """'ZYX' and ['Z', 'Y', 'X'] both name the order, as on the nodes."""
        table = _two_end_sites_table()
        table[0]["rot_channels"] = "ZYX"
        table[0]["pos_channels"] = "XYZ"
        table[1]["rot_channels"] = "XYZ"
        nodes = nodes_from_table(table)
        assert nodes[0].rot_channels == ("Z", "Y", "X")
        assert nodes[1].rot_channels == ("X", "Y", "Z")

    def test_offsets_may_be_lists(self):
        table = _two_end_sites_table()
        table[1]["offset"] = [0, 1, 0]
        nodes = nodes_from_table(table)
        np.testing.assert_array_equal(nodes[1].offset, [0.0, 1.0, 0.0])
        assert nodes[1].offset.dtype == np.float64

    def test_root_may_omit_its_parent_key(self):
        table = _two_end_sites_table()
        del table[0]["parent"]
        assert nodes_from_table(table)[0].parent is None


# =============================================================================
# The entry table of #186: position channels on any joint, a static root
# =============================================================================


def _translating_joint_table():
    """The two-end-sites rig whose joint 'Hand' declares position channels."""
    table = _two_end_sites_table()
    table[1] = {
        "name": "Hand",
        "parent": 0,
        "offset": np.array([0.0, 1.0, 0.0]),
        "pos_channels": ["X", "Y", "Z"],
        "rot_channels": ["Z", "Y", "X"],
    }
    return table


def _translating_joint_rig():
    root = BvhRoot("Hips", [0, 0, 0], "XYZ", "ZYX", [])
    hand = _attach(root, BvhJoint("Hand", [0, 1, 0], "ZYX", [], pos_channels="XYZ"))
    _attach(hand, BvhEndSite("EndSiteHand", [1, 0, 0]))
    _attach(hand, BvhEndSite("EndSiteHand", [0, 0, 2]))
    return [root, hand, hand.children[0], hand.children[1]]


class TestTranslatingJoint:
    """A non-root joint with ``pos_channels: ['X', 'Y', 'Z']`` and an order."""

    def test_the_table_builds_it(self):
        hand = nodes_from_table(_translating_joint_table())[1]
        assert type(hand) is BvhJoint
        assert hand.pos_channels == ("X", "Y", "Z")
        assert hand.rot_channels == ("Z", "Y", "X")

    def test_a_hand_built_node_list_holds_it(self):
        bvh = Bvh(_translating_joint_rig())
        assert bvh.nodes[1].pos_channels == ("X", "Y", "Z")

    def test_the_table_round_trips(self):
        table = nodes_to_table(nodes_from_table(_translating_joint_table()))
        assert table[1]["pos_channels"] == ["X", "Y", "Z"]
        assert list(table[1]) == ["name", "parent", "offset", "pos_channels", "rot_channels"]

    def test_bvh_to_node_table_writes_it(self):
        table = Bvh(_translating_joint_rig()).to_node_table()
        assert table[1]["pos_channels"] == ["X", "Y", "Z"]


def _static_root_table():
    """A CHANNELS 0 root above the two-end-sites rig's joint."""
    table = _two_end_sites_table()
    table[0]["pos_channels"] = []
    table[0]["rot_channels"] = []
    return table


def _static_root_rig():
    root = BvhRoot("Hips", [0, 0, 0], (), (), [])
    hand = _attach(root, BvhJoint("Hand", [0, 1, 0], "ZYX", []))
    _attach(hand, BvhEndSite("EndSiteHand", [1, 0, 0]))
    _attach(hand, BvhEndSite("EndSiteHand", [0, 0, 2]))
    return [root, hand, hand.children[0], hand.children[1]]


class TestStaticRoot:
    """A root with ``pos_channels: []`` and ``rot_channels: []`` (CHANNELS 0)."""

    def test_the_table_builds_it(self):
        root = nodes_from_table(_static_root_table())[0]
        assert type(root) is BvhRoot
        assert root.pos_channels == ()
        assert root.rot_channels == ()
        assert root.is_static()

    def test_a_root_with_channels_is_not_static(self):
        assert not nodes_from_table(_two_end_sites_table())[0].is_static()

    def test_a_hand_built_node_list_holds_it(self):
        bvh = Bvh(_static_root_rig())
        assert bvh.root.is_static()

    def test_the_table_round_trips(self):
        table = nodes_to_table(nodes_from_table(_static_root_table()))
        assert table[0]["pos_channels"] == []
        assert table[0]["rot_channels"] == []

    def test_bvh_to_node_table_writes_it(self):
        table = Bvh(_static_root_rig()).to_node_table()
        assert table[0]["pos_channels"] == [] and table[0]["rot_channels"] == []


class TestEntryTableRefusals:
    """Each refusal of the entry table names the entry, or the node."""

    @pytest.mark.parametrize(
        "pos, rot",
        [
            pytest.param([], ["Z", "Y", "X"], id="pos_empty"),
            pytest.param(["X", "Y", "Z"], [], id="rot_empty"),
            pytest.param(None, [], id="rot_empty_pos_absent"),
        ],
    )
    def test_a_root_with_one_order_empty(self, pos, rot):
        table = _two_end_sites_table()
        if pos is None:
            del table[0]["pos_channels"]
        else:
            table[0]["pos_channels"] = pos
        table[0]["rot_channels"] = rot
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*both.*or neither"):
            nodes_from_table(table)

    @pytest.mark.parametrize("pos, rot", [((), "ZYX"), ("XYZ", ())])
    def test_a_hand_built_root_with_one_order_empty(self, pos, rot):
        with pytest.raises(ValueError, match=r"Root 'Hips'.*both.*or neither"):
            BvhRoot("Hips", [0, 0, 0], pos, rot, [])

    def test_a_root_emptied_by_one_assignment_is_refused_by_bvh(self):
        """The setters let a root pass through one order empty; Bvh does not."""
        nodes = _two_end_sites_rig()
        nodes[0].rot_channels = ()
        with pytest.raises(ValueError, match=r"Root 'Hips'.*both.*or neither"):
            Bvh(nodes)

    def test_a_root_turned_static_by_two_assignments(self):
        nodes = _two_end_sites_rig()
        nodes[0].rot_channels = ()
        nodes[0].pos_channels = ()
        assert Bvh(nodes).root.is_static()

    def test_a_joint_without_rotation_channels(self):
        table = _translating_joint_table()
        table[1]["rot_channels"] = []
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*channels"):
            nodes_from_table(table)

    @pytest.mark.parametrize("empty", [[], ()])
    def test_empty_pos_channels_on_an_end_site(self, empty):
        table = _two_end_sites_table()
        table[2]["pos_channels"] = empty
        with pytest.raises(ValueError, match=r"entry 2 \('EndSiteHand'\).*pos_channels.*end site"):
            nodes_from_table(table)

    @pytest.mark.parametrize("index, name", [(0, "Hips"), (1, "Hand")])
    @pytest.mark.parametrize("order", ["ZYX", ["Y", "X", "Z"]])
    def test_a_position_order_other_than_xyz(self, index, name, order):
        table = _translating_joint_table()
        table[index]["pos_channels"] = order
        with pytest.raises(
            ValueError,
            match=rf"entry {index} \('{name}'\).*position channel order .* is not supported\. "
            rf"Expected 'XYZ'",
        ):
            nodes_from_table(table)

    def test_a_hand_built_joint_with_a_position_order_other_than_xyz(self):
        with pytest.raises(
            ValueError, match=r"order 'ZXY' on joint 'Hand' is not supported\. Expected 'XYZ'"
        ):
            BvhJoint("Hand", [0, 1, 0], "ZYX", [], pos_channels="ZXY")

    def test_a_hand_built_root_with_a_position_order_other_than_xyz(self):
        with pytest.raises(
            ValueError,
            match=r"root position channel order 'ZYX' is not supported\. Expected 'XYZ'.*'Hips'",
        ):
            BvhRoot("Hips", pos_channels="ZYX")

    def test_pos_channels_none_on_a_joint(self):
        table = _translating_joint_table()
        table[1]["pos_channels"] = None
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*pos_channels None"):
            nodes_from_table(table)


def _as_json(table):
    return json.dumps(
        [{**entry, "offset": np.asarray(entry["offset"]).tolist()} for entry in table]
    )


# Each pair holds every row of the entry table: the two-end-sites rig a root
# with channels, a rotating joint and end sites; the others a translating
# joint and a static root.
_ENTRY_TABLE_RIGS = [
    pytest.param(_two_end_sites_rig, _two_end_sites_table, id="root_rotating_joint_end_sites"),
    pytest.param(_translating_joint_rig, _translating_joint_table, id="translating_joint"),
    pytest.param(_static_root_rig, _static_root_table, id="static_root"),
]


class TestEntryTableRoundTrip:
    @pytest.mark.parametrize("rig, table", _ENTRY_TABLE_RIGS)
    def test_a_hand_built_node_list_gives_the_table(self, rig, table):
        assert _as_json(nodes_to_table(rig())) == _as_json(table())

    @pytest.mark.parametrize("rig, table", _ENTRY_TABLE_RIGS)
    def test_the_table_gives_back_the_same_nodes(self, rig, table):
        nodes = rig()
        rebuilt = nodes_from_table(nodes_to_table(nodes))
        assert [type(n) for n in rebuilt] == [type(n) for n in nodes]
        assert _as_json(nodes_to_table(rebuilt)) == _as_json(nodes_to_table(nodes))
        for before, after in zip(nodes, rebuilt, strict=True):
            for attribute in ("pos_channels", "rot_channels"):
                assert getattr(after, attribute, None) == getattr(before, attribute, None)


# The SHA-256 of each clip's node table as JSON (offsets as lists), computed
# on the code before #186 let a joint other than the root translate. A table
# of today's clips must not change: pos_channels stays on the root only.
_TABLE_DIGESTS_BEFORE_186 = {
    "bvh_data/bvh_test1.bvh": "5470628d5644396c01a4f4e18ad91f0fb80fb7b01f386762fca3c07435bbb5c8",
    "bvh_data/bvh_test2.bvh": "7d563e900a6a5443cd5d3fd36c532c7803594a4e4e06377e1a9511bcb37a5d7f",
    "bvh_data/bvh_test3.bvh": "6568b02d93786ea39b290ae0ae3733cca3366cff1e0c6d4a8848131f1d0abca5",
    "bvh_data/bvh_test4.bvh": "5470628d5644396c01a4f4e18ad91f0fb80fb7b01f386762fca3c07435bbb5c8",
    "bvh_data/cmu_12_01_walk.bvh": (
        "493ae9b1ac426b39e4edc8e0489ec8ce9e5fa1275a40238f61c649fdaf7bd4e8"
    ),
    "bvh_data/standard_skeleton.bvh": (
        "51ab720756687ff66bb4b5cc4f7f9c92e5a8203be27654c1da46622908b3089f"
    ),
    "bvh_data/synthetic/bvh_synthetic_lying_down.bvh": (
        "5470628d5644396c01a4f4e18ad91f0fb80fb7b01f386762fca3c07435bbb5c8"
    ),
    "bvh_data/synthetic/reference_props_rotation_only.bvh": (
        "1dd9cd0c3aa5ead64352e048d91bb304fdb1c129fc3d4871c538cafe278cd019"
    ),
    "bvh_data/synthetic/rotation_only_control.bvh": (
        "b6ef97068290508d192d5c0b5afd267bb18ca3dd2f3172f92f3655542537d69d"
    ),
    "tests/fixtures/full_precision_frame_time.bvh": (
        "d0714b55df3d9d3d9556ea4d8463341b35640ea97ddad816682ecd31085a92c5"
    ),
    "tests/fixtures/rotation_first_root.bvh": (
        "dd413c6f7c1812d238177e16a1bb2815ac5e85f9d904bd2f684f70b4684ce5f4"
    ),
}


@pytest.mark.filterwarnings("ignore:Rest pose suggests world up")
@pytest.mark.parametrize("relative", sorted(_TABLE_DIGESTS_BEFORE_186))
def test_the_node_table_of_a_fixture_clip_is_unchanged(relative):
    table = read_bvh_file(BVH_DATA.parent / relative).to_node_table()
    digest = hashlib.sha256(_as_json(table).encode()).hexdigest()
    assert digest == _TABLE_DIGESTS_BEFORE_186[relative]
