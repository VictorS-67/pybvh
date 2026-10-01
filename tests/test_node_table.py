"""Tests for the node table: `nodes_to_table` and `nodes_from_table`.

The table is the positional twin of `FkTopology`: one entry per node in
`nodes` order, parents referenced by index. Position is identity, so the
rigs of `test_name_collisions.py` (duplicate joint names, two end sites
under one joint) must come back exactly. The builder is where a node
tree is validated once; every rule it enforces has a test here, and each
message names the entry it rejects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import (
    Bvh, FkTopology, nodes_from_table, nodes_to_table, read_bvh_file,
)
from pybvh.bvhnode import BvhEndSite, BvhJoint, BvhRoot

BVH_DATA = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture
def bvh_example():
    return read_bvh_file(BVH_DATA / "bvh_example.bvh")


def _attach(parent, child):
    parent.children = parent.children + [child]
    child.parent = parent
    return child


def _two_end_sites_rig():
    """The #16 rig: one joint carrying two end sites with one display name."""
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    hand = _attach(root, BvhJoint('Hand', [0, 1, 0], 'ZYX', []))
    _attach(hand, BvhEndSite('EndSiteHand', [1, 0, 0]))
    _attach(hand, BvhEndSite('EndSiteHand', [0, 0, 2]))
    return [root, hand, hand.children[0], hand.children[1]]


def _collision_rig():
    """A joint named like another joint's generated end site (see
    `test_name_collisions.py`)."""
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    shadowed = _attach(root, BvhJoint('EndSiteHips', [1, 0, 0], 'ZYX', []))
    child = _attach(shadowed, BvhJoint('Child', [0, 1, 0], 'ZYX', []))
    _attach(child, BvhEndSite('EndSiteChild', [0, 1, 0]))
    _attach(root, BvhEndSite('EndSiteHips', [0, 0, 1]))
    return [root, shadowed, child, child.children[0], root.children[1]]


def _duplicate_joint_rig():
    """Two sibling joints sharing the name 'Arm'."""
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    first = _attach(root, BvhJoint('Arm', [1, 0, 0], 'ZYX', []))
    grandchild = _attach(first, BvhJoint('ArmChild', [0, 1, 0], 'ZYX', []))
    _attach(grandchild, BvhEndSite('EndSiteArmChild', [0, 1, 0]))
    second = _attach(root, BvhJoint('Arm', [-1, 0, 0], 'XYZ', []))
    _attach(second, BvhEndSite('EndSiteArm', [0, 1, 0]))
    return [root, first, grandchild, grandchild.children[0],
            second, second.children[0]]


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
        {'name': 'Hips', 'parent': None, 'offset': np.array([0., 0., 0.]),
         'pos_channels': ['X', 'Y', 'Z'], 'rot_channels': ['Z', 'Y', 'X']},
        {'name': 'Hand', 'parent': 0, 'offset': np.array([0., 1., 0.]),
         'rot_channels': ['Z', 'Y', 'X']},
        {'name': 'EndSiteHand', 'parent': 1, 'offset': np.array([1., 0., 0.])},
        {'name': 'EndSiteHand', 'parent': 1, 'offset': np.array([0., 0., 2.])},
    ]


# =============================================================================
# nodes_to_table: the format
# =============================================================================

class TestNodesToTable:

    def test_two_end_sites_rig_matches_the_documented_table(self):
        table = nodes_to_table(_two_end_sites_rig())
        expected = _two_end_sites_table()
        assert len(table) == len(expected)
        for entry, want in zip(table, expected):
            assert list(entry) == list(want)
            for key in want:
                if key == 'offset':
                    np.testing.assert_array_equal(entry[key], want[key])
                else:
                    assert entry[key] == want[key]

    def test_offsets_are_copies(self):
        nodes = _two_end_sites_rig()
        table = nodes_to_table(nodes)
        table[1]['offset'][0] = 99.0
        table[1]['rot_channels'][0] = 'Q'
        assert nodes[1].offset[0] == 0.0
        assert nodes[1].rot_channels[0] == 'Z'

    def test_bvh_example_entries(self, bvh_example):
        table = nodes_to_table(bvh_example.nodes)
        assert len(table) == len(bvh_example.nodes)
        assert table[0]['parent'] is None
        assert 'pos_channels' in table[0]
        for i, (entry, node) in enumerate(zip(table[1:], bvh_example.nodes[1:]), 1):
            assert entry['parent'] == bvh_example.nodes.index(node.parent)
            assert 'pos_channels' not in entry
            assert ('rot_channels' in entry) == (not node.is_end_site())

    def test_parent_outside_the_list_raises(self, bvh_example):
        with pytest.raises(ValueError, match="not in the node list"):
            nodes_to_table(bvh_example.nodes[1:])


# =============================================================================
# Round trip
# =============================================================================

class TestRoundTrip:
    """`nodes_from_table(nodes_to_table(nodes))` rebuilds the tree."""

    @pytest.fixture(params=[
        pytest.param(_two_end_sites_rig, id="two_end_sites"),
        pytest.param(_collision_rig, id="collision"),
        pytest.param(_duplicate_joint_rig, id="duplicate_joint"),
    ])
    def rig(self, request):
        return _as_bvh(request.param())

    def test_bvh_example(self, bvh_example):
        rebuilt = nodes_from_table(nodes_to_table(bvh_example.nodes))
        rebuilt_bvh = Bvh(rebuilt, bvh_example.root_pos, bvh_example.joint_angles,
                          bvh_example.frame_time)
        assert rebuilt_bvh.matches_hierarchy(bvh_example)
        assert rebuilt_bvh.matches_channels(bvh_example)
        _assert_same_topology(FkTopology.from_nodes(rebuilt), bvh_example.fk_topology)

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
        assert nodes[0].pos_channels == ['X', 'Y', 'Z']
        assert nodes[1].rot_channels == ['Z', 'Y', 'X']

    def test_table_is_not_shared_with_the_result(self):
        table = _two_end_sites_table()
        nodes = nodes_from_table(table)
        table[1]['offset'][1] = 99.0
        table[1]['rot_channels'][0] = 'Q'
        assert nodes[1].offset[1] == 1.0
        assert nodes[1].rot_channels == ['Z', 'Y', 'X']


# =============================================================================
# Validation: every rule names the entry it rejects
# =============================================================================

class TestNodesFromTableRejects:

    def test_root_with_a_parent(self):
        table = _two_end_sites_table()
        table[0]['parent'] = 1
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*parent"):
            nodes_from_table(table)

    def test_second_root(self):
        table = _two_end_sites_table()
        table[1]['parent'] = None
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*one root"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_parent", [2, 3, 7, -1])
    def test_parent_index_outside_the_entries_before(self, bad_parent):
        table = _two_end_sites_table()
        table[2]['parent'] = bad_parent
        with pytest.raises(ValueError, match=r"entry 2 \('EndSiteHand'\).*\[0, 2\)"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_parent", ['Hips', 1.0, True])
    def test_parent_that_is_not_an_index(self, bad_parent):
        table = _two_end_sites_table()
        table[1]['parent'] = bad_parent
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*not an int"):
            nodes_from_table(table)

    def test_parent_that_is_an_end_site(self):
        table = _two_end_sites_table()
        table[3]['parent'] = 2
        with pytest.raises(ValueError, match=r"entry 3 \('EndSiteHand'\).*entry 2.*end site"):
            nodes_from_table(table)

    def test_root_without_rot_channels(self):
        table = _two_end_sites_table()
        del table[0]['rot_channels']
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*rot_channels"):
            nodes_from_table(table)

    def test_breadth_first_order(self):
        """Hips, Arm, Leg, EndSiteArm, EndSiteLeg: valid parents, wrong order."""
        table = [
            {'name': 'Hips', 'parent': None, 'offset': [0, 0, 0], 'rot_channels': 'ZYX'},
            {'name': 'Arm', 'parent': 0, 'offset': [1, 0, 0], 'rot_channels': 'ZYX'},
            {'name': 'Leg', 'parent': 0, 'offset': [0, -1, 0], 'rot_channels': 'ZYX'},
            {'name': 'EndSiteArm', 'parent': 1, 'offset': [1, 0, 0]},
            {'name': 'EndSiteLeg', 'parent': 2, 'offset': [0, -1, 0]},
        ]
        with pytest.raises(ValueError, match=r"entry 2 \('Leg'\).*depth-first.*entry 3 \('EndSiteArm'\)"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_offset", [[1.0, 2.0], "abc", None])
    def test_malformed_offset(self, bad_offset):
        table = _two_end_sites_table()
        table[1]['offset'] = bad_offset
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*offset"):
            nodes_from_table(table)

    def test_missing_offset(self):
        table = _two_end_sites_table()
        del table[1]['offset']
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*offset"):
            nodes_from_table(table)

    @pytest.mark.parametrize("bad_channels", ['ZYQ', ['Z', 'Y'], 'XXYZ', 3])
    def test_malformed_rot_channels(self, bad_channels):
        table = _two_end_sites_table()
        table[1]['rot_channels'] = bad_channels
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*channels"):
            nodes_from_table(table)

    def test_malformed_pos_channels(self):
        table = _two_end_sites_table()
        table[0]['pos_channels'] = 'XYQ'
        with pytest.raises(ValueError, match=r"entry 0 \('Hips'\).*channels"):
            nodes_from_table(table)

    def test_pos_channels_on_a_joint(self):
        table = _two_end_sites_table()
        table[1]['pos_channels'] = ['X', 'Y', 'Z']
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*pos_channels.*root"):
            nodes_from_table(table)

    def test_pos_channels_on_an_end_site(self):
        table = _two_end_sites_table()
        table[2]['pos_channels'] = ['X', 'Y', 'Z']
        with pytest.raises(ValueError, match=r"entry 2 \('EndSiteHand'\).*pos_channels.*end site"):
            nodes_from_table(table)

    def test_joint_without_a_name(self):
        table = _two_end_sites_table()
        del table[1]['name']
        with pytest.raises(ValueError, match=r"entry 1\b.*name"):
            nodes_from_table(table)

    def test_name_that_is_not_a_string(self):
        table = _two_end_sites_table()
        table[1]['name'] = 7
        with pytest.raises(ValueError, match=r"entry 1\b.*name.*string"):
            nodes_from_table(table)

    def test_unknown_key(self):
        """A misspelt `rot_channels` must not silently make an end site."""
        table = _two_end_sites_table()
        table[1]['rot_channel'] = table[1].pop('rot_channels')
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*'rot_channel'"):
            nodes_from_table(table)

    def test_children_key_from_the_hierarchy_dict(self):
        table = _two_end_sites_table()
        table[1]['children'] = ['EndSiteHand']
        with pytest.raises(ValueError, match=r"entry 1 \('Hand'\).*'children'"):
            nodes_from_table(table)

    def test_entry_that_is_not_a_dict(self):
        table = _two_end_sites_table()
        table[2] = ['EndSiteHand', 1, [1, 0, 0]]
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
        del table[0]['pos_channels']
        nodes = nodes_from_table(table)
        assert nodes[0].pos_channels == ['X', 'Y', 'Z']

    def test_end_site_name_defaults_to_the_parser_rule(self):
        table = _two_end_sites_table()
        del table[2]['name']
        del table[3]['name']
        nodes = nodes_from_table(table)
        assert [node.name for node in nodes[2:]] == ['EndSiteHand', 'EndSiteHand']

    def test_channel_strings_are_accepted(self):
        """'ZYX' and ['Z', 'Y', 'X'] both name the order, as on the nodes."""
        table = _two_end_sites_table()
        table[0]['rot_channels'] = 'ZYX'
        table[0]['pos_channels'] = 'XYZ'
        table[1]['rot_channels'] = 'XYZ'
        nodes = nodes_from_table(table)
        assert nodes[0].rot_channels == ['Z', 'Y', 'X']
        assert nodes[1].rot_channels == ['X', 'Y', 'Z']

    def test_offsets_may_be_lists(self):
        table = _two_end_sites_table()
        table[1]['offset'] = [0, 1, 0]
        nodes = nodes_from_table(table)
        np.testing.assert_array_equal(nodes[1].offset, [0., 1., 0.])
        assert nodes[1].offset.dtype == np.float64

    def test_root_may_omit_its_parent_key(self):
        table = _two_end_sites_table()
        del table[0]['parent']
        assert nodes_from_table(table)[0].parent is None
