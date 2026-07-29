"""Regression tests: topology is resolved by node identity, never by name.

Node names are not unique in general. The parser derives every end site's
display name from its parent joint (`'EndSite' + parent.name`), so two end
sites under one joint collide outright, and a real joint may be named after
one. Nothing in the parser or the `Bvh` constructor rejects a duplicate.

Anything that resolves a *parent* — or any node it already holds the object
for — must therefore key on `id(node)`. A name-keyed lookup silently returns
the wrong node: no exception, no shape change, no nan, just a limb attached
somewhere else. These tests pin that behaviour across every surface that
derives topology.
"""
from __future__ import annotations

import numpy as np
import pytest

from pybvh import Bvh
from pybvh.bvhnode import BvhEndSite, BvhJoint, BvhRoot
from pybvh.bvhplot._common import get_skeleton_lines


def _attach(parent, child):
    parent.children = parent.children + [child]
    child.parent = parent
    return child


@pytest.fixture
def collision_rig():
    """A joint named after another joint's generated end-site name.

    Node order (depth-first)::

        0 Hips             (root)
        1 EndSiteHips      (a real JOINT, named like an end site)
        2 Child
        3 EndSiteChild     (end site of Child)
        4 EndSiteHips      (end site of Hips — collides with node 1)

    `node_index` keeps the last occurrence, so a name lookup for the
    *joint* 'EndSiteHips' returns node 4 — an end site, which cannot have
    children at all.
    """
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    shadowed = _attach(root, BvhJoint('EndSiteHips', [1, 0, 0], 'ZYX', []))
    child = _attach(shadowed, BvhJoint('Child', [0, 1, 0], 'ZYX', []))
    _attach(child, BvhEndSite('EndSiteChild', [0, 1, 0]))
    _attach(root, BvhEndSite('EndSiteHips', [0, 0, 1]))
    nodes = [root, shadowed, child, child.children[0], root.children[1]]
    return Bvh(nodes, np.zeros((3, 3)), np.zeros((3, 3, 3)), 1 / 30)


@pytest.fixture
def duplicate_joint_rig():
    """Two sibling joints sharing the name 'Arm'.

    Node order::

        0 Hips  1 Arm  2 ArmChild  3 EndSiteArmChild  4 Arm  5 EndSiteArm
    """
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    first = _attach(root, BvhJoint('Arm', [1, 0, 0], 'ZYX', []))
    grandchild = _attach(first, BvhJoint('ArmChild', [0, 1, 0], 'ZYX', []))
    _attach(grandchild, BvhEndSite('EndSiteArmChild', [0, 1, 0]))
    second = _attach(root, BvhJoint('Arm', [-1, 0, 0], 'ZYX', []))
    _attach(second, BvhEndSite('EndSiteArm', [0, 1, 0]))
    nodes = [root, first, grandchild, grandchild.children[0],
             second, second.children[0]]
    return Bvh(nodes, np.zeros((3, 3)), np.zeros((3, 4, 3)), 1 / 30)


def _expected_node_edges(bvh):
    """The truth, computed the only way that cannot be fooled."""
    position = {id(node): i for i, node in enumerate(bvh.nodes)}
    return [(i, position[id(node.parent)])
            for i, node in enumerate(bvh.nodes) if node.parent is not None]


# =============================================================================
# The rigs really are ambiguous
# =============================================================================

def test_collision_rig_has_a_lossy_node_index(collision_rig):
    """Precondition: the name map genuinely loses a node."""
    assert len(collision_rig.node_index) < len(collision_rig.nodes)
    assert collision_rig.node_index['EndSiteHips'] == 4
    assert collision_rig.nodes[4].is_end_site()


def test_duplicate_joint_rig_has_a_lossy_joint_index(duplicate_joint_rig):
    assert len(duplicate_joint_rig.joint_index) < duplicate_joint_rig.joint_count


# =============================================================================
# Edge lists
# =============================================================================

class TestEdgeLists:

    def test_node_edges_under_end_site_collision(self, collision_rig):
        assert collision_rig.node_edges == _expected_node_edges(collision_rig)
        assert collision_rig.node_edges == [(1, 0), (2, 1), (3, 2), (4, 0)]

    def test_node_edges_under_duplicate_joint_names(self, duplicate_joint_rig):
        assert duplicate_joint_rig.node_edges == _expected_node_edges(
            duplicate_joint_rig)

    def test_no_edge_ever_points_at_an_end_site(self, collision_rig):
        """The failure this prevents: an end site given a child."""
        for _child, parent in collision_rig.node_edges:
            assert not collision_rig.nodes[parent].is_end_site()

    def test_edges_under_duplicate_joint_names(self, duplicate_joint_rig):
        """Joint-space edges: 'Arm' at column 1 must keep its own child."""
        assert duplicate_joint_rig.edges == [(1, 0), (2, 1), (3, 0)]

    def test_edges_ignores_end_site_collisions(self, collision_rig):
        """End sites are absent from joint space, so they cannot shadow a joint."""
        assert collision_rig.edges == [(1, 0), (2, 1)]

    def test_edge_counts(self, collision_rig):
        assert len(collision_rig.edges) == collision_rig.joint_count - 1
        assert len(collision_rig.node_edges) == len(collision_rig.nodes) - 1


# =============================================================================
# Everything else that derives topology
# =============================================================================

class TestOtherTopologyConsumers:

    def test_plot_bone_list(self, collision_rig):
        """The drawn skeleton is the posed skeleton."""
        assert get_skeleton_lines(collision_rig) == [
            (parent, child) for child, parent in _expected_node_edges(collision_rig)]

    def test_plot_bone_list_draws_every_bone_once(self, collision_rig):
        lines = get_skeleton_lines(collision_rig)
        assert len(lines) == len(set(lines)) == len(collision_rig.nodes) - 1

    def test_forward_kinematics(self, collision_rig):
        """FK was already identity-keyed; this pins it against the same rig."""
        coords = collision_rig.node_positions(frame=0)
        np.testing.assert_allclose(coords, [
            [0, 0, 0],    # Hips
            [1, 0, 0],    # EndSiteHips (joint), offset from Hips
            [1, 1, 0],    # Child
            [1, 2, 0],    # EndSiteChild
            [0, 0, 1],    # EndSiteHips (end site), offset from Hips
        ], atol=1e-12)

    def test_joint_tips(self, collision_rig):
        assert collision_rig.joint_tips == {
            'Hips': 4, 'EndSiteHips': None, 'Child': 3}

    def test_fk_topology_parent_array(self, collision_rig):
        np.testing.assert_array_equal(
            collision_rig.fk_topology.parent_idx, [-1, 0, 1, 2, 0])

    def test_extract_joints_keeps_the_right_end_site_offset(self, collision_rig):
        """The synthesized end site comes from the original node, by identity."""
        reduced = collision_rig.extract_joints(['Hips', 'EndSiteHips'])
        assert reduced.node_edges == _expected_node_edges(reduced)
        assert [n.name for n in reduced.nodes if not n.is_end_site()] == [
            'Hips', 'EndSiteHips']


# =============================================================================
# Mirroring
# =============================================================================

@pytest.fixture
def two_tips_rig():
    """A symmetric skeleton whose hands each carry two end sites.

    Both end sites of a hand get the same generated name, so a name-keyed
    lookup returns one of them twice. The rig is symmetric about x, so a
    correct mirror is the identity on the offsets.
    """
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    left = _attach(root, BvhJoint('LeftHand', [1, 0, 0], 'ZYX', []))
    right = _attach(root, BvhJoint('RightHand', [-1, 0, 0], 'ZYX', []))
    _attach(left, BvhEndSite('EndSiteLeftHand', [0.5, 1.0, 0.0]))
    _attach(left, BvhEndSite('EndSiteLeftHand', [0.5, 2.0, 0.0]))
    _attach(right, BvhEndSite('EndSiteRightHand', [-0.5, 1.0, 0.0]))
    _attach(right, BvhEndSite('EndSiteRightHand', [-0.5, 2.0, 0.0]))
    nodes = [root, left, left.children[0], left.children[1],
             right, right.children[0], right.children[1]]
    return Bvh(nodes, np.zeros((3, 3)), np.zeros((3, 3, 3)), 1 / 30)


class TestMirrorWithRepeatedEndSiteNames:

    def test_mirror_reflects_every_tip(self, two_tips_rig):
        """A symmetric rig mirrors back onto itself — every end site swapped."""
        mirrored = two_tips_rig.mirror(lateral_axis='x')
        for original, result in zip(two_tips_rig.nodes, mirrored.nodes):
            np.testing.assert_allclose(
                result.offset, original.offset, atol=1e-12,
                err_msg=f"node {original.name!r} was not reflected correctly")

    def test_round_trip_alone_would_not_catch_it(self, two_tips_rig):
        """Why the test above asserts against the reflection, not a round trip.

        Mirroring twice negates the lateral component twice, so an
        unswapped end site returns to its original value regardless — a
        round-trip assertion passes on the broken implementation too.
        """
        twice = two_tips_rig.mirror(lateral_axis='x').mirror(lateral_axis='x')
        for original, result in zip(two_tips_rig.nodes, twice.nodes):
            np.testing.assert_allclose(result.offset, original.offset, atol=1e-12)

    def test_unequal_end_site_counts_still_raise(self, two_tips_rig):
        """The domain error survives the move into the shared resolver."""
        lonely = [n for n in two_tips_rig.nodes if n.name == 'RightHand'][0]
        lonely.children = lonely.children[:1]
        with pytest.raises(ValueError, match="Cannot pair end sites"):
            two_tips_rig.mirror(lateral_axis='x')


# =============================================================================
# node_lr_pairs
# =============================================================================

class TestNodeLrPairs:

    def test_covers_joints_and_end_sites(self, two_tips_rig):
        assert two_tips_rig.node_lr_pairs == [(1, 4), (2, 5), (3, 6)]

    def test_order_is_joints_then_end_sites(self, two_tips_rig):
        pairs = two_tips_rig.node_lr_pairs
        is_end_site = [two_tips_rig.nodes[left].is_end_site() for left, _ in pairs]
        assert is_end_site == sorted(is_end_site)

    def test_matches_lr_pairs_on_the_joint_half(self, two_tips_rig):
        joint_pairs = [
            (left, right) for left, right in two_tips_rig.node_lr_pairs
            if not two_tips_rig.nodes[left].is_end_site()]
        node_names = [(two_tips_rig.nodes[left].name, two_tips_rig.nodes[right].name)
                      for left, right in joint_pairs]
        joint_names = two_tips_rig.joint_names
        assert node_names == [
            (joint_names[left], joint_names[right])
            for left, right in two_tips_rig.lr_pairs]

    def test_none_when_no_mapping(self, two_tips_rig):
        two_tips_rig.lr_mapping = None
        assert two_tips_rig.node_lr_pairs is None
        assert two_tips_rig.lr_pairs is None

    def test_unequal_end_site_counts_are_filtered_not_raised(self, two_tips_rig):
        """The property drops what it cannot pair; only `mirror` refuses."""
        lonely = [n for n in two_tips_rig.nodes if n.name == 'RightHand'][0]
        lonely.children = lonely.children[:1]
        pairs = two_tips_rig.node_lr_pairs
        assert pairs == [(1, 4)]  # the joint pair survives, its tips do not

    def test_real_skeleton(self):
        """On a normal rig every pair resolves and points at matching nodes."""
        from pathlib import Path
        from pybvh import read_bvh_file
        bvh = read_bvh_file(
            Path(__file__).parent.parent / "bvh_data" / "bvh_example.bvh")
        pairs = bvh.node_lr_pairs
        assert pairs
        for left, right in pairs:
            assert left != right
            assert (bvh.nodes[left].is_end_site()
                    == bvh.nodes[right].is_end_site())
        assert len(pairs) == len(set(pairs))
