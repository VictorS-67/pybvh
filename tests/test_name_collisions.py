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
import pandas as pd
import pytest

from pybvh import Bvh, df_to_bvh, read_bvh_file, write_bvh_file
from pybvh.bvhnode import BvhEndSite, BvhJoint, BvhRoot
from pybvh.bvhplot._from_bvh import get_skeleton_lines


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
    # Every joint gets its own angles, so a column swap between the two
    # 'Arm' joints shows. Powers of two survive the degrees round trip of
    # the DataFrame bit-exactly, which `==` on a Bvh requires.
    joint_angles = (2.0 ** -np.arange(36, dtype=float)).reshape(3, 4, 3)
    return Bvh(nodes, np.zeros((3, 3)), joint_angles, 1 / 30)


@pytest.fixture
def one_joint_two_end_sites_rig():
    """One joint carrying two end sites, which share their display name.

    Node order::

        0 Hips  1 Hand  2 EndSiteHand  3 EndSiteHand
    """
    root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
    hand = _attach(root, BvhJoint('Hand', [0, 1, 0], 'ZYX', []))
    _attach(hand, BvhEndSite('EndSiteHand', [1, 0, 0]))
    _attach(hand, BvhEndSite('EndSiteHand', [0, 0, 2]))
    nodes = [root, hand, hand.children[0], hand.children[1]]
    return Bvh(nodes, np.zeros((3, 3)), np.zeros((3, 2, 3)), 1 / 30)


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

    def test_extract_joints_wires_parents_by_identity(self):
        """A joint nested under a joint of the same name: 'Hand' hangs from
        the outer 'Arm', its real ancestor, not from the inner one that a
        name lookup would return as the latest 'Arm' built."""
        root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
        outer = _attach(root, BvhJoint('Arm', [1, 0, 0], 'ZYX', []))
        inner = _attach(outer, BvhJoint('Arm', [0, 1, 0], 'ZYX', []))
        _attach(inner, BvhEndSite('EndSiteArm', [0, 1, 0]))
        hand = _attach(outer, BvhJoint('Hand', [0, 0, 1], 'ZYX', []))
        _attach(hand, BvhEndSite('EndSiteHand', [0, 0, 1]))
        nodes = [root, outer, inner, inner.children[0], hand, hand.children[0]]
        bvh = Bvh(nodes, np.zeros((2, 3)), np.zeros((2, 4, 3)), 1 / 30)

        reduced = bvh.extract_joints(['Hips', 'Arm', 'Hand'])

        assert [n.name for n in reduced.nodes] == [
            'Hips', 'Arm', 'Arm', 'EndSiteArm', 'Hand', 'EndSiteHand']
        assert reduced.node_edges == [(1, 0), (2, 1), (3, 2), (4, 1), (5, 4)]

    def test_extract_joints_selects_columns_by_position(self, duplicate_joint_rig):
        """Two kept joints named 'Arm' each keep their own joint_angles column."""
        rig = duplicate_joint_rig
        angles = np.zeros((3, 4, 3))
        angles[:, 1, 0] = 10.0   # the first Arm, column 1
        angles[:, 3, 0] = 30.0   # the second Arm, column 3
        rig = Bvh(rig.nodes, rig.root_pos, angles, rig.frame_time)

        reduced = rig.extract_joints(['Hips', 'Arm'])

        assert reduced.joint_names == ['Hips', 'Arm', 'Arm']
        np.testing.assert_array_equal(reduced.joint_angles[:, 1, 0], 10.0)
        np.testing.assert_array_equal(reduced.joint_angles[:, 2, 0], 30.0)


# =============================================================================
# DataFrame column labels
# =============================================================================

class TestDataFrameColumns:
    """`to_df_dict` exports one column per channel of every node, however
    the nodes are named: a repeated name is labelled `X`, `X.1`, `X.2` in
    node order, pandas' rule for repeated CSV headers."""

    def test_two_end_sites_each_get_their_columns(self, one_joint_two_end_sites_rig):
        columns = list(one_joint_two_end_sites_rig.to_df_dict(mode='coordinates'))
        assert len(columns) == 13
        assert 'EndSiteHand_X' in columns
        assert 'EndSiteHand.1_X' in columns

    def test_two_joints_sharing_a_name_each_get_their_columns(self, duplicate_joint_rig):
        columns = list(duplicate_joint_rig.to_df_dict(mode='euler'))
        rotation_columns = [c for c in columns if c.endswith('_rot')]
        assert len(rotation_columns) == 12
        assert 'Arm_Z_rot' in columns
        assert 'Arm.1_Z_rot' in columns

    def test_suffix_follows_the_nodes_the_mode_exports(self, collision_rig):
        """The joint 'EndSiteHips' (node 1) precedes the end site of that
        name (node 4): coordinates mode suffixes the end site, and euler
        mode, where end sites have no columns, suffixes nothing."""
        coordinates = list(collision_rig.to_df_dict(mode='coordinates'))
        assert coordinates == [
            'time',
            'Hips_X', 'Hips_Y', 'Hips_Z',
            'EndSiteHips_X', 'EndSiteHips_Y', 'EndSiteHips_Z',
            'Child_X', 'Child_Y', 'Child_Z',
            'EndSiteChild_X', 'EndSiteChild_Y', 'EndSiteChild_Z',
            'EndSiteHips.1_X', 'EndSiteHips.1_Y', 'EndSiteHips.1_Z']

        euler = list(collision_rig.to_df_dict(mode='euler'))
        assert euler == [
            'time',
            'Hips_X_pos', 'Hips_Y_pos', 'Hips_Z_pos',
            'Hips_Z_rot', 'Hips_Y_rot', 'Hips_X_rot',
            'EndSiteHips_Z_rot', 'EndSiteHips_Y_rot', 'EndSiteHips_X_rot',
            'Child_Z_rot', 'Child_Y_rot', 'Child_X_rot']

    def test_a_node_named_like_a_suffixed_label_keeps_its_name(self):
        """Joints 'Arm', 'Arm', 'Arm.1': the second 'Arm' skips the label
        the third joint owns, as pandas reads the header `Arm,Arm,Arm.1`
        as `Arm, Arm.2, Arm.1`."""
        root = BvhRoot('Hips', [0, 0, 0], 'XYZ', 'ZYX', [])
        for name, offset in [('Arm', [1, 0, 0]), ('Arm', [-1, 0, 0]), ('Arm.1', [0, 1, 0])]:
            joint = _attach(root, BvhJoint(name, offset, 'ZYX', []))
            _attach(joint, BvhEndSite('EndSite' + name, [0, 1, 0]))
        nodes = [root]
        for joint in root.children:
            nodes.extend([joint, joint.children[0]])
        bvh = Bvh(nodes, np.zeros((2, 3)), np.zeros((2, 4, 3)), 1 / 30)

        rotation_columns = [c for c in bvh.to_df_dict(mode='euler') if c.endswith('_Z_rot')]

        assert rotation_columns == ['Hips_Z_rot', 'Arm_Z_rot', 'Arm.2_Z_rot', 'Arm.1_Z_rot']


# =============================================================================
# Round trips through a file and a DataFrame
# =============================================================================

class TestRoundTrips:
    """Written and read back, or rebuilt from a DataFrame, a rig keeps its
    topology: the writer walks `children`, the reader and `df_to_bvh`
    build by position, and none of them keys on a name."""

    @pytest.fixture(params=[
        "collision_rig", "duplicate_joint_rig", "one_joint_two_end_sites_rig"])
    def rig(self, request):
        return request.getfixturevalue(request.param)

    def test_file_round_trip(self, rig, tmp_path):
        path = tmp_path / "rig.bvh"
        write_bvh_file(rig, path)
        back = read_bvh_file(path)
        assert back.matches_hierarchy(rig)
        assert back.matches_channels(rig)
        # The file carries six decimals, so the posed skeleton is equal to
        # that precision; a limb attached elsewhere is off by a bone length.
        np.testing.assert_allclose(
            back.node_positions(), rig.node_positions(), atol=1e-6)

    @pytest.fixture(params=[
        "collision_rig", "duplicate_joint_rig", "one_joint_two_end_sites_rig"])
    def dataframe_rig(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def df(self, dataframe_rig):
        return pd.DataFrame(dataframe_rig.to_df_dict(mode='euler'))

    def test_dataframe_round_trip_through_the_node_list(self, dataframe_rig, df):
        rebuilt = df_to_bvh(dataframe_rig.nodes, df)
        assert rebuilt.matches_hierarchy(dataframe_rig)
        assert rebuilt == dataframe_rig

    def test_dataframe_round_trip_through_the_node_table(self, dataframe_rig, df):
        rebuilt = df_to_bvh(dataframe_rig.to_node_table(), df)
        assert rebuilt.matches_hierarchy(dataframe_rig)
        assert rebuilt == dataframe_rig

    def test_from_df_takes_the_node_list(self, dataframe_rig, df):
        rebuilt = Bvh.from_df(dataframe_rig.nodes, df)
        assert rebuilt.matches_hierarchy(dataframe_rig)
        assert rebuilt == dataframe_rig

    def test_from_df_takes_the_node_table(self, dataframe_rig, df):
        rebuilt = Bvh.from_df(dataframe_rig.to_node_table(), df)
        assert rebuilt.matches_hierarchy(dataframe_rig)
        assert rebuilt == dataframe_rig

    def test_name_keyed_dict_is_refused(self, one_joint_two_end_sites_rig):
        """The hierarchy dict of v0.9.0, which held one end site of two,
        names the migration instead of rebuilding a wrong clip."""
        rig = one_joint_two_end_sites_rig
        hier = {
            'Hips': {'offset': [0, 0, 0], 'parent': None, 'children': ['Hand'],
                     'pos_channels': ['X', 'Y', 'Z'], 'rot_channels': ['Z', 'Y', 'X']},
            'Hand': {'offset': [0, 1, 0], 'parent': 'Hips',
                     'children': ['EndSiteHand'], 'rot_channels': ['Z', 'Y', 'X']},
            'EndSiteHand': {'offset': [0, 0, 2], 'parent': 'Hand'},
        }
        df = pd.DataFrame(rig.to_df_dict(mode='euler'))
        with pytest.raises(TypeError, match="to_node_table"):
            df_to_bvh(hier, df)
        with pytest.raises(TypeError, match="to_node_table"):
            Bvh.from_df(hier, df)

    def test_empty_hierarchy_is_refused(self, one_joint_two_end_sites_rig):
        df = pd.DataFrame(one_joint_two_end_sites_rig.to_df_dict(mode='euler'))
        with pytest.raises(ValueError, match="to_node_table"):
            df_to_bvh([], df)

    def test_repeated_literal_labels_get_the_suffix_rule(self, duplicate_joint_rig):
        """A DataFrame labelling both 'Arm' joints 'Arm_X_rot', as a
        v0.9.0 export or a hand-built frame would, is refused with the
        missing labels and the rule that makes the second one 'Arm.1'."""
        rig = duplicate_joint_rig
        df = pd.DataFrame(rig.to_df_dict(mode='euler'))
        df.columns = [c.replace('Arm.1_', 'Arm_') for c in df.columns]
        assert list(df.columns).count('Arm_X_rot') == 2

        with pytest.raises(ValueError, match=r"missing.*'Arm\.1_Z_rot'") as excinfo:
            df_to_bvh(rig.nodes, df)
        assert "X, X.1, X.2" in str(excinfo.value)


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
