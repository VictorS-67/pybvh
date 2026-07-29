"""Tests for `FkTopology` — the array-signature forward-kinematics input.

Two things are under test. First, that a topology extracted from a `Bvh`
reproduces `node_positions()` exactly when fed back through FK with no node
objects in sight — that is the whole point of the type. Second, that the
constructor rejects every malformed topology that would otherwise produce
silently wrong geometry, since the train-time caller builds one from arrays
with no node tree to check itself against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import FkTopology, frames_to_node_positions, read_bvh_file

BVH_DATA = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture
def bvh_example():
    return read_bvh_file(BVH_DATA / "bvh_example.bvh")


@pytest.fixture
def topology(bvh_example):
    return bvh_example.fk_topology


# =============================================================================
# Extraction
# =============================================================================

class TestFromNodes:
    """`FkTopology.from_nodes` / `Bvh.fk_topology` describe the node tree."""

    def test_shapes_and_counts(self, bvh_example, topology):
        num_nodes = len(bvh_example.nodes)
        assert topology.offsets.shape == (num_nodes, 3)
        assert topology.parent_idx.shape == (num_nodes,)
        assert topology.joint_idx.shape == (num_nodes,)
        assert len(topology.euler_orders) == bvh_example.joint_count

    def test_euler_orders_matches_the_bvh_property(self, bvh_example, topology):
        """The list is the same one `Bvh.euler_orders` returns, element for element."""
        assert topology.euler_orders == bvh_example.euler_orders

    def test_offsets_match_the_nodes(self, bvh_example, topology):
        expected = np.array([n.offset for n in bvh_example.nodes])
        np.testing.assert_allclose(topology.offsets, expected)

    def test_parent_idx_is_node_edges(self, bvh_example, topology):
        """`node_edges` is the edge-list view of the same parent array."""
        derived = [(child, int(parent))
                   for child, parent in enumerate(topology.parent_idx)
                   if parent >= 0]
        assert derived == bvh_example.node_edges

    def test_end_sites_are_exactly_the_negative_joint_idx(self, bvh_example, topology):
        is_end_site = np.array([n.is_end_site() for n in bvh_example.nodes])
        np.testing.assert_array_equal(topology.joint_idx < 0, is_end_site)

    def test_joint_idx_counts_up_in_node_order(self, topology):
        joints = topology.joint_idx[topology.joint_idx >= 0]
        np.testing.assert_array_equal(joints, np.arange(joints.size))

    def test_parent_outside_the_list_raises(self, bvh_example):
        with pytest.raises(ValueError, match="not in the node list"):
            FkTopology.from_nodes(bvh_example.nodes[1:])

    def test_property_is_not_cached(self, bvh_example):
        """Recomputed per access, so it cannot go stale against the skeleton."""
        before = bvh_example.fk_topology
        scaled = bvh_example.scale(2.0)
        np.testing.assert_allclose(
            scaled.fk_topology.offsets, before.offsets * 2.0)


# =============================================================================
# FK from arrays alone
# =============================================================================

class TestForwardKinematicsFromArrays:
    """A topology reproduces the node-object path exactly."""

    @pytest.mark.parametrize("filename", [
        "bvh_example.bvh", "cmu_12_01_walk.bvh", "bvh_test3.bvh"])
    def test_matches_the_bvh_path(self, filename):
        """Bit-exact: the array path is the same computation, not an approximation."""
        bvh = read_bvh_file(BVH_DATA / filename)
        result = frames_to_node_positions(
            bvh.fk_topology, bvh.root_pos, bvh.joint_angles, centered="world")
        np.testing.assert_array_equal(result, bvh.node_positions())

    @pytest.mark.parametrize("filename", [
        "bvh_example.bvh", "cmu_12_01_walk.bvh", "bvh_test3.bvh"])
    def test_matches_the_bvh_path_skeleton_centered(self, filename):
        """Same geometry, but not bit-exact by design.

        `Bvh.node_positions` reaches skeleton-centered by subtracting the
        root from its cached world-frame FK, where this path simply never
        adds it — one rounding apart, which is the price of that cache.
        """
        bvh = read_bvh_file(BVH_DATA / filename)
        result = frames_to_node_positions(
            bvh.fk_topology, bvh.root_pos, bvh.joint_angles,
            centered="skeleton")
        np.testing.assert_allclose(
            result, bvh.node_positions(centered="skeleton"), atol=1e-12)

    def test_matches_the_node_list_path(self, bvh_example):
        from_nodes = frames_to_node_positions(
            bvh_example.nodes, bvh_example.root_pos, bvh_example.joint_angles)
        from_arrays = frames_to_node_positions(
            bvh_example.fk_topology, bvh_example.root_pos,
            bvh_example.joint_angles)
        np.testing.assert_array_equal(from_arrays, from_nodes)

    def test_single_frame(self, bvh_example, topology):
        result = frames_to_node_positions(
            topology, bvh_example.root_pos[0], bvh_example.joint_angles[0])
        assert result.shape == (len(bvh_example.nodes), 3)
        np.testing.assert_array_equal(result, bvh_example.node_positions(frame=0))

    def test_survives_a_serialization_round_trip(self, bvh_example, topology, tmp_path):
        """Every field stores and reloads — the persistence requirement."""
        path = tmp_path / "topology.npz"
        np.savez(path, offsets=topology.offsets, parent_idx=topology.parent_idx,
                 joint_idx=topology.joint_idx, euler_orders=topology.euler_orders)
        loaded = np.load(path)
        rebuilt = FkTopology(
            loaded["offsets"], loaded["parent_idx"], loaded["joint_idx"],
            list(loaded["euler_orders"]))
        np.testing.assert_array_equal(
            frames_to_node_positions(
                rebuilt, bvh_example.root_pos, bvh_example.joint_angles),
            bvh_example.node_positions())

    def test_permuted_joint_columns_need_permuted_orders(self, bvh_example, topology):
        """`euler_orders` is indexed by joint column, not by node.

        A caller repacking the joint axis must carry the orders along. This
        pins the documented contract: permute both and the geometry is
        unchanged.
        """
        num_joints = len(topology.euler_orders)
        rng = np.random.default_rng(0)
        permutation = rng.permutation(num_joints)

        joint_idx = topology.joint_idx.copy()
        is_joint = joint_idx >= 0
        joint_idx[is_joint] = permutation[joint_idx[is_joint]]
        inverse = np.argsort(permutation)
        repacked = topology._replace(
            joint_idx=joint_idx,
            euler_orders=[topology.euler_orders[j] for j in inverse])

        np.testing.assert_allclose(
            frames_to_node_positions(
                repacked, bvh_example.root_pos,
                bvh_example.joint_angles[:, inverse]),
            bvh_example.node_positions(), atol=1e-12)

    def test_joint_count_mismatch_raises(self, bvh_example, topology):
        with pytest.raises(ValueError, match="but the skeleton declares"):
            frames_to_node_positions(
                topology, bvh_example.root_pos,
                bvh_example.joint_angles[:, :-1])

    def test_motion_is_required(self, topology):
        with pytest.raises(ValueError, match="must be provided"):
            frames_to_node_positions(topology)

    def test_rejects_other_types(self):
        with pytest.raises(ValueError, match="skeleton must be"):
            frames_to_node_positions("not a skeleton", np.zeros((1, 3)),
                                     np.zeros((1, 1, 3)))


# =============================================================================
# The up axis for centered="first"
# =============================================================================

class TestFirstCenteringUpAxis:
    """`centered='first'` never guesses a gravity direction."""

    def test_bvh_supplies_its_own_world_up(self, bvh_example):
        """A Bvh knows its up axis, so no argument is needed — and '+y' is not assumed."""
        assert bvh_example.world_up == '+z'
        np.testing.assert_allclose(
            frames_to_node_positions(bvh_example, centered="first"),
            bvh_example.node_positions(centered="first"))

    def test_topology_without_up_raises(self, bvh_example, topology):
        with pytest.raises(ValueError, match="carries no gravity direction"):
            frames_to_node_positions(
                topology, bvh_example.root_pos, bvh_example.joint_angles,
                centered="first")

    def test_node_list_without_up_raises(self, bvh_example):
        with pytest.raises(ValueError, match="carries no gravity direction"):
            frames_to_node_positions(
                bvh_example.nodes, bvh_example.root_pos,
                bvh_example.joint_angles, centered="first")

    def test_explicit_up_is_honoured(self, bvh_example, topology):
        np.testing.assert_allclose(
            frames_to_node_positions(
                topology, bvh_example.root_pos, bvh_example.joint_angles,
                centered="first", up=bvh_example.world_up),
            bvh_example.node_positions(centered="first"))


# =============================================================================
# Validation
# =============================================================================

class TestValidation:
    """Every invariant the FK loop relies on is checked at construction."""

    def test_forward_parent_reference_raises(self, topology):
        parent_idx = topology.parent_idx.copy()
        parent_idx[1] = 5  # points ahead of itself
        with pytest.raises(ValueError, match="does not precede its child"):
            topology._replace(parent_idx=parent_idx)

    def test_two_roots_raise(self, topology):
        parent_idx = topology.parent_idx.copy()
        parent_idx[3] = -1
        with pytest.raises(ValueError, match="exactly one -1"):
            topology._replace(parent_idx=parent_idx)

    def test_root_marked_as_end_site_raises(self, topology):
        """The -1 sentinel is a legal negative index; unchecked it reads the last joint."""
        joint_idx = topology.joint_idx.copy()
        joint_idx[0] = -1
        with pytest.raises(ValueError, match="marks the root as an end site"):
            topology._replace(joint_idx=joint_idx)

    def test_child_of_an_end_site_raises(self, topology):
        """End sites accumulate no rotation, so a child would read uninitialized memory."""
        joint_idx = topology.joint_idx.copy()
        end_site = int(np.flatnonzero(joint_idx < 0)[0])
        parent_idx = topology.parent_idx.copy()
        child = int(np.flatnonzero(parent_idx > end_site)[0])
        parent_idx[child] = end_site
        with pytest.raises(ValueError, match="marks as an end site"):
            topology._replace(parent_idx=parent_idx)

    def test_joint_columns_must_be_a_complete_range(self, topology):
        joint_idx = topology.joint_idx.copy()
        joint_idx[joint_idx == 1] = 0  # duplicate column 0, column 1 unused
        with pytest.raises(ValueError, match="exactly 0..J-1"):
            topology._replace(joint_idx=joint_idx)

    def test_euler_orders_length_must_match(self, topology):
        with pytest.raises(ValueError, match="euler_orders has"):
            topology._replace(euler_orders=topology.euler_orders[:-1])

    def test_euler_order_must_be_a_permutation_of_xyz(self, topology):
        orders = list(topology.euler_orders)
        orders[2] = 'XXY'
        with pytest.raises(ValueError, match="not a permutation"):
            topology._replace(euler_orders=orders)

    def test_offsets_shape(self, topology):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            topology._replace(offsets=topology.offsets[:, :2])

    def test_array_lengths_must_agree(self, topology):
        with pytest.raises(ValueError, match="to match offsets"):
            topology._replace(parent_idx=topology.parent_idx[:-1])

    def test_empty_topology_raises(self):
        with pytest.raises(ValueError, match="at least one node"):
            FkTopology(np.zeros((0, 3)), np.zeros(0), np.zeros(0), [])

    def test_negative_parent_other_than_root_raises(self, topology):
        parent_idx = topology.parent_idx.copy()
        parent_idx[4] = -2
        with pytest.raises(ValueError, match="only legal negative value is -1"):
            topology._replace(parent_idx=parent_idx)

    def test_replace_rejects_unknown_fields(self, topology):
        with pytest.raises(ValueError, match="no field"):
            topology._replace(offsetz=topology.offsets)

    def test_constructor_coerces_dtypes(self, topology):
        """A caller reloading ints from disk should not have to cast."""
        rebuilt = FkTopology(
            topology.offsets.astype(np.float32),
            topology.parent_idx.astype(np.int16),
            topology.joint_idx.astype(np.int16),
            topology.euler_orders)
        assert rebuilt.offsets.dtype == np.float64
        assert rebuilt.parent_idx.dtype == np.intp
        assert rebuilt.joint_idx.dtype == np.intp

    def test_is_a_tuple(self, topology):
        offsets, parent_idx, joint_idx, euler_orders = topology
        assert offsets is topology.offsets
        assert euler_orders is topology.euler_orders
