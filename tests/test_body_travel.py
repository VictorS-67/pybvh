"""Travel and heading read the body anchor (#205).

``reference_humanoid_6ch`` holds the body of ``rotation_only_control`` at
the same world positions, under a reference root that never moves: the
descriptors of where the body goes and which way it turns give the
control's values on it. On a clip whose anchor is its root they give what
they gave when they read the root, bit for bit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file
from pybvh.spatial_coord import frames_to_node_positions

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"


def _read(name):
    """The corpus clip, read without the world-up warnings some give."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(CORPUS / name)


@pytest.fixture
def reference():
    return _read("reference_humanoid_6ch.bvh")


@pytest.fixture
def control():
    return _read("rotation_only_control.bvh")


class TestAReferenceRootGivesTheControlsTravel:
    """A root above the hips: the readers describe the hips, as the control's do."""

    def test_root_trajectory(self, reference, control):
        np.testing.assert_allclose(
            reference.root_trajectory(include_velocities=True),
            control.root_trajectory(include_velocities=True),
            atol=1e-9,
        )

    @pytest.mark.parametrize("centered", ["anchor", "first_ground"])
    def test_the_centred_positions(self, reference, control, centered):
        # The reference root is node 0; the body follows it in node order.
        np.testing.assert_allclose(
            reference.node_positions(centered=centered)[:, 1:],
            control.node_positions(centered=centered),
            atol=1e-9,
        )

    @pytest.mark.parametrize("centered", ["anchor", "first_ground"])
    def test_the_centred_positions_of_one_frame(self, reference, control, centered):
        for frame in (0, 3):
            np.testing.assert_allclose(
                reference.node_positions(frame, centered=centered)[1:],
                control.node_positions(frame, centered=centered),
                atol=1e-9,
            )

    def test_the_facing_frame(self, reference, control):
        for ours, theirs in zip(reference.facing_frame(), control.facing_frame(), strict=True):
            np.testing.assert_allclose(ours, theirs, atol=1e-9)

    def test_the_gait_descriptors(self, reference, control):
        assert reference.walking_pace() == pytest.approx(control.walking_pace(), abs=1e-9)
        assert reference.walking_pace() > 0
        feet = ["LeftToeBase", "RightToeBase"]
        contacts = control.foot_contacts(feet)
        np.testing.assert_allclose(
            reference.gait_parameters(feet, contacts=contacts),
            control.gait_parameters(feet, contacts=contacts),
            atol=1e-9,
            equal_nan=True,
        )

    @pytest.mark.parametrize("centered", ["anchor", "first_ground"])
    def test_frames_to_node_positions_on_the_clip(self, reference, control, centered):
        from pybvh.spatial_coord import frames_to_node_positions

        np.testing.assert_allclose(
            frames_to_node_positions(reference, centered=centered)[:, 1:],
            frames_to_node_positions(control, centered=centered),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            frames_to_node_positions(
                reference,
                reference.root_pos[2],
                reference.joint_angles[2],
                centered=centered,
                joint_translations=reference.joint_translations[2],
            )[1:],
            control.node_positions(2, centered=centered),
            atol=1e-9,
        )

    @pytest.mark.parametrize("centered", ["world", "anchor", "first_ground"])
    def test_the_root_block_of_the_feature_array_is_the_root_s_centred_row(
        self, reference, centered
    ):
        np.testing.assert_allclose(
            reference.to_feature_array(centered=centered)[:, :3],
            reference.node_positions(centered=centered)[:, 0],
            atol=1e-9,
        )


class TestABodyMovedAwayFromTheRoot:
    """``reference_props_humanoid_6ch``: the same body, moved by (120, 0, 600)
    from a reference root at the origin, with props beside it."""

    def test_first_ground_centring_starts_the_body_above_the_origin(self):
        clip = _read("reference_props_humanoid_6ch.bvh")
        alone = _read("reference_humanoid_6ch.bvh")
        clip.world_up = "+y"  # what it is authored in, whatever the guess
        body = [clip.node_index[node.name] for node in alone.nodes[1:]]
        np.testing.assert_allclose(
            clip.node_positions(centered="first_ground")[:, body],
            alone.node_positions(centered="first_ground")[:, 1:],
            atol=1e-9,
        )


def test_the_heading_is_the_anchor_s_world_rotation_applied_to_rest_forward(reference):
    # The reference root and the hips both turn: the hips' world rotation
    # is the two composed. Checked against forward kinematics by hand: the
    # hips' world rotation carries each child's offset onto the child's
    # world position less the hips', so it carries rest forward, written as
    # a sum of three children's offsets, onto the same sum of their
    # positions less the hips'.
    angles = reference.joint_angles.copy()
    frames = np.arange(reference.frame_count)
    angles[:, 0] = np.deg2rad(np.column_stack([10.0 * frames, 20.0 + 0 * frames, 35.0 * frames]))
    angles[:, 1] = np.deg2rad(np.column_stack([5.0 + 0 * frames, -15.0 * frames, 50.0 * frames]))
    reference.joint_angles = angles
    reference.world_up = "+y"
    assert reference.rest_forward == "+z"

    hips = reference.node_index["Hips"]
    children = [reference.node_index[name] for name in ("Spine", "LeftUpLeg", "RightUpLeg")]
    offsets = np.array([reference.nodes[c].offset for c in children])  # (3, 3), rows
    weights = np.linalg.solve(offsets.T, [0.0, 0.0, 1.0])
    positions = reference.node_positions()
    forward = np.einsum("c,fcj->fj", weights, positions[:, children] - positions[:, [hips]])
    ground = forward[:, [0, 2]] / np.linalg.norm(forward[:, [0, 2]], axis=1, keepdims=True)

    trajectory = reference.root_trajectory()
    # Columns: ground x, ground z, then sin and cos of the angle from x toward z.
    np.testing.assert_allclose(trajectory[:, 3], ground[:, 0], atol=1e-9)
    np.testing.assert_allclose(trajectory[:, 2], ground[:, 1], atol=1e-9)
    # The hips' ground position is the travel.
    np.testing.assert_allclose(trajectory[:, :2], positions[:, hips][:, [0, 2]], atol=1e-9)


ROOT_ANCHORED = [
    REPO / "bvh_data" / "bvh_test1.bvh",
    REPO / "bvh_data" / "bvh_test3.bvh",
    REPO / "bvh_data" / "cmu_12_01_walk.bvh",
    CORPUS / "rotation_only_control.bvh",
    REPO / "tests" / "fixtures" / "rotation_first_root.bvh",
]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("path", ROOT_ANCHORED, ids=[p.name for p in ROOT_ANCHORED])
def test_a_clip_whose_anchor_is_its_root_reads_the_root_bit_for_bit(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clip = read_bvh_file(path)
    assert clip.body_anchor is clip.root
    up = clip.world_up_axis
    ground = [i for i in range(3) if i != up.index]
    world = clip.node_positions()
    root_pos = clip.root_pos

    np.testing.assert_array_equal(clip.root_trajectory()[:, :2], root_pos[:, ground])
    np.testing.assert_array_equal(
        clip.node_positions(centered="anchor"), world - root_pos[:, np.newaxis]
    )
    start = root_pos[0].copy()
    start[up.index] = 0.0
    np.testing.assert_array_equal(clip.node_positions(centered="first_ground"), world - start)
    features = clip.to_feature_array(centered="anchor")
    np.testing.assert_array_equal(features[:, :3], np.zeros_like(root_pos))
    on_ground = root_pos - np.outer(root_pos @ up.vector, up.vector)
    path_length = np.sum(np.linalg.norm(np.diff(on_ground, axis=0), axis=1))
    assert clip.walking_pace() == pytest.approx(
        path_length / ((clip.frame_count - 1) * clip.frame_time), rel=1e-12
    )


def _with_a_body():
    """The corpus clips that have a body, by name."""
    names = []
    for path in sorted(CORPUS.glob("*.bvh")):
        if _read(path.name).body_anchor is not None:
            names.append(path.name)
    return names


@pytest.mark.parametrize("centered", ["world", "anchor", "first_ground"])
@pytest.mark.parametrize("name", _with_a_body())
def test_the_root_block_and_the_rotations_give_back_the_centred_positions(name, centered):
    # The root block is the root's row of node_positions(centered), so
    # forward kinematics from it, the rotations and the joint translations
    # gives back every centred position, also where the root lies above
    # the body and the block is not zero.
    clip = _read(name)
    features = clip.to_feature_array(
        representation="euler", centered=centered, include_joint_translations=True
    )
    layout = clip.feature_array_layout(representation="euler", include_joint_translations=True)
    angles = features[:, layout["rotations"]].reshape(clip.joint_angles.shape)
    np.testing.assert_array_equal(angles, clip.joint_angles)
    positions = frames_to_node_positions(
        clip.fk_topology,
        root_pos=features[:, layout["root_pos"]],
        joint_angles=angles,
        joint_translations=features[:, layout["joint_translations"]].reshape(
            clip.joint_translations.shape
        ),
    )
    np.testing.assert_allclose(positions, clip.node_positions(centered=centered), atol=1e-9)
