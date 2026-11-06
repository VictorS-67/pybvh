"""The feature block of joint translations (#186).

``include_joint_translations=True`` adds to the feature array a block that
holds :attr:`Bvh.joint_translations` flattened per frame, ``3 * P`` columns
right after the root position, and :func:`feature_array_layout` names it
``"joint_translations"``. The block is opt-in: with it off, the feature array
and its layout are those of a clip whose joints only rotate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import Bvh, batch_to_numpy, features, nodes_from_table, read_bvh_file

BVH_DATA = Path(__file__).parent.parent / "bvh_data"
FIXTURE_FILES = [
    *sorted(BVH_DATA.glob("*.bvh")),
    *sorted((BVH_DATA / "synthetic").glob("*.bvh")),
    *sorted((Path(__file__).parent / "fixtures").glob("*.bvh")),
]


def _clip(frames: int = 4) -> Bvh:
    """Reference root -> Hips (translates) -> Spine (translates) -> Head -> end site."""
    table = [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        {
            "name": "Spine",
            "parent": 1,
            "offset": [0, 10, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        {"name": "Head", "parent": 2, "offset": [0, 20, 0], "rot_channels": "ZYX"},
        {"parent": 3, "offset": [0, 5, 0]},
    ]
    rng = np.random.default_rng(0)
    return Bvh(
        nodes=nodes_from_table(table),
        root_pos=rng.normal(size=(frames, 3)),
        joint_angles=rng.uniform(-60, 60, size=(frames, 4, 3)),
        joint_translations=rng.normal(size=(frames, 2, 3)),
        frame_time=0.1,
        world_up="+y",
    )


class TestLayout:
    def test_the_block_follows_the_root_position(self):
        layout = features.feature_array_layout(
            num_joints=4,
            num_translating_joints=2,
            representation="6d",
            include_joint_translations=True,
        )
        assert layout == {
            "root_pos": slice(0, 3),
            "joint_translations": slice(3, 9),
            "rotations": slice(9, 33),
        }

    def test_the_block_needs_the_translating_joint_count(self):
        with pytest.raises(ValueError, match="num_translating_joints"):
            features.feature_array_layout(num_joints=4, include_joint_translations=True)

    def test_a_count_of_zero_gives_an_empty_block(self):
        layout = features.feature_array_layout(
            num_joints=4, num_translating_joints=0, include_joint_translations=True
        )
        assert layout["joint_translations"] == slice(3, 3)


class TestBlock:
    def test_the_block_holds_the_translations_where_the_layout_says(self):
        clip = _clip()
        feat = clip.to_feature_array(representation="euler", include_joint_translations=True)
        layout = clip.feature_array_layout(representation="euler", include_joint_translations=True)
        assert layout["joint_translations"] == slice(3, 9)
        assert feat.shape == (4, 3 + 6 + 4 * 3)
        # Frame 2: Hips' then Spine's X Y Z.
        np.testing.assert_array_equal(
            feat[2, layout["joint_translations"]],
            [*clip.joint_translations[2, 0], *clip.joint_translations[2, 1]],
        )
        np.testing.assert_array_equal(feat[:, layout["root_pos"]], clip.root_pos)
        np.testing.assert_array_equal(
            feat[:, layout["rotations"]], clip.joint_angles.reshape(4, 12)
        )

    def test_a_clip_without_translating_joints_gets_an_empty_block(self):
        clip = read_bvh_file(BVH_DATA / "bvh_test1.bvh")
        on = clip.to_feature_array(include_joint_translations=True)
        np.testing.assert_array_equal(on, clip.to_feature_array())
        layout = clip.feature_array_layout(include_joint_translations=True)
        assert layout["joint_translations"] == slice(3, 3)

    def test_the_block_is_trimmed_with_the_velocities(self):
        clip = _clip(frames=6)
        feat = clip.to_feature_array(
            include_joint_translations=True, include_velocities=True, pad="none"
        )
        layout = clip.feature_array_layout(include_joint_translations=True, include_velocities=True)
        assert feat.shape[0] == 4
        np.testing.assert_array_equal(
            feat[:, layout["joint_translations"]], clip.joint_translations[1:5].reshape(4, 6)
        )


def _rotating_spine_twin(clip: Bvh) -> Bvh:
    """``clip``'s skeleton with the spine declaring rotation channels only."""
    table = clip.to_node_table()
    del table[2]["pos_channels"]
    return Bvh(
        nodes=nodes_from_table(table),
        root_pos=clip.root_pos,
        joint_angles=clip.joint_angles,
        joint_translations=clip.joint_translations[:, :1],
        frame_time=clip.frame_time,
        world_up="+y",
    )


class TestBatch:
    def test_each_array_holds_the_block(self):
        clips = [_clip(4), _clip(6)]
        arrays = batch_to_numpy(clips, representation="6d", include_joint_translations=True)
        for clip, array in zip(clips, arrays, strict=True):
            np.testing.assert_array_equal(
                array, clip.to_feature_array(representation="6d", include_joint_translations=True)
            )

    @pytest.mark.parametrize("representation", ["euler", "axisangle", "6d", "quat", "rotmat"])
    def test_with_the_block_the_translating_joints_must_match(self, representation):
        clip = _clip()
        with pytest.raises(ValueError, match=r"index 2.*joint 'Spine' declares position channels"):
            batch_to_numpy(
                [clip, clip, _rotating_spine_twin(clip)],
                representation=representation,
                include_joint_translations=True,
            )

    @pytest.mark.parametrize("representation", ["euler", "axisangle", "6d", "quat", "rotmat"])
    def test_without_the_block_the_translating_joints_may_differ(self, representation):
        clip = _clip()
        arrays = batch_to_numpy([clip, _rotating_spine_twin(clip)], representation=representation)
        np.testing.assert_array_equal(arrays[0], arrays[1])

    def test_with_the_block_6d_still_ignores_the_rotation_order(self):
        clip = _clip()
        reordered = clip.change_euler_order("XYZ")
        arrays = batch_to_numpy(
            [clip, reordered], representation="6d", include_joint_translations=True
        )
        np.testing.assert_allclose(arrays[0], arrays[1], atol=1e-9)


@pytest.mark.filterwarnings("ignore:Rest pose suggests world up")
@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda path: path.name)
def test_off_the_array_holds_the_root_position_and_rotations_only(path):
    """With the block off, a clip's array and layout are those of before #186."""
    clip = read_bvh_file(path)
    feat = clip.to_feature_array(representation="euler")
    np.testing.assert_array_equal(
        feat,
        np.concatenate([clip.root_pos, clip.joint_angles.reshape(clip.frame_count, -1)], axis=1),
    )
    assert clip.feature_array_layout(representation="euler") == {
        "root_pos": slice(0, 3),
        "rotations": slice(3, 3 + 3 * clip.joint_count),
    }
