"""Frame operations and skeleton edits on a clip with a translating joint (#186).

Each operation that changes frames or offsets keeps ``joint_translations``
right, by the rule of #186's operation table. The clip: a reference root,
a ``Hips`` joint that translates, a ``Spine`` joint that only rotates and
an end site. The root stands still and nothing rotates, so the ``Hips``
position in each frame is its translation, and every expected value is
read off the frames by hand.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from six_channel_fixtures import reference_hips_table

from pybvh import Bvh, nodes_from_table, transforms

FRAMES = 6

# Hips' x travels 0, 1, 4, 9, 16, 25 (f squared): not linear, so an
# interpolated frame differs from the frame it replaces.
HIPS_X = np.array([0.0, 1, 4, 9, 16, 25])


def _table() -> list[dict]:
    return reference_hips_table()


def _translations(hips_x) -> np.ndarray:
    """Hips at ``(x, 90, 0)`` in each frame, as ``(F, 1, 3)``."""
    hips_x = np.asarray(hips_x, dtype=np.float64)
    return np.stack([hips_x, np.full_like(hips_x, 90.0), np.zeros_like(hips_x)], axis=1)[:, None, :]


def _clip(hips_x=HIPS_X, table=None) -> Bvh:
    frames = len(hips_x)
    return Bvh(
        nodes=nodes_from_table(_table() if table is None else table),
        root_pos=np.zeros((frames, 3)),
        joint_angles=np.zeros((frames, 3, 3)),
        joint_translations=_translations(hips_x),
        frame_time=0.1,
        world_up="+y",
    )


def _hips_x(clip: Bvh) -> list[float]:
    return clip.joint_translations[:, 0, 0].tolist()


class _FixedDraws:
    """A stand-in for ``rng`` whose draws are fixed: at ``drop_rate=0.5``,
    frames 1, 2 and 4 of six are dropped and 0, 3 and 5 kept."""

    def random(self, size):
        return np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.9])


class TestFrameOperations:
    def test_slicing_keeps_the_selected_frames(self):
        assert _hips_x(_clip()[1:3]) == [1.0, 4.0]

    def test_joining_appends_the_frames(self):
        joined = _clip([0.0, 1]) + _clip([7.0, 8])
        assert _hips_x(joined) == [0.0, 1.0, 7.0, 8.0]

    def test_joining_in_place_appends_the_frames(self):
        clip = _clip([0.0, 1])
        clip += _clip([7.0, 8])
        assert _hips_x(clip) == [0.0, 1.0, 7.0, 8.0]

    def test_splicing_writes_the_frames(self):
        clip = _clip()
        clip[1:3] = _clip([7.0, 8])
        assert _hips_x(clip) == [0.0, 7.0, 8.0, 9.0, 16.0, 25.0]

    def test_a_copy_keeps_the_translations(self):
        assert _hips_x(_clip().copy()) == HIPS_X.tolist()

    def test_a_pickle_keeps_the_translations(self):
        assert _hips_x(pickle.loads(pickle.dumps(_clip()))) == HIPS_X.tolist()

    def test_resampling_interpolates_linearly(self):
        # 10 fps to 20 fps: a new frame halfway between each pair.
        resampled = _clip().resample(20)
        assert _hips_x(resampled) == pytest.approx([0, 0.5, 1, 2.5, 4, 6.5, 9, 12.5, 16, 20.5, 25])

    def test_dropped_frames_are_interpolated_linearly(self):
        # Frames 1 and 2 lie between kept frames 0 (x 0) and 3 (x 9);
        # frame 4 between 3 (x 9) and 5 (x 25).
        dropped = _clip().drop_frames(0.5, rng=_FixedDraws())
        assert _hips_x(dropped) == [0.0, 3.0, 6.0, 9.0, 17.0, 25.0]


def _rotating_hips_table() -> list[dict]:
    table = _table()
    del table[1]["pos_channels"]
    table[2]["pos_channels"] = "XYZ"
    return table


def _clip_on(table) -> Bvh:
    return Bvh(
        nodes=nodes_from_table(table),
        root_pos=np.zeros((2, 3)),
        joint_angles=np.zeros((2, 3, 3)),
        joint_translations=np.zeros((2, 1, 3)),
        frame_time=0.1,
    )


class TestJoinsRequireTheSameTranslatingJoints:
    def test_joining_in_place_refuses_another_set(self):
        clip = _clip([0.0, 1])
        with pytest.raises(ValueError, match="Hips"):
            clip += _clip_on(_rotating_hips_table())

    def test_the_way_the_refusal_names_keeps_each_displacement(self):
        # The second clip's Hips rest at (0, 100, 0), not (0, 90, 0): the
        # join refuses and names other.retarget(self), which moves its
        # frames by (0, -10, 0), so (7, 100, 0) becomes (7, 90, 0).
        table = _table()
        table[1]["offset"] = [0, 100, 0]
        other = Bvh(
            nodes=nodes_from_table(table),
            root_pos=np.zeros((2, 3)),
            joint_angles=np.zeros((2, 3, 3)),
            joint_translations=np.array([[[7.0, 100, 0]], [[8.0, 100, 0]]]),
            frame_time=0.1,
            world_up="+y",
        )
        clip = _clip([0.0, 1])
        with pytest.raises(ValueError, match=r"other\.retarget\(self\)"):
            clip + other
        joined = clip + other.retarget(clip)
        assert joined.joint_translations.tolist() == [
            [[0.0, 90, 0]],
            [[1.0, 90, 0]],
            [[7.0, 90, 0]],
            [[8.0, 90, 0]],
        ]

    def test_splicing_refuses_another_set(self):
        clip = _clip([0.0, 1])
        with pytest.raises(ValueError, match="Hips"):
            clip[0:2] = _clip_on(_rotating_hips_table())


class TestScale:
    @pytest.mark.parametrize("inplace", [False, True])
    def test_the_translations_are_multiplied(self, inplace):
        clip = _clip([0.0, 3])
        scaled = clip.scale(2, inplace=inplace) or clip
        assert scaled.joint_translations.tolist() == [[[0.0, 180, 0]], [[6.0, 180, 0]]]

    def test_positions_scale_with_the_skeleton(self):
        # Spine sits 10 above Hips, the end site 5 above Spine: twice that.
        scaled = _clip([0.0, 3]).scale(2)
        assert scaled.node_positions()[1].tolist() == [
            [0.0, 0, 0],
            [6.0, 180, 0],
            [6.0, 200, 0],
            [6.0, 210, 0],
        ]


class TestTheOffsetReplacingPath:
    def test_it_refuses_to_leave_the_translations_unsaid(self):
        # The one path every offset edit takes: on a clip with translating
        # joints it must be told what happens to their frames.
        clip = _clip([0.0, 3])
        with pytest.raises(TypeError, match="translations="):
            clip._replace_offsets(clip._offsets() * 2)
        assert clip.nodes[1].offset.tolist() == [0.0, 90.0, 0.0]

    def test_it_writes_the_translations_it_is_given(self):
        clip = _clip([0.0, 3])
        clip._replace_offsets(clip._offsets(), translations=np.ones((2, 1, 3)))
        assert clip.joint_translations.tolist() == [[[1.0, 1, 1]], [[1.0, 1, 1]]]


def _reference(hips_offset, spine_offset) -> Bvh:
    """A skeleton to retarget onto, with Hips rotating only: retarget
    reads its offsets, not its channels."""
    table = _table()
    del table[1]["pos_channels"]
    table[1]["offset"] = hips_offset
    table[2]["offset"] = spine_offset
    return Bvh(nodes=nodes_from_table(table))


class TestRetarget:
    @pytest.mark.parametrize("inplace", [False, True])
    def test_a_translating_joint_keeps_its_displacement(self, inplace):
        # Hips' offset goes from (0, 90, 0) to (0, 100, 0): each frame
        # moves by the same (0, 10, 0), so (3, 90, 0) becomes (3, 100, 0).
        clip = _clip([0.0, 3])
        retargeted = clip.retarget(_reference([0, 100, 0], [0, 20, 0]), inplace=inplace) or clip
        assert retargeted.nodes[1].offset.tolist() == [0.0, 100, 0]
        assert retargeted.joint_translations.tolist() == [[[0.0, 100, 0]], [[3.0, 100, 0]]]

    def test_positions_follow_the_new_skeleton(self):
        retargeted = _clip([0.0, 3]).retarget(_reference([1, 100, 0], [0, 20, 0]))
        assert retargeted.node_positions()[1].tolist() == [
            [0.0, 0, 0],
            [4.0, 100, 0],
            [4.0, 120, 0],
            [4.0, 125, 0],
        ]

    def test_an_unmatched_translating_joint_keeps_its_frames(self):
        clip = _clip([0.0, 3])
        retargeted = clip.retarget(
            _reference([0, 100, 0], [0, 20, 0]), name_mapping={"Hips": "Pelvis"}
        )
        assert retargeted.nodes[1].offset.tolist() == [0.0, 90, 0]
        assert retargeted.joint_translations.tolist() == clip.joint_translations.tolist()


class TestCopyOnANewSkeleton:
    def test_it_refuses_to_move_position_channels_to_another_joint(self):
        # Same count of translating joints, but Spine's column would be
        # Hips' frames.
        with pytest.raises(ValueError, match="'Hips'.*translat"):
            _clip().copy(nodes=_rotating_hips_table())

    def test_it_refuses_to_change_the_root_channel_set(self):
        table = _table()
        table[0]["pos_channels"] = []
        table[0]["rot_channels"] = []
        with pytest.raises(ValueError, match="channels of the root 'Reference'"):
            _clip().copy(nodes=table)

    def test_a_new_offset_leaves_the_frames_as_they_are(self):
        # The frames are the joint's whole position: a new rest offset
        # does not move it in any frame.
        table = _table()
        table[1]["offset"] = [0, 100, 0]
        edited = _clip([0.0, 3]).copy(nodes=table)
        assert edited.joint_translations.tolist() == [[[0.0, 90, 0]], [[3.0, 90, 0]]]


ROOT_AND_ROTATION_EDITS = {
    "change_euler_order": lambda clip: clip.change_euler_order("XYZ"),
    "add_rotation_noise": lambda clip: transforms.add_rotation_noise(
        clip, 0.1, rng=np.random.default_rng(0)
    ),
    "translate_root": lambda clip: transforms.translate_root(clip, [1.0, 2, 3]),
    "add_position_noise": lambda clip: transforms.add_position_noise(
        clip, 1.0, rng=np.random.default_rng(0)
    ),
    "rotate_vertical": lambda clip: transforms.rotate_vertical(clip, 90, degrees=True),
}


@pytest.mark.parametrize("edit", ROOT_AND_ROTATION_EDITS.values(), ids=ROOT_AND_ROTATION_EDITS)
def test_rotation_and_root_edits_leave_the_translations_unchanged(edit):
    # The translations are each joint's position in its parent's frame:
    # a rotation, or a move of the root, does not change them.
    edited = edit(_clip([0.0, 3]))
    assert edited.joint_translations.tolist() == [[[0.0, 90, 0]], [[3.0, 90, 0]]]


def _static_root_table(root_offset) -> list[dict]:
    return [
        {
            "name": "Armature",
            "parent": None,
            "offset": root_offset,
            "pos_channels": [],
            "rot_channels": [],
        },
        {"name": "Hips", "parent": 0, "offset": [0, 90, 0], "rot_channels": "ZYX"},
        {"parent": 1, "offset": [0, 5, 0]},
    ]


def test_a_static_root_moves_its_frames_with_its_offset():
    # A static root's root_pos is its offset in every frame: an offset
    # edit moves both, or root_pos goes stale.
    clip = Bvh(
        nodes=nodes_from_table(_static_root_table([0, 0, 0])),
        root_pos=np.zeros((2, 3)),
        joint_angles=np.zeros((2, 2, 3)),
        world_up="+y",
    )
    clip.retarget(Bvh(nodes=nodes_from_table(_static_root_table([1, 2, 3]))), inplace=True)
    assert clip.root_pos.tolist() == [[1.0, 2, 3], [1.0, 2, 3]]


def test_scaling_a_static_root_scales_its_frames_once():
    clip = Bvh(
        nodes=nodes_from_table(_static_root_table([1, 2, 3])),
        root_pos=np.tile([1.0, 2, 3], (2, 1)),
        joint_angles=np.zeros((2, 2, 3)),
        world_up="+y",
    )
    clip.scale(2, inplace=True)
    assert clip.root_pos.tolist() == [[2.0, 4, 6], [2.0, 4, 6]]


def _two_translating_table() -> list[dict]:
    table = _table()
    table[2]["pos_channels"] = "XYZ"
    return table


def _three_channel_table() -> list[dict]:
    table = _table()
    del table[1]["pos_channels"]
    return table


SKELETONS = {"two_translating": _two_translating_table, "three_channel": _three_channel_table}

MOTION_FREE_EDITS = {
    "retarget": lambda clip: clip.retarget(clip.scale(2)),
    "scale": lambda clip: clip.scale(2),
    "rotate_vertical": lambda clip: clip.rotate_vertical(90),
    "reorient_world_up": lambda clip: clip.reorient_world_up("+z"),
    "reorient_rest_up": lambda clip: clip.reorient_rest_up("+z"),
    "reorient_rest_forward": lambda clip: clip.reorient_rest_forward("+x"),
    "change_euler_order": lambda clip: clip.change_euler_order("XYZ", joint="Spine"),
    "copy_nodes": lambda clip: clip.copy(nodes=clip.to_node_table()),
}


@pytest.mark.parametrize("table", SKELETONS.values(), ids=SKELETONS)
@pytest.mark.parametrize("edit", MOTION_FREE_EDITS.values(), ids=MOTION_FREE_EDITS)
def test_a_skeleton_edit_on_a_clip_without_motion_gives_a_clip_without_motion(table, edit):
    clip = Bvh(nodes=nodes_from_table(table()), world_up="+y")
    edited = edit(clip)
    assert edited.frame_count == 0
    # Still the arrays of a clip built without motion, which bind no joint.
    assert edited.joint_angles.shape == (0, 0, 3)
    assert edited.joint_translations.shape == (0, 0, 3)
    assert edited.node_positions().shape == (0, len(edited.nodes), 3)


@pytest.mark.parametrize("table", SKELETONS.values(), ids=SKELETONS)
def test_extract_joints_on_a_clip_without_motion_gives_no_frame(table):
    # As before #186, the result holds no frame of the kept joints.
    clip = Bvh(nodes=nodes_from_table(table()), world_up="+y")
    extracted = clip.extract_joints(["Reference", "Spine"])
    assert extracted.frame_count == 0
    assert extracted.joint_angles.shape == (0, 2, 3)
    assert extracted.node_positions().shape == (0, len(extracted.nodes), 3)


EULER_COLUMNS = {
    "two_translating": "Reference_pos Reference_rot Hips_pos Hips_rot Spine_pos Spine_rot",
    "three_channel": "Reference_pos Reference_rot Hips_rot Spine_rot",
}


@pytest.mark.parametrize("name", SKELETONS)
def test_a_clip_without_motion_gives_its_euler_columns_with_no_row(name):
    clip = Bvh(nodes=nodes_from_table(SKELETONS[name]()), world_up="+y")
    columns = clip.to_df_dict()
    expected = ["time"]
    for block in EULER_COLUMNS[name].split():
        joint, kind = block.split("_")
        axes = "XYZ" if kind == "pos" else "ZYX"
        expected += [f"{joint}_{axis}_{kind}" for axis in axes]
    assert list(columns) == expected
    assert all(len(values) == 0 for values in columns.values())

    coordinates = clip.to_df_dict(mode="coordinates")
    assert all(len(values) == 0 for values in coordinates.values())
