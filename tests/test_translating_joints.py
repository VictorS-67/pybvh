"""Translating joints: position channels on a joint other than the root (#186).

A joint whose ``CHANNELS`` line declares 3 position + 3 rotation channels
is placed, in each frame, at its position channels' values relative to its
parent; its ``OFFSET`` is the rest pose only (ADR 0010). The values are
``Bvh.joint_translations``, shape ``(F, P, 3)``, one column per translating
joint in node order. These tests cover the clip that holds them, the file
that declares them, forward kinematics and the comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import Bvh, nodes_from_table, read_bvh_file

BVH_DATA = Path(__file__).parent.parent / "bvh_data"


def _table(spine_translates: bool = True) -> list[dict]:
    """Reference root -> Hips (translates) -> Spine (translates or not) -> end site."""
    spine = {"name": "Spine", "parent": 1, "offset": [0, 10, 0], "rot_channels": "ZYX"}
    if spine_translates:
        spine["pos_channels"] = "XYZ"
    return [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        spine,
        {"parent": 2, "offset": [0, 5, 0]},
    ]


def _clip(translations=None, frames: int = 2, spine_translates: bool = True) -> Bvh:
    count = 2 if spine_translates else 1
    if translations is None:
        translations = np.tile([[0.0, 90, 0], [0, 10, 0]][:count], (frames, 1, 1))
    return Bvh(
        nodes=nodes_from_table(_table(spine_translates)),
        root_pos=np.zeros((frames, 3)),
        joint_angles=np.zeros((frames, 3, 3)),
        joint_translations=translations,
        frame_time=0.1,
        world_up="+y",
    )


class TestStorage:
    def test_a_clip_holds_its_translations(self):
        translations = np.arange(12.0).reshape(2, 2, 3)
        np.testing.assert_array_equal(_clip(translations).joint_translations, translations)

    def test_a_three_channel_clip_has_no_translating_joint(self):
        clip = read_bvh_file(BVH_DATA / "bvh_test1.bvh")
        assert clip.joint_translations.shape == (clip.frame_count, 0, 3)

    def test_the_keyword_defaults_to_no_translating_joint(self):
        table = _table()
        for row in table:
            row.pop("pos_channels", None)
        table[0]["pos_channels"] = "XYZ"
        clip = Bvh(
            nodes=nodes_from_table(table),
            root_pos=np.zeros((4, 3)),
            joint_angles=np.zeros((4, 3, 3)),
        )
        assert clip.joint_translations.shape == (4, 0, 3)

    def test_a_clip_without_motion_holds_none(self):
        assert Bvh(nodes=nodes_from_table(_table())).joint_translations.shape == (0, 0, 3)

    def test_the_skeleton_fixes_the_translating_joint_count(self):
        with pytest.raises(ValueError, match="joint_translations has 1 translating joints"):
            _clip(np.zeros((2, 1, 3)))

    def test_a_translating_skeleton_needs_its_translations(self):
        with pytest.raises(ValueError, match="joint_translations has 0 translating joints"):
            Bvh(
                nodes=nodes_from_table(_table()),
                root_pos=np.zeros((2, 3)),
                joint_angles=np.zeros((2, 3, 3)),
            )

    def test_the_frame_count_must_match(self):
        with pytest.raises(ValueError, match="disagree on frame count"):
            _clip(np.zeros((3, 2, 3)))

    def test_translations_alone_are_refused(self):
        with pytest.raises(ValueError, match="must be provided together"):
            Bvh(nodes=nodes_from_table(_table()), joint_translations=np.zeros((2, 2, 3)))

    def test_the_view_is_read_only(self):
        with pytest.raises(ValueError, match="read-only"):
            _clip().joint_translations[0, 0, 0] = 1.0

    def test_the_setter_replaces_the_whole_array(self):
        clip = _clip()
        clip.joint_translations = np.ones((2, 2, 3))
        np.testing.assert_array_equal(clip.joint_translations, np.ones((2, 2, 3)))

    def test_the_setter_checks_the_shape_against_the_skeleton(self):
        clip = _clip()
        with pytest.raises(ValueError, match="joint_translations has 1 translating joints"):
            clip.joint_translations = np.zeros((2, 1, 3))
        with pytest.raises(ValueError, match=r"must have shape \(F, P, 3\)"):
            clip.joint_translations = np.zeros((2, 2))

    def test_the_setter_starts_the_world_up_guess_again(self):
        clip = read_bvh_file(BVH_DATA / "bvh_test1.bvh")[0:10]
        assert clip.orientation_report.world_up_inherited is True
        clip.joint_translations = clip.joint_translations
        assert clip.orientation_report.world_up_inherited is False

    def test_the_setter_clears_the_positions(self):
        clip = _clip()
        before = clip.node_positions()
        clip.joint_translations = np.tile([[0.0, 92, 0], [0, 12, 0]], (2, 1, 1))
        np.testing.assert_allclose(clip.node_positions()[:, 3] - before[:, 3], [[0, 4, 0]] * 2)


class TestForwardKinematics:
    def test_positions_follow_the_translations(self):
        # Frame 1: the hips travel to (10, 92, 5) and turn 90 degrees about
        # Z, so the spine (0, 12, 0) above them points along -X.
        clip = _clip(np.array([[[0.0, 92, 0], [0, 12, 0]], [[10.0, 92, 5], [0, 12, 0]]]))
        angles = clip.joint_angles.copy()
        angles[1, 1, 0] = np.pi / 2
        clip.joint_angles = angles
        expected = np.array(
            [
                [[0, 0, 0], [0, 92, 0], [0, 104, 0], [0, 109, 0]],
                [[0, 0, 0], [10, 92, 5], [-2, 92, 5], [-7, 92, 5]],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(clip.node_positions(), expected, atol=1e-12)
        np.testing.assert_allclose(clip.node_positions(frame=1), expected[1], atol=1e-12)

    def test_a_stretching_bone(self):
        lengths = np.array([10.0, 15.0, 20.0])
        translations = np.zeros((3, 2, 3))
        translations[:, 0] = [0, 90, 0]
        translations[:, 1, 1] = lengths
        tip = _clip(translations, frames=3).node_positions()[:, 3]
        np.testing.assert_allclose(tip[:, 1], 90 + lengths + 5)

    def test_the_rest_pose_reads_the_offsets(self):
        rest = _clip(np.full((2, 2, 3), 50.0)).rest_pose_positions()
        np.testing.assert_allclose(rest[:, 1], [0, 90, 100, 105])

    def test_a_rigid_clip_has_the_positions_of_its_three_channel_twin(self):
        rng = np.random.default_rng(0)
        angles = rng.uniform(-1, 1, (5, 3, 3))
        root_pos = rng.uniform(-1, 1, (5, 3))
        rigid = _clip(frames=5, spine_translates=False)
        rigid.root_pos, rigid.joint_angles = root_pos, angles
        table = _table(spine_translates=False)
        del table[1]["pos_channels"]
        twin = Bvh(nodes=nodes_from_table(table), root_pos=root_pos, joint_angles=angles)
        np.testing.assert_array_equal(rigid.node_positions(), twin.node_positions())
