"""A static root: a root whose ``CHANNELS`` line declares no channel (#186).

Blender's exporter writes ``ROOT __0``, ``OFFSET 0 0 0``, ``CHANNELS 0``
above an armature with several root bones. Such a root stays at its
``OFFSET`` with no rotation in every frame: ``root_pos`` holds the offset
and its row of ``joint_angles`` is zero. These tests cover the file that
declares it, forward kinematics, the write back, and the operations and
setters that would move it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from six_channel_fixtures import blender_style_text

from pybvh import Bvh, harmonize, nodes_from_table, read_bvh_file

_BLENDER_STYLE = blender_style_text(2)


def _blender_style_file(tmp_path) -> Path:
    path = tmp_path / "blender_style.bvh"
    path.write_text(_BLENDER_STYLE)
    return path


class TestReading:
    def test_the_root_is_static(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        assert clip.root.is_static()
        assert clip.joint_names == ["__0", "Hips", "Prop"]

    def test_the_root_stays_at_its_offset_with_no_rotation(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        np.testing.assert_array_equal(clip.root_pos, [[0, 2, 0], [0, 2, 0]])
        np.testing.assert_array_equal(clip.joint_angles[:, 0], np.zeros((2, 3)))

    def test_the_other_joints_read_their_values(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        np.testing.assert_array_equal(clip.joint_translations, [[[0, 90, 0]], [[10, 90, 5]]])
        np.testing.assert_allclose(
            np.rad2deg(clip.joint_angles[:, 1:]),
            [[[0, 0, 0], [0, 0, 0]], [[90, 0, 0], [0, 0, 90]]],
        )

    def test_forward_kinematics(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        # By hand. Nodes: __0, Hips, its tip, Prop, its tip. Frame 1: the
        # hips at 10 92 5, turned 90 degrees about Z, so their tip 10
        # along -X; the prop at 30 2 0, turned 90 degrees about X, so its
        # tip 4 along +Z.
        expected = np.array(
            [
                [[0, 2, 0], [0, 92, 0], [0, 102, 0], [30, 2, 0], [30, 6, 0]],
                [[0, 2, 0], [10, 92, 5], [0, 92, 5], [30, 2, 0], [30, 2, 4]],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(clip.node_positions(), expected, atol=1e-12)


class TestWriting:
    def test_the_root_is_written_with_no_channel(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        out = tmp_path / "out.bvh"
        clip.write(out)
        lines = [line.strip() for line in out.read_text().splitlines()]
        assert lines[1:5] == ["ROOT __0", "{", "OFFSET 0.000000 2.000000 0.000000", "CHANNELS 0"]
        # The frame lines hold the hips' 6 values and the prop's 3.
        assert lines[-1].split() == [
            "10.000000",
            "90.000000",
            "5.000000",
            "90.000000",
            "0.000000",
            "0.000000",
            "0.000000",
            "0.000000",
            "90.000000",
        ]

    def test_a_round_trip_gives_an_equal_clip_and_stable_bytes(self, tmp_path):
        clip = read_bvh_file(_blender_style_file(tmp_path))
        first = tmp_path / "first.bvh"
        clip.write(first)
        reread = read_bvh_file(first)
        assert reread == clip
        second = tmp_path / "second.bvh"
        reread.write(second)
        assert second.read_bytes() == first.read_bytes()

    def test_a_static_root_on_a_joint_other_than_the_root_stays_refused(self, tmp_path):
        path = tmp_path / "joint_channels_0.bvh"
        path.write_text(
            _BLENDER_STYLE.replace("CHANNELS 3 Zrotation Yrotation Xrotation", "CHANNELS 0")
        )
        with pytest.raises(ValueError, match="channels of the joint Prop"):
            read_bvh_file(path)


_STATIC = "root '__0' is static"


@pytest.fixture
def clip(tmp_path):
    return read_bvh_file(_blender_style_file(tmp_path))


class TestRefusals:
    """The operations and setters that would move or turn the static root."""

    def test_translate_root(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.translate_root([1, 0, 0])

    def test_add_position_noise(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.add_position_noise(1.0, rng=np.random.default_rng(0))

    def test_rotate_vertical(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.rotate_vertical(np.pi / 2)

    def test_the_root_pos_setter(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.root_pos = np.zeros((2, 3))

    def test_the_joint_angles_setter_refuses_a_rotation_of_the_root(self, clip):
        angles = clip.joint_angles.copy()
        angles[1, 0, 2] = 0.1
        with pytest.raises(ValueError, match=_STATIC):
            clip.joint_angles = angles

    def test_the_joint_angles_setter_takes_the_other_joints(self, clip):
        angles = clip.joint_angles.copy()
        angles[1, 2] = [0.1, 0.2, 0.3]
        clip.joint_angles = angles
        np.testing.assert_array_equal(clip.joint_angles, angles)

    @pytest.mark.parametrize("joint", ["__0", 0])
    def test_change_euler_order_of_the_root(self, clip, joint):
        selector = clip.nodes[0] if joint == 0 else joint
        with pytest.raises(ValueError, match=_STATIC):
            clip.change_euler_order("XYZ", joint=selector)

    def test_change_euler_order_of_another_joint(self, clip):
        changed = clip.change_euler_order("XYZ", joint="Prop")
        assert changed.nodes[3].rot_channels == ("X", "Y", "Z")
        assert changed.root.is_static()

    def test_the_constructor_refuses_a_root_away_from_its_offset(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            Bvh(
                nodes=nodes_from_table(clip.to_node_table()),
                root_pos=np.zeros((2, 3)),
                joint_angles=clip.joint_angles,
                joint_translations=clip.joint_translations,
            )

    def test_the_constructor_refuses_a_rotation_of_the_root(self, clip):
        angles = clip.joint_angles.copy()
        angles[0, 0, 1] = 0.1
        with pytest.raises(ValueError, match=_STATIC):
            Bvh(
                nodes=nodes_from_table(clip.to_node_table()),
                root_pos=clip.root_pos,
                joint_angles=angles,
                joint_translations=clip.joint_translations,
            )


class TestRotationNoise:
    def test_the_root_row_stays_zero(self, clip):
        noised = clip.add_rotation_noise(0.1, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(noised.joint_angles[:, 0], np.zeros((2, 3)))

    def test_the_other_joints_get_noise(self, clip):
        noised = clip.add_rotation_noise(0.1, rng=np.random.default_rng(0))
        assert np.all(noised.joint_angles[:, 1:] != clip.joint_angles[:, 1:])


class TestOffsetEdits:
    def test_a_copy_with_a_new_root_offset_moves_root_pos(self, clip):
        table = clip.to_node_table()
        table[0]["offset"] = [5.0, 0.0, 0.0]
        edited = clip.copy(nodes=table)
        np.testing.assert_array_equal(edited.root_pos, [[5, 0, 0], [5, 0, 0]])
        # The whole body moves with the root: by hand, (5, -2, 0) from the source.
        np.testing.assert_allclose(
            edited.node_positions(), clip.node_positions() + [5, -2, 0], atol=1e-12
        )
        np.testing.assert_array_equal(clip.root_pos, [[0, 2, 0], [0, 2, 0]])


class TestReorientation:
    def test_reorient_rest_up_refuses(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.reorient_rest_up("+z")

    def test_reorient_rest_forward_refuses(self, clip):
        with pytest.raises(ValueError, match=_STATIC):
            clip.reorient_rest_forward("+x")

    def test_reorient_world_up_turns_the_root_with_the_world(self, clip):
        clip.world_up = "+y"
        turned = clip.reorient_world_up("+z")
        assert turned.root.is_static()
        # +Y up to +Z up turns (x, y, z) into (x, -z, y): by hand, the
        # root's offset 0 2 0 becomes 0 0 2.
        np.testing.assert_array_equal(turned.root.offset, [0, 0, 2])
        np.testing.assert_array_equal(turned.root_pos, [[0, 0, 2], [0, 0, 2]])
        np.testing.assert_array_equal(turned.joint_angles[:, 0], np.zeros((2, 3)))
        before = clip.node_positions()
        after = np.stack([before[..., 0], -before[..., 2], before[..., 1]], axis=-1)
        np.testing.assert_allclose(turned.node_positions(), after, atol=1e-9)


class TestHarmonize:
    """``harmonize(target_euler_order=...)`` re-expresses the rotating joints.

    A static root declares no rotation channel, so it has no order to
    change: it stays static, and its row of ``joint_angles`` stays zero.
    """

    def test_the_other_joints_take_the_target_order(self, clip):
        (harmonized,), report = harmonize(
            [clip], target_euler_order="XYZ", verbose=False, return_report=True
        )
        assert harmonized.root.is_static()
        assert [node.rot_channels for node in harmonized.nodes if not node.is_end_site()] == [
            (),
            ("X", "Y", "Z"),
            ("X", "Y", "Z"),
        ]
        np.testing.assert_allclose(harmonized.node_positions(), clip.node_positions(), atol=1e-9)
        assert report.applied_stages[0] == {"euler_order": "→XYZ"}

    def test_a_clip_whose_joints_have_the_target_order_is_left_alone(self, clip):
        # The joints rotate in ZYX; the static root's '' in euler_orders is
        # no declared order, so it asks for no change.
        (harmonized,), report = harmonize(
            [clip], target_euler_order="ZYX", verbose=False, return_report=True
        )
        assert harmonized == clip
        assert report.applied_stages[0] == {}


class TestEulerOrder:
    """A static root declares no rotation channel: its Euler order is ``''``."""

    def test_the_clip_and_its_topology_read_the_empty_order(self, clip):
        assert clip.euler_orders == ["", "ZYX", "ZYX"]
        assert clip.fk_topology.euler_orders == ["", "ZYX", "ZYX"]

    def test_change_euler_order_changes_every_other_joint(self, clip):
        changed = clip.change_euler_order("XYZ")
        assert changed.euler_orders == ["", "XYZ", "XYZ"]
        assert changed.root.is_static()
        np.testing.assert_array_equal(changed.joint_angles[:, 0], np.zeros((2, 3)))
        np.testing.assert_allclose(changed.node_positions(), clip.node_positions(), atol=1e-9)

    def test_change_euler_order_in_place(self, clip):
        positions = clip.node_positions()
        assert clip.change_euler_order("XZY", inplace=True) is None
        assert clip.euler_orders == ["", "XZY", "XZY"]
        np.testing.assert_allclose(clip.node_positions(), positions, atol=1e-9)

    def test_harmonize_gives_the_same_clip(self, clip):
        (harmonized,) = harmonize([clip], target_euler_order="XYZ", verbose=False)
        assert harmonized == clip.change_euler_order("XYZ")

    def test_the_rotation_conversions_read_the_identity(self, clip):
        identity = np.broadcast_to(np.eye(3), (2, 3, 3))
        np.testing.assert_array_equal(clip.to_rotmat()[:, 0], identity)
        np.testing.assert_array_equal(clip.to_quat()[:, 0], [[1, 0, 0, 0]] * 2)
        np.testing.assert_array_equal(clip.to_6d()[:, 0], [[1, 0, 0, 0, 1, 0]] * 2)
        np.testing.assert_array_equal(clip.to_axisangle()[:, 0], np.zeros((2, 3)))

    @pytest.mark.parametrize("representation", ["rotmat", "quat", "6d", "axisangle"])
    def test_a_round_trip_through_the_rotation_conversions(self, clip, representation):
        rotation = getattr(clip, f"to_{representation}")()
        back = getattr(clip, f"from_{representation}")(rotation)
        np.testing.assert_array_equal(back.joint_angles[:, 0], np.zeros((2, 3)))
        np.testing.assert_allclose(back.node_positions(), clip.node_positions(), atol=1e-9)

    def test_from_rotmat_refuses_a_rotation_of_the_root(self, clip):
        rotmats = clip.to_rotmat()
        rotmats[1, 0] = rotmats[1, 1]
        with pytest.raises(ValueError, match=_STATIC):
            clip.from_rotmat(rotmats)

    def test_resample_interpolates_the_other_joints(self, clip):
        resampled = clip.resample(clip.fps * 2)
        assert resampled.frame_count == 3
        np.testing.assert_array_equal(resampled.joint_angles[:, 0], np.zeros((3, 3)))
        # By hand: the prop turns 90 degrees about X between the frames, so 45 halfway.
        np.testing.assert_allclose(np.rad2deg(resampled.joint_angles[1, 2]), [0, 0, 45], atol=1e-9)

    def test_root_trajectory_reads_the_root_as_the_identity(self, clip):
        from pybvh.analysis import root_trajectory

        trajectory = root_trajectory(clip)
        # The root never turns: its heading is the same in both frames.
        np.testing.assert_array_equal(trajectory[0, 2:], trajectory[1, 2:])
