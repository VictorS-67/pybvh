"""Operations that change a clip's skeleton shape or orientation, on a clip
with translating joints (#186).

``extract_joints``, ``mirror`` and the rest and world-up reorientations edit
the offsets; each one also edits ``joint_translations``, so that the forward
kinematics of the result follow from those of the source: equal for the kept
joints of an extraction and for a rest reorientation, reflected for a mirror,
rotated for a world-up reorientation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import Bvh, nodes_from_table, read_bvh_file

SYNTHETIC = Path(__file__).parent.parent / "bvh_data" / "synthetic"

_FRAMES = 4


def _positions_by_name(clip: Bvh) -> dict[str, np.ndarray]:
    """FK positions of every joint, ``(F, 3)`` each, keyed by name."""
    positions = clip.node_positions()
    return {
        node.name: positions[:, i] for i, node in enumerate(clip.nodes) if not node.is_end_site()
    }


# =============================================================================
# extract_joints
# =============================================================================


def _chain_table() -> list[dict]:
    """Reference -> Hips (translates) -> Spine -> Chest (translates) -> Neck -> end."""
    return [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        {"name": "Spine", "parent": 1, "offset": [0, 10, 1], "rot_channels": "ZYX"},
        {
            "name": "Chest",
            "parent": 2,
            "offset": [0, 12, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        {"name": "Neck", "parent": 3, "offset": [0, 8, 0], "rot_channels": "ZYX"},
        {"parent": 4, "offset": [0, 5, 0]},
    ]


def _chain_clip(still: tuple[str, ...]) -> Bvh:
    """A chain clip with random motion; the joints named in ``still`` do not
    rotate, so extracting them away keeps the FK positions of the rest."""
    rng = np.random.default_rng(186)
    nodes = nodes_from_table(_chain_table())
    names = [node.name for node in nodes if not node.is_end_site()]
    angles = rng.uniform(-1, 1, (_FRAMES, len(names), 3))
    for name in still:
        angles[:, names.index(name)] = 0.0
    return Bvh(
        nodes=nodes,
        root_pos=rng.uniform(-5, 5, (_FRAMES, 3)),
        joint_angles=angles,
        joint_translations=rng.uniform(-20, 20, (_FRAMES, 2, 3)) + [[0, 90, 0], [0, 12, 0]],
        frame_time=0.1,
        world_up="+y",
    )


def _assert_kept_positions_equal(clip: Bvh, extracted: Bvh, kept: list[str]) -> None:
    source, result = _positions_by_name(clip), _positions_by_name(extracted)
    for name in kept:
        np.testing.assert_allclose(result[name], source[name], atol=1e-9, err_msg=name)


class TestExtractJoints:
    def test_a_kept_translating_joint_keeps_its_translations(self):
        clip = _chain_clip(still=())
        kept = ["Reference", "Hips", "Spine", "Chest"]
        extracted = clip.extract_joints(kept)
        np.testing.assert_array_equal(extracted.joint_translations, clip.joint_translations)
        _assert_kept_positions_equal(clip, extracted, kept)

    def test_a_removed_translating_joint_moves_its_kept_descendant(self):
        # Hips translates and goes: Spine, below it, takes over its
        # translation plus its own offset, per frame.
        clip = _chain_clip(still=("Hips",))
        kept = ["Reference", "Spine", "Chest", "Neck"]
        extracted = clip.extract_joints(kept)
        spine = extracted.nodes[extracted.index("Spine", space="node")]
        assert spine.pos_channels == ("X", "Y", "Z")
        assert extracted.joint_translations.shape == (_FRAMES, 2, 3)
        np.testing.assert_allclose(
            extracted.joint_translations[:, 0], clip.joint_translations[:, 0] + [0, 10, 1]
        )
        _assert_kept_positions_equal(clip, extracted, kept)

    def test_a_removed_chain_sums_into_a_translating_descendant(self):
        # Hips (translating) and Spine (rotating) go: Chest's translation
        # gains the Hips' translation and the Spine's offset.
        clip = _chain_clip(still=("Hips", "Spine"))
        kept = ["Reference", "Chest", "Neck"]
        extracted = clip.extract_joints(kept)
        assert extracted.joint_translations.shape == (_FRAMES, 1, 3)
        _assert_kept_positions_equal(clip, extracted, kept)

    def test_extraction_keeps_a_three_channel_clip_without_translations(self):
        table = _chain_table()
        for row in table[1:]:
            row.pop("pos_channels", None)
        clip = Bvh(
            nodes=nodes_from_table(table),
            root_pos=np.zeros((_FRAMES, 3)),
            joint_angles=np.zeros((_FRAMES, 5, 3)),
        )
        extracted = clip.extract_joints(["Reference", "Chest"])
        assert extracted.joint_translations.shape == (_FRAMES, 0, 3)
        assert not extracted.nodes[1].pos_channels


# =============================================================================
# mirror
# =============================================================================


def _swap_side(name: str) -> str:
    if name.startswith("Left"):
        return "Right" + name[len("Left") :]
    if name.startswith("Right"):
        return "Left" + name[len("Right") :]
    return name


class TestMirror:
    def test_a_symmetric_translating_pair_mirrors(self):
        # Blender-style: both shoulders and both upper legs translate.
        # Their translations move off the rest pose, unlike each other,
        # so a swap without the negation (or the reverse) shows.
        clip = read_bvh_file(SYNTHETIC / "blender_mixed.bvh")
        rng = np.random.default_rng(5)
        clip.joint_translations = clip.joint_translations + rng.uniform(
            -3, 3, clip.joint_translations.shape
        )
        mirrored = clip.mirror(lateral="x")
        source, result = _positions_by_name(clip), _positions_by_name(mirrored)
        for name, positions in source.items():
            expected = positions * [-1, 1, 1]
            np.testing.assert_allclose(result[_swap_side(name)], expected, atol=1e-9, err_msg=name)

    def test_a_pair_that_translates_on_one_side_is_refused(self):
        # LeftArm translates, RightArm does not.
        clip = read_bvh_file(SYNTHETIC / "synthetic_stretch.bvh")
        with pytest.raises(ValueError, match="LeftArm.*RightArm"):
            clip.mirror(lateral="x")


# =============================================================================
# Reorientation
# =============================================================================


def _stretch_clip() -> Bvh:
    """Three translating joints whose translations change in every frame."""
    clip = read_bvh_file(SYNTHETIC / "synthetic_stretch.bvh")
    clip.world_up = "+y"
    return clip


class TestReorientation:
    def test_world_up_rotates_the_positions(self):
        clip = _stretch_clip()
        turned = clip.reorient_world_up("+z")
        # The quarter turn about X that carries +y onto +z: (x, y, z) -> (x, -z, y).
        x, y, z = np.moveaxis(clip.node_positions(), -1, 0)
        np.testing.assert_allclose(
            turned.node_positions(), np.stack([x, -z, y], axis=-1), atol=1e-9
        )

    @pytest.mark.parametrize(
        ("reorient", "axis"),
        [("reorient_rest_up", "+z"), ("reorient_rest_forward", "+x")],
    )
    def test_a_rest_reorientation_keeps_the_positions(self, reorient, axis):
        clip = _stretch_clip()
        turned = getattr(clip, reorient)(axis)
        np.testing.assert_allclose(turned.node_positions(), clip.node_positions(), atol=1e-9)
