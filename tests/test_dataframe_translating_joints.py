"""DataFrame columns of translating joints and of a static root (#186).

``to_df_dict(mode='euler')`` lays out, per joint in node order, its
position columns if it declares position channels, then its rotation
columns. ``df_to_bvh`` reads which joints translate from the skeleton it
is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybvh import Bvh, df_to_bvh, nodes_from_table, read_bvh_file

BVH_DATA = Path(__file__).parent.parent / "bvh_data"


def _translating_table() -> list[dict]:
    """Reference root -> Hips (translates) -> Spine (rotates) -> end site."""
    return [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZXY",
        },
        {"name": "Spine", "parent": 1, "offset": [0, 10, 0], "rot_channels": "XYZ"},
        {"parent": 2, "offset": [0, 5, 0]},
    ]


def _translating_clip(frames: int = 3) -> Bvh:
    rng = np.random.default_rng(0)
    return Bvh(
        nodes=nodes_from_table(_translating_table()),
        root_pos=rng.normal(size=(frames, 3)),
        joint_angles=rng.uniform(-1, 1, size=(frames, 3, 3)),
        joint_translations=rng.normal(size=(frames, 1, 3)) + [0, 90, 0],
        frame_time=0.1,
        world_up="+y",
    )


class TestTranslatingJoint:
    def test_the_columns(self):
        assert list(_translating_clip().to_df_dict()) == [
            "time",
            "Reference_X_pos",
            "Reference_Y_pos",
            "Reference_Z_pos",
            "Reference_Z_rot",
            "Reference_Y_rot",
            "Reference_X_rot",
            "Hips_X_pos",
            "Hips_Y_pos",
            "Hips_Z_pos",
            "Hips_Z_rot",
            "Hips_X_rot",
            "Hips_Y_rot",
            "Spine_X_rot",
            "Spine_Y_rot",
            "Spine_Z_rot",
        ]

    def test_the_values(self):
        clip = _translating_clip()
        columns = clip.to_df_dict()
        np.testing.assert_array_equal(
            np.stack([columns[f"Hips_{ax}_pos"] for ax in "XYZ"], axis=1),
            clip.joint_translations[:, 0],
        )
        np.testing.assert_allclose(
            np.stack([columns[f"Hips_{ax}_rot"] for ax in "ZXY"], axis=1),
            np.rad2deg(clip.joint_angles[:, 1]),
        )

    @pytest.mark.parametrize("hier", ["table", "nodes"])
    def test_round_trip(self, hier):
        clip = _translating_clip()
        skeleton = clip.to_node_table() if hier == "table" else clip.nodes
        back = df_to_bvh(skeleton, pd.DataFrame(clip.to_df_dict()))
        back.world_up = "+y"
        np.testing.assert_allclose(back.joint_angles, clip.joint_angles, rtol=0, atol=1e-12)
        back.joint_angles = clip.joint_angles
        assert back == clip

    def test_a_missing_position_column_is_refused_by_name(self):
        df = pd.DataFrame(_translating_clip().to_df_dict()).drop(columns="Hips_Y_pos")
        with pytest.raises(ValueError, match="Hips_Y_pos"):
            df_to_bvh(_translating_table(), df)

    def test_the_skeleton_decides_which_joints_translate(self):
        """Position columns of a joint the skeleton does not translate are
        ignored, as any column outside the expected set."""
        df = pd.DataFrame(_translating_clip().to_df_dict())
        df["Spine_X_pos"] = 1.0
        table = _translating_table()
        back = df_to_bvh(table, df)
        assert back.joint_translations.shape == (3, 1, 3)


def _static_root_table() -> list[dict]:
    """A Blender-style static root above a translating Hips joint."""
    return [
        {
            "name": "__0",
            "parent": None,
            "offset": [1, 2, 3],
            "pos_channels": [],
            "rot_channels": [],
        },
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZXY",
        },
        {"parent": 1, "offset": [0, 5, 0]},
    ]


def _static_root_clip(frames: int = 3) -> Bvh:
    rng = np.random.default_rng(1)
    angles = rng.uniform(-1, 1, size=(frames, 2, 3))
    angles[:, 0] = 0
    return Bvh(
        nodes=nodes_from_table(_static_root_table()),
        root_pos=np.tile([1.0, 2, 3], (frames, 1)),
        joint_angles=angles,
        joint_translations=rng.normal(size=(frames, 1, 3)),
        frame_time=0.1,
        world_up="+y",
    )


class TestStaticRoot:
    def test_the_root_has_no_columns(self):
        assert list(_static_root_clip().to_df_dict()) == [
            "time",
            "Hips_X_pos",
            "Hips_Y_pos",
            "Hips_Z_pos",
            "Hips_Z_rot",
            "Hips_X_rot",
            "Hips_Y_rot",
        ]

    # FK on a static root raises until ticket 03 of #186 lands, and the
    # constructor df_to_bvh calls guesses the world up axis by FK.
    @pytest.mark.xfail(
        strict=True, raises=ValueError, reason="FK on a static root: ticket 03 of #186"
    )
    def test_round_trip(self):
        clip = _static_root_clip()
        back = df_to_bvh(clip.to_node_table(), pd.DataFrame(clip.to_df_dict()))
        back.world_up = "+y"
        back.joint_angles = clip.joint_angles
        assert back == clip


@pytest.mark.parametrize(
    "path",
    sorted(BVH_DATA.glob("*.bvh")),
    ids=lambda path: path.name,
)
def test_the_columns_of_a_three_channel_clip_are_unchanged(path):
    """The root's position columns, then each joint's rotation columns."""
    clip = read_bvh_file(path)
    joints = [node for node in clip.nodes if not node.is_end_site()]
    expected = ["time"] + [f"{joints[0].name}_{ax}_pos" for ax in "XYZ"]
    for joint in joints:
        expected += [f"{joint.name}_{ax}_rot" for ax in joint.rot_channels]
    assert list(clip.to_df_dict()) == expected
