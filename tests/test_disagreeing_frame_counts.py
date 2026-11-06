"""A clip whose motion arrays disagree on frame count is never packed.

The setters check no frame count, so ``clip.joint_angles =
clip.joint_angles[:1]`` leaves a clip whose arrays disagree. Every path
that packs the motion into columns refuses it with the frame-count
``ValueError`` of the motion list, as ``node_positions`` and ``copy`` do,
instead of broadcasting the shorter array across the frames.
"""

from __future__ import annotations

import numpy as np
import pytest
from six_channel_fixtures import reference_hips_table

from pybvh import Bvh, nodes_from_table
from pybvh.features import to_feature_array

FRAMES = 4


def _clip() -> Bvh:
    """Reference root -> translating Hips -> Spine, ``FRAMES`` frames."""
    rng = np.random.default_rng(0)
    return Bvh(
        nodes=nodes_from_table(reference_hips_table()),
        root_pos=rng.normal(size=(FRAMES, 3)),
        joint_angles=rng.uniform(-1, 1, size=(FRAMES, 3, 3)),
        joint_translations=rng.normal(size=(FRAMES, 1, 3)) + [0, 90, 0],
        frame_time=0.1,
        world_up="+y",
    )


def _disagreeing_clip(array: str, frames: int) -> Bvh:
    """``_clip()`` with *array* reassigned to *frames* frames."""
    clip = _clip()
    value = getattr(clip, array)
    # Frames cut off, or the last frame repeated.
    resized = value[:frames] if frames <= FRAMES else np.concatenate([value, value[-1:]])
    setattr(clip, array, resized)
    return clip


DISAGREEMENTS = [
    pytest.param(array, frames, id=f"{array}-{frames}")
    for array in ["root_pos", "joint_angles", "joint_translations"]
    for frames in [1, FRAMES + 1]
]


@pytest.mark.parametrize(("array", "frames"), DISAGREEMENTS)
class TestDisagreeingFrameCounts:
    def test_to_df_dict_refuses_the_clip(self, array, frames):
        clip = _disagreeing_clip(array, frames)
        with pytest.raises(ValueError, match="disagree on frame count"):
            clip.to_df_dict()

    def test_the_writer_refuses_the_clip_and_leaves_the_file_alone(self, array, frames, tmp_path):
        path = tmp_path / "clip.bvh"
        path.write_text("an earlier file")
        clip = _disagreeing_clip(array, frames)
        with pytest.raises(ValueError, match="disagree on frame count"):
            clip.write(path)
        assert path.read_text() == "an earlier file"

    def test_the_feature_array_refuses_the_clip(self, array, frames):
        clip = _disagreeing_clip(array, frames)
        with pytest.raises(ValueError, match="disagree on frame count"):
            to_feature_array(clip, include_joint_translations=True)
