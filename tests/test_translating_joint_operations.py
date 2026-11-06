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

from pybvh import Bvh, nodes_from_table

FRAMES = 6

# Hips' x travels 0, 1, 4, 9, 16, 25 (f squared): not linear, so an
# interpolated frame differs from the frame it replaces.
HIPS_X = np.array([0.0, 1, 4, 9, 16, 25])


def _table() -> list[dict]:
    return [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": "ZYX",
        },
        {"name": "Spine", "parent": 1, "offset": [0, 10, 0], "rot_channels": "ZYX"},
        {"parent": 2, "offset": [0, 5, 0]},
    ]


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

    def test_splicing_refuses_another_set(self):
        clip = _clip([0.0, 1])
        with pytest.raises(ValueError, match="Hips"):
            clip[0:2] = _clip_on(_rotating_hips_table())
