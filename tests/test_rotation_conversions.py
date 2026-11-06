"""The rotation conversions: ``to_*`` and ``from_*`` carry rotations only (#186).

``to_rotmat``, ``to_6d``, ``to_quat`` and ``to_axisangle`` return the
rotation array; ``from_rotmat``, ``from_6d``, ``from_quat`` and
``from_axisangle`` take it back, write ``joint_angles`` only, and keep
``root_pos`` and ``joint_translations``. A clip with a static root and a
translating joint shows that the arrays the conversions do not touch stay
as they were.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file, rotations

BVH_DATA_DIR = Path(__file__).parent.parent / "bvh_data"

# A static root 2 above the origin; under it, the hips translate and turn,
# and a prop only rotates.
_BLENDER_STYLE = """HIERARCHY
ROOT __0
{
\tOFFSET 0 2 0
\tCHANNELS 0
\tJOINT Hips
\t{
\t\tOFFSET 0 90 0
\t\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 10 0
\t\t}
\t}
\tJOINT Prop
\t{
\t\tOFFSET 30 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 4 0
\t\t}
\t}
}
MOTION
Frames: 3
Frame Time: 0.1
0 90 0 0 0 0 0 0 0
10 90 5 90 0 0 0 0 90
-3 85 2 30 -60 170 45 89 -10
"""

# (name, the representation's dimensions after (F, J), its to_* and from_*)
REPRESENTATIONS = {
    "rotmat": ((3, 3), "to_rotmat", "from_rotmat"),
    "6d": ((6,), "to_6d", "from_6d"),
    "quat": ((4,), "to_quat", "from_quat"),
    "axisangle": ((3,), "to_axisangle", "from_axisangle"),
}


@pytest.fixture
def static_root_clip(tmp_path):
    path = tmp_path / "blender_style.bvh"
    path.write_text(_BLENDER_STYLE)
    return read_bvh_file(path)


@pytest.fixture
def moving_root_clip():
    return read_bvh_file(BVH_DATA_DIR / "bvh_test1.bvh")


def _rotmats(clip):
    return rotations.euler_to_rotmat(clip.joint_angles, clip.euler_orders)


@pytest.mark.parametrize("name", REPRESENTATIONS)
class TestEachRepresentation:
    def test_to_returns_the_rotation_array_only(self, name, static_root_clip):
        dims, to, _ = REPRESENTATIONS[name]
        rot = getattr(static_root_clip, to)()
        assert isinstance(rot, np.ndarray)
        assert rot.shape == (3, 3, *dims)

    @pytest.mark.parametrize("clip_name", ["static_root_clip", "moving_root_clip"])
    def test_from_of_to_keeps_the_clip(self, name, clip_name, request):
        clip = request.getfixturevalue(clip_name)
        _, to, frm = REPRESENTATIONS[name]
        back = getattr(clip, frm)(getattr(clip, to)())
        np.testing.assert_array_equal(back.root_pos, clip.root_pos)
        np.testing.assert_array_equal(back.joint_translations, clip.joint_translations)
        np.testing.assert_allclose(_rotmats(back), _rotmats(clip), atol=1e-10)

    def test_from_writes_the_new_rotations(self, name, static_root_clip):
        # Turn the prop 90 degrees about Y in every frame: the hips keep
        # their translations, the root its offset.
        _, to, frm = REPRESENTATIONS[name]
        target = _rotmats(static_root_clip)
        target[:, 2] = rotations.euler_to_rotmat(np.radians([0.0, 90.0, 0.0]), "XYZ")
        rot = {
            "rotmat": target,
            "6d": rotations.rotmat_to_rot6d(target),
            "quat": rotations.rotmat_to_quat(target),
            "axisangle": rotations.rotmat_to_axisangle(target),
        }[name]
        back = getattr(static_root_clip, frm)(rot)
        np.testing.assert_allclose(_rotmats(back), target, atol=1e-10)
        np.testing.assert_array_equal(back.root_pos, static_root_clip.root_pos)
        np.testing.assert_array_equal(back.joint_translations, static_root_clip.joint_translations)

    def test_from_in_place_returns_none(self, name, static_root_clip):
        _, to, frm = REPRESENTATIONS[name]
        rot = getattr(static_root_clip, to)()
        assert getattr(static_root_clip, frm)(rot, inplace=True) is None

    def test_a_stale_two_array_call_names_the_new_signature(self, name, moving_root_clip):
        _, to, frm = REPRESENTATIONS[name]
        rot = getattr(moving_root_clip, to)()
        with pytest.raises(TypeError, match=rf"{frm}\(joint_\w+, inplace=False\)"):
            getattr(moving_root_clip, frm)(moving_root_clip.root_pos, rot)

    def test_from_refuses_a_rotation_of_the_static_root(self, name, static_root_clip):
        _, to, frm = REPRESENTATIONS[name]
        target = _rotmats(static_root_clip)
        target[1, 0] = rotations.euler_to_rotmat(np.radians([0.0, 0.0, 10.0]), "XYZ")
        rot = {
            "rotmat": target,
            "6d": rotations.rotmat_to_rot6d(target),
            "quat": rotations.rotmat_to_quat(target),
            "axisangle": rotations.rotmat_to_axisangle(target),
        }[name]
        with pytest.raises(ValueError, match=rf"{frm} rotates the root.*'__0' is static"):
            getattr(static_root_clip, frm)(rot)

    def test_from_refuses_another_frame_count(self, name, moving_root_clip):
        _, to, frm = REPRESENTATIONS[name]
        rot = getattr(moving_root_clip, to)()
        with pytest.raises(ValueError, match="125 frames.*126"):
            getattr(moving_root_clip, frm)(rot[:-1])


def test_from_starts_the_world_up_guess_again():
    """A from_* call is a user motion replacement: a slice that reported its
    source's world up guesses again from its own motion (ADR 0003)."""
    clip = read_bvh_file(BVH_DATA_DIR / "bvh_test3.bvh", warn_on_world_up_disagreement=False)
    rotmats = clip.to_rotmat()
    tilt = rotations.euler_to_rotmat(np.radians([45.0, 0.0, 0.0]), "XYZ")
    rotmats[5:, 0] = tilt @ rotmats[5:, 0]
    tilted = clip.from_rotmat(rotmats)
    sliced = tilted[5:]
    assert sliced.world_up == "+z"
    assert sliced.from_quat(sliced.to_quat()).world_up == "+y"
