"""Translating joints: position channels on a joint other than the root (#186).

A joint whose ``CHANNELS`` line declares 3 position + 3 rotation channels
is placed, in each frame, at its position channels' values relative to its
parent; its ``OFFSET`` is the rest pose only (ADR 0010). The values are
``Bvh.joint_translations``, shape ``(F, P, 3)``, one column per translating
joint in node order. These tests cover the clip that holds them, the file
that declares them, forward kinematics and the comparisons.
"""

from __future__ import annotations

import hashlib
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


# =============================================================================
# The file: reading, writing, round trip
# =============================================================================

_HIPS_TRAVEL = """\
HIERARCHY
ROOT Reference
{
\tOFFSET 0 0 0
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\tJOINT Hips
\t{
\t\tOFFSET 0 90 0
\t\tCHANNELS 6 {hips_channels}
\t\tJOINT Spine
\t\t{
\t\t\tOFFSET 0 10 0
\t\t\tCHANNELS {spine_channels}
\t\t\tEnd Site
\t\t\t{
\t\t\t\tOFFSET 0 5 0
\t\t\t}
\t\t}
\t}
}
MOTION
Frames: 2
Frame Time: 0.1
"""

_XYZ_ZYX = "Xposition Yposition Zposition Zrotation Yrotation Xrotation"


def _write(tmp_path, text: str, name: str = "clip.bvh") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _hips_travel(
    tmp_path,
    hips_channels=_XYZ_ZYX,
    hips_rows=None,
    spine_channels="3 Zrotation Yrotation Xrotation",
    spine_rows=None,
):
    """The reference root at rest, the hips travelling and turning 90
    degrees about Z in frame 1; *hips_rows* are the hips' six values per
    frame in the order *hips_channels* declares them."""
    if hips_rows is None:
        hips_rows = ["0 90 0 0 0 0", "10 92 5 90 0 0"]
    if spine_rows is None:
        spine_rows = ["0 0 0", "0 0 0"]
    text = _HIPS_TRAVEL.replace("{hips_channels}", hips_channels).replace(
        "{spine_channels}", spine_channels
    )
    text += "".join(
        f"1 0 0 0 0 0 {hips} {spine}\n" if f else f"0 0 0 0 0 0 {hips} {spine}\n"
        for f, (hips, spine) in enumerate(zip(hips_rows, spine_rows, strict=True))
    )
    return _write(tmp_path, text)


def _assert_round_trip(clip: Bvh, tmp_path) -> None:
    """write -> read gives an equal clip, and a second write the same bytes."""
    first = tmp_path / "first.bvh"
    clip.write(first)
    reread = read_bvh_file(first)
    assert reread == clip
    second = tmp_path / "second.bvh"
    reread.write(second)
    assert second.read_bytes() == first.read_bytes()


class TestReading:
    def test_a_travelling_hips_joint(self, tmp_path):
        clip = read_bvh_file(_hips_travel(tmp_path))
        assert clip.nodes[1].pos_channels == ("X", "Y", "Z")
        assert clip.nodes[2].pos_channels == ()
        np.testing.assert_array_equal(clip.joint_translations, [[[0, 90, 0]], [[10, 92, 5]]])
        np.testing.assert_array_equal(clip.root_pos, [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(np.rad2deg(clip.joint_angles[:, 1]), [[0, 0, 0], [90, 0, 0]])
        # By hand: frame 1, the reference at (1, 0, 0), the hips 10 92 5
        # from it, the spine and the tip along -X once the hips turn.
        expected = np.array(
            [
                [[0, 0, 0], [0, 90, 0], [0, 100, 0], [0, 105, 0]],
                [[1, 0, 0], [11, 92, 5], [1, 92, 5], [-4, 92, 5]],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(clip.node_positions(), expected, atol=1e-12)

    def test_a_stretching_bone(self, tmp_path):
        path = _hips_travel(
            tmp_path,
            hips_rows=["0 90 0 0 0 0", "0 90 0 0 0 0"],
            spine_channels="6 " + _XYZ_ZYX,
            spine_rows=["0 10 0 0 0 0", "0 25 0 0 0 0"],
        )
        clip = read_bvh_file(path)
        tip = clip.node_positions()[:, 3] - clip.root_pos
        np.testing.assert_allclose(tip, [[0, 105, 0], [0, 120, 0]], atol=1e-12)

    def test_position_channels_in_another_order(self, tmp_path):
        path = _hips_travel(
            tmp_path,
            hips_channels="Zposition Xposition Yposition Zrotation Yrotation Xrotation",
            hips_rows=["0 0 90 0 0 0", "5 10 92 90 0 0"],
        )
        clip = read_bvh_file(path)
        assert clip.nodes[1].pos_channels == ("X", "Y", "Z")
        np.testing.assert_array_equal(clip.joint_translations, [[[0, 90, 0]], [[10, 92, 5]]])

    def test_rotation_channels_first(self, tmp_path):
        path = _hips_travel(
            tmp_path,
            hips_channels="Zrotation Yrotation Xrotation Xposition Yposition Zposition",
            hips_rows=["0 0 0 0 90 0", "90 0 0 10 92 5"],
        )
        clip = read_bvh_file(path)
        np.testing.assert_array_equal(clip.joint_translations, [[[0, 90, 0]], [[10, 92, 5]]])
        np.testing.assert_allclose(np.rad2deg(clip.joint_angles[1, 1]), [90, 0, 0])

    def test_interleaved_channels_read_translate_then_rotate(self, tmp_path):
        path = _hips_travel(
            tmp_path,
            hips_channels="Zrotation Xposition Xrotation Yposition Yrotation Zposition",
            hips_rows=["0 0 0 90 0 0", "90 10 0 92 0 5"],
        )
        clip = read_bvh_file(path)
        assert clip.nodes[1].rot_channels == ("Z", "X", "Y")
        np.testing.assert_array_equal(clip.joint_translations, [[[0, 90, 0]], [[10, 92, 5]]])
        np.testing.assert_allclose(np.rad2deg(clip.joint_angles[1, 1]), [90, 0, 0])
        np.testing.assert_allclose(clip.node_positions()[1, 2], [1, 92, 5], atol=1e-12)

    def test_a_joint_without_position_values_matches_its_offset_twin(self, tmp_path):
        rigid = read_bvh_file(
            _hips_travel(tmp_path, spine_channels="6 " + _XYZ_ZYX, spine_rows=["0 10 0 0 0 0"] * 2)
        )
        twin = read_bvh_file(_hips_travel(tmp_path))
        np.testing.assert_array_equal(rigid.node_positions(), twin.node_positions())
        assert rigid.joint_translations.shape == (2, 2, 3)

    @pytest.mark.parametrize(
        "channels",
        [
            "2 Xposition Yposition",
            "3 Xposition Yposition Zposition",
            "6 Zrotation Yrotation Xrotation Zrotation Yrotation Xrotation",
            "5 Xposition Yposition Zposition Zrotation Yrotation",
            "4 Xposition Zrotation Yrotation Xrotation",
            "6 Xposition Xposition Yposition Zrotation Yrotation Xrotation",
            "0",
        ],
        ids=[
            "two_positions",
            "positions_only",
            "six_rotations",
            "two_rotations",
            "one_position",
            "repeated_axis",
            "no_channel",
        ],
    )
    def test_a_refused_declaration_names_the_joint(self, tmp_path, channels):
        path = _hips_travel(tmp_path, spine_channels=channels)
        with pytest.raises(
            ValueError, match="Could not read the offset or channels of the joint Spine"
        ):
            read_bvh_file(path)


class TestWriting:
    def test_a_translating_joint_is_written_with_its_position_channels(self, tmp_path):
        clip = read_bvh_file(
            _hips_travel(
                tmp_path,
                hips_channels="Zposition Xposition Yposition Zrotation Yrotation Xrotation",
                hips_rows=["0 0 90 0 0 0", "5 10 92 90 0 0"],
            )
        )
        out = tmp_path / "out.bvh"
        clip.write(out)
        text = out.read_text()
        assert (
            "\t\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n" in text
        )
        assert "\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation\n" in text
        rows = text.splitlines()[-2:]
        assert rows[1].split()[6:12] == [
            "10.000000",
            "92.000000",
            "5.000000",
            "90.000000",
            "0.000000",
            "0.000000",
        ]

    @pytest.mark.parametrize(
        ("hips_channels", "hips_rows"),
        [
            (_XYZ_ZYX, None),
            (
                "Zposition Xposition Yposition Zrotation Yrotation Xrotation",
                ["0 0 90 0 0 0", "5 10 92 90 0 0"],
            ),
            (
                "Zrotation Xposition Xrotation Yposition Yrotation Zposition",
                ["0 0 0 90 0 0", "90 10 0 92 0 5"],
            ),
        ],
        ids=["xyz", "zxy_positions", "interleaved"],
    )
    def test_round_trip(self, tmp_path, hips_channels, hips_rows):
        clip = read_bvh_file(
            _hips_travel(tmp_path, hips_channels=hips_channels, hips_rows=hips_rows)
        )
        _assert_round_trip(clip, tmp_path)

    def test_round_trip_of_a_stretching_bone(self, tmp_path):
        clip = read_bvh_file(
            _hips_travel(
                tmp_path,
                spine_channels="6 " + _XYZ_ZYX,
                spine_rows=["0 10 0 0 0 0", "0 25 0 0 0 0"],
            )
        )
        _assert_round_trip(clip, tmp_path)


# The SHA-256 of each 3-channel clip's motion as read (root_pos then
# joint_angles, as bytes) and of its file as written, computed on the code
# before #186 let a joint other than the root translate.
_DIGESTS_BEFORE_186 = {
    "bvh_data/bvh_test1.bvh": (
        "cb7c826da269cc23f350768ad6e0c8d79459a22b5c1cc126da18b820516ac335",
        "09184d649ef8293b9e7557ede84d9e0c4e1c77eb74e3fa1d27c57c04269d5118",
    ),
    "bvh_data/bvh_test2.bvh": (
        "1a929cf239e0ac5f38367dbf82d7e4bd290f17d93885efb26191f0a73b7ffa80",
        "6df524ff87de75bfd08f6e08a8ad4b9d6b71cb7901b15e8a72a32493ff12f49d",
    ),
    "bvh_data/bvh_test3.bvh": (
        "326adac285c0821088a3ccd05b6bded676b5740327adc3801556f9bd25892591",
        "e573cb6e952f384bdc1ad46aaebe8852a100be3f8069804a55f86b805730928c",
    ),
    "bvh_data/bvh_test4.bvh": (
        "0e36d1ca675bfe4a99890f85f480bba9321d1171be8b51d6b7d11491547ade5c",
        "58788ba2dad66c4712700f572fd3431fb0020360de455f8cb943b6bafd754c8b",
    ),
    "bvh_data/cmu_12_01_walk.bvh": (
        "01b906624ef73e2837f4c9b736481222deded51e78dfa71456f6506ceb1b3bf5",
        "05187a97aa09dd6c31d985d15c73acb14e5ae30703b397e2302c6ecc4ae81b36",
    ),
    "bvh_data/standard_skeleton.bvh": (
        "bd50e12c55dda3ee443c1cb6d71c7bcf6351c4ec96f7bc8d6adec015d1192eea",
        "3c379cbec8029c296fc083beb662669c22fe0f6a8102db64740305020f2ff1e5",
    ),
    "bvh_data/synthetic/bvh_synthetic_lying_down.bvh": (
        "d2c06565e878e786b4aa3b05dbc7138e3317c208cfa45c8fd74b6fd2213a83c5",
        "07c97915d56cefc9ed846af191b5c88bb52d968be1795162bc637bda5e2d9704",
    ),
    "bvh_data/synthetic/reference_props_rotation_only.bvh": (
        "efc68e23a0b6d04d8cb36264a6fef6e106961bf2729e40f28c803e26941b06be",
        "eb168e81b4676045db8af759da73c4f951d07f9ee2e7399a3b5125d3d38e8642",
    ),
    "bvh_data/synthetic/rotation_only_control.bvh": (
        "5c2d56d5d06645f20a182427cf4bb789ac44d484a42dfc0d257560256830e5c2",
        "b9b29df9396629779b9b6a308c3a82c00f17a4e4b4b4f2b86c58c53fedf7fbc7",
    ),
    "tests/fixtures/full_precision_frame_time.bvh": (
        "765571a587ccc88e864c5368118a027dce3cf24984796b6c7fcecfee3e31d0b7",
        "c07930d3be4dca5e084be81c29406f9b559e3f4e896c841d68149a3018542f9d",
    ),
    "tests/fixtures/rotation_first_root.bvh": (
        "abe2876717a859ffff27c063c026a3c5bf46dc70fa1d8b1437b5adde8f037116",
        "c6452a49f364323a5a8ea7f3e04761b63fc0c9993c32082f363af5fc2c890492",
    ),
}


@pytest.mark.filterwarnings("ignore:Rest pose suggests world up")
@pytest.mark.parametrize("relative", sorted(_DIGESTS_BEFORE_186))
def test_a_three_channel_clip_reads_and_writes_as_before(relative, tmp_path):
    clip = read_bvh_file(BVH_DATA.parent / relative)
    read_digest, written_digest = _DIGESTS_BEFORE_186[relative]
    motion = clip.root_pos.tobytes() + clip.joint_angles.tobytes()
    assert hashlib.sha256(motion).hexdigest() == read_digest
    out = tmp_path / "out.bvh"
    clip.write(out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == written_digest
    np.testing.assert_array_equal(clip.fk_topology.translation_idx, -1)
