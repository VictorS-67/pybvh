"""The body of a clip: what is looked for in it (#205, Plan items 2 and 3).

The body is the body anchor and every node below it, started higher when
no joint below the anchor has a head name or a foot name. It has no public
name; it is computed on plain data in ``tests/test_orientation.py``
(``TestBody``). Here it is seen through what a clip finds in it: the head
joint (through rest up and the world-up guess), the L/R pairs of rest left
and rest forward, and the feet :meth:`Bvh.auto_detect_feet` detects.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pybvh import Bvh, nodes_from_table, read_bvh_file

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"


def _read(path):
    """The clip, read without the world-up warnings some corpus clips give."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(path)


def _feet(clip):
    return [foot.name for foot in clip.auto_detect_feet()]


@pytest.mark.parametrize(
    ("name", "feet"),
    [
        ("footsteps_humanoid_6ch.bvh", ["LeftToeBase", "RightToeBase"]),
        ("footsteps_humanoid_offset_6ch.bvh", ["LeftToeBase", "RightToeBase"]),
        ("quadruped_tails_6ch.bvh", ["LeftToe", "RightToe"]),
    ],
)
def test_a_footsteps_node_beside_the_body_is_not_a_foot(name, feet):
    assert _feet(_read(CORPUS / name)) == feet


def test_hips_beside_the_spine_read_up_from_the_head_joint():
    # The hips hold no head joint: the body starts at the root, where the
    # spine hangs beside them.
    clip = _read(CORPUS / "hips_beside_spine.bvh")
    report = clip.orientation_report
    assert clip.body_anchor.name == "Hips"
    assert report.rest_up_source == "named_joint"
    assert (report.world_up, report.world_up_source) == ("+y", "first_frame")
    assert _feet(clip) == ["LeftToeBase", "RightToeBase"]


def test_chest_and_hips_at_the_same_depth_find_the_feet():
    # The chest is the anchor and holds no foot: the body starts at the
    # root, where the hips hang beside it.
    clip = _read(CORPUS / "chest_and_hips_same_depth.bvh")
    assert clip.body_anchor.name == "Chest"
    assert _feet(clip) == ["LeftToeBase", "RightToeBase"]


@pytest.mark.parametrize("name", ["reference_props_humanoid_6ch", "reference_props_rotation_only"])
def test_rest_left_and_rest_forward_read_the_pairs_of_the_body(name):
    # The props hang from the reference root, beside the body: the clip
    # faces as the body alone does.
    report = _read(CORPUS / f"{name}.bvh").orientation_report
    control = _read(CORPUS / "rotation_only_control.bvh").orientation_report
    assert (report.rest_left, report.rest_forward, report.rest_forward_source) == (
        control.rest_left,
        control.rest_forward,
        "lr_geometry",
    )


# The clips whose body is the whole skeleton, with what they gave before
# the searches looked in the body (#205, at the merge of its ticket 05):
# rest up and its source, rest left, rest forward, world up, the feet.
WHOLE_BODY = [
    ("bvh_data/bvh_test1.bvh", "+z", "named_joint", "-x", "+y", "+z", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/bvh_test2.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToe", "RightToe")),
    ("bvh_data/bvh_test3.bvh", "+y", "named_joint", "+x", "+z", "+z", ("LeftForeFoot", "RightForeFoot")),
    ("bvh_data/bvh_test4.bvh", "+z", "named_joint", "-x", "+y", "+z", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/cmu_12_01_walk.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/standard_skeleton.bvh", "+z", "named_joint", "-x", "+y", "+z", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/blender_mixed.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/bvh_synthetic_lying_down.bvh", "+z", "named_joint", "-x", "+y", "+z", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/hips_hands_6ch.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/mixed_zxy_positions.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/rigid_6ch_control.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/rotation_only_control.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("bvh_data/synthetic/synthetic_stretch.bvh", "+y", "named_joint", "+x", "+z", "+y", ("LeftToeBase", "RightToeBase")),
    ("tests/fixtures/full_precision_frame_time.bvh", "+y", "named_joint", None, "+z", "+y", ()),
    ("tests/fixtures/pre_186_root_and_child.bvh", "+y", "named_joint", None, "+z", "+y", ()),
    ("tests/fixtures/rotation_first_root.bvh", "+y", "named_joint", None, "+z", "+y", ()),
]  # fmt: skip


@pytest.mark.parametrize(
    ("path", "rest_up", "source", "rest_left", "rest_forward", "world_up", "feet"),
    WHOLE_BODY,
    ids=[row[0] for row in WHOLE_BODY],
)
def test_a_clip_whose_body_is_its_whole_skeleton_gives_what_it_gave(
    path, rest_up, source, rest_left, rest_forward, world_up, feet
):
    clip = _read(REPO / path)
    report = clip.orientation_report
    assert (
        report.rest_up,
        report.rest_up_source,
        report.rest_left,
        report.rest_forward,
        report.world_up,
    ) == (rest_up, source, rest_left, rest_forward, world_up)
    assert tuple(_feet(clip)) == feet


def test_the_largest_spread_reads_the_body_the_user_names():
    # No head name, no pairs: the body the user names stands along +y, and
    # the prop beside it, under the root, spreads farther along +x.
    rows = [
        {"name": "Scene", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {"name": "Body", "parent": 0, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {"name": "Torso", "parent": 1, "offset": [0, 1, 0], "rot_channels": "ZYX"},
        {"name": "Skull", "parent": 2, "offset": [0, 1, 0]},
        {"name": "Prop", "parent": 0, "offset": [100, 0, 0]},
    ]
    clip = Bvh(nodes_from_table(rows))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clip.body_anchor = "Body"
        report = clip.orientation_report
    assert (report.rest_up, report.rest_up_source) == ("+y", "largest_spread")
    assert (report.world_up, report.world_up_source) == ("+y", "rest_up")
