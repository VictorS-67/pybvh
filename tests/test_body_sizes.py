"""Sizes read the joint the feet branch from (#205).

``skeleton_size``, the threshold scale of the contacts and their "feet
below hips" check read the size reference: the lowest joint that has the
feet in two or more of its branches, else the body anchor. So the same
body gives the same answers, whatever lies above it:
``reference_humanoid_6ch`` holds the body of ``rotation_only_control`` at
the same world positions under a reference root, and gives its sizes and
its contacts. On a clip whose root is where its feet branch, the sizes and
the contacts are those it gave when they read the root.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"

# The control's size: its feet's mean rest distance to its hips, its root.
CONTROL_SIZE = 93.34880824091971
METHODS = ["combined", "velocity", "height"]


def _read(path):
    """The clip, read without the world-up warnings some give."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(path)


@pytest.fixture
def control():
    return _read(CORPUS / "rotation_only_control.bvh")


class TestAReferenceRootGivesTheControlsSizes:
    """A root above the hips: the sizes and the contacts are the control's."""

    def test_skeleton_size(self, control):
        reference = _read(CORPUS / "reference_humanoid_6ch.bvh")
        assert reference.skeleton_size() == pytest.approx(control.skeleton_size(), rel=1e-12)

    @pytest.mark.parametrize("method", METHODS)
    def test_the_contacts(self, control, method):
        reference = _read(CORPUS / "reference_humanoid_6ch.bvh")
        ours, our_report = reference.foot_contacts(method=method, return_info=True)
        theirs, their_report = control.foot_contacts(method=method, return_info=True)
        np.testing.assert_array_equal(ours, theirs)
        assert our_report.skeleton_scale == pytest.approx(their_report.skeleton_scale, rel=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        "reference_props_humanoid_6ch.bvh",
        "reference_props_rotation_only.bvh",
        "footsteps_humanoid_offset_6ch.bvh",
    ],
)
class TestARootBesideTheBody:
    """A root that carries props, or a Bip01 root at an offset from the hips."""

    def test_skeleton_size_is_the_controls(self, name):
        assert _read(CORPUS / name).skeleton_size() == pytest.approx(CONTROL_SIZE, rel=1e-9)

    def test_the_height_method_returns(self, name):
        clip = _read(CORPUS / name)
        contacts = clip.foot_contacts(method="height")
        assert contacts.shape == (clip.frame_count, 2)


def test_chest_and_hips_at_one_depth_measure_from_the_hips():
    # The anchor is the chest; the feet branch at the hips.
    clip = _read(CORPUS / "chest_and_hips_same_depth.bvh")
    assert clip.body_anchor.name == "Chest"
    feet = clip.auto_detect_feet()
    rest = clip.rest_pose_positions()
    hips = rest[clip.index("Hips", space="node")]
    expected = np.mean(
        [np.linalg.norm(rest[clip.index(foot, space="node")] - hips) for foot in feet]
    )
    assert clip.skeleton_size(feet) == pytest.approx(expected, rel=1e-12)


# The clips whose root is where their detected feet branch, with what they
# gave when the sizes read the root (#205, before its ticket 07): the
# skeleton size, then the contact frames of each foot under the combined,
# the velocity and the height methods.
ROOT_BRANCHED = [
    ("bvh_data/bvh_test1.bvh", 35.565995560180795, (86, 92), (86, 92), (98, 105)),
    ("bvh_data/bvh_test2.bvh", 96.88008741281922, (61, 61), (61, 61), (61, 61)),
    ("bvh_data/bvh_test3.bvh", 33.682804865030164, (100, 100), (100, 100), (100, 100)),
    ("bvh_data/bvh_test4.bvh", 35.565995560180795, (70, 68), (70, 68), (74, 71)),
    ("bvh_data/cmu_12_01_walk.bvh", 16.7092517264102, (102, 118), (152, 171), (72, 84)),
    ("bvh_data/standard_skeleton.bvh", 31.893243798648015, (1, 1), (1, 1), (1, 1)),
    ("bvh_data/synthetic/blender_mixed.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
    ("bvh_data/synthetic/bvh_synthetic_lying_down.bvh", 35.565995560180795, (40, 63), (120, 120), (40, 63)),
    ("bvh_data/synthetic/hips_hands_6ch.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
    ("bvh_data/synthetic/mixed_zxy_positions.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
    ("bvh_data/synthetic/rigid_6ch_control.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
    ("bvh_data/synthetic/rotation_only_control.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
    ("bvh_data/synthetic/synthetic_stretch.bvh", 93.34880824091971, (0, 0), (0, 0), (1, 1)),
]  # fmt: skip


@pytest.mark.parametrize(
    ("path", "size", "combined", "velocity", "height"),
    ROOT_BRANCHED,
    ids=[row[0] for row in ROOT_BRANCHED],
)
def test_a_clip_whose_root_is_where_its_feet_branch_gives_what_it_gave(
    path, size, combined, velocity, height
):
    clip = _read(REPO / path)
    assert clip.skeleton_size() == size
    for method, frames in zip(METHODS, (combined, velocity, height), strict=True):
        contacts, report = clip.foot_contacts(method=method, return_info=True)
        assert tuple(int(n) for n in contacts.sum(axis=0)) == frames
        assert report.skeleton_scale == size
