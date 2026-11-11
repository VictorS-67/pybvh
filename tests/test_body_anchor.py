"""The body anchor of a clip: ``Bvh.body_anchor`` and the report's fields (#205).

The rule itself is tested on plain data in ``tests/test_orientation.py``;
here it is read through a clip: on every clip of the synthetic corpus,
against the anchor and source its manifest expects, on the fixture clips
whose names give no L/R pair, and through its lifecycle (ADR 0003): found
once, found again after a change of the skeleton or of the L/R mapping, and
never inherited by a derived clip.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text())
EXPECTED = {
    entry["file"]: (entry["expected_body_anchor"], entry["expected_body_anchor_source"])
    for entry in MANIFEST["files"]
}
FIXTURES = REPO / "tests" / "fixtures"
# The fixture clips whose joint names give no L/R pair.
NO_PAIRS = [
    "full_precision_frame_time.bvh",
    "pre_186_root_and_child.bvh",
    "rotation_first_root.bvh",
]


@pytest.fixture
def bvh_example():
    return read_bvh_file(REPO / "bvh_data" / "bvh_test1.bvh")


def _read(path):
    """The clip, read without the world-up warnings some corpus clips give."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(path)


def test_the_manifest_lists_the_twelve_generated_clips():
    assert len(EXPECTED) == 12


@pytest.mark.parametrize("name", sorted(EXPECTED), ids=sorted(EXPECTED))
def test_a_corpus_clip_gives_the_anchor_its_manifest_expects(name):
    clip = _read(CORPUS / name)
    report = clip.orientation_report
    assert (clip.body_anchor.name, report.body_anchor_source) == EXPECTED[name]
    assert report.body_anchor_candidates == (clip.body_anchor,)


@pytest.mark.parametrize("name", NO_PAIRS)
def test_a_clip_without_pairs_gives_the_root(name):
    clip = _read(FIXTURES / name)
    assert clip.lr_mapping is None
    report = clip.orientation_report
    assert clip.body_anchor is clip.root
    assert (report.body_anchor_source, report.body_anchor_candidates) == ("default", ())


def test_the_anchor_is_a_node_of_the_clip_and_the_report_s(bvh_example):
    # bvh_example's root is its hips.
    anchor = bvh_example.body_anchor
    assert any(anchor is node for node in bvh_example.nodes)
    assert anchor is bvh_example.orientation_report.body_anchor
    assert anchor is bvh_example.root


def test_a_root_above_the_hips_gives_the_hips():
    clip = _read(CORPUS / "reference_humanoid_6ch.bvh")
    assert clip.root.name == "Reference"
    assert clip.body_anchor is clip.nodes[clip.node_index["Hips"]]


def test_the_mapping_in_effect_is_read():
    # The arms alone meet at the chest; the mapping found from the names
    # holds the legs too, which meet at the hips.
    clip = _read(CORPUS / "rotation_only_control.bvh")
    assert clip.body_anchor.name == "Hips"
    clip.lr_mapping = {"LeftArm": "RightArm"}
    assert clip.body_anchor.name == "Chest"
    assert clip.orientation_report.body_anchor.name == "Chest"
    clip.lr_mapping = None
    assert clip.body_anchor is clip.root
    assert clip.orientation_report.body_anchor_source == "default"


@pytest.mark.parametrize(
    "name",
    [
        "reference_props_humanoid_6ch.bvh",
        "reference_props_rotation_only.bvh",
        "footsteps_humanoid_offset_6ch.bvh",
    ],
)
def test_up_is_read_from_the_anchor_on_a_root_beside_the_body(name):
    # The body stands along +y, as authored; the root carries props or
    # lies at an offset, and read from it the head is along +z.
    clip = _read(CORPUS / name)
    report = clip.orientation_report
    assert (clip.world_up, report.world_up_source) == ("+y", "first_frame")
    assert clip.rest_up == "+y"


class TestLifecycle:
    """Derived state (ADR 0003): found from the skeleton and the L/R mapping
    in effect, kept until one of them changes, never inherited."""

    def _read_anchor(self, bvh):
        _ = (bvh.body_anchor, bvh.orientation_report.body_anchor)

    def test_found_once_across_its_readers(self, computations):
        bvh = read_bvh_file(REPO / "bvh_data" / "bvh_test1.bvh")
        self._read_anchor(bvh)
        self._read_anchor(bvh)
        assert computations.body_anchor == 1

    @pytest.mark.parametrize(
        "change",
        [
            lambda b: b.reorient_rest_up("+y", inplace=True),
            lambda b: setattr(b, "lr_mapping", None),
            lambda b: setattr(b, "lr_mapping", {"LeftUpLeg": "RightUpLeg"}),
        ],
        ids=["skeleton", "mapping cleared", "mapping set"],
    )
    def test_found_again_after_each_clearing_change(self, bvh_example, computations, change):
        self._read_anchor(bvh_example)
        found = computations.body_anchor
        change(bvh_example)
        self._read_anchor(bvh_example)
        assert computations.body_anchor == found + 1

    @pytest.mark.parametrize(
        "change",
        [
            lambda b: setattr(b, "joint_angles", b.joint_angles),
            lambda b: setattr(b, "root_pos", b.root_pos),
            lambda b: b.translate_root(np.array([1.0, 0.0, 0.0]), inplace=True),
            lambda b: setattr(b, "world_up", "+x"),
        ],
        ids=["joint_angles assigned", "root_pos assigned", "translate_root", "world up set"],
    )
    def test_kept_by_the_other_changes(self, bvh_example, computations, change):
        self._read_anchor(bvh_example)
        found = computations.body_anchor
        change(bvh_example)
        self._read_anchor(bvh_example)
        assert computations.body_anchor == found

    @pytest.mark.parametrize(
        "derive",
        [lambda b: b[0:10], lambda b: b.copy(), lambda b: b.scale(2.0)],
        ids=["slice", "copy", "scale"],
    )
    def test_found_again_on_a_derived_clip(self, bvh_example, computations, derive):
        self._read_anchor(bvh_example)
        found = computations.body_anchor
        derived = derive(bvh_example)
        self._read_anchor(derived)
        assert computations.body_anchor == found + 1
        assert derived.body_anchor is derived.root
