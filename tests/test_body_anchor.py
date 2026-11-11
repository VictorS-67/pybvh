"""The body anchor of a clip: ``Bvh.body_anchor`` and the report's fields (#205).

The rule itself is tested on plain data in ``tests/test_orientation.py``;
here it is read through a clip: on every clip of the synthetic corpus,
against the anchor and source its manifest expects, on the fixture clips
whose names give no L/R pair, and through its lifecycle (ADR 0003): found
once, found again after a change of the skeleton or of the L/R mapping, and
never inherited by a derived clip. The anchor the user sets, on the clip or
at load, is tested here too: the resolver's refusals, and its lifecycle as
one of the user's settings, which every derived clip carries.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybvh import Bvh, nodes_from_table, read_bvh_directory, read_bvh_file

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


def test_the_manifest_lists_the_fourteen_generated_clips():
    assert len(EXPECTED) == 14


@pytest.mark.parametrize("name", sorted(EXPECTED), ids=sorted(EXPECTED))
def test_a_corpus_clip_gives_the_anchor_its_manifest_expects(name):
    clip = _read(CORPUS / name)
    report = clip.orientation_report
    assert (clip.body_anchor.name, report.body_anchor_source) == EXPECTED[name]
    assert report.body_anchor_candidates[0] is clip.body_anchor


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


class TestSetting:
    """``bvh.body_anchor = ...``: a joint the user names, by name or node,
    through the resolver of every selection (ADR 0009)."""

    @pytest.mark.parametrize("by", ["name", "node"])
    def test_a_set_joint_is_the_anchor_with_source_set(self, by):
        clip = _read(CORPUS / "rotation_only_control.bvh")
        chest = clip.nodes[clip.node_index["Chest"]]
        clip.body_anchor = "Chest" if by == "name" else chest
        report = clip.orientation_report
        assert clip.body_anchor is chest
        assert report.body_anchor is chest
        assert (report.body_anchor_source, report.body_anchor_candidates) == ("set", ())

    @staticmethod
    def _with_a_repeated_name():
        clip = _read(CORPUS / "rotation_only_control.bvh")
        table = clip.to_node_table()
        table[clip.node_index["Spine1"]]["name"] = "Spine"
        return clip.copy(nodes=table)

    @pytest.mark.parametrize(
        ("selector", "error", "match"),
        [
            (lambda c: 3, TypeError, "bvh.nodes"),
            (lambda c: True, TypeError, "bvh.nodes"),
            (lambda c: next(n for n in c.nodes if n.is_end_site()), ValueError, "end site"),
            (lambda c: next(n for n in c.nodes if n.is_end_site()).name, ValueError, "end site"),
            (lambda c: "Pelvis", ValueError, "Pelvis"),
            (lambda c: "Spine", ValueError, "Spine"),
            (
                lambda c: _read(CORPUS / "rotation_only_control.bvh").nodes[1],
                ValueError,
                "not one of this clip's nodes",
            ),
        ],
        ids=[
            "integer",
            "bool",
            "end site",
            "end site name",
            "missing name",
            "repeated name",
            "node of another clip",
        ],
    )
    def test_the_resolver_refuses(self, selector, error, match):
        clip = self._with_a_repeated_name()
        before = clip.orientation_report
        with pytest.raises(error, match=match):
            clip.body_anchor = selector(clip)
        assert clip.orientation_report is before
        assert clip.orientation_report.body_anchor_source == "lr_pairs"

    def test_none_returns_to_the_rule(self):
        clip = _read(CORPUS / "reference_humanoid_6ch.bvh")
        clip.body_anchor = "Chest"
        clip.body_anchor = None
        report = clip.orientation_report
        assert clip.body_anchor.name == "Hips"
        assert (report.body_anchor_source, report.body_anchor_candidates) == (
            "lr_pairs",
            (clip.body_anchor,),
        )


class TestAtLoad:
    """``read_bvh_file(..., body_anchor=...)``: the setting, at load."""

    def _load(self, reader, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return reader(CORPUS / "reference_humanoid_6ch.bvh", **kwargs)

    @pytest.mark.parametrize(
        "reader",
        [
            read_bvh_file,
            Bvh.from_file,
            lambda path, **kw: read_bvh_directory(path.parent, pattern=path.name, **kw)[0],
        ],
        ids=["read_bvh_file", "Bvh.from_file", "read_bvh_directory"],
    )
    def test_equals_reading_then_setting(self, reader):
        loaded = self._load(reader, body_anchor="Chest")
        then_set = _read(CORPUS / "reference_humanoid_6ch.bvh")
        then_set.body_anchor = "Chest"
        assert loaded == then_set
        assert loaded.body_anchor.name == "Chest"
        assert loaded.orientation_report.body_anchor_source == "set"
        assert loaded.world_up == then_set.world_up

    def test_a_missing_name_raises_the_resolver_s_error(self):
        clip = _read(CORPUS / "reference_humanoid_6ch.bvh")
        with pytest.raises(ValueError) as expected:
            clip.body_anchor = "Pelvis"
        with pytest.raises(ValueError) as raised:
            self._load(read_bvh_file, body_anchor="Pelvis")
        assert str(raised.value) == str(expected.value)


class TestSetLifecycle:
    """A set anchor is one of the user's settings (ADR 0003): kept by the
    changes of the clip, carried by every derived clip and a pickle, and
    dropped by a derived clip that no longer holds the joint."""

    @staticmethod
    def _set_on_chest():
        clip = _read(CORPUS / "rotation_only_control.bvh")
        clip.body_anchor = "Chest"
        return clip

    @staticmethod
    def _assert_set(clip, name):
        report = clip.orientation_report
        assert clip.body_anchor.name == name
        assert clip.body_anchor is report.body_anchor
        assert (report.body_anchor_source, report.body_anchor_candidates) == ("set", ())

    @pytest.mark.parametrize(
        "change",
        [
            lambda b: setattr(b, "joint_angles", b.joint_angles),
            lambda b: setattr(b, "root_pos", b.root_pos),
            lambda b: b.reorient_rest_up("+z", inplace=True),
            lambda b: setattr(b, "lr_mapping", {"LeftUpLeg": "RightUpLeg"}),
            lambda b: setattr(b, "lr_mapping", None),
            lambda b: setattr(b, "world_up", "+x"),
        ],
        ids=["joint_angles", "root_pos", "skeleton", "mapping set", "mapping cleared", "world up"],
    )
    def test_kept_by_a_change_of_the_clip(self, change):
        clip = self._set_on_chest()
        change(clip)
        self._assert_set(clip, "Chest")

    def test_follows_its_joint_renamed_through_copy_nodes(self):
        clip = self._set_on_chest()
        table = clip.to_node_table()
        table[clip.node_index["Chest"]]["name"] = "UpperChest"
        self._assert_set(clip.copy(nodes=table), "UpperChest")

    @pytest.mark.parametrize(
        "derive",
        [
            lambda b: b[0:10],
            lambda b: b.copy(),
            lambda b: b.resample(60),
            lambda b: b + b,
            lambda b: b.scale(2.0),
            lambda b: pickle.loads(pickle.dumps(b)),
        ],
        ids=["slice", "copy", "resample", "concatenation", "scale", "pickle"],
    )
    def test_carried_by_a_derived_clip(self, derive):
        self._assert_set(derive(self._set_on_chest()), "Chest")

    def test_carried_by_extract_joints_when_the_joint_is_kept(self):
        clip = self._set_on_chest()
        kept = clip.extract_joints(["Hips", "Spine", "Spine1", "Chest", "Neck", "Head"])
        self._assert_set(kept, "Chest")

    def test_dropped_by_extract_joints_when_the_joint_is_not_kept(self):
        clip = self._set_on_chest()
        kept = clip.extract_joints(["Hips", "Spine", "LeftUpLeg", "RightUpLeg"])
        report = kept.orientation_report
        assert (kept.body_anchor.name, report.body_anchor_source) == ("Hips", "lr_pairs")

    @pytest.mark.parametrize(("kept", "anchor"), [("Chest", "Chest"), ("Spine", "Hips")])
    def test_copy_nodes_of_another_joint_count_goes_by_name(self, kept, anchor):
        # A clip without motion binds no joint, so copy(nodes=) may change
        # the joint count, and the anchor, like a user L/R mapping, is
        # found by name; the rule is used when no joint carries it.
        rows = [
            {"name": "Hips", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
            {"name": "Chest", "parent": 0, "offset": [0, 1, 0], "rot_channels": "ZYX"},
            {"name": "Head", "parent": 1, "offset": [0, 1, 0], "rot_channels": "ZYX"},
            {"name": "HeadEnd", "parent": 2, "offset": [0, 1, 0]},
        ]
        clip = Bvh(nodes_from_table(rows))
        clip.body_anchor = "Chest"
        edited = clip.copy(nodes=[rows[0], {**rows[1], "name": kept}, {**rows[3], "parent": 1}])
        assert edited.body_anchor.name == anchor
        expected = "set" if kept == anchor else "default"
        assert edited.orientation_report.body_anchor_source == expected
