"""A clip with no body (#205, ticket 09).

A clip whose hierarchy holds two figures of comparable size has no body:
``bvh.body_anchor`` is ``None``. Every body-level operation refuses it with
one error, which names the candidates and the way out, ``bvh.body_anchor =
...``; every hierarchy-level operation works as on any clip. Once a
candidate is named, the clip describes that figure, as a clip of its own
does. The two corpus profiles hold the same figure twice, under a root that
moves and under a static root.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import Bvh, bvhplot, nodes_from_table, read_bvh_file
from pybvh._orientation import _up_marker_index
from pybvh.analysis import (
    cadence,
    foot_contacts,
    gait_parameters,
    ground_contacts,
    root_trajectory,
    skeleton_size,
    stride_length,
    walking_pace,
)
from pybvh.bvhnode import BvhNode
from pybvh.features import to_feature_array
from pybvh.spatial_coord import frames_to_node_positions
from pybvh.transforms import mirror

CORPUS = Path(__file__).resolve().parent.parent / "bvh_data" / "synthetic"
TWO_FIGURES = ["two_figures_moving_root.bvh", "two_figures_static_root.bvh"]
TOES = ["LeftToeBase", "RightToeBase"]
# The L/R pairs of the first figure alone: with them, the rule would find
# a body, where the clip's own pairs find none.
FIRST_FIGURE_PAIRS = {
    f"Left{part}": f"Right{part}"
    for part in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase")
}


def _read(name, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(CORPUS / name, **kwargs)


# The body-level entry points, each a call on a clip. Those that read world
# up alone are only body-level while world up is guessed.
READ_WORLD_UP = {
    "world_up": lambda b: b.world_up,
    "world_up_inferred": lambda b: b.world_up_inferred,
    "world_up_axis": lambda b: b.world_up_axis,
}
BODY_LEVEL = {
    "rest_up": lambda b: b.rest_up,
    "rest_up_axis": lambda b: b.rest_up_axis,
    "rest_forward": lambda b: b.rest_forward,
    "rest_forward_axis": lambda b: b.rest_forward_axis,
    "has_lr_geometry (rest left)": lambda b: b.has_lr_geometry,
    "forward_at": lambda b: b.forward_at(0),
    "left_at": lambda b: b.left_at(0),
    "facing_frame": lambda b: b.facing_frame(),
    "root_trajectory": lambda b: root_trajectory(b),
    "node_positions(centered='anchor')": lambda b: b.node_positions(centered="anchor"),
    "node_positions(centered='first_ground')": lambda b: b.node_positions(centered="first_ground"),
    "node_positions(frame=0, centered='anchor')": lambda b: b.node_positions(
        frame=0, centered="anchor"
    ),
    "node_positions(centered='anchor_ground')": lambda b: b.node_positions(
        centered="anchor_ground"
    ),
    "node_positions(frame=0, centered='anchor_ground')": lambda b: b.node_positions(
        frame=0, centered="anchor_ground"
    ),
    "joint_positions(centered='anchor')": lambda b: b.joint_positions(centered="anchor"),
    "node_velocities(centered='anchor_ground')": lambda b: b.node_velocities(
        centered="anchor_ground"
    ),
    "frames_to_node_positions(centered='anchor_ground')": lambda b: frames_to_node_positions(
        b, centered="anchor_ground"
    ),
    "to_feature_array(centered='anchor_ground')": lambda b: to_feature_array(
        b, centered="anchor_ground"
    ),
    "node_velocities(centered='anchor')": lambda b: b.node_velocities(centered="anchor"),
    "frames_to_node_positions(centered='anchor')": lambda b: frames_to_node_positions(
        b, b.root_pos, b.joint_angles, centered="anchor", joint_translations=b.joint_translations
    ),
    "to_feature_array(centered='first_ground')": lambda b: to_feature_array(
        b, centered="first_ground"
    ),
    "auto_detect_feet": lambda b: b.auto_detect_feet(),
    "foot_contacts, the feet given": lambda b: foot_contacts(b, TOES),
    "foot_contacts, the feet detected": lambda b: foot_contacts(b),
    "ground_contacts": lambda b: ground_contacts(b, TOES),
    "skeleton_size, the feet given": lambda b: skeleton_size(b, TOES),
    "walking_pace": lambda b: walking_pace(b),
    "walking_pace, one frame": lambda b: walking_pace(b[:1]),
    "cadence": lambda b: cadence(b, TOES),
    "stride_length": lambda b: stride_length(b, TOES),
    "gait_parameters": lambda b: gait_parameters(b, TOES),
    "gait_parameters, one frame, the contacts given": lambda b: gait_parameters(
        b[:1], TOES, contacts=np.zeros((1, 2))
    ),
    "mirror": lambda b: mirror(b),
    "mirror(lateral='x')": lambda b: mirror(b, lateral="x"),
    "mirror, the pairs of one figure given": lambda b: mirror(b, lr_mapping=FIRST_FIGURE_PAIRS),
}


def _ids(entries):
    return list(entries)


def _named_alike(clip):
    """*clip* with its second figure carrying the first figure's names."""
    table = clip.to_node_table()
    for row in table:
        row["name"] = row["name"].removesuffix("_2")
    return clip.copy(nodes=table)


class TestBodyLevelRaises:
    """Every body-level entry point refuses a clip with no body."""

    @pytest.mark.parametrize("name", TWO_FIGURES)
    @pytest.mark.parametrize(
        "call",
        list({**READ_WORLD_UP, **BODY_LEVEL}.values()),
        ids=_ids({**READ_WORLD_UP, **BODY_LEVEL}),
    )
    def test_with_world_up_guessed(self, name, call):
        clip = _read(name)
        with pytest.raises(ValueError, match="no body") as raised:
            call(clip)
        assert "'Hips'" in str(raised.value)
        assert "'Hips_2'" in str(raised.value)
        assert "bvh.body_anchor = " in str(raised.value)

    @pytest.mark.parametrize("name", TWO_FIGURES)
    @pytest.mark.parametrize("call", list(BODY_LEVEL.values()), ids=_ids(BODY_LEVEL))
    def test_with_world_up_set(self, name, call):
        clip = _read(name, world_up="+y")
        with pytest.raises(ValueError, match="no body") as raised:
            call(clip)
        assert "'Hips'" in str(raised.value)
        assert "'Hips_2'" in str(raised.value)
        assert "bvh.body_anchor = " in str(raised.value)


HIERARCHY_LEVEL = {
    "node_positions": lambda b: b.node_positions(),
    "node_positions(frame=0)": lambda b: b.node_positions(frame=0),
    "joint_positions": lambda b: b.joint_positions(),
    "node_velocities": lambda b: b.node_velocities(),
    "node_jerk": lambda b: b.node_jerk(),
    "a slice": lambda b: b[1:3],
    "a concatenation": lambda b: b + b,
    "resample": lambda b: b.resample(2 * b.fps),
    "change_euler_order": lambda b: b.change_euler_order("XYZ", joint="Hips"),
    "scale": lambda b: b.scale(2.0),
    "extract_joints": lambda b: b.extract_joints([b.root, "Hips", "Spine", "Hips_2"]),
    "copy": lambda b: b.copy(),
    "to_rotmat": lambda b: b.to_rotmat(),
    "to_feature_array": lambda b: to_feature_array(b),
    "bounding_box": lambda b: b.bounding_box(),
    "lr_mapping": lambda b: b.lr_mapping,
    "body_anchor": lambda b: b.body_anchor,
    "orientation_report": lambda b: b.orientation_report,
}


class TestTheErrorsWayOut:
    """The way out the no-body error names works on the clip."""

    def test_unique_names_are_named(self):
        with pytest.raises(ValueError) as raised:
            _ = _read("two_figures_moving_root.bvh").rest_up
        assert str(raised.value) == (
            "This clip has no body: its hierarchy holds several figures of "
            "comparable size, whose body anchors would be 'Hips' and 'Hips_2'. "
            "Name the one to describe, for example `bvh.body_anchor = 'Hips'`, "
            "or pass `body_anchor=` to read_bvh_file."
        )

    @staticmethod
    def _shared_names():
        """``two_figures_static_root`` with the second figure carrying the
        first's names."""
        return _named_alike(_read("two_figures_static_root.bvh"))

    @staticmethod
    def _suggested_node(clip, message):
        """The node the error's suggested assignment names, after running it."""
        match = re.search(r"`(bvh\.body_anchor = bvh\.nodes\[\d+\])`", message)
        assert match is not None, message
        exec(match.group(1), {"bvh": clip})
        return clip.body_anchor

    def test_shared_names_are_named_by_the_node(self):
        # The second figure carries the first's names, so a name selects
        # no candidate: the error names the candidate's node.
        clip = self._shared_names()
        with pytest.raises(ValueError, match="no body") as raised:
            _ = clip.rest_up
        message = str(raised.value)
        assert "'Hips' and 'Hips'" in message
        assert "read_bvh_file" not in message
        first = clip.orientation_report.body_anchor_candidates[0]
        assert self._suggested_node(clip, message) is first
        assert clip.rest_up == "+y"

    def test_shared_names_under_another_mapping_name_one_of_its_candidates(self):
        # The clip's own report holds no candidate (no mapping, so the root
        # by default); mirror reads the anchor of the mapping it is given,
        # whose candidates are the two hips.
        clip = self._shared_names()
        clip.lr_mapping = None
        assert clip.orientation_report.body_anchor_candidates == ()
        with pytest.raises(ValueError, match="no body") as raised:
            mirror(clip, lr_mapping={"LeftUpLeg": "RightUpLeg"})
        hips = [node for node in clip.nodes if node.name == "Hips"]
        assert self._suggested_node(clip, str(raised.value)) in hips


class TestHierarchyLevelReturns:
    """Every hierarchy-level operation works on a clip with no body."""

    @pytest.mark.parametrize("name", TWO_FIGURES)
    @pytest.mark.parametrize("call", list(HIERARCHY_LEVEL.values()), ids=_ids(HIERARCHY_LEVEL))
    def test_returns(self, name, call):
        call(_read(name))

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_a_write_then_read_gives_an_equal_clip(self, name, tmp_path):
        clip = _read(name)
        clip.write(tmp_path / name)
        assert _read(tmp_path / name) == clip

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_the_report_holds_no_body_fact(self, name):
        report = _read(name).orientation_report
        assert report.body_anchor is None
        assert report.body_anchor_source == "ambiguous"
        assert [node.name for node in report.body_anchor_candidates] == ["Hips", "Hips_2"]
        body_facts = (
            report.rest_up,
            report.rest_up_source,
            report.rest_left,
            report.rest_forward,
            report.rest_forward_source,
            report.world_up,
            report.world_up_source,
        )
        assert body_facts == (None,) * 7

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_the_scene_ground_reads_world_up_once_set(self, name):
        clip = _read(name)
        with pytest.raises(ValueError, match="no body"):
            _ = clip.floor_height
        clip.world_up = "+y"
        assert clip.floor_height < clip.node_positions()[..., 1].mean()


class TestDrawing:
    """Drawing is hierarchy-level, but it reads world up, whose guess is body-level."""

    def _draw(self, clip, tmp_path):
        fig, _ax = bvhplot.frame(clip, 0, style=bvhplot.Style("paper", color_mode="chains"))
        plt.close(fig)
        return bvhplot.render(clip[0:4], tmp_path / "clip.gif", backend="matplotlib")

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_raises_while_world_up_is_guessed(self, name, tmp_path):
        with pytest.raises(ValueError, match="no body"):
            self._draw(_read(name), tmp_path)

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_draws_once_world_up_is_set(self, name, tmp_path):
        clip = _read(name)
        clip.world_up = "+y"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._draw(clip, tmp_path).exists()

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_draws_once_the_anchor_is_named(self, name, tmp_path):
        clip = _read(name)
        clip.body_anchor = "Hips"
        assert self._draw(clip, tmp_path).exists()


def _figure_alone(clip, suffix):
    """The clip cut to its root and one figure: the joints whose names end
    with *suffix* (``""`` for the first figure)."""
    joints = [
        node
        for node in clip.nodes[1:]
        if not node.is_end_site() and node.name.endswith("_2") == (suffix == "_2")
    ]
    return clip.extract_joints([clip.root, *joints])


def _body_facts(clip):
    """What the clip says of its body, with joints by name, the suffix of the
    second figure removed so the two figures compare."""

    def name(node_or_index):
        node = clip.nodes[node_or_index] if isinstance(node_or_index, int) else node_or_index
        return node.name.removesuffix("_2")

    feet = clip.auto_detect_feet()
    suffix = "_2" if clip.body_anchor.name.endswith("_2") else ""
    torso = [clip.node_index[f"{joint}{suffix}"] for joint in ("Hips", "Spine", "Head")]
    head = _up_marker_index(clip._rest_skeleton(), clip._body_anchor_index())
    return {
        "anchor": name(clip.body_anchor),
        "head joint": None if head is None else name(int(head)),
        "feet": [name(foot) for foot in feet],
        "world_up": clip.world_up,
        "rest_up": clip.rest_up,
        "rest_forward": clip.rest_forward,
        "skeleton_size": round(skeleton_size(clip, feet), 9),
        "contacts": foot_contacts(clip, feet, method="height").tolist(),
        "walking_pace": round(walking_pace(clip), 9),
        "heading": np.round(root_trajectory(clip)[:, 2:4], 9).tolist(),
        "centred": np.round(clip.node_positions(centered="anchor")[:, torso], 9).tolist(),
        "facing": np.round(np.stack(clip.facing_frame()[:3]), 9).tolist(),
        "forward_at": clip.forward_at(0),
        "left_at": clip.left_at(0),
    }


def _second_figure_turned(clip):
    """*clip* with its second figure turned a quarter turn about its hips'
    Y axis, so the two figures face two ways."""
    hips = clip.nodes[clip.node_index["Hips_2"]]
    angles = clip.joint_angles.copy()
    angles[:, clip.joint_names.index("Hips_2"), hips.rot_channels.index("Y")] += np.pi / 2
    clip.joint_angles = angles
    return clip


class TestNamingOneFigure:
    """Once a candidate is named, the clip describes that figure."""

    @pytest.mark.parametrize("name", TWO_FIGURES)
    @pytest.mark.parametrize("suffix", ["", "_2"])
    def test_gives_the_figure_alone(self, name, suffix):
        clip = _read(name)
        alone = _figure_alone(clip, suffix)
        clip.body_anchor = f"Hips{suffix}"
        assert alone.body_anchor.name == f"Hips{suffix}"
        assert _body_facts(clip) == _body_facts(alone)

    @pytest.mark.parametrize("name", TWO_FIGURES)
    @pytest.mark.parametrize("suffix", ["", "_2"])
    def test_faces_as_the_figure_alone(self, name, suffix):
        clip = _second_figure_turned(_read(name))
        alone = _figure_alone(clip, suffix)
        clip.body_anchor = f"Hips{suffix}"
        assert _body_facts(clip) == _body_facts(alone)

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_naming_the_other_gives_the_other(self, name):
        clip = _read(name)
        clip.body_anchor = "Hips"
        _body_facts(clip)
        clip.body_anchor = "Hips_2"
        assert _body_facts(clip) == _body_facts(_figure_alone(clip, "_2"))

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_naming_it_at_load_gives_the_same(self, name):
        then_named = _read(name)
        then_named.body_anchor = "Hips_2"
        at_load = _read(name, body_anchor="Hips_2")
        assert at_load == then_named
        assert _body_facts(at_load) == _body_facts(then_named)


class TestFiguresNamedAlike:
    """Two figures that share their joint names are two figures (#216): a
    name pair binds one node pair in each figure, by the tree."""

    @pytest.mark.parametrize("name", TWO_FIGURES)
    def test_have_no_body(self, name):
        clip = _named_alike(_read(name))
        report = clip.orientation_report
        assert clip.body_anchor is None
        assert report.body_anchor_source == "ambiguous"
        hips = [node for node in clip.nodes if node.name == "Hips"]
        assert list(report.body_anchor_candidates) == hips


class TestTheAnchorsEvent:
    """Setting the anchor clears what was computed from it (ADR 0003, 0004)."""

    PATH = CORPUS / "reference_humanoid_6ch.bvh"

    def test_a_guess_of_the_clip_s_own_is_made_again(self, computations):
        clip = _read(self.PATH)
        assert clip.orientation_report.world_up_source == "first_frame"
        guesses = computations.world_up
        clip.body_anchor = "Chest"
        assert clip.world_up == "+y"
        assert computations.world_up == guesses + 1

    def test_an_override_stays(self, computations):
        clip = _read(self.PATH, world_up="-z")
        guesses = computations.world_up
        clip.body_anchor = "Chest"
        assert clip.world_up == "-z"
        assert computations.world_up == guesses

    def test_an_inherited_guess_stays(self, computations):
        clip = _read(self.PATH)[1:5]
        assert clip.orientation_report.world_up_inherited
        guesses = computations.world_up
        clip.body_anchor = "Chest"
        assert clip.world_up == "+y"
        assert clip.orientation_report.world_up_inherited
        assert computations.world_up == guesses


def _rooted_at(clip, anchor_name):
    """The figure below *anchor_name* as a clip of its own whose root is that
    joint, and the node and joint positions it keeps from *clip*.

    *clip* must have a static root at the origin, so the joint's file
    values are its world position.
    """
    rows = clip.to_node_table()
    anchor = clip.node_index[anchor_name]
    below = {anchor}
    for i, row in enumerate(rows):
        if row["parent"] in below:
            below.add(i)
    keep = sorted(below)
    position = {old: new for new, old in enumerate(keep)}
    table = [
        {**rows[old], "parent": None if old == anchor else position[rows[old]["parent"]]}
        for old in keep
    ]
    topology = clip.fk_topology
    joints = [int(topology.joint_idx[i]) for i in keep if topology.joint_idx[i] >= 0]
    rooted = Bvh(
        nodes=nodes_from_table(table),
        root_pos=clip.joint_translations[:, int(topology.translation_idx[anchor])],
        joint_angles=clip.joint_angles[:, joints],
        frame_time=clip.frame_time,
    )
    np.testing.assert_allclose(rooted.node_positions(), clip.node_positions()[:, keep], atol=1e-9)
    return rooted, keep, joints


def _comparable(value, clip=None, keep=None, joints=None):
    """*value* as plain data, rounded; with *clip*, an array or clip of
    *clip*'s is cut to the rows of the nodes *keep* or the joints *joints*."""
    if isinstance(value, Bvh):
        value = value.node_positions()
    if isinstance(value, np.ndarray):
        if clip is not None and value.ndim == 2 and value.shape[0] == len(clip.nodes):
            value = value[keep]  # one frame
        elif clip is not None and value.ndim >= 2 and value.shape[1] == len(clip.nodes):
            value = value[:, keep]
        elif clip is not None and value.ndim >= 2 and value.shape[1] == len(clip.joint_angles[0]):
            value = value[:, joints]
        return _comparable(np.round(value.astype(float), 6).tolist())
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: _comparable(v, clip, keep, joints) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(v, clip, keep, joints) for v in value]
    if isinstance(value, BvhNode):
        return value.name
    if isinstance(value, float):
        return "nan" if np.isnan(value) else round(value, 6)
    return value


# Every body-level entry point but to_feature_array, whose root block is
# the root's row (an open question of #205), and the world-up readers,
# which the other entries read.
GUARDED = {
    name: call for name, call in BODY_LEVEL.items() if not name.startswith("to_feature_array")
}


@pytest.mark.parametrize("call", list(GUARDED.values()), ids=_ids(GUARDED))
def test_no_body_level_entry_point_reads_node_0_as_the_body(call):
    # The clip's root, node 0, is a static root at the origin, away from
    # the figure named as the body; the figure rooted at its own hips is
    # the reference. An entry point that read node 0 as the body would
    # read a point that never moves and never turns.
    clip = _read("two_figures_static_root.bvh", body_anchor="Hips")
    rooted, keep, joints = _rooted_at(clip, "Hips")
    assert _comparable(call(clip), clip, keep, joints) == _comparable(call(rooted))


# The members of Bvh that BODY_LEVEL and READ_WORLD_UP reach, by name.
BODY_LEVEL_MEMBERS = {
    "auto_detect_feet",
    "cadence",
    "facing_frame",
    "foot_contacts",
    "forward_at",
    "gait_parameters",
    "has_lr_geometry",
    "left_at",
    "mirror",
    "rest_forward",
    "rest_forward_axis",
    "rest_up",
    "rest_up_axis",
    "root_trajectory",
    "skeleton_size",
    "stride_length",
    "walking_pace",
    "world_up_inferred",
}


@pytest.mark.parametrize("name", TWO_FIGURES)
def test_every_member_that_refuses_a_clip_with_no_body_is_listed(name):
    # Finds the body-level entry points among the members of Bvh that take
    # no argument (the drawing calls aside), by calling each on a clip with
    # no body and world up set: a member that reaches the anchor raises the
    # error. The list above is the one the tests of this file check, so a
    # new body-level member is tested once it is listed.
    refused = set()
    for member, value in inspect.getmembers(Bvh):
        if member.startswith("_") or member in ("play", "render") or member.startswith("plot_"):
            continue
        is_method = inspect.isfunction(value) and not isinstance(
            inspect.getattr_static(Bvh, member), (staticmethod, classmethod)
        )
        if not (is_method or isinstance(value, property)):
            continue
        if is_method and any(
            p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in list(inspect.signature(value).parameters.values())[1:]
        ):
            continue
        clip = _read(name, world_up="+y")
        try:
            attribute = getattr(clip, member)
            if is_method:
                attribute()
        except ValueError as error:
            if "has no body" in str(error):
                refused.add(member)
    assert refused == BODY_LEVEL_MEMBERS
