"""Every entry point that selects nodes takes a name or a ``BvhNode`` (#127).

A name must match exactly one node of the index space the entry point works
in, and a node is resolved by identity among the clip's ``nodes``. One table,
``ENTRIES``, lists every selecting parameter of ``Bvh``, ``analysis`` and
``features``, the modules whose functions select nodes of a clip; each test
below runs over the rows it applies to.

To add an entry point, add one row per selecting parameter: ``call`` takes the
clip and the value of that parameter and calls the entry point with valid
values for every other parameter. ``space`` is the index space the entry point
resolves the parameter in, and ``takes_set`` says whether the parameter is a
sequence of selectors: the tests then pass each selector inside a list, and
add the refusals of a bare ``str`` or ``BvhNode`` and of a node selected twice.

The guard at the end of this file reads the public signatures of ``Bvh``,
``analysis`` and ``features``. It finds a selecting parameter by its name
(``node``, ``joints``, ``feet``, ...) or by ``NodeSelector`` in its annotation,
and fails on a selecting parameter without its row in ``ENTRIES`` or not typed
with ``NodeSelector``, on a row whose parameter is gone, on a parameter typed
with ``int`` that ``_INTEGERS_NOT_NODES`` does not list, and on a parameter
named ``coords``. It calls each row with an end site: a parameter named
``joint`` or ``joints`` must refuse it, and every other must accept it,
``Bvh.index`` excepted, whose caller gives the space.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest

from pybvh import Bvh, analysis, features, nodes_from_table

# The rig, in node order. Its names are chosen to collide:
#   0 Hips    root
#   1 Spine   joint
#   2 Head    joint      } a joint and an end site share the name "Head"
#   3 Head    end site   }
#   4 Arm     joint      } two joints share the name "Arm"
#   5 Hand    end site
#   6 Arm     joint      }
#   7 Finger  end site
# In joint space the joints are Hips 0, Spine 1, Head 2, Arm 3, Arm 4.
_RIG = [
    {"name": "Hips", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
    {"name": "Spine", "parent": 0, "offset": [0, 1, 0], "rot_channels": "ZYX"},
    {"name": "Head", "parent": 1, "offset": [0, 1, 0], "rot_channels": "ZYX"},
    {"name": "Head", "parent": 2, "offset": [0, 0.5, 0]},
    {"name": "Arm", "parent": 1, "offset": [1, 0, 0], "rot_channels": "ZYX"},
    {"name": "Hand", "parent": 4, "offset": [1, 0, 0]},
    {"name": "Arm", "parent": 1, "offset": [-1.5, 0.3, 0.2], "rot_channels": "ZYX"},
    {"name": "Finger", "parent": 6, "offset": [-1, 0, 0.2]},
]
_FRAMES = 40
_SPINE, _HEAD_JOINT, _HEAD_END_SITE, _ARM, _HAND, _OTHER_ARM = 1, 2, 3, 4, 5, 6


def _rig() -> Bvh:
    """A clip on ``_RIG`` whose nodes all move, each differently."""
    rng = np.random.default_rng(127)
    joint_count = sum("rot_channels" in row for row in _RIG)
    time = np.linspace(0.0, 2.0 * np.pi, _FRAMES)
    frequencies = rng.uniform(0.5, 2.0, size=(1, joint_count, 3))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(1, joint_count, 3))
    angles = 0.6 * np.sin(frequencies * time[:, None, None] + phases)
    root_pos = np.stack([np.linspace(0.0, 2.0, _FRAMES), np.ones(_FRAMES), np.zeros(_FRAMES)], 1)
    return Bvh(nodes_from_table(_RIG), root_pos, angles, frame_time=1 / 30, world_up="+y")


def _renamed_rig() -> Bvh:
    """``_rig()`` with unique names: the second "Arm" is "Arm2", the end site "Head" is "HeadTip"."""
    clip = _rig()
    table = clip.to_node_table()
    table[_OTHER_ARM]["name"] = "Arm2"
    table[_HEAD_END_SITE]["name"] = "HeadTip"
    return clip.copy(nodes=table)


def _with_root(clip: Bvh, joints: Any) -> Any:
    """``joints`` with the root first, which ``extract_joints`` requires.

    A value that is not a list is passed on as it is, so that the refusal of a
    bare ``str`` or ``BvhNode`` reaches ``extract_joints``.
    """
    return [clip.root, *joints] if isinstance(joints, list) else joints


def _swinging(clip: Bvh) -> Bvh:
    """``clip``, edited in place, so that the default contact detection finds contacts on it.

    The detectors' height test refuses nodes above the root, and the rig's
    motion is too fast for their speed test. Here the rig hangs from a still
    root (world up ``-y``) and its spine swings between two held turns: the
    default detection then labels contacts on the first "Arm" and none on the
    other. The edit keeps the clip's nodes, so a selection taken before it
    still selects them.
    """
    held = np.repeat([0.0, 30.0, 0.0, 30.0], _FRAMES // 4)
    ramped = np.convolve(np.pad(held, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    angles = np.zeros_like(clip.joint_angles)
    angles[:, _SPINE, 0] = ramped
    clip.world_up = "-y"
    clip.root_pos = np.tile([0.0, 1.0, 0.0], (_FRAMES, 1))
    clip.joint_angles = angles
    return clip


def _landings(selection: Any) -> np.ndarray:
    """Contacts with three landings in each column, one column per entry of ``selection``."""
    columns = len(selection) if isinstance(selection, list) else 1
    landed = np.zeros(_FRAMES)
    landed[[*range(5, 10), *range(18, 23), *range(31, 36)]] = 1.0
    return np.tile(landed[:, None], (1, columns))


@dataclass(frozen=True)
class Entry:
    """One selecting parameter of one public entry point."""

    function: str
    parameter: str
    space: Literal["joint", "node"]
    takes_set: bool
    call: Callable[[Bvh, Any], Any]

    @property
    def id(self) -> str:
        return f"{self.function}({self.parameter}, space={self.space})"


ENTRIES = [
    Entry("Bvh.index", "node", "node", False, lambda bvh, s: bvh.index(s, space="node")),
    Entry("Bvh.index", "node", "joint", False, lambda bvh, s: bvh.index(s, space="joint")),
    *(
        Entry(f"Bvh.{method}", "node", "node", False, lambda bvh, s, m=method: getattr(bvh, m)(s))
        for method in (
            "curvature",
            "torsion",
            "movement_phase",
            "path_length",
            "directness",
            "ground_path",
            "smoothness",
            "velocity_reductions",
        )
    ),
    Entry(
        "Bvh.segment_axis_angle",
        "node_a",
        "node",
        False,
        lambda bvh, s: bvh.segment_axis_angle(s, "Spine"),
    ),
    Entry(
        "Bvh.segment_axis_angle",
        "node_b",
        "node",
        False,
        lambda bvh, s: bvh.segment_axis_angle("Hips", s),
    ),
    Entry(
        "Bvh.inter_joint_distance",
        "pairs",
        "node",
        False,
        lambda bvh, s: bvh.inter_joint_distance([("Hips", "Spine"), (s, "Hips")]),
    ),
    Entry(
        "Bvh.joint_angle", "a", "node", False, lambda bvh, s: bvh.joint_angle(s, "Spine", "Hips")
    ),
    Entry(
        "Bvh.joint_angle",
        "vertex",
        "node",
        False,
        lambda bvh, s: bvh.joint_angle("Hips", s, "Spine"),
    ),
    Entry(
        "Bvh.joint_angle", "b", "node", False, lambda bvh, s: bvh.joint_angle("Hips", "Spine", s)
    ),
    Entry(
        "Bvh.triangle_area",
        "a",
        "node",
        False,
        lambda bvh, s: bvh.triangle_area(s, "Spine", "Hips"),
    ),
    Entry(
        "Bvh.triangle_area",
        "b",
        "node",
        False,
        lambda bvh, s: bvh.triangle_area("Hips", s, "Spine"),
    ),
    Entry(
        "Bvh.triangle_area",
        "c",
        "node",
        False,
        lambda bvh, s: bvh.triangle_area("Hips", "Spine", s),
    ),
    Entry("Bvh.range_of_motion", "joint", "joint", False, lambda bvh, s: bvh.range_of_motion(s)),
    # The contact detectors return each node's speed, a signal that differs
    # between any two nodes of the rig, where the labels could agree.
    # method="velocity": the rig's nodes stand above its root, which the
    # height test of foot_contacts refuses as feet above the hips.
    Entry(
        "Bvh.ground_contacts",
        "nodes",
        "node",
        True,
        lambda bvh, s: bvh.ground_contacts(nodes=s, method="velocity", return_info=True)[1].speed,
    ),
    Entry(
        "analysis.ground_contacts",
        "nodes",
        "node",
        True,
        lambda bvh, s: (
            analysis.ground_contacts(bvh, nodes=s, method="velocity", return_info=True)[1].speed
        ),
    ),
    Entry(
        "Bvh.foot_contacts",
        "feet",
        "node",
        True,
        lambda bvh, s: bvh.foot_contacts(feet=s, method="velocity", return_info=True)[1].speed,
    ),
    Entry(
        "analysis.foot_contacts",
        "feet",
        "node",
        True,
        lambda bvh, s: (
            analysis.foot_contacts(bvh, feet=s, method="velocity", return_info=True)[1].speed
        ),
    ),
    Entry("Bvh.skeleton_size", "feet", "node", True, lambda bvh, s: bvh.skeleton_size(feet=s)),
    Entry(
        "analysis.skeleton_size",
        "feet",
        "node",
        True,
        lambda bvh, s: analysis.skeleton_size(bvh, feet=s),
    ),
    # Cadence counts the landings alone, and the feature array's feet only
    # label contacts, so both detect them on a swinging rig. With contacts
    # given, a stride is measured between the selected feet's positions at the
    # landings, which differ between any two nodes.
    Entry("Bvh.cadence", "feet", "node", True, lambda bvh, s: _swinging(bvh).cadence(feet=s)),
    Entry(
        "analysis.cadence",
        "feet",
        "node",
        True,
        lambda bvh, s: analysis.cadence(_swinging(bvh), feet=s),
    ),
    Entry(
        "Bvh.stride_length",
        "feet",
        "node",
        True,
        lambda bvh, s: bvh.stride_length(feet=s, contacts=_landings(s)),
    ),
    Entry(
        "analysis.stride_length",
        "feet",
        "node",
        True,
        lambda bvh, s: analysis.stride_length(bvh, feet=s, contacts=_landings(s)),
    ),
    Entry(
        "Bvh.gait_parameters",
        "feet",
        "node",
        True,
        lambda bvh, s: bvh.gait_parameters(feet=s, contacts=_landings(s)),
    ),
    Entry(
        "analysis.gait_parameters",
        "feet",
        "node",
        True,
        lambda bvh, s: analysis.gait_parameters(bvh, feet=s, contacts=_landings(s)),
    ),
    Entry(
        "Bvh.to_feature_array",
        "feet",
        "node",
        True,
        lambda bvh, s: _swinging(bvh).to_feature_array(include_foot_contacts=True, feet=s),
    ),
    Entry(
        "features.to_feature_array",
        "feet",
        "node",
        True,
        lambda bvh, s: features.to_feature_array(
            _swinging(bvh), include_foot_contacts=True, feet=s
        ),
    ),
    # The angles show which joint changed order: the skeleton's positions do
    # not change with it.
    Entry(
        "Bvh.change_euler_order",
        "joint",
        "joint",
        False,
        lambda bvh, s: bvh.change_euler_order("XYZ", joint=s).joint_angles,
    ),
    Entry(
        "Bvh.extract_joints",
        "joints",
        "joint",
        True,
        lambda bvh, s: bvh.extract_joints(_with_root(bvh, s)).node_positions(),
    ),
]


def _rows(predicate: Callable[[Entry], bool] = lambda entry: True) -> list[Any]:
    return [pytest.param(entry, id=entry.id) for entry in ENTRIES if predicate(entry)]


def _selection(entry: Entry, selector: Any) -> Any:
    """The value of ``entry``'s parameter that selects ``selector`` alone."""
    return [selector] if entry.takes_set else selector


def _arrays(result: Any) -> list[np.ndarray]:
    """A result as arrays: each field of a named tuple, or the result itself."""
    fields = result if isinstance(result, tuple) else (result,)
    return [np.asarray(field, dtype=np.float64) for field in fields]


def _assert_same(got: Any, want: Any) -> None:
    for got_array, want_array in zip(_arrays(got), _arrays(want), strict=True):
        np.testing.assert_allclose(got_array, want_array, equal_nan=True)


def _differ(a: Any, b: Any) -> bool:
    return any(
        not np.allclose(x, y, equal_nan=True) for x, y in zip(_arrays(a), _arrays(b), strict=True)
    )


@pytest.mark.parametrize("entry", _rows())
def test_a_repeated_name_is_reached_through_its_node(entry):
    # Each "Arm" joint, passed as a node, gives what its unique name gives on
    # the same clip with the namesake renamed.
    clip, renamed = _rig(), _renamed_rig()
    first = entry.call(clip, _selection(entry, clip.nodes[_ARM]))
    second = entry.call(clip, _selection(entry, clip.nodes[_OTHER_ARM]))
    _assert_same(first, entry.call(renamed, _selection(entry, "Arm")))
    _assert_same(second, entry.call(renamed, _selection(entry, "Arm2")))
    assert _differ(first, second)


@dataclass(frozen=True)
class Refusal:
    """A selection every row of a kind refuses, and the way out its message names."""

    name: str
    error: type[Exception]
    way_out: str
    applies: Callable[[Entry], bool]
    selection: Callable[[Entry, Bvh], Any]


def _stale_node(clip: Bvh) -> Any:
    """A node of ``clip`` taken before an in-place edit replaced its skeleton."""
    node = clip.nodes[_SPINE]
    clip.scale(2.0, inplace=True)
    return node


REFUSALS = [
    Refusal(
        "integer", TypeError, r"bvh\.nodes\[i\]", lambda e: True, lambda e, c: _selection(e, 3)
    ),
    Refusal(
        "bool", TypeError, r"bvh\.nodes\[i\]", lambda e: True, lambda e, c: _selection(e, True)
    ),
    Refusal(
        "foreign node",
        ValueError,
        r"one of bvh\.nodes",
        lambda e: True,
        lambda e, c: _selection(e, c.copy().nodes[_SPINE]),
    ),
    Refusal(
        "stale node",
        ValueError,
        r"one of bvh\.nodes",
        lambda e: True,
        lambda e, c: _selection(e, _stale_node(c)),
    ),
    Refusal(
        "missing name",
        ValueError,
        r"bvh\.node_index lists",
        lambda e: e.space == "node",
        lambda e, c: _selection(e, "Tail"),
    ),
    Refusal(
        "missing name",
        ValueError,
        r"bvh\.joint_names lists",
        lambda e: e.space == "joint",
        lambda e, c: _selection(e, "Tail"),
    ),
    Refusal(
        "repeated name",
        ValueError,
        rf"bvh\.nodes\[{_ARM}\], bvh\.nodes\[{_OTHER_ARM}\]\. Pass the node",
        lambda e: True,
        lambda e, c: _selection(e, "Arm"),
    ),
    Refusal(
        "bare str",
        TypeError,
        r"\['Spine'\]",
        lambda e: e.takes_set,
        lambda e, c: "Spine",
    ),
    Refusal(
        "bare node",
        TypeError,
        r"\[bvh\.nodes\[i\]\]",
        lambda e: e.takes_set,
        lambda e, c: c.nodes[_SPINE],
    ),
    Refusal(
        "node selected twice",
        ValueError,
        rf"bvh\.nodes\[{_SPINE}\] \('Spine'\) more than once\. Select each node once",
        lambda e: e.takes_set,
        # The node between the two is a joint, which every space accepts.
        lambda e, c: [c.nodes[_SPINE], c.nodes[_ARM], c.nodes[_SPINE]],
    ),
    Refusal(
        "name and its node",
        ValueError,
        rf"bvh\.nodes\[{_SPINE}\] \('Spine'\) more than once\. Select each node once",
        lambda e: e.takes_set,
        lambda e, c: ["Spine", c.nodes[_SPINE]],
    ),
    Refusal(
        "end site by name",
        ValueError,
        rf"its parent bvh\.nodes\[{_ARM}\]",
        lambda e: e.space == "joint",
        lambda e, c: _selection(e, "Hand"),
    ),
    Refusal(
        "end site by node",
        ValueError,
        rf"its parent bvh\.nodes\[{_ARM}\]",
        lambda e: e.space == "joint",
        lambda e, c: _selection(e, c.nodes[_HAND]),
    ),
]


@pytest.mark.parametrize(
    ("entry", "refusal"),
    [
        pytest.param(entry, refusal, id=f"{entry.id}-{refusal.name}")
        for entry in ENTRIES
        for refusal in REFUSALS
        if refusal.applies(entry)
    ],
)
def test_a_selection_is_refused_with_its_way_out(entry, refusal):
    clip = _rig()
    selection = refusal.selection(entry, clip)
    with pytest.raises(refusal.error, match=refusal.way_out):
        entry.call(clip, selection)


@pytest.mark.parametrize("entry", _rows(lambda entry: entry.space == "joint"))
def test_a_name_a_joint_and_an_end_site_share_selects_the_joint_in_joint_space(entry):
    clip = _rig()
    _assert_same(
        entry.call(clip, _selection(entry, "Head")),
        entry.call(clip, _selection(entry, clip.nodes[_HEAD_JOINT])),
    )


@pytest.mark.parametrize("entry", _rows(lambda entry: entry.space == "node"))
def test_a_name_a_joint_and_an_end_site_share_is_refused_in_node_space(entry):
    clip = _rig()
    way_out = rf"bvh\.nodes\[{_HEAD_JOINT}\], bvh\.nodes\[{_HEAD_END_SITE}\]\. Pass the node"
    with pytest.raises(ValueError, match=way_out):
        entry.call(clip, _selection(entry, "Head"))


def test_index_counts_a_node_in_the_requested_space():
    # The two "Arm" joints are nodes 4 and 6; in joint space the end sites
    # before them (node 3, and node 5 for the second) are skipped: 3 and 4.
    clip = _rig()
    arms = [clip.nodes[_ARM], clip.nodes[_OTHER_ARM]]
    assert [clip.index(arm, space="node") for arm in arms] == [4, 6]
    assert [clip.index(arm, space="joint") for arm in arms] == [3, 4]


def test_change_euler_order_changes_only_the_joint_it_is_given():
    clip = _rig()
    # The first "Arm": joint_index keeps the last node of a repeated name.
    changed = clip.change_euler_order("XYZ", joint=clip.nodes[_ARM])
    orders = ["".join(node.rot_channels) for node in changed.nodes if not node.is_end_site()]
    assert orders == ["ZYX", "ZYX", "ZYX", "XYZ", "ZYX"]


def test_extract_joints_keeps_one_of_two_joints_that_share_a_name():
    clip = _rig()
    # The first "Arm": joint_index keeps the last node of a repeated name.
    reduced = clip.extract_joints([clip.root, clip.nodes[_SPINE], clip.nodes[_ARM]])
    assert [node.name for node in reduced.nodes] == ["Hips", "Spine", "Arm", "Hand"]
    np.testing.assert_array_equal(reduced.joint_angles, clip.joint_angles[:, [0, 1, 3]])


# The guard over the public signatures. It reads the signatures of every
# public function and method of Bvh, analysis and features, so that a new one
# cannot select nodes outside the rule, nor skip ENTRIES.

# A parameter whose name says it selects nodes. The annotation finds the others
# (a, b, c, vertex, pairs): they are typed with NodeSelector.
_SELECTING_NAME = re.compile(r"(joint|node)s?(_[a-z])?|feet")
# Parameters named like a selection that take a whole skeleton, not a selection
# of the clip's nodes.
_SKELETONS = {("Bvh.__init__", "nodes"), ("Bvh.copy", "nodes")}
# The parameters typed with int, each of which takes a frame, an axis, a lag,
# a padding level or a count, not a node. The name and the annotation miss a
# selection named otherwise ("foot", "end_site"), so a new parameter typed with
# int fails the guard until it is listed here.
_INTEGERS_NOT_NODES = {
    ("Bvh.forward_at", "frame"),
    ("Bvh.left_at", "frame"),
    ("Bvh.joint_positions", "frame"),
    ("Bvh.node_positions", "frame"),
    ("Bvh.feature_array_layout", "num_feet"),
    ("analysis.lagged_covariance", "lag"),
    ("analysis.range_of_motion", "axis"),
    ("analysis.sparc", "padlevel"),
    ("analysis.zero_crossings", "axis"),
    ("features.feature_array_layout", "num_joints"),
    ("features.feature_array_layout", "num_feet"),
}


def _public_callables() -> list[tuple[str, Callable[..., Any]]]:
    """Each public function of analysis and features and each public method of Bvh, by name."""
    callables = []
    for name, member in vars(Bvh).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if inspect.isfunction(member):
            callables.append((f"Bvh.{name}", member))
    for module in (analysis, features):
        short = module.__name__.rpartition(".")[2]
        for name, member in vars(module).items():
            if (
                not name.startswith("_")
                and inspect.isfunction(member)
                and member.__module__ == module.__name__
            ):
                callables.append((f"{short}.{name}", member))
    return callables


def _parameters() -> list[tuple[str, inspect.Parameter]]:
    """Each parameter of each public callable, with the callable's name; ``self`` left out.

    The modules import ``annotations`` from ``__future__``, so each annotation
    is the source text, in which ``NodeSelector`` keeps its name.
    """
    return [
        (function, parameter)
        for function, member in _public_callables()
        for parameter in inspect.signature(member).parameters.values()
        if parameter.name != "self"
    ]


def _selecting_parameters() -> set[tuple[str, str]]:
    return {
        (function, parameter.name)
        for function, parameter in _parameters()
        if (
            _SELECTING_NAME.fullmatch(parameter.name) or "NodeSelector" in str(parameter.annotation)
        )
        and (function, parameter.name) not in _SKELETONS
    }


def test_the_guard_finds_the_public_callables():
    # A walk that found nothing would let every check below pass.
    functions = {function for function, _ in _public_callables()}
    assert {"Bvh.index", "Bvh.path_length", "analysis.ground_contacts"} <= functions
    assert "features.to_feature_array" in functions


def test_no_public_signature_takes_coords():
    # The clip's own positions are read from it; positions the caller holds
    # take bvh.index, then pybvh.geometry or pybvh.signal.
    assert [function for function, parameter in _parameters() if parameter.name == "coords"] == []


def test_every_selecting_parameter_is_typed_with_node_selector_and_never_int():
    annotations = {
        (function, parameter.name): str(parameter.annotation)
        for function, parameter in _parameters()
    }
    wrong = {
        selecting: annotations[selecting]
        for selecting in _selecting_parameters()
        if "NodeSelector" not in annotations[selecting]
        or re.search(r"\bint\b", annotations[selecting])
    }
    assert wrong == {}


def test_a_parameter_typed_with_int_is_listed_as_not_a_node():
    typed_int = {
        (function, parameter.name)
        for function, parameter in _parameters()
        if re.search(r"\bint\b", str(parameter.annotation))
    }
    assert typed_int - _INTEGERS_NOT_NODES == set()


def test_every_selecting_parameter_has_its_rows_in_entries():
    selecting = _selecting_parameters()
    listed = {(entry.function, entry.parameter) for entry in ENTRIES}
    assert selecting - listed == set(), "selecting parameters without a row in ENTRIES"
    assert listed - selecting == set(), "rows of ENTRIES whose parameter is gone"


@pytest.mark.parametrize("entry", _rows(lambda entry: entry.parameter in ("joint", "joints")))
def test_a_parameter_named_joint_refuses_an_end_site(entry):
    clip = _rig()
    with pytest.raises(ValueError, match=r"is an end site"):
        entry.call(clip, _selection(entry, clip.nodes[_HAND]))


# Bvh.index takes its space from its caller, so its name says nothing of it.
@pytest.mark.parametrize(
    "entry",
    _rows(
        lambda entry: entry.parameter not in ("joint", "joints") and entry.function != "Bvh.index"
    ),
)
def test_a_parameter_not_named_joint_accepts_an_end_site(entry):
    clip = _rig()
    entry.call(clip, _selection(entry, clip.nodes[_HAND]))
