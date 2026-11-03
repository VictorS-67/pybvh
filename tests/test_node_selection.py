"""Every entry point that selects nodes takes a name or a ``BvhNode`` (#127).

A name must match exactly one node of the index space the entry point works
in, and a node is resolved by identity among the clip's ``nodes``. One table,
``ENTRIES``, lists every selecting parameter of the public surface; each test
below runs over the rows it applies to.

To add an entry point, add one row per selecting parameter: ``call`` takes the
clip and the value of that parameter and calls the entry point with valid
values for every other parameter. ``space`` is the index space the entry point
resolves the parameter in, and ``takes_set`` says whether the parameter is a
sequence of selectors: the tests then pass each selector inside a list, and
add the refusals of a bare ``str`` or ``BvhNode`` and of a node selected twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest

from pybvh import Bvh, nodes_from_table

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
    {"name": "Arm", "parent": 1, "offset": [-1, 0, 0], "rot_channels": "ZYX"},
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
    # The plural form, which every entry point that takes a set calls. It has
    # no public caller yet: the contact detectors' rows replace this one.
    Entry("Bvh._indices", "nodes", "node", True, lambda bvh, s: bvh._indices(s, space="node")),
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
        lambda e, c: [c.nodes[_SPINE], c.nodes[_HAND], c.nodes[_SPINE]],
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
