"""The lifecycle of a clip's derived state (ADR 0003).

A :class:`~pybvh.bvh.Bvh` computes some values from its inputs (its
skeleton, its motion, its world-up override and its L/R mapping setting)
and keeps them until an input changes. This module holds those values and knows when each one is
stale; ``Bvh`` keeps the code that computes them, fills an empty value on
read, and reports every change of an input to its :class:`DerivedState`.

The table :data:`_CLEARED_BY` is the one place that says which change
clears which value.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._orientation import OrientationReport, _WorldUpGuess


class _Change(Enum):
    """A change to one of a clip's inputs."""

    SKELETON = auto()
    MOTION = auto()
    USER_REPLACED_MOTION = auto()
    WORLD_UP_OVERRIDE = auto()
    LR_MAPPING_SETTING = auto()


_SKELETON_OR_MOTION: Final = frozenset(
    {_Change.SKELETON, _Change.MOTION, _Change.USER_REPLACED_MOTION}
)

# Each derived value, and the changes that clear it.
_CLEARED_BY: Final[dict[str, frozenset[_Change]]] = {
    "node_index": frozenset({_Change.SKELETON}),
    "joint_index": frozenset({_Change.SKELETON}),
    "names_lr_mapping": frozenset({_Change.SKELETON}),
    "rest_pose_positions": frozenset({_Change.SKELETON}),
    "node_positions": _SKELETON_OR_MOTION,
    # Measured along world up, so cleared when the axis set or guessed may change.
    "floor_height": _SKELETON_OR_MOTION | {_Change.WORLD_UP_OVERRIDE, _Change.LR_MAPPING_SETTING},
    # The guessed world up, inferred here or inherited from the clip this
    # one was derived from, outlives pybvh's own edits: none of them
    # changes which way is up but reorient_world_up, which sets the axis
    # as an override and turns the guess with the frames (ADR 0003,
    # decision 3). The user setting the L/R mapping starts the clip's own
    # guess again: on a rig with no up marker it is read from rest up,
    # which is read from the legs of the mapping. A guess from the first
    # frame or the house default does not read the mapping, so the guess
    # made again is the same one. The inherited guess stays, as it does
    # through a skeleton change: copy(nodes=), extract_joints and
    # unpickling set the mapping on a clip that has just inherited it.
    "inferred_world_up": frozenset({_Change.USER_REPLACED_MOTION, _Change.LR_MAPPING_SETTING}),
    "inherited_world_up": frozenset({_Change.USER_REPLACED_MOTION}),
    # The guess inferred here that Bvh.world_up has warned about (it warns
    # once per guess), so it goes with a guess the motion replaced. A
    # mapping change keeps it: the guess made again after it warns only
    # where it differs from this one, which a rest-up guess read from the
    # new legs can.
    "world_up_guess_announced": frozenset({_Change.USER_REPLACED_MOTION}),
    # Read from the rest pose, the L/R mapping in effect and world up, set
    # or guessed (ADR 0004); pybvh's own motion writes change none of them.
    "orientation_report": frozenset(
        {
            _Change.SKELETON,
            _Change.USER_REPLACED_MOTION,
            _Change.WORLD_UP_OVERRIDE,
            _Change.LR_MAPPING_SETTING,
        }
    ),
}


class DerivedState:
    """Every value a clip derives from its inputs, each ``None`` until filled.

    Attributes
    ----------
    node_positions : ndarray or None
        World-frame forward kinematics of every frame, shape ``(F, N, 3)``.
        Stored read-only: a write into the array raises at the line that
        made it.
    rest_pose_positions : ndarray or None
        The rest pose (every angle zero, root at the origin), shape
        ``(N, 3)``. Stored read-only, as ``node_positions`` is.
    floor_height : float or None
        The scene ground, along world up (``Bvh.floor_height``).
    inferred_world_up : _WorldUpGuess or None
        The world up guessed from this clip's own motion, with its source,
        used when the user set none and the clip inherited none.
    inherited_world_up : _WorldUpGuess or None
        The world up of the clip this one was derived from, guessed there,
        with the source of that guess, used when the user set none.
    world_up_guess_announced : _WorldUpGuess or None
        The guess inferred here that ``Bvh.world_up`` last gave its
        warnings about, which it gives once per guess; ``None`` until
        then.
    node_index, joint_index : dict or None
        Node name to its index in ``nodes``; joint name to its index in
        ``joint_angles`` axis 1.
    names_lr_mapping : dict or None
        The L/R pairs detected from the joint names, one direction (left to
        right); ``{}`` when the names give none.
    orientation_report : OrientationReport or None
        Each orientation fact of the clip with its source
        (``Bvh.orientation_report``).
    """

    # Slots make a name in _CLEARED_BY that is not a field raise on the
    # first change instead of setting a stray attribute.
    __slots__ = (
        "_node_positions",
        "_rest_pose_positions",
        "floor_height",
        "inferred_world_up",
        "inherited_world_up",
        "world_up_guess_announced",
        "node_index",
        "joint_index",
        "names_lr_mapping",
        "orientation_report",
    )

    def __init__(self) -> None:
        self._node_positions: npt.NDArray[np.float64] | None = None
        self._rest_pose_positions: npt.NDArray[np.float64] | None = None
        self.floor_height: float | None = None
        self.inferred_world_up: _WorldUpGuess | None = None
        self.inherited_world_up: _WorldUpGuess | None = None
        self.world_up_guess_announced: _WorldUpGuess | None = None
        self.node_index: dict[str, int] | None = None
        self.joint_index: dict[str, int] | None = None
        self.names_lr_mapping: dict[str, str] | None = None
        self.orientation_report: OrientationReport | None = None

    @property
    def node_positions(self) -> npt.NDArray[np.float64] | None:
        return self._node_positions

    @node_positions.setter
    def node_positions(self, value: npt.NDArray[np.float64] | None) -> None:
        self._node_positions = _read_only(value)

    @property
    def rest_pose_positions(self) -> npt.NDArray[np.float64] | None:
        return self._rest_pose_positions

    @rest_pose_positions.setter
    def rest_pose_positions(self, value: npt.NDArray[np.float64] | None) -> None:
        self._rest_pose_positions = _read_only(value)

    def skeleton_changed(self) -> None:
        """The skeleton was replaced or edited."""
        self._clear(_Change.SKELETON)

    def motion_changed(self) -> None:
        """pybvh rewrote the motion (``root_pos`` or ``joint_angles``)."""
        self._clear(_Change.MOTION)

    def user_replaced_motion(self) -> None:
        """The user assigned motion: the ``root_pos`` or ``joint_angles``
        setter, or a frame assignment (``bvh[i] = other``)."""
        self._clear(_Change.USER_REPLACED_MOTION)

    def world_up_changed(self) -> None:
        """The user set or cleared the world-up override, or
        ``reorient_world_up`` turned the guessed axis with the frames."""
        self._clear(_Change.WORLD_UP_OVERRIDE)

    def lr_mapping_changed(self) -> None:
        """The user set or cleared the L/R mapping (``Bvh.lr_mapping``)."""
        self._clear(_Change.LR_MAPPING_SETTING)

    def _clear(self, change: _Change) -> None:
        for name, changes in _CLEARED_BY.items():
            if change in changes:
                setattr(self, name, None)


def _read_only(value: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64] | None:
    """Mark a derived array read-only, so a write raises where it is made."""
    if value is not None:
        value.flags.writeable = False
    return value
