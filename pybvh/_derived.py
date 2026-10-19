"""The lifecycle of a clip's derived state (ADR 0003).

A :class:`~pybvh.bvh.Bvh` computes some values from its inputs (its
skeleton, its motion and its world-up override) and keeps them until an
input changes. This module holds those values and knows when each one is
stale; ``Bvh`` keeps the code that computes them, fills an empty value on
read, and reports every change of an input to its :class:`DerivedState`.

The table :data:`_CLEARED_BY` is the one place that says which change
clears which value. Until the guessed world up is inherited by derived
clips (#112), it is cleared like the other motion dependents; ADR 0003
records the row it ends on, cleared only when the user replaces the
motion.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Final

import numpy as np
import numpy.typing as npt


class _Change(Enum):
    """A change to one of a clip's inputs."""

    SKELETON = auto()
    MOTION = auto()
    USER_REPLACED_MOTION = auto()
    WORLD_UP_OVERRIDE = auto()


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
    # ADR 0003 ends this row at USER_REPLACED_MOTION alone, once derived
    # clips inherit the guess.
    "guessed_world_up": _SKELETON_OR_MOTION,
    "floor_height": _SKELETON_OR_MOTION | {_Change.WORLD_UP_OVERRIDE},
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
    guessed_world_up : str or None
        The world up inferred from the motion, used when the user set none.
    node_index, joint_index : dict or None
        Node name to its index in ``nodes``; joint name to its index in
        ``joint_angles`` axis 1.
    names_lr_mapping : dict or None
        The L/R pairs detected from the joint names, one direction (left to
        right); ``{}`` when the names give none.
    """

    # Slots make a name in _CLEARED_BY that is not a field raise on the
    # first change instead of setting a stray attribute.
    __slots__ = (
        "_node_positions",
        "_rest_pose_positions",
        "floor_height",
        "guessed_world_up",
        "node_index",
        "joint_index",
        "names_lr_mapping",
    )

    def __init__(self) -> None:
        self._node_positions: npt.NDArray[np.float64] | None = None
        self._rest_pose_positions: npt.NDArray[np.float64] | None = None
        self.floor_height: float | None = None
        self.guessed_world_up: str | None = None
        self.node_index: dict[str, int] | None = None
        self.joint_index: dict[str, int] | None = None
        self.names_lr_mapping: dict[str, str] | None = None

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
        """The user set or cleared the world-up override."""
        self._clear(_Change.WORLD_UP_OVERRIDE)

    def _clear(self, change: _Change) -> None:
        for name, changes in _CLEARED_BY.items():
            if change in changes:
                setattr(self, name, None)


def _read_only(value: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64] | None:
    """Mark a cached array read-only, so a write raises where it is made."""
    if value is not None:
        value.flags.writeable = False
    return value
