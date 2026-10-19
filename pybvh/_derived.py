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


_MOTION_CHANGES: Final = frozenset({_Change.SKELETON, _Change.MOTION, _Change.USER_REPLACED_MOTION})

# Each derived value, and the changes that clear it.
_CLEARED_BY: Final[dict[str, frozenset[_Change]]] = {
    "node_positions": _MOTION_CHANGES,
    "guessed_world_up": _MOTION_CHANGES,
    "floor_height": _MOTION_CHANGES,
}


class DerivedState:
    """Every value a clip derives from its inputs, each ``None`` until filled.

    Attributes
    ----------
    node_positions : ndarray or None
        World-frame forward kinematics of every frame, shape ``(F, N, 3)``.
        Stored read-only: a write into the array raises at the line that
        made it.
    floor_height : float or None
        The scene ground, along world up (``Bvh.floor_height``).
    guessed_world_up : str or None
        The world up inferred from the motion, used when the user set none.
    """

    def __init__(self) -> None:
        self._node_positions: npt.NDArray[np.float64] | None = None
        self.floor_height: float | None = None
        self.guessed_world_up: str | None = None

    @property
    def node_positions(self) -> npt.NDArray[np.float64] | None:
        return self._node_positions

    @node_positions.setter
    def node_positions(self, value: npt.NDArray[np.float64] | None) -> None:
        if value is not None:
            value.flags.writeable = False
        self._node_positions = value

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

    def _clear(self, change: _Change) -> None:
        for name, changes in _CLEARED_BY.items():
            if change in changes:
                setattr(self, name, None)
