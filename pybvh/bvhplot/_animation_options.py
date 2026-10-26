"""What an animation draws beside its Scene.

The router gathers the options of ``render`` and ``play`` that say what
is drawn over the frames into one :class:`AnimationOptions`, which the
backend reads; where the frames go is the video module's
:class:`~._video.VideoTarget`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._viewport import Turntable, check_schedule


@dataclass(frozen=True)
class AnimationOptions:
    """What is drawn over an animation's frames, besides the Scene's
    skeletons; frozen, and checked when built.

    Attributes
    ----------
    schedule : {"fixed", "follow", "turntable"} or Turntable
        The camera's schedule, handed to the viewport untouched
        (:func:`~._viewport.make_viewport`): ``"turntable"`` is one
        orbit over the frames, a :class:`~._viewport.Turntable` one
        every ``period`` frames.
    ghost : int
        Faded trailing poses behind each live skeleton, 0 or more.
    trajectory : bool
        Draw each root's trail on the floor, growing with playback.
    frame_counter : bool
        Stamp a ``Frame f/F`` counter in the corner.
    """

    schedule: str | Turntable = "fixed"
    ghost: int = 0
    trajectory: bool = False
    frame_counter: bool = False

    def __post_init__(self) -> None:
        check_schedule(self.schedule)
        if not (isinstance(self.ghost, int) and self.ghost >= 0):
            raise ValueError(f"ghost must be an integer >= 0, got {self.ghost!r}")
