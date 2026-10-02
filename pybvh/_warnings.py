"""Where a pybvh warning points: the user's line, whatever the path to it."""

from __future__ import annotations

import inspect
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def user_stacklevel() -> int:
    """The ``stacklevel`` that makes a ``warnings.warn`` in the caller
    point at the first frame outside the pybvh package: the user's call.

    A fixed ``stacklevel`` counts frames from the ``warn`` call, so it
    is right from one call path only. A warning reached both from a
    public function and through a convenience method that wraps it
    (``bvhplot.play`` and ``Bvh.play``), or from a method a user may
    call directly, needs the count taken at run time. Frames are
    walked from the function that calls this one until a frame's file
    is not under the package's directory. Python 3.12's
    ``warnings.warn(skip_file_prefixes=...)`` does the same, and would
    replace this once 3.12 is the oldest version supported.
    """
    frame = inspect.currentframe()
    level = 0
    try:
        while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
            _PACKAGE_DIR
        ):
            frame = frame.f_back
            level += 1
    finally:
        del frame
    return level
