"""The pixels of a vedo picture where the skeleton hides the floor.

For the tests that a floor drawn behind the skeleton leaves it as it
is, on the offscreen renderer and on the viewer alike.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# The viewer's fast quality draws its joints at an alpha of 0.9, so a
# joint lets a tenth of what lies behind it through: at most 26 levels.
# Where a joint covers the edge of a bone, some of the floor comes
# through with the bone (up to 15 levels seen, and none with the window's
# multisampling off). A floor drawn over the skeleton changed it by up to
# 108 levels (the floor over a joint) and 205 (a grid line over a capsule).
CHANGED_LEVELS = 26


def skeleton_pixels(bare: npt.NDArray, opaque: npt.NDArray, margin: int = 2) -> npt.NDArray:
    """The pixels where the skeleton hides any floor behind it.

    *bare* is the picture without a floor, *opaque* the same picture
    over an opaque solid floor. The skeleton is where *bare* is not the
    background and *opaque* draws it unchanged, which leaves out what
    the floor covers: the lowest joints' spheres reach through it, and
    where two capsules meet a sliver can show what lies behind them.
    A pixel counts at least *margin* pixels in from that outline:
    antialiasing blends in what lies behind, and where a sphere passes
    through the floor the two are at one depth, which either can win.
    """
    bare = bare.astype(int)
    drawn = (bare != bare[0, 0]).any(axis=-1)
    unchanged = np.abs(opaque.astype(int) - bare).max(axis=-1) <= CHANGED_LEVELS
    inside = drawn & unchanged
    for _ in range(margin):
        inner = np.zeros_like(inside)
        inner[1:-1, 1:-1] = (
            inside[1:-1, 1:-1]
            & inside[:-2, 1:-1]
            & inside[2:, 1:-1]
            & inside[1:-1, :-2]
            & inside[1:-1, 2:]
        )
        inside = inner
    return inside
