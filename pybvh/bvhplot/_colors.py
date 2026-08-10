"""Color resolution shared by every backend.

_common.py stays free of plotting imports (enforced by test); this
module owns the pieces that need matplotlib's color parser: the
light/dark background split, the floor palette, and the conversion of
per-bone/per-node style colors to 0-255 RGB. Backends do only their
own format packing at their border (BGR flip, uint32 shift).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._common import Style, bone_colors_for_view

if TYPE_CHECKING:
    from ._common import SkeletonView


def is_dark_background(background: object) -> bool:
    """Whether a background color needs the dark floor/accent palette.

    Decided by relative luminance (< 0.5), not string matching, so
    "#F5F5F7", "snow", or (1, 1, 1) all correctly count as light.
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(background)  # type: ignore[arg-type]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5


# Floor palettes: face/edge for the solid plane, grid line color, and
# the checkerboard's two alternating shades.
FLOOR_LIGHT = {
    "face": "#E8E8EC",
    "edge": "#D0D0D8",
    "grid": "#C8C8D0",
    "checker": ("#EDEDF1", "#DCDCE3"),
}
FLOOR_DARK = {
    "face": "#2A2E36",
    "edge": "#3A3F4A",
    "grid": "#3A3F4A",
    "checker": ("#2A2E36", "#1E2127"),
}


def floor_palette(style: Style) -> dict:
    """The floor color set matching the style's background luminance."""
    return FLOOR_DARK if is_dark_background(style.background) else FLOOR_LIGHT


def rgb255(color: object) -> tuple[int, int, int]:
    """Any matplotlib-parseable color -> (r, g, b) 0-255 ints."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)  # type: ignore[arg-type]
    return (int(r * 255), int(g * 255), int(b * 255))


def bone_colors_255(
    view: SkeletonView,
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> list[tuple[int, int, int]]:
    """Per-bone colors as 0-255 RGB, parallel to ``view.bones``."""
    return [rgb255(c) for c in
            bone_colors_for_view(view, style, view_index, n_skeletons)]


def node_colors_255(
    view: SkeletonView,
    style: Style,
    view_index: int,
    n_skeletons: int,
    bone_rgb: list[tuple[int, int, int]] | None = None,
) -> npt.NDArray[np.uint8]:
    """Per-node colors as a (N, 3) uint8 array.

    The single owner of the node-coloring rule: a node takes the color
    of the bone whose child it is; nodes that are nobody's child (the
    root) fall back to the style's spine color. Pass *bone_rgb* to
    reuse an already-computed :func:`bone_colors_255` result.
    """
    if bone_rgb is None:
        bone_rgb = bone_colors_255(view, style, view_index, n_skeletons)
    spine = rgb255(style.chain_colors.get("spine", "#3A3F4A"))
    n_nodes = view.coords.shape[1]
    out = np.empty((n_nodes, 3), dtype=np.uint8)
    out[:] = spine
    for (_parent, child), color in zip(view.bones, bone_rgb):
        out[child] = color
    return out
