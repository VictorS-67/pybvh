"""Color resolution shared by every backend.

_style.py stays free of plotting imports (enforced by test); this
module owns the pieces that need matplotlib's color parser: the
light/dark background split, the floor palette, the grid box colors
derived from the background, and the conversion of per-bone/per-node
style colors to 0-255 RGB. Backends do only their
own format packing at their border (BGR flip, uint32 shift).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._style import (
    Style,
    bone_colors_for_view,
    skeleton_color,
    spine_color,
)

if TYPE_CHECKING:
    from ._scene import SkeletonView


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


# How far the grid box's lines and labels step from the background
# toward black (light background) or white (dark one), as a fraction
# of the way. Chosen so that on white they land on k3d's own defaults,
# 0xE6E6E6 and 0x444444.
GRID_BOX_LINE_STEP = (0xFF - 0xE6) / 0xFF
GRID_BOX_LABEL_STEP = (0xFF - 0x44) / 0xFF


def grid_box_colors(
    style: Style,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The (line, label) 0-255 RGB colors of a grid box on the style's
    background.

    Both are the background moved toward black on a light background
    and toward white on a dark one (the floor palette's luminance
    split), the lines a short step so they frame the scene without
    competing with the skeleton, the labels most of the way so they
    read clearly.
    """
    background = np.array(rgb255(style.background))
    pole = 255 if is_dark_background(style.background) else 0

    def step_toward_pole(fraction: float) -> tuple[int, int, int]:
        r, g, b = (int(v) for v in np.rint(
            background + fraction * (pole - background)))
        return (r, g, b)

    return (step_toward_pole(GRID_BOX_LINE_STEP),
            step_toward_pole(GRID_BOX_LABEL_STEP))


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


def skeleton_color_255(
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> tuple[int, int, int]:
    """:func:`~._style.skeleton_color` as 0-255 RGB: what a skeleton's
    label and root trail are drawn in."""
    return rgb255(skeleton_color(style, view_index, n_skeletons))


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
    spine = rgb255(spine_color(style))
    n_nodes = view.coords.shape[1]
    out = np.empty((n_nodes, 3), dtype=np.uint8)
    out[:] = spine
    for (_parent, child), color in zip(view.bones, bone_rgb):
        out[child] = color
    return out
