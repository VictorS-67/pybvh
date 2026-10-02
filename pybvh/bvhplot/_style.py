"""The look of a figure: :class:`Style`, its presets, the palettes and
the ghost and trace conventions every backend shares.

Pure data, no plotting library imports; the pieces that need
matplotlib's color parser live in :mod:`._colors`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._scene import SkeletonView


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# RGB is the canonical channel order everywhere in bvhplot; the OpenCV
# backend converts to BGR at its own border (its channel-order quirk
# stays its own concern).

# Per-skeleton comparison palette (multi-skeleton figures)
PALETTE_RGB = [
    (50, 120, 255),   # blue
    (220, 50, 50),    # red
    (50, 180, 50),    # green
    (50, 130, 200),   # teal
    (200, 100, 50),   # orange
    (200, 50, 200),   # magenta
]
PALETTE_MPL = [(r / 255, g / 255, b / 255) for (r, g, b) in PALETTE_RGB]

# Per-chain palette (single-skeleton figures): left = warm, right = cool,
# spine dark — side is encoded by temperature, chain by shade. Derived
# from the Okabe-Ito colorblind-safe palette.
CHAIN_COLORS = {
    "spine": "#3A3F4A",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}
# Dark-background variant: same warm/cool encoding, spine and joints
# lightened so they read against a near-black ground.
CHAIN_COLORS_DARK = {
    "spine": "#C8CCD6",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# The bone width at which the backends that size bones in scene units
# (the vedo capsules, k3d's lines) draw them at their base fraction of
# the body: the paper default. Backends that size bones in pixels or
# points (matplotlib, OpenCV, the vedo viewer's fast mode) take
# bone_width as it is.
ANCHOR_BONE_WIDTH = 3.0


def bone_width_scale(bone_width: float) -> float:
    """The factor a style's *bone_width* scales bones drawn in scene
    units by: 1 at ``ANCHOR_BONE_WIDTH``, the paper default."""
    return bone_width / ANCHOR_BONE_WIDTH


# Preset field values. "paper" is the publication-grade default;
# "debug" reproduces the pre-0.9.0 output exactly (single blue, full
# axes, no floor); "dark" is the paper look on a near-black ground.
_STYLE_PRESETS: dict[str, dict[str, object]] = {
    "paper": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#23262E",
        floor="solid",
        floor_alpha=0.85,
        background="white",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
    "debug": dict(
        bone_width=2.5,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="single",
        chain_colors=CHAIN_COLORS,
        joint_markers=False,
        joint_size=9.0,
        joint_color="#23262E",
        floor=None,
        floor_alpha=0.85,
        background="white",
        axes="full",
        projection="persp",
        dpi=None,
        supersample=1,
        shadow=False,
        ghost_spacing=0.3,
    ),
    "dark": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS_DARK,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#E8E8EC",
        floor="solid",
        floor_alpha=0.85,
        background="#16181D",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
}

_VALID_COLOR_MODES = {"auto", "chains", "skeleton", "single"}
_VALID_FLOORS = {None, "solid", "grid", "checker"}
_VALID_AXES = {"off", "full"}
_VALID_PROJECTIONS = {"persp", "ortho"}


@dataclass(frozen=True, init=False)
class Style:
    """Visual styling for every bvhplot function.

    Construct from a preset name plus any field overrides::

        Style("paper")                    # the defaults
        Style("paper", floor=None)        # paper look, no ground plane
        Style("dark", bone_width=4.0)

    Presets: ``"paper"`` (publication-grade default: ground plane,
    per-chain colors, joint markers, axes off), ``"debug"`` (the
    pre-0.9.0 look: single blue skeleton, full axes and ticks, no
    floor — for coordinate inspection), ``"dark"`` (paper on a
    near-black ground, for slides and project pages).

    Fields split into two documented groups. **Look** fields apply to
    every figure/video output (matplotlib, OpenCV, vedo offscreen):
    ``bone_width``, ``bone_color``, ``color_mode``, ``chain_colors``,
    ``joint_markers``, ``joint_size``, ``joint_color``, ``floor``,
    ``floor_alpha``, ``background``, ``axes``, ``projection``,
    ``ghost_spacing`` (seconds of clip time, not playback time,
    between the faded trailing poses that ``render(ghost=...)`` draws;
    see there). The two *interactive viewers* apply the subset that
    has meaning in a live window: background, bone width, and
    single-skeleton chain colors in both; in the vedo viewer,
    ``color_mode`` and ``bone_color`` in full, in both qualities, and
    floor kind, where ``"checker"`` falls back to ``"grid"``. Fields
    outside that subset (``axes``, ``projection``, ``joint_markers``,
    ...) do not alter the viewers. **Output** fields apply only where
    raster output is produced: ``dpi`` (matplotlib figures),
    ``supersample`` (OpenCV export), ``shadow`` (vedo offscreen
    renders).

    ``background`` also colors the grid box of ``play(backend="k3d")``:
    its lines and labels are the background stepped toward black on a
    light background and toward white on a dark one (luminance below
    one half, the split that picks the floor colors), the lines about
    10% of the way, the labels about 73%. Those fractions put white
    exactly on k3d's own defaults, ``#E6E6E6`` lines and ``#444444``
    labels, so a white background looks as it always did; on
    ``"dark"`` (``#16181D``) they give ``#2D2F33`` lines and
    ``#C1C1C3`` labels. The alternative, one fixed pair for all light
    and one for all dark backgrounds, differs as soon as the
    background is not white: fixed lines can vanish into an off-white
    background, and the fixed dark pair ignores how dark it is.

    ``color_mode``: ``"auto"`` uses per-chain colors for a single
    skeleton and flat per-skeleton palette colors for multi-skeleton
    comparisons (the GT-vs-generated convention); ``"chains"`` forces
    chain colors everywhere; ``"skeleton"`` forces the flat palette;
    ``"single"`` draws one skeleton in ``bone_color`` (multi-skeleton
    still uses the palette — the pre-0.9.0 behavior). The vedo
    backends draw a skeleton's label, and the vedo viewer its root
    trail, in that flat color; under chain colors, which give a
    skeleton no single color, in the spine color rather than the
    color of whichever bone the file lists first.

    ``axes`` and ``floor`` are independent, so ``Style("paper",
    axes="full")`` keeps the ground plane — and on the matplotlib
    backend that plane paints over the axis panes and grid lines,
    because any floor forces manual draw order (mplot3d's computed
    z-order would wash the skeleton out under the semi-transparent
    plane). For clean coordinate-inspection axes use ``"debug"``, or
    add ``floor=None``.
    """

    bone_width: float
    bone_color: tuple[float, float, float]
    color_mode: str
    chain_colors: dict[str, str]
    joint_markers: bool
    joint_size: float
    joint_color: str
    floor: str | None
    floor_alpha: float
    background: str
    axes: str
    projection: str
    dpi: int | None
    supersample: int
    shadow: bool
    ghost_spacing: float

    def __init__(self, preset: str = "paper", **overrides: object) -> None:
        if preset not in _STYLE_PRESETS:
            raise ValueError(
                f"Unknown style preset {preset!r}. "
                f"Choose from: {sorted(_STYLE_PRESETS)}")
        self._assign_fields(dict(_STYLE_PRESETS[preset]), overrides)

    def _assign_fields(
        self,
        fields: dict[str, object],
        overrides: dict[str, object],
    ) -> None:
        """Shared construction path for __init__ and replace: unknown-
        field check, defensive copies, assignment, validation."""
        unknown = set(overrides) - set(fields)
        if unknown:
            raise TypeError(
                f"Unknown Style field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(fields)}")
        fields.update(overrides)
        for name, value in fields.items():
            # Copy mutable field values (chain_colors) so no instance
            # aliases the module-level preset dicts — mutating one
            # Style must never restyle every other figure.
            if isinstance(value, dict):
                value = dict(value)
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {sorted(_VALID_COLOR_MODES)}, "
                f"got {self.color_mode!r}")
        if self.floor not in _VALID_FLOORS:
            raise ValueError(
                f"floor must be one of "
                f"{sorted(f for f in _VALID_FLOORS if f)} or None, "
                f"got {self.floor!r}")
        if self.axes not in _VALID_AXES:
            raise ValueError(
                f"axes must be one of {sorted(_VALID_AXES)}, "
                f"got {self.axes!r}")
        if self.projection not in _VALID_PROJECTIONS:
            raise ValueError(
                f"projection must be one of {sorted(_VALID_PROJECTIONS)}, "
                f"got {self.projection!r}")
        if not self.bone_width > 0:
            raise ValueError(
                f"bone_width must be positive, got {self.bone_width}")
        if not (isinstance(self.supersample, int) and self.supersample >= 1):
            raise ValueError(
                f"supersample must be an integer >= 1, "
                f"got {self.supersample!r}")
        if not self.ghost_spacing > 0:
            raise ValueError(
                f"ghost_spacing must be positive (seconds), "
                f"got {self.ghost_spacing!r}")

    def replace(self, **overrides: object) -> Style:
        """A new Style with the given fields changed."""
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)}
        new = object.__new__(Style)
        new._assign_fields(fields, overrides)
        return new


def resolve_style(style: Style | str) -> Style:
    """Accept a preset name or a Style instance; return a Style."""
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        return Style(style)
    raise TypeError(
        f"style must be a Style or a preset name string, "
        f"got {type(style).__name__}")


def effective_color_mode(style: Style, n_skeletons: int) -> str:
    """Resolve ``"auto"``/``"single"`` to the concrete mode for n panels.

    The multi-skeleton auto-switch rule: comparisons get flat
    per-skeleton palette colors unless chains are forced explicitly.
    """
    if style.color_mode == "auto":
        return "chains" if n_skeletons == 1 else "skeleton"
    if style.color_mode == "single":
        return "single" if n_skeletons == 1 else "skeleton"
    return style.color_mode


def spine_color(style: Style) -> str:
    """The style's spine color: the color of every bone no limb chain
    claims, and of the root joint, under chain colors.

    A ``chain_colors`` without a ``"spine"`` entry falls back to the
    paper preset's.
    """
    return style.chain_colors.get("spine", CHAIN_COLORS["spine"])


def skeleton_color(
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> tuple[float, float, float] | str:
    """The one color that stands for a whole skeleton, in
    matplotlib-friendly form.

    Under the flat color modes it is the color all its bones share:
    the view's palette color (``"skeleton"``) or ``style.bone_color``
    (``"single"``). Under ``"chains"`` no bone color stands for the
    whole skeleton, so it is :func:`spine_color`, the color the root
    joint takes, rather than the first bone's color, which would
    depend on the order the file lists the root's children in.
    """
    mode = effective_color_mode(style, n_skeletons)
    if mode == "skeleton":
        return PALETTE_MPL[view_index % len(PALETTE_MPL)]
    if mode == "single":
        return style.bone_color
    return spine_color(style)


def bone_colors_for_view(
    view: SkeletonView,
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> list:
    """Per-bone colors for one view, in matplotlib-friendly form.

    Each entry is a hex string or an RGB float tuple, parallel to
    ``view.bones``. The OpenCV backend converts these at its border.
    Under the flat color modes every bone takes :func:`skeleton_color`.
    """
    if effective_color_mode(style, n_skeletons) != "chains":
        flat = skeleton_color(style, view_index, n_skeletons)
        return [flat] * len(view.bones)
    # A skeleton with no L/R pairs is all-"spine" in view.bone_chains,
    # i.e. a single dark color (the documented fallback).
    spine = spine_color(style)
    return [style.chain_colors.get(chain_name, spine)
            for chain_name in view.bone_chains]


# ---------------------------------------------------------------------------
# Bone and joint sizes in pixels (OpenCV, the vedo viewer's fast mode)
# ---------------------------------------------------------------------------

# A joint marker's disc is this many pixels wider in radius than the
# bones are wide.
JOINT_DISC_MARGIN_PX = 2


def bone_width_px(bone_width: float, px_scale: float = 1.0) -> int:
    """A style's *bone_width* in whole pixels: ``bone_width`` pixels at
    the 1080p anchor (*px_scale* 1), scaled by *px_scale*, rounded half
    up as the OpenCV backend has always rounded it (2.5 draws 3 pixels,
    where Python's ``round``, half to even, gives 2) and at least 1."""
    return max(1, int(bone_width * px_scale + 0.5))


# ---------------------------------------------------------------------------
# Ghost trails and trajectory traces (shared backend conventions)
# ---------------------------------------------------------------------------

# The trace and ghost look must be identical across backends: one
# render() call, different sinks. These are the single definitions.
TRACE_COLOR = "#7A8090"
TRACE_BLEND = 0.9          # blended toward the background at this weight
GHOST_WIDTH_FACTOR = 0.75  # ghosts draw thinner than the live skeleton


def ghost_schedule(
    style: Style,
    frame_time: float,
    n_ghosts: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    """Frame lag and fade weights for a ghost trail.

    Ghost slot ``j`` trails the live pose by ``(j+1) * lag`` frames;
    weights fade from 0.32 (nearest, darkest) to 0.15 (oldest).
    ``frame_time`` must be positive: the router refuses ghosts on a
    clip whose rate is unset.
    """
    lag = max(1, round(style.ghost_spacing / frame_time))
    weights = np.linspace(0.32, 0.15, n_ghosts)
    return lag, weights
