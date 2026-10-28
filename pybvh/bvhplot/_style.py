"""The look of a figure: :class:`Style`, its presets, the palettes and
the ghost, trace and frame-counter conventions every backend shares.

Pure data, no plotting library imports but one: the Style checks its
color fields with matplotlib's color parser, imported where it checks
them. Turning colors into a backend's values, which needs that parser
too, lives in :mod:`._colors`.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._scene import Scene, SkeletonView


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# RGB is the canonical channel order everywhere in bvhplot; the OpenCV
# backend converts to BGR at its own border (its channel-order quirk
# stays its own concern).

# Per-skeleton comparison palette (multi-skeleton figures)
PALETTE_RGB = [
    (50, 120, 255),  # blue
    (220, 50, 50),  # red
    (50, 180, 50),  # green
    (50, 130, 200),  # teal
    (200, 100, 50),  # orange
    (200, 50, 200),  # magenta
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


# The joint size at which every backend that sizes joints itself (k3d's
# points, OpenCV's discs) draws them at its base size: the presets'.
ANCHOR_JOINT_SIZE = 9.0


def joint_size_scale(joint_size: float) -> float:
    """The factor a style's *joint_size* scales a joint's diameter by:
    1 at ``ANCHOR_JOINT_SIZE``.

    ``joint_size`` is an area, as matplotlib's marker size is (square
    points), so the diameter goes as its square root, as a matplotlib
    marker's does: four times the joint size draws joints twice as wide
    on every backend. The alternative, a diameter in proportion to the
    size as ``bone_width`` gives the bones, would make the same change
    twice as large on k3d and OpenCV as on matplotlib.
    """
    return float((joint_size / ANCHOR_JOINT_SIZE) ** 0.5)


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

# The fields with a fixed set of values, and those values. A backend
# declares which of these values it draws, where it declares any other
# field drawn whole (see DrawnStyle): OpenCV draws "ortho" exactly, and
# so honours a Style asking for it, without drawing "persp".
STYLE_VALUES: dict[str, frozenset[object]] = {
    "color_mode": frozenset(_VALID_COLOR_MODES),
    "joint_markers": frozenset({True, False}),
    "floor": frozenset(_VALID_FLOORS),
    "axes": frozenset(_VALID_AXES),
    "projection": frozenset(_VALID_PROJECTIONS),
}


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

    A Style keeps which fields its caller set: those passed to the
    constructor, and to every ``replace`` since. A field set at a value
    the backend drawing it does not draw is refused, as an option the
    backend does not read is: ``frame`` and a named ``render`` backend
    raise ``ValueError`` naming it, ``render(backend="auto")`` warns.
    A preset's own values are never refused, so ``Style("paper")``
    draws on every backend, each drawing what it can of it; ``dpi``,
    ``supersample`` and ``ghost_spacing`` are accepted by every
    backend, drawn or not. Which fields were set does not count for
    ``==`` or ``repr``: two Styles of the same values are the same
    look. The alternative, refusing a field whose value differs from
    its preset's, would let ``Style("paper", projection="persp")``
    pass unrefused on OpenCV, which draws orthographic.

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
    floor kind and ``floor_alpha``. Fields
    outside that subset (``axes``, ``projection``, ``joint_markers``,
    ...) do not alter the viewers. **Output** fields apply only where
    raster output is produced: ``dpi`` (matplotlib figures),
    ``supersample`` (OpenCV export), ``shadow`` (vedo offscreen
    renders).

    ``bone_color``, ``joint_color``, ``background`` and the values of
    ``chain_colors`` take any color matplotlib parses (a name, a hex
    string, an RGB tuple in [0, 1]), the parser every backend reads them
    with; one it cannot parse raises ``ValueError`` when the Style is
    made, by the constructor or by ``replace``. ``chain_colors`` may
    leave chains out: those are drawn in its ``"spine"`` color, or the
    paper preset's spine color when it has none.

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
    matplotlib's panel titles (``labels``) take the label colour on a
    dark background; on a light one they keep matplotlib's black rather
    than the label colour (``#444444`` on white), so the light pictures
    look as they always did.

    ``color_mode``: ``"auto"`` uses per-chain colors for a single
    skeleton and flat per-skeleton palette colors for multi-skeleton
    comparisons (the GT-vs-generated convention); ``"chains"`` forces
    chain colors everywhere; ``"skeleton"`` forces the flat palette;
    ``"single"`` draws one skeleton in ``bone_color`` (multi-skeleton
    still uses the palette — the pre-0.9.0 behavior). The vedo and
    OpenCV backends draw a skeleton's label, and the vedo viewer its
    root trail, in that flat color; under chain colors, which give a
    skeleton no single color, in the spine color rather than the
    color of whichever bone the file lists first.

    ``joint_size`` is matplotlib's marker size, an area in points
    squared, so four times the size draws joints twice as wide. OpenCV
    draws the joint discs through the scale that turns ``bone_width``
    into pixels: their radius grows with the square root of
    ``joint_size``, 4 pixels for the presets' 9.0 on a drawing surface
    1080 rows high, which it draws at ``supersample`` times the
    resolution and scales down.

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
    # The fields the caller set, by the constructor's overrides and every
    # replace since: the fields a backend that does not draw them refuses.
    # Not a field of the look: equality and repr leave it out.
    _set_fields: frozenset[str] = dataclasses.field(
        default=frozenset(), init=False, repr=False, compare=False
    )

    def __init__(self, preset: str = "paper", **overrides: object) -> None:
        if preset not in _STYLE_PRESETS:
            raise ValueError(
                f"Unknown style preset {preset!r}. Choose from: {sorted(_STYLE_PRESETS)}"
            )
        self._assign_fields(dict(_STYLE_PRESETS[preset]), overrides)

    def _assign_fields(
        self,
        fields: dict[str, object],
        overrides: dict[str, object],
        set_before: frozenset[str] = frozenset(),
    ) -> None:
        """Shared construction path for __init__ and replace: unknown-
        field check, defensive copies, assignment, validation, and the
        fields set: *set_before* and the overrides."""
        unknown = set(overrides) - set(fields)
        if unknown:
            raise TypeError(
                f"Unknown Style field(s): {sorted(unknown)}. Valid fields: {sorted(fields)}"
            )
        fields.update(overrides)
        for name, value in fields.items():
            # Copy mutable field values (chain_colors) so no instance
            # aliases the module-level preset dicts — mutating one
            # Style must never restyle every other figure.
            if isinstance(value, dict):
                value = dict(value)
            object.__setattr__(self, name, value)
        self._validate()
        object.__setattr__(self, "_set_fields", set_before | frozenset(overrides))

    def _validate(self) -> None:
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {sorted(_VALID_COLOR_MODES)}, got {self.color_mode!r}"
            )
        if self.floor not in _VALID_FLOORS:
            raise ValueError(
                f"floor must be one of "
                f"{sorted(f for f in _VALID_FLOORS if f)} or None, "
                f"got {self.floor!r}"
            )
        if self.axes not in _VALID_AXES:
            raise ValueError(f"axes must be one of {sorted(_VALID_AXES)}, got {self.axes!r}")
        if self.projection not in _VALID_PROJECTIONS:
            raise ValueError(
                f"projection must be one of {sorted(_VALID_PROJECTIONS)}, got {self.projection!r}"
            )
        if not self.bone_width > 0:
            raise ValueError(f"bone_width must be positive, got {self.bone_width}")
        if not (isinstance(self.supersample, int) and self.supersample >= 1):
            raise ValueError(f"supersample must be an integer >= 1, got {self.supersample!r}")
        if not self.ghost_spacing > 0:
            raise ValueError(
                f"ghost_spacing must be positive (seconds), got {self.ghost_spacing!r}"
            )
        if not (isinstance(self.floor_alpha, numbers.Real) and 0 <= self.floor_alpha <= 1):
            raise ValueError(f"floor_alpha must be a number in [0, 1], got {self.floor_alpha!r}")
        if not (isinstance(self.joint_size, numbers.Real) and self.joint_size > 0):
            raise ValueError(f"joint_size must be a positive number, got {self.joint_size!r}")
        if not (self.dpi is None or (isinstance(self.dpi, numbers.Real) and self.dpi > 0)):
            raise ValueError(f"dpi must be a positive number or None, got {self.dpi!r}")
        for name in ("joint_markers", "shadow"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be True or False, got {value!r}")
        if not isinstance(self.chain_colors, dict):
            raise ValueError(
                f"chain_colors must be a dict from chain name to color, got {self.chain_colors!r}"
            )
        colors = {
            "bone_color": self.bone_color,
            "joint_color": self.joint_color,
            "background": self.background,
            **{f"chain_colors[{chain!r}]": c for chain, c in self.chain_colors.items()},
        }
        # Every backend turns a color into its own format through
        # matplotlib's parser, so a value it parses is one each can draw.
        # Imported here, not at the top: importing bvhplot loads no
        # matplotlib.
        from matplotlib.colors import is_color_like

        for name, value in colors.items():
            if not is_color_like(value):
                raise ValueError(
                    f"{name} must be a matplotlib color, such as 'white', '#16181D' "
                    f"or (0.1, 0.2, 0.8), got {value!r}"
                )

    def replace(self, **overrides: object) -> Style:
        """A new Style with the given fields changed. The fields changed
        join those the caller set before (see :class:`Style`)."""
        # The fields of the look, the ones compared: not _set_fields.
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.compare}
        new = object.__new__(Style)
        new._assign_fields(fields, overrides, self._set_fields)
        return new


def fields_set(style: Style) -> dict[str, object]:
    """The fields of *style* its caller set, by the constructor's
    overrides or by :meth:`Style.replace`, with their values: none for a
    preset alone."""
    return {name: getattr(style, name) for name in sorted(style._set_fields)}


@dataclass(frozen=True, init=False)
class DrawnStyle:
    """The Style values one backend draws for one call, declared beside
    its entry function: each field drawn whole, and of each field of
    :data:`STYLE_VALUES` it does not draw whole, the values it draws.

    A value drawn is drawn as the Style says: OpenCV draws orthographic
    whatever the projection asked, so it declares ``projection`` with
    ``"ortho"`` only, and the vedo renderer, which sizes and colours
    every joint by its bones, declares none of the joint fields. A
    field absent draws none of its values.
    """

    fields: frozenset[str]
    values: dict[str, frozenset[object]]

    def __init__(self, fields: Collection[str], **values: Collection[object]) -> None:
        object.__setattr__(self, "fields", frozenset(fields))
        object.__setattr__(
            self, "values", {name: frozenset(drawn) for name, drawn in values.items()}
        )

    def draws(self, field: str, value: object) -> bool:
        """Whether *field* at *value* is drawn as the Style asks."""
        return field in self.fields or value in self.values.get(field, frozenset())


def resolve_style(style: Style | str) -> Style:
    """Accept a preset name or a Style instance; return a Style."""
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        return Style(style)
    raise TypeError(f"style must be a Style or a preset name string, got {type(style).__name__}")


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
    return [style.chain_colors.get(chain_name, spine) for chain_name in view.bone_chains]


# ---------------------------------------------------------------------------
# Bone and joint sizes in pixels (OpenCV, the vedo viewer's fast mode)
# ---------------------------------------------------------------------------

# A joint disc of ANCHOR_JOINT_SIZE has this radius in pixels at the
# 1080p anchor. At render's default 1920 x 1080, drawn twice the size,
# that is 8 pixels against bones 6 wide: a joint stands 2 pixels out of
# its bones on each side.
ANCHOR_JOINT_RADIUS_PX = 4.0


def bone_width_px(bone_width: float, px_scale: float = 1.0) -> int:
    """A style's *bone_width* in whole pixels: ``bone_width`` pixels at
    the 1080p anchor (*px_scale* 1), scaled by *px_scale*, rounded half
    up as the OpenCV backend has always rounded it (2.5 draws 3 pixels,
    where Python's ``round``, half to even, gives 2) and at least 1."""
    return max(1, int(bone_width * px_scale + 0.5))


def joint_radius_px(joint_size: float, px_scale: float = 1.0) -> int:
    """A style's *joint_size* as a joint disc's radius in whole pixels:
    ``ANCHOR_JOINT_RADIUS_PX`` at the 1080p anchor (*px_scale* 1) scaled
    by :func:`joint_size_scale`, which grows with the square root of
    the size, and by *px_scale*, as :func:`bone_width_px` scales the
    bones; rounded half up and at least 1."""
    radius = ANCHOR_JOINT_RADIUS_PX * joint_size_scale(joint_size)
    return max(1, int(radius * px_scale + 0.5))


# ---------------------------------------------------------------------------
# Ghost trails and trajectory traces (shared backend conventions)
# ---------------------------------------------------------------------------

# The trace and ghost look must be identical across backends: one
# render() call, different sinks. These are the single definitions.
TRACE_COLOR = "#7A8090"
TRACE_BLEND = 0.9  # blended toward the background at this weight
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
    weights = np.linspace(0.32, 0.15, n_ghosts, dtype=np.float64)
    return lag, weights


# ---------------------------------------------------------------------------
# Frame counter (shared backend convention)
# ---------------------------------------------------------------------------

# How every backend that draws the counter stamps it, in pixels of the
# written frame: gray text, its baseline this far above the bottom edge
# and its left edge where counter_left_px puts it.
COUNTER_RGB = (150, 150, 150)
COUNTER_BASELINE_PX = 15


def counter_left_px(width: int) -> int:
    """The left edge of the counter on a frame *width* pixels wide:
    200 pixels in from the right edge, or 5 in from the left on a frame
    narrower than 205."""
    return max(5, width - 200)


def frame_counter_text(scene: Scene, frame: int) -> str:
    """The counter stamped on *frame*: ``Frame f/F``, ``f`` the frame
    of the clip it shows and ``F`` the clip's last frame, so a looped
    Scene counts from 0 again on each pass
    (:meth:`~._scene.Scene.pass_start`)."""
    return f"Frame {frame - scene.pass_start(frame)}/{scene.pass_length - 1}"
