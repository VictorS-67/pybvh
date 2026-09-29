"""Matplotlib visualization backend.

Provides static frame plots, animated renders (to file), interactive
playback via plt.show(), and 2D trajectory plots.
"""
from __future__ import annotations

import dataclasses
import inspect
import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from typing import Any, TYPE_CHECKING

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._common import (
    GHOST_WIDTH_FACTOR,
    PALETTE_MPL,
    Style,
    TRACE_BLEND,
    TRACE_COLOR,
    bone_colors_for_view,
    floor_trace_points,
    framing_bounds,
    ghost_schedule,
)
from ._scene import Scene, SkeletonView, UP_AXIS_INDEX
from ._colors import floor_palette

# mplot3d sizes its default margins for a cube free to rotate to any
# angle; span-fitted boxes are much tighter, so they can be zoomed in
# without the drawing ever reaching the axes edge.
BOX_ZOOM = 1.25

# matplotlib 3.10 made add_collection3d autoscale the axes to every collection
# it adds (autolim=True), by concatenating the collection's segments — which
# raises on an empty one, and ghost trails start empty (at frame 0 there is no
# history to show). bvhplot sets every axis limit itself, so nothing it adds
# should autoscale; matplotlib before 3.10 has no autolim and never did.
_NO_AUTOSCALE = (
    {"autolim": False}
    if "autolim" in inspect.signature(Axes3D.add_collection3d).parameters
    else {})


def _add_collection(ax: matplotlib.axes.Axes, collection: Any) -> None:
    """Add a 3D collection without letting it rescale the axes."""
    ax.add_collection3d(collection, **_NO_AUTOSCALE)

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes


# ---------------------------------------------------------------------------
# Style application helpers
# ---------------------------------------------------------------------------

def _manual_zorder(style: Style) -> bool:
    """Whether this style takes manual control of the 3D draw order.

    True whenever a floor exists or the axes are hidden — the same
    condition under which ``_apply_axes_style`` disables mplot3d's
    computed z-order and the bone collections must depth-sort their own
    segments (:class:`_DepthSortedLine3DCollection`).
    """
    return style.axes == "off" or style.floor is not None


class _DepthSortedLine3DCollection(Line3DCollection):
    """A bone collection that draws its segments far-to-near.

    mplot3d depth-sorts *artists* against each other (computed z-order)
    but never the segments inside one collection, so a fixed segment
    order lets a far-side limb paint over a near-side one. This
    subclass re-sorts segments (and their per-segment colors) by
    projected depth at draw time, which keeps occlusion correct for any
    camera — including per-frame follow/turntable azimuths, where the
    right order changes during the animation.

    Only used when the style controls draw order manually
    (:func:`_manual_zorder`); the debug preset keeps the plain
    fixed-order collection for pixel parity with pre-0.9.0 output.
    """

    def __init__(self, segments, **kwargs: Any) -> None:
        super().__init__(segments, **kwargs)
        self._base_edgecolors = np.array(self.get_edgecolor())

    def set_segments(self, segments) -> None:
        self._segments_sortable = np.asarray(segments, dtype=float)
        super().set_segments(segments)

    def do_3d_projection(self) -> float:
        segs = self._segments_sortable
        if len(segs) == 0:
            return super().do_3d_projection()
        pts = segs.reshape(-1, 3)
        tx, ty, tz = proj3d.proj_transform(
            pts[:, 0], pts[:, 1], pts[:, 2], self.axes.M)
        depth = np.asarray(tz).reshape(len(segs), 2).mean(axis=1)
        # In mpl's projected space larger z is farther from the viewer
        # (Axes3D draws artists in decreasing do_3d_projection order),
        # so far-to-near means descending depth.
        order = np.argsort(depth)[::-1]
        segs_2d = np.stack(
            [np.asarray(tx).reshape(len(segs), 2),
             np.asarray(ty).reshape(len(segs), 2)], axis=-1)
        LineCollection.set_segments(self, list(segs_2d[order]))
        if len(self._base_edgecolors) == len(segs):
            LineCollection.set_color(self, self._base_edgecolors[order])
        return float(depth.min())


def _make_bone_collection(
    segments,
    style: Style,
    **kwargs: Any,
) -> Line3DCollection:
    """The right bone-collection class for the style: depth-sorted under
    manual z-order, the plain fixed-order collection otherwise (debug
    pixel parity)."""
    cls = (_DepthSortedLine3DCollection if _manual_zorder(style)
           else Line3DCollection)
    return cls(segments, **kwargs)


def _apply_axes_style(
    ax: matplotlib.axes.Axes,
    style: Style,
) -> None:
    """Apply the axes/projection/background part of a Style to a 3D axes.

    ``axes="full"`` (the debug look) keeps default panes, ticks, and
    labels; the caller adds the x/y/z labels itself. ``axes="off"``
    hides everything and takes manual control of draw order so the
    floor can be painted behind the skeleton.
    """
    if style.projection == "ortho":
        ax.set_proj_type("ortho")  # type: ignore[attr-defined]
    if style.axes == "off":
        ax.set_axis_off()
    if _manual_zorder(style):
        # Manual draw order via explicit zorders (mplot3d ignores
        # zorder unless computed_zorder is off): floor 0.5, trace 0.8,
        # ghosts 1.5, bones 2 (the artist default), joints 3. Required
        # whenever a floor exists — under computed z-order the huge
        # semi-transparent plane's average depth beats the skeleton
        # and washes it out.
        ax.computed_zorder = False  # type: ignore[attr-defined]
    # Leave the default white patch untouched (pixel parity for the
    # debug preset); only non-white backgrounds need painting.
    if style.background != "white":
        ax.patch.set_facecolor(style.background)


def _floor_limits(
    view: SkeletonView,
    style: Style,
) -> tuple[npt.NDArray[np.float64], float]:
    """The view's cubic box, shifted so the floor sits just inside it.

    Without the shift a floor below the box bottom would be invisible;
    with it, the plane sits ~2% of the half-span above the bottom edge.
    """
    center = view.center.copy()
    half_span = view.half_span
    up = view.up_index
    sign = view.up_sign
    # The box edge visually below the skeleton is center - sign*half:
    # for a negative up axis the ground sits at the coordinate MAXIMUM.
    bottom = center[up] - sign * half_span
    target = view.below_floor(0.02 * half_span)
    delta = target - bottom
    if sign * delta < 0:
        center[up] += delta
    return center, half_span


def _draw_floor_mpl(
    ax: matplotlib.axes.Axes,
    view: SkeletonView,
    style: Style,
    bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
) -> None:
    """Draw the ground plane for one view (solid, grid, or checker).

    The plane is horizontal in the view's two ground axes at
    ``view.floor_height``. Without *bounds* it extends 1.8 x half_span
    around the box center (fills the frame at typical camera
    elevations); with ``bounds=(mins, maxs)`` it is clipped to that
    box — the sequence figure needs this, where a full-extent plane
    reads as a backdrop wall.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    up = view.up_index
    ground = [i for i in range(3) if i != up]
    y = view.below_floor(0.001 * view.half_span)
    palette = floor_palette(style)

    if bounds is None:
        center, half_span = _floor_limits(view, style)
        g0_lo = float(center[ground[0]]) - half_span * 1.8
        g0_hi = float(center[ground[0]]) + half_span * 1.8
        g1_lo = float(center[ground[1]]) - half_span * 1.8
        g1_hi = float(center[ground[1]]) + half_span * 1.8
    else:
        mins, maxs = bounds
        g0_lo, g0_hi = float(mins[ground[0]]), float(maxs[ground[0]])
        g1_lo, g1_hi = float(mins[ground[1]]), float(maxs[ground[1]])

    def quad(a_lo: float, a_hi: float, b_lo: float, b_hi: float):
        pts = []
        for a, b in ((a_lo, b_lo), (a_hi, b_lo), (a_hi, b_hi),
                     (a_lo, b_hi)):
            p = [0.0, 0.0, 0.0]
            p[ground[0]] = a
            p[ground[1]] = b
            p[up] = y
            pts.append(tuple(p))
        return pts

    if style.floor == "solid":
        _add_collection(ax, Poly3DCollection(
            [quad(g0_lo, g0_hi, g1_lo, g1_hi)],
            facecolors=palette["face"], edgecolors=palette["edge"],
            linewidths=0.5, alpha=style.floor_alpha, zorder=0.5))
    elif style.floor == "checker":
        n = 16
        s0 = (g0_hi - g0_lo) / n
        s1 = (g1_hi - g1_lo) / n
        squares = []
        colors = []
        for i in range(n):
            for j in range(n):
                squares.append(quad(g0_lo + i * s0, g0_lo + (i + 1) * s0,
                                    g1_lo + j * s1, g1_lo + (j + 1) * s1))
                colors.append(palette["checker"][(i + j) % 2])
        _add_collection(ax, Poly3DCollection(
            squares, facecolors=colors, alpha=style.floor_alpha,
            zorder=0.5))
    elif style.floor == "grid":
        # Square cells: divisions per axis follow its extent, so a wide
        # sequence floor doesn't get dense crosshatching on the short
        # axis.
        cell = max(g0_hi - g0_lo, g1_hi - g1_lo) / 20
        n0 = max(1, round((g0_hi - g0_lo) / cell))
        n1 = max(1, round((g1_hi - g1_lo) / cell))
        s0 = (g0_hi - g0_lo) / n0
        s1 = (g1_hi - g1_lo) / n1
        segs = []
        for i in range(n0 + 1):
            segs.append([_pt3(ground, up, g0_lo + i * s0, g1_lo, y),
                         _pt3(ground, up, g0_lo + i * s0, g1_hi, y)])
        for j in range(n1 + 1):
            segs.append([_pt3(ground, up, g0_lo, g1_lo + j * s1, y),
                         _pt3(ground, up, g0_hi, g1_lo + j * s1, y)])
        _add_collection(ax, Line3DCollection(
            segs, colors=palette["grid"], linewidths=0.7, alpha=0.8,
            zorder=0.5))


def _pt3(ground: list[int], up: int, a: float, b: float, y: float):
    p = [0.0, 0.0, 0.0]
    p[ground[0]] = a
    p[ground[1]] = b
    p[up] = y
    return tuple(p)


def _draw_joint_markers(
    ax: matplotlib.axes.Axes,
    frame_data: npt.NDArray[np.float64],
    style: Style,
) -> None:
    # zorder 3: joint dots sit on top of the bone lines (scatter's
    # default of 1 would put them underneath once computed_zorder is
    # off in paper mode).
    ax.scatter(
        frame_data[:, 0], frame_data[:, 1], frame_data[:, 2],
        s=style.joint_size, c=style.joint_color, depthshade=False,
        zorder=3)


def _fade_toward_background(color: object, weight: float, style: Style):
    """Blend a color toward the background: weight=1 is the full color,
    weight->0 disappears into the ground. This is the lighter-equals-past
    time encoding used by sequence figures and ghost trails."""
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))  # type: ignore[arg-type]
    bg = np.array(to_rgb(style.background))
    return tuple(bg + (rgb - bg) * weight)


def _draw_pose(
    ax: matplotlib.axes.Axes,
    pose: npt.NDArray[np.float64],
    view: SkeletonView,
    style: Style,
    colors: list,
    *,
    weight: float = 1.0,
    line_width: float | None = None,
    joints: bool = True,
) -> None:
    """Draw one skeleton pose, optionally faded toward the background."""
    if weight < 1.0:
        colors = [_fade_toward_background(c, weight, style) for c in colors]
    segments = pose[np.asarray(view.bones, dtype=int)]
    _add_collection(ax, _make_bone_collection(
        segments, style, colors=colors,
        linewidths=line_width if line_width is not None else style.bone_width))
    if joints and style.joint_markers:
        joint_color = (style.joint_color if weight >= 1.0 else
                       _fade_toward_background(style.joint_color, weight,
                                               style))
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2],
                   s=style.joint_size, c=[joint_color], depthshade=False,
                   zorder=3)


def _draw_floor_trace(
    ax: matplotlib.axes.Axes,
    view: SkeletonView,
    style: Style,
    start: int = 0,
    upto: int | None = None,
):
    """Dashed root-trajectory trace on the floor. Returns the artist."""
    path = floor_trace_points(view, start, upto)
    trace_color = _fade_toward_background(TRACE_COLOR, TRACE_BLEND, style)
    (line,) = ax.plot(
        path[:, 0], path[:, 1], path[:, 2],
        c=trace_color, lw=1.4, ls=(0, (4, 2)), zorder=0.8)
    return line


def sequence_mpl(
    scene: Scene,
    style: Style,
    sample_frames: npt.NDArray[np.intp],
    layout: str,
    *,
    trajectory: bool = True,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """The motion-paper sequence still: sampled poses, lighter = past.

    ``layout="offset"``: poses at their world positions (locomotion
    spreads them naturally), orthographic, equal-scale but NON-cubic
    bounds so wide travel fills the frame instead of shrinking into a
    cube. ``layout="overlay"``: poses superimposed with per-pose
    horizontal root-centering, cubic bounds, perspective.

    Poses draw back-to-front (oldest first) under manual z-order — the
    correct painter's-algorithm order for the fade encoding.
    """
    view = scene.views[0]
    n_samples = len(sample_frames)

    # Per-pose coordinates under the chosen layout — one fancy index,
    # no Python loop over frames.
    stack = view.coords[sample_frames]              # (S, N, 3)
    if layout == "overlay":
        shifts = stack[:, :1, :].copy()             # (S, 1, 3) root positions
        shifts[..., view.up_index] = 0.0
        stack = stack - shifts
    poses = list(stack)

    if ax is not None:
        fig = ax.get_figure()
        assert fig is not None
    else:
        if figsize is None:
            figsize = (13.0, 4.5) if layout == "offset" else (7.0, 6.5)
        fig = plt.figure(figsize=figsize, dpi=style.dpi)
        ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor(style.background)
    _apply_axes_style(ax, style)
    ax.computed_zorder = False  # type: ignore[attr-defined]
    if layout == "offset":
        # The offset figure is orthographic by design (D4): perspective
        # would shrink distant poses and break the left-to-right read.
        ax.set_proj_type("ortho")  # type: ignore[attr-defined]

    ax.view_init(  # type: ignore[attr-defined]
        elev=view.elevation, azim=view.azimuth, vertical_axis=view.up_axis)

    up = view.up_index
    sign = view.up_sign
    flat = stack.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    # Overlay layouts re-center heights, so the world floor does not
    # apply — use the poses' own lowest point (sign-aware: for a
    # negative up axis the ground is at the coordinate maximum).
    if layout == "offset":
        floor_y = view.floor_height
    else:
        floor_y = float(flat[:, up].min() if sign > 0
                        else flat[:, up].max())
    if style.floor is not None:
        if sign > 0:
            mins[up] = min(mins[up], floor_y)
        else:
            maxs[up] = max(maxs[up], floor_y)
    pad = 0.04 * float((maxs - mins).max())
    mins, maxs = mins - pad, maxs + pad
    spans = maxs - mins

    if layout == "offset":
        # Equal-scale, non-cubic: box aspect follows the data spans.
        # zoom compensates for mplot3d's generous default margins.
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(  # type: ignore[attr-defined]
            tuple(spans / spans.max()), zoom=BOX_ZOOM)
    else:
        center = (mins + maxs) / 2
        half = float(spans.max()) / 2
        _set_axis_limits(ax, center, half)
        ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]

    # Floor clipped to the box (a full-extent plane reads as a backdrop
    # wall in wide orthographic views), honoring the style's floor kind
    # via the one shared floor renderer. Overlay layouts re-center
    # heights, so the floor is re-anchored to the poses' lowest point.
    if style.floor is not None:
        floor_view = view if layout == "offset" else dataclasses.replace(
            view, floor_height=floor_y)
        _draw_floor_mpl(ax, floor_view, style, bounds=(mins, maxs))

    if trajectory and layout == "offset":
        # Trace only the sampled range — a frames= restriction must not
        # leak the whole clip's path into (and beyond) the figure.
        _draw_floor_trace(ax, view, style,
                          start=int(sample_frames[0]),
                          upto=int(sample_frames[-1]))

    colors = bone_colors_for_view(view, style, 0, 1)
    for k, pose in enumerate(poses):
        # lighter = past: the last sampled pose is fully saturated
        weight = 0.3 + 0.7 * (k / (n_samples - 1) if n_samples > 1 else 1.0)
        _draw_pose(ax, pose, view, style, colors, weight=weight,
                   joints=(k == n_samples - 1))

    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Static frame
# ---------------------------------------------------------------------------

def frame_mpl(
    scene: Scene,
    style: Style,
    *,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]:
    """Render one or more skeletons as static 3D subplots.

    Parameters
    ----------
    scene : Scene
        Prepared visualization. Only the first frame of each view's
        coords is plotted; each view supplies its own bounding box and
        camera so mixed-axis side-by-side comparisons render correctly.
    style : Style
        Visual styling (colors, floor, axes, background, projection).
    figsize : (float, float) or None
        Figure size.
    show : bool
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw on. If provided, no new figure is
        created. Only supported for a single skeleton (``n == 1``).

    Returns
    -------
    fig : Figure
    axs : Axes or list[Axes]
    """
    n = scene.num_skeletons

    if ax is not None:
        if n > 1:
            raise ValueError(
                "ax is only supported for single skeletons; pass a single "
                "Bvh object (not a list) when using ax."
            )
        if not hasattr(ax, 'get_zlim'):
            raise ValueError(
                "ax must be a 3D axes. Create one with "
                "plt.subplots(..., subplot_kw={'projection': '3d'}) or "
                "fig.add_subplot(..., projection='3d')."
            )
        fig = ax.get_figure()
        assert fig is not None
        axs_flat: list[matplotlib.axes.Axes] = [ax]
    else:
        if figsize is None:
            # 6.5" per subplot gives 3D axis labels room without ballooning.
            figsize = (6.5 * n, 6)

        fig, axs = plt.subplots(
            1, n, subplot_kw=dict(projection="3d"), figsize=figsize,
            dpi=style.dpi, squeeze=False)
        axs_flat = list(axs[0])

    fig.patch.set_facecolor(style.background)

    for i, (view, ax_i) in enumerate(zip(scene.views, axs_flat)):
        frame_data = view.coords[0]  # (N, 3) — first frame

        _apply_axes_style(ax_i, style)

        if style.floor is not None:
            _draw_floor_mpl(ax_i, view, style)
            center, half_span = _floor_limits(view, style)
        else:
            center, half_span = view.center, view.half_span

        colors = bone_colors_for_view(view, style, i, n)
        _draw_pose(ax_i, frame_data, view, style, colors)

        _set_axis_limits(ax_i, center, half_span)
        ax_i.view_init(  # type: ignore[attr-defined]
            elev=view.elevation, azim=view.azimuth,
            vertical_axis=view.up_axis)
        if style.axes == "full":
            ax_i.set_xlabel('x')
            ax_i.set_ylabel('y')
            ax_i.set_zlabel('z')  # type: ignore[attr-defined]
            # 3D axis labels and tick labels are clipped to the axes patch
            # by default. With certain camera angles (e.g. azim ≈ 160°) the
            # labels are positioned just outside the axes rectangle and
            # become invisible — the tick numbers may still appear, but the
            # 'x'/'y'/'z' label can disappear entirely. Disabling clipping
            # renders them into the surrounding figure margin instead.
            _disable_3d_label_clipping(ax_i)

        if view.label is not None:
            ax_i.set_title(view.label)

    if ax is None and n > 1:
        # tight_layout / constrained_layout under-estimate 3D axis tick-label
        # and pane extent, leaving the inside of neighboring subplots
        # overlapping AND the outside (z labels of the rightmost subplot,
        # y labels of the leftmost) clipped by the figure edge. Explicit
        # margins tuned for 3D solve both. Only needed when there are
        # neighbors; the single-subplot case fits comfortably in defaults.
        fig.subplots_adjust(
            left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.1,
        )
    if ax is None:
        # Jupyter's inline backend saves with bbox_inches='tight', which
        # crops to fig.get_tightbbox() — which by default doesn't include
        # 3D axis labels positioned outside the axes rectangle. Extend it.
        _extend_fig_tightbbox_with_3d_labels(fig, axs_flat)
    if show:
        plt.show()

    return (fig, axs_flat[0]) if n == 1 else (fig, axs_flat)


# ---------------------------------------------------------------------------
# Animated render (save to file)
# ---------------------------------------------------------------------------

def _setup_animated_panel(
    ax: matplotlib.axes.Axes,
    view: SkeletonView,
    style: Style,
    bones: npt.NDArray[np.intp],
    view_index: int,
    n_skeletons: int,
    rotating: bool = False,
) -> tuple[Line3DCollection, object | None]:
    """Shared per-panel setup for animated matplotlib output.

    Draws the static floor, the frame-0 bone collection, and the
    frame-0 joint scatter; applies limits, camera, and axes style.
    Returns the artists that get updated every frame.

    The panel is framed to the clip's own extents (see
    :func:`~._common.framing_bounds`) rather than to a cube, and the
    floor is clipped to that box; *rotating* squares off the ground
    axes for orbiting cameras. Stills keep the cubic box: they frame a
    single pose, whose extents are already tight.
    """
    _apply_axes_style(ax, style)

    lo, hi = framing_bounds(view, rotating=rotating)
    if style.floor is not None:
        # Full-extent floor, not one clipped to the framing box: the box
        # hugs the motion, so a clipped plane would end just past the
        # feet and read as a platform the character stands on rather
        # than as ground. (sequence() clips its floor for the opposite
        # reason — a wide ortho still turns a full plane into a wall.)
        _draw_floor_mpl(ax, view, style)

    colors = bone_colors_for_view(view, style, view_index, n_skeletons)
    collection = _make_bone_collection(
        view.coords[0][bones], style, colors=colors,
        linewidths=style.bone_width)
    _add_collection(ax, collection)

    joint_scatter = None
    if style.joint_markers:
        frame0 = view.coords[0]
        joint_scatter = ax.scatter(
            frame0[:, 0], frame0[:, 1], frame0[:, 2],
            s=style.joint_size, c=style.joint_color, depthshade=False)

    # view_init first: set_box_aspect stores the aspect rolled to whatever
    # vertical axis is current, and get_proj rolls it back the same way. Set
    # it before view_init picks a non-z vertical axis and the two rolls no
    # longer cancel — the aspect ends up paired with the wrong axis limits,
    # which scales one screen direction against the others and stretches the
    # skeleton. Invisible on a z-up rig, where the roll is the identity.
    ax.view_init(  # type: ignore[attr-defined]
        elev=view.elevation, azim=view.azimuth, vertical_axis=view.up_axis)
    _set_span_limits(ax, lo, hi)

    if style.axes == "full":
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  # type: ignore[attr-defined]
        # See frame_mpl: prevent rotated views from clipping their axis
        # labels against the axes patch.
        _disable_3d_label_clipping(ax)

    if view.label is not None:
        ax.set_title(view.label)

    return collection, joint_scatter


def _setup_render_extras(
    scene: Scene,
    style: Style,
    axs_flat: list[matplotlib.axes.Axes],
    ghost: int,
    trajectory: bool,
) -> tuple[list, list]:
    """Create the ghost collections and trace lines for animated output.

    Ghost slot ``j`` trails the live pose by ``(j+1) * ghost_spacing``
    seconds; nearer ghosts are darker (weights fade from 0.32 down to
    0.15 toward the oldest).
    """
    ghost_slots: list = []   # per skeleton: list of (collection, lag_frames)
    trace_lines: list = []   # per skeleton: Line3D or None
    n = scene.num_skeletons

    for i, (view, ax) in enumerate(zip(scene.views, axs_flat)):
        colors = bone_colors_for_view(view, style, i, n)
        lag, weights = ghost_schedule(style, view.frame_time, ghost)
        slots = []
        for j in range(ghost):
            faded = [_fade_toward_background(c, float(weights[j]), style)
                     for c in colors]
            # zorder 1.5: ghosts sit behind the live skeleton (bones at
            # the default 2) regardless of artist creation order.
            collection = _make_bone_collection(
                np.empty((0, 2, 3)), style, colors=faded,
                linewidths=style.bone_width * GHOST_WIDTH_FACTOR,
                zorder=1.5)
            _add_collection(ax, collection)
            slots.append((collection, (j + 1) * lag))
        ghost_slots.append(slots)
        trace_lines.append(
            _draw_floor_trace(ax, view, style, upto=0) if trajectory
            else None)

    return ghost_slots, trace_lines


def _wrap_update_with_extras(
    base_update,
    scene: Scene,
    bones_arrays,
    ghost_slots,
    trace_lines,
):
    """Extend an animation update fn with ghost and trace updates."""
    trace_paths = [floor_trace_points(v) for v in scene.views]
    empty = np.empty((0, 2, 3))

    def update(f: int):
        artists = base_update(f)
        for view, bones, slots, trace, path in zip(
                scene.views, bones_arrays, ghost_slots, trace_lines,
                trace_paths):
            for collection, lag in slots:
                gf = f - lag
                collection.set_segments(
                    view.coords[gf][bones] if gf >= 0 else empty)
                artists.append(collection)
            if trace is not None:
                upto = path[:f + 1]
                trace.set_data_3d(upto[:, 0], upto[:, 1], upto[:, 2])
                artists.append(trace)
        return artists

    return update


def render_mpl(
    scene: Scene,
    style: Style,
    filepath: Path,
    fps: float,
    *,
    follow: bool = False,
    turntable: bool = False,
    resolution: tuple[int, int] = (1920, 1080),
    ghost: int = 0,
    trajectory: bool = False,
) -> Path:
    """Render animation to a video/GIF/HTML file via matplotlib.

    Each subplot uses its own bounding box and camera orientation so
    that mixed-up-axis side-by-side comparisons render correctly.

    When ``follow`` is True, the camera orientation is recomputed every
    frame using each skeleton's current facing direction, so the view
    orbits with the character.

    Returns
    -------
    filepath : Path
        The actual output path (may differ from input if ffmpeg is missing
        and the format was changed to GIF).
    """
    filepath, writer_name = _resolve_writer(filepath)
    num_frames = scene.num_frames

    n = scene.num_skeletons
    # matplotlib sizes figures in inches; convert the requested pixel
    # resolution via the figure dpi so the saved frames honor it.
    dpi = float(plt.rcParams['figure.dpi'])
    w, h = resolution
    fig, axs = plt.subplots(
        1, n, subplot_kw=dict(projection="3d"),
        figsize=(w / dpi, h / dpi), squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    fig.patch.set_facecolor(style.background)

    coords_list = [v.coords for v in scene.views]
    bones_arrays = [np.asarray(v.bones, dtype=int) for v in scene.views]
    bone_collections: list[Line3DCollection] = []
    joint_scatters: list = []
    for i, (view, bones, ax) in enumerate(
            zip(scene.views, bones_arrays, axs_flat)):
        collection, joint_scatter = _setup_animated_panel(
            ax, view, style, bones, i, n, rotating=follow or turntable)
        bone_collections.append(collection)
        joint_scatters.append(joint_scatter)

    if n > 1:
        # Same 3D-aware spacing as frame_mpl — tight_layout under-estimates
        # 3D tick-label and pane extent, causing neighbours to overlap each
        # other's axes and the outer subplots to clip against the figure
        # edge. Outer margins handle the latter.
        fig.subplots_adjust(
            left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.1,
        )
    if style.axes == "full":
        # Same tight-bbox adjustment as frame_mpl, in case the animation
        # writer (jshtml/HTML) uses bbox_inches='tight' for its frames.
        _extend_fig_tightbbox_with_3d_labels(fig, axs_flat)

    if follow or turntable:
        from ._common import compute_follow_azimuths, turntable_azimuths

        if follow:
            per_frame_azimuths = [
                compute_follow_azimuths(v, v.azimuth)
                for v in scene.views]
        else:
            per_frame_azimuths = [
                turntable_azimuths(v.azimuth, num_frames)
                for v in scene.views]
        update = _make_orbit_update_fn(
            scene, bones_arrays, bone_collections, joint_scatters,
            axs_flat, per_frame_azimuths)
    else:
        update = _make_update_fn(
            coords_list, bones_arrays, bone_collections, joint_scatters)

    if ghost > 0 or trajectory:
        ghost_slots, trace_lines = _setup_render_extras(
            scene, style, axs_flat, ghost, trajectory)
        update = _wrap_update_with_extras(
            update, scene, bones_arrays, ghost_slots, trace_lines)

    interval = int(1000.0 / fps)
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval)

    if writer_name == "jshtml":
        html_content = anim.to_jshtml()
        with open(filepath, 'w') as f:
            f.write(html_content)
    else:
        # Pass fps explicitly: the writer's rate would otherwise be
        # derived from the integer-millisecond interval, quantizing e.g.
        # 24 fps (41.67 ms) to 1000/41 ≈ 24.4 fps.
        anim.save(filepath, writer=writer_name, fps=round(fps))

    plt.close(fig)
    return filepath


def _make_orbit_update_fn(
    scene: Scene,
    bones_arrays,
    bone_collections,
    joint_scatters,
    axs_flat,
    per_frame_azimuths: list[npt.NDArray[np.float64]],
):
    """Build an animation update fn that also recomputes view_init per frame.

    Used by both follow mode (azimuths from
    :func:`~._common.compute_follow_azimuths` — continuous rotation
    tracking around ``world_up``) and turntable mode (a constant-rate
    ramp from :func:`~._common.turntable_azimuths`).
    """
    base_update = _make_update_fn(
        [v.coords for v in scene.views], bones_arrays, bone_collections,
        joint_scatters)

    def update(frame):
        artists = base_update(frame)
        for az_per_frame, view, ax in zip(
                per_frame_azimuths, scene.views, axs_flat):
            ax.view_init(elev=view.elevation, azim=az_per_frame[frame],
                         vertical_axis=view.up_axis)
        return artists

    return update


# ---------------------------------------------------------------------------
# Interactive playback (matplotlib fallback)
# ---------------------------------------------------------------------------

def play_mpl(
    scene: Scene,
    style: Style,
    fps: float,
    *,
    in_notebook: bool = False,
) -> None:
    """Playback via matplotlib.

    Each subplot uses its own bounding box and camera orientation.

    In a notebook, renders the animation as inline HTML with playback
    controls (play/pause/scrub). In a script, opens an animated window
    via ``plt.show()``.
    """
    num_frames = scene.num_frames
    n = scene.num_skeletons

    fig, axs = plt.subplots(
        1, n, subplot_kw=dict(projection="3d"),
        figsize=(6 * n, 6), squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    fig.patch.set_facecolor(style.background)

    coords_list = [v.coords for v in scene.views]
    bones_arrays = [np.asarray(v.bones, dtype=int) for v in scene.views]
    bone_collections: list[Line3DCollection] = []
    joint_scatters: list = []
    for i, (view, bones, ax) in enumerate(
            zip(scene.views, bones_arrays, axs_flat)):
        collection, joint_scatter = _setup_animated_panel(
            ax, view, style, bones, i, n)
        bone_collections.append(collection)
        joint_scatters.append(joint_scatter)

    plt.tight_layout()

    update = _make_update_fn(
        coords_list, bones_arrays, bone_collections, joint_scatters)

    interval = int(1000.0 / fps)
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval)

    if in_notebook:
        # Render as inline HTML with play/pause/scrub controls
        from IPython.display import display, HTML  # type: ignore[import-untyped]
        display(HTML(anim.to_jshtml()))
        plt.close(fig)
    else:
        # Script: open animated window
        update(0)
        fig._pybvh_anim = anim  # type: ignore[attr-defined]
        plt.show()


# ---------------------------------------------------------------------------
# 2D trajectory
# ---------------------------------------------------------------------------

def trajectory_mpl(
    scene: Scene,
    style: Style,
    *,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    facing_arrows: bool = False,
    tight: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot 2D top-down trajectory of the root joint.

    Each skeleton's trajectory is projected onto its own horizontal
    plane (dropping its up axis). When skeletons share the same up
    axis the plot axes are labelled accordingly; when they differ,
    generic "horizontal" labels are used.

    Parameters
    ----------
    scene : Scene
        Prepared visualization; per-view labels become legend labels.
    figsize : (float, float) or None
        Figure size.
    show : bool
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing 2D axes to draw on. If provided, no new figure is
        created. Works with single or multiple skeletons.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    labels = scene.labels

    axis_names = ['x', 'y', 'z']

    if ax is not None:
        if hasattr(ax, 'get_zlim'):
            raise ValueError(
                "ax must be a 2D axes for trajectory(). "
                "Do not pass subplot_kw={'projection': '3d'} when creating it."
            )
        fig = ax.get_figure()
        assert fig is not None
    else:
        if figsize is None:
            figsize = _trajectory_figsize(
                _trajectory_data_aspect(scene.views))
        # constrained_layout handles external (bbox_to_anchor) legends
        # without clipping; tight_layout does not.
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    # trajectory() is a 2D data plot: axes, ticks, and grid carry the
    # information, so only Style's background applies here (the plot
    # keeps its axes regardless of style.axes).
    if style.background != "white":
        fig.patch.set_facecolor(style.background)
        ax.set_facecolor(style.background)

    # Track which horizontal axes are used across all skeletons
    all_horiz: set[tuple[int, int]] = set()
    # Track full-skeleton horizontal extents for the non-tight limit mode
    skeleton_h0_bounds: list[tuple[float, float]] = []
    skeleton_h1_bounds: list[tuple[float, float]] = []

    for i, view in enumerate(scene.views):
        coords = view.coords
        # Per-skeleton up axis: the view's, which honors any manual
        # world_up override on the Bvh it was built from.
        up_idx = view.up_index
        horiz = [j for j in range(3) if j != up_idx]
        all_horiz.add((horiz[0], horiz[1]))

        root_traj = coords[:, 0, :]  # (F, 3)
        h0 = root_traj[:, horiz[0]]
        h1 = root_traj[:, horiz[1]]

        # Record full-skeleton extent (all joints, all frames) so we can
        # set axis limits that include arm-span / leg-swing context
        # rather than zooming to just the root path.
        sk_h0 = coords[:, :, horiz[0]]
        sk_h1 = coords[:, :, horiz[1]]
        skeleton_h0_bounds.append((float(sk_h0.min()), float(sk_h0.max())))
        skeleton_h1_bounds.append((float(sk_h1.min()), float(sk_h1.max())))

        color = PALETTE_MPL[i % len(PALETTE_MPL)]
        label = labels[i] if labels and i < len(labels) else None

        ax.plot(h0, h1, c=color, lw=1.5, label=label)
        ax.scatter(h0[0], h1[0], c=[color], marker='o', s=60, zorder=5)
        ax.scatter(h0[-1], h1[-1], c=[color], marker='s', s=60, zorder=5)

        if facing_arrows and h0.shape[0] >= 2:
            # Overlay ~10 facing-direction arrows along this skeleton's path.
            # root_trajectory() returns [ground_a, ground_b, sin, cos] with
            # sin/cos being the facing direction in the ground-plane basis
            # (a, b = non-up axes in natural x,y,z order with the up axis
            # removed) — i.e. cos along axis a, sin along axis b.  Our local
            # horiz[] uses the same convention, so the trig components map
            # directly to (h0, h1) plot coordinates. view.root_heading is
            # that [sin, cos] pair, already aligned to the coords by
            # make_scene (truncated or padded with them).
            F_plot = h0.shape[0]
            if view.root_heading is None:
                raise ValueError(
                    "facing_arrows needs a Scene built from clip frames; "
                    "caller-supplied coordinates carry no root heading.")
            facing_sin = view.root_heading[:, 0]        # y-component (h1)
            facing_cos = view.root_heading[:, 1]        # x-component (h0)
            step = max(1, F_plot // 10)
            idx = np.arange(0, F_plot, step)
            # Arrow length: 8 % of the larger ground-plane span.  Using
            # the larger span (not each axis independently) keeps arrows
            # visually proportionate on highly asymmetric paths.
            span = max(float(np.ptp(h0)), float(np.ptp(h1)))
            if span == 0.0:  # stationary root — fall back to a small default
                span = 1.0
            arrow_len = span * 0.08
            ax.quiver(
                h0[idx], h1[idx],
                facing_cos[idx] * arrow_len,
                facing_sin[idx] * arrow_len,
                color=color,
                angles='xy', scale_units='xy', scale=1,
                width=0.005, zorder=4,
            )

    # Label axes — if all skeletons share the same horizontal pair, name them
    if len(all_horiz) == 1:
        h0_idx, h1_idx = all_horiz.pop()
        ax.set_xlabel(f'{axis_names[h0_idx]} axis')
        ax.set_ylabel(f'{axis_names[h1_idx]} axis')
    else:
        ax.set_xlabel('horizontal axis 1')
        ax.set_ylabel('horizontal axis 2')

    ax.set_aspect('equal')
    ax.set_title('Root Trajectory (top-down)')

    if not tight and skeleton_h0_bounds:
        # Union of per-skeleton horizontal extents, matching the bounding box
        # bvh.play() uses for the horizontal plane.  Adds a small visual pad.
        h0_min = min(b[0] for b in skeleton_h0_bounds)
        h0_max = max(b[1] for b in skeleton_h0_bounds)
        h1_min = min(b[0] for b in skeleton_h1_bounds)
        h1_max = max(b[1] for b in skeleton_h1_bounds)
        pad = 0.05 * max(h0_max - h0_min, h1_max - h1_min)
        if pad == 0.0:  # degenerate (all joints at one point) — avoid 0-span axes
            pad = 1.0
        ax.set_xlim(h0_min - pad, h0_max + pad)
        ax.set_ylim(h1_min - pad, h1_max + pad)

    # Build legend handles: skeleton labels (if any) + start/end marker key.
    # The start/end markers are shown in gray so the legend communicates
    # "shape → meaning" without being tied to any one skeleton's color.
    from matplotlib.lines import Line2D
    handles: list[Line2D] = []
    if labels:
        for i, label in enumerate(labels):
            if label is None:
                continue
            color = PALETTE_MPL[i % len(PALETTE_MPL)]
            handles.append(Line2D([0], [0], color=color, lw=2, label=label))
    handles.append(Line2D(
        [0], [0], marker='o', color='w', markerfacecolor='gray',
        markersize=9, label='start', linestyle=''))
    handles.append(Line2D(
        [0], [0], marker='s', color='w', markerfacecolor='gray',
        markersize=9, label='end', linestyle=''))
    # Legend is anchored outside the axes so it can never obstruct the
    # data — important for wide-flat trajectories where set_aspect('equal')
    # collapses the axes box into a thin strip.
    ax.legend(
        handles=handles, loc='center left',
        bbox_to_anchor=(1.02, 0.5), borderaxespad=0, framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_update_fn(
    coords_list: list[npt.NDArray[np.float64]],
    bones_arrays: list[npt.NDArray[np.intp]],
    bone_collections: list[Line3DCollection],
    joint_scatters: list | None = None,
) -> Any:
    """Create a FuncAnimation update function for bone rendering.

    One ``Line3DCollection`` per skeleton: a single ``set_segments``
    call replaces a Python loop over ~60 individual line artists.
    Joint scatters (when the style draws them) update in the same pass.
    """
    scatters = joint_scatters or [None] * len(coords_list)

    def update(f: int) -> list[Any]:
        for coords, bones, collection, scatter in zip(
                coords_list, bones_arrays, bone_collections, scatters):
            frame_data = coords[f]
            collection.set_segments(frame_data[bones])
            if scatter is not None:
                scatter._offsets3d = (  # noqa: SLF001 — mpl's supported idiom
                    frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        artists: list[Any] = list(bone_collections)
        artists.extend(s for s in scatters if s is not None)
        return artists
    return update


def _disable_3d_label_clipping(ax: matplotlib.axes.Axes) -> None:
    """Render axis labels and tick labels even when positioned outside the
    axes patch.

    Matplotlib's 3D axes position labels relative to the projected pane.
    Some camera angles place the label just outside the axes rectangle,
    where the default ``clip_on=True`` makes them invisible. Disabling
    clipping lets them render into the surrounding figure margin.
    """
    for axis_name in ('xaxis', 'yaxis', 'zaxis'):
        axis = getattr(ax, axis_name)
        axis.label.set_clip_on(False)
        for tick in axis.get_major_ticks():
            tick.label1.set_clip_on(False)


def _measuring_renderer(
    fig: matplotlib.figure.Figure,
    args: tuple,
    kwargs: dict,
) -> object | None:
    """A renderer to measure label extents with, or None if there is none.

    Prefer the one the caller passed: ``savefig`` hands its renderer to
    ``get_tightbbox``, and on a vector canvas (PDF, SVG, PS) it is the
    only one available — those canvases have no ``get_renderer`` at all,
    so reaching for one there raised ``AttributeError`` and took down
    every ``savefig(..., bbox_inches="tight")`` to a vector format, which
    is exactly how figures get saved for print.
    """
    renderer = kwargs.get('renderer', args[0] if args else None)
    if renderer is not None:
        return renderer
    get_renderer = getattr(fig.canvas, 'get_renderer', None)
    return get_renderer() if get_renderer is not None else None


def _extend_fig_tightbbox_with_3d_labels(
    fig: matplotlib.figure.Figure,
    axes_list: list[matplotlib.axes.Axes],
) -> None:
    """Patch ``fig.get_tightbbox`` so it includes 3D axis labels.

    Jupyter's inline backend saves figures with ``bbox_inches='tight'``,
    which crops to ``fig.get_tightbbox()``. For a 3D axes, that tight
    bbox does not include axis labels positioned *outside* the axes
    rectangle — even when those labels have ``clip_on=False`` and render
    correctly in interactive or plain-save contexts. The result is that
    inline notebook renders crop off labels that are visible elsewhere.
    This patch unions the axis label extents into the tight bbox so
    Jupyter's crop respects them.
    """
    from matplotlib.transforms import Bbox

    original_get_tightbbox = fig.get_tightbbox

    def patched(*args, **kwargs):
        bb = original_get_tightbbox(*args, **kwargs)
        renderer = _measuring_renderer(fig, args, kwargs)
        if renderer is None:
            return bb
        to_inches = fig.dpi_scale_trans.inverted()
        extras = []
        for ax in axes_list:
            if not hasattr(ax, 'zaxis'):
                continue
            for axis_name in ('xaxis', 'yaxis', 'zaxis'):
                axis = getattr(ax, axis_name)
                if not axis.label.get_visible():
                    continue
                ext_px = axis.label.get_window_extent(renderer)
                extras.append(ext_px.transformed(to_inches))
        if extras:
            bb = Bbox.union([bb] + extras)
        return bb

    fig.get_tightbbox = patched  # type: ignore[method-assign]


def _set_axis_limits(
    ax: matplotlib.axes.Axes,
    center: npt.NDArray[np.float64],
    half_span: float,
) -> None:
    """Set equal axis limits on a 3D axes from center and half_span."""
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)  # type: ignore[attr-defined]


def _set_span_limits(
    ax: matplotlib.axes.Axes,
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
) -> None:
    """Fit a 3D axes to a box, equal-scale but not cubic.

    The box aspect follows the data spans, so the drawing fills the
    canvas instead of padding the short axes out to the longest one;
    ``zoom`` compensates for mplot3d's generous default margins, which
    are sized for a rotating cube.
    """
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])  # type: ignore[attr-defined]
    spans = hi - lo
    ax.set_box_aspect(  # type: ignore[attr-defined]
        tuple(spans / spans.max()), zoom=BOX_ZOOM)


def _resolve_writer(filepath: Path) -> tuple[Path, str]:
    """Determine the matplotlib animation writer from file extension.

    Returns
    -------
    filepath : Path
        Possibly modified path (e.g. .mp4 → .gif if ffmpeg missing).
    writer : str
        Matplotlib writer name.
    """
    ext = filepath.suffix.lower()

    if ext in ('.mp4', '.mov', '.avi'):
        if animation.writers.is_available('ffmpeg'):
            return filepath, 'ffmpeg'
        # Fallback to GIF
        filepath = filepath.with_suffix('.gif')
        warnings.warn(
            f"FFmpeg not found — cannot save as {ext}. "
            f"Falling back to GIF: '{filepath}'. "
            f".webp and .html are also available.")
        return filepath, 'pillow'

    if ext in ('.gif', '.webp', '.apng'):
        return filepath, 'pillow'

    if ext == '.html':
        return filepath, 'jshtml'

    raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Trajectory layout helpers
# ---------------------------------------------------------------------------

def _trajectory_data_aspect(views: list[SkeletonView]) -> float:
    """Aspect ratio (dx / dy) of the combined trajectory data.

    Each skeleton is projected onto its own horizontal plane (dropping
    its own up axis). The aggregate dx and dy are computed from the
    union of all projected root paths. Returns ``1.0`` for degenerate
    (single-point) data.
    """
    h0_values: list[npt.NDArray[np.float64]] = []
    h1_values: list[npt.NDArray[np.float64]] = []
    for view in views:
        up_idx = view.up_index
        horiz = [j for j in range(3) if j != up_idx]
        root_traj = view.coords[:, 0, :]
        h0_values.append(root_traj[:, horiz[0]])
        h1_values.append(root_traj[:, horiz[1]])

    if not h0_values:
        return 1.0

    h0 = np.concatenate(h0_values)
    h1 = np.concatenate(h1_values)
    dx = float(np.ptp(h0))
    dy = float(np.ptp(h1))

    # Guard degenerate single-point or single-axis data so the ratio
    # remains finite and roughly 1:1 in that case.
    eps = max(1e-9, max(dx, dy, 1.0) * 1e-6)
    dx = max(dx, eps)
    dy = max(dy, eps)
    return dx / dy


def _trajectory_figsize(
    data_aspect: float,
    base: float = 5.5,
    legend_margin: float = 1.8,
    max_ratio: float = 2.5,
) -> tuple[float, float]:
    """Figure size matching trajectory data aspect, clamped for sanity.

    ``set_aspect('equal')`` (the right choice for spatial honesty in a
    top-down root path) means the axes box visual aspect equals the
    data aspect. On a fixed square figure, wide-flat or narrow-tall
    data collapses the axes box into an unreadable strip. Sizing the
    figure to track the data aspect keeps the plot area balanced.

    Parameters
    ----------
    data_aspect : float
        ``dx / dy`` of the combined trajectory data.
    base : float, optional
        Base dimension (inches) used for the shorter figure side.
    legend_margin : float, optional
        Extra horizontal inches reserved for the external legend.
    max_ratio : float, optional
        Cap on the figure width-to-height ratio. Extreme data aspects
        (e.g. 13:1) still produce a plot where the trajectory occupies
        a thin strip — that's honest — but the figure itself does not
        become absurdly wide.

    Returns
    -------
    (width, height) : tuple of float
        Figure size in inches, including legend margin.
    """
    capped = max(1.0 / max_ratio, min(max_ratio, data_aspect))

    if capped >= 1.0:
        width = base * capped
        height = base
    else:
        width = base
        height = base / capped

    return (width + legend_margin, height)
