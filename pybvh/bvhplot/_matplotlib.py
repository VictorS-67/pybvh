"""Matplotlib visualization backend.

Provides static frame plots, animated renders (to file), interactive
playback via plt.show(), and 2D trajectory plots.
"""
from __future__ import annotations

import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from typing import Any, TYPE_CHECKING

from ._common import PALETTE_MPL

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from ..bvh import Bvh


# ---------------------------------------------------------------------------
# Static frame
# ---------------------------------------------------------------------------

def frame_mpl(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    labels: list[str] | None,
    figsize: tuple[float, float] | None,
    show: bool,
    skeleton_lines_list: list[list[tuple[int, int]]],
    centers: list[npt.NDArray[np.float64]],
    half_spans: list[float],
    azimuths: list[float],
    elevations: list[float],
    up_axes: list[str],
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]:
    """Render one or more skeletons as static 3D subplots.

    Parameters
    ----------
    bvh_list : list[Bvh]
        Skeleton objects.
    coords_list : list[ndarray]
        Spatial coordinates, each ``(F, N, 3)``. Only the first frame
        of each is plotted.
    labels : list[str] or None
        Subplot titles.
    figsize : (float, float) or None
        Figure size.
    show : bool
        Whether to call ``plt.show()``.
    skeleton_lines_list : list
        Precomputed bone index pairs per skeleton.
    centers, half_spans : list[ndarray], list[float]
        Per-skeleton bounding boxes. Each subplot uses its own.
    azimuths, elevations : list[float]
        Per-skeleton camera angles in degrees.
    up_axes : list[str]
        Per-skeleton vertical axis, each ``'x'``, ``'y'``, or ``'z'``.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw on. If provided, no new figure is
        created. Only supported for a single skeleton (``n == 1``).

    Returns
    -------
    fig : Figure
    axs : Axes or list[Axes]
    """
    n = len(bvh_list)

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
            squeeze=False)
        axs_flat = list(axs[0])

    for i, (coords, bones, ax_i) in enumerate(
            zip(coords_list, skeleton_lines_list, axs_flat)):
        frame_data = coords[0]  # (N, 3) — first frame
        color = PALETTE_MPL[i % len(PALETTE_MPL)] if n > 1 else (0.1, 0.2, 0.8)

        _draw_bones(ax_i, frame_data, bones, color)
        _set_axis_limits(ax_i, centers[i], half_spans[i])
        ax_i.view_init(  # type: ignore[attr-defined]
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])
        ax_i.set_xlabel('x')
        ax_i.set_ylabel('y')
        ax_i.set_zlabel('z')  # type: ignore[attr-defined]
        # 3D axis labels and tick labels are clipped to the axes patch by
        # default. With certain camera angles (e.g. azim ≈ 160°) the labels
        # are positioned just outside the axes rectangle and become invisible
        # — the tick numbers may still appear, but the 'x'/'y'/'z' label can
        # disappear entirely. Disabling clipping renders them into the
        # surrounding figure margin instead.
        _disable_3d_label_clipping(ax_i)

        if labels and i < len(labels):
            ax_i.set_title(labels[i])

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

def render_mpl(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    filepath: Path,
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    centers: list[npt.NDArray[np.float64]],
    half_spans: list[float],
    azimuths: list[float],
    elevations: list[float],
    up_axes: list[str],
    show_axis: bool,
    follow: bool = False,
    camera: str | tuple[float, float] = "front",
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
    num_frames = coords_list[0].shape[0]

    n = len(bvh_list)
    fig, axs = plt.subplots(
        1, n, subplot_kw=dict(projection="3d"),
        figsize=(6.5 * n, 6), squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    all_line_artists: list[list[Any]] = []
    for i, (bones, ax) in enumerate(zip(skeleton_lines_list, axs_flat)):
        color = PALETTE_MPL[i % len(PALETTE_MPL)] if n > 1 else (0.1, 0.2, 0.8)
        line_artists = [ax.plot([], [], [], c=color, lw=2.5)[0] for _ in bones]
        all_line_artists.append(line_artists)

        _set_axis_limits(ax, centers[i], half_spans[i])
        ax.view_init(  # type: ignore[attr-defined]
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])

        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')  # type: ignore[attr-defined]
            # See frame_mpl: prevent rotated views from clipping their axis
            # labels against the axes patch.
            _disable_3d_label_clipping(ax)

        if labels and i < len(labels):
            ax.set_title(labels[i])

    if n > 1:
        # Same 3D-aware spacing as frame_mpl — tight_layout under-estimates
        # 3D tick-label and pane extent, causing neighbours to overlap each
        # other's axes and the outer subplots to clip against the figure
        # edge. Outer margins handle the latter.
        fig.subplots_adjust(
            left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.1,
        )
    if show_axis:
        # Same tight-bbox adjustment as frame_mpl, in case the animation
        # writer (jshtml/HTML) uses bbox_inches='tight' for its frames.
        _extend_fig_tightbbox_with_3d_labels(fig, axs_flat)

    if follow:
        update = _make_follow_update_fn(
            bvh_list, coords_list, skeleton_lines_list,
            all_line_artists, axs_flat, camera)
    else:
        update = _make_update_fn(
            coords_list, skeleton_lines_list, all_line_artists)

    interval = int(1000.0 / fps)
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval)

    if writer_name == "jshtml":
        html_content = anim.to_jshtml()
        with open(filepath, 'w') as f:
            f.write(html_content)
    else:
        anim.save(filepath, writer=writer_name)

    plt.close(fig)
    return filepath


def _make_follow_update_fn(
    bvh_list,
    coords_list,
    skeleton_lines_list,
    all_line_artists,
    axs_flat,
    camera,
):
    """Build an animation update fn that also recomputes view_init per frame.

    Uses CONTINUOUS rotation tracking: for each skeleton, the base camera
    angle is computed once from frame 0, then on every frame we add the
    signed rotation delta between frame 0's lateral axis and the current
    frame's lateral axis (measured around ``world_up``). This gives a
    smooth orbit that tracks the character's actual rotation — not a
    snap-every-90°-to-a-signed-axis.
    """
    from ._common import get_camera_angles
    from ..tools import (
        _axis_to_vector,
        _signed_rotation_delta_around_axis,
        _world_leftward_unit_at_frame,
    )

    base_update = _make_update_fn(
        coords_list, skeleton_lines_list, all_line_artists)

    # Precompute per-skeleton base camera and frame-0 lateral unit vectors.
    base_angles: list[tuple[float, float, str]] = []
    base_lefts: list[np.ndarray | None] = []
    up_vecs: list[np.ndarray] = []
    for bvh_obj, coords in zip(bvh_list, coords_list):
        az, el, up = get_camera_angles(bvh_obj, coords[0], camera)
        base_angles.append((az, el, up))
        base_lefts.append(
            _world_leftward_unit_at_frame(bvh_obj, coords[0], bvh_obj.world_up))
        up_vecs.append(_axis_to_vector(bvh_obj.world_up))

    def update(frame):
        artists = base_update(frame)
        for i, (bvh_obj, coords, ax) in enumerate(
                zip(bvh_list, coords_list, axs_flat)):
            az0, el0, up0 = base_angles[i]
            left_0 = base_lefts[i]
            if left_0 is None:
                ax.view_init(elev=el0, azim=az0, vertical_axis=up0)
                continue
            left_f = _world_leftward_unit_at_frame(
                bvh_obj, coords[frame], bvh_obj.world_up)
            if left_f is None:
                ax.view_init(elev=el0, azim=az0, vertical_axis=up0)
                continue
            delta = _signed_rotation_delta_around_axis(
                left_0, left_f, up_vecs[i])
            ax.view_init(elev=el0, azim=az0 + delta, vertical_axis=up0)
        return artists

    return update


# ---------------------------------------------------------------------------
# Interactive playback (matplotlib fallback)
# ---------------------------------------------------------------------------

def play_mpl(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    centers: list[npt.NDArray[np.float64]],
    half_spans: list[float],
    azimuths: list[float],
    elevations: list[float],
    up_axes: list[str],
    in_notebook: bool = False,
) -> None:
    """Playback via matplotlib.

    Each subplot uses its own bounding box and camera orientation.

    In a notebook, renders the animation as inline HTML with playback
    controls (play/pause/scrub). In a script, opens an animated window
    via ``plt.show()``.
    """
    num_frames = coords_list[0].shape[0]
    n = len(bvh_list)

    fig, axs = plt.subplots(
        1, n, subplot_kw=dict(projection="3d"),
        figsize=(6 * n, 6), squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    all_line_artists: list[list[Any]] = []
    for i, (bones, ax) in enumerate(zip(skeleton_lines_list, axs_flat)):
        color = PALETTE_MPL[i % len(PALETTE_MPL)] if n > 1 else (0.1, 0.2, 0.8)
        line_artists = [ax.plot([], [], [], c=color, lw=2.5)[0] for _ in bones]
        all_line_artists.append(line_artists)

        _set_axis_limits(ax, centers[i], half_spans[i])
        ax.view_init(  # type: ignore[attr-defined]
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  # type: ignore[attr-defined]

        if labels and i < len(labels):
            ax.set_title(labels[i])

    plt.tight_layout()

    update = _make_update_fn(coords_list, skeleton_lines_list, all_line_artists)

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
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    labels: list[str] | None,
    figsize: tuple[float, float] | None,
    show: bool,
    up_axis: str,
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
    bvh_list : list[Bvh]
        Skeleton objects.
    coords_list : list[ndarray]
        Spatial coordinates, each ``(F, N, 3)``.
    labels : list[str] or None
        Legend labels.
    figsize : (float, float) or None
        Figure size.
    show : bool
        Whether to call ``plt.show()``.
    up_axis : str
        ``'x'``, ``'y'``, or ``'z'`` — from the first skeleton.
    ax : matplotlib.axes.Axes, optional
        Existing 2D axes to draw on. If provided, no new figure is
        created. Works with single or multiple skeletons.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    from ..tools import _AXIS_CHAR_TO_IDX

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
                _trajectory_data_aspect(bvh_list, coords_list))
        # constrained_layout handles external (bbox_to_anchor) legends
        # without clipping; tight_layout does not.
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    # Track which horizontal axes are used across all skeletons
    all_horiz: set[tuple[int, int]] = set()
    # Track full-skeleton horizontal extents for the non-tight limit mode
    skeleton_h0_bounds: list[tuple[float, float]] = []
    skeleton_h1_bounds: list[tuple[float, float]] = []

    for i, (bvh_obj, coords) in enumerate(zip(bvh_list, coords_list)):
        # Per-skeleton up axis, honoring any manual world_up override
        up_idx = _AXIS_CHAR_TO_IDX[bvh_obj.world_up[1]]
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
            # directly to (h0, h1) plot coordinates.
            #
            # Multi-skeleton plots truncate coords to the shortest clip
            # (see _prepare), so we slice root_trajectory to match h0/h1.
            F_plot = h0.shape[0]
            traj = bvh_obj.root_trajectory()[:F_plot]   # (F_plot, 4)
            facing_cos = traj[:, 3]                     # x-component (h0)
            facing_sin = traj[:, 2]                     # y-component (h1)
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
    skeleton_lines_list: list[list[tuple[int, int]]],
    all_line_artists: list[list[Any]],
) -> Any:
    """Create a FuncAnimation update function for bone rendering."""
    def update(f: int) -> list[Any]:
        artists: list[Any] = []
        for coords, bones, line_artists in zip(
                coords_list, skeleton_lines_list, all_line_artists):
            frame_data = coords[f]
            for line, (p_idx, c_idx) in zip(line_artists, bones):
                p = frame_data[p_idx]
                c = frame_data[c_idx]
                line.set_data_3d([p[0], c[0]], [p[1], c[1]], [p[2], c[2]])
            artists.extend(line_artists)
        return artists
    return update


def _draw_bones(
    ax: matplotlib.axes.Axes,
    frame_data: npt.NDArray[np.float64],
    bones: list[tuple[int, int]],
    color: tuple[float, float, float],
) -> None:
    """Draw all skeleton bones for a single frame on a 3D axes."""
    for p_idx, c_idx in bones:
        p = frame_data[p_idx]
        c = frame_data[c_idx]
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                c=color, lw=2.5)


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
        renderer = fig.canvas.get_renderer()
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

def _trajectory_data_aspect(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
) -> float:
    """Aspect ratio (dx / dy) of the combined trajectory data.

    Each skeleton is projected onto its own horizontal plane (dropping
    its own up axis). The aggregate dx and dy are computed from the
    union of all projected root paths. Returns ``1.0`` for degenerate
    (single-point) data.
    """
    from ..tools import _AXIS_CHAR_TO_IDX

    h0_values: list[npt.NDArray[np.float64]] = []
    h1_values: list[npt.NDArray[np.float64]] = []
    for bvh_obj, coords in zip(bvh_list, coords_list):
        up_idx = _AXIS_CHAR_TO_IDX[bvh_obj.world_up[1]]
        horiz = [j for j in range(3) if j != up_idx]
        root_traj = coords[:, 0, :]
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
