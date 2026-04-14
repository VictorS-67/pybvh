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
        axs_flat: list[matplotlib.axes.Axes] = [ax]
    else:
        if figsize is None:
            figsize = (6 * n, 6)

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
        ax_i.view_init(
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])
        ax_i.set_xlabel('x')
        ax_i.set_ylabel('y')
        ax_i.set_zlabel('z')

        if labels and i < len(labels):
            ax_i.set_title(labels[i])

    if ax is None:
        plt.tight_layout()
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
        figsize=(6 * n, 6), squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    all_line_artists: list[list[Any]] = []
    for i, (bones, ax) in enumerate(zip(skeleton_lines_list, axs_flat)):
        color = PALETTE_MPL[i % len(PALETTE_MPL)] if n > 1 else (0.1, 0.2, 0.8)
        line_artists = [ax.plot([], [], [], c=color, lw=2.5)[0] for _ in bones]
        all_line_artists.append(line_artists)

        _set_axis_limits(ax, centers[i], half_spans[i])
        ax.view_init(
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])

        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        if labels and i < len(labels):
            ax.set_title(labels[i])

    plt.tight_layout()

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
        _world_lateral_unit_at_frame,
    )

    base_update = _make_update_fn(
        coords_list, skeleton_lines_list, all_line_artists)

    # Precompute per-skeleton base camera and frame-0 lateral unit vectors.
    base_angles: list[tuple[float, float, str]] = []
    base_laterals: list[np.ndarray | None] = []
    up_vecs: list[np.ndarray] = []
    for bvh_obj, coords in zip(bvh_list, coords_list):
        az, el, up = get_camera_angles(bvh_obj, coords[0], camera)
        base_angles.append((az, el, up))
        base_laterals.append(
            _world_lateral_unit_at_frame(bvh_obj, coords[0], bvh_obj.world_up))
        up_vecs.append(_axis_to_vector(bvh_obj.world_up))

    def update(frame):
        artists = base_update(frame)
        for i, (bvh_obj, coords, ax) in enumerate(
                zip(bvh_list, coords_list, axs_flat)):
            az0, el0, up0 = base_angles[i]
            lateral_0 = base_laterals[i]
            if lateral_0 is None:
                ax.view_init(elev=el0, azim=az0, vertical_axis=up0)
                continue
            lateral_f = _world_lateral_unit_at_frame(
                bvh_obj, coords[frame], bvh_obj.world_up)
            if lateral_f is None:
                ax.view_init(elev=el0, azim=az0, vertical_axis=up0)
                continue
            delta = _signed_rotation_delta_around_axis(
                lateral_0, lateral_f, up_vecs[i])
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
        ax.view_init(
            elev=elevations[i], azim=azimuths[i], vertical_axis=up_axes[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

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

    ax_provided = ax is not None
    if ax_provided:
        if hasattr(ax, 'get_zlim'):
            raise ValueError(
                "ax must be a 2D axes for trajectory(). "
                "Do not pass subplot_kw={'projection': '3d'} when creating it."
            )
        fig = ax.get_figure()
    else:
        if figsize is None:
            figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)

    # Track which horizontal axes are used across all skeletons
    all_horiz: set[tuple[int, int]] = set()

    for i, (bvh_obj, coords) in enumerate(zip(bvh_list, coords_list)):
        # Per-skeleton up axis, honoring any manual world_up override
        up_idx = _AXIS_CHAR_TO_IDX[bvh_obj.world_up[1]]
        horiz = [j for j in range(3) if j != up_idx]
        all_horiz.add((horiz[0], horiz[1]))

        root_traj = coords[:, 0, :]  # (F, 3)
        h0 = root_traj[:, horiz[0]]
        h1 = root_traj[:, horiz[1]]

        color = PALETTE_MPL[i % len(PALETTE_MPL)]
        label = labels[i] if labels and i < len(labels) else None

        ax.plot(h0, h1, c=color, lw=1.5, label=label)
        ax.scatter(h0[0], h1[0], c=[color], marker='o', s=60, zorder=5)
        ax.scatter(h0[-1], h1[-1], c=[color], marker='s', s=60, zorder=5)

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
    ax.legend(handles=handles, loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3)

    if not ax_provided:
        plt.tight_layout()
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
