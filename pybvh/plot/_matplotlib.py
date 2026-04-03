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
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
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
    center, half_span : ndarray, float
        Bounding box from :func:`_common.compute_unified_limits`.
    azimuth, elevation : float
        Camera angles in degrees.
    up_axis : str
        ``'x'``, ``'y'``, or ``'z'``.

    Returns
    -------
    fig : Figure
    axs : Axes or list[Axes]
    """
    n = len(bvh_list)
    if figsize is None:
        figsize = (6 * n, 6)

    fig, axs = plt.subplots(
        1, n, subplot_kw=dict(projection="3d"), figsize=figsize,
        squeeze=False)
    axs_flat: list[matplotlib.axes.Axes] = list(axs[0])

    for i, (coords, bones, ax) in enumerate(
            zip(coords_list, skeleton_lines_list, axs_flat)):
        frame_data = coords[0]  # (N, 3) — first frame
        color = PALETTE_MPL[i % len(PALETTE_MPL)] if n > 1 else (0.1, 0.2, 0.8)

        _draw_bones(ax, frame_data, bones, color)
        _set_axis_limits(ax, center, half_span)
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis=up_axis)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if labels and i < len(labels):
            ax.set_title(labels[i])

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
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
    show_axis: bool,
) -> Path:
    """Render animation to a video/GIF/HTML file via matplotlib.

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

        _set_axis_limits(ax, center, half_span)
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis=up_axis)

        if not show_axis:
            ax.axis('off')
        else:
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

    if writer_name == "jshtml":
        html_content = anim.to_jshtml()
        with open(filepath, 'w') as f:
            f.write(html_content)
    else:
        anim.save(filepath, writer=writer_name)

    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# Interactive playback (matplotlib fallback)
# ---------------------------------------------------------------------------

def play_mpl(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
    in_notebook: bool = False,
) -> None:
    """Playback via matplotlib.

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

        _set_axis_limits(ax, center, half_span)
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis=up_axis)
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

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    from ..tools import get_up_axis_index

    axis_names = ['x', 'y', 'z']

    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    # Track which horizontal axes are used across all skeletons
    all_horiz: set[tuple[int, int]] = set()

    for i, (bvh_obj, coords) in enumerate(zip(bvh_list, coords_list)):
        # Per-skeleton up axis
        up_idx = get_up_axis_index(bvh_obj, coords[0])
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
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)

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
