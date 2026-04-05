"""Visualization module for pybvh.

Provides three main functions:

- :func:`frame` — static 3D skeleton snapshot (matplotlib).
- :func:`play` — interactive playback with camera controls.
- :func:`render` — fast export to video/GIF/HTML.
- :func:`trajectory` — 2D top-down root trajectory plot.

Backends
--------
``render`` supports ``"opencv"`` (fast, optional dep) and ``"matplotlib"``
(default fallback). When *backend* is ``"auto"`` (the default), OpenCV is
used if available.

``play`` supports ``"k3d"`` (Jupyter notebooks, optional dep), ``"vedo"``
(desktop window, optional dep), and ``"matplotlib"`` (fallback). When
*backend* is ``"auto"``, the best available backend for the current
environment is selected automatically.

Install optional backends::

    pip install pybvh[opencv]       # fast video rendering
    pip install pybvh[interactive]  # k3d for Jupyter
    pip install pybvh[viewer]       # vedo for desktop
    pip install pybvh[all-viz]      # all of the above
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TYPE_CHECKING

from ._common import (
    get_skeleton_lines,
    normalize_input,
    compute_unified_limits,
    get_camera_angles,
    align_frame_counts,
)

# Re-export get_forw_up_axis so existing code that does
# `from pybvh.bvhplot import get_forw_up_axis` (or tests that check
# `bvhplot.get_forw_up_axis`) continues to work.
from ..tools import get_forw_up_axis  # noqa: F401

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from ..bvh import Bvh


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_notebook() -> bool:
    """Check if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (ImportError, AttributeError):
        return False


def _has_display() -> bool:
    """Check if a display server is available."""
    import os
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _resolve_render_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import cv2  # noqa: F401
        return "opencv"
    except ImportError:
        return "matplotlib"


def _resolve_play_backend(requested: str) -> tuple[str, int]:
    """Resolve the play backend and its fallback tier.

    Returns
    -------
    backend_name : str
        One of ``"k3d"``, ``"vedo"``, ``"opencv_notebook"``,
        ``"matplotlib"``.
    tier : int
        0 = explicit (no warnings), 1 = best auto,
        2 = fast fallback, 3 = slow fallback.
    """
    if requested != "auto":
        return requested, 0

    in_notebook = _detect_notebook()

    if in_notebook:
        try:
            import k3d  # noqa: F401
            return "k3d", 1
        except ImportError:
            pass
        try:
            import cv2  # noqa: F401
            return "opencv_notebook", 2
        except ImportError:
            pass
        return "matplotlib", 3

    # Script path
    if _has_display():
        try:
            import vedo  # noqa: F401
            return "vedo", 1
        except ImportError:
            pass
    return "matplotlib", 2


# ---------------------------------------------------------------------------
# Common preparation
_VALID_SYNC = {"truncate", "pad"}


def _validate_sync(sync: str) -> None:
    if sync not in _VALID_SYNC:
        raise ValueError(
            f"Unknown sync mode {sync!r}. "
            f"Choose from: {sorted(_VALID_SYNC)}")


# ---------------------------------------------------------------------------

def _prepare(
    bvh: Bvh | list[Bvh],
    frames: int | npt.NDArray[np.floating] | None,
    centered: str,
    camera: str | tuple[float, float],
    pad: bool = False,
) -> tuple[
    list,                                     # bvh_list
    list[npt.NDArray[np.float64]],            # coords_list
    list[list[tuple[int, int]]],              # skeleton_lines_list
    npt.NDArray[np.float64],                  # center
    float,                                    # half_span
    float,                                    # azimuth
    float,                                    # elevation
    str,                                      # up_axis
]:
    """Shared setup for all visualization functions."""
    _VALID_CENTERED = {"world", "skeleton", "first"}
    if centered not in _VALID_CENTERED:
        raise ValueError(
            f"Unknown centered mode {centered!r}. "
            f"Choose from: {sorted(_VALID_CENTERED)}")

    bvh_list, coords_list = normalize_input(bvh, frames, centered)
    coords_list = align_frame_counts(coords_list, pad=pad)
    skeleton_lines_list = [get_skeleton_lines(b) for b in bvh_list]
    center, half_span = compute_unified_limits(coords_list)

    # Use first skeleton's first frame as reference for camera
    ref_frame = coords_list[0][0]
    azimuth, elevation, up_axis = get_camera_angles(
        bvh_list[0], ref_frame, camera)

    return (bvh_list, coords_list, skeleton_lines_list,
            center, half_span, azimuth, elevation, up_axis)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rest_pose(
    bvh: Bvh | list[Bvh],
    *,
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    camera: str | tuple[float, float] = "front",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]:
    """Plot the rest pose (T-pose / bind pose) of one or more skeletons.

    All joint angles are zero and root is at the origin.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for side-by-side comparison.
    labels : list[str], optional
        Subplot titles for side-by-side comparison.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True`` (default), call ``plt.show()``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : Axes or list[Axes]
        Single axes when one skeleton, list when multiple.
    """
    bvh_list = bvh if isinstance(bvh, list) else [bvh]

    # Build rest-pose coords as (1, N, 3) arrays and go through the
    # same pipeline as frame(), bypassing get_spatial_coord.
    from ._matplotlib import frame_mpl

    coords_list = [b.get_rest_pose(mode='coordinates')[np.newaxis]
                   for b in bvh_list]
    skeleton_lines_list = [get_skeleton_lines(b) for b in bvh_list]
    center, half_span = compute_unified_limits(coords_list)
    azimuth, elevation, up_axis = get_camera_angles(
        bvh_list[0], coords_list[0][0], camera)

    return frame_mpl(
        bvh_list, coords_list, labels, figsize, show,
        skeleton_lines_list, center, half_span, azimuth, elevation, up_axis)


def frame(
    bvh: Bvh | list[Bvh],
    frame: int | npt.NDArray[np.floating] = 0,
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    camera: str | tuple[float, float] = "front",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]:
    """Plot a static 3D skeleton snapshot.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for side-by-side comparison.
    frame : int or ndarray, optional
        Frame index (default 0) or pre-computed spatial coordinates.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Subplot titles for side-by-side comparison.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True`` (default), call ``plt.show()``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : Axes or list[Axes]
        Single axes when one skeleton, list when multiple.
    """
    from ._matplotlib import frame_mpl

    (bvh_list, coords_list, skeleton_lines_list,
     center, half_span, azimuth, elevation, up_axis) = _prepare(
        bvh, frame, centered, camera)

    return frame_mpl(
        bvh_list, coords_list, labels, figsize, show,
        skeleton_lines_list, center, half_span, azimuth, elevation, up_axis)


def render(
    bvh: Bvh | list[Bvh],
    filepath: str | Path = Path("./anim.mp4"),
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    fps: int = -1,
    backend: str = "auto",
    camera: str | tuple[float, float] = "front",
    resolution: tuple[int, int] = (1920, 1080),
    show_axis: bool = False,
    sync: str = "truncate",
) -> Path:
    """Render animation to a video, GIF, or HTML file.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for side-by-side comparison.
    filepath : str or Path, optional
        Output file path (default ``"./anim.mp4"``). Format is inferred
        from the extension: ``.mp4``, ``.mov``, ``.avi``, ``.gif``,
        ``.webp``, ``.apng``, ``.html``.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Labels for each skeleton when comparing.
    fps : int, optional
        Frames per second. ``-1`` (default) uses the BVH frame rate.
    backend : str, optional
        ``"auto"`` (default), ``"opencv"``, or ``"matplotlib"``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
    resolution : (int, int), optional
        Output resolution ``(width, height)`` in pixels.
        Default ``(1920, 1080)``. Only used by the OpenCV backend.
    show_axis : bool, optional
        Show 3D axes (default ``False``). Only used by matplotlib backend.
    sync : str, optional
        How to handle different frame counts in side-by-side comparison:
        ``"truncate"`` (default) stops at the shortest clip;
        ``"pad"`` continues to the longest clip (shorter clips freeze
        on their last frame).

    Returns
    -------
    Path
        The path to the written file.
    """
    filepath = Path(filepath)
    _validate_sync(sync)
    pad = sync == "pad"

    (bvh_list, coords_list, skeleton_lines_list,
     center, half_span, azimuth, elevation, up_axis) = _prepare(
        bvh, None, centered, camera, pad=pad)

    # Resolve FPS
    actual_fps: float
    if fps == -1:
        actual_fps = 1.0 / bvh_list[0].frame_frequency
    else:
        actual_fps = float(fps)

    backend_name = _resolve_render_backend(backend)

    if backend_name != backend and backend == "auto" and backend_name == "matplotlib":
        import warnings
        warnings.warn(
            "OpenCV not found for fast rendering. "
            "Install with: pip install pybvh[opencv]. "
            "Falling back to matplotlib (slower).",
            stacklevel=2)

    if backend_name == "opencv":
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError(
                "OpenCV backend requires opencv-python. "
                "Install with: pip install pybvh[opencv]")
        from ._opencv import render_opencv
        return render_opencv(
            bvh_list, coords_list, filepath, actual_fps,
            resolution, labels, show_axis, skeleton_lines_list,
            center, half_span, azimuth, elevation, up_axis)

    else:  # matplotlib
        from ._matplotlib import render_mpl
        return render_mpl(
            bvh_list, coords_list, filepath, actual_fps, labels,
            skeleton_lines_list, center, half_span,
            azimuth, elevation, up_axis, show_axis)


def play(
    bvh: Bvh | list[Bvh],
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    fps: int = -1,
    backend: str = "auto",
    camera: str | tuple[float, float] = "front",
    sync: str = "truncate",
    resolution: tuple[int, int] = (960, 540),
    quality: str = "high",
) -> object:
    """Play back motion data.

    Auto-detects the best backend for the current environment:

    - **Tier 1 (interactive):** k3d in Jupyter notebooks, vedo on desktop.
    - **Tier 2 (fast fallback):** OpenCV renders to an inline video
      (notebook) or matplotlib animated window (script).
    - **Tier 3 (slow fallback):** matplotlib jshtml inline (notebook) or
      animated window (script).

    When falling back, warnings indicate which packages to install for
    a better experience.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for side-by-side comparison.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Labels for each skeleton when comparing.
    fps : int, optional
        Frames per second. ``-1`` (default) uses the BVH frame rate,
        capped at 30 for matplotlib backends (via frame subsampling).
    backend : str, optional
        ``"auto"`` (default), ``"k3d"``, ``"vedo"``, or ``"matplotlib"``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
    sync : str, optional
        How to handle different frame counts in side-by-side comparison:
        ``"truncate"`` (default) stops at the shortest clip;
        ``"pad"`` continues to the longest clip (shorter clips freeze
        on their last frame).
    resolution : (int, int), optional
        Output resolution ``(width, height)`` in pixels for the OpenCV
        notebook fallback. Default ``(960, 540)``. Ignored by
        interactive backends (k3d, vedo) and matplotlib.
    quality : str, optional
        Visual quality for the vedo desktop backend:
        ``"high"`` (default) uses 3D tubes and spheres with lighting;
        ``"fast"`` uses flat lines and points for maximum performance.
        Ignored by other backends.

    Returns
    -------
    object
        Backend-specific return value (widget, plotter, or None).
    """
    import math
    import warnings

    valid_backends = {"auto", "k3d", "vedo", "matplotlib"}
    if backend not in valid_backends:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Choose from: {sorted(valid_backends)}")

    _VALID_QUALITY = {"fast", "high"}
    if quality not in _VALID_QUALITY:
        raise ValueError(
            f"Unknown quality {quality!r}. "
            f"Choose from: {sorted(_VALID_QUALITY)}")

    _validate_sync(sync)
    pad = sync == "pad"
    (bvh_list, coords_list, skeleton_lines_list,
     center, half_span, azimuth, elevation, up_axis) = _prepare(
        bvh, None, centered, camera, pad=pad)

    bvh_fps = 1.0 / bvh_list[0].frame_frequency
    actual_fps = float(fps) if fps > 0 else bvh_fps

    backend_name, tier = _resolve_play_backend(backend)

    # --- Warnings (auto mode only, tier > 0) ---
    if tier >= 2:
        warnings.warn(
            "No interactive backend (k3d, vedo) found. "
            "Install with: pip install pybvh[interactive]",
            stacklevel=2)
    if tier >= 3:
        warnings.warn(
            "OpenCV not found for fast rendering. "
            "Install with: pip install pybvh[opencv]. "
            "Falling back to matplotlib (slow for long clips).",
            stacklevel=2)

    # --- Subsample to 30fps when fps is auto ---
    # Notebooks (k3d, jshtml) and matplotlib windows can't keep up with
    # high frame rates (120fps). Cap at 30fps for correct playback speed.
    # opencv_notebook uses a video player that handles any fps natively.
    # vedo uses persistent actors + timer, handles high fps well.
    _PLAY_MAX_FPS = 30.0
    if (fps == -1
            and backend_name not in ("opencv_notebook", "vedo")
            and bvh_fps > _PLAY_MAX_FPS):
        subsample_step = math.ceil(bvh_fps / _PLAY_MAX_FPS)
        coords_list = [c[::subsample_step] for c in coords_list]
        actual_fps = bvh_fps / subsample_step

    # --- Dispatch ---
    if backend_name == "k3d":
        try:
            import k3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "k3d backend requires k3d and ipywidgets. "
                "Install with: pip install pybvh[interactive]")
        from ._k3d import play_k3d
        return play_k3d(
            bvh_list, coords_list, actual_fps, labels,
            skeleton_lines_list, center, half_span)

    elif backend_name == "vedo":
        try:
            import vedo  # noqa: F401
        except ImportError:
            raise ImportError(
                "vedo backend requires vedo. "
                "Install with: pip install pybvh[viewer]")
        from ._vedo import play_vedo
        return play_vedo(
            bvh_list, coords_list, actual_fps, labels,
            skeleton_lines_list, center, half_span,
            up_axis=up_axis, azimuth=azimuth, elevation=elevation,
            quality=quality)

    elif backend_name == "opencv_notebook":
        import tempfile
        from ._opencv import render_opencv
        from IPython.display import display, Video  # type: ignore[import-untyped]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        render_opencv(
            bvh_list, coords_list, tmp_path, actual_fps,
            resolution, labels, False, skeleton_lines_list,
            center, half_span, azimuth, elevation, up_axis)

        display(Video(str(tmp_path), embed=True, mimetype="video/mp4"))
        tmp_path.unlink(missing_ok=True)
        return None

    else:  # matplotlib
        from ._matplotlib import play_mpl
        play_mpl(
            bvh_list, coords_list, actual_fps, labels,
            skeleton_lines_list, center, half_span,
            azimuth, elevation, up_axis,
            in_notebook=_detect_notebook())
        return None


def trajectory(
    bvh: Bvh | list[Bvh],
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot 2D top-down trajectory of the root joint.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for overlaid comparison.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Legend labels.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True`` (default), call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    (bvh_list, coords_list, skeleton_lines_list,
     center, half_span, azimuth, elevation, up_axis) = _prepare(
        bvh, None, centered, "front")

    from ._matplotlib import trajectory_mpl
    return trajectory_mpl(
        bvh_list, coords_list, labels, figsize, show, up_axis)
