"""Visualization module for pybvh.

Provides five main functions:

- :func:`rest_pose` — T-pose / bind pose visualization (matplotlib).
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
    _UP_AXIS_INDEX,
)

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

def _match_frame_rates(
    bvh_list: list[Bvh],
    match_fps: str | None,
) -> list[Bvh]:
    """Warn on frame-rate mismatch and optionally resample to a common rate.

    Parameters
    ----------
    bvh_list : list[Bvh]
        Input clips (not modified in place).
    match_fps : str or None
        ``None`` — warn only, no resampling.
        ``"lowest"`` — resample all clips to the lowest frame rate.
        ``"highest"`` — resample all clips to the highest frame rate.

    Returns
    -------
    list[Bvh]
        Possibly resampled clips (originals returned when no resampling
        needed).
    """
    if len(bvh_list) <= 1:
        return bvh_list

    rates = [1.0 / b.frame_time if b.frame_time > 0 else 0.0 for b in bvh_list]
    if all(abs(r - rates[0]) < 0.5 for r in rates):
        return bvh_list  # all close enough

    import warnings
    rate_strs = ", ".join(f"{r:.1f}" for r in rates)
    if match_fps is None:
        warnings.warn(
            f"Frame rates differ across clips ({rate_strs} fps). \n"
            f"Playback speed will not match real time for all clips. \n"
            f"Use match_fps='lowest' or match_fps='highest' to resample \n"
            f"automatically, or call bvh.resample(target_fps) manually.",
            UserWarning,
            stacklevel=3,
        )
        return bvh_list

    valid = {"lowest", "highest"}
    if match_fps not in valid:
        raise ValueError(f"match_fps must be None, 'lowest', or 'highest', got {match_fps!r}")

    target_fps = min(rates) if match_fps == "lowest" else max(rates)
    result = []
    for b, r in zip(bvh_list, rates):
        if abs(r - target_fps) < 0.5:
            result.append(b)
        else:
            result.append(b.resample(target_fps))
    return result


def _apply_scene_spacing(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    spacing: float | str,
    up_axis_char: str,
    centered: str,
) -> list[npt.NDArray[np.float64]]:
    """Offset each skeleton laterally so they don't overlap in a shared 3D scene.

    Used by single-scene backends (k3d, vedo). Multi-panel backends
    (matplotlib, opencv) don't need this — they already use separate viewports.

    Parameters
    ----------
    bvh_list : list[Bvh]
        Skeleton objects (used to determine forward direction).
    coords_list : list[ndarray]
        Spatial coordinates per skeleton, each ``(F, N, 3)``.
    spacing : float or "auto"
        ``"auto"`` computes spacing from skeleton 0's lateral bounding-box
        width × 1.2. A float is used directly (in scene units).
        ``"auto"`` with ``centered="world"`` returns the list unchanged.
    up_axis_char : str
        Single character ``'x'``, ``'y'``, or ``'z'``.
    centered : str
        Centering mode — used to determine whether auto-spacing applies.

    Returns
    -------
    list[ndarray]
        Possibly offset coordinate arrays (new arrays; originals unchanged).
    """
    if len(bvh_list) <= 1:
        return coords_list

    if spacing == "auto" and centered == "world":
        return coords_list  # respect raw world coordinates

    # Lateral axis = the one that is neither up nor forward
    up_idx = _UP_AXIS_INDEX.get(up_axis_char, 2)
    fwd_str = bvh_list[0].forward_at(frame=0)
    fwd_idx = _UP_AXIS_INDEX.get(fwd_str[1], 0)
    lat_idx = next(i for i in range(3) if i != up_idx and i != fwd_idx)

    if spacing == "auto":
        c0 = coords_list[0].reshape(-1, 3)
        bbox_width = float(c0[:, lat_idx].max() - c0[:, lat_idx].min())
        effective_spacing = max(bbox_width, 0.1) * 1.2
    else:
        effective_spacing = float(spacing)

    if effective_spacing == 0.0:
        return coords_list

    offset_unit = np.zeros(3)
    offset_unit[lat_idx] = 1.0  # always positive lateral direction

    return [
        coords + (offset_unit * k * effective_spacing)[np.newaxis, np.newaxis, :]
        for k, coords in enumerate(coords_list)
    ]


def _warn_world_up_mismatch(
    bvh_list: list[Bvh],
) -> None:
    """Warn when skeletons have different world_up values."""
    import warnings
    if len(bvh_list) <= 1:
        return
    world_ups = [b.world_up for b in bvh_list]
    if len(set(world_ups)) > 1:
        warnings.warn(
            f"Clips have different world_up values ({', '.join(world_ups)}). \n"
            "Use pybvh.reorient_world_up() to normalize before comparing.",
            UserWarning,
            stacklevel=3,
        )


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
    list[npt.NDArray[np.float64]],            # centers (per skeleton)
    list[float],                              # half_spans (per skeleton)
    list[float],                              # azimuths (per skeleton)
    list[float],                              # elevations (per skeleton)
    list[str],                                # up_axes (per skeleton)
]:
    """Shared setup for all visualization functions.

    Returns per-skeleton camera angles and bounding boxes so that
    side-by-side comparisons of skeletons with different up or forward
    axes render each one correctly in its own subplot.
    """
    _VALID_CENTERED = {"world", "skeleton", "first"}
    if centered not in _VALID_CENTERED:
        raise ValueError(
            f"Unknown centered mode {centered!r}. "
            f"Choose from: {sorted(_VALID_CENTERED)}")

    bvh_list, coords_list = normalize_input(bvh, frames, centered)
    coords_list = align_frame_counts(coords_list, pad=pad)
    skeleton_lines_list = [get_skeleton_lines(b) for b in bvh_list]

    # Per-skeleton bounding box: each subplot gets its own cubic box
    # centered on its own skeleton. For same-skeleton comparisons the
    # boxes end up identical; for mixed skeletons this prevents the
    # unified box from swallowing both into a tiny corner.
    centers: list[npt.NDArray[np.float64]] = []
    half_spans: list[float] = []
    for c in coords_list:
        ctr, hs = compute_unified_limits([c])
        centers.append(ctr)
        half_spans.append(hs)

    # Per-skeleton camera: each subplot oriented for its own detected
    # forward/up axes. matplotlib's view_init(vertical_axis=...) needs
    # to match the skeleton's actual vertical component, which can
    # differ across BVH files (Y-up vs Z-up).
    azimuths: list[float] = []
    elevations: list[float] = []
    up_axes: list[str] = []
    for b, c in zip(bvh_list, coords_list):
        az, el, ua = get_camera_angles(b, c[0], camera)
        azimuths.append(az)
        elevations.append(el)
        up_axes.append(ua)

    return (bvh_list, coords_list, skeleton_lines_list,
            centers, half_spans, azimuths, elevations, up_axes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rest_pose(
    bvh: Bvh | list[Bvh],
    *,
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    camera: str | tuple[float, float] = "front",
    ax: matplotlib.axes.Axes | None = None,
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
        If ``True``, call ``plt.show()``. Default ``False``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw on. If provided, no new figure is
        created. Only supported for a single skeleton (raises
        ``ValueError`` when ``bvh`` is a list).

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

    coords_list = [b.rest_pose_coords(mode='coordinates')[np.newaxis]
                   for b in bvh_list]
    skeleton_lines_list = [get_skeleton_lines(b) for b in bvh_list]

    # Per-skeleton bounding box and camera so mixed-up-axis comparisons
    # render each skeleton in its own orientation.
    centers: list = []
    half_spans: list[float] = []
    for c in coords_list:
        ctr, hs = compute_unified_limits([c])
        centers.append(ctr)
        half_spans.append(hs)

    azimuths: list[float] = []
    elevations: list[float] = []
    up_axes: list[str] = []
    for b, c in zip(bvh_list, coords_list):
        az, el, ua = get_camera_angles(b, c[0], camera)
        azimuths.append(az)
        elevations.append(el)
        up_axes.append(ua)

    return frame_mpl(
        bvh_list, coords_list, labels, figsize, show,
        skeleton_lines_list, centers, half_spans,
        azimuths, elevations, up_axes,
        ax=ax)


def frame(
    bvh: Bvh | list[Bvh],
    frame: int | npt.NDArray[np.floating] = 0,
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    camera: str | tuple[float, float] = "front",
    ax: matplotlib.axes.Axes | None = None,
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
        If ``True``, call ``plt.show()``. Default ``False``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw on. If provided, no new figure is
        created. Only supported for a single skeleton (raises
        ``ValueError`` when ``bvh`` is a list).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : Axes or list[Axes]
        Single axes when one skeleton, list when multiple.
    """
    from ._matplotlib import frame_mpl

    (bvh_list, coords_list, skeleton_lines_list,
     centers, half_spans, azimuths, elevations, up_axes) = _prepare(
        bvh, frame, centered, camera)

    return frame_mpl(
        bvh_list, coords_list, labels, figsize, show,
        skeleton_lines_list, centers, half_spans,
        azimuths, elevations, up_axes,
        ax=ax)


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
    follow: bool = False,
    match_fps: str | None = None,
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
    follow : bool, optional
        If ``True``, the camera orientation is recomputed every frame
        using each skeleton's current facing direction (via
        :meth:`~pybvh.bvh.Bvh.forward_at`), so the view orbits with the
        character. Only affects preset cameras (``"front"``, ``"side"``,
        ``"top"``); custom ``(azimuth, elevation)`` tuples are fixed and
        ignore ``follow``. Default ``False`` (stable camera).
    match_fps : str or None, optional
        How to handle clips with different frame rates in side-by-side
        rendering.  ``None`` (default) emits a warning but does not
        resample.  ``"lowest"`` resamples all clips to the lowest frame
        rate.  ``"highest"`` resamples all clips to the highest frame rate
        (using SLERP interpolation for added frames).

    Returns
    -------
    Path
        The path to the written file.
    """
    filepath = Path(filepath)
    _validate_sync(sync)
    pad = sync == "pad"

    # Handle frame-rate mismatch before computing FK coordinates
    if not isinstance(bvh, list):
        bvh_input = [bvh]
    else:
        bvh_input = bvh
    bvh_input = _match_frame_rates(bvh_input, match_fps)
    bvh = bvh_input if len(bvh_input) > 1 else bvh_input[0]

    (bvh_list, coords_list, skeleton_lines_list,
     centers, half_spans, azimuths, elevations, up_axes) = _prepare(
        bvh, None, centered, camera, pad=pad)

    # A custom (azim, elev) tuple means the camera is fixed; follow is
    # a no-op in that case because there's no orientation to track.
    effective_follow = follow and not isinstance(camera, tuple)

    # Resolve FPS
    actual_fps: float
    if fps == -1:
        actual_fps = 1.0 / bvh_list[0].frame_time
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
            centers, half_spans, azimuths, elevations, up_axes,
            follow=effective_follow, camera=camera)

    else:  # matplotlib
        from ._matplotlib import render_mpl
        return render_mpl(
            bvh_list, coords_list, filepath, actual_fps, labels,
            skeleton_lines_list, centers, half_spans,
            azimuths, elevations, up_axes, show_axis,
            follow=effective_follow, camera=camera)


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
    match_fps: str | None = None,
    spacing: float | str = "auto",
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
    match_fps : str or None, optional
        How to handle clips with different frame rates.  ``None``
        (default) emits a warning.  ``"lowest"`` or ``"highest"``
        resamples all clips to match.
    spacing : float or "auto", optional
        Lateral separation between skeletons in single-scene backends (k3d,
        vedo). ``"auto"`` (default) spaces skeletons by 1.2 × the lateral
        bounding-box width of the first skeleton when ``centered`` is
        ``"first"`` or ``"skeleton"``; no spacing is applied when
        ``centered="world"`` (raw world coordinates are honoured). Pass a
        float (in scene units) to override. Ignored by multi-panel backends
        (matplotlib, OpenCV).

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

    if spacing != "auto":
        try:
            spacing_val = float(spacing)
        except (TypeError, ValueError):
            raise ValueError(
                f"spacing must be 'auto' or a non-negative number, got {spacing!r}")
        if spacing_val < 0:
            raise ValueError(
                f"spacing must be non-negative, got {spacing_val}")
        spacing = spacing_val

    _validate_sync(sync)
    pad = sync == "pad"

    # Handle frame-rate mismatch before computing FK coordinates
    if not isinstance(bvh, list):
        bvh_input = [bvh]
    else:
        bvh_input = bvh
    bvh_input = _match_frame_rates(bvh_input, match_fps)
    bvh = bvh_input if len(bvh_input) > 1 else bvh_input[0]

    (bvh_list, coords_list, skeleton_lines_list,
     centers, half_spans, azimuths, elevations, up_axes) = _prepare(
        bvh, None, centered, camera, pad=pad)

    bvh_fps = 1.0 / bvh_list[0].frame_time
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

    # --- world_up consistency check (all backends) ---
    _warn_world_up_mismatch(bvh_list)

    # --- Dispatch ---
    # For single-scene backends (vedo, k3d) there can only be ONE camera
    # and ONE bounding box. We apply lateral spacing so skeletons don't
    # overlap, then compute a unified bounding box from the spread coords.
    if backend_name == "k3d":
        try:
            import k3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "k3d backend requires k3d and ipywidgets. "
                "Install with: pip install pybvh[interactive]")
        from ._k3d import play_k3d
        spread_coords = _apply_scene_spacing(
            bvh_list, coords_list, spacing, up_axes[0], centered)
        shared_center, shared_half_span = compute_unified_limits(spread_coords)
        return play_k3d(
            bvh_list, spread_coords, actual_fps, labels,
            skeleton_lines_list, shared_center, shared_half_span,
            azimuths[0], elevations[0], up_axes[0])

    elif backend_name == "vedo":
        try:
            import vedo  # noqa: F401
        except ImportError:
            raise ImportError(
                "vedo backend requires vedo. "
                "Install with: pip install pybvh[viewer]")
        from ._vedo import play_vedo
        spread_coords = _apply_scene_spacing(
            bvh_list, coords_list, spacing, up_axes[0], centered)
        shared_center, shared_half_span = compute_unified_limits(spread_coords)
        return play_vedo(
            bvh_list, spread_coords, actual_fps, labels,
            skeleton_lines_list, shared_center, shared_half_span,
            up_axis=up_axes[0], azimuth=azimuths[0], elevation=elevations[0],
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
            centers, half_spans, azimuths, elevations, up_axes)

        display(Video(str(tmp_path), embed=True, mimetype="video/mp4"))
        tmp_path.unlink(missing_ok=True)
        return None

    else:  # matplotlib
        from ._matplotlib import play_mpl
        play_mpl(
            bvh_list, coords_list, actual_fps, labels,
            skeleton_lines_list, centers, half_spans,
            azimuths, elevations, up_axes,
            in_notebook=_detect_notebook())
        return None


def trajectory(
    bvh: Bvh | list[Bvh],
    *,
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
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
        If ``True``, call ``plt.show()``. Default ``False``.
    ax : matplotlib.axes.Axes, optional
        Existing 2D axes to draw on. If provided, no new figure is
        created. Works with single or multiple skeletons (overlaid).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    (bvh_list, coords_list, skeleton_lines_list,
     centers, half_spans, azimuths, elevations, up_axes) = _prepare(
        bvh, None, centered, "front")

    # trajectory_mpl() computes its own per-skeleton horizontal axes
    # internally (drop each skeleton's own up axis), so we just pass
    # the first up axis for the axis-labels fallback.
    from ._matplotlib import trajectory_mpl
    return trajectory_mpl(
        bvh_list, coords_list, labels, figsize, show, up_axes[0], ax=ax)
