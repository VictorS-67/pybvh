"""Visualization module for pybvh.

Provides six main functions, all accepting ``style=`` (a
:class:`Style` preset name or instance; ``"paper"`` is the default):

- :func:`rest_pose` — T-pose / bind pose visualization (matplotlib).
- :func:`frame` — static 3D skeleton snapshot (matplotlib, or a
  shadowed capsule render via ``backend="vedo"``).
- :func:`sequence` — the motion-paper still: sampled poses in one
  figure, lightness encoding time.
- :func:`play` — interactive playback with camera controls.
- :func:`render` — export to video/GIF/HTML.
- :func:`trajectory` — 2D top-down root trajectory plot.

Backends
--------
``render`` supports ``"opencv"`` (fast, optional dep),
``"matplotlib"`` (default fallback), and ``"vedo"`` (shadowed capsule
skeletons, never auto-selected). When *backend* is ``"auto"`` (the
default), OpenCV is used if available.

``play`` supports ``"k3d"`` (Jupyter notebooks, optional dep),
``"vedo"`` (desktop window, optional dep), ``"opencv"`` (notebook
inline video), and ``"matplotlib"`` (fallback). When *backend* is
``"auto"``, the best available backend for the current environment is
selected automatically.

Install optional backends::

    pip install pybvh[opencv]       # fast video rendering
    pip install pybvh[interactive]  # k3d for Jupyter
    pip install pybvh[viewer]       # vedo desktop viewer + renders
    pip install pybvh[all-viz]      # all of the above
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TYPE_CHECKING

from ._from_bvh import (
    as_clip_list,
    get_skeleton_lines,   # noqa: F401 — re-export (public since 0.5.0)
    make_scene,
    normalize_input,
)
from ._style import Style, resolve_style
from ._scene import Scene, align_frame_counts

__all__ = [
    "Style", "rest_pose", "frame", "sequence", "render", "play",
    "trajectory",
]


def _resolve_sample_frames(
    num_frames: int,
    n_poses: int,
    frames: slice | tuple[int, int] | None,
) -> npt.NDArray[np.intp]:
    """Equidistant sample indices for sequence(), honoring a range spec."""
    if not (isinstance(n_poses, int) and n_poses >= 1):
        raise ValueError(f"n_poses must be an integer >= 1, got {n_poses!r}")
    if frames is None:
        start, stop = 0, num_frames
    elif isinstance(frames, slice):
        start, stop, step = frames.indices(num_frames)
        if step != 1:
            raise ValueError(
                "frames slice must have step 1 — n_poses controls the "
                "sampling density.")
    elif isinstance(frames, tuple) and len(frames) == 2:
        start, stop = frames
        start = start if start >= 0 else num_frames + start
        stop = stop if stop >= 0 else num_frames + stop
    else:
        raise TypeError(
            f"frames must be a slice, a (start, stop) tuple, or None, "
            f"got {frames!r}")
    if not 0 <= start < stop <= num_frames:
        raise ValueError(
            f"frames range [{start}, {stop}) is empty or outside the "
            f"clip's {num_frames} frames.")
    return np.unique(np.linspace(start, stop - 1, n_poses).round()
                     .astype(np.intp))

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from ..bvh import Bvh
    from ._viewport import Turntable


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
    import sys
    if sys.platform in ('darwin', 'win32'):
        return True  # macOS/Windows always have a windowing system
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _module_importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# Formats only the matplotlib/pillow pipeline can write — OpenCV's
# VideoWriter handles video containers only (its .gif support is a
# dedicated pillow-based path inside render_opencv).
_MPL_ONLY_EXTENSIONS = {'.html', '.webp', '.apng', '.gif'}
_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi'}   # containers codec= applies to


def _resolve_render_backend(requested: str, ext: str) -> str:
    """Resolve the render backend from the request and the file extension.

    Under ``"auto"``, extensions OpenCV cannot write route to matplotlib
    even when cv2 is installed (previously they hit a misleading codec
    error); everything else prefers OpenCV when available. ``"vedo"``
    (the shadowed capsule renderer) is never auto-selected — it is a
    deliberate look, not a fallback.
    """
    valid = {"auto", "opencv", "matplotlib", "vedo"}
    if requested not in valid:
        raise ValueError(
            f"Unknown backend {requested!r}. "
            f"Choose from: {sorted(valid)}")
    if requested != "auto":
        return requested
    if ext in _MPL_ONLY_EXTENSIONS:
        return "matplotlib"
    return "opencv" if _module_importable("cv2") else "matplotlib"


def _written_fps(backend_name: str, suffix: str, fps: float) -> float:
    """The rate *backend_name* writes a ``suffix`` file at, given the
    resolved *fps*: matplotlib rounds it (:func:`._matplotlib.written_fps`),
    OpenCV and vedo write it as given."""
    if backend_name == "matplotlib":
        from ._matplotlib import written_fps
        return written_fps(fps, suffix)
    return fps


# A turntable turning 180 degrees or more a frame looks frozen or turning
# backwards: its period must be longer than this many frames.
_MIN_TURNTABLE_FRAMES = 2


def _resolve_fps(fps: float | None, frame_time: float) -> float:
    """Resolve the shared ``fps`` parameter of ``play()`` and ``render()``.

    ``None`` means "use the BVH frame rate", which the caller must have
    checked is set (:func:`_require_frame_rates`); anything else must be
    a positive number (fractional rates like 119.88 are fine).
    """
    if fps is None:
        return 1.0 / frame_time
    fps = float(fps)
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return fps


def _unset_rate_subject(clips: list[Bvh]) -> str | None:
    """Name the clips whose ``frame_time`` is 0 (unset), for an error
    message ("the clip at index 1 has"), or ``None`` when all have one.
    """
    unset = [i for i, clip in enumerate(clips) if clip.frame_time == 0]
    if not unset:
        return None
    indices = ", ".join(str(i) for i in unset)
    if len(clips) == 1:
        return "the clip has"
    if len(unset) == 1:
        return f"the clip at index {indices} has"
    return f"the clips at indices {indices} have"


def _require_frame_rates(
    clips: list[Bvh],
    entry: str,
    fps: float | None,
    ghost: int = 0,
    match_fps: str | None = None,
) -> None:
    """Raise when an animated entry point needs a rate a clip lacks.

    ``frame_time == 0`` is the "unset" value of a Bvh built in memory.
    Timing the animation needs a rate unless the caller gives ``fps``.
    Two options need every clip's own rate whatever ``fps`` says: a
    ghost trail, spaced in seconds of clip time, and ``match_fps`` on a
    comparison, which resamples each clip from its rate. The error
    offers only ways out that work, so ``fps=`` alone is offered only
    when neither option is asked for. Every clip is checked, not only
    the first that drives playback.
    """
    subject = _unset_rate_subject(clips)
    if subject is None:
        return
    reasons = {}
    if ghost:
        reasons["ghost="] = (
            "ghost= spaces its trailing poses style.ghost_spacing seconds "
            "apart in each clip's own time")
    if match_fps is not None and len(clips) > 1:
        reasons["match_fps="] = (
            f"match_fps={match_fps!r} resamples each clip from its own "
            f"rate")
    if reasons:
        options = " and ".join(reasons)
        drop = f"drop {options}"
        if fps is None:
            drop += f" and pass fps= to {entry}()"
        raise ValueError(
            f"{entry}() needs every clip's frame rate even when fps= is "
            f"given, because {'; '.join(reasons.values())}; but {subject} "
            f"frame_time 0 (unset). Set bvh.frame_time (or bvh.fps), or "
            f"{drop}.")
    if fps is None:
        raise ValueError(
            f"{entry}() needs a frame rate to time the animation, but "
            f"{subject} frame_time 0 (unset). Set bvh.frame_time (or "
            f"bvh.fps), or pass fps= to {entry}().")


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
        # "opencv" is the user-facing name (matching render()) for the
        # inline-video backend auto-selection calls "opencv_notebook" —
        # everything auto can choose must be nameable.
        if requested == "opencv":
            return "opencv_notebook", 0
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

    A clip whose ``frame_time`` is 0 has no rate, not a rate of 0 fps:
    rates are compared among the clips that have one, the warning names
    the others, and resampling refuses them (the router's
    :func:`_require_frame_rates` does so first, with advice that fits
    the call).

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

    unset_subject = _unset_rate_subject(bvh_list)
    if match_fps is not None and unset_subject is not None:
        raise ValueError(
            f"match_fps={match_fps!r} resamples each clip from its own "
            f"rate, but {unset_subject} frame_time 0 (unset). Set "
            f"bvh.frame_time (or bvh.fps), or drop match_fps=.")

    rates = [1.0 / b.frame_time for b in bvh_list if b.frame_time > 0]
    rates_agree = all(abs(r - rates[0]) < 0.5 for r in rates)
    if rates_agree and (unset_subject is None or not rates):
        return bvh_list  # all close enough, or none has a rate to compare

    rate_strs = ", ".join(f"{r:.1f}" for r in rates)
    if match_fps is None:
        if unset_subject is None:
            warnings.warn(
                f"Frame rates differ across clips ({rate_strs} fps). \n"
                f"Playback speed will not match real time for all clips. \n"
                f"Use match_fps='lowest' or match_fps='highest' to resample \n"
                f"automatically, or call bvh.resample(target_fps) manually.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"Frame rates differ across clips ({rate_strs} fps, and "
                f"{unset_subject} frame_time 0 (unset)). \n"
                f"Playback speed will not match real time for all clips. \n"
                f"Set bvh.frame_time (or bvh.fps) on every clip to compare "
                f"them in real time.",
                UserWarning,
                stacklevel=3,
            )
        return bvh_list

    valid = {"lowest", "highest"}
    if match_fps not in valid:
        raise ValueError(f"match_fps must be None, 'lowest', or 'highest', got {match_fps!r}")

    target_fps = min(rates) if match_fps == "lowest" else max(rates)
    result = []
    for b in bvh_list:
        if abs(1.0 / b.frame_time - target_fps) < 0.5:
            result.append(b)
        else:
            result.append(b.resample(target_fps))
    return result


def _spread_for_single_scene(
    scene: Scene,
    spacing: float | str,
    centered: str,
) -> Scene:
    """Router policy around :meth:`Scene.spread` for k3d and vedo.

    ``"auto"`` spacing respects raw world coordinates: two clips drawn
    under ``centered="world"`` are left exactly where their files put
    them. Every other combination spreads the views laterally so they
    do not overlap in the one shared scene.
    """
    if spacing == "auto" and centered == "world":
        return scene
    return scene.spread(spacing)


def _warn_world_up_mismatch(
    bvh_list: list[Bvh],
) -> None:
    """Warn when skeletons have different world_up values."""
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
    labels: list[str] | None = None,
    pad: bool = False,
) -> Scene:
    """Shared setup for all visualization functions.

    Returns a :class:`~._scene.Scene` with per-skeleton camera angles
    so that side-by-side comparisons of skeletons with different up or
    forward axes render each one correctly in its own subplot.
    """
    _VALID_CENTERED = {"world", "skeleton", "first"}
    if centered not in _VALID_CENTERED:
        raise ValueError(
            f"Unknown centered mode {centered!r}. "
            f"Choose from: {sorted(_VALID_CENTERED)}")

    bvh_list, coords_list = normalize_input(bvh, frames, centered)
    coords_list = align_frame_counts(coords_list, pad=pad)

    # Canonical (cached, robust) floor only where heights are world
    # units: FK-computed coords under "world" or "first" centering
    # (first-centering is ground-plane-only since 0.8.0). Root-relative
    # or caller-supplied coords use the min of the coords in use.
    canonical_floor = (
        not isinstance(frames, np.ndarray) and centered in ("world", "first"))

    # Which clip frames the coords are: one frame, the whole clip, or
    # none of them when the caller supplied the array.
    if isinstance(frames, np.ndarray):
        clip_frames = None
    elif isinstance(frames, int):
        clip_frames = frames
    else:
        clip_frames = slice(None)

    return make_scene(
        bvh_list, coords_list, camera, labels,
        canonical_floor=canonical_floor, clip_frames=clip_frames)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rest_pose(
    bvh: Bvh | list[Bvh],
    *,
    style: Style | str = "paper",
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
    style : Style or str, optional
        Visual styling: a preset name (``"paper"``, ``"debug"``,
        ``"dark"``) or a :class:`Style` instance for field-level
        control, e.g. ``Style("paper", floor=None)``. Default
        ``"paper"``.
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
    clips = as_clip_list(bvh)

    # Build rest-pose coords as (1, N, 3) arrays and go through the
    # same pipeline as frame(), bypassing spatial_coords.
    from ._matplotlib import frame_mpl

    coords_list = [b.rest_pose_positions()[np.newaxis]
                   for b in clips]
    # Rest-pose coords put the root at the origin, so the canonical
    # world floor does not apply — the floor is the pose's lowest point.
    scene = make_scene(clips, coords_list, camera, labels,
                       canonical_floor=False)

    return frame_mpl(scene, resolve_style(style),
                     figsize=figsize, show=show, ax=ax)


def frame(
    bvh: Bvh | list[Bvh],
    frame: int = 0,
    *,
    style: Style | str = "paper",
    backend: str = "matplotlib",
    coords: npt.NDArray[np.floating] | None = None,
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    camera: str | tuple[float, float] = "front",
    resolution: tuple[int, int] = (1100, 1000),
    filepath: str | Path | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]] | npt.NDArray[np.uint8]:
    """Plot a static 3D skeleton snapshot.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects. Pass a list for side-by-side comparison.
    frame : int, optional
        Frame index (default 0). Negative indices count from the end.
        Ignored when *coords* is given.
    style : Style or str, optional
        Visual styling: a preset name (``"paper"``, ``"debug"``,
        ``"dark"``) or a :class:`Style` instance for field-level
        control, e.g. ``Style("paper", floor=None)``. ``"paper"``
        (default) draws a ground plane, per-chain bone colors, and
        joint markers with axes hidden; ``"debug"`` reproduces the
        pre-0.9.0 look (single blue, full axes/ticks, no floor).
    backend : str, optional
        ``"matplotlib"`` (default) returns ``(fig, ax)``. ``"vedo"``
        renders a shadowed 3D capsule skeleton offscreen (headless-
        safe, requires ``pybvh[viewer]``) and returns an ``(H, W, 3)``
        uint8 RGB image instead — display it with ``plt.imshow`` or
        save it via *filepath*. Shadows are hard-edged projections
        (``Style.shadow``); ``figsize``/``ax`` do not apply, and all
        skeletons share one scene rather than side-by-side panels.
    coords : ndarray, optional
        Pre-computed spatial coordinates to plot instead of computing
        forward kinematics from *bvh*: ``(N, 3)`` for one frame, or
        ``(F, N, 3)`` of which the first frame is drawn. Only valid
        when *bvh* is a single Bvh object.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
        Ignored when *coords* is given.
    labels : list[str], optional
        Subplot titles for side-by-side comparison.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True``, call ``plt.show()``. Default ``False``.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``) or
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
    resolution : (int, int), optional
        Image size in pixels for ``backend="vedo"`` (ignored by
        matplotlib, which sizes via *figsize*/*dpi*).
    filepath : str or Path, optional
        Also write the rendered image here (``fig.savefig`` on the
        matplotlib backend, a PNG write on vedo).
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw on. If provided, no new figure is
        created. Only supported for a single skeleton (raises
        ``ValueError`` when ``bvh`` is a list). The style is applied
        to the provided axes — the default paper style hides its
        ticks and panes; pass ``style="debug"`` to draw into an axes
        whose full axis machinery you want to keep.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : Axes or list[Axes]
        Single axes when one skeleton, list when multiple.
        With ``backend="vedo"``: an ``(H, W, 3)`` uint8 RGB image
        array instead.
    """
    clips = as_clip_list(bvh)
    _VALID_FRAME_BACKENDS = {"matplotlib", "vedo"}
    if backend not in _VALID_FRAME_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Choose from: {sorted(_VALID_FRAME_BACKENDS)}")

    frame_spec = coords if coords is not None else frame
    scene = _prepare(clips, frame_spec, centered, camera, labels)

    if backend == "vedo":
        if not _module_importable("vedo"):
            raise ImportError(
                "vedo backend requires vedo. "
                "Install with: pip install pybvh[viewer]")
        from ._vedo_offscreen import frame_vedo
        return frame_vedo(scene, resolve_style(style),
                          resolution=resolution, filepath=filepath)

    from ._matplotlib import frame_mpl
    style_obj = resolve_style(style)
    fig, axs = frame_mpl(scene, style_obj,
                         figsize=figsize, show=show, ax=ax)
    if filepath is not None:
        fig.savefig(filepath, dpi=style_obj.dpi,
                    facecolor=fig.get_facecolor())
    return fig, axs


def sequence(
    bvh: Bvh,
    *,
    n_poses: int = 8,
    frames: slice | tuple[int, int] | None = None,
    layout: str = "offset",
    style: Style | str = "paper",
    centered: str = "world",
    camera: str | tuple[float, float] | None = None,
    trajectory: bool = True,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """The motion-paper sequence still: sampled poses in one figure,
    with lightness encoding time (lighter = earlier, the TEMOS
    convention).

    Parameters
    ----------
    bvh : Bvh
        A single BVH object (sequence figures are single-skeleton).
    n_poses : int, optional
        Number of equidistantly sampled poses (default 8).
    frames : slice or (start, stop), optional
        Restrict sampling to a frame range. Default: the whole clip.
    layout : str, optional
        ``"offset"`` (default): poses at their world positions —
        locomotion spreads them left to right; drawn orthographic from
        the side with equal-scale non-cubic bounds so the travel fills
        the frame. ``"overlay"``: poses superimposed (per-pose
        horizontal root-centering), seen from the front in the style's
        projection (perspective unless the style asks otherwise) — the
        right mode for in-place motion; on locomotion the legs tangle.
    style : Style or str, optional
        Visual styling preset or instance. Default ``"paper"``.
    centered : str, optional
        Centering mode for the underlying coords (default ``"world"``).
    camera : str, (float, float), or None, optional
        ``None`` (default) picks the layout's natural view: ``"side"``
        for offset, ``"front"`` for overlay. Presets and explicit
        ``(azimuth, elevation)`` tuples override it.
    trajectory : bool, optional
        Draw the dashed root trace on the floor (offset layout only).
        Default ``True``.
    figsize : (float, float), optional
        Figure size; defaults to a wide figure for offset layout.
    show : bool, optional
        Call ``plt.show()``. Default ``False``.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to draw into.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if isinstance(bvh, list):
        raise ValueError(
            "sequence() takes a single Bvh — multi-skeleton sequence "
            "figures are not supported.")
    _VALID_LAYOUTS = {"offset", "overlay"}
    if layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"Unknown layout {layout!r}. "
            f"Choose from: {sorted(_VALID_LAYOUTS)}")

    if camera is None:
        camera = "side" if layout == "offset" else "front"

    scene = _prepare(bvh, None, centered, camera, None)
    sample_frames = _resolve_sample_frames(
        scene.num_frames, n_poses, frames)

    from ._matplotlib import sequence_mpl
    return sequence_mpl(
        scene, resolve_style(style), sample_frames, layout,
        trajectory=trajectory, figsize=figsize, show=show, ax=ax)


def render(
    bvh: Bvh | list[Bvh],
    filepath: str | Path = Path("./anim.mp4"),
    *,
    style: Style | str = "paper",
    centered: str = "world",
    labels: list[str] | None = None,
    fps: float | None = None,
    backend: str = "auto",
    camera: str | tuple[float, float] = "front",
    turntable_period: float | None = None,
    resolution: tuple[int, int] = (1920, 1080),
    sync: str = "truncate",
    follow: bool = False,
    ghost: int = 0,
    trajectory: bool = False,
    frame_counter: bool = False,
    match_fps: str | None = None,
    codec: str = "auto",
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
    style : Style or str, optional
        Visual styling: a preset name (``"paper"``, ``"debug"``,
        ``"dark"``) or a :class:`Style` instance, e.g.
        ``Style("paper", floor=None)``. Axes visibility follows
        ``style.axes`` (the former ``show_axis=True`` is
        ``Style("paper", axes="full")``).
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Labels for each skeleton when comparing.
    fps : float, optional
        Frames per second (fractional rates like 119.88 are fine).
        ``None`` (default) uses the BVH frame rate, which then must be
        set: a clip whose ``frame_time`` is 0 (unset, as on a Bvh built
        in memory) raises ``ValueError`` unless ``fps`` is given.
    backend : str, optional
        ``"auto"`` (default), ``"opencv"``, or ``"matplotlib"``.
        Under ``"auto"``, formats OpenCV cannot write (``.gif``,
        ``.webp``, ``.apng``, ``.html``) always use matplotlib.
    camera : str or (float, float), optional
        Camera preset (``"front"``, ``"side"``, ``"top"``), the
        special ``"turntable"`` (an orbit starting from the front view,
        by default one full 360-degree orbit over the clip duration;
        see ``turntable_period``), or an
        ``(azimuth_deg, elevation_deg)`` tuple. Default ``"front"``.
        The vedo backend has no turntable.
    turntable_period : float, optional
        Seconds per revolution of ``camera="turntable"``; any other
        camera raises ``ValueError``. ``None`` (default) makes one
        orbit over the clip, so the speed follows the clip's length
        (82 degrees per second on the 4.4 s bundled walk). A period
        sets the speed instead. Shorter than the clip, the camera
        makes several orbits, the last one possibly partial, while the
        clip plays once. Longer, the clip loops: it is played again
        from its first frame until the orbit completes, so the video
        lasts one period, rounded to the nearest frame, and the last
        pass of the clip may be cut short. Rounding up instead would
        always complete the orbit, but a period a hair over a whole
        number of frames (as a rate read back from a frame time often
        gives) would then end on a frame showing nearly the first view
        again, which a looping GIF shows twice in a row.
        The seconds are seconds of the video at the rate it is written
        at: ``fps``, or the clip's own rate by default, except that the
        matplotlib backend writes whole frames per second (``fps=29.5``
        is written at 30) and an ``.html`` page whole milliseconds per
        frame. GIF frame durations are further quantized to 10 ms by
        the format, so a GIF may play slightly faster than written.
        They are not seconds of the clip: the two differ when ``fps``
        plays the clip faster or slower than recorded, and the period
        is the speed the viewer sees. A clip whose ``frame_time`` is 0
        (unset) is therefore timed by the ``fps`` it needs anyway. This
        differs from ``ghost`` spacing and ``follow`` smoothing, which
        are seconds of clip time because they describe the motion.
        A period of 2 frames of the video or less raises
        ``ValueError``: at 180 degrees a frame or more the camera looks
        frozen or turning backwards. The default orbit is not checked,
        so a clip of one or two frames still renders.
        At each loop seam the pose jumps from the clip's last frame
        back to its first, with no blending, while the camera turns on
        without a break. Ghosts, the root trace and the frame counter
        restart with the clip, so each pass is drawn as the first was
        and only the camera differs; carrying them across the seam
        would draw ghosts of the previous pass where the character no
        longer is and a trace segment from the clip's end back to its
        start. The clips of a comparison loop together, after
        ``sync`` has given them one length. A number of turns over the clip was the
        rejected alternative: it cannot slow a short clip down without
        cutting the orbit short.
    resolution : (int, int), optional
        Output resolution ``(width, height)`` in pixels.
        Default ``(1920, 1080)``. The OpenCV backend draws at
        ``style.supersample`` times this and downsamples for
        anti-aliasing; primitive sizes scale with resolution (1080p is
        the 1:1 anchor).
    sync : str, optional
        How to handle different frame counts in side-by-side comparison:
        ``"truncate"`` (default) stops at the shortest clip;
        ``"pad"`` continues to the longest clip (shorter clips freeze
        on their last frame).
    follow : bool, optional
        If ``True``, the camera turns with each skeleton's heading, so
        the view orbits with the character: the first frame is seen
        from the preset, later frames turned by the rotation of the
        body's left-to-right axis about the up axis since then. That
        axis also sways with every stride, so its rotation is smoothed
        with a centred Gaussian window of 0.5 s standard deviation
        (reaching ±1.5 s) of clip time, from the clip's own
        ``frame_time``, or from ``fps`` when that is 0 (unset). The
        camera follows a turn in place, which the direction of travel
        would not, and follows a turn without the lag a window over
        past frames only would have, at the cost of starting a little
        before the character does. The rotation is unwrapped, so a
        turn past half a turn is followed all the way round, and a
        frame whose left-to-right axis cannot be measured takes its
        heading from the measured frames around it (interpolated, not
        held at the last one, so the camera does not step when
        measurement resumes); if the first frame cannot be measured,
        there is no reference and the camera stays fixed. The first
        and last frames keep their heading, so the net turn is exact,
        and the smoothing weakens toward them. Only affects preset
        cameras (``"front"``, ``"side"``, ``"top"``); custom
        ``(azimuth, elevation)`` tuples are fixed and ignore
        ``follow``. Default ``False`` (stable camera).
    ghost : int, optional
        Number of faded trailing poses drawn behind the live skeleton
        (default 0 — none). Spacing between ghosts is
        ``style.ghost_spacing`` seconds of clip time, taken from each
        clip's own ``frame_time``, not playback time from ``fps``; the
        two differ when ``fps`` plays the clip faster or slower than
        recorded, and clip time keeps the trail the same poses at any
        playback speed. Ghosts therefore need every clip's frame rate
        set, even when ``fps`` is given. Older ghosts fade further
        toward the background.
    trajectory : bool, optional
        Draw the root trace on the floor, growing with playback
        (default ``False``).
    frame_counter : bool, optional
        Stamp a ``Frame f/F`` counter in the corner (OpenCV backend
        only), counting frames of the clip: a looped turntable
        (``turntable_period``) counts from 0 again on each pass.
        Default ``False`` — publication output never stamps
        text; pass ``True`` to restore the pre-0.9.0 counter.
    match_fps : str or None, optional
        How to handle clips with different frame rates in side-by-side
        rendering.  ``None`` (default) emits a warning but does not
        resample.  ``"lowest"`` resamples all clips to the lowest frame
        rate.  ``"highest"`` resamples all clips to the highest frame rate
        (using SLERP interpolation for added frames). Resampling starts
        from each clip's own rate, so a clip whose ``frame_time`` is 0
        (unset) raises ``ValueError`` under ``"lowest"`` or
        ``"highest"``, even when ``fps`` is given; under ``None`` the
        warning names it.
    codec : str, optional
        Video codec for ``.mp4``/``.mov``/``.avi`` output (invalid for
        other formats). ``"auto"`` (default) writes H.264 through the
        system ``ffmpeg`` executable when one is on PATH, and falls
        back to OpenCV's MPEG-4 Part 2 (``mp4v``) otherwise — OpenCV
        builds cannot encode H.264 themselves (patent licensing). The
        two diverge in where the file plays: H.264 plays everywhere
        (browsers, VSCode, notebook embeds, GitHub); ``mp4v`` only in
        desktop players such as VLC or mpv. Pass ``"h264"`` when you
        need the guarantee (raises with an install hint if no ffmpeg
        is found) or ``"mpeg4"`` to force the OpenCV writer. The
        matplotlib backend always writes H.264 (its mp4 writer *is*
        ffmpeg), so ``"mpeg4"`` is rejected there.

    Returns
    -------
    Path
        The path to the written file.
    """
    clips = as_clip_list(bvh)
    filepath = Path(filepath)
    _validate_sync(sync)
    pad = sync == "pad"
    style_obj = resolve_style(style)
    if not (isinstance(ghost, int) and ghost >= 0):
        raise ValueError(f"ghost must be an integer >= 0, got {ghost!r}")

    from ._opencv import VIDEO_CODECS
    if codec not in VIDEO_CODECS:
        raise ValueError(
            f"Unknown codec {codec!r}. Choose from: {sorted(VIDEO_CODECS)}")
    if codec != "auto" and filepath.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(
            f"codec= applies to video containers "
            f"({sorted(_VIDEO_EXTENSIONS)}), not {filepath.suffix!r}.")

    # "turntable" is a camera *motion*, not an angle: orbit from the
    # front view. It overrides follow (both prescribe the azimuth).
    turntable = camera == "turntable"
    if turntable_period is not None:
        if not turntable:
            raise ValueError(
                f"turntable_period= sets the speed of camera='turntable', "
                f"not of camera={camera!r}.")
        if not (math.isfinite(turntable_period) and turntable_period > 0):
            raise ValueError(
                f"turntable_period must be a positive number of seconds, "
                f"got {turntable_period!r}.")
    if turntable:
        camera = "front"

    backend_name = _resolve_render_backend(backend, filepath.suffix.lower())

    # Handle frame-rate mismatch before computing FK coordinates
    _require_frame_rates(clips, "render", fps, ghost, match_fps)
    clips = _match_frame_rates(clips, match_fps)

    scene = _prepare(clips, None, centered, camera, labels, pad=pad)

    actual_fps = _resolve_fps(fps, scene.frame_time)

    # The one value the backends hand to the viewport. Turntable
    # overrides follow — both prescribe the azimuth. A custom (azim,
    # elev) tuple means the camera is fixed; follow is a no-op in that
    # case because there's no orientation to track.
    motion: str | Turntable
    if turntable:
        from ._viewport import Turntable
        if turntable_period is None:
            period = float(scene.num_frames)
        else:
            video_fps = _written_fps(
                backend_name, filepath.suffix, actual_fps)
            period = turntable_period * video_fps
            if period <= _MIN_TURNTABLE_FRAMES:
                shortest = _MIN_TURNTABLE_FRAMES / video_fps
                raise ValueError(
                    f"turntable_period must be longer than {shortest:g} s "
                    f"at {video_fps:g} fps, the rate the video is written "
                    f"at ({_MIN_TURNTABLE_FRAMES} frames), got "
                    f"{turntable_period!r}: a camera turning 180 degrees "
                    f"or more a frame looks frozen or turning backwards.")
        # Rounded, not ceiled: a period of 360.0000001 frames (a rate
        # read back from a frame time) must not add a 361st frame.
        video_frames = round(period)
        if video_frames > scene.num_frames:
            scene = scene.looped(video_frames)
        motion = Turntable(period=period)
    elif follow and not isinstance(camera, tuple):
        motion = "follow"
    else:
        motion = "fixed"

    if (backend == "auto" and backend_name == "matplotlib"
            and filepath.suffix.lower() not in _MPL_ONLY_EXTENSIONS):
        warnings.warn(
            "OpenCV not found for fast rendering. "
            "Install with: pip install pybvh[opencv]. "
            "Falling back to matplotlib (slower).",
            stacklevel=2)

    if backend_name == "vedo":
        if not _module_importable("vedo"):
            raise ImportError(
                "vedo backend requires vedo. "
                "Install with: pip install pybvh[viewer]")
        unsupported = []
        if motion != "fixed":
            unsupported.append("follow/turntable cameras")
        if ghost:
            unsupported.append("ghost trails")
        if trajectory:
            unsupported.append("trajectory traces")
        if unsupported:
            raise ValueError(
                f"The vedo render backend does not support "
                f"{', '.join(unsupported)}. Use backend='opencv' or "
                f"'matplotlib' for those.")
        from ._vedo_offscreen import render_vedo
        return render_vedo(
            scene, style_obj, filepath, actual_fps, resolution,
            codec=codec)

    if backend_name == "opencv":
        if not _module_importable("cv2"):
            raise ImportError(
                "OpenCV backend requires opencv-python. "
                "Install with: pip install pybvh[opencv]")
        from ._opencv import render_opencv
        return render_opencv(
            scene, style_obj, filepath, actual_fps, resolution,
            motion=motion,
            frame_counter=frame_counter,
            ghost=ghost, trajectory=trajectory, codec=codec)

    else:  # matplotlib
        if codec == "mpeg4":
            raise ValueError(
                "codec='mpeg4' is only available on the OpenCV and "
                "vedo backends — the matplotlib backend writes video "
                "through ffmpeg, which encodes H.264.")
        from ._matplotlib import render_mpl
        return render_mpl(
            scene, style_obj, filepath, actual_fps,
            motion=motion,
            resolution=resolution,
            ghost=ghost, trajectory=trajectory)


def play(
    bvh: Bvh | list[Bvh],
    *,
    style: Style | str = "paper",
    centered: str = "world",
    labels: list[str] | None = None,
    fps: float | None = None,
    backend: str = "auto",
    camera: str | tuple[float, float] = "front",
    sync: str = "truncate",
    resolution: tuple[int, int] = (960, 540),
    quality: str = "high",
    frame_counter: bool = False,
    match_fps: str | None = None,
    spacing: float | str = "auto",
) -> None:
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
    style : Style or str, optional
        Visual styling: a preset name (``"paper"``, ``"debug"``,
        ``"dark"``) or a :class:`Style` instance. Applies fully to the
        matplotlib and OpenCV-notebook fallbacks; the k3d and vedo
        interactive viewers apply Style's look fields.
    centered : str, optional
        Centering mode: ``"world"`` (default), ``"skeleton"``, or ``"first"``.
    labels : list[str], optional
        Labels for each skeleton when comparing.
    fps : float, optional
        Frames per second (fractional rates like 119.88 are fine).
        ``None`` (default) uses the BVH frame rate, capped at 30 for
        the k3d and matplotlib backends (via frame subsampling) —
        notebook widgets and matplotlib windows can't keep up with
        high frame rates. That rate then must be set: a clip whose
        ``frame_time`` is 0 (unset, as on a Bvh built in memory)
        raises ``ValueError`` unless ``fps`` is given.
    backend : str, optional
        ``"auto"`` (default), ``"k3d"``, ``"vedo"``, ``"opencv"``
        (the notebook inline-video fallback — nameable so a backend
        auto-selection picked can be pinned), or ``"matplotlib"``.
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
    frame_counter : bool, optional
        Stamp a ``Frame f/F`` counter on the OpenCV notebook preview
        (default ``False``; pass ``True`` to restore the pre-0.9.0
        counter). Ignored by the other backends.
    match_fps : str or None, optional
        How to handle clips with different frame rates.  ``None``
        (default) emits a warning.  ``"lowest"`` or ``"highest"``
        resamples all clips to match. Resampling starts from each
        clip's own rate, so a clip whose ``frame_time`` is 0 (unset)
        raises ``ValueError`` under ``"lowest"`` or ``"highest"``, even
        when ``fps`` is given; under ``None`` the warning names it.
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
    None
        All backends display or open their viewer as a side effect.
    """
    clips = as_clip_list(bvh)

    valid_backends = {"auto", "k3d", "vedo", "opencv", "matplotlib"}
    if backend not in valid_backends:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Choose from: {sorted(valid_backends)}")
    if backend == "opencv":
        if not _module_importable("cv2"):
            raise ImportError(
                "The opencv play backend requires opencv-python. "
                "Install with: pip install pybvh[opencv]")
        if not _detect_notebook():
            raise ValueError(
                "backend='opencv' plays an inline video and only works "
                "inside a Jupyter notebook. In a script, use "
                "backend='vedo' (interactive window) or render() to a "
                "file instead.")

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
    style_obj = resolve_style(style)

    # Handle frame-rate mismatch before computing FK coordinates
    _require_frame_rates(clips, "play", fps, match_fps=match_fps)
    clips = _match_frame_rates(clips, match_fps)

    scene = _prepare(clips, None, centered, camera, labels, pad=pad)

    actual_fps = _resolve_fps(fps, scene.frame_time)

    backend_name, tier = _resolve_play_backend(backend)

    # --- Warnings (auto mode only, tier > 0) ---
    # Gate the install hint on vedo actually being missing: on a
    # headless display-less machine vedo may well be installed (the
    # fallback is about the display, not the install).
    if tier >= 2 and not _module_importable("vedo"):
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
    if (fps is None
            and backend_name not in ("opencv_notebook", "vedo")
            and actual_fps > _PLAY_MAX_FPS):
        subsample_step = math.ceil(actual_fps / _PLAY_MAX_FPS)
        scene = scene.subsampled(subsample_step)
        actual_fps = 1.0 / scene.frame_time

    # --- world_up consistency check (all backends) ---
    _warn_world_up_mismatch(clips)

    # --- Dispatch ---
    # For single-scene backends (vedo, k3d) there can only be ONE camera
    # and ONE bounding box. We apply lateral spacing so skeletons don't
    # overlap; the backends make one viewport of the spread scene.
    if backend_name == "k3d":
        try:
            import k3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "k3d backend requires k3d and ipywidgets. "
                "Install with: pip install pybvh[interactive]")
        from ._k3d import play_k3d
        play_k3d(_spread_for_single_scene(scene, spacing, centered),
                 style_obj, actual_fps)
        return None

    elif backend_name == "vedo":
        try:
            import vedo  # noqa: F401
        except ImportError:
            raise ImportError(
                "vedo backend requires vedo. "
                "Install with: pip install pybvh[viewer]")
        from ._vedo import play_vedo
        play_vedo(_spread_for_single_scene(scene, spacing, centered),
                  style_obj, actual_fps, quality=quality)
        return None

    elif backend_name == "opencv_notebook":
        import tempfile
        from ._opencv import render_opencv
        from IPython.display import display, Video  # type: ignore[import-untyped]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        render_opencv(scene, style_obj, tmp_path, actual_fps, resolution,
                      frame_counter=frame_counter)

        display(Video(str(tmp_path), embed=True, mimetype="video/mp4"))
        tmp_path.unlink(missing_ok=True)
        return None

    else:  # matplotlib
        from ._matplotlib import play_mpl
        play_mpl(scene, style_obj, actual_fps,
                 in_notebook=_detect_notebook())
        return None


def trajectory(
    bvh: Bvh | list[Bvh],
    *,
    style: Style | str = "paper",
    centered: str = "world",
    labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    facing_arrows: bool = False,
    tight: bool = False,
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
    facing_arrows : bool, optional
        If True, overlay small arrowheads along each skeleton's path
        showing the character's facing direction at ~10 evenly-spaced
        frames.  Arrows use the same color as the trajectory line and
        are sized at ~8 % of the path's span.  Default False.
    tight : bool, optional
        If False (default), the axis range matches the full horizontal
        extent of the skeleton across all joints and frames — the same
        bounding box ``bvh.play()`` uses.  Keeps the motion scale
        honest relative to the character's body so a near-stationary
        clip doesn't get auto-zoomed into looking like a large walk.
        If True, axes auto-scale to just the root path — gives maximum
        detail on the path shape but can exaggerate small motions.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    clips = as_clip_list(bvh)
    scene = _prepare(clips, None, centered, "front", labels)

    # trajectory_mpl() computes its own per-skeleton horizontal axes
    # internally (drop each skeleton's own up axis).
    from ._matplotlib import trajectory_mpl
    return trajectory_mpl(
        scene, resolve_style(style), figsize=figsize, show=show, ax=ax,
        facing_arrows=facing_arrows, tight=tight)
