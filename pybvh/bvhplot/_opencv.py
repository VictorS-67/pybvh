"""OpenCV fast render backend.

Renders skeleton animations to video files using orthographic 2D
projection and OpenCV drawing primitives. Orders of magnitude faster
than matplotlib for video export.

Requires ``opencv-python >= 4.5``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TYPE_CHECKING

from ._common import (
    Scene,
    SkeletonView,
    Style,
    build_view_matrix,
    compute_follow_azimuths,
    ortho_project,
    bone_colors_for_view,
    PALETTE_RGB,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from ..bvh import Bvh

# RGB is canonical in _common; the channel flip for OpenCV's BGR
# drawing API happens here, at this backend's border.
PALETTE_BGR = [(b, g, r) for (r, g, b) in PALETTE_RGB]


def _to_bgr(color: object) -> tuple[int, int, int]:
    """Any matplotlib-parseable color -> OpenCV BGR uint8 tuple."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)  # type: ignore[arg-type]
    return (int(b * 255), int(g * 255), int(r * 255))


def _blend_bgr(
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Emulate alpha compositing (cv2 has none) by pre-blending."""
    return tuple(int(f * alpha + b * (1.0 - alpha))
                 for f, b in zip(fg, bg))  # type: ignore[return-value]


def _draw_floor_opencv(
    img: npt.NDArray[np.uint8],
    view: SkeletonView,
    style: Style,
    view_matrix: npt.NDArray[np.float64],
    panel_w: int,
    h: int,
    x_offset: int,
    bg_bgr: tuple[int, int, int],
    fixed_view_half: tuple[float, float] | None,
) -> None:
    """Project and draw the ground plane into one panel.

    Mirrors the matplotlib floor geometry (same extents, same grays,
    ``floor_alpha`` emulated by pre-blending toward the background).
    """
    import cv2

    up = view.up_index
    ground = [i for i in range(3) if i != up]
    ext = view.half_span * 1.8
    y = view.floor_height
    dark = style.background not in ("white", "#FFFFFF", "#ffffff")

    def project(world_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        pts = ortho_project(
            world_pts, view_matrix, view.center, view.half_span,
            (panel_w, h), fixed_view_half=fixed_view_half)
        pts[:, 0] += x_offset
        return pts

    def quad(c0: float, c1: float, s: float) -> npt.NDArray[np.float64]:
        pts = np.zeros((4, 3))
        for k, (d0, d1) in enumerate(((-s, -s), (s, -s), (s, s), (-s, s))):
            pts[k, ground[0]] = c0 + d0
            pts[k, ground[1]] = c1 + d1
            pts[k, up] = y
        return pts

    c0 = float(view.center[ground[0]])
    c1 = float(view.center[ground[1]])

    if style.floor == "solid":
        face = _blend_bgr(_to_bgr("#2A2E36" if dark else "#E8E8EC"),
                          bg_bgr, style.floor_alpha)
        cv2.fillPoly(img, [project(quad(c0, c1, ext))], face,
                     lineType=cv2.LINE_AA)
    elif style.floor == "checker":
        n = 8
        s = ext / n
        shades = (("#2A2E36", "#1E2127") if dark else ("#EDEDF1", "#DCDCE3"))
        for i in range(-n, n):
            for j in range(-n, n):
                face = _blend_bgr(_to_bgr(shades[(i + j) % 2]), bg_bgr,
                                  style.floor_alpha)
                sq = quad(c0 + (i + 0.5) * s, c1 + (j + 0.5) * s, s / 2)
                cv2.fillPoly(img, [project(sq)], face, lineType=cv2.LINE_AA)
    elif style.floor == "grid":
        color = _blend_bgr(_to_bgr("#3A3F4A" if dark else "#C8C8D0"),
                           bg_bgr, 0.8)
        n = 10
        s = ext / n
        for i in range(-n, n + 1):
            for horiz in (True, False):
                seg = np.zeros((2, 3))
                if horiz:
                    seg[0, ground[0]], seg[1, ground[0]] = c0 - ext, c0 + ext
                    seg[:, ground[1]] = c1 + i * s
                else:
                    seg[:, ground[0]] = c0 + i * s
                    seg[0, ground[1]], seg[1, ground[1]] = c1 - ext, c1 + ext
                seg[:, up] = y
                p = project(seg)
                cv2.line(img, tuple(p[0]), tuple(p[1]), color, 1,
                         cv2.LINE_AA)


# Extensions this backend can actually write: video containers via
# cv2.VideoWriter, plus GIF via a dedicated Pillow path.
_OPENCV_EXTENSIONS = {'.mp4', '.mov', '.avi', '.gif'}


def _compute_fixed_view_halves_for_follow(
    follow_azimuths: list[npt.NDArray[np.float64]],
    elevations: list[float],
    up_axes: list[str],
    half_spans: list[float],
) -> list[tuple[float, float]]:
    """For follow mode: precompute per-skeleton view-space half extents
    that stay constant across the whole animation.

    At every frame, follow rotates the camera around the world-up axis
    (via a signed rotation delta in azimuth). The projection of the
    cubic bounding box onto the screen varies with that rotation: it
    is widest at 45° off-axis and narrowest axis-aligned. If we let
    ``ortho_project`` compute the scale per frame from the current
    view matrix, the scale oscillates and the character appears to
    zoom in and out. To avoid this we compute the MAX (view_half_u,
    view_half_v) across every frame ahead of time and reuse those as
    a fixed scale at render time.
    """
    result: list[tuple[float, float]] = []
    for azimuths_per_frame, el, ua, half_span in zip(
            follow_azimuths, elevations, up_axes, half_spans):
        corners = np.array([[sx, sy, sz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                           dtype=np.float64) * half_span

        max_u = 0.0
        max_v = 0.0
        for az_f in azimuths_per_frame:
            vm = build_view_matrix(az_f, el, ua)
            cv_corners = corners @ vm.T
            u = float(np.abs(cv_corners[:, 0]).max())
            v = float(np.abs(cv_corners[:, 1]).max())
            if u > max_u:
                max_u = u
            if v > max_v:
                max_v = v
        result.append((max_u, max_v))
    return result


def _draw_skeletons_on_frame(
    img: npt.NDArray[np.uint8],
    frame_idx: int,
    scene: Scene,
    style: Style,
    view_matrices: list[npt.NDArray[np.float64]],
    panel_w: int,
    h: int,
    bg_bgr: tuple[int, int, int],
    fixed_view_halves: list[tuple[float, float]] | None = None,
    ghost: int = 0,
    trajectory: bool = False,
) -> None:
    """Draw all skeletons for one frame onto *img* (mutates in place).

    Each skeleton is projected with its own view matrix, so skeletons
    with different forward/up axes all render correctly side by side.

    When ``fixed_view_halves`` is provided, each skeleton's projection
    uses that pre-computed ``(view_half_u, view_half_v)`` instead of
    deriving it from the current view matrix. This keeps the character
    size constant across frames when the camera rotates (follow mode).

    Joint dots are always drawn in this backend — at raster resolution
    they are load-bearing for line joins (matplotlib omits them when
    ``style.joint_markers`` is off). With markers off they are small
    and bone-colored (the pre-0.9.0 look); with markers on they use
    ``style.joint_color``.

    ``ghost`` faded trailing poses draw behind the live skeleton
    (oldest first); ``trajectory`` draws the root trace on the floor up
    to the current frame (solid thin line — cv2 has no dashes).
    """
    import cv2

    n_skeletons = scene.num_skeletons
    thickness = max(1, round(style.bone_width))

    for s, view in enumerate(scene.views):
        frame_data = view.coords[frame_idx]
        view_matrix = view_matrices[s]
        fixed = fixed_view_halves[s] if fixed_view_halves is not None else None
        x_offset = s * panel_w

        if style.floor is not None:
            _draw_floor_opencv(
                img, view, style, view_matrix, panel_w, h, x_offset,
                bg_bgr, fixed)

        def project(world_pts):
            pts = ortho_project(
                world_pts, view_matrix, view.center, view.half_span,
                (panel_w, h), fixed_view_half=fixed)
            pts[:, 0] += x_offset
            return pts

        bone_colors = [
            _to_bgr(c)
            for c in bone_colors_for_view(view, style, s, n_skeletons)]

        if trajectory:
            path = view.coords[:frame_idx + 1, 0, :].copy()
            path[:, view.up_index] = view.floor_height
            if len(path) >= 2:
                trace_color = _blend_bgr(_to_bgr("#7A8090"), bg_bgr, 0.9)
                cv2.polylines(img, [project(path)], False, trace_color, 1,
                              cv2.LINE_AA)

        if ghost > 0:
            lag = max(1, round(
                style.ghost_spacing / view.bvh.frame_time))
            weights = np.linspace(0.32, 0.15, ghost)
            ghost_thickness = max(1, round(style.bone_width * 0.75))
            for j in reversed(range(ghost)):     # oldest first
                gf = frame_idx - (j + 1) * lag
                if gf < 0:
                    continue
                gpts = project(view.coords[gf])
                for (p_idx, c_idx), color in zip(view.bones, bone_colors):
                    faded = _blend_bgr(color, bg_bgr, float(weights[j]))
                    cv2.line(img, tuple(gpts[p_idx]), tuple(gpts[c_idx]),
                             faded, ghost_thickness, cv2.LINE_AA)

        pts_2d = project(frame_data)

        for (p_idx, c_idx), color in zip(view.bones, bone_colors):
            pt1 = (int(pts_2d[p_idx, 0]), int(pts_2d[p_idx, 1]))
            pt2 = (int(pts_2d[c_idx, 0]), int(pts_2d[c_idx, 1]))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

        if style.joint_markers:
            joint_bgr = _to_bgr(style.joint_color)
            for pt in pts_2d:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, joint_bgr,
                           -1, cv2.LINE_AA)
        else:
            # bone-colored r=4 dots: a joint takes the color of the bone
            # whose child it is (falls back to the first bone's color).
            joint_color_by_node = dict(
                (c_idx, col)
                for (_p, c_idx), col in zip(view.bones, bone_colors))
            default = bone_colors[0] if bone_colors else (0, 0, 0)
            for j, pt in enumerate(pts_2d):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4,
                           joint_color_by_node.get(j, default), -1,
                           cv2.LINE_AA)

        if view.label is not None:
            label_color = bone_colors[0] if bone_colors else (0, 0, 0)
            cv2.putText(
                img, view.label, (x_offset + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)

    if n_skeletons > 1:
        for s in range(1, n_skeletons):
            x = s * panel_w
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)


def _generate_frames(
    scene: Scene,
    style: Style,
    resolution: tuple[int, int],
    *,
    follow: bool = False,
    frame_counter: bool = True,
    ghost: int = 0,
    trajectory: bool = False,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield one rendered BGR image ``(H, W, 3)`` per animation frame.

    Shared by the video and GIF paths of :func:`render_opencv` — only
    the sink (``cv2.VideoWriter`` vs Pillow) differs between them.

    Parameters
    ----------
    scene : Scene
        Prepared visualization (per-view bounding boxes and cameras).
    style : Style
        Visual styling; ``style.axes == "full"`` draws the per-panel
        axis indicator (the pre-0.9.0 ``show_axis``).
    resolution : (int, int)
        ``(width, height)`` in pixels.
    follow : bool, optional
        If ``True``, per-frame view matrices track each skeleton's
        rotation (continuous azimuth tracking around ``world_up``).
    frame_counter : bool, optional
        Draw a ``Frame f/F`` counter in the bottom-right corner.
        Default ``True`` (the GIF path opts out to preserve its
        historical counter-free output).
    """
    import cv2

    w, h = resolution
    bg_bgr = _to_bgr(style.background)
    num_frames = scene.num_frames
    n_skeletons = scene.num_skeletons
    panel_w = w // n_skeletons if n_skeletons > 1 else w

    base_view_matrices = [
        build_view_matrix(v.azimuth, v.elevation, v.up_axis)
        for v in scene.views]

    # Follow mode: per-frame azimuths precomputed once per skeleton,
    # plus the MAX view-space half extents across every frame so the
    # projection scale stays constant and the character doesn't zoom
    # in and out as the camera orbits.
    follow_azimuths: list[npt.NDArray[np.float64]] | None = None
    fixed_view_halves: list[tuple[float, float]] | None = None
    if follow:
        follow_azimuths = [
            compute_follow_azimuths(v.bvh, v.coords, v.azimuth)
            for v in scene.views]
        fixed_view_halves = _compute_fixed_view_halves_for_follow(
            follow_azimuths,
            [v.elevation for v in scene.views],
            [v.up_axis for v in scene.views],
            [v.half_span for v in scene.views])

    for f in range(num_frames):
        if follow_azimuths is not None:
            view_matrices = [
                build_view_matrix(az_per_frame[f], v.elevation, v.up_axis)
                for az_per_frame, v
                in zip(follow_azimuths, scene.views)]
        else:
            view_matrices = base_view_matrices

        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = bg_bgr

        _draw_skeletons_on_frame(
            img, f, scene, style, view_matrices, panel_w, h, bg_bgr,
            fixed_view_halves=fixed_view_halves,
            ghost=ghost, trajectory=trajectory)

        if frame_counter:
            fc_text = f"Frame {f}/{num_frames - 1}"
            fc_x = max(5, w - 200)
            cv2.putText(
                img, fc_text,
                (fc_x, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                cv2.LINE_AA)

        if style.axes == "full":
            for s, v in enumerate(scene.views):
                _draw_axis_indicator(
                    img, view_matrices[s], v.up_axis,
                    panel_w, h, panel_idx=s)

        yield img


def render_opencv(
    scene: Scene,
    style: Style,
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    *,
    follow: bool = False,
    ghost: int = 0,
    trajectory: bool = False,
) -> Path:
    """Render skeleton animation to a video or GIF file using OpenCV.

    Each panel uses its own bounding box and camera so that mixed-up-axis
    side-by-side comparisons render correctly. If ``follow`` is True, the
    per-panel view matrices are recomputed every frame so each camera
    tracks its skeleton's current facing direction.

    Frame generation is shared with the GIF path via
    :func:`_generate_frames`; this function only picks the sink.

    Parameters
    ----------
    scene : Scene
        Prepared visualization (per-view bounding boxes and cameras).
    style : Style
        Visual styling (colors, floor, background; ``axes="full"``
        draws the per-panel axis indicator).
    filepath : Path
        Output file path. Must end in one of ``.mp4``, ``.mov``,
        ``.avi``, or ``.gif``.
    fps : float
        Frames per second.
    resolution : (int, int)
        ``(width, height)`` in pixels.
    follow : bool, optional
        If ``True``, recompute view matrices each frame so the camera
        follows the character's orientation. Default ``False``.

    Returns
    -------
    Path
        The path to the written video file.

    Raises
    ------
    ValueError
        If the file extension is not one this backend can write
        (``.html``/``.webp``/``.apng`` need the matplotlib backend).
    """
    ext = filepath.suffix.lower()
    if ext not in _OPENCV_EXTENSIONS:
        raise ValueError(
            f"The OpenCV backend cannot write {ext!r} files. "
            f"Supported extensions: {sorted(_OPENCV_EXTENSIONS)}. "
            f"Use backend='matplotlib' for other formats.")

    # Pillow sink for GIF output (cv2.VideoWriter doesn't support GIF).
    if ext == '.gif':
        frames = _generate_frames(
            scene, style, resolution, follow=follow, frame_counter=False,
            ghost=ghost, trajectory=trajectory)
        return _render_gif(frames, filepath, fps)

    frames = _generate_frames(scene, style, resolution, follow=follow,
                              ghost=ghost, trajectory=trajectory)

    writer = _open_writer(filepath, fps, resolution)
    for img in frames:
        writer.write(img)  # type: ignore[attr-defined]
    writer.release()  # type: ignore[attr-defined]
    return filepath


def _open_writer(
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
) -> object:
    """Open a cv2.VideoWriter with codec fallback.

    Tries MPEG-4 first (widely supported, no noisy codec probing),
    then H.264, then XVID.
    """
    import cv2

    codecs = ['mp4v', 'avc1', 'XVID']
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, resolution)
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"Could not open video writer for {filepath}. "
        f"Tried codecs: {codecs}. Ensure OpenCV has video codec support.")


def _render_gif(
    frames: Iterator[npt.NDArray[np.uint8]],
    filepath: Path,
    fps: float,
) -> Path:
    """Pillow sink for GIF output (cv2.VideoWriter doesn't support GIF).

    Consumes the BGR frames from :func:`_generate_frames`, converting
    each to RGB for Pillow.
    """
    from PIL import Image

    duration_ms = int(1000.0 / fps)
    pil_frames = (Image.fromarray(img[:, :, ::-1]) for img in frames)
    first_frame = next(pil_frames)
    first_frame.save(
        filepath,
        save_all=True,
        append_images=pil_frames,
        duration=duration_ms,
        loop=0)

    return filepath


def _draw_axis_indicator(
    img: npt.NDArray[np.uint8],
    view_matrix: npt.NDArray[np.float64],
    up_axis: str,
    panel_w: int,
    h: int,
    panel_idx: int = 0,
) -> None:
    """Draw a small 3D axis indicator in the bottom-left corner of a panel.

    For side-by-side renders, one indicator is drawn per panel so that
    each skeleton's own camera orientation is visible.
    """
    import cv2

    x_offset = panel_idx * panel_w
    origin = np.array([x_offset + 50, h - 50])
    axis_len = 30

    axis_colors = {
        'x': (50, 50, 220),    # red
        'y': (50, 180, 50),    # green
        'z': (220, 120, 50),   # blue
    }

    for i, axis_name in enumerate('xyz'):
        direction_3d = np.zeros(3)
        direction_3d[i] = 1.0
        projected = view_matrix @ direction_3d
        end = origin + np.array([projected[0], -projected[1]]) * axis_len
        end = end.astype(int)

        cv2.line(img, tuple(origin), tuple(end),
                 axis_colors[axis_name], 2, cv2.LINE_AA)
        cv2.putText(img, axis_name, tuple(end + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    axis_colors[axis_name], 1, cv2.LINE_AA)
