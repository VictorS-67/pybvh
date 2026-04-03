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
    build_view_matrix,
    compute_unified_limits,
    ortho_project,
    PALETTE_BGR,
)

if TYPE_CHECKING:
    from ..bvh import Bvh


def _draw_skeletons_on_frame(
    img: npt.NDArray[np.uint8],
    frame_idx: int,
    coords_list: list[npt.NDArray[np.float64]],
    skeleton_lines_list: list[list[tuple[int, int]]],
    view_matrix: npt.NDArray[np.float64],
    per_skeleton_limits: list[tuple[npt.NDArray[np.float64], float]],
    panel_w: int,
    h: int,
    labels: list[str] | None,
) -> None:
    """Draw all skeletons for one frame onto *img* (mutates in place)."""
    import cv2

    n_skeletons = len(coords_list)

    for s, (coords, bones) in enumerate(
            zip(coords_list, skeleton_lines_list)):
        frame_data = coords[frame_idx]
        sk_center, sk_half_span = per_skeleton_limits[s]
        pts_2d = ortho_project(
            frame_data, view_matrix, sk_center, sk_half_span, (panel_w, h))

        x_offset = s * panel_w
        pts_2d[:, 0] += x_offset

        color = PALETTE_BGR[s % len(PALETTE_BGR)]

        for p_idx, c_idx in bones:
            pt1 = (int(pts_2d[p_idx, 0]), int(pts_2d[p_idx, 1]))
            pt2 = (int(pts_2d[c_idx, 0]), int(pts_2d[c_idx, 1]))
            cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

        for pt in pts_2d:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1,
                       cv2.LINE_AA)

        if labels and s < len(labels):
            cv2.putText(
                img, labels[s], (x_offset + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if n_skeletons > 1:
        for s in range(1, n_skeletons):
            x = s * panel_w
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)


def render_opencv(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    labels: list[str] | None,
    show_axis: bool,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
) -> Path:
    """Render skeleton animation to video using OpenCV.

    Parameters
    ----------
    bvh_list : list[Bvh]
        Skeleton objects.
    coords_list : list[ndarray]
        Spatial coordinates per skeleton, each ``(F, N, 3)``.
    filepath : Path
        Output file path.
    fps : float
        Frames per second.
    resolution : (int, int)
        ``(width, height)`` in pixels.
    labels : list[str] or None
        Labels for each skeleton.
    show_axis : bool
        If ``True``, draw simple axis indicator.
    skeleton_lines_list : list
        Precomputed bone index pairs per skeleton.
    center : ndarray (3,)
        Bounding box center.
    half_span : float
        Half side of cubic bounding box.
    azimuth, elevation : float
        Camera angles in degrees.
    up_axis : str
        ``'x'``, ``'y'``, or ``'z'``.

    Returns
    -------
    Path
        The path to the written video file.
    """
    import cv2

    ext = filepath.suffix.lower()

    # Use Pillow for GIF output (cv2.VideoWriter doesn't support GIF)
    if ext == '.gif':
        return _render_gif(
            bvh_list, coords_list, filepath, fps, resolution, labels,
            show_axis, skeleton_lines_list, center, half_span,
            azimuth, elevation, up_axis)

    w, h = resolution
    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)

    # Build view matrix once
    view_matrix = build_view_matrix(azimuth, elevation, up_axis)

    # Side-by-side: divide canvas into panels
    if n_skeletons > 1:
        panel_w = w // n_skeletons
        # Per-skeleton limits so each is centered in its own panel
        per_skeleton_limits = [
            compute_unified_limits([c]) for c in coords_list]
    else:
        panel_w = w
        per_skeleton_limits = [(center, half_span)]

    # Open video writer with codec fallback
    writer = _open_writer(filepath, fps, (w, h))

    for f in range(num_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        _draw_skeletons_on_frame(
            img, f, coords_list, skeleton_lines_list,
            view_matrix, per_skeleton_limits, panel_w, h, labels)

        fc_text = f"Frame {f}/{num_frames - 1}"
        fc_x = max(5, w - 200)
        cv2.putText(
            img, fc_text,
            (fc_x, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        if show_axis:
            _draw_axis_indicator(img, view_matrix, up_axis, w, h)

        writer.write(img)

    writer.release()
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
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, resolution)
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"Could not open video writer for {filepath}. "
        f"Tried codecs: {codecs}. Ensure OpenCV has video codec support.")


def _render_gif(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    labels: list[str] | None,
    show_axis: bool,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
) -> Path:
    """Render to GIF using Pillow (cv2 doesn't support GIF)."""
    import cv2
    from PIL import Image

    w, h = resolution
    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)
    view_matrix = build_view_matrix(azimuth, elevation, up_axis)

    if n_skeletons > 1:
        panel_w = w // n_skeletons
        per_skeleton_limits = [
            compute_unified_limits([c]) for c in coords_list]
    else:
        panel_w = w
        per_skeleton_limits = [(center, half_span)]

    duration_ms = int(1000.0 / fps)

    def _generate_frames():
        for f in range(num_frames):
            img = np.ones((h, w, 3), dtype=np.uint8) * 255

            _draw_skeletons_on_frame(
                img, f, coords_list, skeleton_lines_list,
                view_matrix, per_skeleton_limits, panel_w, h, labels)

            yield Image.fromarray(img[:, :, ::-1])

    frames_iter = _generate_frames()
    first_frame = next(frames_iter)
    first_frame.save(
        filepath,
        save_all=True,
        append_images=frames_iter,
        duration=duration_ms,
        loop=0)

    return filepath


def _draw_axis_indicator(
    img: npt.NDArray[np.uint8],
    view_matrix: npt.NDArray[np.float64],
    up_axis: str,
    w: int,
    h: int,
) -> None:
    """Draw a small 3D axis indicator in the bottom-left corner."""
    import cv2

    origin = np.array([50, h - 50])
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
