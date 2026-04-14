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
    ortho_project,
    PALETTE_BGR,
)

if TYPE_CHECKING:
    from ..bvh import Bvh


def _compute_fixed_view_halves_for_follow(
    bvh_list,
    coords_list,
    centers,
    half_spans,
    azimuths,
    elevations,
    up_axes,
    base_laterals,
    up_vecs,
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
    from ..tools import _signed_rotation_delta_around_axis, _world_lateral_unit_at_frame

    n_skeletons = len(bvh_list)
    result: list[tuple[float, float]] = []
    for i, bvh_obj in enumerate(bvh_list):
        az0, el0, ua = azimuths[i], elevations[i], up_axes[i]
        lateral_0 = base_laterals[i]
        half_span = half_spans[i]

        corners = np.array([[sx, sy, sz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                           dtype=np.float64) * half_span

        max_u = 0.0
        max_v = 0.0
        num_frames = coords_list[i].shape[0]
        for f in range(num_frames):
            if lateral_0 is None:
                az_f = az0
            else:
                lateral_f = _world_lateral_unit_at_frame(
                    bvh_obj, coords_list[i][f], bvh_obj.world_up)
                if lateral_f is None:
                    az_f = az0
                else:
                    delta = _signed_rotation_delta_around_axis(
                        lateral_0, lateral_f, up_vecs[i])
                    az_f = az0 + delta
            vm = build_view_matrix(az_f, el0, ua)
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
    coords_list: list[npt.NDArray[np.float64]],
    skeleton_lines_list: list[list[tuple[int, int]]],
    view_matrices: list[npt.NDArray[np.float64]],
    per_skeleton_limits: list[tuple[npt.NDArray[np.float64], float]],
    panel_w: int,
    h: int,
    labels: list[str] | None,
    fixed_view_halves: list[tuple[float, float]] | None = None,
) -> None:
    """Draw all skeletons for one frame onto *img* (mutates in place).

    Each skeleton is projected with its own view matrix, so skeletons
    with different forward/up axes all render correctly side by side.

    When ``fixed_view_halves`` is provided, each skeleton's projection
    uses that pre-computed ``(view_half_u, view_half_v)`` instead of
    deriving it from the current view matrix. This keeps the character
    size constant across frames when the camera rotates (follow mode).
    """
    import cv2

    n_skeletons = len(coords_list)

    for s, (coords, bones) in enumerate(
            zip(coords_list, skeleton_lines_list)):
        frame_data = coords[frame_idx]
        sk_center, sk_half_span = per_skeleton_limits[s]
        view_matrix = view_matrices[s]
        fixed = fixed_view_halves[s] if fixed_view_halves is not None else None
        pts_2d = ortho_project(
            frame_data, view_matrix, sk_center, sk_half_span, (panel_w, h),
            fixed_view_half=fixed)

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
    centers: list[npt.NDArray[np.float64]],
    half_spans: list[float],
    azimuths: list[float],
    elevations: list[float],
    up_axes: list[str],
    follow: bool = False,
    camera: str | tuple[float, float] = "front",
) -> Path:
    """Render skeleton animation to video using OpenCV.

    Each panel uses its own bounding box and camera so that mixed-up-axis
    side-by-side comparisons render correctly. If ``follow`` is True, the
    per-panel view matrices are recomputed every frame so each camera
    tracks its skeleton's current facing direction.

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
        If ``True``, draw a simple axis indicator in each panel.
    skeleton_lines_list : list
        Precomputed bone index pairs per skeleton.
    centers, half_spans : list[ndarray], list[float]
        Per-skeleton bounding boxes.
    azimuths, elevations : list[float]
        Per-skeleton camera angles in degrees.
    up_axes : list[str]
        Per-skeleton vertical axis, each ``'x'``, ``'y'``, or ``'z'``.
    follow : bool, optional
        If ``True``, recompute view matrices each frame so the camera
        follows the character's orientation. Default ``False``.
    camera : str or (float, float), optional
        Camera spec passed through to ``get_camera_angles`` when
        ``follow=True``. Ignored otherwise.

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
            show_axis, skeleton_lines_list, centers, half_spans,
            azimuths, elevations, up_axes,
            follow=follow, camera=camera)

    w, h = resolution
    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)

    # Per-skeleton view matrices: each panel's camera is built from its
    # own skeleton's detected forward/up axes. In follow mode these get
    # recomputed every frame inside the loop (continuous rotation tracking).
    view_matrices = [
        build_view_matrix(az, el, ua)
        for az, el, ua in zip(azimuths, elevations, up_axes)]

    panel_w = w // n_skeletons if n_skeletons > 1 else w
    per_skeleton_limits = list(zip(centers, half_spans))

    # Follow-mode precomputation: base azim per skeleton and frame-0 lateral
    # unit vectors, used to apply a continuous rotation delta per frame.
    # Also precompute the MAX view_half_u/v across every frame so the
    # scale is constant and the character doesn't zoom in and out as
    # the camera orbits.
    fixed_view_halves: list[tuple[float, float]] | None = None
    if follow:
        from ..tools import (
            _axis_to_vector,
            _signed_rotation_delta_around_axis,
            _world_lateral_unit_at_frame,
        )
        base_laterals = [
            _world_lateral_unit_at_frame(b, coords_list[i][0], b.world_up)
            for i, b in enumerate(bvh_list)
        ]
        up_vecs = [_axis_to_vector(b.world_up) for b in bvh_list]

        fixed_view_halves = _compute_fixed_view_halves_for_follow(
            bvh_list, coords_list, centers, half_spans,
            azimuths, elevations, up_axes,
            base_laterals, up_vecs)

    # Open video writer with codec fallback
    writer = _open_writer(filepath, fps, (w, h))

    for f in range(num_frames):
        if follow:
            view_matrices = []
            frame_up_axes: list[str] = list(up_axes)
            for i, bvh_obj in enumerate(bvh_list):
                az0, el0, ua = azimuths[i], elevations[i], up_axes[i]
                lateral_0 = base_laterals[i]
                if lateral_0 is None:
                    az_f = az0
                else:
                    lateral_f = _world_lateral_unit_at_frame(
                        bvh_obj, coords_list[i][f], bvh_obj.world_up)
                    if lateral_f is None:
                        az_f = az0
                    else:
                        delta = _signed_rotation_delta_around_axis(
                            lateral_0, lateral_f, up_vecs[i])
                        az_f = az0 + delta
                view_matrices.append(build_view_matrix(az_f, el0, ua))
        else:
            frame_up_axes = up_axes

        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        _draw_skeletons_on_frame(
            img, f, coords_list, skeleton_lines_list,
            view_matrices, per_skeleton_limits, panel_w, h, labels,
            fixed_view_halves=fixed_view_halves)

        fc_text = f"Frame {f}/{num_frames - 1}"
        fc_x = max(5, w - 200)
        cv2.putText(
            img, fc_text,
            (fc_x, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        if show_axis:
            for s in range(n_skeletons):
                _draw_axis_indicator(
                    img, view_matrices[s], frame_up_axes[s],
                    panel_w, h, panel_idx=s)

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
    centers: list[npt.NDArray[np.float64]],
    half_spans: list[float],
    azimuths: list[float],
    elevations: list[float],
    up_axes: list[str],
    follow: bool = False,
    camera: str | tuple[float, float] = "front",
) -> Path:
    """Render to GIF using Pillow (cv2 doesn't support GIF)."""
    from PIL import Image

    w, h = resolution
    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)

    base_view_matrices = [
        build_view_matrix(az, el, ua)
        for az, el, ua in zip(azimuths, elevations, up_axes)]

    panel_w = w // n_skeletons if n_skeletons > 1 else w
    per_skeleton_limits = list(zip(centers, half_spans))

    duration_ms = int(1000.0 / fps)

    fixed_view_halves: list[tuple[float, float]] | None = None
    if follow:
        from ..tools import (
            _axis_to_vector,
            _signed_rotation_delta_around_axis,
            _world_lateral_unit_at_frame,
        )
        base_laterals = [
            _world_lateral_unit_at_frame(b, coords_list[i][0], b.world_up)
            for i, b in enumerate(bvh_list)
        ]
        up_vecs = [_axis_to_vector(b.world_up) for b in bvh_list]

        fixed_view_halves = _compute_fixed_view_halves_for_follow(
            bvh_list, coords_list, centers, half_spans,
            azimuths, elevations, up_axes,
            base_laterals, up_vecs)

    def _generate_frames():
        for f in range(num_frames):
            if follow:
                view_matrices = []
                for i, bvh_obj in enumerate(bvh_list):
                    az0, el0, ua = azimuths[i], elevations[i], up_axes[i]
                    lateral_0 = base_laterals[i]
                    if lateral_0 is None:
                        az_f = az0
                    else:
                        lateral_f = _world_lateral_unit_at_frame(
                            bvh_obj, coords_list[i][f], bvh_obj.world_up)
                        if lateral_f is None:
                            az_f = az0
                        else:
                            delta = _signed_rotation_delta_around_axis(
                                lateral_0, lateral_f, up_vecs[i])
                            az_f = az0 + delta
                    view_matrices.append(build_view_matrix(az_f, el0, ua))
            else:
                view_matrices = base_view_matrices

            img = np.ones((h, w, 3), dtype=np.uint8) * 255

            _draw_skeletons_on_frame(
                img, f, coords_list, skeleton_lines_list,
                view_matrices, per_skeleton_limits, panel_w, h, labels,
                fixed_view_halves=fixed_view_halves)

            if show_axis:
                for s in range(n_skeletons):
                    _draw_axis_indicator(
                        img, view_matrices[s], up_axes[s],
                        panel_w, h, panel_idx=s)

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
