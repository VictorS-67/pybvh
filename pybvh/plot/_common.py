"""Shared helpers for all visualization backends.

Pure-data operations: skeleton topology, bounding boxes, camera math,
and orthographic projection. No plotting library imports.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bvh import Bvh

_UP_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

def get_skeleton_lines(bvh: Bvh) -> list[tuple[int, int]]:
    """Precompute (parent_node_idx, child_node_idx) pairs for bone drawing.

    Computed once per skeleton, reused every frame by all backends.

    Parameters
    ----------
    bvh : Bvh
        The BVH object containing the skeleton hierarchy.

    Returns
    -------
    lines : list of (int, int)
        Each tuple is ``(parent_index, child_index)`` into the flat
        ``nodes`` list (i.e. the same indexing used by spatial coordinates).
    """
    node_index = bvh.node_index
    lines: list[tuple[int, int]] = []
    for node in bvh.nodes[1:]:  # skip root (has no parent bone to draw)
        child_idx = node_index[node.name]
        parent_idx = node_index[node.parent.name]  # type: ignore[union-attr]
        lines.append((parent_idx, child_idx))
    return lines


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def normalize_input(
    bvh: Bvh | list[Bvh],
    frames: int | npt.NDArray[np.floating] | None,
    centered: str,
) -> tuple[list[Bvh], list[npt.NDArray[np.float64]]]:
    """Normalize single/list Bvh + frame spec into parallel lists.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects to visualize.
    frames : int, ndarray, or None
        - ``None`` or ``-1``: all frames (spatial coords for entire motion).
        - Non-negative int: single frame index.
        - 2-D array ``(N, 3)``: single frame of spatial coordinates
          (only valid when *bvh* is a single Bvh).
        - 3-D array ``(F, N, 3)``: pre-computed spatial coordinates
          (only valid when *bvh* is a single Bvh).
    centered : str
        Centering mode passed to ``bvh.get_spatial_coord()``.

    Returns
    -------
    bvh_list : list[Bvh]
        Always a list (length >= 1).
    coords_list : list[ndarray]
        Parallel list of spatial coordinates, each ``(F, N, 3)``.
    """
    # Wrap single Bvh
    if not isinstance(bvh, list):
        bvh_list = [bvh]
    else:
        bvh_list = bvh

    if len(bvh_list) == 0:
        raise ValueError("At least one Bvh object is required.")

    coords_list: list[npt.NDArray[np.float64]] = []

    if frames is None or (isinstance(frames, int) and frames == -1):
        # All frames for each Bvh
        for b in bvh_list:
            coords = b.get_spatial_coord(centered=centered)
            if coords.ndim == 2:
                coords = coords[np.newaxis]  # (N, 3) -> (1, N, 3)
            coords_list.append(coords)

    elif isinstance(frames, int):
        # Single frame index
        for b in bvh_list:
            coords = b.get_spatial_coord(frame_num=frames, centered=centered)
            coords_list.append(coords[np.newaxis])  # (N, 3) -> (1, N, 3)

    elif isinstance(frames, np.ndarray):
        if len(bvh_list) != 1:
            raise ValueError(
                "Pre-computed coordinate arrays can only be passed with a "
                "single Bvh object, not a list.")
        arr = np.asarray(frames, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (N, 3) -> (1, N, 3)
        elif arr.ndim != 3:
            raise ValueError(
                f"Expected frames array with 2 or 3 dimensions, got {arr.ndim}.")
        coords_list.append(arr)

    else:
        raise TypeError(
            f"frames must be int, ndarray, or None, got {type(frames).__name__}.")

    return bvh_list, coords_list


# ---------------------------------------------------------------------------
# Bounding box / axis limits
# ---------------------------------------------------------------------------

def compute_unified_limits(
    coords_list: list[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], float]:
    """Compute a cubic bounding box encompassing all skeletons and frames.

    The half-span is the larger of the per-frame body size and the
    trajectory extent from center. This ensures stationary skeletons
    fill the frame while walking skeletons never clip.

    Parameters
    ----------
    coords_list : list of ndarray
        Each element has shape ``(F, N, 3)`` or ``(N, 3)``.

    Returns
    -------
    center : ndarray of shape (3,)
        Center of the bounding box in world coordinates.
    half_span : float
        Half the side length of the cubic bounding box.
    """
    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)
    max_body_span = 0.0

    for coords in coords_list:
        if coords.ndim == 2:
            coords = coords[np.newaxis]
        frame_mins = coords.min(axis=1)
        frame_maxs = coords.max(axis=1)
        global_min = np.minimum(global_min, frame_mins.min(axis=0))
        global_max = np.maximum(global_max, frame_maxs.max(axis=0))
        frame_spans = frame_maxs - frame_mins
        max_body_span = max(max_body_span, float(frame_spans.max()))

    center = (global_min + global_max) / 2.0

    # half_span must cover both body size AND trajectory extent from center
    trajectory_half_span = float(
        np.maximum(global_max - center, center - global_min).max())
    half_span = max(max_body_span / 2.0, trajectory_half_span)
    # Add a small margin (5%) so skeleton doesn't touch the edge
    half_span *= 1.05
    return center, half_span


# ---------------------------------------------------------------------------
# Camera angles
# ---------------------------------------------------------------------------

def get_camera_angles(
    bvh: Bvh,
    ref_frame: npt.NDArray[np.float64],
    camera: str | tuple[float, float] = "front",
) -> tuple[float, float, str]:
    """Resolve a camera specification to (azimuth, elevation, up_axis).

    Parameters
    ----------
    bvh : Bvh
        The BVH object (used for axis detection).
    ref_frame : ndarray of shape (N, 3)
        A reference frame of spatial coordinates for axis heuristics.
    camera : str or (float, float)
        - ``"front"`` — auto-detected front view (default).
        - ``"side"`` — 90 degrees from front.
        - ``"top"`` — bird's-eye view looking down the up axis.
        - ``(azimuth_deg, elevation_deg)`` — custom angles.

    Returns
    -------
    azimuth : float
        Azimuth angle in degrees.
    elevation : float
        Elevation angle in degrees.
    up_axis : str
        Single character: ``'x'``, ``'y'``, or ``'z'``.
    """
    from ..tools import get_forw_up_axis, extract_sign

    directions = get_forw_up_axis(bvh, ref_frame)
    forward_ax = directions['forward']   # e.g. '+z'
    up_ax = directions['upward']         # e.g. '+y'
    up_char = up_ax[1]                   # 'y'
    up_positive = extract_sign(up_ax)
    fwd_char = forward_ax[1]
    fwd_positive = extract_sign(forward_ax)

    if isinstance(camera, tuple):
        return float(camera[0]), float(camera[1]), up_char

    # Compute base azimuth/elevation for the "front" view.
    # The logic: determine which matplotlib azimuth faces the skeleton's
    # forward axis, accounting for which axis is up.

    # Matplotlib's default front-facing axis given the vertical axis
    # Matplotlib's default front-facing axis at azim=0 for each vertical_axis:
    # vertical_axis='z': azim=0 looks along -x, so default front is 'x'
    # vertical_axis='y': azim=0 looks along -z, so default front is 'z'
    # vertical_axis='x': azim=0 looks along -y, so default front is 'y'
    default_up2front = {'z': 'x', 'y': 'z', 'x': 'y'}

    base_azim = -20.0
    base_elev = 20.0

    if fwd_char != default_up2front[up_char]:
        base_azim += 90.0

    if not up_positive:
        base_elev += 180.0
        base_azim += 180.0

    if fwd_char == default_up2front[up_char] and not fwd_positive:
        base_azim += 180.0

    if camera == "front":
        return base_azim, base_elev, up_char
    elif camera == "side":
        return base_azim + 90.0, base_elev, up_char
    elif camera == "top":
        return base_azim, 90.0, up_char
    else:
        raise ValueError(
            f"Unknown camera preset {camera!r}. "
            f"Use 'front', 'side', 'top', or (azimuth, elevation).")


# ---------------------------------------------------------------------------
# Orthographic projection (used by OpenCV backend)
# ---------------------------------------------------------------------------

def build_view_matrix(
    azimuth_deg: float,
    elevation_deg: float,
    up_axis: str,
) -> npt.NDArray[np.float64]:
    """Build a 3x3 rotation that maps world coordinates to view coordinates.

    Uses the same look-at camera math as matplotlib's ``view_init``
    so that both backends produce identical views for the same
    (azimuth, elevation, up_axis) parameters.

    View coordinate convention: x = right on screen, y = up on screen,
    z = out of screen (toward viewer).

    Parameters
    ----------
    azimuth_deg : float
        Azimuth rotation in degrees.
    elevation_deg : float
        Elevation rotation in degrees.
    up_axis : str
        ``'x'``, ``'y'``, or ``'z'``.

    Returns
    -------
    view_matrix : ndarray of shape (3, 3)
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    axis_idx = _UP_AXIS_INDEX.get(up_axis, 2)

    # Eye direction from spherical coordinates, rolled to match
    # vertical axis (same as matplotlib's _roll_to_vertical).
    eye_dir = np.roll(
        [np.cos(el) * np.cos(az),
         np.cos(el) * np.sin(az),
         np.sin(el)],
        axis_idx - 2)

    # w = viewing direction (from eye toward origin = out of screen)
    w = eye_dir / np.linalg.norm(eye_dir)

    # Up vector along the vertical axis
    V = np.zeros(3)
    V[axis_idx] = -1.0 if abs(np.degrees(el)) > 90 else 1.0

    # Right and up via cross products
    u = np.cross(V, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    # View matrix: rows are u (right), v (up), w (out of screen)
    return np.array([u, v, w])


def ortho_project(
    coords_3d: npt.NDArray[np.float64],
    view_matrix: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    half_span: float,
    resolution: tuple[int, int],
) -> npt.NDArray[np.int32]:
    """Orthographic projection from 3D world to 2D pixel coordinates.

    Parameters
    ----------
    coords_3d : ndarray of shape (N, 3)
        World-space joint positions for one frame.
    view_matrix : ndarray of shape (3, 3)
        From :func:`build_view_matrix`.
    center : ndarray of shape (3,)
        World-space center of the bounding box.
    half_span : float
        Half the side length of the cubic bounding box.
    resolution : (width, height)
        Output image dimensions in pixels.

    Returns
    -------
    pixels : ndarray of shape (N, 2)
        Integer pixel coordinates ``(x, y)`` for each joint.
    """
    w, h = resolution
    viewed = (coords_3d - center) @ view_matrix.T  # (N, 3)

    # Compute the view-space half_span by projecting the bounding box
    # corners through the rotation. A world-space cube becomes a larger
    # rotated box in view space.
    corners = np.array([[sx, sy, sz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                       dtype=np.float64) * half_span
    corners_view = corners @ view_matrix.T
    view_half_u = float(np.abs(corners_view[:, 0]).max())
    view_half_v = float(np.abs(corners_view[:, 1]).max())

    # Scale to fit within 90% of each dimension independently
    if view_half_u > 1e-8 and view_half_v > 1e-8:
        scale_u = (w * 0.9) / (2.0 * view_half_u)
        scale_v = (h * 0.9) / (2.0 * view_half_v)
        scale = min(scale_u, scale_v)
    elif half_span > 1e-8:
        scale = min(w, h) * 0.9 / (2.0 * half_span)
    else:
        scale = 1.0

    px = viewed[:, 0] * scale + w / 2.0
    py = h / 2.0 - viewed[:, 1] * scale  # flip y for image coords

    return np.stack([px, py], axis=-1).astype(np.int32)


# ---------------------------------------------------------------------------
# Frame count alignment
# ---------------------------------------------------------------------------

def align_frame_counts(
    coords_list: list[npt.NDArray[np.float64]],
    pad: bool = False,
) -> list[npt.NDArray[np.float64]]:
    """Align all coordinate arrays to the same frame count.

    When comparing multiple skeletons with different frame counts,
    arrays are either truncated to the minimum or padded to the
    maximum (by repeating the last frame).

    Parameters
    ----------
    coords_list : list of ndarray
        Each element has shape ``(F, N, 3)``.
    pad : bool, optional
        If ``False`` (default), truncate to the shortest clip.
        If ``True``, pad shorter clips by repeating their last frame
        so all clips match the longest.

    Returns
    -------
    coords_list : list of ndarray
        Arrays all with the same frame count.
    """
    if len(coords_list) <= 1:
        return coords_list

    if not pad:
        min_frames = min(c.shape[0] for c in coords_list)
        return [c[:min_frames] for c in coords_list]

    max_frames = max(c.shape[0] for c in coords_list)
    result = []
    for c in coords_list:
        if c.shape[0] < max_frames:
            pad_count = max_frames - c.shape[0]
            last_frame = c[-1:].repeat(pad_count, axis=0)
            c = np.concatenate([c, last_frame], axis=0)
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Color palette for multi-skeleton comparison
# ---------------------------------------------------------------------------

# BGR colors for OpenCV, converted to RGB for matplotlib where needed
PALETTE_BGR = [
    (255, 120, 50),   # blue
    (50, 50, 220),    # red
    (50, 180, 50),    # green
    (200, 130, 50),   # teal
    (50, 100, 200),   # orange
    (200, 50, 200),   # magenta
]

PALETTE_RGB = [(r, g, b) for (b, g, r) in PALETTE_BGR]
PALETTE_MPL = [(r / 255, g / 255, b / 255) for (r, g, b) in PALETTE_RGB]
