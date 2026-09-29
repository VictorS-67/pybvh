"""The geometry of the picture: framing, ground path, camera azimuths,
view matrix and projection.

Pure numpy, computed from a :class:`~._scene.SkeletonView`. No plotting
library imports; the one thing taken from the pybvh core is the
array-pure facing kernel the follow camera runs.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._scene import UP_AXIS_INDEX

if TYPE_CHECKING:
    from ._scene import SkeletonView


# ---------------------------------------------------------------------------
# Framing box (animated output)
# ---------------------------------------------------------------------------

# Fraction of the framing box left as breathing room around the motion.
FRAMING_MARGIN = 0.04


def framing_bounds(
    view: SkeletonView,
    rotating: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The world-space box animated output frames to, as ``(lo, hi)``.

    Animated backends frame the *motion*, not a cube around it: each
    axis gets the extent the clip actually uses, so a clip that travels
    two body-lengths sideways no longer forces that same extent
    vertically, shrinking the character to fit. The floor plane is
    included along the up axis so the ground never falls outside the
    box. Scale stays equal on all three axes — this crops empty space,
    it never stretches the skeleton.

    We frame to the box the whole clip sweeps, not to each frame's own
    box. The alternative — re-framing per frame — keeps the character
    largest at every instant but makes the world drift and breathe
    behind it, which reads as camera shake and destroys any sense of
    travel; it is also what makes a still and a video of the same clip
    disagree (:func:`frame` frames one pose, so its box is tighter).

    With *rotating* (turntable or follow), the two ground axes are
    squared off to the motion's circumscribed radius, so the framing is
    invariant to azimuth. Without it, an orbiting camera would sweep a
    long clip's travel axis from across-screen to into-screen and the
    character would appear to zoom in and out.

    Parameters
    ----------
    view : SkeletonView
        The panel's view; ``coords`` supplies the motion, ``up_index``
        and ``floor_height`` place the ground.
    rotating : bool, optional
        Whether the camera azimuth changes during the clip.

    Returns
    -------
    lo, hi : ndarray of shape (3,)
        Opposite corners of the framing box, margin included.
    """
    points = view.coords.reshape(-1, 3)
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    up = view.up_index
    lo[up] = min(lo[up], view.floor_height)
    hi[up] = max(hi[up], view.floor_height)

    if rotating:
        ground = [i for i in range(3) if i != up]
        center = (lo[ground] + hi[ground]) / 2.0
        radius = float(np.hypot(*(points[:, ground] - center).T).max())
        lo[ground] = center - radius
        hi[ground] = center + radius

    pad = FRAMING_MARGIN * float((hi - lo).max())
    return lo - pad, hi + pad


def box_corners(
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The eight corners of an axis-aligned box, shape ``(8, 3)``."""
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Ground path
# ---------------------------------------------------------------------------

def floor_trace_points(
    view: SkeletonView,
    start: int = 0,
    upto: int | None = None,
) -> npt.NDArray[np.float64]:
    """Root path projected onto the floor plane (for the dashed trace).

    ``start``/``upto`` bound the traced frame range (inclusive of
    ``upto``); the default is the whole clip.
    """
    end = view.coords.shape[0] if upto is None else upto + 1
    path = view.coords[start:end, 0, :].copy()
    path[:, view.up_index] = view.floor_height
    return path


# ---------------------------------------------------------------------------
# Azimuth schedules
# ---------------------------------------------------------------------------

def turntable_azimuths(
    base_azim: float,
    num_frames: int,
) -> npt.NDArray[np.float64]:
    """Per-frame azimuths for a full 360-degree orbit over the clip.

    The trivial case of follow: a constant-rate azimuth ramp starting
    at the base camera angle (endpoint excluded so looped playback
    doesn't hold the identical view for two frames).
    """
    return base_azim + np.linspace(0.0, 360.0, num_frames, endpoint=False)


def compute_follow_azimuths(
    view: SkeletonView,
    base_azim: float,
) -> npt.NDArray[np.float64]:
    """Per-frame camera azimuths that track the character's rotation.

    Follow mode uses CONTINUOUS rotation tracking: the base camera azimuth
    corresponds to frame 0, and every frame adds the signed rotation delta
    between frame 0's lateral (left-to-right) axis and the current frame's,
    measured around ``world_up``. This gives a smooth orbit that tracks the
    character's actual rotation — not a snap-every-90°-to-a-signed-axis.

    A pure function of the view: it reads ``coords``, ``lr_pairs`` and
    ``up_vector``, so it can be recomputed after any operation that
    changes the coords and never goes stale.

    Parameters
    ----------
    view : SkeletonView
        The skeleton's coords for the whole clip plus its L/R pairs and
        signed up vector.
    base_azim : float
        The frame-0 azimuth in degrees (from :func:`get_camera_angles`).

    Returns
    -------
    azimuths : ndarray of shape (F,)
        Azimuth in degrees for every frame. Frames where the lateral
        direction is degenerate (parallel to world up, or no L/R pairs)
        fall back to ``base_azim``. Frame 0 is the reference every delta
        is measured from, so when frame 0 itself is degenerate there is
        no reference and the whole sequence stays at ``base_azim``,
        later valid frames included: a fixed camera, not a partial
        follow.
    """
    from ..tools import _leftward_units_from_pairs

    num_frames = view.coords.shape[0]
    azimuths = np.full(num_frames, float(base_azim))

    # World-space leftward unit vector per frame — the shared facing
    # geometry kernel in pybvh.tools. All-invalid when no L/R pairs exist.
    leftward, valid = _leftward_units_from_pairs(
        view.coords, view.lr_pairs, view.up_vector)
    if num_frames == 0 or not valid[0]:
        return azimuths  # no frame-0 reference — camera stays fixed

    up_vec = view.up_vector

    # Signed angle rotating frame 0's leftward onto each frame's,
    # around world_up (vectorized _signed_rotation_delta_around_axis).
    left_0 = leftward[0]
    cos_a = np.clip(leftward @ left_0, -1.0, 1.0)
    sin_a = np.cross(np.broadcast_to(left_0, leftward.shape), leftward) @ up_vec
    deltas = np.degrees(np.arctan2(sin_a, cos_a))
    azimuths[valid] = base_azim + deltas[valid]
    return azimuths


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
    axis_idx = UP_AXIS_INDEX.get(up_axis, 2)

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
    fixed_view_half: tuple[float, float] | None = None,
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
    fixed_view_half : (float, float), optional
        Pre-computed ``(view_half_u, view_half_v)`` to use for the scale
        calculation instead of computing it from the current view matrix.
        Useful for follow-mode rendering where the view rotates every
        frame: pass the max over all frames once to get a stable
        (angle-invariant) scale so the character doesn't appear to
        zoom in and out as the camera orbits. When ``None`` (default),
        the view-space extents are computed from the bounding box
        corners under the current view matrix (the default behavior,
        which gives a tighter fit per frame but oscillates under
        rotation).

    Returns
    -------
    pixels : ndarray of shape (N, 2)
        Integer pixel coordinates ``(x, y)`` for each joint.
    """
    w, h = resolution
    viewed = (coords_3d - center) @ view_matrix.T  # (N, 3)

    if fixed_view_half is not None:
        view_half_u, view_half_v = fixed_view_half
    else:
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
