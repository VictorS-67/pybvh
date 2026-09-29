"""The geometry of the picture: what is framed, where the ground plane
lies, where the camera stands and how it moves.

One :class:`Viewport` is computed from the view or views a picture
shows, and every backend translates its numbers into its toolkit's
calls. Pure numpy. No plotting library imports; the one thing taken
from the pybvh core is the array-pure facing kernel the follow camera
runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from ._scene import GroundFrame, UP_AXIS_INDEX

if TYPE_CHECKING:
    from ._scene import SkeletonView


# ---------------------------------------------------------------------------
# The viewport
# ---------------------------------------------------------------------------

# Every distance below is a fraction or a multiple of the half-span, the
# half side of the cube around everything the picture shows.

# The ground plane reaches this many half-spans from its centre in each
# ground direction: wide enough to fill the frame at the usual camera
# elevations, small enough that its far edge stays in the picture.
FLOOR_EXTENT = 1.8
# A still that draws a floor shifts its cube so the plane sits this far
# above the cube's bottom edge; without the shift a floor below the
# lowest joint would fall outside the axes.
FLOOR_INSET = 0.02
# A perspective camera stands this many half-spans from the cube's
# centre. A constant, not derived from the toolkit's view angle.
# TODO: derive the distance from the framing box and the vertical view
# angle, so the box provably fits whatever the toolkit's default is.
EYE_DISTANCE = 4.0
# The orthographic projection fits the framing box into this fraction
# of the panel, in whichever direction is tighter.
FIT_FRACTION = 0.9

_FRAMINGS = ("still", "clip")
_MOTIONS = ("fixed", "turntable", "follow")


class Camera(NamedTuple):
    """Where a perspective camera stands, what it looks at, which way
    is up on screen. World coordinates, each of shape ``(3,)``."""

    eye: npt.NDArray[np.float64]
    target: npt.NDArray[np.float64]
    up: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Viewport(GroundFrame):
    """The geometry of one picture, in world coordinates.

    Built by :func:`make_viewport` from one view (a panel of a
    multi-panel figure) or from several (the one scene k3d and vedo
    draw everything into). A backend reads the numbers and translates
    them; it computes no box, floor extent or eye position of its own.

    Two boxes, on purpose:

    - ``center`` and ``half_span`` are the cube around every coordinate
      shown, 5% margin included. It is the picture's *size scale*: line
      widths, capsule radii, the floor's extent and the camera distance
      are multiples of ``half_span``, and perspective cameras look at
      ``center``.
    - ``lo`` and ``hi`` are the *framing box*, what the axes or the
      orthographic projection are fitted to. For ``framing="still"`` it
      is that same cube (shifted along up when a floor must sit inside
      it); for ``framing="clip"`` it is the box the motion sweeps, see
      :func:`framing_bounds`.

    The floor is one plane for the whole viewport: at the ground-side
    extreme of the views' floor heights (the coordinate minimum for a
    positive up axis, the maximum for a negative one), exactly there.
    A toolkit that needs the plane nudged to avoid z-fighting applies
    its own epsilon through :meth:`below_floor`.

    ``azimuths`` is the camera's schedule: one azimuth in degrees per
    frame, or ``None`` when the camera does not move. A schedule that
    turns out constant, whatever the reason (a follow camera on a rig
    with no left/right pairs, or on a character that never turns; a
    facing that is degenerate at frame 0; a turntable over a single
    frame), is stored as ``None``, so ``rotating`` means what it says
    and such a clip is framed as a fixed camera frames it.
    """

    lo: npt.NDArray[np.float64]            # (3,) framing box, low corner
    hi: npt.NDArray[np.float64]            # (3,) framing box, high corner
    center: npt.NDArray[np.float64]        # (3,) centre of the cube
    half_span: float                       # half side of the cube
    up: str                                # signed world-up axis, e.g. '+y'
    floor_height: float                    # ground plane along the up axis
    azimuth: float                         # degrees, the camera's base angle
    elevation: float                       # degrees
    azimuths: npt.NDArray[np.float64] | None  # (F,) degrees, or None
    projection: str                        # 'persp' | 'ortho', as drawn

    @property
    def rotating(self) -> bool:
        """Whether the camera's azimuth changes during the clip."""
        return self.azimuths is not None

    def azimuth_at(self, frame: int = 0) -> float:
        """The camera azimuth at *frame*, in degrees."""
        if self.azimuths is None:
            return self.azimuth
        return float(self.azimuths[frame])

    def view_matrix(self, frame: int = 0) -> npt.NDArray[np.float64]:
        """World-to-view rotation at *frame*, see :func:`build_view_matrix`."""
        if self.azimuths is None:
            return self._view_matrices[0]
        return self._view_matrices[frame]

    def camera(self, frame: int = 0) -> Camera:
        """Where a perspective camera stands at *frame*.

        The eye is ``EYE_DISTANCE`` half-spans from the cube's centre
        along the viewing direction, looking at the centre."""
        matrix = self.view_matrix(frame)
        eye = self.center + matrix[2] * (EYE_DISTANCE * self.half_span)
        return Camera(eye=eye, target=self.center.copy(), up=matrix[1].copy())

    def enclosing_cube(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The smallest cube around the framing box, centred on it, as
        ``(lo, hi)``. For axes that must stay cubic around a box that
        is not: the overlaid sequence figure."""
        middle = (self.lo + self.hi) / 2
        half = float((self.hi - self.lo).max()) / 2
        return middle - half, middle + half

    def grounded_box(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The cube with its ground-side face moved to just under the
        ground, as ``(lo, hi)``.

        For a toolkit that draws the box itself, as k3d draws its grid.
        The face under the skeleton sits ``FLOOR_INSET`` half-spans
        below the ground plane, so whatever lies on the ground (the
        floor, the root trail) is seen lying on the box's bottom face.
        Left at the cube's own bottom, that face can be far below the
        ground (a clip that travels has a cube much taller than the
        body), and a trail drawn at the ground then floats inside the
        box and reads as displaced sideways from an oblique angle.
        Only the extent along up changes, so the box is no longer a
        cube. A ground that lies beyond the cube's far face (above the
        head, for a positive up axis) still gives a valid box, from
        that far face to the ground."""
        lo = self.center - self.half_span
        hi = self.center + self.half_span
        face = self.below_floor(FLOOR_INSET * self.half_span)
        if self.up_sign > 0:
            lo[self.up_index] = face
        else:
            hi[self.up_index] = face
        return np.minimum(lo, hi), np.maximum(lo, hi)

    @property
    def floor_reach(self) -> float:
        """How far the ground plane reaches from its centre in each
        ground direction: ``FLOOR_EXTENT`` half-spans."""
        return self.half_span * FLOOR_EXTENT

    def floor_quad(
        self, clip_to_box: bool = False,
    ) -> npt.NDArray[np.float64]:
        """The four corners of the ground plane, shape ``(4, 3)``.

        A square of half side :attr:`floor_reach` around the
        cube's centre (which is also the framing box's centre on the
        ground), at ``floor_height``. With *clip_to_box* the rectangle
        of the framing box on the ground instead: a wide orthographic
        still turns a full-extent plane into a backdrop wall.

        Corners run ``(lo, lo), (hi, lo), (hi, hi), (lo, hi)`` over the
        two :attr:`ground_axes`, so opposite corners are 0 and 2.
        """
        first, second = self.ground_axes
        if clip_to_box:
            a_lo, a_hi = float(self.lo[first]), float(self.hi[first])
            b_lo, b_hi = float(self.lo[second]), float(self.hi[second])
        else:
            reach = self.floor_reach
            a_lo = float(self.center[first]) - reach
            a_hi = float(self.center[first]) + reach
            b_lo = float(self.center[second]) - reach
            b_hi = float(self.center[second]) + reach
        quad = np.zeros((4, 3), dtype=np.float64)
        quad[:, first] = (a_lo, a_hi, a_hi, a_lo)
        quad[:, second] = (b_lo, b_lo, b_hi, b_hi)
        quad[:, self.up_index] = self.floor_height
        return quad

    def ground_path(
        self, points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """*points* dropped onto the ground plane, as a new array.

        For the root trace and the viewers' trails: the path a joint
        takes, drawn where the floor is."""
        path = np.array(points, dtype=np.float64)
        path[..., self.up_index] = self.floor_height
        return path

    def project(
        self,
        points: npt.NDArray[np.float64],
        resolution: tuple[int, int],
        frame: int = 0,
    ) -> npt.NDArray[np.int32]:
        """Orthographic pixel coordinates of *points* at *frame*.

        The framing box's centre lands in the middle of the panel, and
        one scale holds for the whole clip: the box is fitted at the
        widest it gets over the camera's schedule. The alternative,
        fitting it frame by frame, rescales the drawing continuously
        under a rotating camera and reads as the character zooming in
        and out."""
        return ortho_project(
            points, self.view_matrix(frame), (self.lo + self.hi) / 2.0,
            self.view_half, resolution)

    @cached_property
    def view_half(self) -> tuple[float, float]:
        """Half extents of the framing box on screen, in view units:
        the largest over every frame of the camera's schedule."""
        corners = box_corners(self.lo, self.hi) - (self.lo + self.hi) / 2.0
        projected = corners @ np.swapaxes(self._view_matrices, 1, 2)
        return (float(np.abs(projected[..., 0]).max()),
                float(np.abs(projected[..., 1]).max()))

    @cached_property
    def _view_matrices(self) -> npt.NDArray[np.float64]:
        """One view matrix per scheduled frame, or a single one for a
        fixed camera, shape ``(F, 3, 3)``."""
        azimuths = (np.array([self.azimuth]) if self.azimuths is None
                    else self.azimuths)
        return np.stack([
            build_view_matrix(float(azimuth), self.elevation, self.up_axis)
            for azimuth in azimuths])


def make_viewport(
    views: Sequence[SkeletonView],
    *,
    framing: str = "still",
    motion: str = "fixed",
    include_floor: bool = True,
    projection: str = "persp",
) -> Viewport:
    """Compute the :class:`Viewport` of a picture of *views*.

    Parameters
    ----------
    views : sequence of SkeletonView
        What the picture shows: one view for a panel, all of a Scene's
        views for a single-scene backend. The camera angles, the up
        axis and the follow schedule are the first view's.
    framing : {"still", "clip"}
        ``"still"`` frames the cube around the coordinates.
        ``"clip"`` frames the box the motion sweeps
        (:func:`framing_bounds`), so a clip that travels far sideways
        does not shrink the character to fit a cube.
    motion : {"fixed", "turntable", "follow"}
        How the camera moves. ``"turntable"`` orbits once over the
        clip (:func:`turntable_azimuths`); ``"follow"`` tracks the
        character's rotation (:func:`compute_follow_azimuths`).
    include_floor : bool
        Whether the framing box must contain the ground plane. A still
        whose floor lies beyond its cube's ground-side face, or closer
        to that face than ``FLOOR_INSET`` half-spans, shifts the cube
        along up, as far as it takes, until the plane sits that far
        inside; the cube keeps its size. It never shifts the other
        way: a floor beyond the far face (above the head) stays
        outside a still. A clip extends its box to the plane on
        whichever side the plane lies.
    projection : {"persp", "ortho"}
        The projection the picture is drawn with, stated by the
        adapter that draws it: matplotlib passes what the style asks
        for (``sequence``'s offset layout is always orthographic),
        OpenCV always ``"ortho"``, k3d and vedo always ``"persp"``.
        The viewport does not choose it; it records it, so that
        ``viewport.projection`` describes the picture.

    Returns
    -------
    Viewport
    """
    if not views:
        raise ValueError("A viewport needs at least one view.")
    if framing not in _FRAMINGS:
        raise ValueError(
            f"Unknown framing {framing!r}. Choose from: {list(_FRAMINGS)}")
    if motion not in _MOTIONS:
        raise ValueError(
            f"Unknown motion {motion!r}. Choose from: {list(_MOTIONS)}")

    first = views[0]
    center, half_span = compute_unified_limits([v.coords for v in views])
    ground_side = min if first.up_sign > 0 else max
    floor_height = float(ground_side(v.floor_height for v in views))

    if motion == "follow":
        azimuths = compute_follow_azimuths(first, first.azimuth)
    elif motion == "turntable":
        azimuths = turntable_azimuths(first.azimuth, first.coords.shape[0])
    else:
        azimuths = None
    if azimuths is not None and np.all(azimuths == azimuths[0]):
        azimuths = None

    if framing == "clip":
        lo, hi = _swept_box(
            np.concatenate([v.coords.reshape(-1, 3) for v in views]),
            first.up_index, floor_height if include_floor else None,
            rotating=azimuths is not None)
    else:
        box_center = center.copy()
        if include_floor:
            up = first.up_index
            sign = first.up_sign
            # The cube's edge visually below the skeleton is
            # center - sign * half_span: for a negative up axis the
            # ground sits at the coordinate MAXIMUM.
            bottom = box_center[up] - sign * half_span
            target = floor_height - sign * (FLOOR_INSET * half_span)
            if sign * (target - bottom) < 0:
                box_center[up] += target - bottom
        lo, hi = box_center - half_span, box_center + half_span

    return Viewport(
        lo=lo, hi=hi, center=center, half_span=half_span, up=first.up,
        floor_height=floor_height, azimuth=float(first.azimuth),
        elevation=float(first.elevation), azimuths=azimuths,
        projection=projection)


def panel_viewports(
    views: Sequence[SkeletonView],
    **options: object,
) -> list[Viewport]:
    """One :class:`Viewport` per view, for the multi-panel backends.

    Each panel is framed, floored and scheduled from its own view;
    *options* are :func:`make_viewport`'s."""
    return [make_viewport([view], **options)  # type: ignore[arg-type]
            for view in views]


# ---------------------------------------------------------------------------
# The cube (size scale, and the framing of a still)
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
    return _swept_box(
        view.coords.reshape(-1, 3), view.up_index, view.floor_height,
        rotating)


def _swept_box(
    points: npt.NDArray[np.float64],
    up: int,
    floor_height: float | None,
    rotating: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """:func:`framing_bounds` over plain points, ``(P, 3)``.

    *floor_height* ``None`` leaves the ground plane out of the box."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    if floor_height is not None:
        lo[up] = min(lo[up], floor_height)
        hi[up] = max(hi[up], floor_height)

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
# View matrix and orthographic projection
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
    points: npt.NDArray[np.float64],
    view_matrix: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    view_half: tuple[float, float],
    resolution: tuple[int, int],
) -> npt.NDArray[np.int32]:
    """Orthographic projection from 3D world to 2D pixel coordinates.

    The kernel under :meth:`Viewport.project`, which supplies the
    centre and the half extents. One scale serves both screen
    directions, so a world unit covers the same number of pixels
    across and up: the scale is the smaller of the two that would fit
    ``view_half`` into ``FIT_FRACTION`` of the panel's width and of
    its height.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        World-space positions.
    view_matrix : ndarray of shape (3, 3)
        From :func:`build_view_matrix`.
    center : ndarray of shape (3,)
        The world point that lands in the middle of the panel.
    view_half : (float, float)
        Half extents, across and up the screen in view units, of what
        must fit. A direction with no extent (below ``1e-8``) sets no
        constraint; with none in either the scale is 1 pixel per unit.
    resolution : (width, height)
        Panel size in pixels.

    Returns
    -------
    pixels : ndarray of shape (N, 2)
        Integer pixel coordinates ``(x, y)``, y growing downward.
    """
    w, h = resolution
    viewed = (points - center) @ view_matrix.T  # (N, 3)

    scales = [(extent * FIT_FRACTION) / (2.0 * half)
              for extent, half in zip((w, h), view_half) if half > 1e-8]
    scale = min(scales) if scales else 1.0

    px = viewed[:, 0] * scale + w / 2.0
    py = h / 2.0 - viewed[:, 1] * scale  # flip y for image coords

    return np.stack([px, py], axis=-1).astype(np.int32)
