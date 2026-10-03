"""The geometry of the picture: what is framed, where the ground plane
lies, where the camera stands and how it moves.

One :class:`Viewport` is computed from the view or views a picture
shows, and every backend translates its numbers into its toolkit's
calls. Pure numpy. No plotting library imports; the one thing taken
from the pybvh core is the array-pure facing kernel the follow camera
runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from ._scene import UP_AXIS_INDEX, GroundFrame

if TYPE_CHECKING:
    from ._scene import SkeletonView


# ---------------------------------------------------------------------------
# The viewport
# ---------------------------------------------------------------------------

# Every distance below is a fraction or a multiple of the half-span, the
# half side of the cube around everything the picture shows.

# The cube's half side is half the widest extent it must hold, times
# this margin, so the skeleton does not touch the edge.
CUBE_MARGIN = 1.05
# The half-span of a still of a pose whose widest extent is its height,
# in heights. Sizes that were fractions of the half-span until v0.10.0
# and are now fractions of the body size keep their standing-still
# value by being multiplied by this.
STANDING_STILL_HALF_SPAN = CUBE_MARGIN / 2

# The ground plane reaches this many half-spans from its centre in each
# ground direction: wide enough to fill the frame at the usual camera
# elevations, small enough that its far edge stays in the picture.
FLOOR_EXTENT = 1.8
# A still that draws a floor shifts its cube so the plane sits this far
# above the cube's bottom edge; without the shift a floor below the
# lowest joint would fall outside the axes.
FLOOR_INSET = 0.02
# The orthographic projection fits the framing box, and the perspective
# camera the coordinates shown, into this fraction of the picture, in
# whichever direction is tighter.
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
class Turntable:
    """A turntable camera that turns at a set speed: one revolution
    every ``period`` frames, whatever the number of frames.

    The ``motion`` a backend hands to :func:`make_viewport` when the
    caller chose the speed; the plain ``"turntable"`` is the one orbit
    over the frames, ``Turntable(period=num_frames)``. The period is a
    number of frames of the picture, not necessarily whole, positive
    and finite; a period shorter than the frames makes several orbits.
    """

    period: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.period) and self.period > 0):
            raise ValueError(
                f"A turntable's period must be a positive number of frames, got {self.period!r}."
            )


@dataclass(frozen=True)
class Viewport(GroundFrame):
    """The geometry of one picture, in world coordinates.

    Built by :func:`make_viewport` from one view (a panel of a
    multi-panel figure) or from several (the one scene k3d and vedo
    draw everything into). A backend reads the numbers and translates
    them; it computes no box, floor extent or eye position of its own.

    Two boxes, on purpose:

    - ``center`` and ``half_span`` are the cube around every coordinate
      shown, 5% margin included. It is the scale of the picture: the
      floor's extent is a multiple of ``half_span``, and perspective
      cameras look at ``center`` (from a distance fitted to the
      coordinates shown, :meth:`eye_distance`). It is not the scale of
      what is drawn on a body, since it grows with the distance a clip
      travels: the vedo capsules and k3d's lines and joints are sized
      from each skeleton's :attr:`~._scene.SkeletonView.body_size`.
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

    lo: npt.NDArray[np.float64]  # (3,) framing box, low corner
    hi: npt.NDArray[np.float64]  # (3,) framing box, high corner
    center: npt.NDArray[np.float64]  # (3,) centre of the cube
    half_span: float  # half side of the cube
    up: str  # signed world-up axis, e.g. '+y'
    floor_height: float  # ground plane along the up axis
    azimuth: float  # degrees, the camera's base angle
    elevation: float  # degrees
    azimuths: npt.NDArray[np.float64] | None  # (F,) degrees, or None
    projection: str  # 'persp' | 'ortho', as drawn
    # One (F, N, 3) array per view, the views' own read-only arrays:
    # what the perspective camera fits.
    shown_coords: tuple[npt.NDArray[np.float64], ...] = field(repr=False)

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

    def camera(
        self,
        frame: int = 0,
        *,
        view_angle: float,
        aspect: float | None = None,
        band: tuple[float, float] = (0.0, 1.0),
    ) -> Camera:
        """Where a perspective camera stands at *frame*.

        The camera looks at the cube's centre along the viewing
        direction, from :meth:`eye_distance` for the toolkit's vertical
        *view_angle* (degrees), the picture's *aspect* ratio (width
        over height; ``None`` fits the vertical direction alone) and
        the *band* of its height the figure may take."""
        matrix = self.view_matrix(frame)
        distance = self.eye_distance(view_angle, aspect, band)
        eye = self.center + matrix[2] * distance
        return Camera(eye=eye, target=self.center.copy(), up=matrix[1].copy())

    def eye_distance(
        self,
        view_angle: float,
        aspect: float | None = None,
        band: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """How far the perspective camera stands from the cube's centre.

        The smallest distance at which every coordinate shown projects
        inside ``FIT_FRACTION`` of the picture, in whichever direction
        is tighter, as the orthographic projection fits its box; where
        only a *band* of the picture's height is free for the figure,
        inside ``FIT_FRACTION`` of that band, about its middle. Solved
        exactly: a coordinate at ``x`` across, ``y`` up and ``z``
        towards the eye from the target (view units) needs the eye at
        least ``z + max(y / (t b), |x| / (f t aspect))`` away, where
        ``t`` is the tangent of half the view angle, ``f`` the
        fraction, and ``b`` how far the fitted band reaches from the
        picture's middle on the coordinate's side (up for ``y > 0``,
        down, negative, for ``y < 0``; ``f`` either way for the whole
        height). The distance is the largest of these.

        Conventions, and the alternatives they were chosen over:

        - **Only the distance is fitted.** The camera keeps its target,
          the cube's centre, and its viewing direction. The alternative,
          aiming at the middle of the coordinates as they project,
          would let the eye come closer wherever the figure sits off
          the cube's centre on screen (a still whose floor shifts it,
          seen from above), but the target would then depend on the
          view angle, and every backend aims at the same point today.
        - **The target stays at the picture's middle, whatever the
          band.** Around a band whose middle is off the picture's, the
          figure reaches the band's nearer edge first and leaves room
          unused at the farther one. The alternative, shifting the
          projection so that the band's middle is the picture's middle
          (VTK's window centre), would use that room, but a camera
          would then need that shift besides its eye, target and up,
          and every toolkit a way to apply it. The two differ by as
          much as the band's middle is off the picture's: the vedo
          viewer's band is nearly centred, and its figure is about 1.5%
          smaller than the shifted projection would draw it.
        - **The coordinates shown are fitted**, every node of every view
          on every frame, not the corners of the framing cube. The
          cube's depth is mostly empty, and fitting its corners stands
          the camera much further back (about 6.6 half-spans at a 30
          degree view angle, where a standing figure's coordinates need
          4.1) and shrinks every picture. What is drawn around a node, a
          capsule's radius or a joint's sphere, is not fitted, and the
          floor plane is not either.
        - **Fitted to the view angle.** The alternative, a fixed
          multiple of the half-span (4 until v0.10.0), fits one view
          angle only: at 4 half-spans a still fills the height of VTK's
          30 degree picture and about half of k3d's 60 degree one.
        - **One distance for a moving camera**, the largest over the
          schedule (each frame's coordinates seen from that frame's
          direction), so the figure does not zoom in and out as a
          turntable or follow camera turns.
        - **The whole cube stays in front of the eye**: the eye stands
          at least as far as the cube's corner nearest it, so every
          coordinate, inside the cube by its margin, is in front of the
          eye. This binds on a figure flat across the fitted
          direction, whose coordinates land near the middle of the
          picture however close the eye comes (a line of joints seen
          end on, or a figure lying in the plane of the viewing
          direction and the screen's horizontal, fitted vertically
          alone): the fit alone would bring the eye onto, or past, the
          nearest of them. It also binds, fitted vertically alone, on a
          scene much wider than it is tall, whose cube its width sets
          (skeletons side by side in k3d): the fit alone would bring the
          eye inside the cube, and the scene further past the sides of
          a picture whose width is not fitted. None then reaches the
          fraction. The alternative, a small fixed gap in front of the nearest
          coordinate, would need a length of its own and leave the eye
          among the joints.

        Parameters
        ----------
        view_angle : float
            The toolkit's vertical view angle, in degrees: 30 for VTK,
            60 for k3d.
        aspect : float, optional
            The picture's width over its height. ``None`` fits the
            vertical direction alone, for a toolkit that does not know
            its picture's width.
        band : tuple of float, optional
            The part of the picture's height the figure may take, as
            ``(bottom, top)`` fractions of the height from its bottom
            edge: what a toolkit's controls leave free. The default is
            the whole height.

        Returns
        -------
        float
            The distance from the cube's centre to the eye.

        Raises
        ------
        ValueError
            If *band* is not inside the picture's height, or the part of
            it the coordinates are fitted into does not hold the
            picture's middle, where the camera aims.
        """
        bottom, top = band
        if not 0.0 <= bottom < top <= 1.0:
            raise ValueError(f"band must be (bottom, top) with 0 <= bottom < top <= 1, got {band}")
        # The fitted part of the band, in half heights from the
        # picture's middle (up positive).
        band_middle = bottom + top - 1.0
        fitted_half = FIT_FRACTION * (top - bottom)
        fitted_down = band_middle - fitted_half
        fitted_up = band_middle + fitted_half
        if not fitted_down < 0.0 < fitted_up:
            raise ValueError(
                f"band {band} must hold the picture's middle, where the "
                f"camera aims, inside FIT_FRACTION of it"
            )
        tangent = np.tan(np.radians(view_angle) / 2.0)
        matrices = self._view_matrices
        distances = []
        for coords in self.shown_coords:
            viewed = (coords - self.center) @ np.swapaxes(matrices, 1, 2)
            height = viewed[..., 1]
            needed = np.where(
                height >= 0.0, height / (tangent * fitted_up), height / (tangent * fitted_down)
            )
            if aspect is not None:
                reach_across = tangent * FIT_FRACTION * aspect
                needed = np.maximum(needed, np.abs(viewed[..., 0]) / reach_across)
            distances.append(float((viewed[..., 2] + needed).max()))
        # The cube's corner nearest the eye stands half_span * sum|w_k|
        # from its centre along the unit direction w towards the eye.
        towards_eye = matrices[:, 2]
        cube_corner_per_frame = self.half_span * np.abs(towards_eye).sum(axis=1)
        return max(*distances, float(cube_corner_per_frame.max()))

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
        self,
        clip_to_box: bool = False,
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
        self,
        points: npt.NDArray[np.float64],
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
            points, self.view_matrix(frame), (self.lo + self.hi) / 2.0, self.view_half, resolution
        )

    @cached_property
    def view_half(self) -> tuple[float, float]:
        """Half extents of the framing box on screen, in view units:
        the largest over every frame of the camera's schedule."""
        corners = box_corners(self.lo, self.hi) - (self.lo + self.hi) / 2.0
        projected = corners @ np.swapaxes(self._view_matrices, 1, 2)
        return (float(np.abs(projected[..., 0]).max()), float(np.abs(projected[..., 1]).max()))

    @cached_property
    def _view_matrices(self) -> npt.NDArray[np.float64]:
        """One view matrix per scheduled frame, or a single one for a
        fixed camera, shape ``(F, 3, 3)``."""
        azimuths = np.array([self.azimuth]) if self.azimuths is None else self.azimuths
        return np.stack(
            [
                build_view_matrix(float(azimuth), self.elevation, self.up_axis)
                for azimuth in azimuths
            ]
        )


def make_viewport(
    views: Sequence[SkeletonView],
    *,
    framing: str = "still",
    motion: str | Turntable = "fixed",
    include_floor: bool = True,
    projection: str = "persp",
    fps: float | None = None,
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
    motion : {"fixed", "turntable", "follow"} or Turntable
        How the camera moves. ``"turntable"`` orbits once over the
        clip, a :class:`Turntable` once every ``period`` frames
        (:func:`turntable_azimuths`); ``"follow"`` tracks the
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
    fps : float, optional
        The rate the picture plays at. Read only by ``"follow"``, to
        time a view whose ``frame_time`` is 0 (unset); see
        :func:`compute_follow_azimuths`.

    Returns
    -------
    Viewport
    """
    if not views:
        raise ValueError("A viewport needs at least one view.")
    if framing not in _FRAMINGS:
        raise ValueError(f"Unknown framing {framing!r}. Choose from: {list(_FRAMINGS)}")
    if not isinstance(motion, Turntable) and motion not in _MOTIONS:
        raise ValueError(f"Unknown motion {motion!r}. Choose from: {list(_MOTIONS)}")

    first = views[0]
    num_frames = first.coords.shape[0]
    if motion == "turntable":
        motion = Turntable(period=num_frames)
    center, half_span = compute_unified_limits([v.coords for v in views])
    ground_side = min if first.up_sign > 0 else max
    floor_height = float(ground_side(v.floor_height for v in views))

    if motion == "follow":
        azimuths = compute_follow_azimuths(first, first.azimuth, fps=fps)
    elif isinstance(motion, Turntable):
        azimuths = turntable_azimuths(first.azimuth, num_frames, motion.period)
    else:
        azimuths = None
    if azimuths is not None and np.all(azimuths == azimuths[0]):
        azimuths = None

    if framing == "clip":
        lo, hi = _swept_box(
            np.concatenate([v.coords.reshape(-1, 3) for v in views]),
            first.up_index,
            floor_height if include_floor else None,
            rotating=azimuths is not None,
        )
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
        lo=lo,
        hi=hi,
        center=center,
        half_span=half_span,
        up=first.up,
        floor_height=floor_height,
        azimuth=float(first.azimuth),
        elevation=float(first.elevation),
        azimuths=azimuths,
        projection=projection,
        shown_coords=tuple(v.coords for v in views),
    )


def panel_viewports(
    views: Sequence[SkeletonView],
    **options: object,
) -> list[Viewport]:
    """One :class:`Viewport` per view, for the multi-panel backends.

    Each panel is framed, floored and scheduled from its own view;
    *options* are :func:`make_viewport`'s."""
    return [
        make_viewport([view], **options)  # type: ignore[arg-type]
        for view in views
    ]


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
    trajectory_half_span = float(np.maximum(global_max - center, center - global_min).max())
    half_span = max(max_body_span / 2.0, trajectory_half_span)
    half_span *= CUBE_MARGIN
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
    return _swept_box(view.coords.reshape(-1, 3), view.up_index, view.floor_height, rotating)


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
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )


def turntable_azimuths(
    base_azim: float,
    num_frames: int,
    period: float,
) -> npt.NDArray[np.float64]:
    """Per-frame azimuths of a camera turning one revolution every
    *period* frames, for *num_frames* frames.

    The trivial case of follow: a constant-rate azimuth ramp starting
    at the base camera angle, in degrees, not wrapped (a period
    shorter than the frames climbs past 360). With a period of
    *num_frames*, frame ``num_frames`` would be the first view again,
    so a looped playback does not hold the identical view for two
    frames.
    """
    return base_azim + np.arange(num_frames, dtype=np.float64) * (360.0 / period)


# The follow camera's smoothing window: a Gaussian of this standard
# deviation, in seconds of clip time, cut off at FOLLOW_TRUNCATE standard
# deviations. See compute_follow_azimuths for the choice and its
# alternatives.
FOLLOW_SIGMA = 0.5
FOLLOW_TRUNCATE = 3.0


def compute_follow_azimuths(
    view: SkeletonView,
    base_azim: float,
    *,
    fps: float | None = None,
) -> npt.NDArray[np.float64]:
    """Per-frame camera azimuths that follow the character's heading.

    The camera turns with the character's body: frame 0 is seen from
    ``base_azim``, and every later frame from ``base_azim`` plus the
    heading change since frame 0, measured around ``up_vector`` from
    the rotation of the body's left-to-right axis (the mean of the
    view's L/R joint pairs, projected onto the ground).

    That axis also swings with the gait: pelvis, shoulders, knees, feet
    and hands turn back and forth with every stride, by about 30
    degrees peak to peak on the bundled CMU walk. The camera follows the
    heading, not that sway, so the per-frame heading change is smoothed
    before it is used. The conventions of that smoothing:

    - **Heading, not direction of travel.** The body's axis is what is
      smoothed. The alternative, the direction the root travels, sways
      less with the gait, but it has no direction on a clip that turns
      in place and points sideways on a side step.
    - **A centred Gaussian window**, of standard deviation
      ``FOLLOW_SIGMA`` = 0.5 s, so that ±1 standard deviation spans one
      second, about one stride of a walk, cut off at
      ``FOLLOW_TRUNCATE`` = 3 standard deviations (±1.5 s). Centred, it
      follows a steady turn exactly and a real turn without delay, and
      the camera starts to turn a little before the character does.
      The alternative, a causal window over past frames only, never
      looks ahead but lags behind every turn.
    - **Seconds of clip time**, from the view's ``frame_time``, so the
      camera takes the same path at any playback rate. A view whose
      ``frame_time`` is 0 (unset) has no clip time; its window is then
      sized from ``fps``, the rate the picture plays at, which is the
      rate such a clip is drawn at. On such a clip, and only there, the
      camera's path depends on ``fps``. The alternative, refusing the
      clip as ghost spacing does, would take the follow camera away
      from a clip ``render(fps=...)`` otherwise draws, for a window
      whose exact width matters far less than a ghost's spacing.
    - **Unwrapped**: each frame's change is taken within ±180 degrees
      of the previous valid frame's, so a character that turns more
      than half a turn is followed all the way round rather than
      snapped back by a full turn.
    - **Valid frames only.** A frame whose left-to-right axis is
      degenerate (parallel to up) is not measured: its heading change
      is linearly interpolated between the valid frames around it, or
      held at the last valid frame's after it. The alternative, holding
      the last measurement across a gap too, agrees on a character that
      is not turning; on one that turns through the gap it freezes the
      camera and then makes it catch up in a step when measurement
      resumes, where interpolation turns it steadily through the gap.
      When frame 0 itself is not measured the camera stays fixed (see
      Returns).
    - **The ends are held**: the heading change is extended past each
      end of the clip by point reflection about the end frame (the
      ``padtype="odd"`` extension of ``scipy.signal.filtfilt``), so the
      first and last frames keep their heading (the last frame's
      measured one, or the last valid frame's when it cannot be
      measured), the net turn over the clip is kept exactly, and a
      steady turn is followed to the last frame. The alternative,
      averaging only the frames that exist, would move both ends
      toward the middle of the clip and flatten a turn at the ends. Near an end, the smoothing therefore
      weakens: on the walk, the last frame is caught mid-stride, and the
      camera turns about 13 degrees toward it over the last half
      second.

    A clip shorter than the window is smoothed with the window cut to
    the clip's length on each side.

    Parameters
    ----------
    view : SkeletonView
        The skeleton's coords for the whole clip plus its L/R pairs,
        signed up vector and ``frame_time``.
    base_azim : float
        The frame-0 azimuth in degrees (from :func:`get_camera_angles`).
    fps : float, optional
        The rate the picture plays at. Read only when the view's
        ``frame_time`` is 0 (unset), to size the window.

    Returns
    -------
    azimuths : ndarray of shape (F,)
        Azimuth in degrees for every frame, unwrapped, so it may leave
        ``[-180, 180)``. Frame 0 is the reference every change is
        measured from, so when frame 0 is degenerate, or the rig has no
        L/R pairs, there is no reference and the whole sequence stays at
        ``base_azim``, later valid frames included: a fixed camera, not
        a partial follow.

    Raises
    ------
    ValueError
        If the view's ``frame_time`` is 0 and no ``fps`` is given,
        unless frame 0's left-to-right axis cannot be measured (the
        camera is then fixed, and needs no rate).
    """
    from ..tools import _leftward_units_from_pairs

    num_frames = view.coords.shape[0]
    leftward, valid = _leftward_units_from_pairs(view.coords, view.lr_pairs, view.up_vector)
    if not valid[0]:
        return np.full(num_frames, float(base_azim))

    heading_change = _heading_change(leftward, valid, view.up_vector)
    if view.frame_time > 0:
        frames_per_second = 1.0 / view.frame_time
    elif fps is not None:
        frames_per_second = float(fps)
    else:
        raise ValueError(
            "The follow camera smooths the heading over seconds of clip "
            "time, and this view's frame_time is 0 (unset): pass fps, "
            "the rate the picture plays at."
        )
    smoothed = _gaussian_smooth_held_ends(
        heading_change, FOLLOW_SIGMA * frames_per_second, FOLLOW_TRUNCATE
    )
    return base_azim + smoothed


def _heading_change(
    leftward: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    up_vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Unwrapped heading change since frame 0, in degrees, per frame.

    The signed angle rotating frame 0's leftward unit onto each frame's
    around *up_vector*, unwrapped over the valid frames and linearly
    interpolated across the others. Frame 0 must be valid.
    """
    left_0 = leftward[0]
    cos_a = np.clip(leftward @ left_0, -1.0, 1.0)
    sin_a = np.cross(left_0, leftward) @ up_vector
    change = np.degrees(np.arctan2(sin_a, cos_a))

    valid_frames = np.flatnonzero(valid)
    unwrapped = np.unwrap(change[valid_frames], period=360.0)
    return np.interp(np.arange(len(change)), valid_frames, unwrapped)


def _gaussian_smooth_held_ends(
    signal: npt.NDArray[np.float64],
    sigma_frames: float,
    truncate: float,
) -> npt.NDArray[np.float64]:
    """Centred Gaussian smoothing of *signal*, its end values held.

    The signal is extended past each end by point reflection about the
    end sample, which keeps both end values and any straight line
    exactly. The window is cut at *truncate* standard deviations, and
    at the signal's length on each side.
    """
    reach = min(int(np.ceil(truncate * sigma_frames)), len(signal) - 1)
    if reach < 1:
        return signal.copy()
    offsets = np.arange(-reach, reach + 1)
    weights = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    weights /= weights.sum()
    extended = np.pad(signal, reach, mode="reflect", reflect_type="odd")
    return np.convolve(extended, weights, mode="valid")


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
    eye_dir = np.roll([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis_idx - 2)

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

    scales = [
        (extent * FIT_FRACTION) / (2.0 * half)
        for extent, half in zip((w, h), view_half)
        if half > 1e-8
    ]
    scale = min(scales) if scales else 1.0

    px = viewed[:, 0] * scale + w / 2.0
    py = h / 2.0 - viewed[:, 1] * scale  # flip y for image coords

    return np.stack([px, py], axis=-1).astype(np.int32)
