"""Shared helpers for all visualization backends.

Pure-data operations: skeleton topology, bounding boxes, camera math,
orthographic projection, and the :class:`Scene` container every backend
consumes. No plotting library imports.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bvh import Bvh

UP_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


# ---------------------------------------------------------------------------
# Scene container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkeletonView:
    """Everything a backend needs to draw one skeleton in its own panel.

    A dumb data container — no plotting imports, no behavior beyond
    field access. Camera and bounding box are per-view because
    side-by-side comparisons of skeletons with different up or forward
    axes need each panel oriented and framed independently.

    The view is complete: every fact a backend needs about the skeleton
    (its timing, names, rest pose, left/right pairing, orientation and
    chain classification) is a field, so a view built from plain arrays
    with no :class:`Bvh` behind it draws exactly like one made by
    :func:`make_scene`. Frame-indexed fields (``coords``,
    ``root_heading``) always share their first axis.

    Conventions the fields follow:

    - ``root_heading`` is ``[sin(theta), cos(theta)]`` of
      :func:`~pybvh.analysis.root_trajectory`'s heading angle
      ``theta = atan2(f[b], f[a])``, where ``f`` is the rest-pose
      forward rotated by the root rotation and ``(a, b)`` are the two
      ground axes in natural ``x, y, z`` order with the up axis removed
      (``(x, z)`` for a y-up rig). It is a ground-plane angle, not a
      signed rotation about the up vector: for ``+y`` up, a positive
      rotation about ``+y`` decreases it. Orientation-derived and
      continuous; ``None`` when the coords are not the clip's.
    - ``forward_axis`` is a different quantity: the facing of coordinate
      row 0 (not necessarily clip frame 0) computed from the L/R joint
      geometry and snapped to the dominant signed axis, the value the
      ``"front"`` camera preset is derived from. Follow-camera math
      never reads it; it reads ``coords``, ``lr_pairs`` and
      ``up_vector`` directly.
    - ``lr_pairs`` are joint pairs only, in node index space, the pairs
      the facing geometry averages; end-site pairs are deliberately
      excluded so follow azimuths match the Bvh path bit for bit.
    - ``up_axis``, ``up_vector`` and ``up_sign`` describe one signed
      axis three ways and must agree: geometry reads ``up_vector``,
      floor and shadow offsets read ``up_axis`` and ``up_sign``. A view
      built from arrays is responsible for their consistency; nothing
      checks it yet.
    - Arrays are borrowed, not owned: a Scene operation may share array
      storage with its source (``subsampled`` shares ``coords`` and
      ``root_heading``; ``offset`` allocates new coords and shares the
      heading), and ``frozen`` forbids field reassignment only. Change
      a Scene through its operations; never mutate a view's arrays in
      place.
    """

    coords: npt.NDArray[np.float64]        # (F, N, 3)
    bones: list[tuple[int, int]]           # (parent_idx, child_idx) pairs
    label: str | None
    center: npt.NDArray[np.float64]        # (3,) cubic-box center
    half_span: float                       # cubic-box half side
    azimuth: float                         # degrees
    elevation: float                       # degrees
    up_axis: str                           # 'x' | 'y' | 'z'
    floor_height: float                    # scene ground along up_axis
    frame_time: float                      # seconds per frame
    node_names: list[str]                  # parallel to the N axis
    rest_coords: npt.NDArray[np.float64]   # (N, 3) rest pose, root at origin
    lr_pairs: npt.NDArray[np.intp]         # (P, 2) joint L/R pairs in node index space, the facing geometry's; (0, 2) if none
    up_vector: npt.NDArray[np.float64]     # (3,) signed world-up unit vector
    forward_axis: str                      # snapped facing of coords row 0, e.g. '+y'
    bone_chains: list[str]                 # chain name per bone, parallel to ``bones``
    root_heading: npt.NDArray[np.float64] | None  # (F, 2) [sin, cos] or None
    up_sign: float = 1.0                   # +1 for '+y' etc., -1 for '-y'

    @property
    def up_index(self) -> int:
        return UP_AXIS_INDEX.get(self.up_axis, 2)

    def below_floor(self, distance: float) -> float:
        """The coordinate *distance* visually below the floor plane.

        "Below" follows the signed up axis: for a '-y'-up rig the
        ground sits at the coordinate MAXIMUM, so below means +y.
        Backends use this for z-fighting nudges and shadow offsets so
        negative-up rigs get their floor under the feet, not overhead.
        """
        return self.floor_height - self.up_sign * distance


@dataclass(frozen=True)
class Scene:
    """A prepared visualization: the single input every backend consumes.

    Multi-panel backends (matplotlib, OpenCV) iterate :attr:`views`;
    single-scene backends (k3d, vedo) call :meth:`unified_box` for the
    one shared bounding box and take the camera from ``views[0]``.
    """

    views: list[SkeletonView]

    @property
    def num_frames(self) -> int:
        return int(self.views[0].coords.shape[0])

    @property
    def num_skeletons(self) -> int:
        return len(self.views)

    @property
    def labels(self) -> list[str | None] | None:
        """Per-view labels, or ``None`` when no view is labelled."""
        labels = [v.label for v in self.views]
        return labels if any(lbl is not None for lbl in labels) else None

    @property
    def frame_time(self) -> float:
        """Seconds per frame, from the first view.

        The animated entry points offer to resample the clips to a
        common rate before the Scene is built (``match_fps``), but the
        default only warns on a mismatch, so views can disagree. The
        first view's timing then drives playback and the others play at
        its rate. The static entry points never read it.
        """
        return self.views[0].frame_time

    def unified_box(self) -> tuple[npt.NDArray[np.float64], float]:
        """Cubic bounding box covering every view's coords."""
        return compute_unified_limits([v.coords for v in self.views])

    def subsampled(self, step: int) -> Scene:
        """Every ``step``-th frame of every view, as a new Scene.

        Slices ``coords`` and every other frame-indexed field
        (``root_heading``) with ``[::step]`` and scales ``frame_time`` by
        ``step``, so the kept frames stay at their original moments in
        time; up to ``step - 1`` trailing frames are dropped, so the
        span from the first to the last kept sample can shorten by that
        many original intervals and the last frame is not necessarily
        kept. Each view's box is recomputed from the kept
        frames. The floor is kept as is: it describes the whole clip
        (the canonical floor is a clip-wide estimate, the coords floor
        the full clip's extreme), and a plane that sits where the full
        clip's ground was is the honest one for a preview of it. The
        alternative, recomputing from the kept frames, would move the
        ground between the full and the subsampled view of one clip.
        """
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}.")
        views = []
        for v in self.views:
            coords = v.coords[::step]
            center, half_span = compute_unified_limits([coords])
            heading = (None if v.root_heading is None
                       else v.root_heading[::step])
            views.append(dataclasses.replace(
                v, coords=coords, center=center, half_span=half_span,
                frame_time=v.frame_time * step, root_heading=heading))
        return Scene(views=views)

    def offset(self, offsets: list[npt.NDArray[np.float64]]) -> Scene:
        """Translate each view by its own ``(3,)`` vector, as a new Scene.

        Moves ``coords`` and ``center``, and ``floor_height`` by the
        offset's component along the view's up axis, so the plane stays
        under the feet. Orientation, heading and timing are
        translation-invariant and are kept.
        """
        if len(offsets) != len(self.views):
            raise ValueError(
                f"Expected {len(self.views)} offsets, got {len(offsets)}.")
        views = []
        for v, off in zip(self.views, offsets):
            off = np.asarray(off, dtype=np.float64).reshape(3)
            views.append(dataclasses.replace(
                v,
                coords=v.coords + off,
                center=v.center + off,
                floor_height=v.floor_height + float(off[v.up_index])))
        return Scene(views=views)

    def spread(self, spacing: float | str) -> Scene:
        """Offset the views laterally so skeletons sharing one 3-D scene
        do not overlap.

        For the single-scene backends (k3d, vedo); multi-panel backends
        draw each view in its own axes and never need it. The lateral
        axis is the one that is neither the first view's up axis nor its
        forward axis at frame 0. ``"auto"`` spaces by 1.2 × the first
        view's lateral extent (at least 0.1 scene units); a float is
        used directly, in scene units. View ``k`` moves by
        ``k × spacing`` in the positive lateral direction. Whether to
        spread at all is the caller's policy (``play`` respects raw
        world coordinates under ``"auto"``).
        """
        if len(self.views) <= 1:
            return self
        first = self.views[0]
        up_idx = first.up_index
        fwd_idx = UP_AXIS_INDEX.get(first.forward_axis[1], 0)
        lat_idx = next(i for i in range(3) if i != up_idx and i != fwd_idx)

        if spacing == "auto":
            lateral = first.coords[..., lat_idx]
            width = float(lateral.max() - lateral.min())
            effective = max(width, 0.1) * 1.2
        else:
            effective = float(spacing)
        if effective == 0.0:
            return self

        unit = np.zeros(3)
        unit[lat_idx] = 1.0  # always the positive lateral direction
        return self.offset(
            [unit * k * effective for k in range(len(self.views))])


def make_scene(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    camera: str | tuple[float, float],
    labels: list[str] | None,
    *,
    canonical_floor: bool = True,
    clip_frames: int | slice | None = None,
) -> Scene:
    """Assemble a :class:`Scene` from parallel per-skeleton data.

    The Bvh -> Scene adapter. Together with :func:`normalize_input`,
    which turns a Bvh into coords, it is where bvhplot reads a
    :class:`Bvh`; backends only ever see the Scene. Computes, per
    view, the topology, cubic bounding box, camera angles, floor height
    and every skeleton fact the backends draw from (timing, node names,
    rest pose, L/R pairs, orientation, chain classification, root
    heading), so the views are complete once this returns.

    ``clip_frames`` says which frames of the clip the coords are, so
    the clip-derived, frame-indexed facts can be aligned to them. Root
    heading (the ``[sin, cos]`` columns of
    :func:`~pybvh.analysis.root_trajectory`) is the only one today. An
    ``int`` names the single clip frame the coords hold (NumPy
    semantics, negative from the end) and requires one-row coords; a
    ``slice`` selects the frames, and the result is truncated or
    last-row padded to the coords' frame count, mirroring
    :func:`align_frame_counts`, so ``slice(None)`` is the whole clip.
    The default ``None`` means the coords are not the clip's (a rest
    pose, a caller-supplied array, a transformed copy) and no clip frame
    corresponds to them; ``root_heading`` is then ``None`` rather than
    a number that would be wrong. The default is the safe one on
    purpose: a caller has to vouch for the coords' provenance to get
    frame-indexed clip facts attached to them.

    Floor convention: with ``canonical_floor=True`` the floor is the
    cached :attr:`Bvh.floor_height` — the robust 2nd percentile over all
    nodes, end sites included, so the plane sits under the whole
    skeleton rather than at a joint centre. Valid for world-frame
    coords, and for ``centered="first"`` coords because first-centering
    is ground-plane-only (heights stay in world units). An outlier-low
    frame can therefore dip a node slightly below the drawn plane; the
    alternative (the true minimum) would let a single glitched frame
    sink the plane for the whole clip. With ``canonical_floor=False``
    (root-relative or caller-supplied coords) the floor is the minimum
    up-coordinate of the coords in use — the clip-wide estimate does not
    apply to coords in another frame of reference, and the true minimum
    is the safe choice for a pose whose own extent is all we have.

    This is the scene ground, not the reference ``foot_contacts``
    measures clearance against; see :attr:`Bvh.floor_height`.
    """
    from ..analysis import root_trajectory
    from ..tools import _facing_lr_pairs

    views: list[SkeletonView] = []
    for i, (b, coords) in enumerate(zip(bvh_list, coords_list)):
        center, half_span = compute_unified_limits([coords])
        azimuth, elevation, up_axis, forward_axis = _camera_angles_and_forward(
            b, coords[0], camera)
        up_idx = UP_AXIS_INDEX.get(up_axis, 2)
        up_sign = float(b.up_axis.sign)
        bones = get_skeleton_lines(b)

        if clip_frames is None:
            root_heading = None
        else:
            heading = root_trajectory(b)[:, 2:4]
            if isinstance(clip_frames, slice):
                root_heading = _align_rows(heading[clip_frames],
                                           coords.shape[0])
            else:
                root_heading = heading[clip_frames][np.newaxis]
            if root_heading.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"clip_frames={clip_frames!r} names one clip frame but "
                    f"the coords hold {coords.shape[0]} frames; pass a slice "
                    f"for multi-frame coords.")
        if canonical_floor:
            floor_height = float(b.floor_height)
        else:
            # The ground is the signed-lowest point: the coordinate
            # minimum for a positive up axis, the MAXIMUM for a
            # negative one (where larger values point downward).
            extreme = coords[..., up_idx]
            floor_height = float(extreme.min() if up_sign > 0
                                 else extreme.max())
        views.append(SkeletonView(
            coords=coords,
            bones=bones,
            label=labels[i] if labels and i < len(labels) else None,
            center=center,
            half_span=half_span,
            azimuth=azimuth,
            elevation=elevation,
            up_axis=up_axis,
            floor_height=floor_height,
            frame_time=float(b.frame_time),
            node_names=[node.name for node in b.nodes],
            rest_coords=b.rest_pose_positions(),
            lr_pairs=_facing_lr_pairs(b),
            up_vector=np.asarray(b.up_axis.vector, dtype=np.float64),
            forward_axis=forward_axis,
            bone_chains=_chain_per_bone(get_bone_chains(b), len(bones)),
            root_heading=root_heading,
            up_sign=up_sign,
        ))
    return Scene(views=views)


def _align_rows(
    arr: npt.NDArray[np.float64], num_frames: int,
) -> npt.NDArray[np.float64]:
    """Truncate or last-row pad ``arr`` along axis 0 to ``num_frames``.

    The same rule :func:`align_frame_counts` applies to coords, so a
    frame-indexed field follows its coords through either alignment.
    """
    if arr.shape[0] >= num_frames:
        return arr[:num_frames]
    pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _chain_per_bone(chains: dict[str, list[int]], n_bones: int) -> list[str]:
    """Invert :func:`get_bone_chains`' chain -> bones map into one name per bone.

    Bones no chain claims read as ``"spine"``, the documented fallback.
    """
    per_bone = ["spine"] * n_bones
    for chain_name, bone_indices in chains.items():
        for i in bone_indices:
            per_bone[i] = chain_name
    return per_bone


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

    Notes
    -----
    The parent-first view of :attr:`Bvh.node_edges`, so the drawn bones
    are the same topology forward kinematics posed — including on
    skeletons whose node names repeat, where a name-keyed lookup would
    draw one bone twice and omit another.
    """
    return [(parent, child) for child, parent in bvh.node_edges]


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
        - ``None``: all frames (spatial coords for entire motion).
        - int: single frame index (NumPy semantics — negative counts
          from the end).
        - 2-D array ``(N, 3)``: single frame of spatial coordinates
          (only valid when *bvh* is a single Bvh).
        - 3-D array ``(F, N, 3)``: pre-computed spatial coordinates
          (only valid when *bvh* is a single Bvh).
    centered : str
        Centering mode passed to ``bvh.node_positions()``.

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

    if frames is None:
        # All frames for each Bvh
        for b in bvh_list:
            coords = b.node_positions(centered=centered)
            if coords.ndim == 2:
                coords = coords[np.newaxis]  # (N, 3) -> (1, N, 3)
            coords_list.append(coords)

    elif isinstance(frames, int):
        # Single frame index
        for b in bvh_list:
            coords = b.node_positions(frame=frames, centered=centered)
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
# Camera angles
# ---------------------------------------------------------------------------

def get_camera_angles(
    bvh: Bvh,
    ref_frame: npt.NDArray[np.float64],
    camera: str | tuple[float, float] = "front",
) -> tuple[float, float, str]:
    """Resolve a camera specification to (azimuth, elevation, up_axis).

    Thin view of :func:`_camera_angles_and_forward`, which also returns
    the forward axis the azimuth was derived from.

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
    azimuth, elevation, up_axis, _ = _camera_angles_and_forward(
        bvh, ref_frame, camera)
    return azimuth, elevation, up_axis


def _camera_angles_and_forward(
    bvh: Bvh,
    ref_frame: npt.NDArray[np.float64],
    camera: str | tuple[float, float],
) -> tuple[float, float, str, str]:
    """:func:`get_camera_angles` plus the signed forward axis string.

    ``forward_axis`` (e.g. ``'+y'``) is the character's facing at
    *ref_frame* snapped to the dominant signed world axis
    (:func:`~pybvh.tools._compute_forward_at`: leftward from the L/R
    joint geometry, crossed with up, then snapped). When that geometry
    is degenerate (leftward parallel to up, or no L/R pairs) it falls
    back to the rest-pose leftward and then to a fixed per-up-axis
    default, so the string is always one of the six axes, never the
    continuous direction. It is the fact the ``"front"`` preset turns
    into an azimuth; :func:`make_scene` keeps it on the view so lateral
    spacing and the camera never disagree about which way the skeleton
    faces.
    """
    from ..tools import _compute_forward_at, extract_sign

    # World up comes from the Bvh property (auto-detected with manual
    # override). Forward is computed from the given reference frame's
    # actual joint positions, so it tracks the character's orientation
    # as the animation plays (not just the rest-pose topology).
    up_ax = bvh.world_up
    forward_ax = _compute_forward_at(bvh, ref_frame, up_ax)
    up_char = up_ax[1]                   # 'y'
    up_positive = extract_sign(up_ax)
    fwd_char = forward_ax[1]
    fwd_positive = extract_sign(forward_ax)

    if isinstance(camera, tuple):
        return float(camera[0]), float(camera[1]), up_char, forward_ax

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

    # A negative forward (e.g. '-y' instead of '+y') flips the camera to
    # the opposite side of the skeleton. Apply regardless of whether the
    # forward axis matches the up's default front axis.
    if not fwd_positive:
        base_azim += 180.0

    if camera == "front":
        return base_azim, base_elev, up_char, forward_ax
    elif camera == "side":
        return base_azim + 90.0, base_elev, up_char, forward_ax
    elif camera == "top":
        return base_azim, 90.0, up_char, forward_ax
    else:
        raise ValueError(
            f"Unknown camera preset {camera!r}. "
            f"Use 'front', 'side', 'top', or (azimuth, elevation).")


# ---------------------------------------------------------------------------
# Ghost trails and trajectory traces (shared backend conventions)
# ---------------------------------------------------------------------------

# The trace and ghost look must be identical across backends: one
# render() call, different sinks. These are the single definitions.
TRACE_COLOR = "#7A8090"
TRACE_BLEND = 0.9          # blended toward the background at this weight
GHOST_WIDTH_FACTOR = 0.75  # ghosts draw thinner than the live skeleton


def ghost_schedule(
    style: Style,
    frame_time: float,
    n_ghosts: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    """Frame lag and fade weights for a ghost trail.

    Ghost slot ``j`` trails the live pose by ``(j+1) * lag`` frames;
    weights fade from 0.32 (nearest, darkest) to 0.15 (oldest).
    """
    lag = max(1, round(style.ghost_spacing / frame_time))
    weights = np.linspace(0.32, 0.15, n_ghosts)
    return lag, weights


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
# Color palettes
# ---------------------------------------------------------------------------

# RGB is the canonical channel order everywhere in bvhplot; the OpenCV
# backend converts to BGR at its own border (its channel-order quirk
# stays its own concern).

# Per-skeleton comparison palette (multi-skeleton figures)
PALETTE_RGB = [
    (50, 120, 255),   # blue
    (220, 50, 50),    # red
    (50, 180, 50),    # green
    (50, 130, 200),   # teal
    (200, 100, 50),   # orange
    (200, 50, 200),   # magenta
]
PALETTE_MPL = [(r / 255, g / 255, b / 255) for (r, g, b) in PALETTE_RGB]

# Per-chain palette (single-skeleton figures): left = warm, right = cool,
# spine dark — side is encoded by temperature, chain by shade. Derived
# from the Okabe-Ito colorblind-safe palette.
CHAIN_COLORS = {
    "spine": "#3A3F4A",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}
# Dark-background variant: same warm/cool encoding, spine and joints
# lightened so they read against a near-black ground.
CHAIN_COLORS_DARK = {
    "spine": "#C8CCD6",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}


# ---------------------------------------------------------------------------
# Bone chain classification (for per-chain coloring)
# ---------------------------------------------------------------------------

def get_bone_chains(bvh: Bvh) -> dict[str, list[int]]:
    """Classify each drawn bone into a kinematic chain for coloring.

    Chains are ``"spine"`` (every unpaired node — torso, neck, head),
    ``"l_arm"`` / ``"r_arm"``, and ``"l_leg"`` / ``"r_leg"``. Sides come
    from :attr:`Bvh.node_lr_pairs` (the same left/right detection that
    powers ``mirror``); legs are told apart from arms by walking the
    topology from the auto-detected foot joints (paired ancestors of a
    foot, the foot itself, and everything below it), so no joint-name
    heuristics beyond what those two detectors already encode.

    Returns
    -------
    chains : dict[str, list[int]]
        Maps chain name to indices into :func:`get_skeleton_lines`'s
        bone list. Every bone index appears in exactly one chain.

    Notes
    -----
    A limb is the maximal run of *paired* nodes: a bone belongs to a
    limb chain only when both its endpoints are paired, so the
    junction bones connecting the torso to a limb (an unpaired parent
    like ``Spine3`` or ``Hips`` to a paired child like a shoulder or
    hip joint) stay in ``"spine"`` — limbs start at the shoulder ball
    and hip socket, the TEMOS/MDM capsule-figure convention. The
    alternative (a bone belongs to its child node's chain, coloring
    those connectors as limbs) diverges visibly on rigs whose shoulder
    joints sit on the spine axis: the connector is then a vertical
    segment lying *on* the spine, and limb-coloring it paints a stray
    arm-colored "vertebra" onto the torso.

    Fallbacks: with no left/right pairs (``node_lr_pairs is None``)
    every bone lands in ``"spine"`` — the caller should draw a single
    color. With pairs but no detectable feet, every paired-to-paired
    bone is classified as an arm: side coloring survives, the arm/leg
    shade distinction does not.
    """
    bones = get_skeleton_lines(bvh)
    pairs = bvh.node_lr_pairs
    if pairs is None:
        return {"spine": list(range(len(bones)))}

    left_nodes = {l for l, _ in pairs}
    right_nodes = {r for _, r in pairs}

    parent_of = {child: parent for child, parent in bvh.node_edges}
    children_of: dict[int, list[int]] = {}
    for child, parent in bvh.node_edges:
        children_of.setdefault(parent, []).append(child)

    # Leg nodes: from each detected foot, everything below it plus the
    # paired ancestors above it (stopping at the first unpaired node,
    # i.e. where the leg meets the torso).
    leg_nodes: set[int] = set()
    for foot_name in bvh.auto_detect_foot_joints():
        foot_idx = bvh.node_index.get(foot_name)
        if foot_idx is None:
            continue
        stack = [foot_idx]
        while stack:
            n = stack.pop()
            leg_nodes.add(n)
            stack.extend(children_of.get(n, []))
        n2 = parent_of.get(foot_idx)
        while n2 is not None and (n2 in left_nodes or n2 in right_nodes):
            leg_nodes.add(n2)
            n2 = parent_of.get(n2)

    chains: dict[str, list[int]] = {
        "spine": [], "l_arm": [], "l_leg": [], "r_arm": [], "r_leg": []}
    paired = left_nodes | right_nodes
    for i, (parent, child) in enumerate(bones):
        if child in left_nodes:
            side = "l"
        elif child in right_nodes:
            side = "r"
        else:
            chains["spine"].append(i)
            continue
        if parent not in paired:
            # Junction bone (torso -> limb): stays spine-colored, the
            # limb starts at the first fully-paired bone (see Notes).
            chains["spine"].append(i)
            continue
        limb = "leg" if child in leg_nodes else "arm"
        chains[f"{side}_{limb}"].append(i)
    return {name: idxs for name, idxs in chains.items() if idxs}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Preset field values. "paper" is the publication-grade default;
# "debug" reproduces the pre-0.9.0 output exactly (single blue, full
# axes, no floor); "dark" is the paper look on a near-black ground.
_STYLE_PRESETS: dict[str, dict[str, object]] = {
    "paper": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#23262E",
        floor="solid",
        floor_alpha=0.85,
        background="white",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
    "debug": dict(
        bone_width=2.5,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="single",
        chain_colors=CHAIN_COLORS,
        joint_markers=False,
        joint_size=9.0,
        joint_color="#23262E",
        floor=None,
        floor_alpha=0.85,
        background="white",
        axes="full",
        projection="persp",
        dpi=None,
        supersample=1,
        shadow=False,
        ghost_spacing=0.3,
    ),
    "dark": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS_DARK,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#E8E8EC",
        floor="solid",
        floor_alpha=0.85,
        background="#16181D",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
}

_VALID_COLOR_MODES = {"auto", "chains", "skeleton", "single"}
_VALID_FLOORS = {None, "solid", "grid", "checker"}
_VALID_AXES = {"off", "full"}
_VALID_PROJECTIONS = {"persp", "ortho"}


@dataclass(frozen=True, init=False)
class Style:
    """Visual styling for every bvhplot function.

    Construct from a preset name plus any field overrides::

        Style("paper")                    # the defaults
        Style("paper", floor=None)        # paper look, no ground plane
        Style("dark", bone_width=4.0)

    Presets: ``"paper"`` (publication-grade default: ground plane,
    per-chain colors, joint markers, axes off), ``"debug"`` (the
    pre-0.9.0 look: single blue skeleton, full axes and ticks, no
    floor — for coordinate inspection), ``"dark"`` (paper on a
    near-black ground, for slides and project pages).

    Fields split into two documented groups. **Look** fields apply to
    every figure/video output (matplotlib, OpenCV, vedo offscreen):
    ``bone_width``, ``bone_color``, ``color_mode``, ``chain_colors``,
    ``joint_markers``, ``joint_size``, ``joint_color``, ``floor``,
    ``floor_alpha``, ``background``, ``axes``, ``projection``,
    ``ghost_spacing`` (seconds between the faded trailing poses that
    ``render(ghost=...)`` draws). The two *interactive viewers* apply
    the subset that has meaning in a live window: background, bone
    width, and single-skeleton chain colors in both; floor kind in the
    vedo viewer, where ``"checker"`` falls back to ``"grid"``. Fields
    outside that subset (``axes``, ``projection``, ``joint_markers``,
    ...) do not alter the viewers. **Output** fields apply only where
    raster output is produced: ``dpi`` (matplotlib figures),
    ``supersample`` (OpenCV export), ``shadow`` (vedo offscreen
    renders).

    ``color_mode``: ``"auto"`` uses per-chain colors for a single
    skeleton and flat per-skeleton palette colors for multi-skeleton
    comparisons (the GT-vs-generated convention); ``"chains"`` forces
    chain colors everywhere; ``"skeleton"`` forces the flat palette;
    ``"single"`` draws one skeleton in ``bone_color`` (multi-skeleton
    still uses the palette — the pre-0.9.0 behavior).

    ``axes`` and ``floor`` are independent, so ``Style("paper",
    axes="full")`` keeps the ground plane — and on the matplotlib
    backend that plane paints over the axis panes and grid lines,
    because any floor forces manual draw order (mplot3d's computed
    z-order would wash the skeleton out under the semi-transparent
    plane). For clean coordinate-inspection axes use ``"debug"``, or
    add ``floor=None``.
    """

    bone_width: float
    bone_color: tuple[float, float, float]
    color_mode: str
    chain_colors: dict[str, str]
    joint_markers: bool
    joint_size: float
    joint_color: str
    floor: str | None
    floor_alpha: float
    background: str
    axes: str
    projection: str
    dpi: int | None
    supersample: int
    shadow: bool
    ghost_spacing: float

    def __init__(self, preset: str = "paper", **overrides: object) -> None:
        if preset not in _STYLE_PRESETS:
            raise ValueError(
                f"Unknown style preset {preset!r}. "
                f"Choose from: {sorted(_STYLE_PRESETS)}")
        self._assign_fields(dict(_STYLE_PRESETS[preset]), overrides)

    def _assign_fields(
        self,
        fields: dict[str, object],
        overrides: dict[str, object],
    ) -> None:
        """Shared construction path for __init__ and replace: unknown-
        field check, defensive copies, assignment, validation."""
        unknown = set(overrides) - set(fields)
        if unknown:
            raise TypeError(
                f"Unknown Style field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(fields)}")
        fields.update(overrides)
        for name, value in fields.items():
            # Copy mutable field values (chain_colors) so no instance
            # aliases the module-level preset dicts — mutating one
            # Style must never restyle every other figure.
            if isinstance(value, dict):
                value = dict(value)
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {sorted(_VALID_COLOR_MODES)}, "
                f"got {self.color_mode!r}")
        if self.floor not in _VALID_FLOORS:
            raise ValueError(
                f"floor must be one of "
                f"{sorted(f for f in _VALID_FLOORS if f)} or None, "
                f"got {self.floor!r}")
        if self.axes not in _VALID_AXES:
            raise ValueError(
                f"axes must be one of {sorted(_VALID_AXES)}, "
                f"got {self.axes!r}")
        if self.projection not in _VALID_PROJECTIONS:
            raise ValueError(
                f"projection must be one of {sorted(_VALID_PROJECTIONS)}, "
                f"got {self.projection!r}")
        if not self.bone_width > 0:
            raise ValueError(
                f"bone_width must be positive, got {self.bone_width}")
        if not (isinstance(self.supersample, int) and self.supersample >= 1):
            raise ValueError(
                f"supersample must be an integer >= 1, "
                f"got {self.supersample!r}")
        if not self.ghost_spacing > 0:
            raise ValueError(
                f"ghost_spacing must be positive (seconds), "
                f"got {self.ghost_spacing!r}")

    def replace(self, **overrides: object) -> Style:
        """A new Style with the given fields changed."""
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)}
        new = object.__new__(Style)
        new._assign_fields(fields, overrides)
        return new


def resolve_style(style: Style | str) -> Style:
    """Accept a preset name or a Style instance; return a Style."""
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        return Style(style)
    raise TypeError(
        f"style must be a Style or a preset name string, "
        f"got {type(style).__name__}")


def effective_color_mode(style: Style, n_skeletons: int) -> str:
    """Resolve ``"auto"``/``"single"`` to the concrete mode for n panels.

    The multi-skeleton auto-switch rule: comparisons get flat
    per-skeleton palette colors unless chains are forced explicitly.
    """
    if style.color_mode == "auto":
        return "chains" if n_skeletons == 1 else "skeleton"
    if style.color_mode == "single":
        return "single" if n_skeletons == 1 else "skeleton"
    return style.color_mode


def bone_colors_for_view(
    view: SkeletonView,
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> list:
    """Per-bone colors for one view, in matplotlib-friendly form.

    Each entry is a hex string or an RGB float tuple, parallel to
    ``view.bones``. The OpenCV backend converts these at its border.
    """
    n_bones = len(view.bones)
    mode = effective_color_mode(style, n_skeletons)
    if mode == "skeleton":
        return [PALETTE_MPL[view_index % len(PALETTE_MPL)]] * n_bones
    if mode == "single":
        return [style.bone_color] * n_bones
    # chains — a skeleton with no L/R pairs is all-"spine" in
    # view.bone_chains, i.e. a single dark color (the documented fallback).
    spine_color = style.chain_colors.get("spine", "#3A3F4A")
    return [style.chain_colors.get(chain_name, spine_color)
            for chain_name in view.bone_chains]
