"""The Scene every bvhplot backend consumes.

Pure data: :class:`SkeletonView`, :class:`Scene`, the operations that
change a Scene, and the array helpers they rest on. No plotting library
imports and no imports from the pybvh core; a Scene is built from a
:class:`~pybvh.bvh.Bvh` in :mod:`._from_bvh`, or from plain arrays.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

UP_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
_SIGNED_AXES = ('+x', '-x', '+y', '-y', '+z', '-z')


# ---------------------------------------------------------------------------
# Scene container
# ---------------------------------------------------------------------------

class GroundFrame:
    """What follows from a signed up axis and a floor height.

    Mixed into the two values that hold an ``up`` string and a
    ``floor_height``, :class:`SkeletonView` and
    :class:`~._viewport.Viewport`, so both read the up axis the same
    way.
    """

    up: str
    floor_height: float

    @property
    def up_axis(self) -> str:
        """The up axis letter, ``'x'``, ``'y'`` or ``'z'``, sign dropped."""
        return self.up[1]

    @property
    def up_index(self) -> int:
        """The up coordinate's column in a ``(..., 3)`` position array."""
        return UP_AXIS_INDEX[self.up_axis]

    @property
    def up_sign(self) -> float:
        """``+1.0`` for ``'+y'``, ``-1.0`` for ``'-y'``."""
        return -1.0 if self.up[0] == '-' else 1.0

    @property
    def up_vector(self) -> npt.NDArray[np.float64]:
        """The unit vector pointing up, sign included, shape ``(3,)``.

        A fresh array on every access."""
        vector = np.zeros(3, dtype=np.float64)
        vector[self.up_index] = self.up_sign
        return vector

    @property
    def ground_axes(self) -> tuple[int, int]:
        """The columns of the two ground coordinates, in ``x, y, z``
        order with the up axis removed (``(0, 2)`` for a y-up rig)."""
        first, second = (i for i in range(3) if i != self.up_index)
        return first, second

    def below_floor(self, distance: float) -> float:
        """The coordinate *distance* visually below the floor plane.

        "Below" follows the signed up axis: for a '-y'-up rig the
        ground sits at the coordinate MAXIMUM, so below means +y.
        Backends use this for z-fighting nudges and shadow offsets so
        negative-up rigs get their floor under the feet, not overhead.
        """
        return self.floor_height - self.up_sign * distance


@dataclass(frozen=True)
class SkeletonView(GroundFrame):
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

    A view is checked when it is built: the shapes and lengths of its
    fields must agree with each other, every bone and L/R index must be
    an integer that names a node, and the two axes must be signed axis
    strings. An inconsistent view raises ``ValueError`` here, at
    construction, rather than drawing something wrong later. Only
    consistency is checked, never plausibility: a floor above the head
    or a rest pose unrelated to the coords is the builder's business.

    ``frame_time`` follows :attr:`Bvh.frame_time`: seconds per frame,
    with ``0`` meaning "unset", which a rest pose or a still of a
    Bvh built in memory legitimately carries. The static entry points
    never read it; the animated ones divide by it and need a positive
    value. Only a negative or non-finite value is rejected here.

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
    - ``up`` is the one statement of which way is up: a signed axis
      string (``'+y'``, ``'-z'``), the form :attr:`Bvh.world_up` and
      ``forward_axis`` use. The letter (``up_axis``), the column
      (``up_index``), the sign (``up_sign``) and the unit vector
      (``up_vector``) are derived from it, so they cannot disagree.
      The alternative, storing the vector, would admit an off-axis or
      unnormalised up that no backend can draw: matplotlib's vertical
      axis and the floor plane are axis-aligned.
    - ``lr_pairs`` are joint pairs only, in node index space, the pairs
      the facing geometry averages; end-site pairs are deliberately
      excluded so follow azimuths match the Bvh path bit for bit.
    - Arrays are read-only and borrowed. A view stores a read-only
      NumPy view of each array it is given (``coords``, ``center``,
      ``rest_coords``, ``lr_pairs``, ``root_heading``), so writing into
      one through the view raises. That matters because storage is
      shared: no array is copied at construction, and a Scene operation
      may share storage with its source (``subsampled`` shares
      ``coords`` and ``root_heading``; ``offset`` allocates new coords
      and shares the heading), so a write through one Scene would
      change another while both boxes went stale. The alternative,
      copying, would double the memory of every clip drawn. The array
      the caller passed in keeps its own flags; writing into *it*
      afterwards still changes what the view shows, which is the
      caller's to avoid. A copied or unpickled view is protected
      like the original. The lists (``bones``, ``node_names``,
      ``bone_chains``) are not protected: treat them as immutable.
      Change a Scene through its operations.
    """

    coords: npt.NDArray[np.float64]        # (F, N, 3)
    bones: list[tuple[int, int]]           # (parent_idx, child_idx) pairs
    label: str | None
    center: npt.NDArray[np.float64]        # (3,) cubic-box center
    half_span: float                       # cubic-box half side
    azimuth: float                         # degrees
    elevation: float                       # degrees
    up: str                                # signed world-up axis, e.g. '+y', '-z'
    floor_height: float                    # scene ground along the up axis
    frame_time: float                      # seconds per frame
    node_names: list[str]                  # parallel to the N axis
    rest_coords: npt.NDArray[np.float64]   # (N, 3) rest pose, root at origin
    lr_pairs: npt.NDArray[np.intp]         # (P, 2) joint L/R pairs in node index space, the facing geometry's; (0, 2) if none
    forward_axis: str                      # snapped facing of coords row 0, e.g. '+y'
    bone_chains: list[str]                 # chain name per bone, parallel to ``bones``
    root_heading: npt.NDArray[np.float64] | None  # (F, 2) [sin, cos] or None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords)
        if coords.ndim != 3 or coords.shape[2] != 3 or 0 in coords.shape:
            raise ValueError(
                f"coords must have shape (F, N, 3) with at least one frame "
                f"and one node, got {coords.shape}.")
        num_frames, num_nodes = coords.shape[:2]

        for name in ("up", "forward_axis"):
            value = getattr(self, name)
            if value not in _SIGNED_AXES:
                raise ValueError(
                    f"{name} must be one of {', '.join(_SIGNED_AXES)}, "
                    f"got {value!r}.")
        if self.forward_axis[1] == self.up[1]:
            raise ValueError(
                f"forward_axis {self.forward_axis!r} lies along the up "
                f"axis {self.up!r}; forward is a ground direction.")

        if np.shape(self.center) != (3,):
            raise ValueError(
                f"center must have shape (3,), got {np.shape(self.center)}.")
        if len(self.node_names) != num_nodes:
            raise ValueError(
                f"node_names has {len(self.node_names)} entries but coords "
                f"has {num_nodes} nodes.")
        if np.shape(self.rest_coords) != (num_nodes, 3):
            raise ValueError(
                f"rest_coords must have shape ({num_nodes}, 3) to match "
                f"coords, got {np.shape(self.rest_coords)}.")
        if len(self.bone_chains) != len(self.bones):
            raise ValueError(
                f"bone_chains has {len(self.bone_chains)} entries but there "
                f"are {len(self.bones)} bones.")

        for name in ("bones", "lr_pairs"):
            pairs = np.asarray(getattr(self, name))
            if pairs.shape == (0,):
                continue  # an empty list: no pairs
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(
                    f"{name} must be (first, second) node index pairs, "
                    f"shape (P, 2), got {pairs.shape}.")
            if pairs.size and not np.issubdtype(pairs.dtype, np.integer):
                raise ValueError(
                    f"{name} must hold integer node indices, got dtype "
                    f"{pairs.dtype}.")
            outside = np.unique(pairs[(pairs < 0) | (pairs >= num_nodes)])
            if outside.size:
                raise ValueError(
                    f"{name} names nodes {outside.tolist()} but coords has "
                    f"nodes 0 to {num_nodes - 1}.")

        if (self.root_heading is not None
                and np.shape(self.root_heading) != (num_frames, 2)):
            raise ValueError(
                f"root_heading must have shape ({num_frames}, 2) to match "
                f"the {num_frames} frames of coords, got "
                f"{np.shape(self.root_heading)}.")
        if not (np.isfinite(self.frame_time) and self.frame_time >= 0):
            raise ValueError(
                f"frame_time must be a number of seconds, zero (unset) or "
                f"positive, got {self.frame_time!r}.")

        self._protect_arrays()

    _ARRAY_FIELDS = ("coords", "center", "rest_coords", "lr_pairs",
                     "root_heading")

    def _protect_arrays(self) -> None:
        """Swap each array field for a read-only view of itself."""
        for name in self._ARRAY_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            read_only = np.asarray(value).view()
            read_only.flags.writeable = False
            # frozen forbids assignment; this is the sanctioned way in.
            object.__setattr__(self, name, read_only)

    def __setstate__(self, state: dict[str, object]) -> None:
        # copy.deepcopy and pickle rebuild a view from its state without
        # running __post_init__, and NumPy hands them writable arrays.
        self.__dict__.update(state)
        self._protect_arrays()

@dataclass(frozen=True)
class Scene:
    """A prepared visualization: the single input every backend consumes.

    Multi-panel backends (matplotlib, OpenCV) iterate :attr:`views`;
    single-scene backends (k3d, vedo) call :meth:`unified_box` for the
    one shared bounding box and take the camera from ``views[0]``.

    A Scene has at least one view, and all its views hold the same
    number of frames: backends step every view with one frame counter,
    the first view's. Building a Scene that breaks either raises
    ``ValueError``; align the clips first (:func:`align_frame_counts`).
    Frame *times* may differ between views, see :attr:`frame_time`.
    """

    views: list[SkeletonView]

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("A Scene needs at least one view.")
        counts = [int(v.coords.shape[0]) for v in self.views]
        if len(set(counts)) > 1:
            raise ValueError(
                f"All views of a Scene must hold the same number of "
                f"frames, got {counts}.")

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
