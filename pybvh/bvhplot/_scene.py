"""The Scene every bvhplot backend consumes.

Pure data: :class:`SkeletonView`, :class:`Scene`, the operations that
change a Scene, and the helper that aligns frame counts. No plotting library
imports and no imports from the pybvh core; a Scene is built from a
:class:`~pybvh.bvh.Bvh` in :mod:`._from_bvh`, or from plain arrays.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

UP_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
_SIGNED_AXES = ('+x', '-x', '+y', '-y', '+z', '-z')

# Which measure a view's body size took (SkeletonView.body_size_measure).
BodySizeMeasure = Literal["rest height", "rest extent", "clip extent", "default"]
# The body size of a view whose every coordinate is at one point, in the
# unit of its coords: nothing to measure, a stand-in so that nothing
# drawn on it has size zero.
DEFAULT_BODY_SIZE = 1.0


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
    field access. The camera angles are per-view because side-by-side
    comparisons of skeletons with different up or forward axes need
    each panel oriented independently. A view holds no bounding box:
    what a picture frames is computed from the coords when the picture
    is made (:func:`~._viewport.make_viewport`), so no operation on a
    Scene can leave a box behind that no longer fits.

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
    never read it; the animated ones check it before drawing and
    raise when they need a rate that is unset. Only a negative or
    non-finite value is rejected here.

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
      never reads it; it reads ``coords``, ``lr_pairs``, ``up_vector``
      and ``frame_time`` directly.
    - ``up`` is the one statement of which way is up: a signed axis
      string (``'+y'``, ``'-z'``), the form :attr:`Bvh.world_up` and
      ``forward_axis`` use. The letter (``up_axis``), the column
      (``up_index``), the sign (``up_sign``) and the unit vector
      (``up_vector``) are derived from it, so they cannot disagree.
      The alternative, storing the vector, would admit an off-axis or
      unnormalised up that no backend can draw: matplotlib's vertical
      axis and the floor plane are axis-aligned.
    - ``rest_up`` is the up axis of ``rest_coords`` (:attr:`Bvh.rest_up`),
      in the same signed form, or ``None`` when it is not known. It
      usually equals ``up``; a file that authors its rest pose in one
      convention and animates it in another has two, and
      :attr:`body_size` needs the rest pose's. An unknown rest axis is
      not replaced by ``up``, which on such a file would name the rest
      pose's depth; :attr:`body_size` measures without it instead.
    - ``lr_pairs`` are joint pairs only, in node index space, the pairs
      the facing geometry averages; end-site pairs are deliberately
      excluded so follow azimuths match the Bvh path bit for bit.
    - Arrays are read-only and borrowed. A view stores a read-only
      NumPy view of each array it is given (``coords``,
      ``rest_coords``, ``lr_pairs``, ``root_heading``), so writing into
      one through the view raises. That matters because storage is
      shared: no array is copied at construction, and a Scene operation
      may share storage with its source (``subsampled`` shares
      ``coords`` and ``root_heading``; ``offset`` allocates new coords
      and shares the heading; ``looped`` copies both), so a write
      through one Scene would change another. The alternative,
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
    azimuth: float                         # degrees
    elevation: float                       # degrees
    up: str                                # signed world-up axis, e.g. '+y', '-z'
    floor_height: float                    # scene ground along the up axis
    frame_time: float                      # seconds per frame
    node_names: list[str]                  # parallel to the N axis
    rest_coords: npt.NDArray[np.float64]   # (N, 3) rest pose, root at origin
    rest_up: str | None                    # signed up axis of rest_coords, e.g. '+y'; None if unknown
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

        # rest_up may also be None, unknown; body_size measures without it.
        signed_axes = {"up": self.up, "forward_axis": self.forward_axis}
        if self.rest_up is not None:
            signed_axes["rest_up"] = self.rest_up
        for name, value in signed_axes.items():
            if value not in _SIGNED_AXES:
                raise ValueError(
                    f"{name} must be one of {', '.join(_SIGNED_AXES)}, "
                    f"got {value!r}.")
        if self.forward_axis[1] == self.up[1]:
            raise ValueError(
                f"forward_axis {self.forward_axis!r} lies along the up "
                f"axis {self.up!r}; forward is a ground direction.")

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

    @property
    def body_size(self) -> float:
        """The body's height, the length what is drawn on it is sized from.

        The rest pose's extent along the rest pose's own up axis
        (``rest_up``), in the unit of ``coords``. It is a property of
        the skeleton, not of the motion: it does not change with the
        distance the clip travels, with the frames shown, or with the
        other skeletons of a Scene, so a still and the whole clip draw
        a body at the same proportions, and each skeleton of a Scene
        gets its own. It is always positive: a body that has no height
        to measure is measured another way, and
        :attr:`body_size_measure` says which of these it got, first
        that applies:

        - ``"rest height"``: the rest pose's extent along ``rest_up``.
        - ``"rest extent"``: the rest pose's widest extent along any
          axis, when ``rest_up`` is ``None`` (the rest pose's up axis
          could not be inferred) or the rest pose has no extent along
          it. For a standing rest pose it is the height, or the arm
          span of a T-pose whose arms reach wider than it stands tall
          (bvh_test3: 71.4 across the arms, 62.6 tall). The
          alternative, the extent along the animation's ``up``, is
          wrong exactly where the two axes differ: there it measures
          the rest pose's depth (bvh_test3: 10.7).
        - ``"clip extent"``: the widest extent of the box ``coords``
          sweeps over the clip, when the rest pose has no extent at
          all (one node, or nodes that all coincide) or no ratio to
          put it in the unit of ``coords`` (below). There is no body
          to measure, so it grows with the
          distance the clip travels.
        - ``"default"``: ``DEFAULT_BODY_SIZE``, one unit of ``coords``,
          when every coordinate of the clip is at one point. Nothing
          was measured; the size is a stand-in so that nothing drawn
          on the body has size zero.

        Conventions:

        - The alternative to the rest height is the median over the
          clip's frames of the pose's extent along ``up``. It follows
          what the clip does, and the two differ when the clip does
          not stand: a crouched, seated or lying clip reads smaller
          than its rest pose, and a rest pose with raised arms reads
          taller than the character stands. (Taken as the pose's
          widest extent rather than along ``up``, the alternative also
          differs for a T-pose whose arm span exceeds its height.) The
          rest pose is used because what is drawn on a body should not
          thin when the body crouches, and because it needs no pass
          over the motion.
        - The extent is taken along ``rest_up``, not along ``up``. The
          two agree on most files; they differ when a file authors its
          rest pose in one convention and animates it in another
          (``Bvh.rest_up`` against ``Bvh.world_up``), and there the
          rest pose's extent along ``up`` is its depth, not its height.
        - The unit is the coords': the rest pose's measures are scaled
          by :attr:`coords_per_rest_unit`. A view without that ratio
          does not use its rest pose (see there).
        """
        return self._measured_body_size()[0]

    @property
    def body_size_measure(self) -> BodySizeMeasure:
        """Which measure :attr:`body_size` took: ``"rest height"``,
        ``"rest extent"``, ``"clip extent"`` or ``"default"`` (see
        there). Only ``"default"`` is not measured from the view."""
        return self._measured_body_size()[1]

    def _measured_body_size(self) -> tuple[float, BodySizeMeasure]:
        """:attr:`body_size` and :attr:`body_size_measure`, the first
        measure of the chain that is positive."""
        coords_per_rest_unit = self.coords_per_rest_unit
        if coords_per_rest_unit is not None:
            rest_extents = (
                np.ptp(self.rest_coords, axis=0) * coords_per_rest_unit)
            if self.rest_up is not None:
                rest_height = float(
                    rest_extents[UP_AXIS_INDEX[self.rest_up[1]]])
                if rest_height > 0.0:
                    return rest_height, "rest height"
            widest_rest_extent = float(rest_extents.max())
            if widest_rest_extent > 0.0:
                return widest_rest_extent, "rest extent"
        swept_extents = np.ptp(self.coords.reshape(-1, 3), axis=0)
        widest_swept_extent = float(swept_extents.max())
        if widest_swept_extent > 0.0:
            return widest_swept_extent, "clip extent"
        return DEFAULT_BODY_SIZE, "default"

    @property
    def coords_per_rest_unit(self) -> float | None:
        """How many units of ``coords`` one unit of ``rest_coords`` is:
        the factor that puts a rest-pose length in the unit of what is
        drawn, or ``None`` when there is no ratio to measure.

        It is the median, over the bones of positive rest length, of
        each bone's length in coordinate row 0 over its rest length.
        Bone lengths do not change under forward kinematics, so the
        factor is 1 for coords posed from the skeleton, and differs
        only for coordinates a caller supplies in another unit (metres
        against the file's centimetres). Any bone would do; the median
        keeps one odd bone from deciding it, and bones of zero rest
        length (end sites or helper joints placed on their parent) are
        left out because they have no ratio. Taken over every bone
        instead, as a ratio of median lengths, it is 0 over 0 on a rig
        where most bones have zero length.

        ``None`` for a view with no bone of positive rest length, and
        for one whose coordinate row 0 collapses most of those bones to
        zero length (a median ratio of 0 is no unit). Its rest pose is
        then not converted at all, and what would measure it measures
        ``coords`` instead (as :attr:`body_size` does): the
        alternative, taking the rest pose to share the unit of
        ``coords``, draws a body a hundred times too thin or too thick
        when a caller's coordinates are in centimetres against a rest
        pose in metres, or the reverse.
        """
        if not self.bones:
            return None
        parents, children = np.asarray(self.bones).T
        rest_lengths = np.linalg.norm(
            self.rest_coords[children] - self.rest_coords[parents], axis=1)
        measurable = rest_lengths > 0.0
        if not measurable.any():
            return None
        pose = self.coords[0]
        pose_lengths = np.linalg.norm(
            pose[children] - pose[parents], axis=1)
        ratios = pose_lengths[measurable] / rest_lengths[measurable]
        median_ratio = float(np.median(ratios))
        return median_ratio if median_ratio > 0.0 else None

    _ARRAY_FIELDS = ("coords", "rest_coords", "lr_pairs", "root_heading")

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

    Multi-panel backends (matplotlib, OpenCV) iterate :attr:`views`
    and make one viewport per view; single-scene backends (k3d, vedo)
    make one viewport of all the views, whose camera is the first
    view's.

    A Scene has at least one view, and all its views hold the same
    number of frames: backends step every view with one frame counter,
    the first view's. Building a Scene that breaks either raises
    ``ValueError``; align the clips first (:func:`align_frame_counts`).
    Frame *times* may differ between views, see :attr:`frame_time`.

    ``loop_length`` is ``None`` for a Scene that plays its clips once.
    A Scene made by :meth:`looped` plays them again from the first
    frame every ``loop_length`` frames; each such run is a *pass*
    (:meth:`pass_start`), and what trails the live pose (ghosts, the
    root trace, the frame counter) belongs to its pass.
    """

    views: list[SkeletonView]
    loop_length: int | None = None

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("A Scene needs at least one view.")
        counts = [int(v.coords.shape[0]) for v in self.views]
        if len(set(counts)) > 1:
            raise ValueError(
                f"All views of a Scene must hold the same number of "
                f"frames, got {counts}.")
        if self.loop_length is not None and not (
                _is_whole_number(self.loop_length)
                and 1 <= self.loop_length <= counts[0]):
            raise ValueError(
                f"loop_length must be a whole number of frames from 1 to "
                f"the Scene's {counts[0]}, got {self.loop_length!r}.")

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

    @property
    def pass_length(self) -> int:
        """Frames in one pass of the clip: ``loop_length`` for a looped
        Scene, every frame otherwise."""
        if self.loop_length is None:
            return self.num_frames
        return self.loop_length

    def pass_start(self, frame: int) -> int:
        """The first frame of the pass that shows *frame*.

        ``0`` for every frame of a Scene that is not looped. Backends
        draw a frame's ghosts and root trace from this frame on, and
        count frames from it, so every pass is drawn as the first was.
        """
        return frame - frame % self.pass_length

    def looped(self, num_frames: int) -> Scene:
        """The clip played again from its first frame until the Scene
        holds ``num_frames`` frames, as a new Scene.

        Every frame-indexed field is repeated (a copy, not a view: the
        passes are laid end to end), the last pass is cut short where
        ``num_frames`` ends, and nothing is blended at the seam: the
        pose jumps from the clip's last frame back to its first, as a
        looping video does. Timing, floor and camera angles are kept.
        A Scene that is already looped plays its original pass again,
        so looping twice is looping once to the longer length.
        """
        if not _is_whole_number(num_frames):
            raise ValueError(
                f"num_frames must be a whole number of frames, got "
                f"{num_frames!r}.")
        if num_frames < self.num_frames:
            raise ValueError(
                f"looped() plays the clip again and cannot shorten it: "
                f"num_frames must be at least {self.num_frames}, got "
                f"{num_frames}.")
        shown = np.arange(num_frames) % self.pass_length
        views = []
        for v in self.views:
            heading = (None if v.root_heading is None
                       else v.root_heading[shown])
            views.append(dataclasses.replace(
                v, coords=v.coords[shown], root_heading=heading))
        return Scene(views=views, loop_length=self.pass_length)

    def subsampled(self, step: int) -> Scene:
        """Every ``step``-th frame of every view, as a new Scene.

        Slices ``coords`` and every other frame-indexed field
        (``root_heading``) with ``[::step]`` and scales ``frame_time`` by
        ``step``, so the kept frames stay at their original moments in
        time; up to ``step - 1`` trailing frames are dropped, so the
        span from the first to the last kept sample can shorten by that
        many original intervals and the last frame is not necessarily
        kept. The floor is kept as is: it describes the whole clip
        (the canonical floor is a clip-wide estimate, the coords floor
        the full clip's extreme), and a plane that sits where the full
        clip's ground was is the honest one for a preview of it. The
        alternative, recomputing from the kept frames, would move the
        ground between the full and the subsampled view of one clip.

        A looped Scene cannot be subsampled: its every ``step``-th
        frame is not one clip played again, since its passes would no
        longer share a length. Subsample first, then loop.
        """
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}.")
        if self.loop_length is not None:
            raise ValueError(
                "A looped Scene cannot be subsampled: subsample the clip "
                "first, then loop it.")
        views = []
        for v in self.views:
            heading = (None if v.root_heading is None
                       else v.root_heading[::step])
            views.append(dataclasses.replace(
                v, coords=v.coords[::step],
                frame_time=v.frame_time * step, root_heading=heading))
        return Scene(views=views)

    def offset(self, offsets: list[npt.NDArray[np.float64]]) -> Scene:
        """Translate each view by its own ``(3,)`` vector, as a new Scene.

        Moves ``coords``, and ``floor_height`` by the
        offset's component along the view's up axis, so the plane stays
        under the feet. Orientation, heading, timing and the loop are
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
                floor_height=v.floor_height + float(off[v.up_index])))
        return dataclasses.replace(self, views=views)

    def spread(self, spacing: float | str) -> Scene:
        """Offset the views laterally so skeletons sharing one 3-D scene
        do not overlap.

        For the single-scene backends (k3d, vedo); multi-panel backends
        draw each view in its own axes and never need it. View ``k``
        moves by ``k × spacing`` toward the first skeleton's own left:
        ``up × forward`` of the first view, with ``forward`` its
        ``forward_axis`` (the facing of its first frame, snapped to an
        axis). Seen from the ``"front"`` camera that is the viewer's
        right, so the skeletons read left to right in the order given,
        on every rig. The alternative it replaces, a fixed world axis
        (the positive axis that is neither up nor forward), flips with
        the rig: ``+x`` is the left of a ``+y``-up character facing
        ``+z``, but the right of the same character facing ``-z`` or
        turned ``-y``-up. In a Scene made from a Bvh, a facing that
        cannot be measured on the first frame (no left/right joint
        pairs, or pairs lying along up) leaves ``forward_axis`` at the
        fallback the ``"front"`` camera also faces (the rest pose's
        facing, then a fixed default per up axis, with a warning), so
        the next skeleton still lands on the front camera's right,
        whichever side of the body that is.

        ``"auto"`` spaces by 1.2 × the first view's extent along that
        direction (at least 0.1 scene units); a float is used
        directly, in scene units. Whether to spread at all is the
        caller's policy (``play`` respects raw world coordinates under
        ``"auto"``).
        """
        if len(self.views) <= 1:
            return self
        first = self.views[0]
        forward = np.zeros(3)
        forward[UP_AXIS_INDEX[first.forward_axis[1]]] = (
            -1.0 if first.forward_axis[0] == '-' else 1.0)
        leftward = np.cross(first.up_vector, forward)

        if spacing == "auto":
            lateral = first.coords @ leftward
            width = float(lateral.max() - lateral.min())
            effective = max(width, 0.1) * 1.2
        else:
            effective = float(spacing)
        if effective == 0.0:
            return self

        return self.offset(
            [leftward * k * effective for k in range(len(self.views))])


def _is_whole_number(value: object) -> bool:
    """Whether *value* is an integer type (Python or NumPy), bool
    excluded: a frame count, not a float that happens to be whole."""
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, bool))


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
