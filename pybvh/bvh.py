"""The :class:`Bvh` container: one clip's skeleton and motion.

A :class:`Bvh` holds the node hierarchy, the root translation
``root_pos`` (shape ``(F, 3)``, in the file's length unit) and the joint
rotations ``joint_angles`` (shape ``(F, J, 3)``, Euler angles in
radians), and exposes every high-level operation as a method. Most
methods are thin wrappers over the array-level modules
(:mod:`pybvh.analysis`, :mod:`pybvh.transforms`, :mod:`pybvh.geometry`,
:mod:`pybvh.features`, :mod:`pybvh.bvhplot`), which document the
computation itself.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast, overload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from . import geometry
    from ._contacts import ContactReport

import numpy as np
import numpy.typing as npt

from . import rotations
from ._derived import DerivedState as _DerivedState
from ._orientation import (
    OrientationReport,
    _anchor_contradicts_rest_up,
    _BodyAnchor,
    _build_orientation_report,
    _detect_lr_mapping_by_names,
    _facing_basis,
    _facing_labels,
    _find_body,
    _find_body_anchor,
    _infer_rest_up_and_left,
    _infer_world_up,
    _iter_unique_lr_pairs,
    _lr_pairs_in_body,
    _LrMappingSource,
    _resolve_lr_pairs,
    _resolve_node_lr_pairs,
    _RestSkeleton,
    _size_reference,
    _warn_facing_defaulted_at_frames,
    _warn_facing_defaulted_no_pairs,
    _warn_facing_defaulted_on_pose,
    _warn_rest_forward_defaulted,
    _warn_rest_up_contradicted_by_anchor,
    _warn_world_up_defaulted,
    _warn_world_up_disagreement,
    _world_up_disagrees,
    _WorldUpGuess,
)

# Re-exported: pybvh.bvh has long served every node class, end sites included.
from .bvhnode import BvhEndSite as BvhEndSite
from .bvhnode import BvhJoint as BvhJoint
from .bvhnode import (
    BvhNode,
    BvhRoot,
    NodeSelector,
    _euler_order,
    _is_joint,
    _is_translating_joint,
    _pos_channels,
)
from .node_tree import (
    _OFFSET_ATOL,
    _channel_difference,
    _check_node_list,
    _check_node_tree,
    _end_site_name,
    _hierarchy_difference,
    _nodes_from_table_or_list,
    nodes_from_table,
    nodes_to_table,
)
from .spatial_coord import (
    FkTopology,
    _check_centered,
    _ground_plane_offset,
    frames_to_node_positions,
)
from .tools import (
    Axis,
    _axis_to_vector,
    _format_fps,
    _matches_fps,
    _require_rate,
    _same_rate,
    _validate_axis_string,
    _validate_fps,
    _validate_frame_time,
    parse_axis,
)
from .tools import get_main_direction as _get_main_direction


def _unique_labels(names: Sequence[str]) -> list[str]:
    """Make *names* unique in order, the way pandas labels repeated headers.

    The first occurrence keeps its name; a later one gets ``.1``, ``.2``,
    ... appended, counted up until the label is neither a name in
    *names* nor a label already given, so an entry whose own name looks
    like a suffixed label keeps it. ``['Arm', 'Arm', 'Arm']`` gives
    ``['Arm', 'Arm.1', 'Arm.2']`` and ``['Arm', 'Arm', 'Arm.1']`` gives
    ``['Arm', 'Arm.2', 'Arm.1']``, as ``pandas.read_csv`` reads those
    headers.
    """
    reserved = set(names)
    given: set[str] = set()
    labels: list[str] = []
    for name in names:
        label = name
        suffix = 0
        while label in given or (label != name and label in reserved):
            suffix += 1
            label = f"{name}.{suffix}"
        given.add(label)
        labels.append(label)
    return labels


def _euler_to_quat(
    angles: npt.NDArray[np.float64], orders: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Joint angles ``(..., J, 3)`` in radians, each joint in its Euler order
    of ``orders``, as canonical quaternions ``(..., J, 4)``, through
    rotation matrices."""
    return rotations.rotmat_to_quat(rotations.euler_to_rotmat(angles, orders))


class _EulerBlock(NamedTuple):
    """Three motion columns of one joint in ``to_df_dict``'s euler mode."""

    # The joint's column label, unique over the joints.
    label: str
    # The axes in column order, and the kind, ``"pos"`` or ``"rot"``.
    axes: tuple[str, ...]
    kind: str
    # The motion array the values come from and the joint's row in it:
    # ``"root_pos"`` (row unused), ``"joint_translations"`` or ``"joint_angles"``.
    array: str
    row: int

    def columns(self, motion: Mapping[str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """The block's ``(F, 3)`` values in *motion*, the arrays by name: a
        view, which a reader reads and a writer assigns into."""
        array = motion[self.array]
        return array if self.array == "root_pos" else array[:, self.row]


def _euler_blocks(nodes: Sequence[BvhNode]) -> list[_EulerBlock]:
    """The column blocks of the euler layout of *nodes*, in column order.

    Per joint in node order, its position block if it declares position
    channels (the root's from ``root_pos``, another joint's from its row
    of ``joint_translations``), then its rotation block, from its row of
    ``joint_angles``. The layout is that of a frame line in a file, which
    ``io._frame_columns`` reads from these blocks: a clip whose joints
    other than the root only rotate gets the root's position block then
    each joint's rotation block, and a static root, which declares no
    channels, gets none.
    """
    joints = [node for node in nodes if _is_joint(node)]
    root = joints[0]
    if not isinstance(root, BvhRoot):
        raise ValueError(f"nodes[0] must be the BvhRoot, got {type(root).__name__}")
    labels = _unique_labels([joint.name for joint in joints])
    blocks: list[_EulerBlock] = []
    translating = 0
    for row, (label, joint) in enumerate(zip(labels, joints, strict=True)):
        if joint.is_root() and _pos_channels(joint):
            blocks.append(_EulerBlock(label, _pos_channels(joint), "pos", "root_pos", 0))
        elif _is_translating_joint(joint):
            blocks.append(
                _EulerBlock(label, _pos_channels(joint), "pos", "joint_translations", translating)
            )
            translating += 1
        if joint.rot_channels:
            blocks.append(_EulerBlock(label, tuple(joint.rot_channels), "rot", "joint_angles", row))
    return blocks


def _motion_column_names(nodes: Sequence[BvhNode], mode: str) -> list[str]:
    """Column names of the motion *nodes* carry, in either ``to_df_dict`` mode.

    ``'euler'`` names the blocks of :func:`_euler_blocks`: per joint in
    node order, ``<joint>_<axis>_pos`` per position channel if the joint
    declares any, then ``<joint>_<axis>_rot`` per rotation channel, end
    sites having none. ``'coordinates'`` names ``<node>_<axis>`` for X,
    Y and Z per node, end sites included. ``time`` is not a motion
    column and is not listed. Node names are made unique with
    :func:`_unique_labels` over the nodes the mode lists, before the
    axis and kind are appended, so the same joint can be ``Arm`` in one
    mode and ``Arm.1`` in the other when an earlier end site shares its
    name.
    """
    if mode == "euler":
        return [
            f"{block.label}_{ax}_{block.kind}"
            for block in _euler_blocks(nodes)
            for ax in block.axes
        ]
    if mode == "coordinates":
        labels = _unique_labels([node.name for node in nodes])
        return [f"{label}_{ax}" for label in labels for ax in ("X", "Y", "Z")]
    raise ValueError(f"mode must be 'euler' or 'coordinates', got {mode!r}")


class _SkeletonDim(NamedTuple):
    """An axis of a motion array whose length the skeleton fixes."""

    # The axis as a shape names it: ``(F, J, 3)``.
    symbol: str
    count: Callable[[Sequence[BvhNode]], int]
    # What one index of the axis is, and what the skeleton has of it, as
    # the error of a motion that does not fit the skeleton names them.
    unit: str
    skeleton_unit: str

    def __str__(self) -> str:
        return self.symbol


_JOINTS = _SkeletonDim(
    symbol="J",
    count=lambda nodes: sum(1 for node in nodes if _is_joint(node)),
    unit="joints",
    skeleton_unit="non-end-site joints",
)

_TRANSLATING_JOINTS = _SkeletonDim(
    symbol="P",
    count=lambda nodes: sum(1 for node in nodes if _is_translating_joint(node)),
    unit="translating joints",
    skeleton_unit="joints other than the root with position channels",
)

# How a motion array is interpolated between two frames: ``"linear"`` per
# component, ``"euler"`` as rotations, each joint's three angles read in its
# Euler order.
_Interpolation = Literal["linear", "euler"]


class _MotionArray(NamedTuple):
    """One motion array of a clip, as the frame operations carry it.

    ``name`` is the array's property, its keyword in the constructor and
    in ``Bvh._set_motion`` and its key in a pickle's state; the clip holds
    the array as ``_<name>``. ``frame_shape`` is the array's shape after
    the frame axis: an int is a fixed length, a ``_SkeletonDim`` a
    length the skeleton fixes.
    """

    name: str
    frame_shape: tuple[int | _SkeletonDim, ...]
    interpolation: _Interpolation

    @property
    def attribute(self) -> str:
        """The clip attribute that holds the array."""
        return f"_{self.name}"

    @property
    def skeleton_axes(self) -> list[tuple[int, _SkeletonDim]]:
        """Each axis whose length the skeleton fixes, with its dim."""
        return [
            (axis, dim)
            for axis, dim in enumerate(self.frame_shape, start=1)
            if isinstance(dim, _SkeletonDim)
        ]

    def unknown_interpolation(self, operation: str) -> ValueError:
        """The error ``operation`` raises on an interpolation kind it does not know."""
        return ValueError(
            f"{operation} does not know interpolation {self.interpolation!r} "
            f"of motion array {self.name!r}"
        )

    def empty(self) -> npt.NDArray[np.float64]:
        """The array of a clip built without motion: no frame, and 0 on
        each skeleton dim, so that it binds no skeleton."""
        shape = [0 if isinstance(dim, _SkeletonDim) else dim for dim in self.frame_shape]
        return np.empty((0, *shape), dtype=np.float64)

    def checked(self, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """``value`` as a float array, if it has this array's rank and fixed
        lengths; the skeleton dims are checked against a skeleton by
        ``Bvh._check_motion_fits_skeleton``."""
        arr = np.asarray(value, dtype=np.float64)
        fits = arr.ndim == 1 + len(self.frame_shape) and all(
            isinstance(dim, _SkeletonDim) or arr.shape[axis] == dim
            for axis, dim in enumerate(self.frame_shape, start=1)
        )
        if not fits:
            shape = ", ".join(str(dim) for dim in ("F", *self.frame_shape))
            raise ValueError(f"{self.name} must have shape ({shape}), got {arr.shape}")
        return arr


_ROOT_POS = _MotionArray(name="root_pos", frame_shape=(3,), interpolation="linear")
_JOINT_ANGLES = _MotionArray(name="joint_angles", frame_shape=(_JOINTS, 3), interpolation="euler")
_JOINT_TRANSLATIONS = _MotionArray(
    name="joint_translations", frame_shape=(_TRANSLATING_JOINTS, 3), interpolation="linear"
)


def _check_frame_counts(
    entries: Sequence[_MotionArray], motion: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """Raise ``ValueError`` unless every array of *motion*, by the names of
    *entries*, has the frame count of the first."""
    first, *rest = entries
    frame_count = motion[first.name].shape[0]
    for entry in rest:
        if motion[entry.name].shape[0] != frame_count:
            raise ValueError(
                f"{first.name} and {entry.name} disagree on frame count: "
                f"{first.name} has {frame_count} frames, "
                f"{entry.name} has {motion[entry.name].shape[0]}."
            )


def _check_array_fits_skeleton(
    entry: _MotionArray, array: npt.NDArray[np.float64], nodes: Sequence[BvhNode]
) -> None:
    """Raise ``ValueError`` unless each skeleton dim of ``array`` has the
    length ``nodes`` gives it."""
    for axis, dim in entry.skeleton_axes:
        actual = array.shape[axis]
        expected = dim.count(nodes)
        if actual != expected:
            raise ValueError(
                f"{entry.name} has {actual} {dim.unit} "
                f"on axis {axis}, but the skeleton has {expected} "
                f"{dim.skeleton_unit}."
            )


def _node_pairs(pairs: object) -> list[tuple[NodeSelector, NodeSelector]]:
    """``pairs`` as a list of pairs, or ``TypeError`` when it is not a sequence of pairs.

    Without the check, a pair given alone or a ``str`` would be unpacked
    letter by letter or entry by entry, and fail with an unpacking
    ``ValueError`` that does not say what to pass. The entries of each pair
    are left to :meth:`Bvh.index`.
    """
    example = "[('Hips', 'Spine')]"
    if isinstance(pairs, (str, BvhNode)) or not isinstance(pairs, Iterable):
        raise TypeError(
            f"pairs is a sequence of node pairs, not one {type(pairs).__name__}: "
            f"pass {example} for the pair of 'Hips' and 'Spine'."
        )
    checked = []
    for pair in cast("Iterable[Any]", pairs):
        if isinstance(pair, str):
            got = f"the str {pair!r}"
        elif isinstance(pair, BvhNode) or not isinstance(pair, Sized):
            got = f"one {type(pair).__name__}"
        elif len(pair) != 2:
            got = f"a {type(pair).__name__} of length {len(pair)}"
        else:
            a, b = pair
            checked.append((a, b))
            continue
        raise TypeError(
            f"Each entry of pairs is a pair of nodes, each a name or a "
            f"BvhNode, not {got}: pass {example} for the pair of 'Hips' "
            f"and 'Spine'."
        )
    return checked


def _validated_lr_mapping(mapping: dict[str, str], joint_names: Sequence[str]) -> dict[str, str]:
    """The user's lr_mapping dict, checked against ``joint_names``, one
    direction per pair.

    Validation: both names must exist in ``joint_names``; no
    self-pairs; no duplicate names on either side of a pair. The result
    is stored one-directional (left to right); the public ``lr_mapping``
    property gives the symmetric view, so every internal consumer that
    iterates its items gets one (left, right) tuple per pair.
    """
    if not isinstance(mapping, dict):
        raise TypeError(f"lr_mapping must be a dict, got {type(mapping).__name__}")
    if not mapping:
        raise ValueError("lr_mapping must have at least one pair; pass None to clear the mapping.")
    # Accept symmetric input ({L: R, R: L, ...}) — canonicalize to
    # one-directional by deduping pairs by frozenset before
    # validating. Each pair appears exactly once afterward.
    canonical: dict[str, str] = dict(_iter_unique_lr_pairs(mapping))
    joint_name_set = set(joint_names)
    lefts_seen: set[str] = set()
    rights_seen: set[str] = set()
    for left, right in canonical.items():
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError(
                f"lr_mapping keys and values must be str, "
                f"got {type(left).__name__}/{type(right).__name__}"
            )
        if left == right:
            raise ValueError(f"lr_mapping self-pair not allowed: {left!r}")
        if left not in joint_name_set:
            raise ValueError(f"lr_mapping left joint {left!r} not in joint_names")
        if right not in joint_name_set:
            raise ValueError(f"lr_mapping right joint {right!r} not in joint_names")
        if left in lefts_seen or left in rights_seen:
            raise ValueError(f"lr_mapping joint {left!r} appears in multiple pairs")
        if right in lefts_seen or right in rights_seen:
            raise ValueError(f"lr_mapping joint {right!r} appears in multiple pairs")
        lefts_seen.add(left)
        rights_seen.add(right)
    return canonical


def _build_rest_pose(bvh: Bvh) -> npt.NDArray[np.float64]:
    """FK of ``bvh``'s skeleton with every angle zero, the root at the origin
    and each translating joint at its offset, the rest pose (ADR 0010).

    Posed from the topology, not the clip: the rest pose is centred on the
    root, and the rest skeleton that the body anchor is found from holds it."""
    return frames_to_node_positions(
        bvh.fk_topology,
        root_pos=np.zeros(3),
        joint_angles=np.zeros((bvh.joint_count, 3), dtype=np.float64),
        centered="anchor",
        joint_translations=np.array(
            [node.offset for node in bvh.nodes if _is_translating_joint(node)], dtype=np.float64
        ).reshape(-1, 3),
    )


def _static_root_pos(offset: npt.ArrayLike, frame_count: int) -> npt.NDArray[np.float64]:
    """The ``root_pos`` of a static root (``CHANNELS 0``) at *offset*,
    ``(frame_count, 3)``: it stays at its offset in every frame."""
    return np.tile(np.asarray(offset, dtype=np.float64), (frame_count, 1))


def _channel_declaration(root: BvhRoot) -> str:
    """The root's channel set as a ``CHANNELS`` line counts it."""
    return "CHANNELS 0" if root.is_static() else "CHANNELS 6"


def _describe_rate(frame_time: float) -> str:
    """A clip's rate for a message: its fps and its exact frame time, or unset."""
    if frame_time == 0:
        return "unset, frame_time 0"
    return f"{_format_fps(frame_time)} fps, frame_time {float(frame_time)!r}"


class Bvh:
    """Container for BVH motion-capture data.

    The hierarchy is stored as a tuple of ``BvhNode`` objects (one per
    joint / end-site).  Motion data is stored as three arrays:

    - ``root_pos``:     shape ``(F, 3)``    — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)`` — Euler angles **in radians** per joint per frame
    - ``joint_translations``: shape ``(F, P, 3)`` — per frame, the position
      relative to its parent of each joint other than the root that
      declares position channels (P = 0 on most skeletons)

    ``Bvh`` is a sequence of frames: ``len(bvh) == frame_count`` and
    ``bvh[i]`` returns frame ``i`` as a one-frame Bvh. For joint or node
    counts, use ``bvh.joint_count`` or ``len(bvh.node_index)``.

    Attributes
    ----------
    nodes : tuple of BvhNode
        Skeleton hierarchy in depth-first order, as a file lists it.
        The constructor takes a list or a tuple and keeps a tuple of its
        own, so a later edit to the list it was given leaves the clip as
        it was.
        The constructor checks that the tree is wired both ways: the
        depth-first walk of ``children`` from ``nodes[0]`` must visit
        exactly ``nodes``, in order, and each node's ``parent`` must be
        the node it was reached from; ``ValueError`` otherwise. To build
        a tree from plain data, :func:`~pybvh.nodes_from_table` wires it.
        The constructor freezes every node it is given (see
        :mod:`pybvh.bvhnode`), and ``nodes`` cannot be assigned: a new
        skeleton goes to a new ``Bvh``, such as ``bvh.copy(nodes=table)``.
    root : BvhRoot
        The root node (``nodes[0]``), read-only.
    root_pos : ndarray, shape (F, 3)
        Root position per frame.
    joint_angles : ndarray, shape (F, J, 3)
        Euler angles in **radians** per joint per frame. (BVH files
        store angles in degrees; the deg↔rad conversion happens at the
        I/O boundary in :func:`read_bvh_file` / :func:`write_bvh_file`.)
    joint_translations : ndarray, shape (F, P, 3)
        Per frame, the position relative to its parent of each
        translating joint (see :attr:`joint_translations`). Optional in
        the constructor: omitted, it is ``(F, 0, 3)``, for a skeleton
        whose joints other than the root only rotate.
    frame_time : float
        Duration of one frame in seconds.
    frame_count : int
        Number of frames (read-only).
    node_index : dict
        Mapping from node name to its index in ``nodes`` (includes
        end sites). Use this to index ``node_positions()`` output
        (shape ``(F, N, 3)``).
    joint_index : dict
        Mapping from joint name to its index in ``joint_angles`` axis 1
        (joint-only, excludes end sites). Use this to index
        ``joint_angles`` (shape ``(F, J, 3)``).
    joint_names : list of str
        Names of non-end-site joints in topological order (read-only).
    joint_count : int
        Number of non-end-site joints (read-only).
    source_path : str or None
        Path of the file this Bvh was read from, or ``None`` if it was
        constructed in memory. Set by :func:`read_bvh_file`. Preserved
        through ``copy()``, frame slicing (``bvh[a:b]``), and rotations /
        transforms that don't change which file the data originated from.
        Cleared (set to ``None``) when concatenation (``a + b``) joins two
        clips whose ``source_path`` differ. Writable — callers can assign
        manually when constructing a Bvh from arrays.
    """

    # The motion arrays, in the order the frame operations carry them. An
    # operation that carries the arrays as they are (the shape checks, the
    # derived clips and pickles, slicing, joins, splices, ``==``) reads them
    # from here; the first one's frame count is the one the others must
    # match.
    _MOTION_ARRAYS: ClassVar[tuple[_MotionArray, ...]] = (
        _ROOT_POS,
        _JOINT_ANGLES,
        _JOINT_TRANSLATIONS,
    )

    def __init__(
        self,
        nodes: list[BvhNode] | tuple[BvhNode, ...] | None = None,
        root_pos: npt.ArrayLike | None = None,
        joint_angles: npt.ArrayLike | None = None,
        frame_time: float = 0,
        world_up: str = "auto",
        lr_mapping: dict[str, str] | None = None,
        source_path: str | None = None,
        warn_on_world_up_disagreement: bool = True,
        joint_translations: npt.ArrayLike | None = None,
        body_anchor: NodeSelector | None = None,
    ) -> None:
        # The derived state exists before any property setter runs, so the
        # setters can report their change unconditionally. Each value in it
        # is computed on first read, except the guessed world up, which is
        # guessed below so that a clip read from a file has its world up
        # (and any disagreement warning) at once.
        self._derived = _DerivedState()
        self._world_up_override: str | None = None
        # The warn preference lives on the instance, not only in this
        # call, because the guess starts again when the user assigns
        # motion, and world_up_inferred guesses afresh on every read: each
        # of those must still see what the caller asked for at load.
        self._warn_on_world_up_disagreement = warn_on_world_up_disagreement
        # The user's L/R choice: a mapping they set (one direction, left to
        # right), None once they cleared it, or "names" (the default) for
        # the pairs detected from the joint names.
        self._lr_mapping_setting: dict[str, str] | Literal["names"] | None = "names"
        # The body anchor the user set, as its position in ``nodes``, or
        # None for the one the rule finds. A position, not a node: pybvh's
        # own skeleton edits replace the nodes and keep their order.
        self._body_anchor_setting: int | None = None

        if nodes is None:
            nodes = [BvhRoot()]
        if not isinstance(nodes, (list, tuple)):
            raise ValueError("nodes should be a list or tuple of BvhNode objects")
        _check_node_list(nodes)
        # A tuple of the clip's own: neither an edit to the caller's list
        # nor one through ``bvh.nodes`` can change the skeleton behind the
        # indexes and the derived values.
        self._nodes: tuple[BvhNode, ...] = tuple(nodes)
        self.frame_time = frame_time
        self.source_path = source_path

        given = {"root_pos": root_pos, "joint_angles": joint_angles}
        missing = [name for name, value in given.items() if value is None]
        if missing and (len(missing) < len(given) or joint_translations is not None):
            raise ValueError(
                f"{' and '.join(given)} must be provided together, and with "
                f"joint_translations when it is given; {missing[0]} is missing."
            )
        if not missing:
            # Omitted, no joint translates: P = 0 in every frame.
            given["joint_translations"] = (
                np.empty((len(np.asarray(root_pos)), 0, 3))
                if joint_translations is None
                else joint_translations
            )
        # All given or none: a clip built without motion holds the empty arrays.
        for entry in self._MOTION_ARRAYS:
            value = given.get(entry.name)
            setattr(self, entry.attribute, entry.empty() if value is None else entry.checked(value))
        if not missing:
            self._check_motion_fits_skeleton()
            self._check_static_root_motion()

        if world_up != "auto":
            self._world_up_override = _validate_axis_string(world_up)
        if lr_mapping is not None:
            self._validate_and_set_lr_mapping(lr_mapping)
        # Before the guess below, which may read the anchor.
        if body_anchor is not None:
            self.body_anchor = body_anchor

        if world_up == "auto":
            self._settle_world_up()

        # Frozen last, once nothing can raise: a Bvh(...) that raises owns
        # nothing, so the caller's nodes stay editable for a retry. That
        # includes the guess above, whose disagreement warning raises
        # under warnings-as-errors; it reads the same node objects either
        # way.
        self._freeze_skeleton()

    def _check_motion_fits_skeleton(self, nodes: Sequence[BvhNode] | None = None) -> None:
        """Raise ``ValueError`` unless the motion has one frame count and
        one column per joint of ``nodes`` (default: the clip's own).

        The constructor runs it on the motion it is given, the rebuild of
        a derived clip or a pickle on the motion it carries, and
        ``copy(nodes=...)`` on the skeleton it is given, before building
        the copy, so that skeleton is held to the same check.
        """
        if nodes is None:
            nodes = self.nodes
        motion = self._motion()
        for entry in self._MOTION_ARRAYS:
            _check_array_fits_skeleton(entry, motion[entry.name], nodes)

    def _check_static_root_motion(self) -> None:
        """Raise ``ValueError`` unless a static root stays at its offset with no rotation.

        A static root (``CHANNELS 0``) has no motion of its own: its
        ``root_pos`` is its offset and its row of ``joint_angles`` is
        zero, in every frame, exactly. The constructor runs it on the
        motion it is given.
        """
        if not self.root.is_static():
            return
        if np.any(self._root_pos != self.root.offset):
            self._refuse_static_root("root_pos places the root away from its offset")
        self._refuse_root_rotation(self._joint_angles, "joint_angles rotates the root")

    def _refuse_root_rotation(self, joint_angles: npt.NDArray[np.float64], what: str) -> None:
        """Raise ``ValueError`` if the root is static and *joint_angles*,
        ``(F, J, 3)``, rotates it in some frame (its row 0 is not zero);
        *what* says what the caller does to it."""
        if self.root.is_static() and np.any(joint_angles[:, :1] != 0):
            self._refuse_static_root(what)

    def _refuse_static_root(self, what: str) -> None:
        """Raise ``ValueError`` if the root is static, *what* saying what the caller does to it."""
        if self.root.is_static():
            raise ValueError(
                f"{what}, but root {self.root.name!r} is static (CHANNELS 0): it "
                f"stays at its offset with no rotation in every frame."
            )

    def _motion(self) -> dict[str, npt.NDArray[np.float64]]:
        """Each motion array the clip holds, by name, in list order: the
        arrays themselves, not views or copies.

        The one way an operation reads the clip's motion. Raises
        ``ValueError`` if the arrays disagree on frame count, which the
        setters leave possible so that a user replaces the arrays one at a
        time: no operation then reads one array against the frames of
        another, where NumPy would broadcast a one-frame array across them
        or the shorter array would cut the frames short.
        """
        motion = self._motion_unchecked()
        _check_frame_counts(self._MOTION_ARRAYS, motion)
        return motion

    def _motion_unchecked(self) -> dict[str, npt.NDArray[np.float64]]:
        """Each motion array the clip holds, as :meth:`_motion` gives them,
        with no frame-count check.

        For the readers that a clip whose arrays disagree on frame count
        must not stop: the frame count itself, the question whether the
        motion binds the skeleton (which reads skeleton dims only),
        equality (which compares shapes before values) and the check of an
        edit's result (:meth:`_write_motion`). An operation reads
        :meth:`_motion`.
        """
        return {entry.name: getattr(self, entry.attribute) for entry in self._MOTION_ARRAYS}

    def _binds_skeleton(self) -> bool:
        """Whether the motion binds the skeleton's joints.

        A clip built without motion holds the empty arrays
        (:meth:`_MotionArray.empty`) whatever its skeleton, with 0 on each
        skeleton dim: there is no motion to fit, and it takes any skeleton.
        """
        motion = self._motion_unchecked()
        return any(
            motion[entry.name].shape != entry.empty().shape
            for entry in self._MOTION_ARRAYS
            if entry.skeleton_axes
        )

    @property
    def nodes(self) -> tuple[BvhNode, ...]:
        """Skeleton hierarchy, one node per joint and end site.

        In depth-first order as the file lists it: ``nodes[0]`` is the
        :class:`~pybvh.bvhnode.BvhRoot`, and end sites
        (:class:`~pybvh.bvhnode.BvhEndSite`) are included. ``len(nodes)``
        is therefore the node count *N* that indexes
        :meth:`node_positions` (shape ``(F, N, 3)``), not the joint count
        *J* that indexes :attr:`joint_angles`; :attr:`node_index` and
        :attr:`joint_index` map names to each.

        The clip owns these nodes and has frozen them: assigning a
        node's ``name``, ``offset``, ``parent``, ``children`` or channels
        raises ``AttributeError``, and ``nodes`` itself cannot be
        assigned, nor edited in place, being a tuple (``list(bvh.nodes)``
        gives a list). Change the skeleton through the methods that edit it
        (:meth:`scale`, :meth:`retarget`, :meth:`extract_joints`,
        :meth:`change_euler_order`), or edit it as a table and copy the
        clip onto it: ``table = bvh.to_node_table()``, edit the rows, then
        ``bvh.copy(nodes=table)`` (see :meth:`copy`). The
        methods swap in a new tuple of new nodes, even with
        ``inplace=True``, so a tuple or node taken before the call keeps
        the old skeleton.
        """
        return self._nodes

    @property
    def frame_time(self) -> float:
        """Seconds between successive frames.

        ``0`` means "unset": it is the constructor's default, and any
        clip may carry it, with or without frames. Most operations that
        need a rate raise ``ValueError`` on an unset clip, and so does
        :func:`~pybvh.io.write_bvh_file`, since a file must state its
        frame time; :meth:`to_df_dict` gives it NaN times. Assigning a
        negative number, NaN or infinity raises ``ValueError``.

        A clip written by :func:`~pybvh.io.write_bvh_file` and read
        back has the same frame time. A frame time read from another
        writer's file may differ from its text, since
        :func:`~pybvh.io.read_bvh_file` recovers a rate that the writer
        truncated or rounded: ``0.033333`` reads as ``1 / 30`` and
        ``0.033367`` as ``1 / 29.97``. The Notes of
        :func:`~pybvh.io.read_bvh_file` give the rule.
        """
        return self._frame_time

    @frame_time.setter
    def frame_time(self, value: float) -> None:
        _validate_frame_time(value)
        self._frame_time = value

    @property
    def fps(self) -> float:
        """Frames per second — convenience inverse of :attr:`frame_time`.

        Returns ``0.0`` when ``frame_time == 0`` (the "unset" sentinel)
        rather than raising, mirroring the behaviour of :meth:`__str__`.
        Assigning sets ``frame_time`` to ``1 / fps``, the value
        :func:`~pybvh.io.read_bvh_file` gives a file whose rate it
        recovers: after ``bvh.fps = 29.97``, ``bvh.frame_time`` equals
        the frame time read from ``Frame Time: 0.033367``. The rate must
        be a positive finite number, so ``0``, a negative rate, NaN and
        infinity raise ``ValueError`` rather than mark the rate as unset.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
        >>> if bvh.fps != 30:
        ...     bvh = bvh.resample(30)
        """
        return 1.0 / self._frame_time if self._frame_time > 0 else 0.0

    @fps.setter
    def fps(self, value: float) -> None:
        _validate_fps(value)
        self.frame_time = 1.0 / value

    @property
    def root_pos(self) -> npt.NDArray[np.float64]:
        """Root translation per frame, shape ``(F, 3)``.

        Returns a **read-only view** of the underlying array — call
        ``bvh.root_pos.copy()`` if you need a writable array. To
        replace the whole array, assign via the setter
        (``bvh.root_pos = new_arr``); for in-place edits, copy → mutate
        → assign back.

        On a static root (``CHANNELS 0``, see
        :meth:`~pybvh.bvhnode.BvhRoot.is_static`) it holds the root's
        offset in every frame, and the setter raises ``ValueError``.

        Its length is the clip's frame count. To change the frame count,
        assign ``root_pos`` and :attr:`joint_angles`, and on a skeleton
        with translating joints :attr:`joint_translations` too, each with
        the same number of frames: the setters check no frame count, and
        an operation on a clip whose arrays disagree raises
        ``ValueError``. On a skeleton with no translating joint,
        :attr:`joint_translations` holds no value and takes the frame
        count of ``root_pos``.
        """
        view = self._root_pos.view()
        view.flags.writeable = False
        return view

    @root_pos.setter
    def root_pos(self, value: npt.ArrayLike) -> None:
        self._refuse_static_root("Setting root_pos moves the root")
        self._root_pos = _ROOT_POS.checked(value)
        if _TRANSLATING_JOINTS.count(self.nodes) == 0:
            # With no translating joint, joint_translations holds no value,
            # so it has no frame count of its own: it takes root_pos's,
            # the clip's frame count.
            none: npt.NDArray[np.float64] = np.empty((len(self._root_pos), 0, 3))
            self._joint_translations = none
        self._derived.user_replaced_motion()

    @property
    def joint_angles(self) -> npt.NDArray[np.float64]:
        """Per-joint Euler angles, shape ``(F, J, 3)`` (radians).

        Returns a **read-only view** of the underlying array — call
        ``bvh.joint_angles.copy()`` if you need a writable array. To
        replace the whole array, assign via the setter
        (``bvh.joint_angles = new_arr``); for in-place edits, copy →
        mutate → assign back.

        Read-only view protects against the common footgun of
        ``angles = b.joint_angles; angles -= angles.mean(axis=0)``
        silently corrupting the Bvh.

        On a static root (``CHANNELS 0``, see
        :meth:`~pybvh.bvhnode.BvhRoot.is_static`) the root's row is zero
        in every frame, and the setter raises ``ValueError`` on an array
        whose row 0 is not.
        """
        view = self._joint_angles.view()
        view.flags.writeable = False
        return view

    @joint_angles.setter
    def joint_angles(self, value: npt.ArrayLike) -> None:
        checked = _JOINT_ANGLES.checked(value)
        self._refuse_root_rotation(checked, "These joint_angles rotate the root")
        self._joint_angles = checked
        self._derived.user_replaced_motion()

    @property
    def joint_translations(self) -> npt.NDArray[np.float64]:
        """Per-frame positions of the translating joints, shape ``(F, P, 3)``.

        A translating joint is a joint other than the root whose
        ``CHANNELS`` line declares 3 position channels besides its 3
        rotation channels (its ``pos_channels`` is ``('X', 'Y', 'Z')``).
        Column *p* holds the *p*-th such joint in :attr:`nodes` order: its
        position relative to its parent, in its parent's frame, as the
        file's values (``X Y Z``), not the difference from its offset.
        In each frame they replace the joint's ``offset``, which is the
        rest pose only (ADR 0010): a joint with ``OFFSET 0 10 0`` and the
        values ``0 12 0`` is 12 above its parent, not 22. P = 0 on a
        skeleton whose joints other than the root only rotate.

        Returns a **read-only view** of the underlying array, like
        :attr:`root_pos`. To replace the whole array, assign via the
        setter (``bvh.joint_translations = new_arr``); the shape must
        fit the skeleton (``ValueError`` otherwise), and the assignment
        starts the world-up guess again, as for :attr:`root_pos`.
        """
        view = self._joint_translations.view()
        view.flags.writeable = False
        return view

    @joint_translations.setter
    def joint_translations(self, value: npt.ArrayLike) -> None:
        checked = _JOINT_TRANSLATIONS.checked(value)
        # The skeleton fixes P whatever the motion's frame count, which the
        # setters leave to the caller, as for root_pos and joint_angles.
        _check_array_fits_skeleton(_JOINT_TRANSLATIONS, checked, self.nodes)
        self._joint_translations = checked
        self._derived.user_replaced_motion()

    def _set_motion(self, **motion: npt.ArrayLike) -> None:
        """Write motion as pybvh's own edits do, through no public setter.

        The transforms and ``a += b`` write here and report that the
        motion changed. The public setters report that the user replaced
        the motion, which starts the world-up guess again (ADR 0003), as
        setting the L/R mapping does for a guess of the clip's own (ADR
        0004).

        Takes any of the motion arrays by name (``_MOTION_ARRAYS``) and
        leaves the others as they are. On a clip built without motion, an
        array with no frame is written as the empty array
        (:meth:`_MotionArray.empty`): an edit that reads the motion with
        the skeleton's dims (:meth:`_motion_on_skeleton`) leaves the clip
        without motion, binding no skeleton.
        """
        if not self._binds_skeleton():
            empty = {entry.name: entry.empty() for entry in self._MOTION_ARRAYS}
            motion = {
                name: empty[name] if name in empty and len(np.asarray(array)) == 0 else array
                for name, array in motion.items()
            }
        self._write_motion(motion)

    def _write_motion(self, motion: Mapping[str, npt.ArrayLike]) -> None:
        """Write the motion arrays *motion* names, as they are, and report
        that the motion changed; the rebuild of a derived clip writes here.

        Raises ``ValueError``, before writing anything, if the clip's arrays
        would disagree on frame count after the write: pybvh's own edits
        never leave such a clip, and an edit that reads one array of a clip
        whose arrays disagree, as ``translate_root`` reads ``root_pos``, is
        refused here.
        """
        unknown = motion.keys() - {entry.name for entry in self._MOTION_ARRAYS}
        if unknown:
            raise TypeError(f"_set_motion() got no motion array named {sorted(unknown)}")
        written = {
            entry.name: entry.checked(motion[entry.name])
            for entry in self._MOTION_ARRAYS
            if entry.name in motion
        }
        _check_frame_counts(self._MOTION_ARRAYS, {**self._motion_unchecked(), **written})
        for entry in self._MOTION_ARRAYS:
            if entry.name in written:
                setattr(self, entry.attribute, written[entry.name])
        self._derived.motion_changed()

    def _motion_on_skeleton(self) -> dict[str, npt.NDArray[np.float64]]:
        """Each motion array by name, with the skeleton's length on each
        skeleton dim.

        A clip built without motion holds the empty arrays, with 0 on each
        skeleton dim (:meth:`_binds_skeleton`); an operation that walks the
        skeleton's joints reads their rows here, where those arrays have
        no frame and the skeleton's dims. A clip with motion gives its
        arrays themselves.

        Raises ``ValueError`` if the arrays disagree on frame count, as
        :meth:`_motion` does.
        """
        motion = self._motion()
        if self._binds_skeleton():
            return motion
        for entry in self._MOTION_ARRAYS:
            if entry.skeleton_axes:
                dims = [
                    dim.count(self.nodes) if isinstance(dim, _SkeletonDim) else dim
                    for dim in entry.frame_shape
                ]
                motion[entry.name] = np.empty((0, *dims))
        return motion

    @property
    def frame_count(self) -> int:
        """Number of frames (computed from root_pos)."""
        # The first motion array's frame count, which the others match
        # unless the setters left them disagreeing (see _motion).
        return len(self._motion_unchecked()[self._MOTION_ARRAYS[0].name])

    @property
    def root(self) -> BvhRoot:
        """The root joint, ``nodes[0]``.

        The node placed in the world frame: its translation per frame is
        :attr:`root_pos`, and its rotation is joint 0 of
        :attr:`joint_angles`. A translating joint also has position
        channels, in its parent's frame (:attr:`joint_translations`); a
        static root (``CHANNELS 0``) has none and stays at its offset.
        Read-only: the root is whatever ``nodes`` starts with.
        """
        return self._nodes[0]  # type: ignore[return-value]

    @property
    def _euler_column_names(self) -> list[str]:
        """Column names describing root_pos + joint_angles in flat layout order.

        The motion columns of :meth:`to_df_dict` in euler mode, without
        ``time``: useful for inspecting the channel mapping. Generated on
        the fly from the node hierarchy by :func:`_motion_column_names`.
        """
        return _motion_column_names(self.nodes, "euler")

    def __str__(self) -> str:
        source = ""
        if self.source_path is not None:
            source = f", from {Path(self.source_path).name}"
        return (
            f"{self.joint_count} joints, {self.frame_count} frames at "
            f"{self.fps:.1f} fps (frame_time={self.frame_time:.6f}s{source})"
        )

    def __repr__(self) -> str:
        return (
            f"Bvh(joints={self.joint_names!r}, "
            f"frame_count={self.frame_count}, "
            f"frame_time={self.frame_time:.6f})"
        )

    def __eq__(self, other: object) -> bool:
        """Full-content equality: skeleton, channel layout, timing, motion.

        Hierarchy (names, parents, offsets, end sites) is compared via
        :meth:`matches_hierarchy` with ``atol=0``, so offsets must be
        equal exactly, channel layout via
        :meth:`matches_channels`, and the motion (:attr:`root_pos`,
        :attr:`joint_angles`, :attr:`joint_translations`) in every frame,
        exactly. ``source_path`` is ignored.
        """
        if not isinstance(other, Bvh):
            return NotImplemented
        if not self.matches_hierarchy(other, atol=0):
            return False
        if not self.matches_channels(other):
            return False
        if self.frame_time != other.frame_time:
            return False
        # Unchecked: array_equal compares shapes first, so a clip whose
        # arrays disagree on frame count compares without a broadcast.
        other_motion = other._motion_unchecked()
        return all(
            np.array_equal(array, other_motion[name])
            for name, array in self._motion_unchecked().items()
        )

    def matches_hierarchy(
        self, other: Bvh, match_offsets: bool = True, atol: float = _OFFSET_ATOL
    ) -> bool:
        """Whether ``self`` and ``other`` share the same skeleton hierarchy.

        Hierarchy is defined as: same node names in topological order
        (including end sites), each node of the same kind (a joint or an
        end site), each node's parent the node at the same position (not
        merely a node of the same name, since names may repeat), and —
        when ``match_offsets=True`` (default) — same rest-pose offsets,
        each component within the absolute ``atol``. Motion data, Euler
        rotation orders, and frame timing are NOT compared.

        Use this when you need to know that two clips describe the same
        skeleton in the same rest pose — e.g. before batching to a
        rotation-invariant representation (``6d`` / ``quat`` /
        ``rotmat``) whose channel layout doesn't depend on Euler order.

        Pass ``match_offsets=False`` when the caller is about to overwrite
        rest offsets anyway (e.g. retargeting): it loosens the check to
        the skeleton *graph* alone — joint names and parent structure —
        accepting two characters of different bone proportions.

        Parameters
        ----------
        other : Bvh
            The clip to compare against. Anything that is not a ``Bvh``
            returns ``False`` rather than raising.
        match_offsets : bool, optional
            If True (default), require rest-pose offsets to agree within
            ``atol``. If False, ignore offsets and check only the
            skeleton graph (names + parent structure).
        atol : float, optional
            Absolute tolerance for each offset component, in the file's
            length unit (default ``1e-6``, above the writer's rounding to
            six decimals). No relative tolerance is added, so ``atol=0``
            is exact. Ignored when ``match_offsets=False``.

        Returns
        -------
        bool

        See Also
        --------
        matches_channels : Compare per-joint position channels and Euler orders.
        matches_topology : Conjunction of hierarchy + channels.
        """
        if not isinstance(other, Bvh):
            return False
        difference = _hierarchy_difference(
            self.nodes, other.nodes, atol=atol if match_offsets else None
        )
        return difference is None

    def matches_channels(self, other: Bvh) -> bool:
        """Whether ``self`` and ``other`` share the same channel layout.

        Compares, joint by joint in node order, which joints declare
        position channels (which joints translate) and each joint's
        Euler rotation order. Only the declaration is compared, never the
        values of :attr:`joint_translations`: a bone that stretches and
        comes back still matches its own skeleton. This is a
        *serialization* property:
        clips with identical underlying rotations but different stored
        Euler orders have different channel layouts.

        Use this in addition to :meth:`matches_hierarchy` when batching
        to a representation whose channel layout depends on the source
        Euler order (``euler`` / ``axisangle``).

        Parameters
        ----------
        other : Bvh
            The clip to compare against. Anything that is not a ``Bvh``
            returns ``False`` rather than raising.

        Returns
        -------
        bool

        See Also
        --------
        matches_hierarchy : Compare joint hierarchy and rest offsets.
        matches_topology : Conjunction of hierarchy + channels.
        """
        if not isinstance(other, Bvh):
            return False
        return _channel_difference(self.nodes, other.nodes) is None

    def matches_topology(self, other: Bvh) -> bool:
        """Whether ``self`` and ``other`` share both hierarchy and channel layout.

        Convenience for ``matches_hierarchy(other) and matches_channels(other)``.
        Two Bvhs that satisfy this predicate can be batched together for
        any representation (``euler``, ``axisangle``, ``6d``, ``quat``,
        ``rotmat``) without conversion.

        .. note::
            Prior to 0.7.0, ``matches_topology`` checked only
            ``joint_names`` and ``euler_orders`` — it did not catch
            differences in parent structure or rest offsets. The current
            definition is stricter: clips with identical names but
            differing rest offsets no longer match.

        Parameters
        ----------
        other : Bvh
            The clip to compare against. Anything that is not a ``Bvh``
            returns ``False`` rather than raising.

        Returns
        -------
        bool

        See Also
        --------
        matches_hierarchy : The hierarchy half (joints, parents, offsets).
        matches_channels : The channel-layout half (per-joint position channels
            and Euler orders).
        """
        return self.matches_hierarchy(other) and self.matches_channels(other)

    def matches_rate(self, other: Bvh | float) -> bool:
        """Whether ``self`` has the same rate as ``other``, a clip or a rate in fps.

        The same rate means frame times within a relative ``2e-6``:
        ``abs(a - b) <= 2e-6 * max(a, b)``, where a rate in fps stands
        for the frame time ``1 / fps`` that ``bvh.fps = fps`` sets. The
        tolerance is relative because the error a computation or a saved
        table leaves in a frame time scales with the frame time, so one
        tolerance serves 24 fps and 1000 fps alike. ``2e-6`` lets a
        29.97 fps file, read as ``1 / 29.97``, match an exact NTSC clip at
        ``1001 / 30000``, which differ by 1e-6 of the larger; it is still
        far below the 1e-3 between 29.97 and 30 fps. So a clip at
        ``frame_time = 0.013`` matches ``b.resample(1 / 0.013)``, whose
        frame time is ``0.013000000000000001``.

        Two clips whose frame time is ``0`` (unset) have the same rate; an
        unset clip has the rate of no set clip and of no fps. ``==`` on
        two clips is exact, as it is for their arrays.

        Parameters
        ----------
        other : Bvh or float
            The clip, or the rate in frames per second, to compare against.

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If ``other`` is an fps that is not a positive finite number,
            or one so small that ``1 / other`` overflows.

        See Also
        --------
        matches_topology : Whether two clips share a skeleton.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
        >>> if not bvh.matches_rate(30):
        ...     bvh = bvh.resample(30)
        """
        if isinstance(other, Bvh):
            return _same_rate(self.frame_time, other.frame_time)
        return _matches_fps(self.frame_time, other)

    def __len__(self) -> int:
        """Number of frames. Equivalent to ``self.frame_count``."""
        return self.frame_count

    def __getitem__(self, key: int | slice) -> Bvh:
        """Return a new Bvh containing the selected frame(s).

        Integer keys return a single-frame (F=1) Bvh; slice keys return
        the selected frame range. Negative indices and reversed slices
        (``bvh[::-1]``) work as expected. ``frame_time`` is scaled by
        ``abs(step)`` when ``|step| > 1``; reversed playback at ``|step|=1``
        preserves ``frame_time``.

        Parameters
        ----------
        key : int or slice
            Frame index or slice. Fancy indexing (ndarray, list, boolean
            mask) and tuple keys are not supported.

        Raises
        ------
        IndexError
            Integer key outside ``[-frame_count, frame_count)``.
        TypeError
            Key of unsupported type.
        """
        if isinstance(key, (int, np.integer)):
            F = self.frame_count
            k = int(key)
            if k < -F or k >= F:
                raise IndexError(f"frame index {k} out of range for Bvh with {F} frames")
            i = k if k >= 0 else k + F
            return self._slice_frames(i, i + 1)
        if isinstance(key, slice):
            return self._slice_frames(key.start, key.stop, key.step)
        raise TypeError(
            f"Bvh indices must be int or slice, got {type(key).__name__}. "
            "For arbitrary frame selection, build a new Bvh manually from "
            "the required root_pos and joint_angles arrays."
        )

    def __add__(self, other: object) -> Bvh:
        """Concatenate two Bvh clips with the same skeleton.

        Returns a new Bvh with frames from ``self`` followed by ``other``.
        Raises ``ValueError`` unless the two share their topology
        (:meth:`matches_topology`: names, node kinds, parents, rest
        offsets within ``1e-6``, rotation orders), since ``other``'s
        motion is written onto ``self``'s bones; to join motion recorded
        on other bones on purpose, put it on this skeleton first,
        ``self + other.copy(nodes=self.to_node_table())``. Then raises
        ``ValueError`` unless the two have the same rate
        (:meth:`matches_rate`), since ``other``'s frames would play at
        ``self``'s rate: join ``other.resample(self.fps)`` to keep its
        timing on ``self``'s grid,
        or set ``other.frame_time = self.frame_time`` when the difference
        is only metadata. The result keeps ``self``'s frame time. Two
        unset clips (``frame_time`` 0) join, and the result is unset; an
        unset clip and a set one do not.
        """
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        return self._concat(other)

    def __iadd__(self, other: object) -> Bvh:
        """In-place concatenation. Grows ``self`` by appending ``other``'s frames.

        Checks the skeleton, then the rate, just like ``a + b``, and raises
        ``ValueError`` before changing ``self`` when either differs;
        ``self`` keeps its frame time. The values derived from the motion
        are recomputed on their next read.
        """
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        self._check_same_skeleton(other)
        self._check_same_rate(other, "a += b")
        other_motion = other._motion()
        self._set_motion(
            **{
                name: np.concatenate([array, other_motion[name]], axis=0)
                for name, array in self._motion().items()
            }
        )
        if self.source_path != other.source_path:
            self.source_path = None
        return self

    def __setitem__(self, key: int | slice, value: Bvh) -> None:
        """Splice frames from another Bvh into ``self`` in place.

        The skeleton of ``value`` must match ``self``'s topology, as for
        ``a + b`` (:meth:`matches_topology`), ``value`` must have
        ``self``'s rate, as for ``a + b`` (:meth:`matches_rate`; splice
        ``value.resample(self.fps)`` when it was recorded at another
        rate), and the slice length must equal ``value.frame_count``.
        Integer keys require ``value.frame_count == 1``. Each check raises
        before ``self`` changes, and ``self`` keeps its frame time.

        Frames are written into the motion arrays in place (no full-array
        copies), and the values derived from the motion are recomputed on
        their next read.
        """
        # --- key → canonical slice ---
        if isinstance(key, (int, np.integer)):
            F = self.frame_count
            k = int(key)
            if k < -F or k >= F:
                raise IndexError(f"frame index {k} out of range for Bvh with {F} frames")
            i = k if k >= 0 else k + F
            s = slice(i, i + 1)
        elif isinstance(key, slice):
            s = key
        else:
            raise TypeError(
                f"Bvh indices must be int or slice, got {type(key).__name__}. "
                "For array-level writes, assign to bvh.root_pos or bvh.joint_angles directly."
            )

        # --- value must be a Bvh ---
        if not isinstance(value, Bvh):
            raise TypeError(
                f"Bvh __setitem__ requires a Bvh value, got "
                f"{type(value).__name__}. For array-level writes, assign "
                "to bvh.root_pos or bvh.joint_angles directly."
            )

        # --- skeleton + frame_time ---
        self._check_same_skeleton(value)
        self._check_same_rate(value, "a[i:j] = b")

        # Read before the length check, which reads each clip's frame count:
        # on a clip whose arrays disagree it has none to compare.
        motion = self._motion()
        other_motion = value._motion()

        # --- length match (no resizing) ---
        target_len = len(range(*s.indices(self.frame_count)))
        if target_len != value.frame_count:
            raise ValueError(
                f"Cannot assign {value.frame_count} frames to a slice of "
                f"length {target_len}; __setitem__ does not resize. Use "
                "`a + b` to append, or frame slicing (`bvh[a:b]`) + manual "
                "array assignment for more complex splicing."
            )

        # --- in-place splice, bypassing the setters ---
        for name, array in motion.items():
            array[s] = other_motion[name]
        self._derived.user_replaced_motion()

    def copy(self, nodes: Sequence[BvhNode] | Sequence[Mapping[str, Any]] | None = None) -> Bvh:
        """Return an independent copy, on this skeleton or on an edited one.

        The node hierarchy, ``root_pos``, ``joint_angles`` and
        ``joint_translations``, ``frame_time``, ``source_path``, a manual
        ``world_up``, the warn preference, a user-set L/R mapping and a
        set :attr:`body_anchor` are copied. Editing either
        clip afterwards, its arrays or its nodes, never affects the other.
        A guessed world up is inherited (see :attr:`world_up`). Values the
        clip computes from these (forward-kinematics positions, the floor
        height) are not copied: the copy computes them again on first
        read.

        Parameters
        ----------
        nodes : list of dict or sequence of BvhNode, optional
            A skeleton for the copy, the way to change a clip's frozen
            skeleton: ``table = bvh.to_node_table()``, edit the rows, then
            ``bvh.copy(nodes=table)``. A node table or a node list, as
            :meth:`from_df` takes. A table is data: the copy builds nodes
            of its own from it. A node list is taken over as the
            constructor takes one: the copy owns those nodes and freezes
            them (nodes another clip owns are shared, being frozen), and
            a copy that raises leaves them editable. The motion
            binds the joints by position: joint *i* of ``nodes`` takes
            column *i* of ``joint_angles``, so names and offsets may
            change, the joint count may not, and neither may a joint's
            Euler order, since the angles are only meaningful in the order
            they were recorded in (:meth:`change_euler_order` converts
            them). Neither may which joints translate, nor the root's
            channel set: the motion was recorded on them. A translating
            joint's frames are its whole position (ADR 0010), so a new
            offset changes its rest pose only. A static root
            (``CHANNELS 0``) stands at its offset in every frame, so a new
            offset moves its ``root_pos``, and the whole body with it. A user-set L/R mapping follows its joints by position,
            under their new names, so each mapped joint's new name must be
            one no other joint carries (the mapping goes by name); one
            detected from the names is detected again from the new names.
            A set :attr:`body_anchor` follows its joint by position too,
            under its new name; on a clip without motion that takes
            another joint count, the copy drops it and uses the rule.
            World up is inherited, set or guessed: new offsets do not
            change which way is up. To guess it again from the new
            skeleton, ``edited.world_up = edited.world_up_inferred``.

        Returns
        -------
        Bvh

        Raises
        ------
        ValueError
            If ``nodes`` is not one tree in depth-first order (see
            :func:`~pybvh.nodes_from_table`), has another number of joints
            than ``joint_angles`` has columns, changes a joint's Euler
            order, which joints translate or the root's channel set, or
            gives the root position channels other than XYZ; or
            if it gives a joint of a user-set L/R mapping a name another
            joint carries too (on a clip without motion, which binds no
            joint, if it lacks a name the mapping uses).
        TypeError
            If ``nodes`` is neither a node table nor a node list.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
        >>> table = bvh.to_node_table()
        >>> table[1]["offset"] = [0.0, 12.0, 0.0]
        >>> edited = bvh.copy(nodes=table)
        """
        if nodes is None:
            return self._derive_clip({name: array.copy() for name, array in self._motion().items()})
        new_nodes = _nodes_from_table_or_list(nodes, "nodes")
        # Every check runs before the copy is built, as in the constructor:
        # building it freezes the nodes, and a copy that raises must leave
        # a node list it was given editable, as a Bvh(...) that raises does.
        joints = [node for node in self.nodes if _is_joint(node)]
        new_joints = [node for node in new_nodes if _is_joint(node)]
        if self._binds_skeleton():
            self._check_motion_fits_skeleton(new_nodes)
            root, new_root = self.root, new_nodes[0]
            assert isinstance(new_root, BvhRoot)  # one tree, root first
            if root.is_static() != new_root.is_static():
                raise ValueError(
                    f"nodes changes the channels of the root {root.name!r} "
                    f"from {_channel_declaration(root)} to "
                    f"{_channel_declaration(new_root)}, but the clip's motion "
                    f"is recorded on {_channel_declaration(root)}."
                )
            for joint, new_joint in zip(joints, new_joints, strict=True):
                if _is_translating_joint(joint) != _is_translating_joint(new_joint):
                    # A column of joint_translations is a joint's frames:
                    # moving the position channels to another joint would
                    # hand it frames recorded for this one.
                    has, lacks = ("the clip", "nodes")
                    if not _is_translating_joint(joint):
                        has, lacks = lacks, has
                    raise ValueError(
                        f"nodes changes which joints translate: joint "
                        f"{joint.name!r} declares position channels in {has} "
                        f"but not in {lacks}, and joint_translations holds the "
                        f"frames of the clip's own translating joints."
                    )
                old_order = "".join(joint.rot_channels)
                new_order = "".join(new_joint.rot_channels)
                if new_order != old_order:
                    # The node, not its name: names repeat, and
                    # change_euler_order refuses a name two joints share.
                    position = self.index(joint, space="node")
                    raise ValueError(
                        f"nodes changes the Euler order of joint {joint.name!r} "
                        f"from {old_order!r} to {new_order!r}, but the clip's "
                        f"angles are in {old_order!r}. To change an order, "
                        f"convert the angles with it: "
                        f"bvh.change_euler_order({new_order!r}, joint=bvh.nodes[{position}])."
                    )
        setting = self._lr_mapping_setting
        if isinstance(setting, dict):
            new_names = [joint.name for joint in new_joints]
            if len(new_joints) == len(joints):
                index = self.joint_index
                # The mapping goes by name, so it cannot hold a mapped joint
                # whose new name another joint carries too: refuse before
                # the renamed mapping overwrites a pair or names that joint.
                # A name the clip already repeated, on the same joints, is
                # the clip's own ambiguity, which copy() keeps too.
                old_names = self.joint_names
                for name in [name for pair in setting.items() for name in pair]:
                    new_name = new_names[index[name]]
                    carriers = [i for i, other in enumerate(new_names) if other == new_name]
                    before = [i for i, other in enumerate(old_names) if other == name]
                    if len(carriers) > 1 and carriers != before:
                        raise ValueError(
                            f"nodes gives the mapped joint {name!r} the name "
                            f"{new_name!r}, which {len(carriers)} joints carry, but "
                            f"the L/R mapping identifies joints by name. Keep "
                            f"each mapped joint's name unique, or set "
                            f"lr_mapping again on the copy."
                        )
                setting = {
                    new_names[index[left]]: new_names[index[right]]
                    for left, right in setting.items()
                }
            # Unbound (a clip without motion took another joint count), the
            # names are all there is to go by.
            setting = _validated_lr_mapping(setting, new_names)
        motion = {name: array.copy() for name, array in self._motion().items()}
        new_root = new_nodes[0]
        if isinstance(new_root, BvhRoot) and new_root.is_static():
            motion["root_pos"] = _static_root_pos(new_root.offset, self.frame_count)
        edited = self._derive_clip(motion, nodes=new_nodes)
        if isinstance(setting, dict):
            edited._set_lr_mapping_setting(setting)
        anchor = self._body_anchor_setting
        if anchor is not None:
            edited._set_body_anchor_setting(
                self._anchor_on(new_nodes, joints, new_joints, self.nodes[anchor])
            )
        return edited

    @staticmethod
    def _anchor_on(
        new_nodes: Sequence[BvhNode],
        joints: Sequence[BvhNode],
        new_joints: Sequence[BvhNode],
        anchor: BvhNode,
    ) -> int | None:
        """The position in ``new_nodes`` of the set body anchor ``anchor``,
        carried by ``copy(nodes=)``.

        By position among the joints when the joint count stays, so a
        renamed anchor is followed. ``None`` when the joint count changes
        (only a clip without motion allows it): the copy then uses the
        rule. A name is not followed, since a name can repeat.
        """
        if len(new_joints) != len(joints):
            return None
        joint = next(j for j, node in enumerate(joints) if node is anchor)
        carried = new_joints[joint]
        return next(i for i, node in enumerate(new_nodes) if node is carried)

    def _derive_clip(
        self,
        motion: Mapping[str, npt.NDArray[np.float64]],
        nodes: list[BvhNode] | tuple[BvhNode, ...] | None = None,
    ) -> Bvh:
        """Build a clip from new primary state and this clip's settings.

        The one way pybvh derives a clip (ADR 0003): the result carries what
        :meth:`__getstate__` lists, with the given motion and skeleton
        (default: a deep copy of this one's), and none of this clip's
        derived state. On a skeleton given as ``nodes`` it carries no set
        body anchor: the caller carries it over with
        :meth:`_set_body_anchor_setting`. ``motion`` holds every motion array by name
        (``_MOTION_ARRAYS``): an array it leaves out is refused, not taken
        from this clip. The arrays are taken as they are, not copied; the
        caller hands over arrays no other clip holds.
        """
        names = [entry.name for entry in self._MOTION_ARRAYS]
        unknown = motion.keys() - set(names)
        if unknown:
            raise TypeError(f"_derive_clip() got no motion array named {sorted(unknown)}")
        missing = [name for name in names if name not in motion]
        if missing:
            raise TypeError(f"_derive_clip() is missing motion arrays {missing}")
        state = self.__getstate__()
        for entry in self._MOTION_ARRAYS:
            del state[entry.name]
        state.update(motion)
        if nodes is None:
            state["nodes"] = copy.deepcopy(self.nodes)
        else:
            # A position in this clip's nodes says nothing of another
            # skeleton's: the caller carries a set anchor over to it.
            state["nodes"] = nodes
            state["body_anchor_setting"] = None
        setting = state["lr_mapping_setting"]
        if isinstance(setting, dict):
            state["lr_mapping_setting"] = dict(setting)
        new_bvh = type(self).__new__(type(self))
        new_bvh.__setstate__(state)
        return new_bvh

    def __getstate__(self) -> dict[str, Any]:
        """What a pickle and every derived clip carry (ADR 0003).

        The primary state (skeleton, motion, frame time, source path), the
        user's settings (world-up override, warn preference, L/R choice,
        set body anchor)
        and the world up this clip guessed, which the rebuilt clip
        inherits; never a derived value, which the rebuilt clip computes
        again on first read.
        """
        self._settle_world_up()
        derived = self._derived
        guess = derived.inherited_world_up or derived.inferred_world_up
        return {
            "nodes": self.nodes,
            # Keyed by the arrays' names, which are the keys pickles have
            # always held: a pickle reads the same in earlier versions.
            **self._motion(),
            "frame_time": self.frame_time,
            "source_path": self.source_path,
            "world_up_override": self._world_up_override,
            # The guess's source travels with it, so the rebuilt clip
            # reports how the axis it inherits was obtained. A plain
            # (axis, source) pair: a pickle names no private class.
            "world_up_guess": None if guess is None else tuple(guess),
            "warn_on_world_up_disagreement": self._warn_on_world_up_disagreement,
            "lr_mapping_setting": self._lr_mapping_setting,
            # The set body anchor's position in "nodes", or None for the
            # rule: a position, so a name that several nodes carry is
            # carried too.
            "body_anchor_setting": self._body_anchor_setting,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Built with no frames, so the constructor guesses no world up;
        # the motion is assigned after, then the settings and the guess.
        Bvh.__init__(
            self,
            nodes=state["nodes"],
            frame_time=state["frame_time"],
            source_path=state["source_path"],
            warn_on_world_up_disagreement=state["warn_on_world_up_disagreement"],
        )
        if "joint_translations" not in state:
            # Pickled before joints translated (#186): none of them does.
            state = {**state, "joint_translations": np.empty((len(state["root_pos"]), 0, 3))}
        # As they are: a derived clip holds the arrays it is given, a
        # clip with no frame those of its skeleton or the empty ones.
        self._write_motion({entry.name: state[entry.name] for entry in self._MOTION_ARRAYS})
        if self._binds_skeleton():
            self._check_motion_fits_skeleton()
        self._world_up_override = state["world_up_override"]
        self._set_lr_mapping_setting(state["lr_mapping_setting"])
        # Absent from a pickle made before the anchor could be set.
        self._set_body_anchor_setting(state.get("body_anchor_setting"))
        guess = state["world_up_guess"]
        self._derived.inherited_world_up = None if guess is None else _WorldUpGuess(*guess)

    def _settle_world_up(self) -> None:
        """Guess world up now, unless it is set or already guessed.

        A clip with motion guesses at load, and before a clip is derived
        from it, so that the guess is made once, on the source, and
        inherited from there (ADR 0003). A clip with no frames or a lone
        root is left to guess on first read: it has no frame to guess
        from, and a lone root warns that it falls back to ``'+y'``.
        """
        if self.frame_count > 0 and len(self.nodes) > 1 and self._has_body():
            _ = self.world_up

    def _rotate_world_up_guess(self, rotation: npt.NDArray[np.float64]) -> None:
        """Turn the guessed world up, inferred or inherited, by an
        axis-aligned ``rotation`` of the whole scene.

        ``reorient_world_up`` rotates every frame and offset by
        ``rotation``, so the axis guessed before is, exactly, the axis
        those frames would be guessed to have now; turning it keeps the
        guess made once (ADR 0003, decision 3), and the guess
        :attr:`world_up` warned about turns with it, so the turned guess
        does not warn again. A clip that has guessed nothing is left to
        guess from its new frames.
        """
        derived = self._derived
        for name in ("inferred_world_up", "inherited_world_up", "world_up_guess_announced"):
            guess = getattr(derived, name)
            if guess is not None:
                turned = _get_main_direction(rotation @ _axis_to_vector(guess.axis))
                assert turned is not None  # a rotation keeps a unit vector unit
                setattr(derived, name, guess._replace(axis=turned))
        # The turned guess is the world up every value measured along it
        # now reads, as a new override is.
        derived.world_up_changed()

    def _freeze_skeleton(self) -> None:
        """Freeze every node the clip owns, so its skeleton cannot change
        under ``joint_angles`` and the derived values."""
        for node in self.nodes:
            node._freeze()

    def _replace_skeleton(self, nodes: Sequence[BvhNode]) -> None:
        """Swap in a new skeleton: the one way a clip's skeleton changes
        after construction.

        Every skeleton edit pybvh makes (``scale``, ``mirror``,
        ``retarget``, the rest and world-up reorientations,
        ``change_euler_order``) builds a fresh node list with
        ``nodes_from_table`` and hands it here; none writes into a node
        the clip owns, so an edit cannot depend on the order of its
        statements, and a node object taken before the edit keeps its old
        values. The new nodes keep the joint layout of ``joint_angles``.
        """
        _check_node_tree(nodes)
        self._nodes = tuple(nodes)
        self._freeze_skeleton()
        self._derived.skeleton_changed()

    def _replace_offsets(
        self, offsets: npt.ArrayLike, *, translations: npt.ArrayLike | None = None
    ) -> None:
        """Replace every node's offset, given in ``nodes`` order as ``(N, 3)``,
        and the translations that go with them.

        A translating joint's frames replace its offset (ADR 0010), so an
        edit of the offsets says what it does to ``joint_translations``:
        ``translations`` is the new array, which this writes with the
        offsets. It may be omitted only on a clip with no translating
        joint, where there is nothing to leave stale; omitted on another
        clip, it raises ``TypeError`` before anything changes. A static
        root (``CHANNELS 0``) stands at its offset in every frame, so its
        ``root_pos`` follows its new offset.
        """
        current = self._motion_on_skeleton()["joint_translations"]
        if translations is None:
            if current.shape[1] != 0:
                raise TypeError(
                    "_replace_offsets() needs translations= on a clip with "
                    "translating joints: their frames replace their offsets, "
                    "so an offset edit must say what it does to them."
                )
        else:
            translations = _JOINT_TRANSLATIONS.checked(translations)
            if translations.shape != current.shape:
                raise ValueError(
                    f"translations has shape {translations.shape}, but "
                    f"joint_translations has {current.shape}."
                )
        offsets = np.asarray(offsets, dtype=np.float64)
        table = nodes_to_table(self.nodes)
        for row, offset in zip(table, offsets, strict=True):
            row["offset"] = offset
        self._replace_skeleton(nodes_from_table(table))
        motion: dict[str, npt.ArrayLike] = {}
        if translations is not None:
            motion["joint_translations"] = translations
        if self.root.is_static():
            motion["root_pos"] = _static_root_pos(offsets[0], self.frame_count)
        if motion:
            self._set_motion(**motion)

    def _offsets(self) -> npt.NDArray[np.float64]:
        """Every node's offset, in ``nodes`` order, as a new ``(N, 3)`` array."""
        return np.array([node.offset for node in self.nodes], dtype=np.float64).reshape(-1, 3)

    # ------------------------------------------------------------------
    # Orientation: world_up and forward_at
    # ------------------------------------------------------------------

    @property
    def world_up(self) -> str:
        """Gravity axis of the BVH coordinate system.

        Returned as a signed axis string (``'+y'``, ``'-z'``, etc.).
        Constant per file. Guessed from the first animation frame's
        direction from :attr:`body_anchor` to the head, with rest-pose
        topology as fallback. On a clip with no body
        (:attr:`body_anchor` ``None``) the guess raises ``ValueError``
        until you set the axis or name the anchor.
        Issues a ``UserWarning`` if the first frame and rest pose disagree,
        unless the clip (or the clip it derives from) was loaded with
        ``warn_on_world_up_disagreement=False``, and one if nothing says
        which way is up; each warning is given once per guess, on the
        first read of the axis that guess gave (at load, for a clip read
        from a file). :attr:`orientation_report` says which state below
        the axis is in and what a guess was read from.

        A clip's world up is in one of three states:

        - **Set**: an axis you assigned, or passed as ``world_up=`` at
          load, or the target of :meth:`reorient_world_up`. It is kept
          until you assign another, and every clip derived from this one
          keeps it set. :meth:`reorient_world_up` also turns the guessed
          axis with the frames, so clearing its axis gives the one the
          rotated frames show.
        - **Inferred**: guessed from this clip's own first frame, for a
          clip read from a file or built from arrays. It is guessed again
          when you assign :attr:`root_pos`, :attr:`joint_angles`,
          :attr:`joint_translations` or frames (``bvh[i] = other``), the one change that replaces the
          recording, and never by an edit pybvh makes. Setting
          :attr:`lr_mapping` guesses it again too: on a rig with no head,
          neck, chest or spine joint the guess falls back to
          :attr:`rest_up`, which is read from the legs of the mapping, so
          the clip gets the axis that mapping passed at load gives. A guess
          from the first frame comes out the same, and does not warn again.
          Setting :attr:`body_anchor` guesses it again as well, since the
          guess measures from the anchor: the clip gets the axis that
          ``body_anchor=`` passed at load gives.
        - **Inherited**: a clip pybvh derives from this one reports the
          axis guessed here, without guessing from its own frames:
          ``copy()``, slicing, ``a + b`` and ``a += b`` (from ``a``),
          :meth:`resample`, the transforms, :meth:`retarget`,
          :meth:`extract_joints` and a pickle. Cutting, resampling or
          transforming a clip does not change which way is up, though a
          slice whose first frame is a crouch or a fall would be guessed
          differently on its own. Assigning motion to the derived clip
          guesses again, as for an inferred axis; setting
          :attr:`lr_mapping` or :attr:`body_anchor` on it does not.

        :attr:`world_up_inferred` always guesses from this clip's own
        motion: where it differs from ``world_up`` and you set no axis,
        the axis was inherited.

        The detection specifics, since they decide which answer you get:
        "head" is the joint :attr:`rest_up` reads, the first in node
        order whose name holds ``head`` as a word, else ``neck``,
        ``chest`` or ``spine``, in any case and after a ``foo:``
        namespace or a ``.001`` suffix (``mixamorig:Head`` and ``Bip01
        Head`` match, ``Forehead`` does not; end sites are not read),
        skipping one that sits on :attr:`body_anchor` in the rest pose;
        and the frame-0
        reading is accepted only when its largest component is at least
        twice the second-largest — a first frame bent or leaning far
        from every axis is treated as ambiguous and defers to
        :attr:`rest_up`, with a warning when the body anchor's rotation
        at frame 0 carries :attr:`rest_up` onto another axis, since the
        rest pose may then be authored in another convention than the
        motion. A figure straight and horizontal at frame 0 with its
        head near a horizontal axis (lying flat, a plank) does decide,
        on that axis, with the disagreement warning when
        :attr:`rest_up` names another axis; set the axis when the guess
        is wrong. When the rest pose is
        degenerate too, the property warns and returns
        ``'+y'``. Use
        :attr:`world_up_inferred` to see what the heuristic picks
        while an override is in effect.

        Can be overridden manually via the setter when auto-detection
        produces the wrong answer (e.g. authored BVH files where the rest
        pose convention differs from the animation's world orientation):

            >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
            >>> bvh.world_up = '+y'

        Assign ``'auto'`` or ``None`` to clear a previous override and
        return to the guessed axis, inferred or inherited.

        Note: BVH files do not store a world-up field, so manual overrides
        are lost on write→read round trips and must be re-applied.
        """
        report = self.orientation_report
        if report.world_up is None:
            # No body, and nothing set or inherited: the guess refuses.
            return self._guess_world_up().axis
        derived = self._derived
        # A guess of this clip's own warns on the first read of the axis it
        # gave, once; an inherited guess warned on the clip it was made on.
        guessed_here = report.world_up_source != "set" and not report.world_up_inherited
        if guessed_here and derived.world_up_guess_announced != derived.inferred_world_up:
            guess = derived.inferred_world_up
            assert guess is not None  # the report read world up from it
            self._warn_about_world_up_guess(guess, report.rest_up)
            derived.world_up_guess_announced = guess
        return report.world_up

    @world_up.setter
    def world_up(self, value: str | None) -> None:
        if value is None or value == "auto":
            self._world_up_override = None
        else:
            self._world_up_override = _validate_axis_string(value)
        self._derived.world_up_changed()

    @property
    def world_up_inferred(self) -> str:
        """What the auto heuristic *would* pick, regardless of any override.

        Useful for auditing whether a manual ``bvh.world_up = '+x'``
        override was necessary, or for diagnosing skeletons whose
        animation and rest-pose conventions disagree. Always runs the
        inference fresh on this clip's own motion, and neither reads nor
        changes the axis :attr:`world_up` reports, so on a clip that
        inherited its world up it shows what a guess of its own would
        have been. Warns on a disagreement as :attr:`world_up` does, and
        is silenced the same way (``warn_on_world_up_disagreement=False``
        at load).

        Compare against :attr:`world_up` to see whether an override, or
        an inherited axis, is in effect:

            >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
            >>> bvh.world_up = '+y'    # a user override
            >>> bvh.world_up_inferred  # auto's guess
            '+z'
            >>> bvh.world_up
            '+y'
        """
        guess = self._guess_world_up()
        self._warn_about_world_up_guess(guess, self.rest_up)
        return guess.axis

    def _guess_world_up(self) -> _WorldUpGuess:
        """Guess world up from this clip's own first frame and rest pose."""
        anchor = self._body_anchor_index()
        frame0 = self.node_positions(frame=0) if self.frame_count > 0 else None
        return _infer_world_up(self._rest_skeleton(), anchor, frame0)

    def _rest_skeleton(self) -> _RestSkeleton:
        """The skeleton in its rest pose, as the orientation module reads it."""
        topology = self.fk_topology
        return _RestSkeleton(
            self._node_names(),
            self._rest_pose(),
            topology.parent_idx,
            topology.joint_idx < 0,
            self._mapping_lr_pairs(),
            self.root.is_static(),
        )

    def _warn_about_world_up_guess(self, guess: _WorldUpGuess, rest_up: str | None) -> None:
        """Warn when *guess* is the house default; and, unless the clip was
        loaded with ``warn_on_world_up_disagreement=False``, when it
        disagrees with *rest_up*, or when it fell back on rest up and the
        body anchor's world rotation at frame 0 carries rest up onto
        another axis."""
        if guess.source == "default":
            _warn_world_up_defaulted()
        elif not self._warn_on_world_up_disagreement:
            return
        elif _world_up_disagrees(guess, rest_up):
            assert rest_up is not None  # a disagreement needs a rest up
            _warn_world_up_disagreement(rest_up, guess.axis)
        else:
            anchor_rotation = self._body_anchor_rotmats([0])[0] if self.frame_count > 0 else None
            carried = _anchor_contradicts_rest_up(guess, anchor_rotation)
            if carried is not None:
                _warn_rest_up_contradicted_by_anchor(guess.axis, carried)

    @property
    def world_up_axis(self) -> Axis:
        """:attr:`world_up` parsed into numeric form — ``Axis(index, sign, vector)``.

        The three fields are the machine-usable views of the same signed axis string (always derived from the resolved :attr:`world_up`, so manual overrides are respected):

        - ``index`` (int): the up coordinate's column (0 = x, 1 = y, 2 = z) in any ``(..., 3)`` position array.
        - ``sign`` (float): ``+1.0`` or ``-1.0`` — multiply the raw coordinate by it to get an up-positive height.
        - ``vector`` (ndarray, shape ``(3,)``): the unit vector along the up direction, sign included (e.g. ``[0, -1, 0]`` for ``'-y'``). A fresh array on every access — mutating it never corrupts the Bvh.

        Example — up-positive heights of every node in every frame:

            >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
            >>> coords = bvh.node_positions()
            >>> heights = coords[:, :, bvh.world_up_axis.index] * bvh.world_up_axis.sign

        See Also
        --------
        world_up : The signed axis string this is parsed from (settable).
        rest_forward_axis, rest_up_axis : The other two parsed axis properties.
        floor_height : The scene ground level along this axis, in raw (unsigned) coordinates.
        """
        return parse_axis(self.world_up)

    @property
    def rest_forward_axis(self) -> Axis:
        """:attr:`rest_forward` parsed into numeric form — ``Axis(index, sign, vector)``.

        Never ``None``: :attr:`rest_forward` always resolves, because it falls back to an arbitrary-but-stable axis perpendicular to :attr:`rest_up` when the skeleton carries no usable L/R geometry. That fallback is indistinguishable here from a measured result; :attr:`orientation_report` (``rest_forward_source``) and :attr:`has_lr_geometry` tell them apart.

        See Also
        --------
        rest_forward : The signed axis string this is parsed from.
        world_up_axis, rest_up_axis : The other two parsed axis properties.
        """
        return parse_axis(self.rest_forward)

    @property
    def rest_up_axis(self) -> Axis | None:
        """:attr:`rest_up` parsed into numeric form — ``Axis(index, sign, vector)``, or ``None``.

        ``None`` exactly when :attr:`rest_up` is ``None`` — a degenerate rest pose (single-node skeleton, or all joints coincident) that carries no directional information. Each parsed axis property mirrors the nullability of the string it parses, so ``bvh.rest_up is None`` and ``bvh.rest_up_axis is None`` always agree.

        Note this is the skeleton's own up axis, read from the rest pose alone. For the world vertical — animation-derived, and always defined — use :attr:`world_up_axis`.

        See Also
        --------
        rest_up : The signed axis string this is parsed from.
        world_up_axis, rest_forward_axis : The other two parsed axis properties.
        """
        rest_up = self.rest_up
        return None if rest_up is None else parse_axis(rest_up)

    @property
    def floor_height(self) -> float:
        """Scene ground height, in raw world coordinates along ``world_up``.

        A single scalar: where this clip's ground is, in the BVH's own
        coordinate system, signed along the raw up axis (so for
        ``world_up='-y'`` a floor at raw ``y≈5`` returns ``≈5``). This is the
        height to draw a ground plane at.

        Convention: the 2nd percentile of the per-frame minimum height over
        **all nodes**, end sites included. Two choices are being made there,
        and both have alternatives:

        - **All nodes, not the feet.** The ground is where the clip's geometry
          bottoms out, whichever body part gets there — a hand in floor work, a
          hip in a fall, a toe *tip* in ordinary walking. Estimating from
          auto-detected feet instead (what this property did before v0.9.0)
          reads systematically *high*: a toe end site hangs below the toe joint
          on most rigs, and over 406 sampled clips the feet-only estimate sat a
          median 1% of the clip span (max 6.6%) above this one, never below.
        - **2nd percentile, not the true minimum.** Robust to occasional
          glitched-low frames, at the cost of letting a genuinely low frame dip
          slightly below the returned level. For the exact minimum use
          ``coords[..., up].min()``, or ``foot_contacts(floor="min")`` for that
          function's own reference; this property stays 2nd-percentile.

        A *persistently* misplaced node — one that sits below the ground in most
        frames, not just a few — will drag this down; the percentile absorbs
        transient glitches, not systematic ones. Contact detection does not
        move with it: the detectors measure against a reference of their own.

        This is **not** the reference :func:`~pybvh.analysis.foot_contacts`
        compares foot clearance against. That one is estimated per call, over
        the nodes being tested, and is reported as the ``contact_reference``
        of its :class:`~pybvh.ContactReport` (``return_info=True``). The two
        answer different questions — this one belongs to the clip, that one to
        the query — and they differ by roughly a toe length on any rig with toe
        end sites. Passing this value in as ``foot_contacts(floor=...)`` will
        make every foot read as airborne.

        Computed on first read and kept until the skeleton, the motion or
        :attr:`world_up` changes; assigning :attr:`world_up` measures it
        again along the new axis. The contact detectors never compute or
        read it.
        """
        floor = self._derived.floor_height
        if floor is None:
            from . import analysis

            floor = analysis._compute_floor_height(self)
            self._derived.floor_height = floor
        return floor

    @property
    def orientation_report(self) -> OrientationReport:
        """Each orientation fact of the clip, and how it was obtained.

        Whether :attr:`rest_up`, :attr:`rest_forward` and
        :attr:`world_up` were set, measured or defaulted, and how, with
        the rest pose's left and the source of the L/R mapping; see
        :class:`~pybvh.OrientationReport` for each field. Those members
        always answer, a default included, so their return value alone
        cannot say which it was: this can.

        Read without a warning: a warning about a default fires when the
        defaulted fact itself is read (:attr:`world_up`, once per guess;
        :attr:`rest_forward`; the facing), and each points here. Computed on first read and
        kept until the skeleton changes, :attr:`world_up` or
        :attr:`lr_mapping` or :attr:`body_anchor` is assigned, or the
        motion is assigned
        (:attr:`root_pos`, :attr:`joint_angles`, ``bvh[i] = other``),
        which guesses world up again.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
        >>> report = bvh.orientation_report
        >>> if report.rest_forward_source == "default":
        ...     ...  # no usable L/R geometry: rest_forward is a house default
        >>> report.world_up, report.world_up_source
        ('+z', 'first_frame')
        """
        report = self._derived.orientation_report
        if report is None:
            report = self._read_orientation()
            self._derived.orientation_report = report
        return report

    def _read_orientation(self) -> OrientationReport:
        """Gather the report's inputs and build it."""
        derived = self._derived
        world_up_inherited = False
        world_up: str | _WorldUpGuess | None
        if self._world_up_override is not None:
            world_up = self._world_up_override
        elif derived.inherited_world_up is not None:
            world_up = derived.inherited_world_up
            world_up_inherited = True
        elif not self._has_body():
            # The guess is body-level: it refuses a clip with no body, and
            # the report does not.
            world_up = None
        else:
            # Guessed silently: the guess's warnings are world_up's to give
            # when the axis is read, not this report's.
            if derived.inferred_world_up is None:
                derived.inferred_world_up = self._guess_world_up()
            world_up = derived.inferred_world_up
        return _build_orientation_report(
            self._rest_skeleton(),
            world_up,
            world_up_inherited,
            self._lr_mapping_source(),
            self._found_body_anchor(),
            self._nodes,
        )

    def _lr_mapping_source(self) -> _LrMappingSource | None:
        """Where the L/R mapping in effect comes from, ``None`` without one."""
        setting = self._lr_mapping_setting
        if isinstance(setting, dict):
            return "set"
        return "names" if setting == "names" and self._lr_mapping is not None else None

    @property
    def body_anchor(self) -> BvhNode | None:
        """The joint the body's measures start from, a node of the clip,
        or ``None`` when the clip has no body.

        The joint you set (see *Setting it* below), else the one a rule
        finds from the hierarchy and the L/R pairs of
        :attr:`lr_mapping`: each pair meets at the lowest joint that both
        its joints' chains to the root pass through, and the anchor is
        the meeting joint closest to the root. The legs meet at the hips,
        the arms lower, at the chest, so on most rigs the anchor is the
        hips, also when the root is a reference node above them or has a
        helper beside them. Of several meeting joints at the same depth,
        the one with the most joints below it, end sites not counted,
        when each other one has fewer than half as many. Otherwise the
        clip has no body, for instance two figures under one root, and
        the anchor is ``None``; reading it does not raise. Under a static
        root (``CHANNELS 0``), each child of the root heads a tree, and
        the trees are compared first, by the same half rule: a second
        tree with at least half the joints of the largest gives no body,
        and otherwise only the pairs inside the largest tree are read. Without a pair, the root. The rest positions and the motion are not read.
        :attr:`orientation_report` says how it was found, in
        ``body_anchor_source``, and which joints were compared, in
        ``body_anchor_candidates``.

        The rule's anchor is found on first read and kept until the
        skeleton changes or :attr:`lr_mapping` is assigned. A clip derived
        from this one finds its own.

        **Setting it.** Where you know better than the rule, assign a
        joint: its name or the node itself, one of :attr:`nodes`, resolved
        as :meth:`index` resolves one with ``space='joint'``. Pass
        ``body_anchor=`` to :func:`~pybvh.read_bvh_file` to set it at
        load, by name. The joint you set is the anchor, and
        :attr:`orientation_report` gives the source ``"set"`` and no
        candidate. Assign ``None`` to return to the rule, as
        ``world_up = None`` returns to the guess. Setting it, or returning
        to the rule, clears the values computed from the anchor (the
        report, the world up this clip guessed and the floor height),
        which are computed again on first read; a world up set or
        inherited stays (see :attr:`world_up`).

        A set anchor is one of your settings, as a set :attr:`lr_mapping`
        is (ADR 0003). Assigning the motion or :attr:`lr_mapping`, and
        pybvh's own edits of the skeleton (:meth:`scale`,
        :meth:`reorient_rest_up`, ...), keep it. Every clip derived from
        this one carries it: a slice, :meth:`copy`, a concatenation,
        :meth:`resample`, a transformed clip, and a pickle. ``copy(nodes=)``
        carries it to the joint at the same position, under its new name;
        a copy that changes the joint count, which only a clip without
        motion allows, drops it. :meth:`extract_joints`
        carries it when the joint is kept. A derived clip that no longer
        holds the joint drops the setting and uses the rule.

        **A clip with no body.** When the anchor is ``None``, every
        body-level operation raises ``ValueError``, naming the candidates
        and the way out, ``bvh.body_anchor = ...`` (the node,
        ``bvh.nodes[i]``, when the candidates share a name with another
        joint): the world-up guess,
        :attr:`rest_up`, :attr:`rest_forward` and
        :attr:`has_lr_geometry`, the facing (:meth:`forward_at`,
        :meth:`left_at`, :meth:`facing_frame`), :meth:`root_trajectory`,
        the ``centered="anchor"``, ``"first_ground"`` and
        ``"anchor_ground"`` modes,
        :meth:`auto_detect_feet`, :meth:`foot_contacts`,
        :meth:`ground_contacts`, :meth:`skeleton_size`, the gait
        descriptors and :meth:`mirror`. The operations that treat all nodes alike work as on any clip:
        reading, writing, forward kinematics, the derivatives, slicing,
        concatenation, :meth:`resample`, the Euler orders,
        :meth:`scale`, :meth:`extract_joints` and the bounding volumes.
        The scene ground (:attr:`floor_height`) and drawing read world
        up, whose guess is body-level: they work once :attr:`world_up`
        is set or the anchor named. Reading the anchor,
        :attr:`orientation_report` and :attr:`lr_mapping` never raises.

        Raises
        ------
        TypeError
            On assignment, if the value is neither a name, a ``BvhNode``
            nor ``None``. An integer is refused: node ``i`` is
            ``bvh.nodes[i]``.
        ValueError
            On assignment, if the node is not one of :attr:`nodes`, if no
            joint carries the name or several do, or if it selects an end
            site.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/synthetic/reference_humanoid_6ch.bvh')
        >>> bvh.root.name, bvh.body_anchor.name
        ('Reference', 'Hips')
        >>> bvh.body_anchor = 'Chest'
        >>> bvh.orientation_report.body_anchor_source
        'set'
        >>> bvh.body_anchor = None
        >>> bvh.body_anchor.name, bvh.orientation_report.body_anchor_source
        ('Hips', 'lr_pairs')
        """
        node = self._found_body_anchor().node
        return None if node is None else self._nodes[node]

    @body_anchor.setter
    def body_anchor(self, value: NodeSelector | None) -> None:
        if value is None:
            self._set_body_anchor_setting(None)
            return
        joint = self.index(value, space="joint")
        self._set_body_anchor_setting(self._non_end_site_indices()[joint])

    def _set_body_anchor_setting(self, setting: int | None) -> None:
        """Store the user's body anchor, a position in ``nodes`` or
        ``None``, and report the change."""
        self._body_anchor_setting = setting
        self._derived.body_anchor_changed()

    def _found_body_anchor(self) -> _BodyAnchor:
        """The body anchor in effect: the one the user set, else the one
        the rule finds, kept in the derived state."""
        if self._body_anchor_setting is not None:
            return _BodyAnchor(self._body_anchor_setting, "set", ())
        found = self._derived.body_anchor
        if found is None:
            found = _find_body_anchor(self._rest_skeleton())
            self._derived.body_anchor = found
        return found

    def _in_body(self) -> npt.NDArray[np.bool_]:
        """Which nodes are in the body, in node order: the body anchor and
        every node below it, started higher when a head or a foot name is
        missing below it (:func:`~pybvh._orientation._find_body`). Raises
        ``ValueError`` on a clip with no body, as
        :meth:`_body_anchor_index` does."""
        return _find_body(self._rest_skeleton(), self._body_anchor_index())

    def _size_reference(self, feet: Sequence[int]) -> int:
        """The node index the sizes of *feet* (node indices) are read from.

        The lowest joint that has the feet in two or more of its
        branches, else the body anchor, reached through
        :meth:`_body_anchor_index`
        (:func:`~pybvh._orientation._size_reference`). Read by
        ``skeleton_size`` and by the "feet below hips" check and the
        threshold scale of ``foot_contacts``.
        """
        return _size_reference(self.fk_topology.parent_idx, feet, self._body_anchor_index())

    @property
    def rest_up(self) -> str | None:
        """The skeleton's own up axis, read from the rest pose alone.

        Read-only. The main axis from :attr:`body_anchor` to the first
        joint of the body, in node order, whose name holds ``head`` as a
        word, else ``neck``, ``chest`` or ``spine``, in any case and
        after a ``foo:`` namespace or a ``.001`` suffix: ``Head``,
        ``mixamorig:Head``, ``Bip01 Head`` and ``CC_Base_Head`` are
        heads, ``Forehead`` is not. A joint that sits on the anchor is
        skipped, and end sites, whose names are generated, are not read.
        The body is the anchor and every node below it; when no joint
        below the anchor has one of these names, it starts at the
        closest joint above the anchor that has one below it, so a prop
        or a helper beside the body is never the head joint.
        Without such a joint, the axis opposite the legs: the first pair
        of :attr:`lr_pairs` that meets at the anchor, split across two of
        its branches, read from the anchor to the far ends of those
        branches. Without such a pair either, the axis along which the
        rest pose spreads most, a weak reading: a T-pose wider than tall
        spreads most along its arms. :attr:`orientation_report` says
        which, in ``rest_up_source``. The animation is never read.
        :attr:`world_up`'s guess reads the same joint.

        Contrast with :attr:`world_up`, which is *animation*-derived
        (inferred from the first frame's anchor-to-head direction). On
        clean files the two agree; when they disagree, the BVH was
        authored with the rest pose in one convention and animated in
        another, and :meth:`reorient_rest_up` can fix it in place.

        Returns
        -------
        str or None
            Signed axis string (e.g. ``'+y'``, ``'+z'``), or ``None``
            when the rest pose is degenerate (single-node skeletons or
            all-zero offsets) and carries no directional information.
        """
        return self._body_orientation().rest_up

    @property
    def rest_forward(self) -> str:
        """The direction the rest pose faces, in the skeleton's own frame.

        Read-only. ``rest left × rest up``, where rest left is the rest
        pose's L/R geometry (the pairs of :attr:`lr_mapping` whose two
        joints are in the body, as :attr:`rest_up` reads it) measured
        against :attr:`rest_up`. Read from the rest pose alone, like
        :attr:`rest_up`: never from :attr:`world_up` or the animation.
        The rest pose is the skeleton with every rotation zero, so this
        is a direction in the skeleton's frame, not in the world; the
        world rotation of :attr:`body_anchor` at a frame carries it into
        the world, which is
        what the heading of :func:`~pybvh.analysis.root_trajectory`
        reads. On a clean file, where rest up is world up, it is also
        what :meth:`forward_at` measures on the rest pose's own
        positions. On a file whose rest pose is authored in another
        convention than its motion it may lie along world up:
        ``bvh_test3.bvh`` (world up ``'+z'``, rest up ``'+y'``) faces
        ``'+z'``.

        Without usable L/R geometry (no pairs, or pairs that lie along
        rest up) it is a house default for rest up, arbitrary but
        stable, and reading it warns; :attr:`orientation_report`
        (``rest_forward_source``) and :attr:`has_lr_geometry` tell the
        two apart.

        Use this to check whether a skeleton's rest-pose facing matches
        a dataset convention without having to call
        :func:`reorient_rest_forward` and compare results.

        Returns
        -------
        str
            Signed axis string (e.g. ``'+z'``, ``'-x'``), perpendicular to
            :attr:`rest_up`.
        """
        report = self._body_orientation()
        assert report.rest_forward is not None  # a clip with a body has one
        if report.rest_forward_source == "default":
            _warn_rest_forward_defaulted(report.rest_forward, report.rest_up)
        return report.rest_forward

    def _rest_left_of(self, mapping: dict[str, str] | None) -> str | None:
        """The rest pose's left for the pairs of *mapping*, against rest up.

        ``None`` without a usable one. :func:`~pybvh.transforms.mirror`
        reads it for the mapping it mirrors with, which may be another
        than the clip's own; :attr:`orientation_report` holds the rest
        left of the clip's own mapping. Rest up is read with the same
        pairs, as it would be with *mapping* set: on a rig without a
        named up joint it comes from them, and from the body anchor those
        pairs meet at.
        """
        pairs = _resolve_lr_pairs(mapping, self.node_index)
        skeleton = self._rest_skeleton()._replace(
            lr_pairs=np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        )
        anchor = self._body_anchor_index(
            None if self._body_anchor_setting is not None else _find_body_anchor(skeleton)
        )
        _rest_up, rest_left = _infer_rest_up_and_left(skeleton, anchor)
        return rest_left

    @property
    def lr_mapping(self) -> dict[str, str] | None:
        """Left/right joint pair mapping for this skeleton (bidirectional).

        A dict describing the skeleton's bilateral symmetry pairs.
        ``None`` if no pairs could be auto-detected and no explicit
        mapping was provided.

        The dict is **symmetric**: both directions of each pair are
        present, so ``mapping['LeftArm'] == 'RightArm'`` AND
        ``mapping['RightArm'] == 'LeftArm'``. Useful for mirroring-based
        data augmentation, where a lookup can come from either side.

        Detection, run on first use, pairs two joints whose names differ
        only by their side, in one of these conventions, tried in this
        order:

        1. a delimited suffix: ``arm.L``, ``arm_r``, ``arm.Left``,
           ``arm_right``;
        2. the word ``Left`` / ``Right`` anywhere, in any case:
           ``LeftArm``, ``leftArm``, ``LeftEye``;
        3. ``L`` / ``R`` before an uppercase letter: ``LArm``;
        4. ``l``, ``r``, ``left`` or ``right`` in any case, as a word of
           its own between ``.``, ``_`` or spaces: ``b_l_upleg``,
           ``L_legUpper0001_bind_JNT``, ``Bip01 L Thigh``;
        5. ``左`` / ``右`` as the first character: ``左足``;
        6. ``l`` / ``r`` before an uppercase letter: ``lThigh``;
        7. ``l`` / ``r`` with no delimiter: ``lfemur``.

        A name is paired by the first convention that pairs it, and only
        when its partner is also a joint: a joint with no partner stays
        unpaired. A Mixamo ``mixamorig:`` namespace and a numbered
        ``.001`` duplicate suffix are set aside before the match. Two
        conventions are read narrowly, so that a name that only looks like
        a side is not paired: ``左`` / ``右`` count as the first character
        only, and the last convention counts only when it gives two pairs
        or more on the skeleton, since one pair such as ``light`` /
        ``right`` can differ by its first letter by chance. Skeletons with
        conventions the heuristic can't parse have ``lr_mapping = None`` —
        in that case, set it explicitly:

            >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
            >>> bvh.lr_mapping = {'LeftArm': 'RightArm', 'LeftUpLeg': 'RightUpLeg'}

        The assigned dict is one-directional; pybvh symmetrizes it
        internally. Either form works on assignment.

        or pass ``lr_mapping=`` at load time:

            >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh', lr_mapping={'LeftArm': 'RightArm'})

        Consumers: ``mirror()``, ``orientation_report`` (and so
        ``rest_forward`` and ``has_lr_geometry``),
        ``forward_at()``, ``left_at()``, ``facing_frame()``. On a rig with
        no head, neck, chest or spine joint, ``rest_up`` is read from the
        legs of the mapping, and so is the world up guessed from it: setting
        the mapping guesses the clip's own :attr:`world_up` again, and its
        :attr:`floor_height` with it, as passing it at load would have. A
        world up set, or inherited from the clip this one was derived from,
        stays (see :attr:`world_up`).

        Note: BVH files don't store L/R pair info, so user-set mappings
        are lost on ``bvh.write()`` round-trips — same wart as
        ``world_up``. Re-apply after reading.
        """
        if self._lr_mapping is None:
            return None
        symmetric: dict[str, str] = {}
        for left, right in self._lr_mapping.items():
            symmetric[left] = right
            symmetric[right] = left
        return symmetric

    @lr_mapping.setter
    def lr_mapping(self, value: dict[str, str] | None) -> None:
        if value is None:
            self._set_lr_mapping_setting(None)
            return
        self._validate_and_set_lr_mapping(value)

    def _set_lr_mapping_setting(self, setting: dict[str, str] | Literal["names"] | None) -> None:
        """Make *setting* the user's L/R choice, and report the change."""
        self._lr_mapping_setting = setting
        self._derived.lr_mapping_changed()

    @property
    def _lr_mapping(self) -> dict[str, str] | None:
        """The L/R mapping in effect, one direction (left to right), or None.

        The user's mapping when they set one; otherwise the pairs detected
        from the joint names, ``None`` when the names give none or the user
        cleared the mapping.
        """
        setting = self._lr_mapping_setting
        if setting != "names":
            return setting
        detected = self._derived.names_lr_mapping
        if detected is None:
            detected = _detect_lr_mapping_by_names(self.joint_names)
            self._derived.names_lr_mapping = detected
        return detected or None

    @property
    def has_lr_geometry(self) -> bool:
        """Whether the rest pose carries usable left/right direction.

        ``True`` when the skeleton's L/R joint pairs give a lateral axis
        in the rest pose that does not lie along :attr:`rest_up`, i.e.
        when :attr:`rest_forward` is **measured** from this skeleton
        rather than a house default: ``orientation_report.rest_forward_source
        == "lr_geometry"``. It is ``False`` exactly when reading
        :attr:`rest_forward` emits the fallback ``UserWarning``.
        Like :attr:`rest_forward`, it reads the rest pose alone, never
        :attr:`world_up`.

        This is the check :attr:`rest_forward` (and so
        :attr:`rest_forward_axis`) cannot express in its own return
        value: with no usable L/R geometry it returns an
        arbitrary-but-stable axis chosen from :attr:`rest_up` alone,
        which is indistinguishable from a measured result. When it
        matters whether a facing was derived from the data — comparing
        a skeleton against a dataset convention, say, where a fallback
        would "match" every time — check this first.

        Assigning :attr:`lr_mapping` explicitly is what fixes a ``False``
        on a skeleton whose joints simply are not named recognizably.

        This property describes the *rest pose*. The per-frame
        :meth:`forward_at` / :meth:`facing_frame` measure each frame's
        coordinates first and fall back only on frames where that
        fails, so on a skeleton with zero rest offsets but animated L/R
        separation they can still measure while this reports ``False``.

        Examples
        --------
        >>> bvh = Bvh.from_file('bvh_data/bvh_test1.bvh')
        >>> dataset_convention = '+y'  # the rest facing the dataset assumes
        >>> if bvh.has_lr_geometry:
        ...     assert bvh.rest_forward == dataset_convention
        ... else:
        ...     ...  # facing is a default, not a measurement

        See Also
        --------
        rest_forward : The axis whose fallback this reports.
        lr_mapping : The pairs the measurement is derived from.
        """
        return self._body_orientation().rest_forward_source == "lr_geometry"

    @property
    def lr_pairs(self) -> list[tuple[int, int]] | None:
        """Left/right joint pairs as index tuples in ``joint_angles`` space.

        Index-space counterpart of :attr:`lr_mapping`: one
        ``(left, right)`` tuple per pair, in mapping order, resolved as
        :func:`~pybvh.transforms.mirror` resolves its pairs. Returns
        ``None`` when no mapping is available (matches the
        ``lr_mapping`` sentinel — one "no pairs" convention across both
        surfaces); ``bvh.lr_pairs or []`` gives a list either way.

        Useful for graph construction and array-level ops that index
        joints by position rather than by name, such as
        :func:`~pybvh.transforms.mirror_angles`.
        """
        mapping = self._lr_mapping
        if mapping is None:
            return None
        return _resolve_lr_pairs(mapping, self.joint_index)

    @property
    def node_lr_pairs(self) -> list[tuple[int, int]] | None:
        """Left/right node pairs as index tuples in ``nodes`` index space.

        Node-space counterpart of :attr:`lr_pairs`, covering joints **and**
        their end sites — so a mirror over :meth:`node_positions` output
        can swap every paired vertex, including the fingertips, toe tips
        and head top that only exist in node space. ``None`` when no L/R
        mapping is available, the same sentinel as :attr:`lr_pairs` and
        :attr:`lr_mapping`.

        Order is deterministic: every joint pair in :attr:`lr_mapping`
        order first, then the end-site pairs in that same order, so
        metadata built from it reproduces across runs.

        Left and right joints are matched by name (that is what
        :attr:`lr_mapping` is), but looked up among joints only, so an
        end site whose generated display name collides with a joint's
        cannot shadow it. The end sites themselves are matched
        positionally within each joint and resolved by identity.

        A joint pair whose two sides carry **different numbers of end
        sites** has no well-defined tip correspondence. The pair itself is
        still returned and its end sites are dropped, matching
        :attr:`lr_pairs`' habit of filtering what it cannot resolve rather
        than raising from a property. :func:`~pybvh.transforms.mirror`
        raises on the same condition instead, because a half-swapped
        skeleton is a wrong answer rather than a partial one.

        Returns
        -------
        list of (int, int) or None

        See Also
        --------
        lr_pairs : The same pairing in ``joint_angles`` index space.
        joint_tips : Each joint's first end site, for tip trajectories.
        """
        if self._lr_mapping is None:
            return None
        pairs, _mismatches = self._node_lr_pairs_of(self._lr_mapping)
        return pairs

    def _node_lr_pairs_of(
        self, mapping: dict[str, str] | None
    ) -> tuple[list[tuple[int, int]], list[tuple[str, int, str, int]]]:
        """*mapping* resolved in ``nodes`` index space, and its end-site mismatches.

        What :attr:`node_lr_pairs` and :func:`~pybvh.transforms.mirror`
        read, each with its own policy on a mismatch.
        """
        topology = self.fk_topology
        return _resolve_node_lr_pairs(
            mapping,
            self._node_names(),
            topology.parent_idx,
            topology.joint_idx < 0,
        )

    def _mapping_lr_pairs(self) -> npt.NDArray[np.intp]:
        """Every joint pair of the L/R mapping in effect, in ``nodes`` index space.

        Joint pairs only, as :attr:`lr_mapping` names them, resolved into
        ``nodes`` index space so they index :meth:`node_positions` output.
        End sites are not included: that is :attr:`node_lr_pairs`' job,
        for mirroring positions, not for reading the facing. An ``(P, 2)``
        int array, ``(0, 2)`` without an L/R mapping, so array code needs
        no ``None`` branch. The rest skeleton holds them, for the body
        anchor rule and for rest up and rest left, which keep the body's.
        """
        pairs = _resolve_lr_pairs(self._lr_mapping, self.node_index)
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)

    def _facing_lr_pairs(self) -> npt.NDArray[np.intp]:
        """The L/R pairs the facing reads: those of :meth:`_mapping_lr_pairs`
        with both joints in the body (:meth:`_in_body`), in their order.

        A second figure or a prop beside the body is not averaged into the
        body's facing. On a clip whose body holds every pair, they are
        :meth:`_mapping_lr_pairs`. bvhplot stores them on its
        ``SkeletonView``, so the follow camera needs no ``Bvh``. Raises
        ``ValueError`` on a clip with no body, as :meth:`_in_body` does.
        """
        return _lr_pairs_in_body(self._mapping_lr_pairs(), self._in_body())

    def _validate_and_set_lr_mapping(self, mapping: dict[str, str]) -> None:
        """Validate the user's lr_mapping dict and make it the mapping in effect."""
        self._set_lr_mapping_setting(_validated_lr_mapping(mapping, self.joint_names))

    def forward_at(self, frame: int = 0) -> str:
        """Character's world-space forward (facing) direction at a given frame.

        Computed from actual joint positions at the given frame — the
        leftward axis is derived by averaging (left − right) across
        the L/R joint pairs of the body in world space, then crossed with
        ``world_up`` to produce the forward direction (forward =
        leftward × up). This tracks the character's actual facing as
        they rotate through the animation.

        This is the snapped classification — the continuous facing
        vector is quantized to the nearest of the six signed world
        axes. See :meth:`facing_frame` for the continuous per-frame
        basis as unit vectors, and for the fallback when the frame's
        left/right axis cannot be read (the body's heading at that
        frame, then a house default, which warns).

        Parameters
        ----------
        frame : int, optional
            Frame index (default 0). Must be within the animation range.

        Returns
        -------
        str
            Signed axis string (e.g. ``'-z'``) pointing in the character's
            facing direction in world coordinates at the given frame.

        See Also
        --------
        facing_frame : The continuous per-frame basis (vectors, all
            frames at once).
        left_at : Leftward direction.
        """
        return self._facing_labels_at(frame)[0]

    def left_at(self, frame: int = 0) -> str:
        """Character's world-space leftward direction at a given frame.

        Returns the signed axis along which a positive step moves from
        the character's right side toward their left side (e.g.
        right-shoulder → left-shoulder direction). Follows the
        right-hand-rule convention ``leftward = world_up × forward`` so
        the triple (``world_up``, :meth:`forward_at`, ``left_at``) forms
        a consistent orthonormal frame in every axis convention pybvh
        supports.

        Computed from joint positions at the given frame, so it tracks
        hip twist and shoulder rotation as the character moves.

        This is the snapped classification — see :meth:`facing_frame`
        for the continuous per-frame basis as unit vectors, and its
        fallback. The label is ``world_up × forward_at(frame)``, so it is
        never world up itself, even on a frame whose left/right axis is
        vertical.

        Parameters
        ----------
        frame : int, optional
            Frame index (default 0). Must be within the animation range.

        Returns
        -------
        str
            Signed axis string (e.g. ``'-x'``) pointing toward the
            character's left in world coordinates at the given frame.

        See Also
        --------
        forward_at : Facing direction.
        facing_frame : The continuous per-frame basis (vectors, all
            frames at once).
        world_up : World vertical axis.
        """
        return self._facing_labels_at(frame)[1]

    def _facing_labels_at(self, frame: int) -> tuple[str, str]:
        """The snapped ``(forward, left)`` labels at clip frame *frame*."""
        frame_coords = self.node_positions(frame=frame)
        frame = range(self.frame_count)[frame]  # as the warning names it
        forward, _, _, _ = self._facing_at(frame_coords[np.newaxis], [frame])
        return _facing_labels(forward[0], self.world_up)

    def _facing_at(
        self,
        coords: npt.NDArray[np.float64],
        frames: Sequence[int] | slice | None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.bool_],
    ]:
        """The facing chain on *coords*, whose rows are the clip's *frames*.

        Gathers the chain's inputs (the facing L/R pairs, world up and,
        when rest forward is measured, the body anchor's world rotation at
        *frames*, for the heading),
        runs it, and warns when a row carries the house default. With
        *frames* ``None`` the rows are a pose with no clip frame, and the
        heading stage is skipped. Returns ``(forward, left, up, valid)``.
        """
        world_up = self.world_up
        report = self._body_orientation()
        lr_pairs = self._facing_lr_pairs()
        anchor_rotmats = rest_forward = None
        if frames is not None and lr_pairs.shape[0] > 0:
            if report.rest_forward_source == "lr_geometry":
                rest_forward = report.rest_forward
                anchor_rotmats = self._body_anchor_rotmats(frames)
        forward, left, up, valid, defaulted = _facing_basis(
            coords, lr_pairs, world_up, anchor_rotmats, rest_forward
        )
        if defaulted.any():
            if lr_pairs.shape[0] == 0:
                _warn_facing_defaulted_no_pairs(world_up)
            elif frames is None:
                _warn_facing_defaulted_on_pose(world_up)
            else:
                named = np.arange(self.frame_count)[frames][defaulted].tolist()
                _warn_facing_defaulted_at_frames(world_up, named)
        return forward, left, up, valid

    def _root_rotmats(self, frames: Sequence[int] | slice = slice(None)) -> npt.NDArray[np.float64]:
        """The root's rotation matrix at each of *frames*, shape ``(F, 3, 3)``.

        The world rotation of the body anchor on a clip whose anchor is
        its root (:meth:`_body_anchor_rotmats`).
        """
        joint_angles = self._motion()["joint_angles"]
        # The per-joint form, which reads a static root's order '' as the identity.
        return rotations.euler_to_rotmat(joint_angles[frames, :1], [_euler_order(self.root)])[:, 0]

    def _body_anchor_index(self, anchor: _BodyAnchor | None = None) -> int:
        """The node index of :attr:`body_anchor`, for the readers that describe the body.

        The one way the body-level readers reach the anchor and the body:
        the world-up guess, rest up, rest left and rest forward (through
        :meth:`_body_orientation`), the body (:meth:`_in_body`) and so the
        feet, the sizes (:meth:`_size_reference`), the scale of
        :meth:`ground_contacts`, the travel
        (:meth:`_body_anchor_positions`), the heading and the facing
        (:meth:`_body_anchor_rotmats`) and the ``centered`` modes. Raises
        ``ValueError`` on a clip with no body, naming the candidates and
        ``bvh.body_anchor = ...``, so no reader has a check of its own.

        *anchor* is the one to read in place of the clip's
        (:meth:`_found_body_anchor`): ``mirror`` reads rest left for a
        mapping that may not be the clip's.
        """
        if anchor is None:
            anchor = self._found_body_anchor()
        if anchor.node is None:
            names = [self._nodes[i].name for i in anchor.candidates]
            listed = ", ".join(repr(name) for name in names[:-1]) + f" and {names[-1]!r}"
            joint_names = [self._nodes[i].name for i in self._non_end_site_indices()]
            if all(joint_names.count(name) == 1 for name in names):
                way_out = (
                    f"`bvh.body_anchor = {names[0]!r}`, or pass `body_anchor=` to read_bvh_file"
                )
            else:
                # A name several joints carry selects none of them, and
                # body_anchor= at load takes a name; the node, by its
                # position, selects the candidate whatever mapping found it.
                way_out = f"`bvh.body_anchor = bvh.nodes[{anchor.candidates[0]}]`"
            raise ValueError(
                f"This clip has no body: its hierarchy holds several figures of "
                f"comparable size, whose body anchors would be {listed}. Name the "
                f"one to describe, for example {way_out}."
            )
        return anchor.node

    def _has_body(self) -> bool:
        """Whether the clip has a body: :attr:`body_anchor` is not ``None``.

        For the operations that leave a body-level part out on a clip with
        no body, where the readers raise: the world-up guess at load.
        """
        return self._found_body_anchor().node is not None

    def _body_orientation(self) -> OrientationReport:
        """:attr:`orientation_report`, for a reader of the body's facts.

        Rest up, rest left and rest forward are read from the body; the
        report holds ``None`` for them on a clip with no body, where this
        raises through :meth:`_body_anchor_index`.
        """
        self._body_anchor_index()
        return self.orientation_report

    def _body_anchor_positions(self) -> npt.NDArray[np.float64]:
        """The body anchor's world position per frame, shape ``(F, 3)``, read-only.

        The body's travel. On a clip whose anchor is its root it is
        :attr:`root_pos` itself, with no forward kinematics run.
        """
        anchor = self._body_anchor_index()
        if anchor == 0:
            return self.root_pos
        return self._world_node_positions()[:, anchor]

    def _body_anchor_rotmats(
        self, frames: Sequence[int] | slice = slice(None)
    ) -> npt.NDArray[np.float64]:
        """The body anchor's world rotation at each of *frames*, shape ``(F, 3, 3)``.

        What carries a direction of the skeleton's own frame into the
        world at a frame, for the body (ADR 0004): the heading and the
        facing chain's heading stage read it. The rotations of the chain
        from the root to the anchor, composed parent first as forward
        kinematics composes them. On a clip whose anchor is its root it
        is :meth:`_root_rotmats`.
        """
        anchor = self._body_anchor_index()
        if anchor == 0:
            return self._root_rotmats(frames)
        topology = self.fk_topology
        chain = []
        node = anchor
        while node != -1:
            chain.append(int(topology.joint_idx[node]))
            node = int(topology.parent_idx[node])
        chain.reverse()  # root first
        joint_angles = self._motion()["joint_angles"][frames][:, chain]
        local = rotations.euler_to_rotmat(joint_angles, [topology.euler_orders[j] for j in chain])
        world = local[:, 0]
        for k in range(1, len(chain)):
            world = world @ local[:, k]
        return world

    def facing_frame(self):
        """Per-frame facing basis as continuous unit vectors.

        Returns a ``FacingFrame(forward, left, up, valid)`` named tuple:
        three ``(F, 3)`` arrays — the yaw-only, gravity-aligned
        orthonormal basis that :meth:`forward_at` / :meth:`left_at`
        snap to axis labels — plus a ``(F,)`` bool array, ``False`` on
        frames whose basis is the constant fallback rather than a
        measurement. See :func:`pybvh.analysis.facing_frame` for the
        full construction, conventions, and fallback policy.
        """
        from . import analysis

        return analysis.facing_frame(self)

    def write(self, filepath: str | Path, verbose: bool = False, overwrite: bool = True) -> None:
        """Write the Bvh object to a ``.bvh`` file.

        Pass ``overwrite=False`` to raise ``FileExistsError`` rather than
        replace an existing file.  See :func:`pybvh.io.write_bvh_file`."""
        from . import io

        io.write_bvh_file(self, filepath, verbose=verbose, overwrite=overwrite)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        world_up: str = "auto",
        warn_on_world_up_disagreement: bool = True,
        lr_mapping: dict[str, str] | None = None,
        body_anchor: str | None = None,
    ) -> Bvh:
        """Read a Bvh from a ``.bvh`` file — the constructor counterpart of :meth:`write`.

        Delegates to :func:`pybvh.io.read_bvh_file`; see it for parameter
        details.
        """
        from . import io

        return io.read_bvh_file(
            filepath,
            world_up=world_up,
            warn_on_world_up_disagreement=warn_on_world_up_disagreement,
            lr_mapping=lr_mapping,
            body_anchor=body_anchor,
        )

    @classmethod
    def from_df(cls, hier: Sequence[BvhNode] | Sequence[Mapping[str, Any]], df) -> Bvh:
        """Build a Bvh from a skeleton and a motion DataFrame.

        The constructor counterpart of :meth:`to_node_table` and
        :meth:`to_df_dict`: ``hier`` is a node table, or a node list such
        as :attr:`nodes`, and ``df`` holds euler-mode frames. A table is
        data, so the clip builds nodes of its own from it; a node list is
        taken over as the constructor takes one: the clip owns and
        freezes those nodes, and nodes another clip owns are shared.
        Delegates to :func:`pybvh.df_to_bvh`; see it for the two forms of
        ``hier``, the expected column naming and what raises.
        """
        from .dataframe import df_to_bvh

        return df_to_bvh(hier, df)

    def _non_end_site_indices(self) -> list[int]:
        """Indices in ``nodes`` order corresponding to non-end-site joints.

        The same indices select the joint-axis subset of any per-node
        array (e.g. :meth:`node_positions` output of shape ``(F, N, 3)``)
        to produce a joint-aligned ``(F, J, 3)``.
        """
        return [i for i, n in enumerate(self.nodes) if not n.is_end_site()]

    def _world_node_positions(self) -> npt.NDArray[np.float64]:
        """World-frame FK positions for all frames, computed once per input.

        The array is the clip's derived state and is read-only;
        :meth:`node_positions` derives fresh arrays from it.
        """
        positions = self._derived.node_positions
        if positions is None:
            motion = self._motion_on_skeleton()
            positions = frames_to_node_positions(
                self,
                root_pos=motion["root_pos"],
                joint_angles=motion["joint_angles"],
                centered="world",
                joint_translations=motion["joint_translations"],
            )
            self._derived.node_positions = positions
        return positions

    def _all_node_positions(self, centered: str = "world") -> npt.NDArray[np.float64]:
        """Every frame's node positions under *centered*, with no copy of the clip's.

        Under ``"world"`` the result is the clip's read-only derived array
        itself, so a caller hands the user only arrays computed from it,
        never the result or a view of it; :meth:`node_positions` hands out a
        copy.
        """
        _check_centered(centered)
        world = self._world_node_positions()
        if centered == "world":
            return world
        anchor = self._body_anchor_positions()
        if centered == "anchor":
            return world - anchor[:, np.newaxis, :]
        if centered == "anchor_ground":
            return world - _ground_plane_offset(anchor, self.world_up)[:, np.newaxis, :]
        # with no frames there is nothing to center on
        if self.frame_count == 0:
            return world.copy()
        return world - _ground_plane_offset(anchor[0], self.world_up)

    def node_positions(
        self, frame: int | None = None, centered: str = "world"
    ) -> npt.NDArray[np.float64]:
        """Per-node 3D positions (joints + end sites) — shape ``(F, N, 3)``.

        Returns an ndarray of shape ``(N, 3)`` for a single frame or
        ``(F, N, 3)`` for all frames, where *N* is the total number of
        nodes (joints + end sites). Use :attr:`node_index` to look up
        rows by name.

        For the joint-axis subset (excluding end sites) that aligns with
        :attr:`joint_angles` and :meth:`joint_velocities`, use
        :meth:`joint_positions` instead.

        World-frame forward kinematics is cached across calls (invalidated
        whenever motion data changes), so repeated calls — including with
        different ``centered`` modes — only pay for FK once.

        Parameters
        ----------
        frame : int or None
            Frame index to return. ``None`` (default) returns all frames.
            Negative indices count from the end (NumPy semantics), so
            ``-1`` is the **last** frame.
        centered : str
            ``"world"`` – every node at its world position.
            ``"anchor"`` – :attr:`body_anchor` at origin for all frames.
            ``"first_ground"`` – ground-plane centering: the first frame's body
            anchor position is subtracted in the two axes perpendicular
            to :attr:`world_up`; the up coordinate is untouched, so the
            motion starts above the origin at its original height.
            ``"anchor_ground"`` – each frame's body anchor position is
            subtracted in the two axes perpendicular to
            :attr:`world_up`; the up coordinate is untouched, so the
            body's travel is removed and its height kept: the anchor
            stays above the origin, at its own height on every frame.

            The three centred modes read the body anchor, which is the
            root on a clip whose anchor is its root. On a root that
            stays in place with the body below it (a static root,
            ``CHANNELS 0``, or a reference node whose channels do not
            change), they centre on the body, not on the reference
            node: under ``"anchor"`` the body anchor stays at the origin and
            the reference node moves.

            With *frame* given, every mode gives that frame's row of the
            whole clip: ``node_positions(frame=k, centered=c)`` equals
            ``node_positions(centered=c)[k]``. For one pose at the
            origin of the ground plane, pass ``"anchor_ground"``.
        """
        _check_centered(centered)
        if frame is None:
            positions = self._all_node_positions(centered)
            # Only under "world" is the result the clip's own array.
            return positions.copy() if centered == "world" else positions
        if not -self.frame_count <= frame < self.frame_count:
            raise IndexError(
                f"frame {frame} is out of range for "
                f"{self.frame_count} frames. Use frame=None for all frames."
            )
        actual = frame if frame >= 0 else frame + self.frame_count
        motion = self._motion()
        anchor = 0 if centered == "world" else self._body_anchor_index()
        if centered == "first_ground":
            # The frame's row of the whole clip: frame 0's anchor is
            # subtracted, not this frame's.
            start = motion["root_pos"][0] if anchor == 0 else self._frame_world_positions(0)[anchor]
            return self._frame_world_positions(actual) - _ground_plane_offset(start, self.world_up)
        if anchor != 0:
            # Centred on the anchor's row of the frame's world positions.
            world_frame = self._frame_world_positions(actual)
            if centered == "anchor":
                return world_frame - world_frame[anchor]
            return world_frame - _ground_plane_offset(world_frame[anchor], self.world_up)
        if self._derived.node_positions is not None:
            world_frame = self._derived.node_positions[actual]
            if centered == "world":
                return world_frame.copy()
            if centered == "anchor":
                return world_frame - motion["root_pos"][actual]
            return world_frame - _ground_plane_offset(motion["root_pos"][actual], self.world_up)
        return frames_to_node_positions(
            self,
            root_pos=motion["root_pos"][actual],
            joint_angles=motion["joint_angles"][actual],
            centered=centered,
            joint_translations=motion["joint_translations"][actual],
        )

    def _frame_world_positions(self, frame: int) -> npt.NDArray[np.float64]:
        """The world positions of the nodes at *frame* (``0 <= frame < F``), ``(N, 3)``.

        A row of the cached world positions when the clip holds them, so
        do not write into it; else one frame of forward kinematics, with
        no whole-clip run.
        """
        if self._derived.node_positions is not None:
            return self._derived.node_positions[frame]
        motion = self._motion()
        return frames_to_node_positions(
            self,
            root_pos=motion["root_pos"][frame],
            joint_angles=motion["joint_angles"][frame],
            joint_translations=motion["joint_translations"][frame],
        )

    def joint_positions(
        self, frame: int | None = None, centered: str = "world"
    ) -> npt.NDArray[np.float64]:
        """Per-joint 3D positions (end sites excluded) — shape ``(F, J, 3)``.

        Joint-axis subset of :meth:`node_positions`. Index-aligns with
        :attr:`joint_angles` and :meth:`joint_velocities` — use
        :attr:`joint_index` to look up rows by name.

        Parameters
        ----------
        frame : int or None
            Frame index to return. ``None`` (default) returns all frames;
            negative indices count from the end (``-1`` = last frame).
        centered : str
            See :meth:`node_positions`.
        """
        np_arr = self.node_positions(frame=frame, centered=centered)
        keep = self._non_end_site_indices()
        # node_positions output is either (N, 3) or (F, N, 3); slice the
        # node axis with `keep` — works for both shapes.
        return np_arr[..., keep, :]

    def rest_pose_positions(self) -> npt.NDArray[np.float64]:
        """Rest-pose node positions (all angles zero, root at origin) — ``(N, 3)``.

        Derived from the skeleton offsets alone, so it works on Bvh objects
        with no motion data. Use :attr:`node_index` to look up rows by name.

        Computed once per skeleton; each call returns a new array, so
        writing into it leaves the clip unchanged.
        """
        return self._rest_pose().copy()

    def _rest_pose(self) -> npt.NDArray[np.float64]:
        """The rest pose, computed once per skeleton and read-only.

        pybvh's own readers share this array; :meth:`rest_pose_positions`
        hands the user a copy of it.
        """
        rest = self._derived.rest_pose_positions
        if rest is None:
            rest = _build_rest_pose(self)
            self._derived.rest_pose_positions = rest
        return rest

    def rest_pose_angles(self) -> npt.NDArray[np.float64]:
        """Rest-pose joint angles — zeros of shape ``(J, 3)`` (radians).

        Companion of :meth:`rest_pose_positions` in ``joint_angles`` space
        (one single-frame row, matching :attr:`joint_index`).
        """
        return np.zeros((self.joint_count, 3), dtype=np.float64)

    def to_df_dict(
        self, mode: str = "euler", centered: str = "world"
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return a dict of arrays for ``pd.DataFrame(result)``.

        Each key is a column name, each value a 1-D NumPy array of
        length ``frame_count``. The first column is ``time``, each
        frame's time in seconds (its index times :attr:`frame_time`).
        On an unset clip (``frame_time`` 0) every time is NaN, which
        :meth:`from_df` reads back as an unset clip; zeros would read
        back as a constant column, which it refuses, and raising would
        refuse the motion of a clip that has no rate.

        Parameters
        ----------
        mode : str, optional
            ``'euler'`` (default): the motion as a file carries it, in
            degrees, per joint in :attr:`nodes` order:
            ``<joint>_<axis>_pos`` per position channel if the joint
            declares any (the root's from :attr:`root_pos`, another
            joint's from :attr:`joint_translations`), then
            ``<joint>_<axis>_rot`` per rotation channel, the columns of
            a frame line in a file. A joint that does not translate has
            no ``_pos`` columns, and a static root (``CHANNELS 0``) no
            columns at all; a clip whose joints other than the root only
            rotate gets the root's ``_pos`` columns, then each joint's
            ``_rot`` columns. :func:`~pybvh.df_to_bvh` and
            :meth:`from_df` read this form back.
            ``'coordinates'``: the posed positions of
            :meth:`node_positions`, ``<node>_X``, ``<node>_Y`` and
            ``<node>_Z`` per node in :attr:`nodes` order, end sites
            included. This export is one-way: positions do not determine
            the joint angles, and ``df_to_bvh`` reads euler-mode frames.
        centered : str, optional
            ``"world"`` (default), ``"anchor"``, ``"first_ground"`` or
            ``"anchor_ground"`` — see :meth:`node_positions` for their
            semantics. Only used when
            ``mode='coordinates'``.

        Returns
        -------
        dict
            Column-name → 1-D array mapping, ready for ``pd.DataFrame()``.

        Raises
        ------
        ValueError
            If *mode* is not ``'euler'`` or ``'coordinates'``; in
            ``'coordinates'`` mode, if *centered* is not one of its three
            values; or if the motion arrays disagree on frame count (the
            setters do not check it).

        Notes
        -----
        Node names are not unique in a BVH file: joints may repeat a
        name, and the end sites of two joints of one name then share
        their generated display name. Columns are labelled the way ``pandas.read_csv`` labels
        repeated headers: the first node of a name keeps it, each later
        one gets ``.1``, ``.2``, ... before the axis and kind
        (``Arm_Z_rot``, ``Arm.1_Z_rot``; ``Hand_EndSite_X``,
        ``Hand_EndSite.1_X``), counted up until the label is neither
        another node's name nor a label already given. Blender's BVH
        importer does the same with ``.001``. The suffix
        is a column label only: the node keeps its name, which
        :meth:`to_node_table` and :attr:`nodes` carry and
        :func:`~pybvh.df_to_bvh` reads the labels back from. The
        alternative, raising on a repeated name, would refuse every hand
        whose thumb tip and finger tip are both end sites in coordinates
        mode; before v0.10.0 the later node's values silently replaced
        the earlier one's under the earlier one's column. The suffix is
        assigned over the nodes the mode exports, so a joint can be
        ``X`` in euler mode and ``X.1`` in coordinates mode when an
        earlier end site shares its name.
        """
        correct_modes = ["euler", "coordinates"]

        if mode == "euler":
            return self._get_df_constructor_euler_angles()
        elif mode == "coordinates":
            return self._get_df_constructor_spatial_coord(centered=centered)
        else:
            raise ValueError(
                f"The value {mode} is not recognized for the mode argument.\
                             Currently recognized keywords are {correct_modes}"
            )

    def _get_df_constructor_euler_angles(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for Euler-angle DataFrame.

        DataFrame columns are in degrees for human readability — the
        ``_rot`` columns are the rad→deg-converted view of the internal
        radians-valued :attr:`joint_angles`.
        """
        motion = self._motion_on_skeleton()
        arrays = {**motion, "joint_angles": np.rad2deg(motion["joint_angles"])}
        blocks = _euler_blocks(self.nodes)
        flat = np.empty((self.frame_count, 3 * len(blocks)))
        for i, block in enumerate(blocks):
            flat[:, 3 * i : 3 * i + 3] = block.columns(arrays)
        return self._timed_columns(_motion_column_names(self.nodes, "euler"), flat)

    def _get_df_constructor_spatial_coord(
        self, centered: str
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for spatial-coordinate DataFrame."""
        spatial_array = self.node_positions(centered=centered)  # (F, N, 3)
        # The width is spelled out: NumPy cannot infer -1 from no frames.
        flat = spatial_array.reshape(self.frame_count, 3 * len(self.nodes))
        return self._timed_columns(_motion_column_names(self.nodes, "coordinates"), flat)

    def _timed_columns(
        self, columns: list[str], flat: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """``time``, then *columns* over the columns of *flat*."""
        times: npt.NDArray[np.float64]
        if self.frame_time == 0:
            times = np.full(self.frame_count, np.nan)
        else:
            times = np.arange(self.frame_count, dtype=np.float64) * self.frame_time
        result: dict[str, npt.NDArray[np.float64]] = {"time": times}
        result.update(zip(columns, flat.T, strict=True))
        return result

    def to_node_table(self) -> list[dict[str, Any]]:
        """Export the skeleton as a node table: one plain ``dict`` per node.

        The skeleton as plain data, the counterpart of :meth:`from_df`'s
        ``hier`` argument as :meth:`to_df_dict` is of its ``df``. One
        entry per node in :attr:`nodes` order, each with ``name``,
        ``parent`` (the index of the parent's entry, ``None`` on the
        root), ``offset``, ``rot_channels`` on the root and joints and
        ``pos_channels`` on the root and translating joints; an entry
        without ``rot_channels`` is an end site. Position is identity, so two end sites under one
        joint and joints sharing a name are carried as they are, which
        no name-keyed export can do. :func:`~pybvh.nodes_to_table`
        defines the format and names its conventions;
        :func:`~pybvh.nodes_from_table` builds a tree back from it.

        The table is the named twin of :attr:`fk_topology`: ``parent``
        with ``None`` read as ``-1`` is its ``parent_idx``, the stacked
        ``offset`` values are its ``offsets``, and the entries with a
        ``rot_channels`` key (a static root's ``[]`` included) are its
        joint columns, in order.

        Returns
        -------
        list of dict
            One entry per node, in :attr:`nodes` order. Entries are
            fresh objects and each ``offset`` is a copy, safe to mutate.
        """
        return nodes_to_table(self.nodes)

    @property
    def node_index(self) -> dict[str, int]:
        """Mapping from node name to its integer index in ``nodes``.

        Indexes the output of :meth:`node_positions` (shape
        ``(F, N, 3)``), which includes end sites. For indexing
        :attr:`joint_angles` (shape ``(F, J, 3)``, excludes end sites),
        use :attr:`joint_index` instead.

        .. warning::
            ``joint_index`` and ``node_index`` share keys for every
            non-end-site joint but return **different integers** once
            any end site has appeared earlier in the hierarchy. Indexing
            ``joint_angles`` with ``node_index`` (or
            :meth:`node_positions` with ``joint_index``) produces
            silently misaligned data — no shape mismatch, just the
            wrong limb. Pick one consistently per array, or use
            :meth:`Bvh.index` to make the intent explicit at the call
            site.

        Returns
        -------
        dict
            ``{node_name: int}`` for every node (joints and end sites).
        """
        index = self._derived.node_index
        if index is None:
            index = {node.name: i for i, node in enumerate(self.nodes)}
            self._derived.node_index = index
        return index

    @property
    def joint_index(self) -> dict[str, int]:
        """Mapping from joint name to its integer index in ``joint_angles`` axis 1.

        Excludes end sites.  Use this instead of
        ``bvh.joint_names.index(name)`` for joint-axis lookups.

        .. warning::
            ``joint_index`` and ``node_index`` share keys for every
            non-end-site joint but return **different integers** once
            any end site has appeared earlier in the hierarchy. Indexing
            :meth:`node_positions` with ``joint_index`` (or
            ``joint_angles`` with ``node_index``) produces silently
            misaligned data. Pick one consistently per array, or use
            :meth:`Bvh.index` to make the intent explicit at the call
            site.

        Returns
        -------
        dict
            ``{joint_name: int}`` for every non-end-site joint.
            Values cover ``range(joint_count)``.
        """
        index = self._derived.joint_index
        if index is None:
            index = {name: i for i, name in enumerate(self.joint_names)}
            self._derived.joint_index = index
        return index

    def index(self, node: NodeSelector, space: Literal["joint", "node"]) -> int:
        """The integer index of one node in the requested index space.

        Unambiguous alternative to picking between :attr:`joint_index`
        and :attr:`node_index` at the call site. Use ``space='joint'``
        when indexing :attr:`joint_angles` / :meth:`joint_velocities` /
        :meth:`joint_accelerations` / :meth:`joint_positions` /
        :meth:`angular_velocities` (any ``(F, J, ...)`` array). Use
        ``space='node'`` when indexing :meth:`node_positions` /
        :meth:`node_velocities` / :meth:`node_accelerations` (any
        ``(F, N, ...)`` array).

        This is the resolver of every parameter of ``Bvh``, ``analysis``
        and ``features`` that selects a node (see
        :data:`~pybvh.NodeSelector`). A ``BvhNode`` is found by identity
        among :attr:`nodes`. A name must match exactly one node of the
        requested space: :attr:`node_index` and :attr:`joint_index` keep
        the last of the nodes that share a name, where this method raises
        and asks for the node. A name that a joint and an end site share
        therefore selects the joint with ``space='joint'``, and raises
        with ``space='node'``.

        Parameters
        ----------
        node : str or BvhNode
            The node's name, or the node itself, one of :attr:`nodes`.
        space : {'joint', 'node'}
            Which index space to look up. ``'joint'`` excludes end sites.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If ``node`` is neither a ``str`` nor a ``BvhNode``. An integer
            is refused: it does not say which index space it counts in;
            node ``i`` is ``bvh.nodes[i]``.
        ValueError
            If ``space`` is not ``'joint'`` or ``'node'``; if ``node`` is
            a ``BvhNode`` that is not one of :attr:`nodes` (a node of
            another clip, or one taken before an edit replaced this
            clip's skeleton); if no node of the space carries the name, or
            more than one does; or if ``node`` selects an end site with
            ``space='joint'``.
        """
        if space not in ("joint", "node"):
            raise ValueError(f"space must be 'joint' or 'node', got {space!r}")
        position = self._node_position(node, space)
        if space == "node":
            return position
        if self.nodes[position].is_end_site():
            raise ValueError(self._end_site_in_joint_space(position))
        return self._non_end_site_indices().index(position)

    def _indices(self, nodes: Sequence[NodeSelector], space: Literal["joint", "node"]) -> list[int]:
        """The plural form of :meth:`index`: the indices of a set of nodes, in its order.

        Each entry resolves as :meth:`index` resolves one node. Beyond its
        refusals, a bare ``str`` or ``BvhNode`` given as the set raises
        ``TypeError`` (a ``str`` would otherwise be read letter by
        letter), and a set that selects one node twice, by the same
        selector or by a name and its node, raises ``ValueError``.
        """
        if isinstance(nodes, (str, BvhNode)):
            listed = repr(nodes) if isinstance(nodes, str) else "bvh.nodes[i]"
            raise TypeError(
                f"A set of nodes is a sequence of names or BvhNode objects, "
                f"not one {type(nodes).__name__}: pass [{listed}] to select "
                f"one node."
            )
        indices: list[int] = []
        for node in nodes:
            index = self.index(node, space)
            if index in indices:
                position = self._node_position(node, space)
                raise ValueError(
                    f"The set selects bvh.nodes[{position}] "
                    f"({self.nodes[position].name!r}) more than once. Select "
                    f"each node once."
                )
            indices.append(index)
        return indices

    def _node_position(self, node: NodeSelector, space: Literal["joint", "node"]) -> int:
        """The position in :attr:`nodes` of the node ``node`` selects in ``space``.

        A name is looked up among the nodes of ``space``; in joint space,
        a name that only end sites carry gives the first of them, which
        :meth:`index` then refuses as an end site.
        """
        if isinstance(node, BvhNode):
            for position, candidate in enumerate(self.nodes):
                if candidate is node:
                    return position
            raise ValueError(
                f"The node {node.name!r} is not one of this clip's nodes: it "
                f"belongs to another clip, or to this clip's skeleton before an "
                f"in-place edit replaced it. Pass one of bvh.nodes, or the "
                f"node's name."
            )
        if not isinstance(node, str):
            raise TypeError(
                f"A node is selected by its name (str) or by the BvhNode "
                f"itself, not by the {type(node).__name__} {node!r}. An "
                f"integer does not say whether it counts in joint space or in "
                f"node space: node i is bvh.nodes[i]."
            )
        positions = [
            position
            for position, candidate in enumerate(self.nodes)
            if candidate.name == node and (space == "node" or not candidate.is_end_site())
        ]
        if len(positions) > 1:
            listed = ", ".join(f"bvh.nodes[{position}]" for position in positions)
            raise ValueError(
                f"{len(positions)} {space}s of this clip are named {node!r}: "
                f"{listed}. Pass the node you mean in place of its name."
            )
        if positions:
            return positions[0]
        if space == "joint":
            end_sites = [i for i, candidate in enumerate(self.nodes) if candidate.name == node]
            if end_sites:
                return end_sites[0]
        names_at = "bvh.node_index" if space == "node" else "bvh.joint_names"
        raise ValueError(
            f"No {space} of this clip is named {node!r}; {names_at} lists the "
            f"{space}s' names.{self._old_end_site_name_hint(node)}"
        )

    def _old_end_site_name_hint(self, name: str) -> str:
        """The sentence that names the new form of an end-site name of before 0.10.0.

        Until 0.9.0, an end site was named ``'EndSite'`` + its parent's
        name, ``'EndSiteLeftToe'``; since 0.10.0 it is ``'LeftToe_EndSite'``
        (``_end_site_name``). The sentence is given only when ``name`` has
        the old form and a node of this clip carries the new name, and is
        empty otherwise. A later release can remove it.
        """
        old_prefix = "EndSite"
        if not name.startswith(old_prefix) or name == old_prefix:
            return ""
        new_name = _end_site_name(name[len(old_prefix) :], 1)
        if not any(candidate.name == new_name for candidate in self.nodes):
            return ""
        return f" The end sites are named {new_name!r} since 0.10.0."

    def _end_site_in_joint_space(self, position: int) -> str:
        """The refusal of the end site at ``position`` in joint space."""
        parent_position = int(self.fk_topology.parent_idx[position])
        return (
            f"bvh.nodes[{position}] ({self.nodes[position].name!r}) is an end "
            f"site: it has no rotation channels, so no index in joint space. "
            f"Select a joint, such as its parent bvh.nodes[{parent_position}] "
            f"({self.nodes[parent_position].name!r})."
        )

    @property
    def joint_tips(self) -> dict[str, int | None]:
        """Mapping from joint name to its end-site **node** index, or None.

        For every non-end-site joint (root included): the node-space index of the joint's end-site child — a row of :meth:`node_positions` output — or ``None`` for interior joints whose children are all joints. The tip is the bone's far end, so ``bvh.node_positions()[:, bvh.joint_tips["LeftFoot"]]`` is the toe-tip trajectory without knowing the end site's generated display name.

        Resolution is identity-based on the node tree, never by name: end-site display names are cosmetic (the parser generates the parent's name + ``'_EndSite'``) and may even collide with a real joint's name, which would make a name-keyed lookup through :attr:`node_index` silently pick the wrong node. A joint with several end-site children (nonstandard, but the parser accepts it) maps to the **first one in file order**.

        Returns
        -------
        dict
            ``{joint_name: int | None}`` for every non-end-site joint, in topological (``joint_names``) order. A fresh dict per access — mutate freely.

        See Also
        --------
        node_index : Name → node index for every node (joints *and* end sites).
        nodes : The flat depth-first node list these indices point into.
        """
        node_position = {id(node): i for i, node in enumerate(self.nodes)}
        tips: dict[str, int | None] = {}
        for node in self.nodes:
            if node.is_end_site():
                continue
            tips[node.name] = next(
                (
                    node_position[id(child)]
                    for child in node.children  # type: ignore[attr-defined]
                    if child.is_end_site()
                ),
                None,
            )
        return tips

    @property
    def joint_names(self) -> list[str]:
        """Names of non-end-site joints in topological order.

        Returns
        -------
        list of str
        """
        return [n.name for n in self.nodes if not n.is_end_site()]

    def _node_names(self) -> list[str]:
        """Every node's name, in node order, end sites included; names may repeat."""
        return [node.name for node in self.nodes]

    @property
    def joint_count(self) -> int:
        """Number of non-end-site joints.

        Returns
        -------
        int
        """
        return len(self.joint_names)

    @property
    def euler_orders(self) -> list[str]:
        """Per-joint Euler rotation orders as strings.

        Returns
        -------
        list of str
            e.g. ``['ZYX', 'ZYX', ...]``, one per non-end-site joint.
            Order matches ``joint_names`` and ``joint_angles`` axis 1.
            A static root (``CHANNELS 0``) declares no rotation channel
            and never rotates: it reads ``''``, which the rotation
            conversions read as the identity.
        """
        return [_euler_order(n) for n in self.nodes if _is_joint(n)]

    @property
    def fk_topology(self) -> FkTopology:
        """The skeleton as plain arrays — everything forward kinematics reads.

        Bone offsets, parent indices, joint-column indices and Euler
        orders, with no node objects: see :class:`~pybvh.FkTopology`.
        Pass it to :func:`~pybvh.frames_to_node_positions` to run FK from
        a preprocessed dataset, where the source ``.bvh`` is long closed.

        It is also this class's single derivation of skeleton topology —
        :attr:`edges`, :attr:`node_edges` and the bone list the
        :mod:`~pybvh.bvhplot` backends draw are all views of its
        ``parent_idx``, so they cannot disagree with the geometry FK
        produces.

        Recomputed per access rather than cached: it is an O(N) walk over
        a few dozen nodes, and a cache would need invalidating on every
        operation that touches offsets or channel order (:meth:`retarget`,
        :meth:`scale`, :meth:`mirror`, :meth:`change_euler_order`) for a
        saving that does not show up in a profile.

        Returns
        -------
        FkTopology
            ``joint_idx`` counts up in :attr:`nodes` order, so
            ``euler_orders`` matches :attr:`euler_orders` element for
            element.
        """
        return FkTopology.from_nodes(self.nodes)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Skeleton edge list as ``(child_idx, parent_idx)`` tuples.

        Indices use ``joint_angles`` index space (non-end-site joints
        only, matching ``joint_names`` order).  The root joint has no
        parent and produces no edge, so a skeleton with *J* joints
        yields *J - 1* edges.

        Parents are resolved by node identity (via
        :attr:`fk_topology`), never by name — see :attr:`node_edges`.

        See Also
        --------
        node_edges : Same list but in ``nodes`` index space (includes
            end sites) — what graph models over the full visual skeleton
            typically want.
        """
        topology = self.fk_topology
        joint_idx = topology.joint_idx
        return [
            (int(joint_idx[child]), int(joint_idx[parent]))
            for child, parent in enumerate(topology.parent_idx)
            if parent >= 0 and joint_idx[child] >= 0
        ]

    @property
    def node_edges(self) -> list[tuple[int, int]]:
        """Skeleton edge list as ``(child_idx, parent_idx)`` tuples in
        ``nodes`` index space (includes end sites).

        Parallels :attr:`edges` (joint-axis only); use ``node_edges``
        when the downstream graph treats end sites as real leaves
        (visual skeleton, per-bone styling, GCN inputs over the full
        topology).  ``node_edges`` has one more edge per end site than
        ``edges``.

        Parents are resolved by node identity (via
        :attr:`fk_topology`), never by name. Node names are not unique in
        general — the parser derives end-site display names from the
        parent joint's name, so two end sites under one joint collide —
        and a name-keyed lookup would silently emit an edge pointing at
        the wrong parent, yielding a graph of the right length that
        encodes the wrong skeleton.
        """
        return [
            (child, int(parent))
            for child, parent in enumerate(self.fk_topology.parent_idx)
            if parent >= 0
        ]

    @overload
    def retarget(
        self,
        new_skeleton: Bvh,
        name_mapping: dict[str, str] | None = ...,
        strict: bool = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def retarget(
        self,
        new_skeleton: Bvh,
        name_mapping: dict[str, str] | None = ...,
        strict: bool = ...,
        inplace: Literal[False] = ...,
    ) -> Bvh: ...
    def retarget(
        self,
        new_skeleton: Bvh,
        name_mapping: dict[str, str] | None = None,
        strict: bool = False,
        inplace: bool = False,
    ) -> Bvh | None:
        """Copy joint offsets from a reference skeleton.

        A translating joint keeps its displacement from its rest pose: its
        ``joint_translations`` move by the change of its offset,
        ``pos + (new_offset - old_offset)`` in every frame.

        Parameters
        ----------
        new_skeleton : Bvh
            Reference skeleton whose offsets will be copied.
        name_mapping : dict, optional
            Maps self's joint names to ``new_skeleton``'s joint names,
            e.g. ``{'Hips': 'mixamorig:Hips', ...}``.
            Joints not in the mapping are matched by identical name.
            If None (default), all joints are matched by name.
        strict : bool, optional
            If True, raise ``ValueError`` when a joint in self has no
            match in ``new_skeleton``.  If False (default), unmapped
            joints keep their original offsets.
        inplace : bool, optional
            If True, modify self and return None.
            If False (default), return a modified copy.
            With True, the skeleton is replaced, not edited: a node taken from
            ``nodes`` before the call keeps its old values.

        Returns
        -------
        None or Bvh
            The result keeps this clip's world up, set or guessed: a new
            skeleton does not change which way is up, so it is not guessed
            again from the new offsets (see :attr:`world_up`).
        """
        try:
            new_skel_nodes = new_skeleton.nodes
        except AttributeError:
            raise ValueError("new_skeleton must be a Bvh object") from None

        # Build name → index lookup for the reference skeleton
        newnodes2idx = {n.name: i for i, n in enumerate(new_skel_nodes)}

        offsets = self._offsets()
        for i, node in enumerate(self.nodes):
            # Determine the target name in new_skeleton
            if name_mapping and node.name in name_mapping:
                target_name = name_mapping[node.name]
            else:
                target_name = node.name

            if target_name in newnodes2idx:
                offsets[i] = new_skel_nodes[newnodes2idx[target_name]].offset
            elif strict:
                raise ValueError(
                    f"Node '{node.name}' (mapped to '{target_name}') not found "
                    f"in new_skeleton and strict=True."
                )
            # else: keep original offset (lenient mode)

        # A translating joint keeps its displacement from its rest pose:
        # its frames move by the change of its offset (#186).
        translating = [i for i, node in enumerate(self.nodes) if _is_translating_joint(node)]
        translations = (
            self._motion_on_skeleton()["joint_translations"]
            + (offsets - self._offsets())[translating]
        )

        target = self if inplace else self.copy()
        target._replace_offsets(offsets, translations=translations)
        if inplace:
            return None
        return target

    @overload
    def scale(self, scale: float, *, inplace: Literal[True]) -> None: ...
    @overload
    def scale(self, scale: float, inplace: Literal[False] = ...) -> Bvh: ...
    def scale(self, scale: float, inplace: bool = False) -> Bvh | None:
        """Uniformly scale all node offsets, the root translation and
        the joint translations.

        Parameters
        ----------
        scale : float
            Uniform scale factor. Only scalars are accepted: per-axis
            world factors applied to parent-local offsets are not
            geometrically meaningful once joints rotate during animation.
        inplace : bool, optional
            If True, modify self and return None.
            If False (default), return a modified copy.
            With True, the skeleton is replaced, not edited: a node taken from
            ``nodes`` before the call keeps its old values.

        Returns
        -------
        None or Bvh
        """
        if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"scale must be a scalar, got {type(scale).__name__}. "
                "Per-axis scaling is not supported: node offsets are "
                "parent-local, so per-axis world factors do not commute "
                "with joint rotations and would distort the animation."
            )

        factor = float(scale)
        target = self if inplace else self.copy()
        # root_pos first: the offset edit sets a static root's root_pos
        # to its new offset, which a later scaling would scale twice.
        target._set_motion(root_pos=target.root_pos * factor)
        target._replace_offsets(
            target._offsets() * factor,
            translations=target._motion_on_skeleton()["joint_translations"] * factor,
        )
        if inplace:
            return None
        return target

    @overload
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: NodeSelector | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: NodeSelector | None = ...,
        inplace: Literal[False] = ...,
    ) -> Bvh: ...
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: NodeSelector | None = None,
        inplace: bool = False,
    ) -> Bvh | None:
        """
        Change the Euler angle order of one or all joints.

        Converts rotation data via rotation matrices so the resulting
        Euler angles use the new order but represent the same physical
        rotations.

        Parameters
        ----------
        order : str or list of 3 chars
            New rotation order, e.g. 'XYZ' or ['X', 'Y', 'Z'].
        joint : str, BvhNode, or None
            The one joint to change, by its name or as the node itself (see
            :meth:`index`, ``space='joint'``). If None (default), every joint
            that declares rotation channels is changed to the new order: a
            static root (``CHANNELS 0``) declares none, keeps its order
            ``''`` and its zero rotation.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.
            With True, the skeleton is replaced, not edited: a node taken from
            ``nodes`` before the call keeps its old values.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.

        Raises
        ------
        TypeError
            If ``joint`` is neither a name, a ``BvhNode`` nor None.
        ValueError
            If :meth:`index` refuses ``joint`` in joint space, or if
            ``joint`` names a static root (``CHANNELS 0``), which declares
            no rotation channel.
        """
        if isinstance(order, str):
            new_order = list(order.upper())
        else:
            new_order = [c.upper() for c in order]

        if joint is None:
            joint_indices = self._rotating_joint_columns()
        else:
            joint_indices = [self.index(joint, space="joint")]
            if joint_indices == [0]:
                self._refuse_static_root("change_euler_order gives the root an Euler order")
        return self._change_euler_orders(new_order, joint_indices, inplace)

    def _rotating_joint_columns(self) -> list[int]:
        """The joint columns whose joints declare rotation channels: every
        column but a static root's (``CHANNELS 0``), which has no order to
        change.

        The one rule of :meth:`change_euler_order` with no joint and of
        :meth:`_in_euler_order`.
        """
        return [k for k, order in enumerate(self.euler_orders) if order]

    def _in_euler_order(self, order: str) -> Bvh | None:
        """A copy with each joint that declares an Euler order in *order*,
        or ``None`` when each already declares *order*.

        ``harmonize`` calls it. The joints are those
        :meth:`change_euler_order` changes with no joint: a static root
        keeps its order ``''``.
        """
        rotating = self._rotating_joint_columns()
        orders = self.euler_orders
        if all(orders[k] == order for k in rotating):
            return None
        return self._change_euler_orders(list(order.upper()), rotating, inplace=False)

    def _change_euler_orders(
        self, new_order: list[str], joint_indices: list[int], inplace: bool
    ) -> Bvh | None:
        """Re-express the joints at ``joint_indices`` in ``new_order``.

        :meth:`change_euler_order` after its checks, and
        :meth:`_in_euler_order` with the joints that declare rotation
        channels.
        """
        target = self if inplace else self.copy()
        # The new skeleton is built, and the order checked, before any
        # angle is converted.
        table = nodes_to_table(target.nodes)
        joint_rows = [row for row in table if "rot_channels" in row]
        old_orders = [row["rot_channels"] for row in joint_rows]
        for j_idx in joint_indices:
            joint_rows[j_idx]["rot_channels"] = new_order
        new_nodes = nodes_from_table(table)

        # The array the clip holds (the public view is read-only), or on a
        # clip without motion one with no frame to convert.
        angles = target._motion_on_skeleton()["joint_angles"]
        for j_idx in joint_indices:
            old_order = old_orders[j_idx]
            if old_order == new_order:
                continue
            # Convert: old Euler → rotmat → new Euler, then write back.
            R = rotations.euler_to_rotmat(angles[:, j_idx], old_order)
            angles[:, j_idx] = rotations.rotmat_to_euler(R, new_order)
        target._replace_skeleton(new_nodes)

        if inplace:
            return None
        return target

    def to_rotmat(self) -> npt.NDArray[np.float64]:
        """
        Convert the per-joint Euler angles to rotation matrices.

        Returns
        -------
        joint_rotmats : ndarray, shape (num_frames, num_joints, 3, 3)
            Rotation matrix for each joint in each frame.
            Joint order matches ``self.joint_names`` / ``self.joint_index``.

        Notes
        -----
        The conversions carry rotations only: :attr:`root_pos` and
        :attr:`joint_translations` are read from the clip as they are.

        When multiple rotation representations are needed (e.g. 6D for
        the model and quaternions for runtime SLERP), call ``to_rotmat``
        once and apply the relevant ``rotations.rotmat_to_*`` primitives
        directly — the Euler conversion runs once instead of per
        representation.
        """
        return rotations.euler_to_rotmat(self.joint_angles, self.euler_orders)

    def to_6d(self) -> npt.NDArray[np.float64]:
        """
        Convert the per-joint Euler angles to the 6D rotation representation.

        The 6D representation (Zhou et al., CVPR 2019) is continuous and
        well-suited for neural network training.

        Returns
        -------
        joint_rot6d : ndarray, shape (num_frames, num_joints, 6)
            6D rotation for each joint in each frame.

        Notes
        -----
        When multiple representations are needed, call :meth:`to_rotmat`
        once and apply :func:`pybvh.rotations.rotmat_to_rot6d` /
        ``rotmat_to_quat`` / ``rotmat_to_axisangle`` directly to avoid
        running the Euler conversion more than once.
        """
        return rotations.rotmat_to_rot6d(self.to_rotmat())

    def to_quat(self) -> npt.NDArray[np.float64]:
        """
        Convert the per-joint Euler angles to quaternions.

        Returns
        -------
        joint_quats : ndarray, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) for each joint in each frame, in
            canonical form (``w >= 0``).

        Notes
        -----
        The canonical form is applied **per frame**, so the returned
        sequence is not guaranteed to be temporally continuous: a joint
        rotating through 180° flips sign between adjacent frames even
        though the motion is smooth. Every quaternion is still exactly
        the right rotation; it is the representation that jumps. Wrap
        with :func:`pybvh.rotations.quat_unwrap` when feeding the array
        to anything that differences or measures distance on the raw
        values.

        See :meth:`to_rotmat` for the multi-representation reuse pattern.
        """
        return _euler_to_quat(self.joint_angles, self.euler_orders)

    def to_axisangle(self) -> npt.NDArray[np.float64]:
        """
        Convert the per-joint Euler angles to axis-angle vectors.

        The axis-angle representation is the unit rotation axis scaled
        by the rotation angle in radians.  Used in SMPL/SMPL-X body
        models and many pose estimation pipelines.

        Returns
        -------
        joint_aa : ndarray, shape (num_frames, num_joints, 3)
            Axis-angle vector for each joint in each frame.

        Notes
        -----
        See :meth:`to_rotmat` for the multi-representation reuse pattern.
        """
        return rotations.rotmat_to_axisangle(self.to_rotmat())

    def _from_rotations(
        self,
        method: str,
        param_name: str,
        rotation: npt.ArrayLike,
        to_rotmat: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        inplace: object,
    ) -> Bvh | None:
        """The ``from_*`` conversions: check, convert to Euler, write ``joint_angles``.

        *method* and *param_name* name the caller and its rotation
        argument in the errors; *to_rotmat* turns the array into rotation
        matrices. Writes through the ``joint_angles`` setter, so the call
        is a user motion replacement.
        """
        if not isinstance(inplace, (bool, np.bool_)):
            # A call of the old signature, from_*(root_pos, rotation), lands
            # its rotation array here.
            raise TypeError(
                f"{method} takes the rotation array only: "
                f"{method}({param_name}, inplace=False), got {type(inplace).__name__} "
                f"for inplace. It keeps root_pos and joint_translations; to set "
                f"the root position too, assign bvh.root_pos, or build a new clip "
                f"with Bvh(nodes=..., root_pos=..., joint_angles=...)."
            )
        # Through the motion's check: on a clip whose arrays disagree on
        # frame count, the clip has no frame count to hold the rotations to.
        frame_count = len(self._motion()["joint_angles"])
        joint_rotmats = to_rotmat(np.asarray(rotation, dtype=np.float64))
        if joint_rotmats.ndim != 4:
            raise ValueError(
                f"Expected {param_name} with a frame axis and a joint axis before "
                f"the rotation, got shape {np.shape(rotation)}"
            )
        if joint_rotmats.shape[0] != frame_count:
            raise ValueError(
                f"{param_name} has {joint_rotmats.shape[0]} frames, but the clip has "
                f"{frame_count}: {method} replaces the rotations of the clip's "
                f"frames and keeps their count."
            )
        if joint_rotmats.shape[1] != self.joint_count:
            raise ValueError(
                f"Expected {self.joint_count} joints in {param_name}, got {joint_rotmats.shape[1]}"
            )
        if self.root.is_static() and np.any(rotations._off_identity(joint_rotmats[:, 0])):
            # Here, not in the conversion, to name the call and the root.
            self._refuse_static_root(f"{method} rotates the root")
        new_angles = rotations.rotmat_to_euler(joint_rotmats, self.euler_orders)
        target = self if inplace else self.copy()
        target.joint_angles = new_angles
        if inplace:
            return None
        return target

    @overload
    def from_rotmat(self, joint_rotmats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_rotmat(self, joint_rotmats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_rotmat(self, joint_rotmats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """
        Set the joint rotations from rotation matrices.

        Converts rotation matrices to Euler angles in each joint's
        rot_channels order and writes :attr:`joint_angles`. Inverse of
        :meth:`to_rotmat`. :attr:`root_pos` and :attr:`joint_translations`
        stay as they are; to replace them too, assign them, or build a new
        clip with the :class:`Bvh` constructor. Like an assignment of
        :attr:`joint_angles`, the call starts the world-up guess again.

        Parameters
        ----------
        joint_rotmats : array_like, shape (num_frames, num_joints, 3, 3)
            Rotation matrix per joint per frame. The frame count must be
            the clip's. Joint order must match ``self.joint_names``.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.

        Raises
        ------
        TypeError
            If ``inplace`` is not a bool, as in a call of the old
            signature with two arguments,
            ``from_rotmat(root_pos, joint_rotmats)``. Python's own
            ``TypeError`` refuses the other old calls, with ``inplace``
            as a third argument or a keyword.
        ValueError
            If the frame or joint count does not match the clip, or if
            the rotations turn a static root (``CHANNELS 0``): a rotation
            other than the identity, each matrix entry within ``1e-7``.
        """
        return self._from_rotations(
            "from_rotmat", "joint_rotmats", joint_rotmats, lambda r: r, inplace
        )

    @overload
    def from_6d(self, joint_rot6d: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_6d(self, joint_rot6d: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_6d(self, joint_rot6d: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """
        Set the joint rotations from 6D rotation data.

        Converts 6D rotations to Euler angles in each joint's
        rot_channels order and writes :attr:`joint_angles`. Inverse of
        :meth:`to_6d`. :attr:`root_pos` and :attr:`joint_translations`
        stay as they are; to replace them too, assign them, or build a new
        clip with the :class:`Bvh` constructor. Like an assignment of
        :attr:`joint_angles`, the call starts the world-up guess again.

        Parameters
        ----------
        joint_rot6d : array_like, shape (num_frames, num_joints, 6)
            6D rotation per joint per frame. The frame count must be the
            clip's. Joint order must match ``self.joint_names``.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.

        Raises
        ------
        TypeError
            If ``inplace`` is not a bool, as in a call of the old
            signature with two arguments,
            ``from_6d(root_pos, joint_rot6d)``. Python's own
            ``TypeError`` refuses the other old calls, with ``inplace``
            as a third argument or a keyword.
        ValueError
            If the frame or joint count does not match the clip, or if
            the rotations turn a static root (``CHANNELS 0``): a rotation
            other than the identity, each matrix entry within ``1e-7``.
        """
        return self._from_rotations(
            "from_6d", "joint_rot6d", joint_rot6d, rotations.rot6d_to_rotmat, inplace
        )

    @overload
    def from_quat(self, joint_quats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_quat(self, joint_quats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_quat(self, joint_quats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """
        Set the joint rotations from quaternions.

        Converts quaternions to Euler angles in each joint's
        rot_channels order and writes :attr:`joint_angles`. Inverse of
        :meth:`to_quat`. :attr:`root_pos` and :attr:`joint_translations`
        stay as they are; to replace them too, assign them, or build a new
        clip with the :class:`Bvh` constructor. Like an assignment of
        :attr:`joint_angles`, the call starts the world-up guess again.

        Parameters
        ----------
        joint_quats : array_like, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) per joint per frame. The frame count
            must be the clip's. Joint order must match
            ``self.joint_names``.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.

        Raises
        ------
        TypeError
            If ``inplace`` is not a bool, as in a call of the old
            signature with two arguments,
            ``from_quat(root_pos, joint_quats)``. Python's own
            ``TypeError`` refuses the other old calls, with ``inplace``
            as a third argument or a keyword.
        ValueError
            If the frame or joint count does not match the clip, or if
            the rotations turn a static root (``CHANNELS 0``): a rotation
            other than the identity, each matrix entry within ``1e-7``.
        """
        return self._from_rotations(
            "from_quat", "joint_quats", joint_quats, rotations.quat_to_rotmat, inplace
        )

    @overload
    def from_axisangle(self, joint_aa: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_axisangle(self, joint_aa: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_axisangle(self, joint_aa: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """
        Set the joint rotations from axis-angle vectors.

        Converts axis-angle vectors to Euler angles in each joint's
        rot_channels order and writes :attr:`joint_angles`. Inverse of
        :meth:`to_axisangle`. :attr:`root_pos` and
        :attr:`joint_translations` stay as they are; to replace them too,
        assign them, or build a new clip with the :class:`Bvh`
        constructor. Like an assignment of :attr:`joint_angles`, the call
        starts the world-up guess again.

        Parameters
        ----------
        joint_aa : array_like, shape (num_frames, num_joints, 3)
            Axis-angle vector per joint per frame. The frame count must be
            the clip's. Joint order must match ``self.joint_names``.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.

        Raises
        ------
        TypeError
            If ``inplace`` is not a bool, as in a call of the old
            signature with two arguments,
            ``from_axisangle(root_pos, joint_aa)``. Python's own
            ``TypeError`` refuses the other old calls, with ``inplace``
            as a third argument or a keyword.
        ValueError
            If the frame or joint count does not match the clip, or if
            the rotations turn a static root (``CHANNELS 0``): a rotation
            other than the identity, each matrix entry within ``1e-7``.
        """
        return self._from_rotations(
            "from_axisangle", "joint_aa", joint_aa, rotations.axisangle_to_rotmat, inplace
        )

    # ----------------------------------------------------------------
    # Frame slicing, concatenation, and resampling
    # ----------------------------------------------------------------

    def _slice_frames(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Bvh:
        """Implementation of ``bvh[start:end:step]`` (see :meth:`__getitem__`).

        Parameters
        ----------
        start, end, step : int or None
            Slice parameters (same semantics as ``array[start:end:step]``).

        Returns
        -------
        Bvh
            New Bvh object with the sliced frames and same skeleton.
        """
        s = slice(start, end, step)
        new_bvh = self._derive_clip(
            {name: array[s].copy() for name, array in self._motion().items()}
        )
        # Adjust frame frequency if step changes the sampling rate
        if step is not None and abs(step) != 1:
            new_bvh.frame_time = self.frame_time * abs(step)
        return new_bvh

    def _check_same_skeleton(self, other: Bvh) -> None:
        """Raise ``ValueError`` unless ``other`` has this clip's topology.

        The hierarchy (names, node kinds, parents, rest offsets within
        ``1e-6``) and the channel layout (which joints translate, each
        joint's rotation order),
        as :meth:`matches_topology` compares them: ``+``, ``+=`` and slice
        assignment write ``other``'s motion onto this clip's bones, which
        would silently change it where the two skeletons differ. The
        error names the first difference.
        """
        difference = _hierarchy_difference(self.nodes, other.nodes) or _channel_difference(
            self.nodes, other.nodes
        )
        if difference is None:
            return
        message = f"Cannot join clips whose skeletons differ: {difference.message}."
        if difference.kind == "offset":
            message += (
                " To put the other clip's motion on this clip's bones on purpose, "
                "join other.copy(nodes=self.to_node_table()) instead, after "
                "scaling the whole clip if the two are in different units."
            )
        raise ValueError(message)

    def _check_same_rate(self, other: Bvh, join: str) -> None:
        """Raise unless ``other`` has ``self``'s rate, before a join changes anything.

        ``a + b``, ``a += b`` and ``a[i:j] = b`` lay ``other``'s frames on
        ``self``'s timeline, so frames recorded at another rate would play
        too fast or too slow. The same rate is :meth:`matches_rate`'s rule.
        The message speaks of the clips as *a* (``self``) and *b*
        (``other``), as *join* writes them, and names the ways out. A
        clip with no rate cannot be resampled, so when one of the two is
        unset the only way out is to give it the rate its frames were
        recorded at.
        """
        if self.matches_rate(other):
            return
        message = (
            f"Cannot join clips whose rates differ in {join} "
            f"(a: {_describe_rate(self.frame_time)}; b: {_describe_rate(other.frame_time)})."
        )
        if self.frame_time == 0 or other.frame_time == 0:
            unset, peer = ("a", "b") if self.frame_time == 0 else ("b", "a")
            message += (
                f" {unset} has no rate: set {unset}.frame_time = {peer}.frame_time "
                f"(or {unset}.fps) when its frames were recorded at {peer}'s rate."
            )
        else:
            message += (
                " Join b.resample(a.fps) to keep b's timing on a's grid, or set "
                "b.frame_time = a.frame_time when the difference is only metadata."
            )
        raise ValueError(message)

    def _concat(self, other: Bvh) -> Bvh:
        """Implementation of ``self + other`` (see :meth:`__add__`).

        Parameters
        ----------
        other : Bvh
            Must have the same topology (:meth:`matches_topology`): the
            same names, node kinds, parents, rest offsets within ``1e-6``,
            translating joints and rotation orders.

        Returns
        -------
        Bvh
            New Bvh with frames from ``self`` followed by ``other``.

        Raises
        ------
        ValueError
            If the topologies differ, the message naming the first
            difference, or if the rates differ (:meth:`matches_rate`).
        """
        self._check_same_skeleton(other)
        self._check_same_rate(other, "a + b")

        other_motion = other._motion()
        new_bvh = self._derive_clip(
            {
                name: np.concatenate([array, other_motion[name]], axis=0)
                for name, array in self._motion().items()
            }
        )
        if self.source_path != other.source_path:
            new_bvh.source_path = None
        return new_bvh

    def _slerp_angles(
        self,
        angles: npt.NDArray[np.float64],
        left: npt.NDArray[np.intp],
        right: npt.NDArray[np.intp],
        alpha: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Frames between frames ``left`` and ``right`` of ``angles``, at ``alpha``.

        ``angles`` is ``(F, J, 3)``, each joint's angles in radians in its
        Euler order (``euler_orders``); ``left``, ``right`` and ``alpha``
        are ``(N,)``, one frame pair and one fraction in ``[0, 1]`` per new
        frame. Each joint goes through quaternions and is slerped on the
        shorter arc, so the result is ``(N, J, 3)``, in radians, read back
        in the joint's Euler order.
        """
        quats = _euler_to_quat(angles, self.euler_orders)
        between = rotations.quat_slerp(quats[left], quats[right], alpha[:, np.newaxis])
        return rotations.quat_to_euler(between, self.euler_orders)

    def resample(self, target_fps: float) -> Bvh:
        """Resample frames to a new frame rate via interpolation.

        Root position is linearly interpolated.  Joint rotations are
        converted to quaternions and interpolated with SLERP for
        smooth, gimbal-lock-free results.  This is the rotation-aware
        alternative to naive per-channel linear interpolation on Euler
        angles, which produces wobble and gimbal-lock artifacts.

        The new timestamps are ``0, 1/fps, 2/fps, …`` up to the original
        clip's duration — anchored at ``t = 0``, and the last sample is
        the largest multiple of the new period that still fits. The
        original final frame is reproduced only when the duration is an
        exact multiple of that period; otherwise the clip is shortened
        by up to one new frame period. The alternative convention —
        stretch the grid to land exactly on the final frame, giving
        ``round(duration · fps) + 1`` samples and an irregular last
        interval — keeps the endpoint at the cost of an inexact rate.
        A clip that already has the rate, by :meth:`matches_rate`'s rule,
        is not interpolated: the result keeps its samples and takes
        ``1 / target_fps``, the
        frame time ``bvh.fps = target_fps`` sets. So does a clip shorter
        than two frames, which has nothing to interpolate.

        Parameters
        ----------
        target_fps : float
            Target frames per second.

        Returns
        -------
        Bvh
            New Bvh with resampled frames.

        Raises
        ------
        ValueError
            If ``target_fps`` is not a positive finite number, or the
            clip's ``frame_time`` is 0 (unset), whatever its length.
        """
        _validate_fps(target_fps, name="target_fps")
        _require_rate(self.frame_time, "resample()")

        new_freq = 1.0 / target_fps
        if _matches_fps(self.frame_time, target_fps, "target_fps") or self.frame_count < 2:
            # The clip already has the rate, or has nothing to interpolate:
            # it keeps its samples and adopts the target's exact frame time.
            new_bvh = self.copy()
            new_bvh.frame_time = new_freq
            return new_bvh

        # Original and target timestamps. The epsilon on the arange stop
        # keeps the final original timestamp reachable despite float
        # rounding (np.arange guarantees values strictly below the stop).
        t_orig = np.arange(self.frame_count) * self.frame_time
        t_new = np.arange(0, t_orig[-1] + 1e-12, new_freq)

        num_new = len(t_new)

        # Find surrounding frame indices for each new timestamp
        idx_right = np.searchsorted(t_orig, t_new, side="right")
        idx_right = np.clip(idx_right, 1, self.frame_count - 1)
        idx_left = idx_right - 1

        # Compute interpolation parameter t in [0, 1]
        t_left = t_orig[idx_left]
        t_right = t_orig[idx_right]
        dt = t_right - t_left
        # Avoid division by zero for duplicate timestamps
        dt = np.where(dt < 1e-15, 1.0, dt)
        alpha = (t_new - t_left) / dt  # (num_new,)

        motion = self._motion()
        new_motion = {}
        for entry in self._MOTION_ARRAYS:
            array = motion[entry.name]
            if entry.interpolation == "linear":
                # np.interp per component, over the array flattened to
                # (F, components) and given back its shape. drop_frames'
                # weighted sum of the two neighbours would round otherwise,
                # so each operation keeps its own linear rule and its values.
                components = array.reshape(self.frame_count, -1)
                resampled = np.empty((num_new, components.shape[1]), dtype=np.float64)
                for column in range(components.shape[1]):
                    resampled[:, column] = np.interp(t_new, t_orig, components[:, column])
                new_motion[entry.name] = resampled.reshape(num_new, *array.shape[1:])
            elif entry.interpolation == "euler":
                new_motion[entry.name] = self._slerp_angles(array, idx_left, idx_right, alpha)
            else:
                raise entry.unknown_interpolation("resample()")

        new_bvh = self._derive_clip(new_motion)
        new_bvh.frame_time = new_freq
        return new_bvh

    # ----------------------------------------------------------------
    # Joint subsetting
    # ----------------------------------------------------------------

    def extract_joints(self, joints: Sequence[NodeSelector]) -> Bvh:
        """Extract a subset of joints into a new Bvh.

        Removed joints' offsets are collapsed into their nearest kept
        descendant via vector addition (valid at rest pose).  Their
        rotation contribution during animation is lost.

        A kept joint keeps its position channels. A removed translating
        joint's translation is summed into its nearest kept descendant,
        per frame, in place of its offset, and that descendant becomes a
        translating joint (see :attr:`joint_translations`); a removed
        joint that does not translate adds its offset to every frame of a
        translating descendant. So a kept joint's FK position is the
        source's whenever the removed joints above it do not rotate. An
        end site added below a childless kept joint reads the rest
        offsets of the removed joints between them.

        The result lists its nodes in depth-first order, as a file
        would, so writing it and reading it back keeps
        :meth:`matches_hierarchy`. A kept joint left with nothing below
        it (no kept descendant, no end site of its own) gets an end site
        named ``name + '_EndSite'``, placed right after it, at the offset
        of its nearest original end-site descendant (zero when it has
        none).

        ``source_path``, a manual ``world_up`` override, and a user-set
        ``lr_mapping`` (filtered to pairs whose joints are both kept) and a
        set :attr:`body_anchor` whose joint is kept are preserved on the
        result; a set anchor whose joint is removed is dropped, and the
        result uses the rule. A guessed world up is inherited, not
        guessed again from the kept joints (see :attr:`world_up`).

        Parameters
        ----------
        joints : sequence of str or BvhNode
            The joints to keep, each by its name or as the node itself
            (see :meth:`index`, ``space='joint'``); the order does not
            matter. The root must be included. End sites are handled
            automatically (kept if their parent is kept, otherwise
            removed) and cannot be selected. Of two joints that share a
            name, pass the node to keep one alone.

        Returns
        -------
        Bvh
            New Bvh with the reduced skeleton and corresponding motion
            data.

        Raises
        ------
        TypeError
            If ``joints`` is a single ``str`` or ``BvhNode`` rather than a
            sequence of them, or if an entry is neither (an integer
            included).
        ValueError
            If the root is not among ``joints``, if :meth:`index` refuses
            an entry in joint space, or if two entries select the same
            joint.
        """
        position_of_column = self._non_end_site_indices()
        kept = {
            id(self.nodes[position_of_column[column]])
            for column in self._indices(joints, space="joint")
        }

        if id(self.root) not in kept:
            raise ValueError(f"Root joint '{self.root.name}' must be in joints.")

        def survives_extraction(node: BvhNode) -> bool:
            if node.is_end_site():
                return node.parent is not None and id(node.parent) in kept
            return id(node) in kept

        # A kept joint ends childless in the new tree when nothing below
        # it survives: its end site then goes right after it, which is
        # its depth-first place. Walking the list backwards settles each
        # node's flag before its parent reads it, since parents precede
        # their children in ``nodes``.
        survives = [survives_extraction(node) for node in self.nodes]
        # The rows of the joints, which a clip without motion holds too.
        motion = self._motion_on_skeleton()
        position = {id(node): i for i, node in enumerate(self.nodes)}
        survivor_below = [False] * len(self.nodes)
        for i in range(len(self.nodes) - 1, 0, -1):
            parent_position = position[id(self.nodes[i].parent)]
            if survives[i] or survivor_below[i]:
                survivor_below[parent_position] = True

        translation_column_of = {
            id(node): column
            for column, node in enumerate(n for n in self.nodes if _is_translating_joint(n))
        }

        rows: list[dict[str, Any]] = []
        kept_old_j_indices: list[int] = []
        new_translations: list[npt.NDArray[np.float64]] = []
        # Original joint (by identity: names repeat) -> its row index
        row_of: dict[int, int] = {}

        joint_column = 0
        for i, node in enumerate(self.nodes):
            if node.is_end_site():
                if survives[i]:
                    rows.append(
                        {
                            "name": node.name,
                            "parent": row_of[id(node.parent)],
                            "offset": node.offset,
                        }
                    )
                continue

            column = joint_column
            joint_column += 1
            if not survives[i]:
                continue

            row: dict[str, Any] = {"name": node.name}
            if node is self.root:
                row["parent"] = None
                row["offset"] = node.offset
                row["pos_channels"] = node.pos_channels  # type: ignore[attr-defined]
            else:
                # Removed ancestors collapse into this joint's offset,
                # summed up to the nearest kept one (the root at the latest).
                # Each frame sums the same chain, with a translating joint's
                # translation in place of its offset; the joint translates
                # when one joint of the chain does.
                chain = [node]
                walker = node.parent
                while walker is not None and id(walker) not in kept:
                    chain.append(walker)
                    walker = walker.parent
                row["parent"] = row_of[id(walker)]
                row["offset"] = np.sum([link.offset for link in chain], axis=0)
                if any(_is_translating_joint(link) for link in chain):
                    row["pos_channels"] = ["X", "Y", "Z"]
                    new_translations.append(
                        np.sum(
                            [
                                motion["joint_translations"][:, translation_column_of[id(link)]]
                                if _is_translating_joint(link)
                                else np.broadcast_to(link.offset, (self.frame_count, 3))
                                for link in chain
                            ],
                            axis=0,
                        )
                    )
            row["rot_channels"] = node.rot_channels  # type: ignore[attr-defined]

            row_of[id(node)] = len(rows)
            rows.append(row)
            kept_old_j_indices.append(column)

            if not survivor_below[i]:
                # No name: the builder names it the joint's name +
                # '_EndSite', as the parser would.
                rows.append(
                    {"parent": row_of[id(node)], "offset": self._find_end_site_offset(node)}
                )

        new_nodes = nodes_from_table(rows)

        # Selecting the kept columns makes a new array.
        new_bvh = self._derive_clip(
            {
                "root_pos": self._root_pos.copy(),
                "joint_angles": motion["joint_angles"][:, kept_old_j_indices, :],
                "joint_translations": np.stack(new_translations, axis=1)
                if new_translations
                else np.empty((self.frame_count, 0, 3)),
            },
            nodes=new_nodes,
        )
        # A user-set L/R mapping keeps the pairs whose joints are both kept;
        # with none left, the pairs are detected from the kept names, as
        # they are when the user set none. The mapping goes by name, so a
        # name is kept when one of its joints is.
        user_mapping = self._lr_mapping_setting
        if isinstance(user_mapping, dict):
            kept_names = {node.name for node in self.nodes if id(node) in kept}
            kept_pairs = {
                left: right
                for left, right in user_mapping.items()
                if left in kept_names and right in kept_names
            }
            new_bvh._set_lr_mapping_setting(kept_pairs or "names")
        # A set body anchor stays when its joint is kept, at the joint's
        # row of the new table, which is its position in the new nodes.
        anchor = self._body_anchor_setting
        if anchor is not None and id(self.nodes[anchor]) in row_of:
            new_bvh._set_body_anchor_setting(row_of[id(self.nodes[anchor])])
        return new_bvh

    def _find_end_site_offset(self, node: BvhNode) -> npt.NDArray[np.float64]:
        """Find accumulated offset to the nearest end-site descendant."""
        # BFS through children
        queue = [(child, child.offset.copy()) for child in node.children]  # type: ignore[attr-defined]
        while queue:
            child, acc = queue.pop(0)
            if child.is_end_site():
                return acc
            for grandchild in child.children:
                queue.append((grandchild, acc + grandchild.offset))
        # Fallback: zero offset
        return np.zeros(3, dtype=np.float64)

    # ----------------------------------------------------------------
    #  Feature export (delegate to analysis / features modules)
    # ----------------------------------------------------------------

    def joint_velocities(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position velocities — shape ``(F, J, 3)``. See :func:`pybvh.analysis.joint_velocities`."""
        from . import analysis

        return analysis.joint_velocities(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def node_velocities(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position velocities (joints + end sites) — shape ``(F, N, 3)``. See :func:`pybvh.analysis.node_velocities`."""
        from . import analysis

        return analysis.node_velocities(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def joint_accelerations(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position accelerations — shape ``(F, J, 3)``. See :func:`pybvh.analysis.joint_accelerations`."""
        from . import analysis

        return analysis.joint_accelerations(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def node_accelerations(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position accelerations (joints + end sites) — shape ``(F, N, 3)``. See :func:`pybvh.analysis.node_accelerations`."""
        from . import analysis

        return analysis.node_accelerations(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def joint_speed_derivative(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint rate of change of speed ``d‖v‖/dt`` — shape ``(F, J)``. See :func:`pybvh.analysis.joint_speed_derivative`."""
        from . import analysis

        return analysis.joint_speed_derivative(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def node_speed_derivative(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node rate of change of speed ``d‖v‖/dt`` (joints + end sites) — shape ``(F, N)``. See :func:`pybvh.analysis.node_speed_derivative`."""
        from . import analysis

        return analysis.node_speed_derivative(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def angular_velocities(
        self,
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint angular velocities.  See :func:`pybvh.analysis.angular_velocities`."""
        from . import analysis

        return analysis.angular_velocities(
            self, in_frames=in_frames, stencil=stencil, pad=pad, degrees=degrees
        )

    def root_trajectory(
        self,
        world_up: str | None = None,
        include_velocities: bool = False,
        stencil: str = "central",
        pad: str = "edge",
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Extract root trajectory features.  See :func:`pybvh.analysis.root_trajectory`."""
        from . import analysis

        return analysis.root_trajectory(
            self,
            world_up=world_up,
            include_velocities=include_velocities,
            stencil=stencil,
            pad=pad,
            degrees=degrees,
        )

    @overload
    def foot_contacts(
        self,
        feet: Sequence[NodeSelector] | None = ...,
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: Literal[False] = ...,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def foot_contacts(
        self,
        feet: Sequence[NodeSelector] | None = ...,
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], ContactReport]: ...
    @overload
    def foot_contacts(
        self,
        feet: Sequence[NodeSelector] | None = ...,
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: bool,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ContactReport]: ...
    def foot_contacts(
        self,
        feet: Sequence[NodeSelector] | None = None,
        method: str = "combined",
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = None,
        vel_smooth_duration: float | None = None,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = None,
        floor: float | str | None = None,
        min_contact_duration: float = 0.1,
        min_gap_duration: float = 0.1,
        hysteresis: float = 0.25,
        adaptive: bool = False,
        height_reference: str | None = None,
        return_info: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ContactReport]:
        """Detect foot contact labels.  See :func:`pybvh.analysis.foot_contacts`."""
        from . import analysis

        return analysis.foot_contacts(
            self,
            feet=feet,
            method=method,
            vel_threshold=vel_threshold,
            vel_smooth_duration=vel_smooth_duration,
            height_threshold=height_threshold,
            floor=floor,
            min_contact_duration=min_contact_duration,
            min_gap_duration=min_gap_duration,
            hysteresis=hysteresis,
            adaptive=adaptive,
            height_reference=height_reference,
            return_info=return_info,
        )

    @overload
    def ground_contacts(
        self,
        nodes: Sequence[NodeSelector],
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: Literal[False] = ...,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def ground_contacts(
        self,
        nodes: Sequence[NodeSelector],
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], ContactReport]: ...
    @overload
    def ground_contacts(
        self,
        nodes: Sequence[NodeSelector],
        method: str = ...,
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        vel_smooth_duration: float | None = ...,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = ...,
        floor: float | str | None = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str | None = ...,
        return_info: bool,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ContactReport]: ...
    def ground_contacts(
        self,
        nodes: Sequence[NodeSelector],
        method: str = "combined",
        *,
        vel_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = None,
        vel_smooth_duration: float | None = None,
        height_threshold: float | Sequence[float] | npt.NDArray[np.float64] | None = None,
        floor: float | str | None = None,
        min_contact_duration: float = 0.1,
        min_gap_duration: float = 0.1,
        hysteresis: float = 0.25,
        adaptive: bool = False,
        height_reference: str | None = None,
        return_info: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ContactReport]:
        """Detect ground contacts for an arbitrary node set.  See :func:`pybvh.analysis.ground_contacts`."""
        from . import analysis

        return analysis.ground_contacts(
            self,
            nodes,
            method=method,
            vel_threshold=vel_threshold,
            vel_smooth_duration=vel_smooth_duration,
            height_threshold=height_threshold,
            floor=floor,
            min_contact_duration=min_contact_duration,
            min_gap_duration=min_gap_duration,
            hysteresis=hysteresis,
            adaptive=adaptive,
            height_reference=height_reference,
            return_info=return_info,
        )

    def auto_detect_feet(self) -> list[BvhNode]:
        """Auto-detect the clip's feet, as nodes, from skeleton topology.  See :func:`pybvh.analysis.auto_detect_feet`."""
        from . import analysis

        return analysis.auto_detect_feet(self)

    def to_feature_array(
        self,
        representation: str = "6d",
        include_root_pos: bool = True,
        include_velocities: bool = False,
        include_foot_contacts: bool = False,
        centered: str = "world",
        feet: Sequence[NodeSelector] | None = None,
        stencil: str = "central",
        pad: str = "edge",
        include_joint_translations: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Export motion as a flat feature array.  See :func:`pybvh.features.to_feature_array`."""
        from . import features

        return features.to_feature_array(
            self,
            representation=representation,
            include_root_pos=include_root_pos,
            include_velocities=include_velocities,
            include_foot_contacts=include_foot_contacts,
            centered=centered,
            feet=feet,
            stencil=stencil,
            pad=pad,
            include_joint_translations=include_joint_translations,
        )

    def feature_array_layout(
        self,
        *,
        num_feet: int = 0,
        representation: str = "6d",
        include_root_pos: bool = True,
        include_joint_translations: bool = False,
        include_velocities: bool = False,
        include_foot_contacts: bool = False,
    ) -> dict[str, slice]:
        """Column layout of :meth:`to_feature_array` output.  See :func:`pybvh.features.feature_array_layout`."""
        from . import features

        return features.feature_array_layout(
            num_joints=self.joint_count,
            num_feet=num_feet,
            num_translating_joints=_TRANSLATING_JOINTS.count(self.nodes),
            representation=representation,
            include_root_pos=include_root_pos,
            include_joint_translations=include_joint_translations,
            include_velocities=include_velocities,
            include_foot_contacts=include_foot_contacts,
        )

    # ----------------------------------------------------------------
    #  Motion descriptors (delegate to geometry / analysis modules)
    # ----------------------------------------------------------------
    #  Relational / trajectory queries select in NODE space, so end sites
    #  (fingertips, toe tips, head top) are first-class: each takes a node's
    #  name or the node itself, resolved by index(). Every descriptor reads
    #  the node positions the clip holds as derived state.

    def curvature(
        self,
        node: NodeSelector,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-frame trajectory curvature of ``node``. See :func:`pybvh.geometry.curvature`."""
        from . import geometry

        traj = self._all_node_positions()[:, self.index(node, "node"), :]
        return geometry.curvature(traj, stencil, pad)

    def torsion(
        self,
        node: NodeSelector,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-frame trajectory torsion of ``node``. See :func:`pybvh.geometry.torsion`."""
        from . import geometry

        traj = self._all_node_positions()[:, self.index(node, "node"), :]
        return geometry.torsion(traj, stencil, pad)

    def movement_phase(
        self,
        node: NodeSelector,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-frame movement phase (``speed · curvature``) of ``node``.

        Raises ``ValueError`` when the clip's ``frame_time`` is 0 (unset).
        See :func:`pybvh.geometry.movement_phase`."""
        from . import geometry

        _require_rate(self.frame_time, "movement_phase()")
        traj = self._all_node_positions()[:, self.index(node, "node"), :]
        return geometry.movement_phase(traj, self.frame_time, stencil, pad)

    def path_length(self, node: NodeSelector) -> float:
        """Arc length travelled by ``node``. See :func:`pybvh.geometry.path_length`."""
        from . import geometry

        return float(
            geometry.path_length(self._all_node_positions()[:, self.index(node, "node"), :])
        )

    def directness(self, node: NodeSelector) -> float:
        """Directness of ``node``'s path (net displacement ÷ path length).

        See :func:`pybvh.geometry.directness`."""
        from . import geometry

        return float(
            geometry.directness(self._all_node_positions()[:, self.index(node, "node"), :])
        )

    def ground_path(self, node: NodeSelector) -> geometry.GroundPath:
        """Ground-plane path of ``node`` (uses ``world_up``). See :func:`pybvh.geometry.ground_path`."""
        from . import geometry

        traj = self._all_node_positions()[:, self.index(node, "node"), :]
        return geometry.ground_path(traj, _axis_to_vector(self.world_up))

    def inter_joint_distance(
        self,
        pairs: Sequence[tuple[NodeSelector, NodeSelector]],
    ) -> npt.NDArray[np.float64]:
        """Per-frame distances between node pairs, each node a name or a ``BvhNode``.

        ``pairs`` is a sequence of pairs, such as ``[('Hips', 'Spine')]``;
        anything else raises ``TypeError``. A pair may name one node twice,
        and its distance is 0. See :func:`pybvh.geometry.inter_joint_distance`."""
        from . import geometry

        idx_pairs = [[self.index(a, "node"), self.index(b, "node")] for a, b in _node_pairs(pairs)]
        return geometry.inter_joint_distance(self._all_node_positions(), idx_pairs)

    def joint_angle(
        self,
        a: NodeSelector,
        vertex: NodeSelector,
        b: NodeSelector,
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Per-frame angle at ``vertex`` in ``a–vertex–b``. See :func:`pybvh.geometry.joint_angle`."""
        from . import geometry

        pos = self._all_node_positions()
        return geometry.joint_angle(
            pos[:, self.index(a, "node")],
            pos[:, self.index(vertex, "node")],
            pos[:, self.index(b, "node")],
            degrees=degrees,
        )

    def triangle_area(
        self, a: NodeSelector, b: NodeSelector, c: NodeSelector
    ) -> npt.NDArray[np.float64]:
        """Per-frame area of triangle ``(a, b, c)``. See :func:`pybvh.geometry.triangle_area`."""
        from . import geometry

        pos = self._all_node_positions()
        return geometry.triangle_area(
            pos[:, self.index(a, "node")],
            pos[:, self.index(b, "node")],
            pos[:, self.index(c, "node")],
        )

    def segment_axis_angle(
        self,
        node_a: NodeSelector,
        node_b: NodeSelector,
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Per-frame angle of the segment ``node_a→node_b`` to ``world_up``.

        See :func:`pybvh.geometry.segment_axis_angle`."""
        from . import geometry

        pos = self._all_node_positions()
        seg = pos[:, self.index(node_b, "node")] - pos[:, self.index(node_a, "node")]
        return geometry.segment_axis_angle(seg, _axis_to_vector(self.world_up), degrees=degrees)

    def bounding_box(self) -> geometry.BoundingBox:
        """Per-frame axis-aligned bounding box of all nodes. See :func:`pybvh.geometry.bounding_box`."""
        from . import geometry

        return geometry.bounding_box(self._all_node_positions())

    def bounding_sphere(self) -> geometry.BoundingSphere:
        """Per-frame approximate enclosing sphere of all nodes. See :func:`pybvh.geometry.bounding_sphere`."""
        from . import geometry

        return geometry.bounding_sphere(self._all_node_positions())

    def bounding_ellipsoid(self) -> geometry.BoundingEllipsoid:
        """Per-frame PCA-aligned bounding ellipsoid of all nodes. See :func:`pybvh.geometry.bounding_ellipsoid`."""
        from . import geometry

        return geometry.bounding_ellipsoid(self._all_node_positions())

    def center_of_mass(
        self,
        weights: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame centre of mass of all nodes (uniform by default; pass per-node masses).

        See :func:`pybvh.geometry.center_of_mass`."""
        from . import geometry

        return geometry.center_of_mass(self._all_node_positions(), weights=weights)

    def com_displacement(
        self,
        weights: npt.NDArray[np.float64] | None = None,
        com_ref: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame centre-of-mass travel from a reference.

        ``com_ref`` defaults to the **first frame's** centre of mass, in the
        same world frame as :meth:`center_of_mass`, so the result is how far
        the CoM has travelled since the start (``0`` at frame 0). Pass an
        explicit ``com_ref`` (e.g. ``center_of_mass().mean(0)``) for a
        different baseline. See :func:`pybvh.geometry.com_displacement`."""
        from . import geometry

        com = geometry.center_of_mass(self._all_node_positions(), weights=weights)
        if com_ref is None:
            com_ref = com[0]
        return geometry.com_displacement(com, com_ref)

    def verticality(self) -> npt.NDArray[np.float64]:
        """Per-frame height/width ratio along ``world_up``. See :func:`pybvh.geometry.verticality`."""
        from . import geometry

        return geometry.verticality(self._all_node_positions(), _axis_to_vector(self.world_up))

    def node_jerk(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position jerk — ``(F, N, 3)``. See :func:`pybvh.analysis.node_jerk`."""
        from . import analysis

        return analysis.node_jerk(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def joint_jerk(
        self,
        centered: str = "world",
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position jerk — ``(F, J, 3)``. See :func:`pybvh.analysis.joint_jerk`."""
        from . import analysis

        return analysis.joint_jerk(
            self, centered=centered, in_frames=in_frames, stencil=stencil, pad=pad
        )

    def smoothness(
        self,
        node: NodeSelector,
        metric: str = "sparc",
        **kwargs: Any,
    ) -> float:
        """Smoothness of ``node``'s speed profile. See :func:`pybvh.analysis.smoothness`.

        Computes the node's per-frame speed ``‖velocity‖`` and passes it
        to the chosen ``metric`` at sampling rate ``1 / frame_time``.

        ``metric`` is one of ``"sparc"`` (default),
        ``"dimensionless_jerk"``, ``"log_dimensionless_jerk"``,
        ``"integrated_squared_jerk"``, ``"mean_squared_jerk"``,
        ``"rms_squared_jerk"``, ``"number_of_peaks"``, ``"speed_metric"``.
        Metric-specific options are forwarded as keyword arguments —
        ``"sparc"`` accepts ``padlevel`` (FFT zero-padding exponent,
        default ``4``), ``fc`` (max cutoff frequency in Hz, default
        ``10.0``) and ``amp_th`` (amplitude threshold, default ``0.05``);
        ``"dimensionless_jerk"`` and ``"log_dimensionless_jerk"`` accept
        ``normalize`` (``"peak_speed"`` default, ``"mean_speed"`` or
        ``"amplitude"``) and ``amplitude``; ``"number_of_peaks"``
        accepts ``min_height`` (minimum height for a maximum to count;
        default counts all). The remaining metrics take none. Raises
        ``ValueError`` when the clip's ``frame_time`` is 0 (unset)."""
        from . import analysis

        _require_rate(self.frame_time, "smoothness()")
        vel = self.node_velocities()
        speed = np.linalg.norm(vel[:, self.index(node, "node"), :], axis=-1)
        value = analysis.smoothness(speed, 1.0 / self.frame_time, metric=metric, **kwargs)
        # One node's (T,) speed reduces to a scalar; an array's rank is not in its type.
        return cast(float, value)

    def velocity_reductions(self, node: NodeSelector):
        """Scalar reductions of ``node``'s speed profile (peak, mean, …).

        Computes the node's per-frame speed ``‖velocity‖`` and reduces it
        at sampling rate ``1 / frame_time``, and raises ``ValueError``
        when that ``frame_time`` is 0 (unset). See
        :func:`pybvh.analysis.velocity_reductions`."""
        from . import analysis

        _require_rate(self.frame_time, "velocity_reductions()")
        vel = self.node_velocities()
        speed = np.linalg.norm(vel[:, self.index(node, "node"), :], axis=-1)
        return analysis.velocity_reductions(speed, 1.0 / self.frame_time)

    def skeleton_size(self, feet: Sequence[NodeSelector] | None = None) -> float:
        """Absolute skeleton scale — mean rest-pose distance from the feet to the joint they branch from.

        That joint is the lowest one that has the feet in two or more of
        its branches, else :attr:`body_anchor`. Raises ``ValueError`` for a
        skeleton whose size cannot be measured (no feet found, or all feet
        on that joint) rather than
        returning a substitute. See :func:`pybvh.analysis.skeleton_size`."""
        from . import analysis

        return analysis.skeleton_size(self, feet=feet)

    def kinetic_energy(
        self,
        masses: npt.NDArray[np.float64] | Mapping[str, float] | None = None,
        centered: str = "world",
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-frame kinetic energy over joints. ``masses`` may be a ``(J,)`` array
        or a ``{joint_name: mass}`` mapping. See :func:`pybvh.analysis.kinetic_energy`."""
        from . import analysis

        return analysis.kinetic_energy(
            self, masses=masses, centered=centered, stencil=stencil, pad=pad
        )

    def cadence(
        self,
        feet: Sequence[NodeSelector] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """Step rate (onsets/second). See :func:`pybvh.analysis.cadence`."""
        from . import analysis

        return analysis.cadence(self, feet=feet, contacts=contacts)

    def stride_length(
        self,
        feet: Sequence[NodeSelector] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """Mean stride length. See :func:`pybvh.analysis.stride_length`."""
        from . import analysis

        return analysis.stride_length(self, feet=feet, contacts=contacts)

    def walking_pace(self) -> float:
        """Mean horizontal speed. See :func:`pybvh.analysis.walking_pace`."""
        from . import analysis

        return analysis.walking_pace(self)

    def gait_parameters(
        self,
        feet: Sequence[NodeSelector] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ):
        """Spatiotemporal gait parameters. See :func:`pybvh.analysis.gait_parameters`."""
        from . import analysis

        return analysis.gait_parameters(self, feet=feet, contacts=contacts)

    def range_of_motion(self, joint: NodeSelector) -> npt.NDArray[np.float64]:
        """Peak-to-peak range of ``joint``'s Euler angles — ``(3,)`` per channel.

        Selects in JOINT space (rotations exist only on joints), so an end
        site raises ``ValueError``, and a name that a joint and an end site
        share selects the joint. See :func:`pybvh.analysis.range_of_motion`."""
        from . import analysis

        idx = self.index(joint, "joint")
        return analysis.range_of_motion(self.joint_angles[:, idx, :], axis=0)

    # ----------------------------------------------------------------
    #  Spatial Augmentation Transforms (delegate to transforms module)
    # ----------------------------------------------------------------

    @overload
    def translate_root(self, offset: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def translate_root(self, offset: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def translate_root(self, offset: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """Shift root position by a constant offset.  See :func:`pybvh.transforms.translate_root`."""
        from . import transforms

        return transforms.translate_root(self, offset, inplace=inplace)  # type: ignore[call-overload, return-value]

    @overload
    def add_rotation_noise(
        self,
        sigma: float,
        *,
        rng: np.random.Generator | None = ...,
        inplace: Literal[True],
        wrap: bool = ...,
        degrees: bool = ...,
    ) -> None: ...
    @overload
    def add_rotation_noise(
        self,
        sigma: float,
        rng: np.random.Generator | None = ...,
        inplace: Literal[False] = ...,
        wrap: bool = ...,
        degrees: bool = ...,
    ) -> Bvh: ...
    def add_rotation_noise(
        self,
        sigma: float,
        rng: np.random.Generator | None = None,
        inplace: bool = False,
        wrap: bool = False,
        degrees: bool = False,
    ) -> Bvh | None:
        """Add Gaussian noise (``sigma`` in radians, or degrees with ``degrees=True``) to joint angles.  See :func:`pybvh.transforms.add_rotation_noise`."""
        from . import transforms

        return transforms.add_rotation_noise(
            self, sigma, rng=rng, inplace=inplace, wrap=wrap, degrees=degrees
        )  # type: ignore[call-overload, return-value]

    @overload
    def add_position_noise(
        self, sigma: float, *, rng: np.random.Generator | None = ..., inplace: Literal[True]
    ) -> None: ...
    @overload
    def add_position_noise(
        self, sigma: float, rng: np.random.Generator | None = ..., inplace: Literal[False] = ...
    ) -> Bvh: ...
    def add_position_noise(
        self, sigma: float, rng: np.random.Generator | None = None, inplace: bool = False
    ) -> Bvh | None:
        """Add Gaussian noise (``sigma`` in the skeleton's length unit) to the root translation.  See :func:`pybvh.transforms.add_position_noise`."""
        from . import transforms

        return transforms.add_position_noise(self, sigma, rng=rng, inplace=inplace)  # type: ignore[call-overload, return-value]

    def perturb_speed(self, factor: float) -> Bvh:
        """Change motion speed by resampling.  See :func:`pybvh.transforms.perturb_speed`."""
        from . import transforms

        return transforms.perturb_speed(self, factor)

    @overload
    def drop_frames(
        self, drop_rate: float, *, rng: np.random.Generator | None = ..., inplace: Literal[True]
    ) -> None: ...
    @overload
    def drop_frames(
        self, drop_rate: float, rng: np.random.Generator | None = ..., inplace: Literal[False] = ...
    ) -> Bvh: ...
    def drop_frames(
        self, drop_rate: float, rng: np.random.Generator | None = None, inplace: bool = False
    ) -> Bvh | None:
        """Replace dropped frames with SLERP interpolation.  See :func:`pybvh.transforms.drop_frames`."""
        from . import transforms

        return transforms.drop_frames(self, drop_rate, rng=rng, inplace=inplace)  # type: ignore[call-overload, return-value]

    @overload
    def rotate_vertical(
        self,
        angle: float,
        *,
        world_up: str | None = ...,
        degrees: bool = ...,
        pivot: str | npt.ArrayLike = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def rotate_vertical(
        self,
        angle: float,
        world_up: str | None = ...,
        degrees: bool = ...,
        pivot: str | npt.ArrayLike = ...,
        inplace: Literal[False] = ...,
    ) -> Bvh: ...
    def rotate_vertical(
        self,
        angle: float,
        world_up: str | None = None,
        degrees: bool = False,
        pivot: str | npt.ArrayLike = "origin",
        inplace: bool = False,
    ) -> Bvh | None:
        """Rotate entire motion around the vertical axis (``angle`` in radians), about the world origin or ``pivot=``.  See :func:`pybvh.transforms.rotate_vertical`."""
        from . import transforms

        return transforms.rotate_vertical(
            self, angle, world_up=world_up, degrees=degrees, pivot=pivot, inplace=inplace
        )  # type: ignore[call-overload, return-value]

    @overload
    def mirror(
        self,
        *,
        lr_mapping: dict[str, str] | None = ...,
        lateral: str | None = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def mirror(
        self,
        lr_mapping: dict[str, str] | None = ...,
        lateral: str | None = ...,
        inplace: Literal[False] = ...,
    ) -> Bvh: ...
    def mirror(
        self,
        lr_mapping: dict[str, str] | None = None,
        lateral: str | None = None,
        inplace: bool = False,
    ) -> Bvh | None:
        """Mirror motion across the lateral plane.  See :func:`pybvh.transforms.mirror`."""
        from . import transforms

        return transforms.mirror(self, lr_mapping=lr_mapping, lateral=lateral, inplace=inplace)  # type: ignore[call-overload, return-value]

    def random_translate_root(
        self,
        offset_range: tuple[float, float] = (-100.0, 100.0),
        rng: np.random.Generator | None = None,
    ) -> Bvh:
        """Translate root by a random offset.  See :func:`pybvh.transforms.random_translate_root`."""
        from . import transforms

        return transforms.random_translate_root(self, offset_range=offset_range, rng=rng)

    def random_rotate_vertical(
        self,
        angle_range: tuple[float, float] = (-np.pi, np.pi),
        world_up: str | None = None,
        degrees: bool = False,
        pivot: str | npt.ArrayLike = "origin",
        rng: np.random.Generator | None = None,
    ) -> Bvh:
        """Rotate motion by a random angle around the vertical axis (radians).  See :func:`pybvh.transforms.random_rotate_vertical`."""
        from . import transforms

        return transforms.random_rotate_vertical(
            self, angle_range=angle_range, world_up=world_up, degrees=degrees, pivot=pivot, rng=rng
        )

    def random_perturb_speed(
        self, factor_range: tuple[float, float] = (0.8, 1.2), rng: np.random.Generator | None = None
    ) -> Bvh:
        """Apply a random speed change.  See :func:`pybvh.transforms.random_perturb_speed`."""
        from . import transforms

        return transforms.random_perturb_speed(self, factor_range=factor_range, rng=rng)

    # ----------------------------------------------------------------
    #  Coordinate-frame reorientation
    # ----------------------------------------------------------------

    @overload
    def reorient_world_up(self, new_up: str, *, inplace: Literal[True]) -> None: ...
    @overload
    def reorient_world_up(self, new_up: str, inplace: Literal[False] = ...) -> Bvh: ...
    def reorient_world_up(self, new_up: str, inplace: bool = False) -> Bvh | None:
        """Change the world coordinate system's up axis.  See :func:`pybvh.transforms.reorient_world_up`."""
        from . import transforms

        return transforms.reorient_world_up(self, new_up, inplace=inplace)  # type: ignore[call-overload,return-value]

    @overload
    def reorient_rest_up(self, new_up: str, *, inplace: Literal[True]) -> None: ...
    @overload
    def reorient_rest_up(self, new_up: str, inplace: Literal[False] = ...) -> Bvh: ...
    def reorient_rest_up(self, new_up: str, inplace: bool = False) -> Bvh | None:
        """Reorient rest-pose up axis without changing FK positions.  See :func:`pybvh.transforms.reorient_rest_up`."""
        from . import transforms

        return transforms.reorient_rest_up(self, new_up, inplace=inplace)  # type: ignore[call-overload,return-value]

    @overload
    def reorient_rest_forward(self, new_forward: str, *, inplace: Literal[True]) -> None: ...
    @overload
    def reorient_rest_forward(self, new_forward: str, inplace: Literal[False] = ...) -> Bvh: ...
    def reorient_rest_forward(self, new_forward: str, inplace: bool = False) -> Bvh | None:
        """Reorient rest-pose forward direction without changing FK positions.  See :func:`pybvh.transforms.reorient_rest_forward`."""
        from . import transforms

        return transforms.reorient_rest_forward(self, new_forward, inplace=inplace)  # type: ignore[call-overload,return-value]

    # ----------------------------------------------------------------
    #  Visualization (delegate to bvhplot module)
    # ----------------------------------------------------------------

    def plot_rest_pose(self, **kwargs):
        """Plot the rest pose. See :func:`pybvh.bvhplot.rest_pose`."""
        from . import bvhplot

        return bvhplot.rest_pose(self, **kwargs)

    def plot_frame(self, frame=0, **kwargs):
        """Plot a single frame. See :func:`pybvh.bvhplot.frame`."""
        from . import bvhplot

        return bvhplot.frame(self, frame=frame, **kwargs)

    def plot_sequence(self, **kwargs):
        """Sequence still (sampled poses, lighter = past).
        See :func:`pybvh.bvhplot.sequence`."""
        from . import bvhplot

        return bvhplot.sequence(self, **kwargs)

    def plot_trajectory(self, **kwargs):
        """Plot the root trajectory. See :func:`pybvh.bvhplot.trajectory`."""
        from . import bvhplot

        return bvhplot.trajectory(self, **kwargs)

    def render(self, filepath: str | Path = Path("./anim.mp4"), **kwargs):
        """Render animation to file. See :func:`pybvh.bvhplot.render`."""
        from . import bvhplot

        return bvhplot.render(self, filepath, **kwargs)

    def play(self, **kwargs):
        """Interactive playback. See :func:`pybvh.bvhplot.play`."""
        from . import bvhplot

        return bvhplot.play(self, **kwargs)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ----------------------------- end of BVH class-----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
