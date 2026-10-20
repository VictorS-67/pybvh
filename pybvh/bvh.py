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
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from . import geometry

import numpy as np
import numpy.typing as npt

from . import rotations
from ._derived import DerivedState as _DerivedState
from ._orientation import (
    OrientationReport,
    _build_orientation_report,
    _detect_lr_mapping_by_names,
    _facing_basis,
    _facing_labels,
    _infer_rest_left,
    _infer_world_up,
    _iter_unique_lr_pairs,
    _resolve_lr_pairs,
    _resolve_node_lr_pairs,
    _warn_facing_defaulted,
    _warn_rest_forward_defaulted,
    _warn_world_up_defaulted,
    _warn_world_up_disagreement,
    _world_up_disagrees,
    _WorldUpGuess,
)
from ._warnings import user_stacklevel

# Re-exported: pybvh.bvh has long served every node class, end sites included.
from .bvhnode import BvhEndSite as BvhEndSite
from .bvhnode import BvhJoint, BvhNode, BvhRoot, _is_joint
from .node_tree import (
    _check_node_list,
    _check_node_tree,
    _nodes_from_table_or_list,
    nodes_from_table,
    nodes_to_table,
)
from .spatial_coord import (
    FkTopology,
    _ground_plane_offset,
    frames_to_node_positions,
)
from .tools import (
    Axis,
    _axis_to_vector,
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


def _motion_column_names(nodes: Sequence[BvhNode], mode: str) -> list[str]:
    """Column names of the motion *nodes* carry, in either ``to_df_dict`` mode.

    ``'euler'`` names the flat layout of ``root_pos`` and ``joint_angles``:
    ``<root>_<axis>_pos`` per position channel, then ``<joint>_<axis>_rot``
    per rotation channel for each joint in node order, end sites having
    none. ``'coordinates'`` names ``<node>_<axis>`` for X, Y and Z per
    node, end sites included. ``time`` is not a motion column and is not
    listed. Node names are made unique with :func:`_unique_labels` over
    the nodes the mode lists, before the axis and kind are appended, so
    the same joint can be ``Arm`` in one mode and ``Arm.1`` in the other
    when an earlier end site shares its name.
    """
    if mode == "euler":
        joints = [node for node in nodes if isinstance(node, BvhJoint)]
        root = joints[0]
        if not isinstance(root, BvhRoot):
            raise ValueError(f"nodes[0] must be the BvhRoot, got {type(root).__name__}")
        labels = _unique_labels([joint.name for joint in joints])
        names = [f"{labels[0]}_{ax}_pos" for ax in root.pos_channels]
        for label, joint in zip(labels, joints):
            names.extend(f"{label}_{ax}_rot" for ax in joint.rot_channels)
        return names
    if mode == "coordinates":
        labels = _unique_labels([node.name for node in nodes])
        return [f"{label}_{ax}" for label in labels for ax in ("X", "Y", "Z")]
    raise ValueError(f"mode must be 'euler' or 'coordinates', got {mode!r}")


def _checked_root_pos(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``value`` as a float array, if it has the shape of ``root_pos``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (F, 3), got {arr.shape}")
    return arr


def _checked_joint_angles(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``value`` as a float array, if it has the shape of ``joint_angles``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"joint_angles must have shape (F, J, 3), got {arr.shape}")
    return arr


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
    """FK of ``bvh``'s skeleton with every angle zero and the root at the origin."""
    return frames_to_node_positions(
        bvh,
        root_pos=np.zeros(3),
        joint_angles=np.zeros((bvh.joint_count, 3), dtype=np.float64),
        centered="skeleton",
    )


class Bvh:
    """Container for BVH motion-capture data.

    The hierarchy is stored as a tuple of ``BvhNode`` objects (one per
    joint / end-site).  Motion data is stored as two structured arrays:

    - ``root_pos``:     shape ``(F, 3)``    — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)`` — Euler angles **in radians** per joint per frame

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

        # Validate that root position channels are standard XYZ
        pos_order = "".join(self.root.pos_channels)
        if pos_order != "XYZ":
            raise ValueError(
                f"Non-standard root position channel order {pos_order!r} is "
                f"not supported. Expected 'XYZ', the order of every root_pos row."
            )

        # ---------- Determine root_pos / joint_angles ----------
        if (root_pos is None) != (joint_angles is None):
            missing = "joint_angles" if joint_angles is None else "root_pos"
            raise ValueError(
                f"root_pos and joint_angles must be provided together; {missing} is missing."
            )
        if root_pos is not None:
            self.root_pos = np.asarray(root_pos, dtype=np.float64)
            self.joint_angles = np.asarray(joint_angles, dtype=np.float64)
            self._check_motion_fits_skeleton()
        else:
            # Empty object
            self.root_pos = np.empty((0, 3), dtype=np.float64)
            self.joint_angles = np.empty((0, 0, 3), dtype=np.float64)

        if world_up != "auto":
            self._world_up_override = _validate_axis_string(world_up)
        if lr_mapping is not None:
            self._validate_and_set_lr_mapping(lr_mapping)

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
        if self._root_pos.shape[0] != self._joint_angles.shape[0]:
            raise ValueError(
                f"root_pos and joint_angles disagree on frame count: "
                f"root_pos has {self._root_pos.shape[0]} frames, "
                f"joint_angles has {self._joint_angles.shape[0]}."
            )
        joint_count = sum(1 for n in nodes if not n.is_end_site())
        if self._joint_angles.shape[1] != joint_count:
            raise ValueError(
                f"joint_angles has {self._joint_angles.shape[1]} joints "
                f"on axis 1, but the skeleton has {joint_count} "
                f"non-end-site joints."
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

        A value of ``0`` means "unset" and is the default for newly
        constructed empty :class:`Bvh` objects. Writing to a file
        requires a positive value — :func:`~pybvh.io.write_bvh_file`
        raises ``ValueError`` otherwise. Assigning a negative number,
        NaN or infinity raises ``ValueError``.
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
        Assigning sets ``frame_time`` to ``1 / fps``; the rate must be a
        positive finite number, so ``0``, a negative rate, NaN and
        infinity raise ``ValueError`` rather than mark the rate as unset.

        Examples
        --------
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
        """
        view = self._root_pos.view()
        view.flags.writeable = False
        return view

    @root_pos.setter
    def root_pos(self, value: npt.ArrayLike) -> None:
        self._root_pos = _checked_root_pos(value)
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
        """
        view = self._joint_angles.view()
        view.flags.writeable = False
        return view

    @joint_angles.setter
    def joint_angles(self, value: npt.ArrayLike) -> None:
        self._joint_angles = _checked_joint_angles(value)
        self._derived.user_replaced_motion()

    def _set_motion(
        self,
        *,
        root_pos: npt.ArrayLike | None = None,
        joint_angles: npt.ArrayLike | None = None,
    ) -> None:
        """Write motion as pybvh's own edits do, through no public setter.

        The transforms, ``a += b`` and the rebuild of a derived clip write
        here and report that the motion changed. The public setters report
        that the user replaced the motion, the one change that starts the
        world-up guess again (ADR 0003).
        """
        if root_pos is not None:
            self._root_pos = _checked_root_pos(root_pos)
        if joint_angles is not None:
            self._joint_angles = _checked_joint_angles(joint_angles)
        self._derived.motion_changed()

    @property
    def frame_count(self) -> int:
        """Number of frames (computed from root_pos)."""
        return len(self.root_pos)

    @property
    def root(self) -> BvhRoot:
        """The root joint, ``nodes[0]``.

        The one node with position channels: its translation per frame is
        :attr:`root_pos`, and its rotation is joint 0 of
        :attr:`joint_angles`. Read-only: the root is whatever ``nodes``
        starts with.
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
        :meth:`matches_hierarchy` with ``atol=0``, channel layout via
        :meth:`matches_channels`. ``source_path`` is ignored.
        """
        if not isinstance(other, Bvh):
            return NotImplemented
        if not self.matches_hierarchy(other, atol=0):
            return False
        if not self.matches_channels(other):
            return False
        if self.frame_time != other.frame_time:
            return False
        if not np.array_equal(self.root_pos, other.root_pos):
            return False
        if not np.array_equal(self.joint_angles, other.joint_angles):
            return False
        return True

    def matches_hierarchy(self, other: Bvh, match_offsets: bool = True, atol: float = 1e-6) -> bool:
        """Whether ``self`` and ``other`` share the same skeleton hierarchy.

        Hierarchy is defined as: same node names in topological order
        (including end sites), same parent-child structure, and — when
        ``match_offsets=True`` (default) — same rest-pose offsets within
        ``atol``. Motion data, Euler rotation orders, and frame timing
        are NOT compared.

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
            Absolute tolerance for offset comparison (default ``1e-6``).
            Ignored when ``match_offsets=False``.

        Returns
        -------
        bool

        See Also
        --------
        matches_channels : Compare per-joint Euler rotation orders.
        matches_topology : Conjunction of hierarchy + channels.
        """
        if not isinstance(other, Bvh):
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1.name != n2.name:
                return False
            parent1 = n1.parent.name if n1.parent is not None else None
            parent2 = n2.parent.name if n2.parent is not None else None
            if parent1 != parent2:
                return False
            if match_offsets and not np.allclose(n1.offset, n2.offset, atol=atol):
                return False
        return True

    def matches_channels(self, other: Bvh) -> bool:
        """Whether ``self`` and ``other`` share the same channel layout.

        Compares per-joint Euler rotation orders and the root's
        position-channel order. This is a *serialization* property:
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
        if self.root.pos_channels != other.root.pos_channels:
            return False
        return self.euler_orders == other.euler_orders

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
        matches_channels : The channel-layout half (Euler orders, root pos channels).
        """
        return self.matches_hierarchy(other) and self.matches_channels(other)

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
        Raises ``ValueError`` if the skeletons are incompatible and warns
        on ``frame_time`` mismatch (``self``'s frame time wins).
        """
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        return self._concat(other)

    def __iadd__(self, other: object) -> Bvh:
        """In-place concatenation. Grows ``self`` by appending ``other``'s frames.

        Validates skeleton compatibility and warns on ``frame_time`` mismatch
        just like ``a + b``. The values derived from the motion are
        recomputed on their next read.
        """
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        self._check_same_skeleton(other)
        if self.frame_time != other.frame_time:
            warnings.warn(
                f"Frame time mismatch: {self.frame_time} vs "
                f"{other.frame_time}. Using self's frame time.",
                stacklevel=user_stacklevel(),
            )
        self._set_motion(
            root_pos=np.concatenate([self.root_pos, other.root_pos], axis=0),
            joint_angles=np.concatenate([self.joint_angles, other.joint_angles], axis=0),
        )
        if self.source_path != other.source_path:
            self.source_path = None
        return self

    def __setitem__(self, key: int | slice, value: Bvh) -> None:
        """Splice frames from another Bvh into ``self`` in place.

        The skeleton of ``value`` must match ``self``, ``value.frame_time``
        must equal ``self.frame_time`` (raises ``ValueError`` otherwise —
        use :meth:`resample` first if they differ), and the slice length
        must equal ``value.frame_count``. Integer keys require
        ``value.frame_count == 1``.

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
        if self.frame_time != value.frame_time:
            raise ValueError(
                f"frame_time mismatch (self={self.frame_time}, value="
                f"{value.frame_time}). Call value.resample(1/self.frame_time) "
                "first, or overwrite self.frame_time explicitly before splicing."
            )

        # --- length match (no resizing) ---
        target_len = len(self.root_pos[s])
        if target_len != value.frame_count:
            raise ValueError(
                f"Cannot assign {value.frame_count} frames to a slice of "
                f"length {target_len}; __setitem__ does not resize. Use "
                "`a + b` to append, or frame slicing (`bvh[a:b]`) + manual "
                "array assignment for more complex splicing."
            )

        # --- in-place splice, bypassing the setters ---
        self._root_pos[s] = value.root_pos
        self._joint_angles[s] = value.joint_angles
        self._derived.user_replaced_motion()

    def copy(self, nodes: Sequence[BvhNode] | Sequence[Mapping[str, Any]] | None = None) -> Bvh:
        """Return an independent copy, on this skeleton or on an edited one.

        The node hierarchy, ``root_pos`` and ``joint_angles``,
        ``frame_time``, ``source_path``, a manual ``world_up``, the warn
        preference and a user-set L/R mapping are copied. Editing either
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
            them). A user-set L/R mapping follows its joints by position,
            under their new names, so each mapped joint's new name must be
            one no other joint carries (the mapping goes by name); one
            detected from the names is detected again from the new names.
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
            order, or gives the root position channels other than XYZ; or
            if it gives a joint of a user-set L/R mapping a name another
            joint carries too (on a clip without motion, which binds no
            joint, if it lacks a name the mapping uses).
        TypeError
            If ``nodes`` is neither a node table nor a node list.

        Examples
        --------
        >>> table = bvh.to_node_table()
        >>> table[1]["offset"] = [0.0, 12.0, 0.0]
        >>> edited = bvh.copy(nodes=table)
        """
        if nodes is None:
            return self._derive_clip(
                root_pos=self._root_pos.copy(), joint_angles=self._joint_angles.copy()
            )
        new_nodes = _nodes_from_table_or_list(nodes, "nodes")
        # Every check runs before the copy is built, as in the constructor:
        # building it freezes the nodes, and a copy that raises must leave
        # a node list it was given editable, as a Bvh(...) that raises does.
        joints = [node for node in self.nodes if _is_joint(node)]
        new_joints = [node for node in new_nodes if _is_joint(node)]
        # A clip built without motion holds (0, 0, 3) angles, which bind no
        # joint: it takes any skeleton.
        has_motion = self._joint_angles.shape[:2] != (0, 0)
        if has_motion:
            self._check_motion_fits_skeleton(new_nodes)
            for joint, new_joint in zip(joints, new_joints, strict=True):
                old_order = "".join(joint.rot_channels)
                new_order = "".join(new_joint.rot_channels)
                if new_order != old_order:
                    raise ValueError(
                        f"nodes changes the Euler order of joint {joint.name!r} "
                        f"from {old_order!r} to {new_order!r}, but the clip's "
                        f"angles are in {old_order!r}. To change an order, "
                        f"convert the angles with it: "
                        f"bvh.change_euler_order({new_order!r}, joint={joint.name!r})."
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
        edited = self._derive_clip(
            root_pos=self._root_pos.copy(),
            joint_angles=self._joint_angles.copy(),
            nodes=new_nodes,
        )
        if isinstance(setting, dict):
            edited._set_lr_mapping_setting(setting)
        return edited

    def _derive_clip(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_angles: npt.NDArray[np.float64],
        nodes: list[BvhNode] | tuple[BvhNode, ...] | None = None,
    ) -> Bvh:
        """Build a clip from new primary state and this clip's settings.

        The one way pybvh derives a clip (ADR 0003): the result carries what
        :meth:`__getstate__` lists, with the given motion and skeleton
        (default: a deep copy of this one's), and none of this clip's
        derived state. The arrays are taken as they are, not copied; the
        caller hands over arrays no other clip holds.
        """
        state = self.__getstate__()
        state["nodes"] = copy.deepcopy(self.nodes) if nodes is None else nodes
        state["root_pos"] = root_pos
        state["joint_angles"] = joint_angles
        setting = state["lr_mapping_setting"]
        if isinstance(setting, dict):
            state["lr_mapping_setting"] = dict(setting)
        new_bvh = type(self).__new__(type(self))
        new_bvh.__setstate__(state)
        return new_bvh

    def __getstate__(self) -> dict[str, Any]:
        """What a pickle and every derived clip carry (ADR 0003).

        The primary state (skeleton, motion, frame time, source path), the
        user's settings (world-up override, warn preference, L/R choice)
        and the world up this clip guessed, which the rebuilt clip
        inherits; never a derived value, which the rebuilt clip computes
        again on first read.
        """
        self._settle_world_up()
        derived = self._derived
        guess = derived.inherited_world_up or derived.inferred_world_up
        return {
            "nodes": self.nodes,
            "root_pos": self._root_pos,
            "joint_angles": self._joint_angles,
            "frame_time": self.frame_time,
            "source_path": self.source_path,
            "world_up_override": self._world_up_override,
            # The guess's source travels with it, so the rebuilt clip
            # reports how the axis it inherits was obtained. A plain
            # (axis, source) pair: a pickle names no private class.
            "world_up_guess": None if guess is None else tuple(guess),
            "warn_on_world_up_disagreement": self._warn_on_world_up_disagreement,
            "lr_mapping_setting": self._lr_mapping_setting,
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
        self._set_motion(root_pos=state["root_pos"], joint_angles=state["joint_angles"])
        # A clip built without motion holds (0, 0, 3) angles whatever its
        # skeleton; there is no motion to fit, as in the constructor.
        if self._joint_angles.shape[:2] != (0, 0):
            self._check_motion_fits_skeleton()
        self._world_up_override = state["world_up_override"]
        self._set_lr_mapping_setting(state["lr_mapping_setting"])
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
        if self.frame_count > 0 and len(self.nodes) > 1:
            _ = self.world_up

    def _rotate_world_up_guess(self, rotation: npt.NDArray[np.float64]) -> None:
        """Turn the guessed world up, inferred or inherited, by an
        axis-aligned ``rotation`` of the whole scene.

        ``reorient_world_up`` rotates every frame and offset by
        ``rotation``, so the axis guessed before is, exactly, the axis
        those frames would be guessed to have now; turning it keeps the
        guess made once (ADR 0003, decision 3). A clip that has guessed
        nothing is left to guess from its new frames.
        """
        derived = self._derived
        for name in ("inferred_world_up", "inherited_world_up"):
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

    def _replace_offsets(self, offsets: npt.ArrayLike) -> None:
        """Replace every node's offset, given in ``nodes`` order as ``(N, 3)``."""
        table = nodes_to_table(self.nodes)
        for row, offset in zip(table, np.asarray(offsets, dtype=np.float64), strict=True):
            row["offset"] = offset
        self._replace_skeleton(nodes_from_table(table))

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
        head-above-hips direction, with rest-pose topology as fallback.
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
          when you assign :attr:`root_pos`, :attr:`joint_angles` or frames
          (``bvh[i] = other``), the one change that replaces the
          recording, and never by an edit pybvh makes.
        - **Inherited**: a clip pybvh derives from this one reports the
          axis guessed here, without guessing from its own frames:
          ``copy()``, slicing, ``a + b`` and ``a += b`` (from ``a``),
          :meth:`resample`, the transforms, :meth:`retarget`,
          :meth:`extract_joints` and a pickle. Cutting, resampling or
          transforming a clip does not change which way is up, though a
          slice whose first frame is a crouch or a fall would be guessed
          differently on its own. Assigning motion to the derived clip
          guesses again, as for an inferred axis.

        :attr:`world_up_inferred` always guesses from this clip's own
        motion: where it differs from ``world_up`` and you set no axis,
        the axis was inherited.

        The detection specifics, since they decide which answer you get:
        "head" is the node :attr:`rest_up` reads, the first node in node
        order named ``head``, else ``neck``, ``chest`` or ``spine`` (a
        whole-name match in any case, so a namespaced ``mixamorig:Head``
        does not match), skipping one that sits at the root in the rest
        pose; "hips" is always the root node whatever it is called; and
        the frame-0 reading is accepted only when its largest component
        exceeds twice the second-largest
        — a crouched, lying or leaning first frame is treated as
        ambiguous and silently defers to :attr:`rest_up`. When the rest
        pose is degenerate too, the property warns and returns ``'+y'``.
        Use :attr:`world_up_inferred` to see what the heuristic picks
        while an override is in effect.

        Can be overridden manually via the setter when auto-detection
        produces the wrong answer (e.g. authored BVH files where the rest
        pose convention differs from the animation's world orientation):

            >>> bvh.world_up = '+y'

        Assign ``'auto'`` or ``None`` to clear a previous override and
        return to the guessed axis, inferred or inherited.

        Note: BVH files do not store a world-up field, so manual overrides
        are lost on write→read round trips and must be re-applied.
        """
        report = self.orientation_report
        derived = self._derived
        # A guess of this clip's own warns on the first read of the axis it
        # gave, once; an inherited guess warned on the clip it was made on.
        guessed_here = report.world_up_source != "set" and not report.world_up_inherited
        if guessed_here and not derived.world_up_guess_announced:
            guess = derived.inferred_world_up
            assert guess is not None  # the report read world up from it
            self._warn_about_world_up_guess(guess, report.rest_up)
            derived.world_up_guess_announced = True
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

            >>> bvh.world_up_inferred  # '+y'  (auto's guess)
            >>> bvh.world_up           # '+z'  (user override)
        """
        guess = self._guess_world_up()
        self._warn_about_world_up_guess(guess, self.rest_up)
        return guess.axis

    def _guess_world_up(self) -> _WorldUpGuess:
        """Guess world up from this clip's own first frame and rest pose."""
        frame0 = self.node_positions(frame=0) if self.frame_count > 0 else None
        return _infer_world_up(self._node_names(), self._rest_pose(), frame0)

    def _warn_about_world_up_guess(self, guess: _WorldUpGuess, rest_up: str | None) -> None:
        """Warn when *guess* is the house default, or when it disagrees with
        *rest_up* and the clip was not loaded with
        ``warn_on_world_up_disagreement=False``."""
        if guess.source == "default":
            _warn_world_up_defaulted()
        elif self._warn_on_world_up_disagreement and _world_up_disagrees(guess, rest_up):
            assert rest_up is not None  # a disagreement needs a rest up
            _warn_world_up_disagreement(rest_up, guess.axis)

    @property
    def world_up_axis(self) -> Axis:
        """:attr:`world_up` parsed into numeric form — ``Axis(index, sign, vector)``.

        The three fields are the machine-usable views of the same signed axis string (always derived from the resolved :attr:`world_up`, so manual overrides are respected):

        - ``index`` (int): the up coordinate's column (0 = x, 1 = y, 2 = z) in any ``(..., 3)`` position array.
        - ``sign`` (float): ``+1.0`` or ``-1.0`` — multiply the raw coordinate by it to get an up-positive height.
        - ``vector`` (ndarray, shape ``(3,)``): the unit vector along the up direction, sign included (e.g. ``[0, -1, 0]`` for ``'-y'``). A fresh array on every access — mutating it never corrupts the Bvh.

        Example — up-positive heights of every node in every frame:

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
        transient glitches, not systematic ones. Pass an explicit ``floor=`` to
        the contact detectors if that ever bites.

        This is **not** the reference :func:`~pybvh.analysis.foot_contacts`
        compares foot clearance against. That one is estimated per call, over
        the joints being tested, and is returned as ``info["floor"]``. The two
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
        :attr:`lr_mapping` is assigned, or the motion is assigned
        (:attr:`root_pos`, :attr:`joint_angles`, ``bvh[i] = other``),
        which guesses world up again.

        Examples
        --------
        >>> report = bvh.orientation_report
        >>> if report.rest_forward_source == "default":
        ...     ...  # no usable L/R geometry: rest_forward is a house default
        >>> report.world_up, report.world_up_source
        ('+y', 'first_frame')
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
        world_up_source: Literal["set", "first_frame", "rest_up", "default"]
        if self._world_up_override is not None:
            world_up, world_up_source = self._world_up_override, "set"
        elif derived.inherited_world_up is not None:
            world_up, world_up_source = derived.inherited_world_up
            world_up_inherited = True
        else:
            # Guessed silently: the guess's warnings are world_up's to give
            # when the axis is read, not this report's.
            if derived.inferred_world_up is None:
                derived.inferred_world_up = self._guess_world_up()
            world_up, world_up_source = derived.inferred_world_up
        return _build_orientation_report(
            self._node_names(),
            self._rest_pose(),
            self._facing_lr_pairs(),
            world_up,
            world_up_source,
            world_up_inherited,
            self._lr_mapping_source(),
        )

    def _lr_mapping_source(self) -> Literal["set", "names"] | None:
        """Where the L/R mapping in effect comes from, ``None`` without one."""
        setting = self._lr_mapping_setting
        if isinstance(setting, dict):
            return "set"
        return "names" if setting == "names" and self._lr_mapping is not None else None

    @property
    def rest_up(self) -> str | None:
        """The skeleton's own up axis, read from the rest pose alone.

        Read-only. The main axis from the root to the first node, in
        node order, named ``head``, else ``neck``, ``chest`` or
        ``spine`` (whole name, any case), skipping one that sits at the
        root; without such a node, the axis along which the rest pose
        spreads most (:attr:`orientation_report` says which, in
        ``rest_up_source``). The animation is never read.
        :attr:`world_up`'s guess reads the same node.

        Contrast with :attr:`world_up`, which is *animation*-derived
        (inferred from the first frame's head-above-hips direction). On
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
        return self.orientation_report.rest_up

    @property
    def rest_forward(self) -> str:
        """The direction the rest pose faces, in the skeleton's own frame.

        Read-only. ``rest left × rest up``, where rest left is the rest
        pose's L/R geometry (the pairs of :attr:`lr_mapping`) measured
        against :attr:`rest_up`. Read from the rest pose alone, like
        :attr:`rest_up`: never from :attr:`world_up` or the animation.
        The rest pose is the skeleton with every rotation zero, so this
        is a direction in the skeleton's frame, not in the world; the
        root rotation of a frame carries it into the world, which is
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
        report = self.orientation_report
        if report.rest_forward_source == "default":
            _warn_rest_forward_defaulted(report.rest_forward, report.rest_up)
        return report.rest_forward

    def _rest_left_of(self, mapping: dict[str, str] | None) -> str | None:
        """The rest pose's left for the pairs of *mapping*, against rest up.

        ``None`` without a usable one. :func:`~pybvh.transforms.mirror`
        reads it for the mapping it mirrors with, which may be another
        than the clip's own; :attr:`orientation_report` holds the rest
        left of the clip's own mapping.
        """
        pairs = _resolve_lr_pairs(mapping, self.node_index)
        return _infer_rest_left(self._rest_pose(), pairs, self.rest_up)

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

        Detection, run on first use, applies the extended name heuristic
        (`Left`/`Right` substring, `L`/`R` prefix, `.L`/`.R` suffix,
        `_l`/`_r` suffix, Mixamo `mixamorig:` namespace, numbered `.001`
        duplicates). Skeletons with conventions the heuristic can't parse
        have ``lr_mapping = None`` — in that case, set it explicitly:

            >>> bvh.lr_mapping = {'arm.L': 'arm.R', 'leg.L': 'leg.R'}

        The assigned dict is one-directional; pybvh symmetrizes it
        internally. Either form works on assignment.

        or pass ``lr_mapping=`` at load time:

            >>> bvh = read_bvh_file('weird.bvh', lr_mapping={...})

        Consumers: ``mirror()``, ``orientation_report`` (and so
        ``rest_forward`` and ``has_lr_geometry``),
        ``forward_at()``, ``left_at()``, ``facing_frame()``.

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
            >>> if bvh.has_lr_geometry:
            ...     assert bvh.rest_forward == dataset_convention
            ... else:
            ...     ...  # facing is a default, not a measurement

        See Also
        --------
        rest_forward : The axis whose fallback this reports.
        lr_mapping : The pairs the measurement is derived from.
        """
        return self.orientation_report.rest_forward_source == "lr_geometry"

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

    def _facing_lr_pairs(self) -> npt.NDArray[np.intp]:
        """The L/R pairs the rest left and the facing read, in ``nodes`` index space.

        Joint pairs only, as :attr:`lr_mapping` names them, resolved into
        ``nodes`` index space so they index :meth:`node_positions` output.
        End sites are not included: that is :attr:`node_lr_pairs`' job,
        for mirroring positions, not for reading the facing. An ``(P, 2)``
        int array, ``(0, 2)`` without an L/R mapping, so array code needs
        no ``None`` branch. bvhplot stores it on its ``SkeletonView``, so
        the follow camera needs no ``Bvh``.
        """
        pairs = _resolve_lr_pairs(self._lr_mapping, self.node_index)
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)

    def _validate_and_set_lr_mapping(self, mapping: dict[str, str]) -> None:
        """Validate the user's lr_mapping dict and make it the mapping in effect."""
        self._set_lr_mapping_setting(_validated_lr_mapping(mapping, self.joint_names))

    def forward_at(
        self,
        frame: int = 0,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> str:
        """Character's world-space forward (facing) direction at a given frame.

        Computed from actual joint positions at the given frame — the
        leftward axis is derived by averaging (left − right) across
        matching L/R joint pairs in world space, then crossed with
        ``world_up`` to produce the forward direction (forward =
        leftward × up). This tracks the character's actual facing as
        they rotate through the animation.

        This is the snapped classification — the continuous facing
        vector is quantized to the nearest of the six signed world
        axes. See :meth:`facing_frame` for the continuous per-frame
        basis as unit vectors, and for the fallback when the frame's
        left/right axis cannot be read (the root's heading at that
        frame, then a house default, which warns).

        Parameters
        ----------
        frame : int, optional
            Frame index (default 0). Must be within the animation range.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates for all frames. When
            provided, skips the per-call forward kinematics — useful for
            computing facing direction across many frames in a hot loop
            (e.g. dataset uniformity diagnostics). The selected frame's
            slice is taken via ``coords[frame]``; the fallback reads the
            clip's root rotation at ``frame``.

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
        return self._facing_labels_at(frame, coords)[0]

    def left_at(
        self,
        frame: int = 0,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> str:
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
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates for all frames. When
            provided, skips the per-call forward kinematics. The selected
            frame's slice is taken via ``coords[frame]``; the fallback
            reads the clip's root rotation at ``frame``.

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
        return self._facing_labels_at(frame, coords)[1]

    def _facing_labels_at(
        self, frame: int, coords: npt.NDArray[np.float64] | None
    ) -> tuple[str, str]:
        """The snapped ``(forward, left)`` labels at clip frame *frame*."""
        if coords is None:
            frame_coords = self.node_positions(frame=frame)
        else:
            frame_coords = coords[frame]
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
        when rest forward is measured, the root's heading at *frames*),
        runs it, and warns when a row carries the house default. With
        *frames* ``None`` the rows are a pose with no clip frame, and the
        heading stage is skipped. Returns ``(forward, left, up, valid)``.
        """
        world_up = self.world_up
        report = self.orientation_report
        lr_pairs = self._facing_lr_pairs()
        root_rotmats = rest_forward = None
        if frames is not None and lr_pairs.shape[0] > 0:
            if report.rest_forward_source == "lr_geometry":
                rest_forward = report.rest_forward
                root_rotmats = self._root_rotmats(frames)
        forward, left, up, valid, defaulted = _facing_basis(
            coords, lr_pairs, world_up, root_rotmats, rest_forward
        )
        if defaulted.any():
            if lr_pairs.shape[0] == 0:
                named: list[int] | None = []
            elif frames is None:
                named = None
            else:
                named = np.arange(self.frame_count)[frames][defaulted].tolist()
            _warn_facing_defaulted(world_up, named)
        return forward, left, up, valid

    def _root_rotmats(self, frames: Sequence[int] | slice = slice(None)) -> npt.NDArray[np.float64]:
        """The root's rotation matrix at each of *frames*, shape ``(F, 3, 3)``.

        What carries a direction of the skeleton's own frame into the
        world at a frame (ADR 0004): the root's heading and the facing
        chain's heading stage read it.
        """
        root_order = self.nodes[0].rot_channels  # type: ignore[attr-defined]
        return rotations.euler_to_rotmat(self.joint_angles[frames, 0], root_order)

    def facing_frame(
        self,
        coords: npt.NDArray[np.float64] | None = None,
    ):
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

        return analysis.facing_frame(self, coords=coords)

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
            positions = frames_to_node_positions(
                self, root_pos=self.root_pos, joint_angles=self.joint_angles, centered="world"
            )
            self._derived.node_positions = positions
        return positions

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
            ``"world"`` – root at actual position.
            ``"skeleton"`` – root at origin for all frames.
            ``"first"`` – ground-plane centering: the first frame's root
            position is subtracted in the two axes perpendicular to
            :attr:`world_up`; the up coordinate is untouched, so the
            motion starts above the origin at its original height.
        """
        centered_options = ["skeleton", "first", "world"]
        if centered not in centered_options:
            raise ValueError(
                f"The value {centered} is not recognized for the centered "
                f"argument. Currently recognized keywords are {centered_options}"
            )

        if frame is None:
            world = self._world_node_positions()
            if centered == "world":
                return world.copy()
            if centered == "skeleton":
                return world - self.root_pos[:, np.newaxis, :]
            # centered == "first"; with no frames there is nothing to center on
            if self.frame_count == 0:
                return world.copy()
            return world - _ground_plane_offset(self.root_pos[0], self.world_up)
        if not -self.frame_count <= frame < self.frame_count:
            raise IndexError(
                f"frame {frame} is out of range for "
                f"{self.frame_count} frames. Use frame=None for all frames."
            )
        actual = frame if frame >= 0 else frame + self.frame_count
        if self._derived.node_positions is not None:
            world_frame = self._derived.node_positions[actual]
            if centered == "world":
                return world_frame.copy()
            if centered == "skeleton":
                return world_frame - self.root_pos[actual]
            return world_frame - _ground_plane_offset(self.root_pos[actual], self.world_up)
        if centered == "first":
            return frames_to_node_positions(
                self,
                root_pos=self.root_pos[actual],
                joint_angles=self.joint_angles[actual],
                centered="first",
                world_up=self.world_up,
            )
        return frames_to_node_positions(
            self,
            root_pos=self.root_pos[actual],
            joint_angles=self.joint_angles[actual],
            centered=centered,
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

        Parameters
        ----------
        mode : str, optional
            ``'euler'`` (default): the motion as a file carries it, in
            degrees: ``<root>_<axis>_pos`` per position channel of the
            root, then ``<joint>_<axis>_rot`` per rotation channel of
            each joint in :attr:`nodes` order, so the columns after
            ``time`` are the flat layout of :attr:`root_pos` and
            :attr:`joint_angles`. :func:`~pybvh.df_to_bvh` and
            :meth:`from_df` read this form back.
            ``'coordinates'``: the posed positions of
            :meth:`node_positions`, ``<node>_X``, ``<node>_Y`` and
            ``<node>_Z`` per node in :attr:`nodes` order, end sites
            included. This export is one-way: positions do not determine
            the joint angles, and ``df_to_bvh`` reads euler-mode frames.
        centered : str, optional
            ``"world"`` (default), ``"skeleton"``, or ``"first"`` — see
            :meth:`node_positions` for their semantics. Only used when
            ``mode='coordinates'``.

        Returns
        -------
        dict
            Column-name → 1-D array mapping, ready for ``pd.DataFrame()``.

        Notes
        -----
        Node names are not unique in a BVH file: the two end sites of one
        joint share their generated display name, and joints may repeat
        a name. Columns are labelled the way ``pandas.read_csv`` labels
        repeated headers: the first node of a name keeps it, each later
        one gets ``.1``, ``.2``, ... before the axis and kind
        (``Arm_Z_rot``, ``Arm.1_Z_rot``; ``EndSiteHand_X``,
        ``EndSiteHand.1_X``), counted up until the label is neither
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
        joint_angles_deg = np.rad2deg(self.joint_angles)
        # The width is spelled out: NumPy cannot infer -1 from no frames.
        flat = np.concatenate(
            [self.root_pos, joint_angles_deg.reshape(self.frame_count, 3 * self.joint_count)],
            axis=1,
        )
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
        result: dict[str, npt.NDArray[np.float64]] = {
            "time": np.arange(self.frame_count, dtype=np.float64) * self.frame_time
        }
        result.update(zip(columns, flat.T))
        return result

    def to_node_table(self) -> list[dict[str, Any]]:
        """Export the skeleton as a node table: one plain ``dict`` per node.

        The skeleton as plain data, the counterpart of :meth:`from_df`'s
        ``hier`` argument as :meth:`to_df_dict` is of its ``df``. One
        entry per node in :attr:`nodes` order, each with ``name``,
        ``parent`` (the index of the parent's entry, ``None`` on the
        root), ``offset``, ``rot_channels`` on the root and joints and
        ``pos_channels`` on the root; an entry without ``rot_channels``
        is an end site. Position is identity, so two end sites under one
        joint and joints sharing a name are carried as they are, which
        no name-keyed export can do. :func:`~pybvh.nodes_to_table`
        defines the format and names its conventions;
        :func:`~pybvh.nodes_from_table` builds a tree back from it.

        The table is the named twin of :attr:`fk_topology`: ``parent``
        with ``None`` read as ``-1`` is its ``parent_idx``, the stacked
        ``offset`` values are its ``offsets``, and the entries with
        ``rot_channels`` are its joint columns, in order.

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

    def index(self, name: str, space: Literal["joint", "node"]) -> int:
        """Look up the integer index for ``name`` in the requested index space.

        Unambiguous alternative to picking between :attr:`joint_index`
        and :attr:`node_index` at the call site. Use ``space='joint'``
        when indexing :attr:`joint_angles` / :meth:`joint_velocities` /
        :meth:`joint_accelerations` / :meth:`joint_positions` /
        :meth:`angular_velocities` (any ``(F, J, ...)`` array). Use
        ``space='node'`` when indexing :meth:`node_positions` /
        :meth:`node_velocities` / :meth:`node_accelerations` (any
        ``(F, N, ...)`` array).

        Parameters
        ----------
        name : str
            Joint or node name.
        space : {'joint', 'node'}
            Which index space to look up. ``'joint'`` excludes end sites.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If ``name`` is not present in the requested index space (e.g.
            an end-site name with ``space='joint'``).
        ValueError
            If ``space`` is not ``'joint'`` or ``'node'``.
        """
        if space == "joint":
            return self.joint_index[name]
        if space == "node":
            return self.node_index[name]
        raise ValueError(f"space must be 'joint' or 'node', got {space!r}")

    @property
    def joint_tips(self) -> dict[str, int | None]:
        """Mapping from joint name to its end-site **node** index, or None.

        For every non-end-site joint (root included): the node-space index of the joint's end-site child — a row of :meth:`node_positions` output — or ``None`` for interior joints whose children are all joints. The tip is the bone's far end, so ``bvh.node_positions()[:, bvh.joint_tips["LeftFoot"]]`` is the toe-tip trajectory without knowing the end site's generated display name.

        Resolution is identity-based on the node tree, never by name: end-site display names are cosmetic (the parser generates ``'EndSite' + parent name``) and may even collide with a real joint's name, which would make a name-keyed lookup through :attr:`node_index` silently pick the wrong node. A joint with several end-site children (nonstandard, but the parser accepts it) maps to the **first one in file order**.

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
        """
        return [
            "".join(n.rot_channels)  # type: ignore[attr-defined]
            for n in self.nodes
            if not n.is_end_site()
        ]

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

        target = self if inplace else self.copy()
        target._replace_offsets(offsets)
        if inplace:
            return None
        return target

    @overload
    def scale(self, scale: float, *, inplace: Literal[True]) -> None: ...
    @overload
    def scale(self, scale: float, inplace: Literal[False] = ...) -> Bvh: ...
    def scale(self, scale: float, inplace: bool = False) -> Bvh | None:
        """Uniformly scale all node offsets and the root translation.

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
        target._replace_offsets(target._offsets() * factor)
        target._set_motion(root_pos=target.root_pos * factor)
        if inplace:
            return None
        return target

    @overload
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: str | BvhNode | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: str | BvhNode | None = ...,
        inplace: Literal[False] = ...,
    ) -> Bvh: ...
    def change_euler_order(
        self,
        order: str | Sequence[str],
        joint: str | BvhNode | None = None,
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
            If a joint name or node is given, only that joint is changed.
            If None (default), all joints are changed to the new order.
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.
            With True, the skeleton is replaced, not edited: a node taken from
            ``nodes`` before the call keeps its old values.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        if isinstance(order, str):
            new_order = list(order.upper())
        else:
            new_order = [c.upper() for c in order]

        if joint is None:
            joint_indices = list(range(self.joint_count))
        else:
            if isinstance(joint, BvhNode):
                joint_name = joint.name
            elif isinstance(joint, str):
                joint_name = joint
            else:
                raise ValueError("joint should be a string (joint name), a BvhNode object, or None")
            if joint_name not in self.joint_index:
                raise ValueError(f"Joint '{joint_name}' not found among non-end-site nodes.")
            joint_indices = [self.joint_index[joint_name]]

        target = self if inplace else self.copy()
        # The new skeleton is built, and the order checked, before any
        # angle is converted.
        table = nodes_to_table(target.nodes)
        joint_rows = [row for row in table if "rot_channels" in row]
        old_orders = [row["rot_channels"] for row in joint_rows]
        for j_idx in joint_indices:
            joint_rows[j_idx]["rot_channels"] = new_order
        new_nodes = nodes_from_table(table)

        for j_idx in joint_indices:
            old_order = old_orders[j_idx]
            if old_order == new_order:
                continue
            # Convert: old Euler → rotmat → new Euler, then write back
            # (private array — the public view is read-only).
            R = rotations.euler_to_rotmat(target.joint_angles[:, j_idx], old_order)
            target._joint_angles[:, j_idx] = rotations.rotmat_to_euler(R, new_order)
        target._replace_skeleton(new_nodes)

        if inplace:
            return None
        return target

    def to_rotmat(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert all per-joint Euler angles in self.frames to rotation matrices.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_rotmats : ndarray, shape (num_frames, num_joints, 3, 3)
            Rotation matrix for each joint in each frame.
            Joint order matches ``self.joint_names`` / ``self.joint_index``.

        Notes
        -----
        When multiple rotation representations are needed (e.g. 6D for
        the model and quaternions for runtime SLERP), call ``to_rotmat``
        once and apply the relevant ``rotations.rotmat_to_*`` primitives
        directly — forward kinematics runs once instead of per
        representation.
        """
        joints = [n for n in self.nodes if not n.is_end_site()]
        per_joint = ["".join(j.rot_channels) for j in joints]  # type: ignore[attr-defined]
        joint_rotmats = rotations.euler_to_rotmat(self.joint_angles, per_joint)
        return self.root_pos.copy(), joint_rotmats

    def to_6d(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert all per-joint Euler angles to 6D rotation representation.

        The 6D representation (Zhou et al., CVPR 2019) is continuous and
        well-suited for neural network training.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_rot6d : ndarray, shape (num_frames, num_joints, 6)
            6D rotation for each joint in each frame.

        Notes
        -----
        When multiple representations are needed, call :meth:`to_rotmat`
        once and apply :func:`pybvh.rotations.rotmat_to_rot6d` /
        ``rotmat_to_quat`` / ``rotmat_to_axisangle`` directly to avoid
        running FK more than once.
        """
        root_pos, joint_rotmats = self.to_rotmat()
        joint_rot6d = rotations.rotmat_to_rot6d(joint_rotmats)
        return root_pos, joint_rot6d

    def to_quat(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert all per-joint Euler angles to quaternions.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
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
        root_pos, joint_rotmats = self.to_rotmat()
        joint_quats = rotations.rotmat_to_quat(joint_rotmats)
        return root_pos, joint_quats

    def to_axisangle(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert all per-joint Euler angles to axis-angle vectors.

        The axis-angle representation is the unit rotation axis scaled
        by the rotation angle in radians.  Used in SMPL/SMPL-X body
        models and many pose estimation pipelines.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_aa : ndarray, shape (num_frames, num_joints, 3)
            Axis-angle vector for each joint in each frame.

        Notes
        -----
        See :meth:`to_rotmat` for the multi-representation reuse pattern.
        """
        root_pos, joint_rotmats = self.to_rotmat()
        joint_aa = rotations.rotmat_to_axisangle(joint_rotmats)
        return root_pos, joint_aa

    def _set_from_rotmats(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_rotmats: npt.NDArray[np.float64],
        param_name: str,
    ) -> None:
        """Shared validation + Euler conversion + assignment for the ``from_*`` importers.

        Writes into ``self`` — callers pass their ``target`` (self or a
        copy). Euler conversion is vectorized across joints in one
        per-joint-order :func:`pybvh.rotations.rotmat_to_euler` call.
        """
        if root_pos.shape[0] != joint_rotmats.shape[0]:
            raise ValueError(
                f"Frame count mismatch: root_pos has {root_pos.shape[0]} frames "
                f"but joint data has {joint_rotmats.shape[0]} frames"
            )
        if joint_rotmats.shape[1] != self.joint_count:
            raise ValueError(
                f"Expected {self.joint_count} joints in {param_name}, got {joint_rotmats.shape[1]}"
            )
        new_angles = rotations.rotmat_to_euler(joint_rotmats, self.euler_orders)
        self.root_pos = root_pos
        self.joint_angles = new_angles

    @overload
    def from_rotmat(
        self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, *, inplace: Literal[True]
    ) -> None: ...
    @overload
    def from_rotmat(
        self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, inplace: Literal[False] = ...
    ) -> Bvh: ...
    def from_rotmat(
        self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, inplace: bool = False
    ) -> Bvh | None:
        """
        Set motion data from root positions and rotation matrices.

        Converts rotation matrices back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.
        Inverse of :meth:`to_rotmat`.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_rotmats : array_like, shape (num_frames, num_joints, 3, 3)
            Rotation matrix per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()
        root_pos_arr = np.asarray(root_pos, dtype=np.float64)
        joint_rotmats_arr = np.asarray(joint_rotmats, dtype=np.float64)
        target._set_from_rotmats(root_pos_arr, joint_rotmats_arr, "joint_rotmats")
        if inplace:
            return None
        return target

    @overload
    def from_6d(
        self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, *, inplace: Literal[True]
    ) -> None: ...
    @overload
    def from_6d(
        self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: Literal[False] = ...
    ) -> Bvh: ...
    def from_6d(
        self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: bool = False
    ) -> Bvh | None:
        """
        Set motion data from root positions and 6D rotation data.

        Converts 6D rotations back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_rot6d : array_like, shape (num_frames, num_joints, 6)
            6D rotation per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()
        root_pos_arr = np.asarray(root_pos, dtype=np.float64)
        joint_rot6d_arr = np.asarray(joint_rot6d, dtype=np.float64)
        target._set_from_rotmats(
            root_pos_arr, rotations.rot6d_to_rotmat(joint_rot6d_arr), "joint_rot6d"
        )
        if inplace:
            return None
        return target

    @overload
    def from_quat(
        self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, *, inplace: Literal[True]
    ) -> None: ...
    @overload
    def from_quat(
        self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: Literal[False] = ...
    ) -> Bvh: ...
    def from_quat(
        self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: bool = False
    ) -> Bvh | None:
        """
        Set motion data from root positions and quaternion data.

        Converts quaternions back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_quats : array_like, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()
        root_pos_arr = np.asarray(root_pos, dtype=np.float64)
        joint_quats_arr = np.asarray(joint_quats, dtype=np.float64)
        target._set_from_rotmats(
            root_pos_arr, rotations.quat_to_rotmat(joint_quats_arr), "joint_quats"
        )
        if inplace:
            return None
        return target

    @overload
    def from_axisangle(
        self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, *, inplace: Literal[True]
    ) -> None: ...
    @overload
    def from_axisangle(
        self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: Literal[False] = ...
    ) -> Bvh: ...
    def from_axisangle(
        self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: bool = False
    ) -> Bvh | None:
        """
        Set motion data from root positions and axis-angle data.

        Converts axis-angle vectors back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_aa : array_like, shape (num_frames, num_joints, 3)
            Axis-angle vector per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()
        root_pos_arr = np.asarray(root_pos, dtype=np.float64)
        joint_aa_arr = np.asarray(joint_aa, dtype=np.float64)
        target._set_from_rotmats(
            root_pos_arr, rotations.axisangle_to_rotmat(joint_aa_arr), "joint_aa"
        )
        if inplace:
            return None
        return target

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
            root_pos=self._root_pos[s].copy(), joint_angles=self._joint_angles[s].copy()
        )
        # Adjust frame frequency if step changes the sampling rate
        if step is not None and abs(step) != 1:
            new_bvh.frame_time = self.frame_time * abs(step)
        return new_bvh

    def _check_same_skeleton(self, other: Bvh) -> None:
        """Raise ValueError if ``other`` has a different skeleton topology.

        Compares node count, per-node names, and per-joint rotation
        orders (end sites excluded since they have no rotation channels).
        Shared by ``__add__``, ``__iadd__``, and ``__setitem__``.
        """
        if len(self.nodes) != len(other.nodes):
            raise ValueError(f"Node count mismatch: {len(self.nodes)} vs {len(other.nodes)}")
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1.name != n2.name:
                raise ValueError(f"Node name mismatch: '{n1.name}' vs '{n2.name}'")
            if _is_joint(n1) and _is_joint(n2):
                if n1.rot_channels != n2.rot_channels:
                    raise ValueError(
                        f"Rotation order mismatch for '{n1.name}': "
                        f"{n1.rot_channels} vs {n2.rot_channels}"
                    )

    def _concat(self, other: Bvh) -> Bvh:
        """Implementation of ``self + other`` (see :meth:`__add__`).

        Parameters
        ----------
        other : Bvh
            Must have the same skeleton (same node names and rotation
            orders).

        Returns
        -------
        Bvh
            New Bvh with frames from ``self`` followed by ``other``.

        Raises
        ------
        ValueError
            If skeletons are incompatible (different node count, names,
            or rotation orders).
        """
        self._check_same_skeleton(other)

        if self.frame_time != other.frame_time:
            warnings.warn(
                f"Frame time mismatch: {self.frame_time} vs "
                f"{other.frame_time}. Using self's frame time.",
                stacklevel=user_stacklevel(),
            )

        new_bvh = self._derive_clip(
            root_pos=np.concatenate([self.root_pos, other.root_pos], axis=0),
            joint_angles=np.concatenate([self.joint_angles, other.joint_angles], axis=0),
        )
        if self.source_path != other.source_path:
            new_bvh.source_path = None
        return new_bvh

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
        Clips shorter than two frames have nothing to interpolate and
        simply adopt the new ``frame_time``.

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
            If ``target_fps`` is not a positive finite number.
        """
        _validate_fps(target_fps, name="target_fps")

        new_freq = 1.0 / target_fps
        if self.frame_count < 2:
            # Nothing to interpolate, but the clip still adopts the new rate.
            new_bvh = self.copy()
            new_bvh.frame_time = new_freq
            return new_bvh

        # Original and target timestamps. The epsilon on the arange stop
        # keeps the final original timestamp reachable despite float
        # rounding (np.arange guarantees values strictly below the stop).
        t_orig = np.arange(self.frame_count) * self.frame_time
        t_new = np.arange(0, t_orig[-1] + 1e-12, new_freq)

        num_new = len(t_new)
        num_joints = self.joint_count

        # --- Root position: linear interpolation ---
        new_root_pos = np.empty((num_new, 3), dtype=np.float64)
        for ax in range(3):
            new_root_pos[:, ax] = np.interp(t_new, t_orig, self.root_pos[:, ax])

        # --- Joint angles: quaternion SLERP ---
        # Convert all joints to quaternions: (F, J, 4)
        _, joint_quats = self.to_quat()

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

        # SLERP for all joints at once: shape (num_new, J, 4)
        q_left = joint_quats[idx_left]  # (num_new, J, 4)
        q_right = joint_quats[idx_right]  # (num_new, J, 4)

        # Broadcast alpha to (num_new, J) for per-joint SLERP
        alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (num_new, num_joints))
        new_quats = rotations.quat_slerp(q_left, q_right, alpha_jt)

        # Convert back to Euler angles — vectorized across joints via the
        # per-joint-order overload of rotmat_to_euler.
        new_angles = rotations.rotmat_to_euler(
            rotations.quat_to_rotmat(new_quats), self.euler_orders
        )

        new_bvh = self._derive_clip(root_pos=new_root_pos, joint_angles=new_angles)
        new_bvh.frame_time = new_freq
        return new_bvh

    # ----------------------------------------------------------------
    # Joint subsetting
    # ----------------------------------------------------------------

    def extract_joints(self, joint_names: list[str]) -> Bvh:
        """Extract a subset of joints into a new Bvh.

        Removed joints' offsets are collapsed into their nearest kept
        descendant via vector addition (valid at rest pose).  Their
        rotation contribution during animation is lost.

        The result lists its nodes in depth-first order, as a file
        would, so writing it and reading it back keeps
        :meth:`matches_hierarchy`. A kept joint left with nothing below
        it (no kept descendant, no end site of its own) gets an end site
        named ``'EndSite' + name``, placed right after it, at the offset
        of its nearest original end-site descendant (zero when it has
        none).

        ``source_path``, a manual ``world_up`` override, and a user-set
        ``lr_mapping`` (filtered to pairs whose joints are both kept) are
        preserved on the result. A guessed world up is inherited, not
        guessed again from the kept joints (see :attr:`world_up`).

        Parameters
        ----------
        joint_names : list of str
            Names of joints to keep.  The root must be included.
            End sites are handled automatically (kept if their parent
            is kept, otherwise removed). Selection is by name, so two
            joints sharing a name are kept or dropped together; their
            parents and ``joint_angles`` columns are still resolved by
            position, never by name.

        Returns
        -------
        Bvh
            New Bvh with the reduced skeleton and corresponding motion
            data.

        Raises
        ------
        ValueError
            If the root joint is not in ``joint_names``.
        """
        keep_set = set(joint_names)

        if self.root.name not in keep_set:
            raise ValueError(f"Root joint '{self.root.name}' must be in joint_names.")

        def survives_extraction(node: BvhNode) -> bool:
            if node.is_end_site():
                return node.parent is not None and node.parent.name in keep_set
            return node.name in keep_set

        # A kept joint ends childless in the new tree when nothing below
        # it survives: its end site then goes right after it, which is
        # its depth-first place. Walking the list backwards settles each
        # node's flag before its parent reads it, since parents precede
        # their children in ``nodes``.
        survives = [survives_extraction(node) for node in self.nodes]
        position = {id(node): i for i, node in enumerate(self.nodes)}
        survivor_below = [False] * len(self.nodes)
        for i in range(len(self.nodes) - 1, 0, -1):
            parent_position = position[id(self.nodes[i].parent)]
            if survives[i] or survivor_below[i]:
                survivor_below[parent_position] = True

        rows: list[dict[str, Any]] = []
        kept_old_j_indices: list[int] = []
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
                acc_offset = node.offset.copy()
                walker = node.parent
                while walker is not None and walker.name not in keep_set:
                    acc_offset = walker.offset + acc_offset
                    walker = walker.parent
                row["parent"] = row_of[id(walker)]
                row["offset"] = acc_offset
            row["rot_channels"] = node.rot_channels  # type: ignore[attr-defined]

            row_of[id(node)] = len(rows)
            rows.append(row)
            kept_old_j_indices.append(column)

            if not survivor_below[i]:
                # No name: the builder names it 'EndSite' + the joint's
                # name, as the parser would.
                rows.append(
                    {"parent": row_of[id(node)], "offset": self._find_end_site_offset(node)}
                )

        new_nodes = nodes_from_table(rows)

        # Selecting the kept columns makes a new array.
        new_bvh = self._derive_clip(
            root_pos=self._root_pos.copy(),
            joint_angles=self._joint_angles[:, kept_old_j_indices, :],
            nodes=new_nodes,
        )
        # A user-set L/R mapping keeps the pairs whose joints are both kept;
        # with none left, the pairs are detected from the kept names, as
        # they are when the user set none.
        user_mapping = self._lr_mapping_setting
        if isinstance(user_mapping, dict):
            kept_pairs = {
                left: right
                for left, right in user_mapping.items()
                if left in keep_set and right in keep_set
            }
            new_bvh._set_lr_mapping_setting(kept_pairs or "names")
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
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position velocities — shape ``(F, J, 3)``. See :func:`pybvh.analysis.joint_velocities`."""
        from . import analysis

        return analysis.joint_velocities(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def node_velocities(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position velocities (joints + end sites) — shape ``(F, N, 3)``. See :func:`pybvh.analysis.node_velocities`."""
        from . import analysis

        return analysis.node_velocities(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def joint_accelerations(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position accelerations — shape ``(F, J, 3)``. See :func:`pybvh.analysis.joint_accelerations`."""
        from . import analysis

        return analysis.joint_accelerations(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def node_accelerations(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position accelerations (joints + end sites) — shape ``(F, N, 3)``. See :func:`pybvh.analysis.node_accelerations`."""
        from . import analysis

        return analysis.node_accelerations(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def joint_speed_derivative(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint rate of change of speed ``d‖v‖/dt`` — shape ``(F, J)``. See :func:`pybvh.analysis.joint_speed_derivative`."""
        from . import analysis

        return analysis.joint_speed_derivative(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def node_speed_derivative(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node rate of change of speed ``d‖v‖/dt`` (joints + end sites) — shape ``(F, N)``. See :func:`pybvh.analysis.node_speed_derivative`."""
        from . import analysis

        return analysis.node_speed_derivative(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
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
        foot_joints: list[str] | None = ...,
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: Literal[False] = ...,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def foot_contacts(
        self,
        foot_joints: list[str] | None = ...,
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], dict]: ...
    @overload
    def foot_contacts(
        self,
        foot_joints: list[str] | None = ...,
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: bool,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]: ...
    def foot_contacts(
        self,
        foot_joints: list[str] | None = None,
        method: str = "combined",
        coords: npt.NDArray[np.float64] | None = None,
        *,
        vel_threshold: float | None = None,
        vel_smooth_duration: float = 1.0 / 30.0,
        height_threshold: float | None = None,
        floor: float | str = "auto",
        min_contact_duration: float = 0.1,
        min_gap_duration: float = 0.1,
        hysteresis: float = 0.25,
        adaptive: bool = False,
        height_reference: str = "velocity",
        return_info: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]:
        """Detect foot contact labels.  See :func:`pybvh.analysis.foot_contacts`."""
        from . import analysis

        return analysis.foot_contacts(
            self,
            foot_joints=foot_joints,
            method=method,
            coords=coords,
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
        joints: Sequence[str | int],
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: Literal[False] = ...,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def ground_contacts(
        self,
        joints: Sequence[str | int],
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], dict]: ...
    @overload
    def ground_contacts(
        self,
        joints: Sequence[str | int],
        method: str = ...,
        coords: npt.NDArray[np.float64] | None = ...,
        *,
        vel_threshold: float | None = ...,
        vel_smooth_duration: float = ...,
        height_threshold: float | None = ...,
        floor: float | str = ...,
        min_contact_duration: float = ...,
        min_gap_duration: float = ...,
        hysteresis: float = ...,
        adaptive: bool = ...,
        height_reference: str = ...,
        return_info: bool,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]: ...
    def ground_contacts(
        self,
        joints: Sequence[str | int],
        method: str = "combined",
        coords: npt.NDArray[np.float64] | None = None,
        *,
        vel_threshold: float | None = None,
        vel_smooth_duration: float = 1.0 / 30.0,
        height_threshold: float | None = None,
        floor: float | str = "auto",
        min_contact_duration: float = 0.1,
        min_gap_duration: float = 0.1,
        hysteresis: float = 0.25,
        adaptive: bool = False,
        height_reference: str = "floor",
        return_info: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]:
        """Detect ground contacts for an arbitrary joint set.  See :func:`pybvh.analysis.ground_contacts`."""
        from . import analysis

        return analysis.ground_contacts(
            self,
            joints,
            method=method,
            coords=coords,
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

    def auto_detect_foot_joints(self) -> list[str]:
        """Auto-detect foot joint names from skeleton topology.  See :func:`pybvh.analysis.auto_detect_foot_joints`."""
        from . import analysis

        return analysis.auto_detect_foot_joints(self)

    def to_feature_array(
        self,
        representation: str = "6d",
        include_root_pos: bool = True,
        include_velocities: bool = False,
        include_foot_contacts: bool = False,
        centered: str = "world",
        foot_joints: list[str] | None = None,
        stencil: str = "central",
        pad: str = "edge",
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
            foot_joints=foot_joints,
            stencil=stencil,
            pad=pad,
        )

    def feature_array_layout(
        self,
        *,
        num_feet: int = 0,
        representation: str = "6d",
        include_root_pos: bool = True,
        include_velocities: bool = False,
        include_foot_contacts: bool = False,
    ) -> dict[str, slice]:
        """Column layout of :meth:`to_feature_array` output.  See :func:`pybvh.features.feature_array_layout`."""
        from . import features

        return features.feature_array_layout(
            num_joints=self.joint_count,
            num_feet=num_feet,
            representation=representation,
            include_root_pos=include_root_pos,
            include_velocities=include_velocities,
            include_foot_contacts=include_foot_contacts,
        )

    # ----------------------------------------------------------------
    #  Motion descriptors (delegate to geometry / analysis modules)
    # ----------------------------------------------------------------
    #  Relational / trajectory queries index in NODE space (node_index), so
    #  end sites (fingertips, toe tips, head top) are first-class; pass a
    #  joint or end-site name. Every descriptor accepts pre-computed
    #  positions via ``coords=`` for hot loops over many descriptors.

    def _descriptor_index(self, joint: str, space: str = "node") -> int:
        """Resolve a name for the descriptor methods — names only, no ints.

        Integer indices are rejected because they are ambiguous between
        joint space and node space (the historical source of silently
        misaligned lookups).
        """
        if not isinstance(joint, str):
            raise TypeError(
                f"joint must be a name (str), got {type(joint).__name__}. "
                f"Integer indices are ambiguous between joint and node index "
                f"spaces — resolve names explicitly with "
                f"bvh.index(name, space={space!r}) and use the functional "
                f"pybvh.geometry / pybvh.analysis API for index-based access."
            )
        return self.index(joint, space=space)  # type: ignore[arg-type]

    def _positions_or(
        self,
        coords: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """``coords`` if given, else the (cached) world-frame node positions."""
        if coords is None:
            return self.node_positions()
        return np.asarray(coords, dtype=np.float64)

    def curvature(
        self,
        joint: str,
        stencil: str = "central",
        pad: str = "edge",
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame trajectory curvature of ``joint``. See :func:`pybvh.geometry.curvature`."""
        from . import geometry

        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.curvature(traj, self.frame_time, stencil, pad)

    def torsion(
        self,
        joint: str,
        stencil: str = "central",
        pad: str = "edge",
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame trajectory torsion of ``joint``. See :func:`pybvh.geometry.torsion`."""
        from . import geometry

        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.torsion(traj, self.frame_time, stencil, pad)

    def movement_phase(
        self,
        joint: str,
        stencil: str = "central",
        pad: str = "edge",
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame movement phase (``speed · curvature``) of ``joint``.

        See :func:`pybvh.geometry.movement_phase`."""
        from . import geometry

        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.movement_phase(traj, self.frame_time, stencil, pad)

    def path_length(self, joint: str, *, coords: npt.NDArray[np.float64] | None = None) -> float:
        """Arc length travelled by ``joint``. See :func:`pybvh.geometry.path_length`."""
        from . import geometry

        return float(
            geometry.path_length(self._positions_or(coords)[:, self._descriptor_index(joint), :])
        )

    def directness(self, joint: str, *, coords: npt.NDArray[np.float64] | None = None) -> float:
        """Directness of ``joint``'s path (net displacement ÷ path length).

        See :func:`pybvh.geometry.directness`."""
        from . import geometry

        return float(
            geometry.directness(self._positions_or(coords)[:, self._descriptor_index(joint), :])
        )

    def ground_path(
        self, joint: str, *, coords: npt.NDArray[np.float64] | None = None
    ) -> geometry.GroundPath:
        """Ground-plane path of ``joint`` (uses ``world_up``). See :func:`pybvh.geometry.ground_path`."""
        from . import geometry

        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.ground_path(traj, _axis_to_vector(self.world_up))

    def inter_joint_distance(
        self,
        pairs: list[tuple[str, str]],
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame distances between node pairs. See :func:`pybvh.geometry.inter_joint_distance`."""
        from . import geometry

        idx_pairs = [[self._descriptor_index(a), self._descriptor_index(b)] for a, b in pairs]
        return geometry.inter_joint_distance(self._positions_or(coords), idx_pairs)

    def joint_angle(
        self,
        a: str,
        vertex: str,
        b: str,
        degrees: bool = False,
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame angle at ``vertex`` in ``a–vertex–b``. See :func:`pybvh.geometry.joint_angle`."""
        from . import geometry

        pos = self._positions_or(coords)
        return geometry.joint_angle(
            pos[:, self._descriptor_index(a)],
            pos[:, self._descriptor_index(vertex)],
            pos[:, self._descriptor_index(b)],
            degrees=degrees,
        )

    def triangle_area(
        self, a: str, b: str, c: str, *, coords: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """Per-frame area of triangle ``(a, b, c)``. See :func:`pybvh.geometry.triangle_area`."""
        from . import geometry

        pos = self._positions_or(coords)
        return geometry.triangle_area(
            pos[:, self._descriptor_index(a)],
            pos[:, self._descriptor_index(b)],
            pos[:, self._descriptor_index(c)],
        )

    def segment_axis_angle(
        self,
        joint_a: str,
        joint_b: str,
        degrees: bool = False,
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame angle of the bone ``joint_a→joint_b`` to ``world_up``.

        See :func:`pybvh.geometry.segment_axis_angle`."""
        from . import geometry

        pos = self._positions_or(coords)
        seg = pos[:, self._descriptor_index(joint_b)] - pos[:, self._descriptor_index(joint_a)]
        return geometry.segment_axis_angle(seg, _axis_to_vector(self.world_up), degrees=degrees)

    def bounding_box(
        self, *, coords: npt.NDArray[np.float64] | None = None
    ) -> geometry.BoundingBox:
        """Per-frame axis-aligned bounding box of all nodes. See :func:`pybvh.geometry.bounding_box`."""
        from . import geometry

        return geometry.bounding_box(self._positions_or(coords))

    def bounding_sphere(
        self, *, coords: npt.NDArray[np.float64] | None = None
    ) -> geometry.BoundingSphere:
        """Per-frame approximate enclosing sphere of all nodes. See :func:`pybvh.geometry.bounding_sphere`."""
        from . import geometry

        return geometry.bounding_sphere(self._positions_or(coords))

    def bounding_ellipsoid(
        self, *, coords: npt.NDArray[np.float64] | None = None
    ) -> geometry.BoundingEllipsoid:
        """Per-frame PCA-aligned bounding ellipsoid of all nodes. See :func:`pybvh.geometry.bounding_ellipsoid`."""
        from . import geometry

        return geometry.bounding_ellipsoid(self._positions_or(coords))

    def center_of_mass(
        self,
        weights: npt.NDArray[np.float64] | None = None,
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame centre of mass of all nodes (uniform by default; pass per-node masses).

        See :func:`pybvh.geometry.center_of_mass`."""
        from . import geometry

        return geometry.center_of_mass(self._positions_or(coords), weights=weights)

    def com_displacement(
        self,
        weights: npt.NDArray[np.float64] | None = None,
        com_ref: npt.NDArray[np.float64] | None = None,
        *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame centre-of-mass travel from a reference.

        ``com_ref`` defaults to the **first frame's** centre of mass, in the
        same world frame as :meth:`center_of_mass`, so the result is how far
        the CoM has travelled since the start (``0`` at frame 0). Pass an
        explicit ``com_ref`` (e.g. ``center_of_mass().mean(0)``) for a
        different baseline. See :func:`pybvh.geometry.com_displacement`."""
        from . import geometry

        com = geometry.center_of_mass(self._positions_or(coords), weights=weights)
        if com_ref is None:
            com_ref = com[0]
        return geometry.com_displacement(com, com_ref)

    def verticality(
        self, *, coords: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """Per-frame height/width ratio along ``world_up``. See :func:`pybvh.geometry.verticality`."""
        from . import geometry

        return geometry.verticality(self._positions_or(coords), _axis_to_vector(self.world_up))

    def node_jerk(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-node position jerk — ``(F, N, 3)``. See :func:`pybvh.analysis.node_jerk`."""
        from . import analysis

        return analysis.node_jerk(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def joint_jerk(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Per-joint position jerk — ``(F, J, 3)``. See :func:`pybvh.analysis.joint_jerk`."""
        from . import analysis

        return analysis.joint_jerk(
            self, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
        )

    def smoothness(
        self,
        joint: str,
        metric: str = "sparc",
        *,
        coords: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> float:
        """Smoothness of ``joint``'s speed profile. See :func:`pybvh.analysis.smoothness`.

        Computes the joint's per-frame speed ``‖velocity‖`` and passes it
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
        default counts all). The remaining metrics take none."""
        from . import analysis

        vel = self.node_velocities(coords=coords)
        speed = np.linalg.norm(vel[:, self._descriptor_index(joint), :], axis=-1)
        value = analysis.smoothness(speed, 1.0 / self.frame_time, metric=metric, **kwargs)
        # One joint's (T,) speed reduces to a scalar; an array's rank is not in its type.
        return cast(float, value)

    def velocity_reductions(self, joint: str, *, coords: npt.NDArray[np.float64] | None = None):
        """Scalar reductions of ``joint``'s speed profile (peak, mean, …).

        Computes the joint's per-frame speed ``‖velocity‖`` and reduces it
        at sampling rate ``1 / frame_time``. See
        :func:`pybvh.analysis.velocity_reductions`."""
        from . import analysis

        vel = self.node_velocities(coords=coords)
        speed = np.linalg.norm(vel[:, self._descriptor_index(joint), :], axis=-1)
        return analysis.velocity_reductions(speed, 1.0 / self.frame_time)

    def skeleton_size(self, foot_joints: list[str] | None = None) -> float:
        """Absolute skeleton scale — mean rest-pose root-to-foot distance.

        Raises ``ValueError`` for a skeleton whose size cannot be
        measured (no feet found, or all feet on the root) rather than
        returning a substitute. See :func:`pybvh.analysis.skeleton_size`."""
        from . import analysis

        return analysis.skeleton_size(self, foot_joints=foot_joints)

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
        foot_joints: list[str] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """Step rate (onsets/second). See :func:`pybvh.analysis.cadence`."""
        from . import analysis

        return analysis.cadence(self, foot_joints=foot_joints, contacts=contacts)

    def stride_length(
        self,
        foot_joints: list[str] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """Mean stride length. See :func:`pybvh.analysis.stride_length`."""
        from . import analysis

        return analysis.stride_length(self, foot_joints=foot_joints, contacts=contacts)

    def walking_pace(self) -> float:
        """Mean horizontal speed. See :func:`pybvh.analysis.walking_pace`."""
        from . import analysis

        return analysis.walking_pace(self)

    def gait_parameters(
        self,
        foot_joints: list[str] | None = None,
        *,
        contacts: npt.NDArray[np.float64] | None = None,
    ):
        """Spatiotemporal gait parameters. See :func:`pybvh.analysis.gait_parameters`."""
        from . import analysis

        return analysis.gait_parameters(self, foot_joints=foot_joints, contacts=contacts)

    def range_of_motion(self, joint: str) -> npt.NDArray[np.float64]:
        """Peak-to-peak range of ``joint``'s Euler angles — ``(3,)`` per channel.

        Indexes in JOINT space (rotations exist only on joints). See
        :func:`pybvh.analysis.range_of_motion`."""
        from . import analysis

        idx = self._descriptor_index(joint, space="joint")
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
