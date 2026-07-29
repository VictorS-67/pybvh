from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence, TYPE_CHECKING, Union, overload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from . import geometry

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhNode, BvhJoint, BvhRoot, BvhEndSite
from .spatial_coord import frames_to_node_positions, _ground_plane_offset
from . import rotations
from .tools import (
    Axis,
    parse_axis,
    _axis_to_vector,
    _compute_forward_at,
    _compute_left_at,
    _detect_lr_mapping_by_names,
    _facing_is_measured,
    _infer_world_up,
    _iter_unique_lr_pairs,
    _rest_upward,
    _validate_axis_string,
)


class Bvh:
    """Container for BVH motion-capture data.

    The hierarchy is stored as a list of ``BvhNode`` objects (one per
    joint / end-site).  Motion data is stored as two structured arrays:

    - ``root_pos``:     shape ``(F, 3)``    — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)`` — Euler angles **in radians** per joint per frame

    ``Bvh`` is a sequence of frames: ``len(bvh) == frame_count`` and
    ``bvh[i]`` returns frame ``i`` as a one-frame Bvh. For joint or node
    counts, use ``bvh.joint_count`` or ``len(bvh.node_index)``.

    Attributes
    ----------
    nodes : list of BvhNode
        Skeleton hierarchy in topological order.
    root : BvhRoot
        The root node (``nodes[0]``).
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
        nodes: list[BvhNode] | None = None,
        root_pos: npt.ArrayLike | None = None,
        joint_angles: npt.ArrayLike | None = None,
        frame_time: float = 0,
        world_up: str = "auto",
        lr_mapping: dict[str, str] | None = None,
        source_path: str | None = None,
        warn_on_disagreement: bool = True,
    ) -> None:
        # All lazy caches exist before any property setter runs, so the
        # setters can invalidate them unconditionally. ``_world_up_override``
        # is set via the public ``world_up`` setter; ``_world_up_cached`` is
        # computed eagerly below (from first animation frame, with rest-pose
        # fallback) so that file-reading paths have a ready-to-use world_up
        # immediately. Floor height and FK positions are lazily computed on
        # first access (FK is too costly to run in every constructor) and
        # invalidated by the motion setters.
        self._world_up_override: str | None = None
        self._world_up_cached: str | None = None
        self._floor_height_cached: float | None = None
        self._node_positions_cached: npt.NDArray[np.float64] | None = None
        self._lr_mapping: dict[str, str] | None = None
        self._lr_mapping_source: str | None = None  # 'names' | 'user' | None

        if nodes is None:
            nodes = [BvhRoot()]
        self.nodes = nodes
        self.frame_time = frame_time
        self.source_path = source_path
        self.root = self.nodes[0]  # type: ignore[assignment]

        # Validate that root position channels are standard XYZ
        if self.root.pos_channels != ['X', 'Y', 'Z']:
            raise ValueError(
                f"Non-standard root position channel order "
                f"{self.root.pos_channels} is not supported. "
                f"Expected ['X', 'Y', 'Z'].")

        # ---------- Determine root_pos / joint_angles ----------
        if (root_pos is None) != (joint_angles is None):
            missing = "joint_angles" if joint_angles is None else "root_pos"
            raise ValueError(
                f"root_pos and joint_angles must be provided together; "
                f"{missing} is missing.")
        if root_pos is not None:
            self.root_pos = np.asarray(root_pos, dtype=np.float64)
            self.joint_angles = np.asarray(joint_angles, dtype=np.float64)
            if self._root_pos.shape[0] != self._joint_angles.shape[0]:
                raise ValueError(
                    f"root_pos and joint_angles disagree on frame count: "
                    f"root_pos has {self._root_pos.shape[0]} frames, "
                    f"joint_angles has {self._joint_angles.shape[0]}.")
            joint_count = sum(1 for n in self.nodes if not n.is_end_site())
            if self._joint_angles.shape[1] != joint_count:
                raise ValueError(
                    f"joint_angles has {self._joint_angles.shape[1]} joints "
                    f"on axis 1, but the skeleton has {joint_count} "
                    f"non-end-site joints.")
        else:
            # Empty object
            self.root_pos = np.empty((0, 3), dtype=np.float64)
            self.joint_angles = np.empty((0, 0, 3), dtype=np.float64)

        # node name → integer index into the spatial-coordinate array
        self._create_node_index()
        # joint name → integer index into joint_angles axis 1
        self._create_joint_index()

        # Freeze channel attributes on all nodes to prevent
        # desynchronization with joint_angles
        for node in self.nodes:
            if hasattr(node, '_frozen'):
                node._frozen = True

        if world_up != "auto":
            self._world_up_override = _validate_axis_string(world_up)
        elif self.frame_count > 0 and len(self.nodes) > 1:
            # warn_on_disagreement=False silences only the rest-pose vs
            # first-frame disagreement warning of this eager inference.
            self._world_up_cached = _infer_world_up(self, warn=warn_on_disagreement)

        # L/R pair mapping — cached. Depends on names + topology only, so
        # no runtime invalidation hooks are needed (no pybvh operation
        # mutates names on an existing Bvh). See also the `lr_mapping`
        # property docstring.
        if lr_mapping is not None:
            # B3 — explicit user mapping at construction time
            self._validate_and_set_lr_mapping(lr_mapping, source='user')
        elif len(self.nodes) > 1:
            # Strategy A — eager name-based detection
            names_mapping = _detect_lr_mapping_by_names(self)
            if names_mapping:
                self._lr_mapping = names_mapping
                self._lr_mapping_source = 'names'


    @property
    def nodes(self) -> list[BvhNode]:
        return self._nodes
    @nodes.setter
    def nodes(self, value: list[BvhNode]) -> None:
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("nodes should be a list of BvhNode class/subclasse objects")
        self._nodes = value 

    @property
    def frame_time(self) -> float:
        """Seconds between successive frames.

        A value of ``0`` means "unset" and is the default for newly
        constructed empty :class:`Bvh` objects. Writing to a file
        requires a positive value — :func:`~pybvh.io.write_bvh_file`
        raises ``ValueError`` otherwise.
        """
        return self._frame_time
    @frame_time.setter
    def frame_time(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"frame_time must be >= 0, got {value}")
        self._frame_time = value

    @property
    def fps(self) -> float:
        """Frames per second — convenience inverse of :attr:`frame_time`.

        Returns ``0.0`` when ``frame_time == 0`` (the "unset" sentinel)
        rather than raising, mirroring the behaviour of :meth:`__str__`.

        Example
        -------
        >>> if bvh.fps != 30:
        ...     bvh = bvh.resample(30)
        """
        return 1.0 / self._frame_time if self._frame_time > 0 else 0.0
    @fps.setter
    def fps(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"fps must be > 0, got {value}")
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
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"root_pos must have shape (F, 3), got {arr.shape}")
        self._root_pos = arr
        self._invalidate_motion_caches()

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
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"joint_angles must have shape (F, J, 3), got {arr.shape}")
        self._joint_angles = arr
        self._invalidate_motion_caches()

    def _invalidate_motion_caches(self) -> None:
        """Drop every cache derived from the motion data.

        Called by the ``root_pos`` / ``joint_angles`` setters and by any
        code path that mutates motion or rest-pose geometry without going
        through them (``__setitem__``, in-place ``retarget``).
        """
        self._world_up_cached = None
        self._floor_height_cached = None
        self._node_positions_cached = None

    @property
    def frame_count(self) -> int:
        """Number of frames (computed from root_pos)."""
        return len(self.root_pos)

    @property
    def root(self) -> BvhRoot:
        return self._root
    @root.setter
    def root(self, value: BvhRoot) -> None:
        if not isinstance(value, BvhRoot):
            raise ValueError("The first element of nodes should be a BvhRoot object")
        self._root = value

    @property
    def _euler_column_names(self) -> list[str]:
        """Column names describing root_pos + joint_angles in flat layout order.

        Useful for building DataFrames or inspecting the channel mapping.
        Generated on the fly from the node hierarchy.
        """
        names = []
        root = self.root
        for ax in root.pos_channels:
            names.append(f'{root.name}_{ax}_pos')
        for node in self.nodes:
            if node.is_end_site():
                continue
            for ax in node.rot_channels:  # type: ignore[attr-defined]
                names.append(f'{node.name}_{ax}_rot')
        return names

            
    def __str__(self) -> str:
        source = ""
        if self.source_path is not None:
            source = f", from {Path(self.source_path).name}"
        return (
            f'{self.joint_count} joints, {self.frame_count} frames at '
            f'{self.fps:.1f} fps (frame_time={self.frame_time:.6f}s{source})'
        )

    def __repr__(self) -> str:
        return (
            f'Bvh(joints={self.joint_names!r}, '
            f'frame_count={self.frame_count}, '
            f'frame_time={self.frame_time:.6f})'
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

    def matches_hierarchy(self, other: Bvh, match_offsets: bool = True,
                          atol: float = 1e-6) -> bool:
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
                raise IndexError(
                    f"frame index {k} out of range for Bvh with {F} frames")
            i = k if k >= 0 else k + F
            return self._slice_frames(i, i + 1)
        if isinstance(key, slice):
            return self._slice_frames(key.start, key.stop, key.step)
        raise TypeError(
            f"Bvh indices must be int or slice, got {type(key).__name__}. "
            "For arbitrary frame selection, build a new Bvh manually from "
            "the required root_pos and joint_angles arrays.")

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
        just like ``a + b``. Mutates ``self.root_pos`` and
        ``self.joint_angles`` via their setters so ``_world_up_cached`` is
        invalidated.
        """
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        self._check_same_skeleton(other)
        if self.frame_time != other.frame_time:
            warnings.warn(
                f"Frame time mismatch: {self.frame_time} vs "
                f"{other.frame_time}. Using self's frame time.")
        self.root_pos = np.concatenate(
            [self.root_pos, other.root_pos], axis=0)
        self.joint_angles = np.concatenate(
            [self.joint_angles, other.joint_angles], axis=0)
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
        copies); motion-derived caches are invalidated explicitly.
        """
        # --- key → canonical slice ---
        if isinstance(key, (int, np.integer)):
            F = self.frame_count
            k = int(key)
            if k < -F or k >= F:
                raise IndexError(
                    f"frame index {k} out of range for Bvh with {F} frames")
            i = k if k >= 0 else k + F
            s = slice(i, i + 1)
        elif isinstance(key, slice):
            s = key
        else:
            raise TypeError(
                f"Bvh indices must be int or slice, got {type(key).__name__}. "
                "For array-level writes, assign to bvh.root_pos or bvh.joint_angles directly.")

        # --- value must be a Bvh ---
        if not isinstance(value, Bvh):
            raise TypeError(
                f"Bvh __setitem__ requires a Bvh value, got "
                f"{type(value).__name__}. For array-level writes, assign "
                "to bvh.root_pos or bvh.joint_angles directly.")

        # --- skeleton + frame_time ---
        self._check_same_skeleton(value)
        if self.frame_time != value.frame_time:
            raise ValueError(
                f"frame_time mismatch (self={self.frame_time}, value="
                f"{value.frame_time}). Call value.resample(1/self.frame_time) "
                "first, or overwrite self.frame_time explicitly before splicing.")

        # --- length match (no resizing) ---
        target_len = len(self.root_pos[s])
        if target_len != value.frame_count:
            raise ValueError(
                f"Cannot assign {value.frame_count} frames to a slice of "
                f"length {target_len}; __setitem__ does not resize. Use "
                "`a + b` to append, or frame slicing (`bvh[a:b]`) + manual "
                "array assignment for more complex splicing.")

        # --- in-place splice + explicit cache invalidation ---
        self._root_pos[s] = value.root_pos
        self._joint_angles[s] = value.joint_angles
        self._invalidate_motion_caches()

    def copy(self) -> Bvh:
        return copy.deepcopy(self)

    def _copy_skeleton(self) -> Bvh:
        """Deep-copy the hierarchy and metadata into a zero-frame Bvh.

        Copies ``nodes`` (deeply), ``frame_time``, ``source_path``, the
        ``world_up`` override, and the L/R mapping — but not the motion
        arrays. Frame-producing operations (slicing, concatenation,
        resampling) use this instead of :meth:`copy` so full motion arrays
        aren't deep-copied only to be overwritten immediately.
        """
        new_bvh = Bvh(
            nodes=copy.deepcopy(self.nodes),
            frame_time=self.frame_time,
            source_path=self.source_path,
        )
        new_bvh._world_up_override = self._world_up_override
        new_bvh._lr_mapping = copy.deepcopy(self._lr_mapping)
        new_bvh._lr_mapping_source = self._lr_mapping_source
        return new_bvh

    # ------------------------------------------------------------------
    # Orientation: world_up and forward_at
    # ------------------------------------------------------------------

    @property
    def world_up(self) -> str:
        """Gravity axis of the BVH coordinate system.

        Returned as a signed axis string (``'+y'``, ``'-z'``, etc.).
        Constant per file. Auto-detected from the first animation frame's
        head-above-hips direction, with rest-pose topology as fallback.
        Issues a ``UserWarning`` if the first frame and rest pose disagree.

        The detection specifics, since they decide which answer you get:
        "head" is the first **exact** lowercase name match among
        ``head``, ``neck``, ``chest``, ``spine`` (so a namespaced
        ``mixamorig:Head`` does not match), "hips" is always the root
        node whatever it is called, and the frame-0 reading is accepted
        only when its largest component exceeds twice the second-largest
        — a crouched, lying or leaning first frame is treated as
        ambiguous and silently defers to :attr:`rest_up`. When the rest
        pose is degenerate too, the property warns and returns ``'+y'``.
        Use :attr:`world_up_inferred` to see what the heuristic picks
        while an override is in effect.

        Can be overridden manually via the setter when auto-detection
        produces the wrong answer (e.g. authored BVH files where the rest
        pose convention differs from the animation's world orientation):

            >>> bvh.world_up = '+y'

        The override is preserved through ``copy()``, frame slicing
        (``bvh[a:b]``), and transforms that don't change the world
        coordinate system (``mirror``, ``rotate_vertical``, ``scale``,
        ``translate_root``). ``retarget()`` re-infers from the new skeleton.

        Assign ``'auto'`` or ``None`` to clear a previous override and
        return to auto-detection.

        Note: BVH files do not store a world-up field, so manual overrides
        are lost on write→read round trips and must be re-applied.
        """
        if self._world_up_override is not None:
            return self._world_up_override
        if self._world_up_cached is None:
            self._world_up_cached = _infer_world_up(self)
        return self._world_up_cached

    @world_up.setter
    def world_up(self, value: str | None) -> None:
        if value is None or value == 'auto':
            self._world_up_override = None
            return
        self._world_up_override = _validate_axis_string(value)

    @property
    def world_up_inferred(self) -> str:
        """What the auto heuristic *would* pick, regardless of any override.

        Useful for auditing whether a manual ``bvh.world_up = '+x'``
        override was necessary, or for diagnosing skeletons whose
        animation and rest-pose conventions disagree. Always runs the
        inference fresh; doesn't consult or write the cache.

        Compare against :attr:`world_up` to see whether an override is
        in effect:

            >>> bvh.world_up_inferred  # '+y'  (auto's guess)
            >>> bvh.world_up           # '+z'  (user override)
        """
        return _infer_world_up(self)

    @property
    def up_axis(self) -> Axis:
        """:attr:`world_up` parsed into numeric form — ``Axis(index, sign, vector)``.

        The three fields are the machine-usable views of the same signed axis string (always derived from the resolved :attr:`world_up`, so manual overrides are respected):

        - ``index`` (int): the up coordinate's column (0 = x, 1 = y, 2 = z) in any ``(..., 3)`` position array.
        - ``sign`` (float): ``+1.0`` or ``-1.0`` — multiply the raw coordinate by it to get an up-positive height.
        - ``vector`` (ndarray, shape ``(3,)``): the unit vector along the up direction, sign included (e.g. ``[0, -1, 0]`` for ``'-y'``). A fresh array on every access — mutating it never corrupts the Bvh.

        Example — up-positive heights of every node in every frame:

            >>> coords = bvh.node_positions()
            >>> heights = coords[:, :, bvh.up_axis.index] * bvh.up_axis.sign

        See Also
        --------
        world_up : The signed axis string this is parsed from (settable).
        forward_axis, rest_up_axis : The other two parsed axis properties.
        floor_height : The estimated ground level along this axis, in raw (unsigned) coordinates.
        """
        return parse_axis(self.world_up)

    @property
    def forward_axis(self) -> Axis:
        """:attr:`rest_forward` parsed into numeric form — ``Axis(index, sign, vector)``.

        Never ``None``: :attr:`rest_forward` always resolves, because forward is defined relative to :attr:`world_up` and falls back to an arbitrary-but-stable horizontal axis when the skeleton carries no usable L/R geometry. That fallback is indistinguishable here from a measured result — see :attr:`rest_forward` for the chain.

        See Also
        --------
        rest_forward : The signed axis string this is parsed from.
        up_axis, rest_up_axis : The other two parsed axis properties.
        """
        return parse_axis(self.rest_forward)

    @property
    def rest_up_axis(self) -> Axis | None:
        """:attr:`rest_up` parsed into numeric form — ``Axis(index, sign, vector)``, or ``None``.

        ``None`` exactly when :attr:`rest_up` is ``None`` — a degenerate rest pose (single-node skeleton, or all joints coincident) that carries no directional information. Each parsed axis property mirrors the nullability of the string it parses, so ``bvh.rest_up is None`` and ``bvh.rest_up_axis is None`` always agree.

        Note this is the *topological* up axis. For the world vertical — animation-derived, and always defined — use :attr:`up_axis`.

        See Also
        --------
        rest_up : The signed axis string this is parsed from.
        up_axis, forward_axis : The other two parsed axis properties.
        """
        rest_up = self.rest_up
        return None if rest_up is None else parse_axis(rest_up)

    @property
    def floor_height(self) -> float:
        """Estimated ground-plane height, in raw world coordinates along ``world_up``.

        A single scalar: the floor level in the BVH's own coordinate system,
        signed along the raw up axis (so for ``world_up='-y'`` a floor at raw
        ``y≈5`` returns ``≈5``). It is the 2nd-percentile of the per-frame
        minimum foot height over auto-detected feet (all nodes for footless
        rigs); see :func:`pybvh.analysis._compute_floor_height`. The 2nd
        percentile is the canonical robust estimate — resistant to
        occasional glitched-low frames; for the true minimum, or any other
        convention, call ``foot_contacts(floor="min")`` / pass an explicit
        float per call (this property stays 2nd-percentile). This is the
        scene's ground plane — `foot_contacts` layers a per-foot stance hover on
        top of it.

        Lazily computed and cached; the cache is invalidated whenever
        ``root_pos`` or ``joint_angles`` is reassigned.
        :func:`~pybvh.analysis.foot_contacts` fills/serves this cache on its
        default world-coords + auto-detected-feet path (with explicit
        ``coords=`` or ``foot_joints=`` it estimates its own per-call floor).
        """
        if self._floor_height_cached is None:
            from . import analysis
            self._floor_height_cached = analysis._compute_floor_height(self)
        return self._floor_height_cached

    @property
    def rest_up(self) -> str | None:
        """Skeleton's topological up axis, derived from the rest pose only.

        Read-only. Inspects rest-pose joint offsets (``"head"``,
        ``"neck"``, ``"chest"``, ``"spine"`` in priority order; falls
        back to the axis with the largest offset spread) and returns
        the dominant signed axis. Pose-independent — the animation
        data is never touched.

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
        return _rest_upward(self)

    @property
    def rest_forward(self) -> str:
        """Skeleton's topological forward axis, derived from the rest pose only.

        Read-only. Computes forward from the rest-pose L/R lateral
        geometry crossed with :attr:`world_up`. Pose-independent — the
        animation data is never touched. Complements :attr:`rest_up`
        (rest-pose up axis) and parallels :meth:`forward_at` (animation-
        derived forward at a given frame).

        Use this to check whether a skeleton's rest-pose facing matches
        a dataset convention without having to call
        :func:`reorient_rest_forward` and compare results.

        Returns
        -------
        str
            Signed axis string (e.g. ``'+z'``, ``'-x'``).
        """
        return _compute_forward_at(self, self.rest_pose_positions(), self.world_up)

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

        Detection at construction time runs the extended name heuristic
        (`Left`/`Right` substring, `L`/`R` prefix, `.L`/`.R` suffix,
        `_l`/`_r` suffix, Mixamo `mixamorig:` namespace, numbered `.001`
        duplicates). Skeletons with conventions the heuristic can't parse
        have ``lr_mapping = None`` — in that case, set it explicitly:

            >>> bvh.lr_mapping = {'arm.L': 'arm.R', 'leg.L': 'leg.R'}

        The assigned dict is one-directional; pybvh symmetrizes it
        internally. Either form works on assignment.

        or pass ``lr_mapping=`` at load time:

            >>> bvh = read_bvh_file('weird.bvh', lr_mapping={...})

        Consumers: ``mirror()``, ``forward_at()``, ``facing_frame()``,
        ``_rest_leftward``, ``_compute_forward_at``,
        ``reorient_rest_forward``.

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

    @property
    def has_lr_geometry(self) -> bool:
        """Whether the rest pose carries usable left/right direction.

        ``True`` when the skeleton's L/R joint pairs give a lateral axis
        that is neither degenerate nor parallel to :attr:`world_up` —
        i.e. when the orientation properties are **measuring** this
        skeleton rather than falling back to a default. The check walks
        the same chain :attr:`rest_forward` computes with, so it is
        ``False`` exactly when deriving a facing from the rest pose
        emits the fallback ``UserWarning``. In particular, on a file
        whose rest-pose and animation up axes disagree (the case
        :attr:`world_up` inference warns about), an L/R axis parallel
        to :attr:`world_up` is unusable and reports ``False`` even
        though the pairs themselves exist.

        This is the check :attr:`rest_forward` (and so
        :attr:`forward_axis`, :meth:`forward_at`, :meth:`left_at` and
        ``facing_frame``) cannot express in its own return value. Those
        always yield an axis: with no usable L/R geometry they return an
        arbitrary-but-stable horizontal axis chosen from
        :attr:`world_up` alone, which is indistinguishable from a
        measured result. When it matters whether a facing was derived
        from the data — comparing a skeleton against a dataset
        convention, say, where a fallback would "match" every time —
        check this first.

        Assigning :attr:`lr_mapping` explicitly is what fixes a ``False``
        on a skeleton whose joints simply are not named recognizably.

        This property describes the *rest-pose* chain. The per-frame
        :meth:`forward_at` / :meth:`facing_frame` measure each frame's
        coordinates first and only fall back to this chain on frames
        where that fails, so on a skeleton with zero rest offsets but
        animated L/R separation they can still measure while this
        reports ``False``.

        Example
        -------
            >>> if bvh.has_lr_geometry:
            ...     assert bvh.rest_forward == dataset_convention
            ... else:
            ...     ...  # facing is a default, not a measurement

        See Also
        --------
        rest_forward : The axis whose fallback this reports.
        lr_mapping : The pairs the measurement is derived from.
        """
        return _facing_is_measured(self, self.world_up)

    @lr_mapping.setter
    def lr_mapping(self, value: dict[str, str] | None) -> None:
        if value is None:
            self._lr_mapping = None
            self._lr_mapping_source = None
            return
        self._validate_and_set_lr_mapping(value, source='user')

    @property
    def lr_pairs(self) -> list[tuple[int, int]] | None:
        """Left/right joint pairs as index tuples in ``joint_angles`` space.

        Index-space counterpart of :attr:`lr_mapping`, derived from the
        same cache.  Returns ``None`` when no mapping is available
        (matches the ``lr_mapping`` sentinel — one "no pairs" convention
        across both surfaces).

        Useful for graph construction and array-level ops that index
        joints by position rather than by name.
        """
        if self._lr_mapping is None:
            return None
        j_name2idx = self.joint_index
        pairs: list[tuple[int, int]] = []
        for left, right in self._lr_mapping.items():
            if left in j_name2idx and right in j_name2idx:
                pairs.append((j_name2idx[left], j_name2idx[right]))
        return pairs

    def _validate_and_set_lr_mapping(
        self, mapping: dict[str, str], source: str,
    ) -> None:
        """Validate an lr_mapping dict and set the cache.

        Validation: both names must exist in ``joint_names``; no
        self-pairs; no duplicate names on either side of a pair.
        """
        if not isinstance(mapping, dict):
            raise TypeError(
                f"lr_mapping must be a dict, got {type(mapping).__name__}")
        if not mapping:
            raise ValueError(
                "lr_mapping must have at least one pair; "
                "pass None to clear the mapping.")
        # Accept symmetric input ({L: R, R: L, ...}) — canonicalize to
        # one-directional by deduping pairs by frozenset before
        # validating. Each pair appears exactly once afterward.
        canonical: dict[str, str] = dict(_iter_unique_lr_pairs(mapping))
        joint_name_set = set(self.joint_names)
        lefts_seen: set[str] = set()
        rights_seen: set[str] = set()
        for left, right in canonical.items():
            if not isinstance(left, str) or not isinstance(right, str):
                raise TypeError(
                    f"lr_mapping keys and values must be str, "
                    f"got {type(left).__name__}/{type(right).__name__}")
            if left == right:
                raise ValueError(
                    f"lr_mapping self-pair not allowed: {left!r}")
            if left not in joint_name_set:
                raise ValueError(
                    f"lr_mapping left joint {left!r} not in joint_names")
            if right not in joint_name_set:
                raise ValueError(
                    f"lr_mapping right joint {right!r} not in joint_names")
            if left in lefts_seen or left in rights_seen:
                raise ValueError(
                    f"lr_mapping joint {left!r} appears in multiple pairs")
            if right in lefts_seen or right in rights_seen:
                raise ValueError(
                    f"lr_mapping joint {right!r} appears in multiple pairs")
            lefts_seen.add(left)
            rights_seen.add(right)
        # Stored one-directional (left → right). Symmetric view is
        # produced by the public `lr_mapping` property — this keeps
        # every internal consumer that iterates `_lr_mapping.items()`
        # working with one (left, right) tuple per pair.
        self._lr_mapping = canonical
        self._lr_mapping_source = source

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
        basis as unit vectors.

        Parameters
        ----------
        frame : int, optional
            Frame index (default 0). Must be within the animation range.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates for all frames. When
            provided, skips the per-call forward kinematics — useful for
            computing facing direction across many frames in a hot loop
            (e.g. dataset uniformity diagnostics). The selected frame's
            slice is taken via ``coords[frame]``.

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
        if coords is None:
            frame_coords = self.node_positions(frame=frame)
        else:
            frame_coords = coords[frame]
        return _compute_forward_at(self, frame_coords, self.world_up)

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
        for the continuous per-frame basis as unit vectors.

        Parameters
        ----------
        frame : int, optional
            Frame index (default 0). Must be within the animation range.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates for all frames. When
            provided, skips the per-call forward kinematics. The selected
            frame's slice is taken via ``coords[frame]``.

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
        if coords is None:
            frame_coords = self.node_positions(frame=frame)
        else:
            frame_coords = coords[frame]
        return _compute_left_at(self, frame_coords, self.world_up)

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

    def write(self, filepath: str | Path, verbose: bool = False,
              overwrite: bool = True) -> None:
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
            filepath, world_up=world_up,
            warn_on_world_up_disagreement=warn_on_world_up_disagreement,
            lr_mapping=lr_mapping)

    @classmethod
    def from_df(cls, hier: list[BvhNode] | dict[str, dict], df) -> Bvh:
        """Build a Bvh from a hierarchy description and a motion DataFrame.

        The constructor counterpart of :meth:`to_df_dict` /
        :meth:`to_hierarchy_dict`. Delegates to :func:`pybvh.df_to_bvh`;
        see it for the expected column naming and hierarchy formats.
        """
        from .df_to_bvh import df_to_bvh
        return df_to_bvh(hier, df)


    def _non_end_site_indices(self) -> list[int]:
        """Indices in ``nodes`` order corresponding to non-end-site joints.

        The same indices select the joint-axis subset of any per-node
        array (e.g. :meth:`node_positions` output of shape ``(F, N, 3)``)
        to produce a joint-aligned ``(F, J, 3)``.
        """
        return [i for i, n in enumerate(self.nodes) if not n.is_end_site()]

    def _world_node_positions(self) -> npt.NDArray[np.float64]:
        """World-frame FK positions for all frames — lazily computed, cached.

        The cache is invalidated by :meth:`_invalidate_motion_caches`
        whenever motion data changes. Callers must not mutate the returned
        array; :meth:`node_positions` derives fresh arrays from it.
        """
        if self._node_positions_cached is None:
            self._node_positions_cached = frames_to_node_positions(
                self, root_pos=self.root_pos,
                joint_angles=self.joint_angles, centered="world")
        return self._node_positions_cached

    def node_positions(self, frame: int | None = None, centered: str = "world") -> npt.NDArray[np.float64]:
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
        centered_options = ['skeleton', 'first', 'world']
        if centered not in centered_options:
            raise ValueError(
                f'The value {centered} is not recognized for the centered '
                f'argument. Currently recognized keywords are {centered_options}')

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
                f"{self.frame_count} frames. Use frame=None for all frames.")
        actual = frame if frame >= 0 else frame + self.frame_count
        if self._node_positions_cached is not None:
            world_frame = self._node_positions_cached[actual]
            if centered == "world":
                return world_frame.copy()
            if centered == "skeleton":
                return world_frame - self.root_pos[actual]
            return world_frame - _ground_plane_offset(
                self.root_pos[actual], self.world_up)
        if centered == "first":
            return frames_to_node_positions(
                self, root_pos=self.root_pos[actual],
                joint_angles=self.joint_angles[actual], centered="first",
                up=self.world_up)
        return frames_to_node_positions(
            self, root_pos=self.root_pos[actual],
            joint_angles=self.joint_angles[actual], centered=centered)

    def joint_positions(self, frame: int | None = None, centered: str = "world") -> npt.NDArray[np.float64]:
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
        """
        return frames_to_node_positions(
            self,
            root_pos=np.zeros(3),
            joint_angles=np.zeros((self.joint_count, 3), dtype=np.float64),
            centered="skeleton")

    def rest_pose_angles(self) -> npt.NDArray[np.float64]:
        """Rest-pose joint angles — zeros of shape ``(J, 3)`` (radians).

        Companion of :meth:`rest_pose_positions` in ``joint_angles`` space
        (one single-frame row, matching :attr:`joint_index`).
        """
        return np.zeros((self.joint_count, 3), dtype=np.float64)



    def to_df_dict(self, mode: str = 'euler', centered: str = "world") -> dict[str, npt.NDArray[np.float64]]:
        """Return a dict of arrays for ``pd.DataFrame(result)``.

        Each key is a column name, each value a 1-D NumPy array of
        length ``frame_count``.

        Parameters
        ----------
        mode : str, optional
            ``'euler'`` — columns are ``'JointName_X_rot'`` etc. (default).
            ``'coordinates'`` — columns are ``'JointName_X'`` etc.,
            including end sites.
        centered : str, optional
            ``"world"`` (default), ``"skeleton"``, or ``"first"`` — see
            :meth:`node_positions` for their semantics. Only used when
            ``mode='coordinates'``.

        Returns
        -------
        dict
            Column-name → 1-D array mapping, ready for ``pd.DataFrame()``.
        """
        correct_modes = ['euler', 'coordinates']

        if mode == 'euler':
            return self._get_df_constructor_euler_angles()
        elif mode == 'coordinates':
            return self._get_df_constructor_spatial_coord(centered=centered)
        else : 
            raise ValueError(f'The value {mode} is not recognized for the mode argument.\
                             Currently recognized keywords are {correct_modes}')
        
    
    def _get_df_constructor_euler_angles(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for Euler-angle DataFrame.

        DataFrame columns are in degrees for human readability — the
        ``_rot`` columns are the rad→deg-converted view of the internal
        radians-valued :attr:`joint_angles`.
        """
        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_time

        root = self.root
        for i, ax in enumerate(root.pos_channels):
            result[f'{root.name}_{ax}_pos'] = self.root_pos[:, i]

        # Convert radians → degrees once for the whole array, then slice.
        joint_angles_deg = np.rad2deg(self.joint_angles)
        j_idx = 0
        for node in self.nodes:
            if node.is_end_site():
                continue
            for i, ax in enumerate(node.rot_channels):  # type: ignore[attr-defined]
                result[f'{node.name}_{ax}_rot'] = joint_angles_deg[:, j_idx, i]
            j_idx += 1

        return result

    def _get_df_constructor_spatial_coord(self, centered: str) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for spatial-coordinate DataFrame."""
        spatial_array = self.node_positions(centered=centered)  # (F, N, 3)

        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_time

        for n_idx, node in enumerate(self.nodes):
            for i, ax in enumerate(['X', 'Y', 'Z']):
                result[f'{node.name}_{ax}'] = spatial_array[:, n_idx, i]

        return result



    
    def to_hierarchy_dict(self) -> dict:
        """Return the skeleton hierarchy as a plain dictionary.

        The inverse-direction counterpart of :meth:`from_df`'s ``hier``
        argument.

        Returns
        -------
        dict
            ``{name: {'offset': [...], 'parent': str|None,
            'rot_channels': [...], 'children': [...]}, ...}``.
            Root entries also include ``'pos_channels'``.
            All values are copies (safe to mutate).
        """
        hier_dict: dict[str, dict[str, object]] = {}
        for node in self.nodes:
            entry: dict[str, object] = {'offset': node.offset.copy()}
            if isinstance(node, BvhRoot):
                entry['pos_channels'] = list(node.pos_channels)
            if isinstance(node, BvhJoint):
                entry['rot_channels'] = list(node.rot_channels)
                entry['children'] = [child.name for child in node.children]
            entry['parent'] = None if node.parent is None else node.parent.name
            hier_dict[node.name] = entry
        return hier_dict


    def _create_node_index(self) -> None:
        """Build ``node_index`` mapping node name to its index in ``nodes``."""
        self._node_index = {node.name: i for i, node in enumerate(self.nodes)}

    def _create_joint_index(self) -> None:
        """Build ``joint_index`` mapping joint name to its index in ``joint_angles`` axis 1."""
        self._joint_index = {name: i for i, name in enumerate(self.joint_names)}

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
        return self._node_index

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
        return self._joint_index

    def index(self, name: str, space: Literal['joint', 'node']) -> int:
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
        if space == 'joint':
            return self._joint_index[name]
        if space == 'node':
            return self._node_index[name]
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
                (node_position[id(child)] for child in node.children  # type: ignore[attr-defined]
                 if child.is_end_site()),
                None)
        return tips

    @property
    def joint_names(self) -> list[str]:
        """Names of non-end-site joints in topological order.

        Returns
        -------
        list of str
        """
        return [n.name for n in self.nodes if not n.is_end_site()]

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
            ''.join(n.rot_channels)  # type: ignore[attr-defined]
            for n in self.nodes
            if not n.is_end_site()
        ]

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Skeleton edge list as ``(child_idx, parent_idx)`` tuples.

        Indices use ``joint_angles`` index space (non-end-site joints
        only, matching ``joint_names`` order).  The root joint has no
        parent and produces no edge, so a skeleton with *J* joints
        yields *J - 1* edges.

        See Also
        --------
        node_edges : Same list but in ``nodes`` index space (includes
            end sites) — what graph models over the full visual skeleton
            typically want.
        """
        joints = [n for n in self.nodes if not n.is_end_site()]
        j_name2idx = {j.name: i for i, j in enumerate(joints)}
        edges: list[tuple[int, int]] = []
        for j_idx, joint in enumerate(joints):
            if joint.parent is not None and joint.parent.name in j_name2idx:
                edges.append((j_idx, j_name2idx[joint.parent.name]))
        return edges

    @property
    def node_edges(self) -> list[tuple[int, int]]:
        """Skeleton edge list as ``(child_idx, parent_idx)`` tuples in
        ``nodes`` index space (includes end sites).

        Parallels :attr:`edges` (joint-axis only); use ``node_edges``
        when the downstream graph treats end sites as real leaves
        (visual skeleton, per-bone styling, GCN inputs over the full
        topology).  ``node_edges`` has one more edge per end site than
        ``edges``.
        """
        edges: list[tuple[int, int]] = []
        for i, node in enumerate(self.nodes):
            if node.parent is not None:
                edges.append((i, self.node_index[node.parent.name]))
        return edges



    @overload
    def retarget(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = ..., strict: bool = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def retarget(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = ..., strict: bool = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def retarget(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = None,
                        strict: bool = False, inplace: bool = False) -> Bvh | None:
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

        Returns
        -------
        None or Bvh
        """
        try:
            new_skel_nodes = new_skeleton.nodes
        except AttributeError:
            raise ValueError('new_skeleton must be a Bvh object')

        # Build name → index lookup for the reference skeleton
        newnodes2idx = {n.name: i for i, n in enumerate(new_skel_nodes)}

        if inplace:
            nodes = self.nodes
        else:
            new_bvh = self.copy()
            nodes = new_bvh.nodes

        for node in nodes:
            # Determine the target name in new_skeleton
            if name_mapping and node.name in name_mapping:
                target_name = name_mapping[node.name]
            else:
                target_name = node.name

            if target_name in newnodes2idx:
                node.offset = new_skel_nodes[newnodes2idx[target_name]].offset
            elif strict:
                raise ValueError(
                    f"Node '{node.name}' (mapped to '{target_name}') not found "
                    f"in new_skeleton and strict=True.")
            # else: keep original offset (lenient mode)

        # Offsets changed without going through the motion setters, so
        # FK-derived caches (positions, floor height, world_up) are stale.
        if inplace:
            self._invalidate_motion_caches()
            return None
        new_bvh._invalidate_motion_caches()
        return new_bvh

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

        Returns
        -------
        None or Bvh
        """
        if isinstance(scale, bool) or not isinstance(
                scale, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"scale must be a scalar, got {type(scale).__name__}. "
                "Per-axis scaling is not supported: node offsets are "
                "parent-local, so per-axis world factors do not commute "
                "with joint rotations and would distort the animation.")

        factor = float(scale)
        target = self if inplace else self.copy()
        for node in target.nodes:
            node.offset = node.offset * factor
        target.root_pos = target.root_pos * factor
        if inplace:
            return None
        return target


    @overload
    def change_euler_order(self, order: Union[str, Sequence[str]], joint: str | BvhNode | None = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def change_euler_order(self, order: Union[str, Sequence[str]], joint: str | BvhNode | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def change_euler_order(self, order: Union[str, Sequence[str]], joint: str | BvhNode | None = None, inplace: bool = False) -> Bvh | None:
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
                raise ValueError(
                    "joint should be a string (joint name), a BvhNode object, or None")
            if joint_name not in self.joint_index:
                raise ValueError(
                    f"Joint '{joint_name}' not found among non-end-site nodes.")
            joint_indices = [self.joint_index[joint_name]]

        target = self if inplace else self.copy()
        joints = [n for n in target.nodes if not n.is_end_site()]

        for j_idx in joint_indices:
            node = joints[j_idx]
            old_order = node.rot_channels  # type: ignore[attr-defined]
            if old_order == new_order:
                continue
            # Convert: old Euler → rotmat → new Euler, then write back
            # (private array — the public view is read-only).
            R = rotations.euler_to_rotmat(target.joint_angles[:, j_idx], old_order)
            target._joint_angles[:, j_idx] = rotations.rotmat_to_euler(R, new_order)
            # Update rot_channels (bypasses the freeze check)
            node._set_rot_channels_internal(new_order)  # type: ignore[attr-defined]

        # Cache invalidation is intentionally skipped: the physical
        # rotations — and therefore all FK-derived caches — are unchanged
        # by an Euler-order re-expression.
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
        joint_rotmats = rotations.euler_to_rotmat(
            self.joint_angles, per_joint)
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
                f"but joint data has {joint_rotmats.shape[0]} frames")
        if joint_rotmats.shape[1] != self.joint_count:
            raise ValueError(
                f"Expected {self.joint_count} joints in {param_name}, "
                f"got {joint_rotmats.shape[1]}")
        new_angles = rotations.rotmat_to_euler(joint_rotmats, self.euler_orders)
        self.root_pos = root_pos
        self.joint_angles = new_angles


    @overload
    def from_rotmat(self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_rotmat(self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_rotmat(self, root_pos: npt.ArrayLike, joint_rotmats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
    def from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
            root_pos_arr, rotations.rot6d_to_rotmat(joint_rot6d_arr), "joint_rot6d")
        if inplace:
            return None
        return target


    @overload
    def from_quat(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_quat(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_quat(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
            root_pos_arr, rotations.quat_to_rotmat(joint_quats_arr), "joint_quats")
        if inplace:
            return None
        return target


    @overload
    def from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
            root_pos_arr, rotations.axisangle_to_rotmat(joint_aa_arr), "joint_aa")
        if inplace:
            return None
        return target


    # ----------------------------------------------------------------
    # Frame slicing, concatenation, and resampling
    # ----------------------------------------------------------------

    def _slice_frames(self, start: int | None = None, end: int | None = None, step: int | None = None) -> Bvh:
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
        new_bvh = self._copy_skeleton()
        s = slice(start, end, step)
        new_bvh.root_pos = self.root_pos[s].copy()
        new_bvh.joint_angles = self.joint_angles[s].copy()
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
            raise ValueError(
                f"Node count mismatch: {len(self.nodes)} vs {len(other.nodes)}")
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1.name != n2.name:
                raise ValueError(
                    f"Node name mismatch: '{n1.name}' vs '{n2.name}'")
            if not n1.is_end_site() and not n2.is_end_site():
                if n1.rot_channels != n2.rot_channels:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"Rotation order mismatch for '{n1.name}': "
                        f"{n1.rot_channels} vs {n2.rot_channels}")  # type: ignore[attr-defined]

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
                f"{other.frame_time}. Using self's frame time.")

        new_bvh = self._copy_skeleton()
        new_bvh.root_pos = np.concatenate(
            [self.root_pos, other.root_pos], axis=0)
        new_bvh.joint_angles = np.concatenate(
            [self.joint_angles, other.joint_angles], axis=0)
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
            If ``target_fps`` is not positive.
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

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
        idx_right = np.searchsorted(t_orig, t_new, side='right')
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
        q_left = joint_quats[idx_left]    # (num_new, J, 4)
        q_right = joint_quats[idx_right]  # (num_new, J, 4)

        # Broadcast alpha to (num_new, J) for per-joint SLERP
        alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (num_new, num_joints))
        new_quats = rotations.quat_slerp(q_left, q_right, alpha_jt)

        # Convert back to Euler angles — vectorized across joints via the
        # per-joint-order overload of rotmat_to_euler.
        new_angles = rotations.rotmat_to_euler(
            rotations.quat_to_rotmat(new_quats), self.euler_orders)

        new_bvh = self._copy_skeleton()
        new_bvh.root_pos = new_root_pos
        new_bvh.joint_angles = new_angles
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

        ``source_path``, a manual ``world_up`` override, and a user-set
        ``lr_mapping`` (filtered to pairs whose joints are both kept) are
        preserved on the result.

        Parameters
        ----------
        joint_names : list of str
            Names of joints to keep.  The root must be included.
            End sites are handled automatically (kept if their parent
            is kept, otherwise removed).

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
            raise ValueError(
                f"Root joint '{self.root.name}' must be in joint_names.")

        # --- Build old joint index for each non-end-site node ---
        old_j_idx = {}
        j = 0
        for node in self.nodes:
            if not node.is_end_site():
                old_j_idx[node.name] = j
                j += 1

        # --- For each kept joint, find nearest kept ancestor and
        #     accumulated offset (sum of intermediate offsets) ---
        # Also collect which old joint indices to keep.
        new_nodes = []
        kept_old_j_indices = []
        # Map old node name → new node object (for parent/children wiring)
        new_node_map: dict[str, BvhNode] = {}

        for node in self.nodes:
            if node.is_end_site():
                # Keep end site only if its parent is kept
                if node.parent is not None and node.parent.name in keep_set:
                    # Walk up from this end site accumulating offset
                    # (in case there were removed intermediates — though
                    # end sites are always direct children, just be safe)
                    acc_offset = node.offset.copy()
                    new_end = BvhEndSite(
                        node.name, offset=acc_offset,
                        parent=new_node_map[node.parent.name])
                    new_nodes.append(new_end)
                    new_node_map[node.name] = new_end
                continue

            if node.name not in keep_set:
                continue

            # This is a kept joint. Find its nearest kept ancestor.
            acc_offset = node.offset.copy()
            walker = node.parent
            while walker is not None and walker.name not in keep_set:
                acc_offset = walker.offset + acc_offset
                walker = walker.parent

            if walker is None:
                # This is the root (no parent)
                if isinstance(node, BvhRoot):
                    new_node = BvhRoot(
                        node.name, offset=acc_offset,
                        pos_channels=list(node.pos_channels),
                        rot_channels=list(node.rot_channels),
                        children=[])
                else:
                    raise ValueError(
                        f"Joint '{node.name}' has no kept ancestor and "
                        f"is not the root.")
            else:
                new_parent = new_node_map[walker.name]
                new_node = BvhJoint(  # type: ignore[assignment]
                    node.name, offset=acc_offset,
                    rot_channels=list(node.rot_channels),  # type: ignore[attr-defined]
                    children=[], parent=new_parent)
                new_parent.children = new_parent.children + [new_node]  # type: ignore[attr-defined]

            new_nodes.append(new_node)
            new_node_map[node.name] = new_node
            kept_old_j_indices.append(old_j_idx[node.name])

        # Wire end-site children into their parents
        for node in new_nodes:
            if node.is_end_site() and node.parent is not None:
                parent = node.parent
                if node not in parent.children:  # type: ignore[attr-defined]
                    parent.children = parent.children + [node]  # type: ignore[attr-defined]

        # If a kept joint has no children at all, create an end site
        # using the offset to its nearest original end-site descendant.
        for node in new_nodes:
            if not node.is_end_site() and not node.children:  # type: ignore[attr-defined]
                orig_node = self.nodes[self.node_index[node.name]]
                end_offset = self._find_end_site_offset(orig_node)
                # 'EndSite<name>' is display-only; end-site identity is the class.
                end_site = BvhEndSite(
                    f'EndSite{node.name}', offset=end_offset, parent=node)
                node.children = [end_site]  # type: ignore[attr-defined]
                new_nodes.append(end_site)

        # --- Build new joint_angles by selecting kept columns ---
        new_joint_angles = self.joint_angles[:, kept_old_j_indices, :]

        new_bvh = Bvh(
            nodes=new_nodes,
            root_pos=self.root_pos.copy(),
            joint_angles=new_joint_angles.copy(),
            frame_time=self.frame_time,
            source_path=self.source_path)
        new_bvh._world_up_override = self._world_up_override
        # A user-set L/R mapping survives, filtered to pairs whose joints
        # are both kept. Name-detected mappings are re-derived by the
        # constructor from the reduced skeleton.
        if self._lr_mapping_source == 'user' and self._lr_mapping:
            kept_pairs = {
                left: right for left, right in self._lr_mapping.items()
                if left in keep_set and right in keep_set
            }
            if kept_pairs:
                new_bvh._validate_and_set_lr_mapping(kept_pairs, source='user')
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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

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
            self, in_frames=in_frames, stencil=stencil, pad=pad, degrees=degrees)

    def root_trajectory(
        self,
        up_axis: str | None = None,
        include_velocities: bool = False,
        stencil: str = "central",
        pad: str = "edge",
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Extract root trajectory features.  See :func:`pybvh.analysis.root_trajectory`."""
        from . import analysis
        return analysis.root_trajectory(
            self, up_axis=up_axis,
            include_velocities=include_velocities,
            stencil=stencil, pad=pad, degrees=degrees)

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
            self, representation=representation,
            include_root_pos=include_root_pos,
            include_velocities=include_velocities,
            include_foot_contacts=include_foot_contacts,
            centered=centered, foot_joints=foot_joints,
            stencil=stencil, pad=pad)

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

    def _descriptor_index(self, joint: str, space: str = 'node') -> int:
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
                f"pybvh.geometry / pybvh.analysis API for index-based access.")
        return self.index(joint, space=space)  # type: ignore[arg-type]

    def _positions_or(
        self, coords: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """``coords`` if given, else the (cached) world-frame node positions."""
        if coords is None:
            return self.node_positions()
        return np.asarray(coords, dtype=np.float64)

    def curvature(self, joint: str, stencil: str = "central",
                  pad: str = "edge", *,
                  coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame trajectory curvature of ``joint``. See :func:`pybvh.geometry.curvature`."""
        from . import geometry
        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.curvature(traj, self.frame_time, stencil, pad)

    def torsion(self, joint: str, stencil: str = "central",
                pad: str = "edge", *,
                coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame trajectory torsion of ``joint``. See :func:`pybvh.geometry.torsion`."""
        from . import geometry
        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.torsion(traj, self.frame_time, stencil, pad)

    def movement_phase(self, joint: str, stencil: str = "central",
                       pad: str = "edge", *,
                       coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame movement phase (``speed · curvature``) of ``joint``.

        See :func:`pybvh.geometry.movement_phase`."""
        from . import geometry
        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.movement_phase(traj, self.frame_time, stencil, pad)

    def path_length(self, joint: str, *,
                    coords: npt.NDArray[np.float64] | None = None) -> float:
        """Arc length travelled by ``joint``. See :func:`pybvh.geometry.path_length`."""
        from . import geometry
        return float(geometry.path_length(
            self._positions_or(coords)[:, self._descriptor_index(joint), :]))

    def directness(self, joint: str, *,
                   coords: npt.NDArray[np.float64] | None = None) -> float:
        """Directness of ``joint``'s path (net displacement ÷ path length).

        See :func:`pybvh.geometry.directness`."""
        from . import geometry
        return float(geometry.directness(
            self._positions_or(coords)[:, self._descriptor_index(joint), :]))

    def ground_path(self, joint: str, *,
                    coords: npt.NDArray[np.float64] | None = None) -> "geometry.GroundPath":
        """Ground-plane path of ``joint`` (uses ``world_up``). See :func:`pybvh.geometry.ground_path`."""
        from . import geometry
        traj = self._positions_or(coords)[:, self._descriptor_index(joint), :]
        return geometry.ground_path(traj, _axis_to_vector(self.world_up))

    def inter_joint_distance(
        self, pairs: list[tuple[str, str]], *,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-frame distances between node pairs. See :func:`pybvh.geometry.inter_joint_distance`."""
        from . import geometry
        idx_pairs = [
            [self._descriptor_index(a), self._descriptor_index(b)]
            for a, b in pairs
        ]
        return geometry.inter_joint_distance(self._positions_or(coords), idx_pairs)

    def joint_angle(self, a: str, vertex: str, b: str,
                    degrees: bool = False, *,
                    coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame angle at ``vertex`` in ``a–vertex–b``. See :func:`pybvh.geometry.joint_angle`."""
        from . import geometry
        pos = self._positions_or(coords)
        return geometry.joint_angle(
            pos[:, self._descriptor_index(a)],
            pos[:, self._descriptor_index(vertex)],
            pos[:, self._descriptor_index(b)], degrees=degrees)

    def triangle_area(self, a: str, b: str, c: str, *,
                      coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame area of triangle ``(a, b, c)``. See :func:`pybvh.geometry.triangle_area`."""
        from . import geometry
        pos = self._positions_or(coords)
        return geometry.triangle_area(
            pos[:, self._descriptor_index(a)],
            pos[:, self._descriptor_index(b)],
            pos[:, self._descriptor_index(c)])

    def segment_axis_angle(self, joint_a: str, joint_b: str,
                           degrees: bool = False, *,
                           coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame angle of the bone ``joint_a→joint_b`` to ``world_up``.

        See :func:`pybvh.geometry.segment_axis_angle`."""
        from . import geometry
        pos = self._positions_or(coords)
        seg = (pos[:, self._descriptor_index(joint_b)]
               - pos[:, self._descriptor_index(joint_a)])
        return geometry.segment_axis_angle(
            seg, _axis_to_vector(self.world_up), degrees=degrees)

    def bounding_box(self, *,
                     coords: npt.NDArray[np.float64] | None = None) -> "geometry.BoundingBox":
        """Per-frame axis-aligned bounding box of all nodes. See :func:`pybvh.geometry.bounding_box`."""
        from . import geometry
        return geometry.bounding_box(self._positions_or(coords))

    def bounding_sphere(self, *,
                        coords: npt.NDArray[np.float64] | None = None) -> "geometry.BoundingSphere":
        """Per-frame approximate enclosing sphere of all nodes. See :func:`pybvh.geometry.bounding_sphere`."""
        from . import geometry
        return geometry.bounding_sphere(self._positions_or(coords))

    def bounding_ellipsoid(self, *,
                           coords: npt.NDArray[np.float64] | None = None) -> "geometry.BoundingEllipsoid":
        """Per-frame PCA-aligned bounding ellipsoid of all nodes. See :func:`pybvh.geometry.bounding_ellipsoid`."""
        from . import geometry
        return geometry.bounding_ellipsoid(self._positions_or(coords))

    def center_of_mass(
        self, weights: npt.NDArray[np.float64] | None = None, *,
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

    def verticality(self, *,
                    coords: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """Per-frame height/width ratio along ``world_up``. See :func:`pybvh.geometry.verticality`."""
        from . import geometry
        return geometry.verticality(
            self._positions_or(coords), _axis_to_vector(self.world_up))

    def node_jerk(self, centered: str = "world", in_frames: bool = False,
                  coords: npt.NDArray[np.float64] | None = None,
                  stencil: str = "central", pad: str = "edge") -> npt.NDArray[np.float64]:
        """Per-node position jerk — ``(F, N, 3)``. See :func:`pybvh.analysis.node_jerk`."""
        from . import analysis
        return analysis.node_jerk(self, centered=centered, in_frames=in_frames,
                                  coords=coords, stencil=stencil, pad=pad)

    def joint_jerk(self, centered: str = "world", in_frames: bool = False,
                   coords: npt.NDArray[np.float64] | None = None,
                   stencil: str = "central", pad: str = "edge") -> npt.NDArray[np.float64]:
        """Per-joint position jerk — ``(F, J, 3)``. See :func:`pybvh.analysis.joint_jerk`."""
        from . import analysis
        return analysis.joint_jerk(self, centered=centered, in_frames=in_frames,
                                   coords=coords, stencil=stencil, pad=pad)

    def smoothness(self, joint: str, metric: str = "sparc", *,
                   coords: npt.NDArray[np.float64] | None = None,
                   **kwargs: Any) -> float:
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
        return analysis.smoothness(speed, 1.0 / self.frame_time, metric=metric, **kwargs)

    def velocity_reductions(self, joint: str, *,
                            coords: npt.NDArray[np.float64] | None = None):
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

    def kinetic_energy(self, masses: npt.NDArray[np.float64] | Mapping[str, float] | None = None,
                       centered: str = "world", stencil: str = "central",
                       pad: str = "edge") -> npt.NDArray[np.float64]:
        """Per-frame kinetic energy over joints. ``masses`` may be a ``(J,)`` array
        or a ``{joint_name: mass}`` mapping. See :func:`pybvh.analysis.kinetic_energy`."""
        from . import analysis
        return analysis.kinetic_energy(self, masses=masses, centered=centered,
                                       stencil=stencil, pad=pad)

    def cadence(self, foot_joints: list[str] | None = None,
                *, contacts: npt.NDArray[np.float64] | None = None) -> float:
        """Step rate (onsets/second). See :func:`pybvh.analysis.cadence`."""
        from . import analysis
        return analysis.cadence(self, foot_joints=foot_joints, contacts=contacts)

    def stride_length(self, foot_joints: list[str] | None = None,
                      *, contacts: npt.NDArray[np.float64] | None = None) -> float:
        """Mean stride length. See :func:`pybvh.analysis.stride_length`."""
        from . import analysis
        return analysis.stride_length(self, foot_joints=foot_joints, contacts=contacts)

    def walking_pace(self) -> float:
        """Mean horizontal speed. See :func:`pybvh.analysis.walking_pace`."""
        from . import analysis
        return analysis.walking_pace(self)

    def gait_parameters(self, foot_joints: list[str] | None = None,
                        *, contacts: npt.NDArray[np.float64] | None = None):
        """Spatiotemporal gait parameters. See :func:`pybvh.analysis.gait_parameters`."""
        from . import analysis
        return analysis.gait_parameters(self, foot_joints=foot_joints, contacts=contacts)

    def range_of_motion(self, joint: str) -> npt.NDArray[np.float64]:
        """Peak-to-peak range of ``joint``'s Euler angles — ``(3,)`` per channel.

        Indexes in JOINT space (rotations exist only on joints). See
        :func:`pybvh.analysis.range_of_motion`."""
        from . import analysis
        idx = self._descriptor_index(joint, space='joint')
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
    def add_rotation_noise(self, sigma: float, *, rng: np.random.Generator | None = ..., inplace: Literal[True], wrap: bool = ..., degrees: bool = ...) -> None: ...
    @overload
    def add_rotation_noise(self, sigma: float, rng: np.random.Generator | None = ..., inplace: Literal[False] = ..., wrap: bool = ..., degrees: bool = ...) -> Bvh: ...
    def add_rotation_noise(self, sigma: float, rng: np.random.Generator | None = None, inplace: bool = False, wrap: bool = False, degrees: bool = False) -> Bvh | None:
        """Add Gaussian noise (``sigma`` in radians, or degrees with ``degrees=True``) to joint angles.  See :func:`pybvh.transforms.add_rotation_noise`."""
        from . import transforms
        return transforms.add_rotation_noise(self, sigma, rng=rng, inplace=inplace, wrap=wrap, degrees=degrees)  # type: ignore[call-overload, return-value]

    @overload
    def add_position_noise(self, sigma: float, *, rng: np.random.Generator | None = ..., inplace: Literal[True]) -> None: ...
    @overload
    def add_position_noise(self, sigma: float, rng: np.random.Generator | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def add_position_noise(self, sigma: float, rng: np.random.Generator | None = None, inplace: bool = False) -> Bvh | None:
        """Add Gaussian noise (``sigma`` in the skeleton's length unit) to the root translation.  See :func:`pybvh.transforms.add_position_noise`."""
        from . import transforms
        return transforms.add_position_noise(self, sigma, rng=rng, inplace=inplace)  # type: ignore[call-overload, return-value]

    def perturb_speed(self, factor: float) -> Bvh:
        """Change motion speed by resampling.  See :func:`pybvh.transforms.perturb_speed`."""
        from . import transforms
        return transforms.perturb_speed(self, factor)

    @overload
    def drop_frames(self, drop_rate: float, *, rng: np.random.Generator | None = ..., inplace: Literal[True]) -> None: ...
    @overload
    def drop_frames(self, drop_rate: float, rng: np.random.Generator | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def drop_frames(self, drop_rate: float, rng: np.random.Generator | None = None, inplace: bool = False) -> Bvh | None:
        """Replace dropped frames with SLERP interpolation.  See :func:`pybvh.transforms.drop_frames`."""
        from . import transforms
        return transforms.drop_frames(self, drop_rate, rng=rng, inplace=inplace)  # type: ignore[call-overload, return-value]

    @overload
    def rotate_vertical(self, angle: float, *, up_axis: str | None = ..., degrees: bool = ..., pivot: str | npt.ArrayLike = ..., inplace: Literal[True]) -> None: ...
    @overload
    def rotate_vertical(self, angle: float, up_axis: str | None = ..., degrees: bool = ..., pivot: str | npt.ArrayLike = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def rotate_vertical(self, angle: float, up_axis: str | None = None, degrees: bool = False, pivot: str | npt.ArrayLike = "origin", inplace: bool = False) -> Bvh | None:
        """Rotate entire motion around the vertical axis (``angle`` in radians), about the world origin or ``pivot=``.  See :func:`pybvh.transforms.rotate_vertical`."""
        from . import transforms
        return transforms.rotate_vertical(self, angle, up_axis=up_axis, degrees=degrees, pivot=pivot, inplace=inplace)  # type: ignore[call-overload, return-value]

    @overload
    def mirror(self, *, lr_mapping: dict[str, str] | None = ..., lateral_axis: str | None = ..., inplace: Literal[True]) -> None: ...
    @overload
    def mirror(self, lr_mapping: dict[str, str] | None = ..., lateral_axis: str | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def mirror(self, lr_mapping: dict[str, str] | None = None, lateral_axis: str | None = None, inplace: bool = False) -> Bvh | None:
        """Mirror motion across the lateral plane.  See :func:`pybvh.transforms.mirror`."""
        from . import transforms
        return transforms.mirror(self, lr_mapping=lr_mapping, lateral_axis=lateral_axis, inplace=inplace)  # type: ignore[call-overload, return-value]

    def random_translate_root(self, offset_range: tuple[float, float] = (-100.0, 100.0), rng: np.random.Generator | None = None) -> Bvh:
        """Translate root by a random offset.  See :func:`pybvh.transforms.random_translate_root`."""
        from . import transforms
        return transforms.random_translate_root(self, offset_range=offset_range, rng=rng)

    def random_rotate_vertical(self, angle_range: tuple[float, float] = (-np.pi, np.pi), up_axis: str | None = None, degrees: bool = False, pivot: str | npt.ArrayLike = "origin", rng: np.random.Generator | None = None) -> Bvh:
        """Rotate motion by a random angle around the vertical axis (radians).  See :func:`pybvh.transforms.random_rotate_vertical`."""
        from . import transforms
        return transforms.random_rotate_vertical(self, angle_range=angle_range, up_axis=up_axis, degrees=degrees, pivot=pivot, rng=rng)

    def random_perturb_speed(self, factor_range: tuple[float, float] = (0.8, 1.2), rng: np.random.Generator | None = None) -> Bvh:
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


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


