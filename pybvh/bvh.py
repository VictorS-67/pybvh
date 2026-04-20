from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Literal, Sequence, Union, overload

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .spatial_coord import frames_to_spatial_coords
from . import rotations


class Bvh:
    """Container for BVH motion-capture data.

    The hierarchy is stored as a list of ``BvhNode`` objects (one per
    joint / end-site).  Motion data is stored as two structured arrays:

    - ``root_pos``:     shape ``(F, 3)``    — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)`` — Euler angles (degrees) per joint per frame

    Attributes
    ----------
    nodes : list of BvhNode
        Skeleton hierarchy in topological order.
    root : BvhRoot
        The root node (``nodes[0]``).
    root_pos : ndarray, shape (F, 3)
        Root position per frame.
    joint_angles : ndarray, shape (F, J, 3)
        Euler angles in degrees per joint per frame.
    frame_time : float
        Duration of one frame in seconds.
    frame_count : int
        Number of frames (read-only).
    node_index : dict
        Mapping from node name to its index in ``nodes`` (includes
        end sites). Use this to index ``spatial_coords()`` output
        (shape ``(F, N, 3)``).
    joint_index : dict
        Mapping from joint name to its index in ``joint_angles`` axis 1
        (joint-only, excludes end sites). Use this to index
        ``joint_angles`` (shape ``(F, J, 3)``).
    joint_names : list of str
        Names of non-end-site joints in topological order (read-only).
    joint_count : int
        Number of non-end-site joints (read-only).
    """
    def __init__(
        self,
        nodes: list[BvhNode] | None = None,
        root_pos: npt.ArrayLike | None = None,
        joint_angles: npt.ArrayLike | None = None,
        frame_time: float = 0,
        world_up: str = "auto",
        lr_mapping: dict[str, str] | None = None,
    ) -> None:
        if nodes is None:
            nodes = [BvhRoot()]
        self.nodes = nodes
        self.frame_time = frame_time
        self.root = self.nodes[0]  # type: ignore[assignment]

        # Validate that root position channels are standard XYZ
        if self.root.pos_channels != ['X', 'Y', 'Z']:
            raise ValueError(
                f"Non-standard root position channel order "
                f"{self.root.pos_channels} is not supported. "
                f"Expected ['X', 'Y', 'Z'].")

        # ---------- Determine root_pos / joint_angles ----------
        if root_pos is not None and joint_angles is not None:
            self.root_pos = np.asarray(root_pos, dtype=np.float64)
            self.joint_angles = np.asarray(joint_angles, dtype=np.float64)
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

        # Orientation cache/override. ``_world_up_override`` is set via the
        # public ``world_up`` setter. ``_world_up_cached`` is computed
        # eagerly below (from first animation frame, with rest-pose fallback)
        # so that file-reading paths have a ready-to-use world_up immediately.
        # For empty/partially-constructed Bvhs the eager compute is skipped
        # and the property will lazily compute on first access.
        self._world_up_override: str | None = None
        self._world_up_cached: str | None = None
        if world_up != "auto":
            from .tools import _validate_axis_string
            self._world_up_override = _validate_axis_string(world_up)
        elif self.frame_count > 0 and len(self.nodes) > 1:
            from .tools import _infer_world_up
            self._world_up_cached = _infer_world_up(self)

        # L/R pair mapping — cached. Depends on names + topology only, so
        # no runtime invalidation hooks are needed (no pybvh operation
        # mutates names on an existing Bvh). See also the `lr_mapping`
        # property docstring.
        self._lr_mapping: dict[str, str] | None = None
        self._lr_mapping_source: str | None = None  # 'names' | 'user' | None
        if lr_mapping is not None:
            # B3 — explicit user mapping at construction time
            self._validate_and_set_lr_mapping(lr_mapping, source='user')
        elif len(self.nodes) > 1:
            # Strategy A — eager name-based detection
            from . import transforms as _transforms
            names_mapping = _transforms._detect_lr_mapping_by_names(self)
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
        return self._root_pos
    @root_pos.setter
    def root_pos(self, value: npt.ArrayLike) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"root_pos must have shape (F, 3), got {arr.shape}")
        self._root_pos = arr
        if hasattr(self, '_world_up_cached'):
            self._world_up_cached = None

    @property
    def joint_angles(self) -> npt.NDArray[np.float64]:
        return self._joint_angles
    @joint_angles.setter
    def joint_angles(self, value: npt.ArrayLike) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"joint_angles must have shape (F, J, 3), got {arr.shape}")
        self._joint_angles = arr
        if hasattr(self, '_world_up_cached'):
            self._world_up_cached = None

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
        count_joints =  0
        for node in self.nodes:
            if not node.is_end_site() : count_joints += 1 
        fps = 1.0 / self.frame_time if self.frame_time > 0 else 0.0
        return f'{count_joints} elements in the Hierarchy, {self.frame_count} frames at {fps:.1f} fps (frame_time={self.frame_time:.6f}s)'
        
    def __repr__(self) -> str:
        nodes_str = []
        for node in self.nodes:
            sep = node.__str__().split()
            if sep[0] == 'ROOT':
                nodes_str += [node.__str__()]
            elif sep[0] == 'JOINT':
                nodes_str += [sep[1]]
        nodes_repr = ''.join(str(nodes_str).split("'"))

        frames_str = f'array(root_pos={self.root_pos.shape}, joint_angles={self.joint_angles.shape}, dtype={self.root_pos.dtype})'

        return f'Bvh(nodes={nodes_repr}, frames={frames_str}, frame_time={self.frame_time:.6f})'
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bvh):
            return NotImplemented
        if self.joint_count != other.joint_count:
            return False
        if self.joint_names != other.joint_names:
            return False
        if self.frame_time != other.frame_time:
            return False
        if self.euler_orders != other.euler_orders:
            return False
        if not np.array_equal(self.root_pos, other.root_pos):
            return False
        if not np.array_equal(self.joint_angles, other.joint_angles):
            return False
        return True

    def matches_topology(self, other: Bvh) -> bool:
        """Whether ``self`` and ``other`` share the same skeleton topology.

        Topology is defined as: same ``joint_names`` (order and names) and
        same per-joint ``euler_orders`` (Euler rotation orders). Bone
        offsets and motion data are NOT compared — for that, use ``==``.

        Use this as a precondition check before batching or retargeting.

        Parameters
        ----------
        other : Bvh

        Returns
        -------
        bool
        """
        if not isinstance(other, Bvh):
            return False
        if self.joint_names != other.joint_names:
            return False
        return self.euler_orders == other.euler_orders

    def __len__(self) -> int:
        """Number of frames. Equivalent to ``self.frame_count``."""
        return self.frame_count

    def __getitem__(self, key: int | slice) -> Bvh:
        """Return a new Bvh containing the selected frame(s).

        Integer keys return a single-frame (F=1) Bvh; slice keys delegate
        to :meth:`slice_frames`. Negative indices and reversed slices
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
            return self.slice_frames(i, i + 1)
        if isinstance(key, slice):
            return self.slice_frames(key.start, key.stop, key.step)
        raise TypeError(
            f"Bvh indices must be int or slice, got {type(key).__name__}. "
            "For arbitrary frame selection, use slice_frames() or build a "
            "new Bvh manually from the required root_pos and joint_angles arrays.")

    def __add__(self, other: object) -> Bvh:
        """Concatenate two Bvh clips. Sugar for :meth:`concat`."""
        if not isinstance(other, Bvh):
            return NotImplemented  # type: ignore[return-value]
        return self.concat(other)

    def __iadd__(self, other: object) -> Bvh:
        """In-place concatenation. Grows ``self`` by appending ``other``'s frames.

        Validates skeleton compatibility and warns on ``frame_time`` mismatch
        just like :meth:`concat`. Mutates ``self.root_pos`` and
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
        return self

    def __setitem__(self, key: int | slice, value: Bvh) -> None:
        """Splice frames from another Bvh into ``self`` in place.

        The skeleton of ``value`` must match ``self``, ``value.frame_time``
        must equal ``self.frame_time`` (raises ``ValueError`` otherwise —
        use :meth:`resample` first if they differ), and the slice length
        must equal ``value.frame_count``. Integer keys require
        ``value.frame_count == 1``.

        Assignment goes through the ``root_pos`` and ``joint_angles``
        setters so the world_up cache is invalidated.
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
                "concat() to append, or slice_frames() + manual array "
                "assignment for more complex splicing.")

        # --- apply via setters (triggers cache invalidation) ---
        new_rp = self.root_pos.copy()
        new_ja = self.joint_angles.copy()
        new_rp[s] = value.root_pos
        new_ja[s] = value.joint_angles
        self.root_pos = new_rp
        self.joint_angles = new_ja

    def copy(self) -> Bvh:
        return copy.deepcopy(self)

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

        Can be overridden manually via the setter when auto-detection
        produces the wrong answer (e.g. authored BVH files where the rest
        pose convention differs from the animation's world orientation):

            >>> bvh.world_up = '+y'

        The override is preserved through ``copy()``, ``slice_frames()``,
        and transforms that don't change the world coordinate system
        (``mirror``, ``rotate_vertical``, ``scale``, ``translate_root``).
        ``retarget()`` re-infers from the new skeleton.

        Note: BVH files do not store a world-up field, so manual overrides
        are lost on write→read round trips and must be re-applied.
        """
        if self._world_up_override is not None:
            return self._world_up_override
        if self._world_up_cached is None:
            from .tools import _infer_world_up
            self._world_up_cached = _infer_world_up(self)
        return self._world_up_cached

    @world_up.setter
    def world_up(self, value: str) -> None:
        from .tools import _validate_axis_string
        self._world_up_override = _validate_axis_string(value)

    @property
    def rest_up(self) -> str:
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
        str
            Signed axis string (e.g. ``'+y'``, ``'+z'``).
        """
        from .tools import _rest_upward
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
        from .tools import _compute_forward_at
        return _compute_forward_at(self, self.rest_pose_coords(), self.world_up)

    @property
    def lr_mapping(self) -> dict[str, str] | None:
        """Left/right joint pair mapping for this skeleton.

        A dict of ``{left_joint_name: right_joint_name, ...}`` describing
        the skeleton's bilateral symmetry pairs. ``None`` if no pairs
        could be auto-detected and no explicit mapping was provided.

        Detection at construction time runs the extended name heuristic
        (`Left`/`Right` substring, `L`/`R` prefix, `.L`/`.R` suffix,
        `_l`/`_r` suffix, Mixamo `mixamorig:` namespace, numbered `.001`
        duplicates). Skeletons with conventions the heuristic can't parse
        have ``lr_mapping = None`` — in that case, set it explicitly:

            >>> bvh.lr_mapping = {'arm.L': 'arm.R', 'leg.L': 'leg.R'}

        or pass ``lr_mapping=`` at load time:

            >>> bvh = read_bvh_file('weird.bvh', lr_mapping={...})

        Consumers: ``mirror()``, ``forward_at()``, ``_rest_leftward``,
        ``_compute_forward_at``, ``reorient_rest_forward``.

        Note: BVH files don't store L/R pair info, so user-set mappings
        are lost on ``bvh.write()`` round-trips — same wart as
        ``world_up``. Re-apply after reading.
        """
        return self._lr_mapping

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
        joint_name_set = set(self.joint_names)
        lefts_seen: set[str] = set()
        rights_seen: set[str] = set()
        for left, right in mapping.items():
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
        self._lr_mapping = dict(mapping)
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
        """
        from .tools import _compute_forward_at
        if coords is None:
            frame_coords = self.spatial_coords(frame_num=frame)
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
        world_up : World vertical axis.
        """
        from .tools import (
            _axis_to_vector, _world_leftward_unit_at_frame, _rest_leftward,
            get_main_direction,
        )
        world_up = self.world_up
        if coords is None:
            frame_coords = self.spatial_coords(frame_num=frame)
        else:
            frame_coords = coords[frame]
        left_vec = _world_leftward_unit_at_frame(self, frame_coords, world_up)
        if left_vec is None:
            rest_left = _rest_leftward(self)
            if rest_left is None:
                # No L/R information at all — pick an arbitrary horizontal
                # axis consistent with the forward_at fallback.
                fwd_fallback = {'y': '+z', 'z': '+x', 'x': '+y'}[world_up[1]]
                fwd_vec = _axis_to_vector(fwd_fallback)
                up_vec = _axis_to_vector(world_up)
                left_ax = get_main_direction(np.cross(up_vec, fwd_vec))
                assert left_ax is not None  # axis-aligned cross never degenerate
                return left_ax
            return rest_left
        left_ax = get_main_direction(left_vec)
        if left_ax is None or left_ax[1] == world_up[1]:
            fwd_fallback = {'y': '+z', 'z': '+x', 'x': '+y'}[world_up[1]]
            fwd_vec = _axis_to_vector(fwd_fallback)
            up_vec = _axis_to_vector(world_up)
            left_ax = get_main_direction(np.cross(up_vec, fwd_vec))
            assert left_ax is not None
        return left_ax

    def write(self, new_filepath: str | Path, verbose: bool = False) -> None:
        """Write the Bvh object to a ``.bvh`` file.  See :func:`pybvh.io.write_bvh_file`."""
        from . import io
        io.write_bvh_file(self, new_filepath, verbose=verbose)


    def spatial_coords(self, frame_num: int = -1, centered: str = "world") -> npt.NDArray[np.float64]:
        """
        Obtain the spatial coordinates of the joints.

        Returns an ndarray of shape ``(N, 3)`` for a single frame or
        ``(F, N, 3)`` for all frames, where *N* is the total number of
        nodes (joints + end sites).

        Parameters
        ----------
        frame_num : int
            Frame index to return.  ``-1`` (default) returns all frames.
        centered : str
            ``"world"`` – root at actual position.
            ``"skeleton"`` – root at origin for all frames.
            ``"first"`` – first-frame root at origin, then moves normally.
        """
        centered_options = ['skeleton', 'first', 'world']
        if centered not in centered_options:
            raise ValueError(
                f'The value {centered} is not recognized for the centered '
                f'argument. Currently recognized keywords are {centered_options}')

        if frame_num == -1:
            # Sentinel for "all frames" — distinct from a negative index,
            # which would return a single frame counted from the end.
            return frames_to_spatial_coords(
                self, root_pos=self.root_pos,
                joint_angles=self.joint_angles, centered=centered)
        if not -self.frame_count <= frame_num < self.frame_count:
            raise IndexError(
                f"frame_num {frame_num} is out of range for "
                f"{self.frame_count} frames. Use -1 for all frames.")
        actual = frame_num if frame_num >= 0 else frame_num + self.frame_count
        return frames_to_spatial_coords(
            self, root_pos=self.root_pos[actual],
            joint_angles=self.joint_angles[actual], centered=centered)

        

    @overload
    def rest_pose_coords(self, mode: Literal['coordinates'] = ...) -> npt.NDArray[np.float64]: ...
    @overload
    def rest_pose_coords(self, mode: Literal['euler']) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def rest_pose_coords(self, mode: str = 'coordinates') -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Return the rest pose of the skeleton (all angles zero, root at origin).

        Parameters
        ----------
        mode : str
            ``'euler'`` – return a tuple ``(root_pos, joint_angles)`` of zeros
            matching the structured shapes.
            ``'coordinates'`` – return spatial coordinates as ``(N, 3)``.
        """
        correct_modes = ['euler', 'coordinates']
        if mode == 'euler':
            return np.zeros(3, dtype=np.float64), np.zeros_like(self.joint_angles[0])
        elif mode == 'coordinates':
            return frames_to_spatial_coords(
                self,
                root_pos=np.zeros(3),
                joint_angles=np.zeros_like(self.joint_angles[0]),
                centered="skeleton")
        else:
            raise ValueError(
                f'The value {mode} is not recognized for the mode argument. '
                f'Currently recognized keywords are {correct_modes}')
        

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
            ``"world"`` (default), ``"skeleton"``, or ``"first"``.
            Only used when ``mode='coordinates'``.

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
        """Return column-name → array dict for Euler-angle DataFrame."""
        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_time

        root = self.root
        for i, ax in enumerate(root.pos_channels):
            result[f'{root.name}_{ax}_pos'] = self.root_pos[:, i]

        j_idx = 0
        for node in self.nodes:
            if node.is_end_site():
                continue
            for i, ax in enumerate(node.rot_channels):  # type: ignore[attr-defined]
                result[f'{node.name}_{ax}_rot'] = self.joint_angles[:, j_idx, i]
            j_idx += 1

        return result

    def _get_df_constructor_spatial_coord(self, centered: str) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for spatial-coordinate DataFrame."""
        spatial_array = self.spatial_coords(centered=centered)  # (F, N, 3)

        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_time

        for n_idx, node in enumerate(self.nodes):
            for i, ax in enumerate(['X', 'Y', 'Z']):
                result[f'{node.name}_{ax}'] = spatial_array[:, n_idx, i]

        return result



    
    def hierarchy_info_as_dict(self) -> dict:
        """Return the skeleton hierarchy as a plain dictionary.

        Returns
        -------
        dict
            ``{name: {'offset': [...], 'parent': str|None,
            'rot_channels': [...], 'children': [...]}, ...}``.
            Root entries also include ``'pos_channels'``.
            The returned dict is a deep copy (safe to mutate).
        """
        hier_dict: dict[str, dict[str, object]] = {}
        for node in self.nodes:
            hier_dict[node.name] = {'offset' : node.offset}
            if isinstance(node, BvhRoot):
                hier_dict[node.name]['pos_channels'] = node.pos_channels
            if isinstance(node, BvhJoint):
                hier_dict[node.name]['rot_channels'] = node.rot_channels
                hier_dict[node.name]['children'] = [child.name for child in node.children]
            hier_dict[node.name]['parent'] = None if node.parent is None else node.parent.name
            
        return copy.deepcopy(hier_dict)
    
    def _create_node_index(self) -> None:
        """Build ``node_index`` mapping node name to its index in ``nodes``."""
        self._node_index = {node.name: i for i, node in enumerate(self.nodes)}

    def _create_joint_index(self) -> None:
        """Build ``joint_index`` mapping joint name to its index in ``joint_angles`` axis 1."""
        self._joint_index = {name: i for i, name in enumerate(self.joint_names)}

    @property
    def node_index(self) -> dict[str, int]:
        """Mapping from node name to its integer index in ``nodes``.

        Indexes the output of :meth:`spatial_coords` (shape
        ``(F, N, 3)``), which includes end sites.  For indexing
        :attr:`joint_angles` (shape ``(F, J, 3)``, excludes end sites),
        use :attr:`joint_index` instead.

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

        Returns
        -------
        dict
            ``{joint_name: int}`` for every non-end-site joint.
            Values cover ``range(joint_count)``.
        """
        return self._joint_index

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

        if inplace:
            return None
        return new_bvh

    @overload
    def scale(self, scale: float | npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def scale(self, scale: float | npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def scale(self, scale: float | npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """Scale all node offsets by a factor.

        Parameters
        ----------
        scale : float or array_like of shape (3,)
            Uniform scalar or per-axis scale factors.
        inplace : bool, optional
            If True, modify self and return None.
            If False (default), return a modified copy.

        Returns
        -------
        None or Bvh
        """
        if isinstance(scale, (int, float)):
            scale_arr: npt.NDArray[np.float64] = np.array([scale, scale, scale], dtype=np.float64)
        else:
            scale_arr = np.asarray(scale, dtype=np.float64)
            if scale_arr.shape != (3,):
                raise ValueError('The scale argument should be a float, or a list/np array of 3 floats')


        if inplace:
            for node in self.nodes:
                node.offset = node.offset * scale_arr
            self.root_pos = self.root_pos * scale_arr
            return None

        else:
            new_bvh = self.copy()
            for node in new_bvh.nodes:
                node.offset = node.offset * scale_arr
            new_bvh.root_pos = new_bvh.root_pos * scale_arr
            return new_bvh
        

        

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
            new_order_list = list(order.upper())
        else:
            new_order_list = [c.upper() for c in order]

        if joint is not None:
            # --- Single joint mode ---
            if isinstance(joint, BvhNode):
                joint_name = joint.name
            elif isinstance(joint, str):
                joint_name = joint
            else:
                raise ValueError("joint should be a string (joint name), a BvhNode object, or None")

            found_joint: BvhNode | None = None
            for node in self.nodes:
                if not node.is_end_site() and node.name == joint_name:
                    found_joint = node
                    break
            if found_joint is None:
                raise ValueError(f"Joint '{joint_name}' not found among non-end-site nodes.")

            old_order = found_joint.rot_channels  # type: ignore[attr-defined]

            # If the order is already the same, nothing to do
            if old_order == new_order_list:
                return None if inplace else self.copy()

            target = self if inplace else self.copy()

            # Find the joint index in joint_angles
            j_idx = 0
            target_joint = None
            for node in target.nodes:
                if node.is_end_site():
                    continue
                if node.name == joint_name:
                    target_joint = node
                    break
                j_idx += 1

            # Convert: old Euler → rotmat → new Euler
            angles_old = target.joint_angles[:, j_idx]  # (num_frames, 3) degrees
            R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
            angles_new = rotations.rotmat_to_euler(R, new_order_list, degrees=True)

            # Write new angles back
            target.joint_angles[:, j_idx] = angles_new

            # Update node's rot_channels (bypass freeze check)
            target_joint._set_rot_channels_internal(new_order_list)  # type: ignore[union-attr]

            if inplace:
                return None
            return target

        else:
            # --- All joints mode ---
            target = self if inplace else self.copy()

            j_idx = 0
            for node in target.nodes:
                if node.is_end_site():
                    continue

                old_order = node.rot_channels  # type: ignore[attr-defined]
                if old_order != new_order_list:
                    angles_old = target.joint_angles[:, j_idx]
                    R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
                    angles_new = rotations.rotmat_to_euler(R, new_order_list, degrees=True)
                    target.joint_angles[:, j_idx] = angles_new
                    node._set_rot_channels_internal(new_order_list)  # type: ignore[attr-defined]
                j_idx += 1

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
            self.joint_angles, per_joint, degrees=True)
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


    def to_quaternions(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert all per-joint Euler angles to quaternions.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_quats : ndarray, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) for each joint in each frame.

        Notes
        -----
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

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_rot6d_arr: npt.NDArray[np.float64] = np.asarray(joint_rot6d, dtype=np.float64)

        if root_pos_arr.shape[0] != joint_rot6d_arr.shape[0]:
            raise ValueError(
                f"Frame count mismatch: root_pos has {root_pos_arr.shape[0]} frames "
                f"but joint data has {joint_rot6d_arr.shape[0]} frames")

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_rot6d_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_rot6d, "
                f"got {joint_rot6d_arr.shape[1]}")

        # Convert 6D -> rotation matrices -> Euler angles per joint
        joint_rotmats = rotations.rot6d_to_rotmat(joint_rot6d_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

        if inplace:
            return None
        return target


    @overload
    def from_quaternions(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def from_quaternions(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def from_quaternions(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_quats_arr: npt.NDArray[np.float64] = np.asarray(joint_quats, dtype=np.float64)

        if root_pos_arr.shape[0] != joint_quats_arr.shape[0]:
            raise ValueError(
                f"Frame count mismatch: root_pos has {root_pos_arr.shape[0]} frames "
                f"but joint data has {joint_quats_arr.shape[0]} frames")

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_quats_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_quats, "
                f"got {joint_quats_arr.shape[1]}")

        joint_rotmats = rotations.quat_to_rotmat(joint_quats_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

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

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_aa_arr: npt.NDArray[np.float64] = np.asarray(joint_aa, dtype=np.float64)

        if root_pos_arr.shape[0] != joint_aa_arr.shape[0]:
            raise ValueError(
                f"Frame count mismatch: root_pos has {root_pos_arr.shape[0]} frames "
                f"but joint data has {joint_aa_arr.shape[0]} frames")

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_aa_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_aa, "
                f"got {joint_aa_arr.shape[1]}")

        joint_rotmats = rotations.axisangle_to_rotmat(joint_aa_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

        if inplace:
            return None
        return target


    # ----------------------------------------------------------------
    # Frame slicing, concatenation, and resampling
    # ----------------------------------------------------------------

    def slice_frames(self, start: int | None = None, end: int | None = None, step: int | None = None) -> Bvh:
        """Return a new Bvh with a slice of frames.

        Parameters
        ----------
        start, end, step : int or None
            Slice parameters (same semantics as ``array[start:end:step]``).

        Returns
        -------
        Bvh
            New Bvh object with the sliced frames and same skeleton.
        """
        new_bvh = self.copy()
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
        Shared by ``concat``, ``__iadd__``, and ``__setitem__``.
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

    def concat(self, other: Bvh) -> Bvh:
        """Concatenate frames from another Bvh with the same skeleton.

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

        new_bvh = self.copy()
        new_bvh.root_pos = np.concatenate(
            [self.root_pos, other.root_pos], axis=0)
        new_bvh.joint_angles = np.concatenate(
            [self.joint_angles, other.joint_angles], axis=0)
        return new_bvh

    def resample(self, target_fps: float) -> Bvh:
        """Resample frames to a new frame rate via interpolation.

        Root position is linearly interpolated.  Joint rotations are
        converted to quaternions and interpolated with SLERP for
        smooth, gimbal-lock-free results.  This is the rotation-aware
        alternative to naive per-channel linear interpolation on Euler
        angles, which produces wobble and gimbal-lock artifacts.

        Parameters
        ----------
        target_fps : float
            Target frames per second.

        Returns
        -------
        Bvh
            New Bvh with resampled frames.
        """
        if self.frame_count < 2:
            return self.copy()

        # Original and target timestamps
        t_orig = np.arange(self.frame_count) * self.frame_time
        new_freq = 1.0 / target_fps
        t_new = np.arange(0, t_orig[-1] + 1e-12, new_freq)
        # Clip to avoid floating-point overshoot
        t_new = t_new[t_new <= t_orig[-1] + 1e-12]

        num_new = len(t_new)
        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)

        # --- Root position: linear interpolation ---
        new_root_pos = np.empty((num_new, 3), dtype=np.float64)
        for ax in range(3):
            new_root_pos[:, ax] = np.interp(t_new, t_orig, self.root_pos[:, ax])

        # --- Joint angles: quaternion SLERP ---
        # Convert all joints to quaternions: (F, J, 4)
        _, joint_quats = self.to_quaternions()

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

        # Convert back to Euler angles per joint
        new_angles = np.empty((num_new, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                rotations.quat_to_rotmat(new_quats[:, j_idx]),
                order, degrees=True)

        new_bvh = self.copy()
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
                    new_end = BvhNode(
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
                # Find original end-site descendant
                orig_node = None
                for n in self.nodes:
                    if n.name == node.name:
                        orig_node = n
                        break
                end_offset = self._find_end_site_offset(orig_node)  # type: ignore[arg-type]
                end_site = BvhNode(
                    f'End Site {node.name}', offset=end_offset, parent=node)
                node.children = [end_site]  # type: ignore[attr-defined]
                new_nodes.append(end_site)

        # --- Build new joint_angles by selecting kept columns ---
        new_joint_angles = self.joint_angles[:, kept_old_j_indices, :]

        new_bvh = Bvh(
            nodes=new_nodes,
            root_pos=self.root_pos.copy(),
            joint_angles=new_joint_angles.copy(),
            frame_time=self.frame_time)
        new_bvh._world_up_override = self._world_up_override
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
    #  ML Pipeline Features (delegate to features module)
    # ----------------------------------------------------------------

    def joint_velocities(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
        stencil: str = "central",
        pad: str = "edge",
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint position velocities.  See :func:`pybvh.features.joint_velocities`."""
        from . import features
        return features.joint_velocities(
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
        """Compute per-joint position accelerations.  See :func:`pybvh.features.joint_accelerations`."""
        from . import features
        return features.joint_accelerations(
            self, centered=centered, in_frames=in_frames, coords=coords,
            stencil=stencil, pad=pad)

    def angular_velocities(
        self,
        in_frames: bool = False,
        stencil: str = "central",
        pad: str = "edge",
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint angular velocities.  See :func:`pybvh.features.angular_velocities`."""
        from . import features
        return features.angular_velocities(
            self, in_frames=in_frames, stencil=stencil, pad=pad, degrees=degrees)

    def root_trajectory(
        self,
        up_axis: str | None = None,
        include_velocities: bool = False,
        stencil: str = "central",
        pad: str = "edge",
        degrees: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Extract root trajectory features.  See :func:`pybvh.features.root_trajectory`."""
        from . import features
        return features.root_trajectory(
            self, up_axis=up_axis,
            include_velocities=include_velocities,
            stencil=stencil, pad=pad, degrees=degrees)

    def foot_contacts(
        self,
        foot_joints: list[str] | None = None,
        method: str = "combined",
        centered: str = "world",
        coords: npt.NDArray[np.float64] | None = None,
        *,
        vel_threshold: float | None = None,
        height_threshold: float | None = None,
        floor: float | str = "auto",
        min_contact_duration: float = 0.1,
        min_gap_duration: float = 0.1,
        return_info: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]:
        """Detect foot contact labels.  See :func:`pybvh.features.foot_contacts`."""
        from . import features
        return features.foot_contacts(
            self,
            foot_joints=foot_joints,
            method=method,
            centered=centered,
            coords=coords,
            vel_threshold=vel_threshold,
            height_threshold=height_threshold,
            floor=floor,
            min_contact_duration=min_contact_duration,
            min_gap_duration=min_gap_duration,
            return_info=return_info,
        )

    def auto_detect_foot_joints(self) -> list[str]:
        """Auto-detect foot joint names from skeleton topology.  See :func:`pybvh.features.auto_detect_foot_joints`."""
        from . import features
        return features.auto_detect_foot_joints(self)

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
            num_nodes=len(self.nodes),
            num_feet=num_feet,
            representation=representation,
            include_root_pos=include_root_pos,
            include_velocities=include_velocities,
            include_foot_contacts=include_foot_contacts,
        )

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
    def add_noise(self, sigma_deg: float, *, sigma_pos: float = ..., rng: np.random.Generator | None = ..., inplace: Literal[True], wrap: bool = ...) -> None: ...
    @overload
    def add_noise(self, sigma_deg: float, sigma_pos: float = ..., rng: np.random.Generator | None = ..., inplace: Literal[False] = ..., wrap: bool = ...) -> Bvh: ...
    def add_noise(self, sigma_deg: float, sigma_pos: float = 0.0, rng: np.random.Generator | None = None, inplace: bool = False, wrap: bool = True) -> Bvh | None:
        """Add Gaussian noise to joint angles.  See :func:`pybvh.transforms.add_noise`."""
        from . import transforms
        return transforms.add_noise(self, sigma_deg, sigma_pos=sigma_pos, rng=rng, inplace=inplace, wrap=wrap)  # type: ignore[call-overload, return-value]

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
    def rotate_vertical(self, angle_deg: float, *, up_axis: str | None = ..., inplace: Literal[True]) -> None: ...
    @overload
    def rotate_vertical(self, angle_deg: float, up_axis: str | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def rotate_vertical(self, angle_deg: float, up_axis: str | None = None, inplace: bool = False) -> Bvh | None:
        """Rotate entire motion around the vertical axis.  See :func:`pybvh.transforms.rotate_vertical`."""
        from . import transforms
        return transforms.rotate_vertical(self, angle_deg, up_axis=up_axis, inplace=inplace)  # type: ignore[call-overload, return-value]

    @overload
    def mirror(self, *, left_right_mapping: dict[str, str] | None = ..., lateral_axis: str | None = ..., inplace: Literal[True]) -> None: ...
    @overload
    def mirror(self, left_right_mapping: dict[str, str] | None = ..., lateral_axis: str | None = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def mirror(self, left_right_mapping: dict[str, str] | None = None, lateral_axis: str | None = None, inplace: bool = False) -> Bvh | None:
        """Mirror motion across the lateral plane.  See :func:`pybvh.transforms.mirror`."""
        from . import transforms
        return transforms.mirror(self, left_right_mapping=left_right_mapping, lateral_axis=lateral_axis, inplace=inplace)  # type: ignore[call-overload, return-value]

    def random_translate_root(self, range_xyz: tuple[float, float] = (-100.0, 100.0), rng: np.random.Generator | None = None) -> Bvh:
        """Translate root by a random offset.  See :func:`pybvh.transforms.random_translate_root`."""
        from . import transforms
        return transforms.random_translate_root(self, range_xyz=range_xyz, rng=rng)

    def random_rotate_vertical(self, angle_range: tuple[float, float] = (-180.0, 180.0), up_axis: str | None = None, rng: np.random.Generator | None = None) -> Bvh:
        """Rotate motion by a random angle around the vertical axis.  See :func:`pybvh.transforms.random_rotate_vertical`."""
        from . import transforms
        return transforms.random_rotate_vertical(self, angle_range=angle_range, up_axis=up_axis, rng=rng)

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

    def render(self, output_path, **kwargs):
        """Render animation to file. See :func:`pybvh.bvhplot.render`."""
        from . import bvhplot
        return bvhplot.render(self, output_path, **kwargs)

    def play(self, **kwargs):
        """Interactive playback. See :func:`pybvh.bvhplot.play`."""
        from . import bvhplot
        return bvhplot.play(self, **kwargs)


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


