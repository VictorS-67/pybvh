from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Union

import numpy as np
import numpy.typing as npt

from .rotations import euler_to_rotmat
from .tools import _validate_axis_string
from .bvhnode import BvhNode

if TYPE_CHECKING:
    from .bvh import Bvh


class _FkTopologyFields(NamedTuple):
    offsets: npt.NDArray[np.float64]
    parent_idx: npt.NDArray[np.intp]
    joint_idx: npt.NDArray[np.intp]
    euler_orders: list[str]


class FkTopology(_FkTopologyFields):
    """A skeleton's topology as plain arrays — everything forward kinematics reads.

    The array-signature counterpart of a node tree: :func:`frames_to_node_positions` accepts one of these in place of a :class:`~pybvh.Bvh` or a ``list[BvhNode]``, so a caller holding only arrays (a data loader reading a preprocessed dataset, an augmentation step that has no source file open) can run forward kinematics without reconstructing node objects.

    This is an **FK input bundle, not a skeleton descriptor**: it carries what the FK loop reads and nothing else — no names, no orientation axes, no channel layout. Every field is independently serializable (three arrays and a list of strings), so a preprocessing step can store the four values in whatever container it already uses; the type itself is not a serialization format. For anything else about a skeleton, keep the :class:`~pybvh.Bvh`.

    Instances validate on construction (see *Raises*), because the train-time
    caller builds one from arrays with no node tree to check against. A
    malformed topology raises here rather than producing silently wrong
    geometry: two of the failure modes are indistinguishable from valid input
    downstream — a root marked as an end site reads ``joint_idx == -1`` as a
    *negative index* into the rotation array, and a node parented to an end
    site reads uninitialized memory for its parent's accumulated rotation.

    Attributes
    ----------
    offsets : ndarray, shape (N, 3)
        Each node's rest-pose offset from its parent, in the skeleton's
        length unit. Node order is the caller's; it need not be
        depth-first, but parents must precede children (see *Raises*).
    parent_idx : ndarray, shape (N,)
        Index of each node's parent, ``-1`` for the root. Exactly one
        ``-1``, and it is necessarily at index 0.
    joint_idx : ndarray, shape (N,)
        Index of each node's rotation along ``joint_angles`` axis 1, and
        ``-1`` for end sites — which is also what marks a node as an end
        site; there is no separate flag.
    euler_orders : list of str
        Per-joint Euler order, e.g. ``['ZYX', ...]``, length *J*.

        **Indexed by joint column, not by node.** ``euler_orders[j]`` is
        the order of the node whose ``joint_idx == j``. The two coincide
        only when ``joint_idx`` counts up in node order, which is what
        :meth:`from_nodes` and :attr:`Bvh.fk_topology` produce. A caller
        that permutes the joint columns — packing vertices into a graph
        layout, say — must permute this list the same way. No check can
        catch a mismatch: every permutation is a valid topology, just not
        the one you meant.

    Raises
    ------
    ValueError
        If the arrays disagree on ``N``, ``offsets`` is not ``(N, 3)``, an
        Euler order is not a permutation of ``'XYZ'``, or any of the
        following invariants is broken:

        - **Parents precede children** (``parent_idx[i] < i``). The FK
          loop fills node ``i`` while reading its parent's already-written
          row, so a forward reference reads an unwritten one.
        - **Exactly one root** (one ``-1`` in ``parent_idx``).
        - **The root is a joint** (``joint_idx[0] >= 0``). A ``-1`` there
          is a valid *negative index* into the rotation array, so it
          silently applies the last joint's rotation to the whole
          skeleton.
        - **No node is parented to an end site.** End sites accumulate no
          rotation, so their children would read an uninitialized frame.
        - **Joint columns are a complete range** — the non-negative
          ``joint_idx`` values are exactly ``0..J-1``, each once — and
          ``len(euler_orders) == J``.

    See Also
    --------
    Bvh.fk_topology : Derive one from a loaded skeleton.
    from_nodes : Derive one from a bare node list.
    frames_to_node_positions : The FK entry point that consumes it.

    Example
    -------
    >>> topology = bvh.fk_topology            # at preprocessing time
    >>> np.savez(path, offsets=topology.offsets, parent_idx=topology.parent_idx,
    ...          joint_idx=topology.joint_idx, euler_orders=topology.euler_orders)
    >>> ...                                   # at train time, no Bvh in sight
    >>> d = np.load(path)
    >>> topology = FkTopology(d['offsets'], d['parent_idx'], d['joint_idx'],
    ...                       list(d['euler_orders']))
    >>> coords = frames_to_node_positions(topology, root_pos, joint_angles)
    """

    __slots__ = ()

    def __new__(
        cls,
        offsets: npt.ArrayLike,
        parent_idx: npt.ArrayLike,
        joint_idx: npt.ArrayLike,
        euler_orders: list[str],
    ) -> FkTopology:
        offsets_arr = np.asarray(offsets, dtype=np.float64)
        parent_arr = np.asarray(parent_idx, dtype=np.intp)
        joint_arr = np.asarray(joint_idx, dtype=np.intp)
        orders = [str(order) for order in euler_orders]
        _validate_fk_topology(offsets_arr, parent_arr, joint_arr, orders)
        return super().__new__(cls, offsets_arr, parent_arr, joint_arr, orders)

    def _replace(self, **changes: object) -> FkTopology:
        """Return a copy with fields replaced, revalidated.

        Overridden so the same invariants hold for a derived topology —
        the natural way to write a bone-length augmentation is
        ``topology._replace(offsets=scaled)``, and that must not be a way
        around the constructor.
        """
        merged = self._asdict()
        unknown = set(changes) - set(merged)
        if unknown:
            raise ValueError(
                f"FkTopology has no field(s) {sorted(unknown)}; "
                f"expected any of {list(merged)}")
        merged.update(changes)
        return type(self)(**merged)  # type: ignore[arg-type]

    @classmethod
    def from_nodes(cls, nodes: list[BvhNode]) -> FkTopology:
        """Derive a topology from a node tree.

        Parameters
        ----------
        nodes : list of BvhNode
            Nodes in an order where every parent precedes its children —
            the depth-first order of :attr:`Bvh.nodes`, for instance.

        Returns
        -------
        FkTopology
            ``joint_idx`` counts up in node order, so ``euler_orders``
            matches :attr:`Bvh.euler_orders` element for element.

        Notes
        -----
        Parents are resolved by object identity, never by name: node names
        need not be unique (the parser generates end-site display names
        from the parent's name, so two end sites under one joint collide),
        and a name-keyed lookup would silently attach a limb to the wrong
        parent.
        """
        num_nodes = len(nodes)
        node_position = {id(node): i for i, node in enumerate(nodes)}
        offsets = np.empty((num_nodes, 3), dtype=np.float64)
        parent_idx = np.empty(num_nodes, dtype=np.intp)
        joint_idx = np.empty(num_nodes, dtype=np.intp)
        euler_orders: list[str] = []

        joint_counter = 0
        for i, node in enumerate(nodes):
            offsets[i] = node.offset
            if node.parent is None:
                parent_idx[i] = -1
            else:
                try:
                    parent_idx[i] = node_position[id(node.parent)]
                except KeyError:
                    raise ValueError(
                        f"Node {node.name!r} (index {i}) has a parent that is "
                        f"not in the node list.") from None

            if node.is_end_site():
                joint_idx[i] = -1
            else:
                joint_idx[i] = joint_counter
                euler_orders.append(''.join(node.rot_channels))  # type: ignore[attr-defined]
                joint_counter += 1

        return cls(offsets, parent_idx, joint_idx, euler_orders)


def _validate_fk_topology(
    offsets: npt.NDArray[np.float64],
    parent_idx: npt.NDArray[np.intp],
    joint_idx: npt.NDArray[np.intp],
    euler_orders: list[str],
) -> None:
    """Check every invariant the FK loop relies on. See :class:`FkTopology`."""
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(
            f"offsets must have shape (N, 3), got {offsets.shape}")
    num_nodes = offsets.shape[0]
    if num_nodes == 0:
        raise ValueError("FkTopology must describe at least one node")
    if parent_idx.shape != (num_nodes,):
        raise ValueError(
            f"parent_idx must have shape ({num_nodes},) to match offsets, "
            f"got {parent_idx.shape}")
    if joint_idx.shape != (num_nodes,):
        raise ValueError(
            f"joint_idx must have shape ({num_nodes},) to match offsets, "
            f"got {joint_idx.shape}")

    # Parents precede children. This also forces parent_idx[0] < 0, so the
    # root is necessarily node 0 and the "exactly one -1" check below pins
    # it there without a separate assertion.
    forward_refs = np.flatnonzero(parent_idx >= np.arange(num_nodes))
    if forward_refs.size:
        i = int(forward_refs[0])
        raise ValueError(
            f"parent_idx[{i}] = {int(parent_idx[i])} does not precede its "
            f"child at index {i}: FkTopology requires parents to come first "
            f"in node order, since forward kinematics fills each node from "
            f"its parent's already-computed frame.")
    if np.any(parent_idx < -1):
        bad = int(np.flatnonzero(parent_idx < -1)[0])
        raise ValueError(
            f"parent_idx[{bad}] = {int(parent_idx[bad])}; the only legal "
            f"negative value is -1 (the root).")
    root_count = int(np.count_nonzero(parent_idx == -1))
    if root_count != 1:
        raise ValueError(
            f"parent_idx must contain exactly one -1 (the root), got "
            f"{root_count}.")

    # The root's rotation is read as local_rotmats[:, joint_idx[0]]. A -1
    # there is a legal negative index, so it would silently apply the last
    # joint's rotation to the entire skeleton instead of raising.
    if joint_idx[0] < 0:
        raise ValueError(
            "joint_idx[0] = -1 marks the root as an end site, but the root "
            "carries the skeleton's base rotation. End sites are leaves.")

    # End sites accumulate no rotation, so acc_rotmats is never written for
    # them; a child of one would read an uninitialized array.
    has_parent = parent_idx >= 0
    parents_of_children = parent_idx[has_parent]
    end_site_parents = np.flatnonzero(joint_idx[parents_of_children] < 0)
    if end_site_parents.size:
        child = int(np.flatnonzero(has_parent)[end_site_parents[0]])
        raise ValueError(
            f"Node {child} is parented to node {int(parent_idx[child])}, "
            f"which joint_idx marks as an end site. End sites are leaves: "
            f"they carry no rotation for a child to inherit.")

    if np.any(joint_idx < -1):
        bad = int(np.flatnonzero(joint_idx < -1)[0])
        raise ValueError(
            f"joint_idx[{bad}] = {int(joint_idx[bad])}; the only legal "
            f"negative value is -1 (an end site).")
    joint_columns = np.sort(joint_idx[joint_idx >= 0])
    num_joints = joint_columns.size
    if not np.array_equal(joint_columns, np.arange(num_joints)):
        raise ValueError(
            f"The non-negative joint_idx values must be exactly 0..J-1, each "
            f"once (they index joint_angles axis 1); got {num_joints} joint "
            f"nodes whose sorted indices are {joint_columns.tolist()}.")
    if len(euler_orders) != num_joints:
        raise ValueError(
            f"euler_orders has {len(euler_orders)} entries but joint_idx "
            f"marks {num_joints} joints.")
    for j, order in enumerate(euler_orders):
        if sorted(order) != ['X', 'Y', 'Z']:
            raise ValueError(
                f"euler_orders[{j}] = {order!r} is not a permutation of "
                f"'XYZ'.")


def frames_to_node_positions(
    skeleton: Union[Bvh, list[BvhNode], FkTopology],
    root_pos: npt.ArrayLike | None = None,
    joint_angles: npt.ArrayLike | None = None,
    centered: str = "world",
    up: str | None = None,
) -> npt.NDArray[np.float64]:
    """
    Return spatial coordinates of all nodes for one or multiple frames.

    Parameters
    ----------
    skeleton : Bvh or list of BvhNode or FkTopology
        The skeleton to pose. Pass an :class:`FkTopology` to run forward
        kinematics from arrays alone, with no node objects — see
        :attr:`Bvh.fk_topology` for producing one.

        Renamed from ``nodes_container`` in 0.8.2, when the array form was
        added and the old name stopped being true.
    root_pos : ndarray, shape (F, 3) or (3,), optional
        Root position per frame.  If None, extracted from *skeleton*
        (which must then be a Bvh object).
    joint_angles : ndarray, shape (F, J, 3) or (J, 3), optional
        Euler angles **in radians** per joint per frame (pybvh's internal
        convention; matches :attr:`Bvh.joint_angles`). If None, extracted
        from *skeleton*. A non-Euler stream converts first — see
        :mod:`pybvh.rotations`.
    centered : str
        ``"world"``  – root at its actual position.
        ``"skeleton"`` – root at origin in every frame.
        ``"first"`` – ground-plane centering: the first frame's root
        position is subtracted in the two non-``up`` axes only, so the
        motion starts above the origin at its original height.
    up : str or None, optional
        Signed world-up axis string (e.g. ``'+y'``) — only read by
        ``centered="first"``, to decide which coordinate is left
        untouched. Defaults to the skeleton's own ``world_up`` when
        *skeleton* is a :class:`~pybvh.Bvh`. There is no default for the
        other two input forms: a node list and an ``FkTopology`` carry no
        gravity direction, and guessing one silently mis-centers every
        skeleton that does not happen to match the guess.

    Returns
    -------
    ndarray, shape (F, N, 3) or (N, 3)
        Spatial coordinates for all nodes (including end sites).
        Returns 2-D ``(N, 3)`` when a single frame is provided,
        3-D ``(F, N, 3)`` otherwise.

    Raises
    ------
    ValueError
        If *centered* is not one of the three modes; if *root_pos* /
        *joint_angles* are omitted for a skeleton that carries no motion;
        if they disagree on frame count or on the joint count the skeleton
        declares; or if ``centered="first"`` is requested without an
        ``up`` axis and the skeleton cannot supply one.
    """
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
        raise ValueError(f"centered argument must be one of {accepted_centered}.")

    # Resolve every accepted input form to the array topology the FK loop
    # actually reads. `source_bvh` is the only thing that can supply motion
    # or a gravity axis when the caller does not.
    topology, source_bvh = _resolve_topology(skeleton)

    # ---- obtain root_pos / joint_angles ----
    single_frame = False

    if root_pos is None or joint_angles is None:
        if source_bvh is None:
            raise ValueError(
                "root_pos and joint_angles must be provided when skeleton is "
                "not a Bvh object.")
        root_pos = source_bvh.root_pos
        joint_angles = source_bvh.joint_angles

    root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
    joint_angles_arr: npt.NDArray[np.float64] = np.asarray(joint_angles, dtype=np.float64)

    if root_pos_arr.ndim == 1:
        root_pos_arr = root_pos_arr.reshape(1, 3)
        single_frame = True
    if joint_angles_arr.ndim == 2:
        joint_angles_arr = joint_angles_arr.reshape(1, *joint_angles_arr.shape)

    # -- From here, root_pos_arr is (F, 3) and joint_angles_arr is (F, J, 3) --

    if root_pos_arr.shape[0] != joint_angles_arr.shape[0]:
        raise ValueError(
            f"root_pos and joint_angles disagree on frame count: "
            f"root_pos has {root_pos_arr.shape[0]} frames, joint_angles "
            f"has {joint_angles_arr.shape[0]}.")
    num_joints = len(topology.euler_orders)
    if joint_angles_arr.shape[1] != num_joints:
        raise ValueError(
            f"joint_angles has {joint_angles_arr.shape[1]} joints on axis 1, "
            f"but the skeleton declares {num_joints}.")

    if centered == "first":
        if up is None:
            if source_bvh is None:
                raise ValueError(
                    "centered='first' needs an up axis to know which "
                    "coordinate to leave untouched, and a skeleton passed as "
                    "a node list or an FkTopology carries no gravity "
                    "direction. Pass `up=` explicitly (e.g. up='+z'), or "
                    "pass the Bvh itself so its `world_up` is used.")
            up = source_bvh.world_up
        up = _validate_axis_string(up)

    positions = _run_forward_kinematics(
        topology, root_pos_arr, joint_angles_arr,
        skel_centered=(centered == "skeleton"))

    # "first" centering: subtract the first frame's root position in the
    # two non-up axes only — the up coordinate stays in world units.
    if centered == "first" and positions.shape[0] > 0:
        assert up is not None  # mypy: set above whenever centered == "first"
        positions -= _ground_plane_offset(root_pos_arr[0], up)

    # Return (N, 3) for single frame, (F, N, 3) for multiple
    if single_frame:
        return positions[0]
    return positions


def _run_forward_kinematics(
    topology: FkTopology,
    root_pos: npt.NDArray[np.float64],
    joint_angles: npt.NDArray[np.float64],
    skel_centered: bool,
) -> npt.NDArray[np.float64]:
    """Vectorized FK over all frames, from arrays only.

    Reads nothing but *topology* — no node objects reach this loop, which
    is what lets :class:`FkTopology` be an entry point rather than an
    internal detail.
    """
    offsets = topology.offsets
    parent_idx = topology.parent_idx
    joint_idx = topology.joint_idx

    num_frames = root_pos.shape[0]
    num_nodes = offsets.shape[0]

    # All local joint rotations in one per-joint-vectorized call: (F, J, 3, 3)
    local_rotmats = euler_to_rotmat(joint_angles, topology.euler_orders)

    # positions: (F, N, 3) - spatial coordinates per node per frame
    # acc_rotmats: (F, N, 3, 3) - accumulated rotation matrices per node per frame
    positions = np.empty((num_frames, num_nodes, 3), dtype=np.float64)
    acc_rotmats = np.empty((num_frames, num_nodes, 3, 3), dtype=np.float64)

    # Node order guarantees a parent's row is written before its children
    # read it (validated by FkTopology).
    for i in range(num_nodes):
        p_idx = parent_idx[i]
        j_idx = joint_idx[i]

        if p_idx == -1:
            # Root node
            positions[:, i, :] = 0.0
            acc_rotmats[:, i] = local_rotmats[:, j_idx]
        elif j_idx == -1:
            # End site: no own rotation
            offset = offsets[i]  # (3,)
            # parent_rot @ offset + parent_pos for all frames
            positions[:, i] = np.einsum('fij,j->fi', acc_rotmats[:, p_idx], offset) + positions[:, p_idx]
        else:
            # Joint node
            offset = offsets[i]  # (3,)
            positions[:, i] = np.einsum('fij,j->fi', acc_rotmats[:, p_idx], offset) + positions[:, p_idx]
            # Accumulate rotation: parent_rot @ this_node_rot
            acc_rotmats[:, i] = acc_rotmats[:, p_idx] @ local_rotmats[:, j_idx]

    # Add root position if not skeleton-centered
    if not skel_centered:
        positions += root_pos[:, np.newaxis, :]  # (F,1,3) broadcasts over (F,N,3)

    return positions


def _ground_plane_offset(
    root_position: npt.NDArray[np.float64],
    up: str,
) -> npt.NDArray[np.float64]:
    """A ``(3,)`` copy of *root_position* with its ``up`` component zeroed.

    Subtracting this vector centers positions on the origin in the ground
    plane while leaving the height above the ground untouched — the
    ``centered="first"`` convention shared by
    :func:`frames_to_node_positions` and :meth:`Bvh.node_positions`.
    """
    up_idx = {'x': 0, 'y': 1, 'z': 2}[up[-1].lower()]
    offset = np.array(root_position, dtype=np.float64)
    offset[up_idx] = 0.0
    return offset


def _resolve_topology(
    skeleton: Union[Bvh, list[BvhNode], FkTopology],
) -> tuple[FkTopology, Bvh | None]:
    """Resolve any accepted skeleton form into an :class:`FkTopology`.

    The single point where the three input forms converge, so the FK loop
    and everything downstream of it see exactly one representation.

    Returns
    -------
    topology : FkTopology
    source_bvh : Bvh or None
        The originating Bvh when there was one — the only form that can
        also supply motion data and a ``world_up``.
    """
    from .bvh import Bvh  # lazy: bvh.py imports this module at top level
    if isinstance(skeleton, FkTopology):
        return skeleton, None
    if isinstance(skeleton, Bvh):
        return FkTopology.from_nodes(skeleton.nodes), skeleton
    if isinstance(skeleton, list):
        if not all(isinstance(n, BvhNode) for n in skeleton):
            raise ValueError('The list must contain BvhNode objects.')
        return FkTopology.from_nodes(skeleton), None
    raise ValueError(
        'skeleton must be a Bvh object, a list of BvhNode objects, or an '
        'FkTopology.')
