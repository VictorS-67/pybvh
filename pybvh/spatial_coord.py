from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

from .rotations import euler_to_rotmat
from .tools import _validate_axis_string
from .bvhnode import BvhNode

if TYPE_CHECKING:
    from .bvh import Bvh


def frames_to_node_positions(
    nodes_container: Union[Bvh, list[BvhNode]],
    root_pos: npt.ArrayLike | None = None,
    joint_angles: npt.ArrayLike | None = None,
    centered: str = "world",
    up: str = '+y',
) -> npt.NDArray[np.float64]:
    """
    Return spatial coordinates of all nodes for one or multiple frames.

    Parameters
    ----------
    nodes_container : Bvh or list of BvhNode
        A Bvh object or a list of Bvh nodes (as obtained from bvh_object.nodes).
    root_pos : ndarray, shape (F, 3) or (3,), optional
        Root position per frame.  If None, extracted from *nodes_container*
        (which must then be a Bvh object).
    joint_angles : ndarray, shape (F, J, 3) or (J, 3), optional
        Euler angles **in radians** per joint per frame (pybvh's internal
        convention; matches :attr:`Bvh.joint_angles`). If None, extracted
        from *nodes_container*.
    centered : str
        ``"world"``  – root at its actual position.
        ``"skeleton"`` – root at origin in every frame.
        ``"first"`` – ground-plane centering: the first frame's root
        position is subtracted in the two non-``up`` axes only, so the
        motion starts above the origin at its original height.
    up : str, optional
        Signed world-up axis string (``'+y'`` default) — only used by
        ``centered="first"`` to decide which coordinate is left untouched.
        :meth:`Bvh.node_positions` passes ``bvh.world_up`` automatically.

    Returns
    -------
    ndarray, shape (F, N, 3) or (N, 3)
        Spatial coordinates for all nodes (including end sites).
        Returns 2-D ``(N, 3)`` when a single frame is provided,
        3-D ``(F, N, 3)`` otherwise.
    """
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
        raise ValueError(f"centered argument must be one of {accepted_centered}.")
    if centered == "first":
        up = _validate_axis_string(up)

    # Resolve nodes_container to a list of nodes
    nodes, is_bvh = _nodes_container_to_nodes_list(nodes_container)

    # ---- obtain root_pos / joint_angles ----
    single_frame = False

    if root_pos is None or joint_angles is None:
        if not is_bvh:
            raise ValueError(
                "root_pos and joint_angles must be provided when "
                "nodes_container is not a Bvh object.")
        root_pos = nodes_container.root_pos  # type: ignore[union-attr, has-type]
        joint_angles = nodes_container.joint_angles  # type: ignore[union-attr, has-type]

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

    num_frames = root_pos_arr.shape[0]
    num_nodes = len(nodes)
    skel_centered = (centered == "skeleton")

    # `joint_angles` is already in radians (pybvh's internal convention).
    all_angles_rad = joint_angles_arr

    # ---- Build topology arrays ----
    # parent_idx[i] = index of parent node (-1 for root)
    # joint_idx[i]  = index into joint_angles for node i (-1 for end sites)
    # offsets[i]    = (3,) offset vector
    # rot_orders[j] = rotation order for joint j

    # Keyed by object identity, not name — duplicate joint names would
    # silently corrupt the parent lookup otherwise.
    node2idx = {}
    parent_idx = np.empty(num_nodes, dtype=np.intp)
    joint_idx = np.empty(num_nodes, dtype=np.intp)
    offsets = np.empty((num_nodes, 3), dtype=np.float64)
    rot_orders = []

    j_counter = 0
    for i, node in enumerate(nodes):
        node2idx[id(node)] = i
        offsets[i] = node.offset
        if node.parent is not None:
            parent_idx[i] = node2idx[id(node.parent)]
        else:
            parent_idx[i] = -1

        if not node.is_end_site():
            joint_idx[i] = j_counter
            rot_orders.append(node.rot_channels)  # type: ignore[attr-defined]
            j_counter += 1
        else:
            joint_idx[i] = -1

    # ---- Vectorized forward kinematics over all frames ----
    # All local joint rotations in one per-joint-vectorized call: (F, J, 3, 3)
    per_joint_orders = [''.join(order) for order in rot_orders]
    local_rotmats = euler_to_rotmat(all_angles_rad, per_joint_orders)

    # positions: (F, N, 3) - spatial coordinates per node per frame
    # acc_rotmats: (F, N, 3, 3) - accumulated rotation matrices per node per frame
    positions = np.empty((num_frames, num_nodes, 3), dtype=np.float64)
    acc_rotmats = np.empty((num_frames, num_nodes, 3, 3), dtype=np.float64)

    for i, node in enumerate(nodes):
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
        positions += root_pos_arr[:, np.newaxis, :]  # (F,1,3) broadcasts over (F,N,3)

    # "first" centering: subtract the first frame's root position in the
    # two non-up axes only — the up coordinate stays in world units.
    if centered == "first" and num_frames > 0:
        positions -= _ground_plane_offset(root_pos_arr[0], up)

    # Return (N, 3) for single frame, (F, N, 3) for multiple
    if single_frame:
        return positions[0]
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


def _nodes_container_to_nodes_list(
    nodes_container: Union[Bvh, list[BvhNode]],
) -> tuple[list[BvhNode], bool]:
    """
    Resolve a Bvh object or list of nodes into a plain list of nodes.

    Returns
    -------
    nodes : list of BvhNode
        The list of nodes.
    is_bvh : bool
        True if *nodes_container* is a Bvh object (has root_pos / joint_angles).
    """
    from .bvh import Bvh  # lazy: bvh.py imports this module at top level
    if isinstance(nodes_container, Bvh):
        return nodes_container.nodes, True
    elif isinstance(nodes_container, list):
        if not all(isinstance(n, BvhNode) for n in nodes_container):
            raise ValueError('The list must contain BvhNode objects.')
        return nodes_container, False
    else:
        raise ValueError(
            'nodes_container must be a Bvh object or a list of BvhNode objects.')
