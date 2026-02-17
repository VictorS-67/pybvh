import numpy as np
from .tools import get_premult_mat_rot
from .bvhnode import BvhNode


def frames_to_spatial_coord(nodes_container, root_pos=None, joint_angles=None, centered="world"):
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
        Euler angles in degrees per joint per frame.  If None, extracted
        from *nodes_container*.
    centered : str
        ``"world"``  – root at its actual position.
        ``"skeleton"`` – root at origin in every frame.
        ``"first"`` – root at origin in the first frame, then moves normally.

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

    # Resolve nodes_container to a list of nodes
    nodes, is_bvh = _nodes_container_to_nodes_list(nodes_container)

    # ---- obtain root_pos / joint_angles ----
    single_frame = False

    if root_pos is None or joint_angles is None:
        if not is_bvh:
            raise ValueError(
                "root_pos and joint_angles must be provided when "
                "nodes_container is not a Bvh object.")
        root_pos = nodes_container.root_pos
        joint_angles = nodes_container.joint_angles

    root_pos = np.asarray(root_pos, dtype=np.float64)
    joint_angles = np.asarray(joint_angles, dtype=np.float64)

    if root_pos.ndim == 1:
        root_pos = root_pos.reshape(1, 3)
        single_frame = True
    if joint_angles.ndim == 2:
        joint_angles = joint_angles.reshape(1, *joint_angles.shape)

    # -- From here, root_pos is (F, 3) and joint_angles is (F, J, 3) --

    num_frames = root_pos.shape[0]
    num_nodes = len(nodes)
    skel_centered = (centered == "skeleton")

    # Pre-allocate output: (F, N, 3)
    output = np.empty((num_frames, num_nodes, 3))

    # Build helper dicts ONCE (reused across all frames)
    node2jointidx = {}
    node2nodeidx = {}
    j_idx = 0
    for n_idx, node in enumerate(nodes):
        node2nodeidx[node.name] = n_idx
        if not node.is_end_site():
            node2jointidx[node.name] = j_idx
            j_idx += 1

    # Convert ALL angles to radians at once
    all_angles_rad = np.radians(joint_angles)

    # Reusable dict for accumulated transforms
    nodes_transfo = {node.name: {'spatial_coor': None, 'acc_rot_mat': None}
                     for node in nodes}

    # Process each frame
    for frame_idx in range(num_frames):
        frame_angles = all_angles_rad[frame_idx]  # (J, 3)
        frame_output = output[frame_idx]           # (N, 3) view

        _fill_spatial_coords_rec(nodes[0], frame_output, nodes_transfo,
                                 node2jointidx, node2nodeidx, frame_angles)

        # Add root position if not skeleton-centered
        if not skel_centered:
            frame_output += root_pos[frame_idx]  # (3,) broadcasts over (N, 3)

    # Handle "first" centering mode – subtract first frame's root position
    if centered == "first":
        output -= output[0:1, 0:1, :]  # (1,1,3) broadcasts over (F,N,3)

    # Return (N, 3) for single frame, (F, N, 3) for multiple
    if single_frame:
        return output[0]
    return output


def _fill_spatial_coords_rec(node, output, nodes_transfo,
                              node2jointidx, node2nodeidx, frame_angles,
                              isroot=True):
    """
    Recursively compute spatial coordinates for one frame.

    Writes directly into *output[node_index]*.

    Parameters
    ----------
    node : BvhNode
        Current node being processed.
    output : ndarray, shape (N, 3)
        Pre-allocated output array for one frame.
    nodes_transfo : dict
        Accumulated rotation matrices and spatial coordinates per node.
    node2jointidx : dict
        Maps node name → joint index in *frame_angles*.
    node2nodeidx : dict
        Maps node name → row index in *output*.
    frame_angles : ndarray, shape (J, 3)
        Euler angles in radians for one frame.
    isroot : bool
        Whether this is the root node.
    """
    n_idx = node2nodeidx[node.name]

    if node.is_end_site():
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']
        nodes_transfo[node.name]['spatial_coor'] = coord
        output[n_idx] = coord
        return

    elif isroot:
        coord = np.array([0.0, 0.0, 0.0])
        nodes_transfo[node.name]['spatial_coor'] = coord
        j_idx = node2jointidx[node.name]
        node_angles = frame_angles[j_idx]
        nodes_transfo[node.name]['acc_rot_mat'] = get_premult_mat_rot(
            node_angles, node.rot_channels)
        output[n_idx] = coord

        for child_node in node.children:
            _fill_spatial_coords_rec(child_node, output, nodes_transfo,
                                      node2jointidx, node2nodeidx,
                                      frame_angles, isroot=False)
        return

    else:
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']
        nodes_transfo[node.name]['spatial_coor'] = coord
        j_idx = node2jointidx[node.name]
        node_angles = frame_angles[j_idx]
        nodes_transfo[node.name]['acc_rot_mat'] = (
            parent_info['acc_rot_mat'] @ get_premult_mat_rot(
                node_angles, node.rot_channels))
        output[n_idx] = coord

        for child_node in node.children:
            _fill_spatial_coords_rec(child_node, output, nodes_transfo,
                                      node2jointidx, node2nodeidx,
                                      frame_angles, isroot=False)
        return


def _nodes_container_to_nodes_list(nodes_container):
    """
    Resolve a Bvh object or list of nodes into a plain list of nodes.

    Returns
    -------
    nodes : list of BvhNode
        The list of nodes.
    is_bvh : bool
        True if *nodes_container* is a Bvh object (has root_pos / joint_angles).
    """
    if hasattr(nodes_container, 'nodes') and hasattr(nodes_container, 'root_pos'):
        return nodes_container.nodes, True
    elif isinstance(nodes_container, list):
        if not all(isinstance(n, BvhNode) for n in nodes_container):
            raise ValueError('The list must contain BvhNode objects.')
        return nodes_container, False
    else:
        raise ValueError(
            'nodes_container must be a Bvh object or a list of BvhNode objects.')