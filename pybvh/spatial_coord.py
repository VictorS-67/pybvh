import numpy as np
from .tools import get_premult_mat_rot


def frames_to_spatial_coord(nodes_container, frames=None, centered="world"):
    """
    Return spatial coordinates of all joints for one or multiple frames.
    
    Input:
    - nodes_container : a Bvh object or a list of Bvh nodes 
                        (as obtained from bvh_object.nodes)
    - frames : Can be:
               - None or empty: use frames from the Bvh object (nodes_container must be Bvh)
               - 1D np array: single frame of euler angles
               - 2D np array: multiple frames of euler angles
               - int: frame index (only if nodes_container is a Bvh object)
    - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton", the root coordinates are [0, 0, 0] in ALL frames.
                If "first", the first frame root position is [0, 0, 0], then moves normally.
                If "world", the root coordinates are the actual saved coordinates.
    
    Returns:
    - 1D np array if single frame input, 2D np array if multiple frames
    """
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
        raise ValueError(f"centered argument must be one of {accepted_centered}.")

    # Resolve nodes_container to a list of nodes
    nodes, nodes_from_bvh = _nodes_container_to_nodes_list(nodes_container)
    
    # Handle frames input - normalize to 2D array
    single_frame = False
    
    if frames is None or (isinstance(frames, (list, np.ndarray)) and len(frames) == 0):
        # No frames provided - get from Bvh object
        if not nodes_from_bvh:
            raise ValueError("The argument frames cannot be empty if nodes_container is not a Bvh object")
        frames = nodes_container.frames
    elif isinstance(frames, int):
        # Frame index provided
        if not nodes_from_bvh:
            raise ValueError("frames cannot be an int if nodes_container is not a Bvh object.")
        frames = nodes_container.frames[frames:frames+1]  # Keep as 2D with 1 row
        single_frame = True
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 1:
            frames = frames.reshape(1, -1)  # Convert 1D to 2D with 1 row
            single_frame = True
        # else: already 2D, keep as-is

    # -- From here, frames is always a 2D np array --
    
    num_frames = frames.shape[0]
    num_nodes = len(nodes)
    root = nodes[0]
    skel_centered = (centered == "skeleton")
    
    # Pre-allocate output array
    output = np.empty((num_frames, num_nodes * 3))
    
    # Build helper dicts ONCE (reused across all frames)
    node2frameidx = {}
    node2outidx = {}
    
    i = 1  # frame column index (skip root position columns)
    out_idx = 0
    for node in nodes:
        node2outidx[node.name] = out_idx
        out_idx += 3
        if not node.is_end_site():
            node2frameidx[node.name] = [3*i, 1+3*i, 2+3*i]
            i += 1

    # Convert ALL angles to radians at once (vectorized)
    all_frame_angles = np.radians(frames)
    
    # Extract all root positions at once
    root_pos_all = frames[:, :3]

    # Create nodes_transfo dict ONCE (values get overwritten each frame, no reset needed)
    nodes_transfo = {node.name: {'spatial_coor': None, 'acc_rot_mat': None} 
                     for node in nodes}

    # Process each frame
    for frame_idx in range(num_frames):
        frame_angles = all_frame_angles[frame_idx]
        frame_output = output[frame_idx]  # View into output row (writes in-place)
        
        _fill_spatial_coords_rec(root, frame_output, nodes_transfo, 
                                  node2frameidx, node2outidx, frame_angles)
        
        # Add root position if not skeleton-centered
        if not skel_centered:
            frame_output += np.tile(root_pos_all[frame_idx], num_nodes)

    # Handle "first" centering mode - subtract first frame's root position
    if centered == "first":
        first_root_pos = output[0, :3]
        output -= np.tile(first_root_pos, num_nodes)

    # Return 1D for single frame, 2D for multiple frames
    if single_frame:
        return output[0]
    return output


def frame_to_spatial_coord(nodes_container, frame, skel_centered=False, nodes_from_bvh=None):
    """
    Return a 1D np array of spatial coordinates for a single frame.
    
    This is a backwards-compatible wrapper around frames_to_spatial_coord.
    
    Input:
    - nodes_container : a Bvh object or a list of Bvh nodes
    - frame : 1D np array of euler angles, or an int frame index (if nodes_container is Bvh)
    - skel_centered : boolean. If True, root at [0,0,0]. If False, root at actual position.
    - nodes_from_bvh : deprecated, kept for compatibility (ignored)
    
    Returns:
    - 1D np array of spatial coordinates
    """
    centered = "skeleton" if skel_centered else "world"
    return frames_to_spatial_coord(nodes_container, frames=frame, centered=centered)


def _fill_spatial_coords_rec(node, output, nodes_transfo, node2frameidx, node2outidx, frame_angles, isroot=True):
    """
    Recursively fill the pre-allocated output array with spatial coordinates.
    Writes directly to output[node2outidx[node.name]:node2outidx[node.name]+3].
    
    Input:
    - node : a bvh node object (contains offset and channel of the node)
    - output : pre-allocated numpy array to write coordinates into
    - nodes_transfo : a nested dict containing the spatial coordinates and 
                      accumulated rotation matrices for each node
    - node2frameidx : dict mapping node name to frame angle indices
    - node2outidx : dict mapping node name to output array start index
    - frame_angles : 1D array of euler angles (in radians) for one frame
    - isroot : boolean indicating if this is the root node

    The spatial coordinate formula is:
    R[root] @ ... @ R[direct parent] @ node_offset + parent_spatial_coordinates
    """
    out_idx = node2outidx[node.name]

    if node.is_end_site():
        # Terminal node - compute coord from parent's transform
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']
        nodes_transfo[node.name]['spatial_coor'] = coord
        output[out_idx:out_idx+3] = coord
        return
    
    elif isroot:
        # Root node - spatial coord at origin, compute rotation matrix
        coord = np.array([0.0, 0.0, 0.0])
        nodes_transfo[node.name]['spatial_coor'] = coord
        node_angles = frame_angles[node2frameidx[node.name]]
        nodes_transfo[node.name]['acc_rot_mat'] = get_premult_mat_rot(node_angles, node.rot_channels)
        output[out_idx:out_idx+3] = coord
        
        for child_node in node.children:
            _fill_spatial_coords_rec(child_node, output, nodes_transfo, node2frameidx, node2outidx, frame_angles, isroot=False)
        return
    
    else:
        # Interior node - compute from parent's accumulated transform
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']
        nodes_transfo[node.name]['spatial_coor'] = coord
        node_angles = frame_angles[node2frameidx[node.name]]
        nodes_transfo[node.name]['acc_rot_mat'] = parent_info['acc_rot_mat'] @ get_premult_mat_rot(node_angles, node.rot_channels)
        output[out_idx:out_idx+3] = coord
        
        for child_node in node.children:
            _fill_spatial_coords_rec(child_node, output, nodes_transfo, node2frameidx, node2outidx, frame_angles, isroot=False)
        return


def _nodes_container_to_nodes_list(nodes_container):
    """
    Input : a Bvh object or a list of Bvh nodes
    Output : 
     - a list of Bvh nodes
     - a bool that indicates if the nodes are from a Bvh object or not
    """
    from_bvh = True
    try:
        nodes = nodes_container.nodes
        root = nodes_container.root
        pos = root.pos_channels
    except:
        from_bvh = False
        try:
            nodes = nodes_container
            root = nodes_container[0]
            pos = root.pos_channels
        except:
            raise ValueError('The nodes container needs to be either a Bvh object, or a list of Bvh nodes (as obtained from bvh_object.nodes for ex)')
    
    return nodes, from_bvh