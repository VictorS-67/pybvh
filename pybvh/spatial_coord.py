import numpy as np
from .tools import get_premult_mat_rot

def frames_to_spatial_coord(nodes_container, frames=[], centered="world"):
    """
    Return a 2d np array of the spatial coordinates of all the joints.
    Input :
    - bvh_object : a Bvh object or a list of Bvh nodes 
                    (as obtained from bvh_object.nodes)
    - frames : 2d np array containing the euler angles of the joints rotation 
                (as obtained from taking multiple line of a bvh_object.frames)
    - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton" , the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0] in ALL frames).
                If "first", the first frame root position is considered to be [0, 0, 0]. From there,
                the skeleton moves in the space normally.
                If "world", the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root saved in the bvh object in all frames).
    """
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
        raise ValueError(f"centered argument must be one of {accepted_centered}.")
    
    if len(frames) == 0:
        try :
            frames = nodes_container.frames
        except:
            raise ValueError("The argument frames cannot be empty if nodes_container is not a Bvh object")

    #the only case not covered by frame_to_spatial_coord is when centered="first"
    if centered == "first":
        # in this case, when calling frame_to_spatial_coord
        # we want to obtain the root with its real coordinates
        # then we substract the first frame root coordinates to the skeleton joints coordinates 
        tiled_first_root_pos = np.tile(frames[0][:3], int(frames.shape[1]/3))
        def arg_exchange_frame_to_spatial_coord(new_frames):
            return frame_to_spatial_coord(nodes_container, new_frames, centered="world")
        return np.apply_along_axis(arg_exchange_frame_to_spatial_coord, 1, frames) - tiled_first_root_pos
    
    else:
        def arg_exchange_frame_to_spatial_coord(new_frames):
            return frame_to_spatial_coord(nodes_container, new_frames, centered=centered)
        return np.apply_along_axis(arg_exchange_frame_to_spatial_coord, 1, frames)
     

def frame_to_spatial_coord(nodes_container, frame, centered="world"):
    """
    Return a 1d np array of the spatial coordinates of all the joints.
    Input :
    - nodes_container : a Bvh object or a list of Bvh nodes 
                    (as obtained from bvh_object.nodes)
    - frame : 1d np array containing the euler angles of the joints rotation 
                (as obtained from taking a line of a bvh_object.frames)
                or
                an int that indicates the index of the frame in the bvh_object.frames 2d np array.
                //!\\This is only possible if nodes_container is a bvh object ! 
    - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton", the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0]) .
                If "first", we return the same result as "skeleton".
                This option is here for easy compatibility with the function frames_to_spatial_coord().
                If "world", the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root).
    """
    try:
            # first we test if it is a bvh object
            nodes = nodes_container.nodes
            root = nodes_container.root
            root.rot_channels
            if isinstance(frame, int):
                # in case frame is a int object,
                # and we have a bvh object as nodes_container,
                # there is no problem
                frame = nodes_container.frames[frame]
    except:
        try:
            # if it was not a bvh object, we try to see
            # if it is a list of nodes
            nodes = nodes_container
            root = nodes_container[0]
            root.pos_channels
            if isinstance(frame, int):
                # in case frame is a int object,
                # and we do not a bvh object as nodes_container,
                # we raise a value error problem
                raise ValueError("The frame cannot be an int if the node container is not a bvh object. \
                                    Either change the node_container argument to a bvh object, \
                                    or change the frame argument to a 1d np array contaning the euler angles for the joints rotations.")
        except:
            raise ValueError('The argument nodes_container needs to be either a Bvh object, or a list of Bvh nodes (as obtained from bvh_object.nodes for ex)')
                
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
            raise ValueError(f"centered argument must be one of {accepted_centered}.")
    
    nodes_transfo = {}
    node2frameidx = {}
    # skip the root spatial position
    i=1 
    #but save them in case we needed later if centered="world"
    root_pos = frame[[0,1,2]]
    for node in nodes:
        # for every node, except end sites,
        # create a dict with their position in the frames
        # and the accumulated (meaning : with all the previous ones already multiplied to it)
        # rotation matrix at this node
        nodes_transfo[node.name] = {'spatial_coor' : None,
                                    'acc_rot_mat' : None
                                    }
        if 'End Site' in node.name:
            continue

        node2frameidx[node.name] = [3*i, 1+3*i, 2+3*i]
        i+=1

    #  frame_angles : a 1D list/np.array of all the euler angles for one frame
    # we need to transform the angle into radian
    frame_angles = np.radians(frame)

    frame_spatial = _get_local_pos_rec(root, nodes_transfo, node2frameidx, frame_angles)

    # finally, we add the root position to all the joints coordinates
    #  if centered == "world"
    if centered == "world":
        for i in range(3):
            frame_spatial[i::3] = frame_spatial[i::3] + root_pos[i]

    # if needed for debugging purpose, we can also return nodes_transfo,
    # which contain the info written during the recursivity process
    # return frame_spatial, nodes_transfo
    return frame_spatial

def _get_local_pos_rec(node, nodes_transfo:dict, node2frameidx:dict, frame_angles, isroot=True) -> np.ndarray:
    """
    Recurring function to calculate the local position of a node
    Input are:
    - node : a bvh node object (contains offset and channel of the node)
    - nodes_transfo : a nested dictionnary containing 
                        the spatial coordinates of the node,
                        as well as the previously calculated rot matrices
                        (and already multiplied together). eg :
                        {
                        'Hips': {
                                'spatial_coor': [x_coord, y_coord, z_coord],
                                'acc_rot_mat': np.array([[...]])
                                },
                        'Spine': ...
                        }
                        for the first call to the function, the value of 
                        'spatial_coor' and 'acc_rot_mat' can be None
    - node2frameidx : a dictionnary containing the name of the nodes and their 
                        euler angles column indices in the frames matrix. eg
                        {
                        'Hips': [3, 4, 5],
                        'Spine': [6, 7, 8],
                        ...
                        }
    - frame_angles : a line of bvh.frames np array, containing the angles info of one bvh frame

    To calculate the spatial coordinates, the formula is
    R[root] @ ... @ R[direct parent of the node] @ node_offset + parent_spatial_coordinates
    this multiplication of rotation matrices is stored in nodes_transfo[node.parent.name],
    to avoid having to redo the same matrix multiplication all the time
    """

    if "End Site" in node.name:
        # we are at a terminal point of the recursive function
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']
        nodes_transfo[node.name]['spatial_coor'] = coord
        #coord is a 3-elt np array
        return coord
    
    elif isroot:
        #if we are dealing with the root, it's the first call of the function
        #we consider a local coordinate for now, so spatial coord are at the origin
        coord = np.array([0.0, 0.0, 0.0])
        nodes_transfo[node.name]['spatial_coor'] = coord
        node_angles = frame_angles[node2frameidx[node.name]]
        nodes_transfo[node.name]['acc_rot_mat'] = get_premult_mat_rot(node_angles, node.rot_channels)

        result = coord
        #start of the recursivity
        for child_node in node.children:
            result = np.append(result, _get_local_pos_rec(child_node, nodes_transfo, node2frameidx, frame_angles, isroot=False))
        return result
    
    else:
        # the node is not the root so has one parent,
        # and has at least one child on which we need to call the function
        parent_info = nodes_transfo[node.parent.name]
        coord = parent_info['acc_rot_mat'] @ node.offset + parent_info['spatial_coor']

        #update the dictionary containing the info
        nodes_transfo[node.name]['spatial_coor'] = coord
        node_angles = frame_angles[node2frameidx[node.name]]
        nodes_transfo[node.name]['acc_rot_mat'] = parent_info['acc_rot_mat'] @ get_premult_mat_rot(node_angles, node.rot_channels)
        
        result = coord
        # print(node.name, node_angles, node.rot_channels)
        # raise ValueError
        for child_node in node.children:
            result = np.append(result, _get_local_pos_rec(child_node, nodes_transfo, node2frameidx, frame_angles, isroot=False))
        return result


