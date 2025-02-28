import numpy as np
import copy
from .tools import get_premult_mat_rot

def frames_to_spatial_coord(nodes_container, frames=[], centered="world", change_skeleton=None):
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
    - change_skeleton : a bvh object.
                If provided, the function will use the bones offset of the skeleton
                in this bvh object to calculate the spatial coordinates of the joints,
                instead of using the original offset of the nodes in the nodes_container
    """
    accepted_centered = ["skeleton", "first", "world"]
    if centered not in accepted_centered:
        raise ValueError(f"centered argument must be one of {accepted_centered}.")
    
    centered2skelCentered = {'world': False, 'first': False, 'skeleton': True}

    # nodes_container is either a bvh object or a list of nodes, get it as a list of nodes
    nodes, nodes_from_bvh = _nodes_container_to_nodes_list(nodes_container)
    
    if len(frames) == 0:
        #if frames is empty, we need to get the frames from the bvh object
        if not nodes_from_bvh:
            raise ValueError("The argument frames cannot be empty if nodes_container is not a Bvh object")
        else:
            frames = nodes_container.frames

    # -- from here, we know that frames is a 2d np array
    # -- and nodes is a list of BvhNode, not a Bvh object anymore
    
    #check if we need to change the offset of the nodes
    if change_skeleton != None:
        nodes = _change_offset(nodes, change_skeleton)

    #the only case not covered by frame_to_spatial_coord is when centered="first"
    if centered == "first":
        # in this case, when calling frame_to_spatial_coord
        # we want to obtain the root with its real coordinates
        # then we substract the first frame root coordinates to the skeleton joints coordinates 
        first_spatial = frame_to_spatial_coord(nodes, frames[0], skel_centered=False, nodes_from_bvh=False)
        tiled_first_root_pos = np.tile(first_spatial[:3], int(first_spatial.shape[0]/3))
        def arg_exchange_frame_to_spatial_coord(new_frames):
            return frame_to_spatial_coord(nodes, new_frames, skel_centered=centered2skelCentered[centered], nodes_from_bvh=False)
        return np.apply_along_axis(arg_exchange_frame_to_spatial_coord, 1, frames) - tiled_first_root_pos
    
    else:
        def arg_exchange_frame_to_spatial_coord(new_frames):
            return frame_to_spatial_coord(nodes, new_frames, skel_centered=centered2skelCentered[centered], nodes_from_bvh=False)
        return np.apply_along_axis(arg_exchange_frame_to_spatial_coord, 1, frames)
     

def frame_to_spatial_coord(nodes_container, frame, skel_centered=False, change_skeleton=None, nodes_from_bvh=None):
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
    - skel_centered : boolean.
                If True, the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0]) .
                If False, the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root).
    - change_skeleton : a bvh object.
                If provided, the function will use the bones offset of the skeleton
                in this bvh object to calculate the spatial coordinates of the joints,
                instead of using the original offset of the nodes in the nodes_container argument.
    - nodes_from_bvh : a boolean. 
                Internal use, to allow skipping the check if the nodes_container is a bvh object
                when calling the function from frames_to_spatial_coord.

    """
    if nodes_from_bvh == None:
        # if from_bvh is None, it means the function is called from outside
        # so we need to check if the nodes_container is a bvh object or not
        nodes, nodes_from_bvh = _nodes_container_to_nodes_list(nodes_container)
    elif nodes_from_bvh == True:
        # if from_bvh is True, it means the nodes_container is a bvh object
        nodes = nodes_container.nodes
    else:
        # if from_bvh is False, it means the nodes_container is a list of nodes
        nodes = nodes_container

    if isinstance(frame, int):
        if nodes_from_bvh:
            # in case frame is a int object,
            # and we have a bvh object as nodes_container,
            # there is no problem
            frame = nodes_container.frames[frame]
        else:
            # in case frame is a int object,
            # and we do not have a bvh object as nodes_container,
            # we raise a value error problem
            raise ValueError("The frame cannot be an int if the node container is not a bvh object. \
                                Either change the nodes_container argument to a bvh object, \
                                or change the frame argument to a 1d np array contaning the euler angles for the joints rotations.")
        

    #check if we need to change the offset of the nodes
    if change_skeleton != None:
        nodes = _change_offset(nodes, change_skeleton)
    

    root = nodes[0]

    nodes_transfo = {}
    node2frameidx = {}
    # skip the root spatial position
    i=1 
    #but save them in case we needed later if skel_centered=False
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
    # if skel_centered==False
    if skel_centered==False:
        frame_spatial = frame_spatial + np.tile(root_pos, int(frame_spatial.shape[0]/3))

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
    - node2frameidx : a dictionnary containing the name of the nodes and the indices 
                        of the euler angles column in the frames matrix. eg
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
    
    
def _nodes_container_to_nodes_list(nodes_container):
    """
    Input : a Bvh object or a list of Bvh nodes
    Output : 
     - a list of Bvh nodes
     - a bool that indicates if the nodes are from a Bvh object or not
    """
    from_bvh = True
    try:
        # first we test if it is a bvh object
        nodes = nodes_container.nodes
        root = nodes_container.root
        pos = root.pos_channels
    except:
        from_bvh = False
        try:
            # if it was not a bvh object, we try to see
            # if it is a list of nodes
            nodes = nodes_container
            root = nodes_container[0]
            pos = root.pos_channels
        except:
            raise ValueError('The nodes container needs to be either a Bvh object, or a list of Bvh nodes (as obtained from bvh_object.nodes for ex)')
    
    return nodes, from_bvh


def _change_offset(nodes, change_skeleton):
        """
        Change the offset of the nodes to the one in the provided bvh object.
        Input :
        - nodes : a list of BvhNode objects
        - use_skeleton : a Bvh object
        Output : 
        Return the nodes with changed offset.

        We check if the use_skeleton argument is actually a Bvh object.
        """
        # we need to check if the use_skeleton argument is a Bvh object first
        try:
            new_skel_nodes = change_skeleton.nodes
            new_skel_root = change_skeleton.root
            new_skel_root.rot_channels
        except:
            raise ValueError('The argument use_skeleton needs to be a Bvh object')
        
        newnodes2idx = {}
        for i, new_skel_node in enumerate(new_skel_nodes):
            newnodes2idx[new_skel_node.name] = i
        
        nodes = copy.deepcopy(nodes)

        for node in nodes:
            try:
                # we check if a node with the same name exists in the new skeleton 
                new_node_offset = new_skel_nodes[newnodes2idx[node.name]].offset
            except:
                raise ValueError(f"Could not find the node {node.name} in the provided change_skeleton bvh object")
            node.offset = new_node_offset

        return nodes

            
