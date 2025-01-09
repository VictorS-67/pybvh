import numpy as np
import matplotlib.pyplot as plt



def get_forw_up_axis(bvh_object, frame)->dict:
    """ 
        Return the upward and forward axis of the bvh file in a dictionnary format. 
        ONLY WORKS WITH HUMAN SKELETON!
        The axis are based on the analysis of the frame given as an argument.
        It works assuming that the skeleton is in a mostly straight or sitting position
        in the frame. In case of lying down, or weird position, will return wrong result.
    """

    axis2vector = {'+x': np.array([1,0,0]), '-x': np.array([-1,0,0]),
               '+y': np.array([0,1,0]), '-y': np.array([0,-1,0]),
               '+z': np.array([0,0,1]), '-z': np.array([0,0,-1])}
    vector2axis = {tuple(v): k for k,v in axis2vector.items()}

    up_body_parts = ["head", "neck", "chest", "spine"]
    for joint in bvh_object.nodes:
        if joint.name.lower() in up_body_parts:
            upward_joint = joint
            break
    # we try if one of the joint had the name in the list
    try:
        idx_up_joint = np.array([bvh_object.name2coord_idx[f'{upward_joint.name}_X'],
                    bvh_object.name2coord_idx[f'{upward_joint.name}_Y'],
                    bvh_object.name2coord_idx[f'{upward_joint.name}_Z']])
    except:
        #if not, we will assume that the first node after the root is the upward joint (spine or chest equivalent)
        idx_up_joint = np.array([3, 4, 5])

    up_joint_coord = frame[idx_up_joint]
    up_ax = _get_main_direction(up_joint_coord)

    # for the front, we suppose that the last joint in the dictionary is one of the feet,
    # so it will point forward. We can obtain the forward direction by subtracting the last two 
    # joints positions to obtain their direction vector

    foot_vector = frame[[-3, -2, -1]] - frame[[-6, -5, -4]]
    forward_ax = _get_main_direction(foot_vector)

    #forward_vector = axis2vector[forward_ax]
    #upward_vector = axis2vector[up_ax]

    return {'forward': forward_ax, 'upward': up_ax}



def setup_plt(frame_plotted, num_subplots=1, directions_dict = {}):
    if num_subplots >= 4:
        raise ValueError("Too many subplots")
    fig, axs = plt.subplots(1, num_subplots, subplot_kw=dict(projection="3d"))
    # if only one subplot we don't have a list so solve this small error
    if num_subplots == 1:
        axs = [axs]
    
    try:
        forward_ax = directions_dict['forward']
        up_ax = directions_dict['upward']
    except:
        forward_ax = "+x"
        up_ax = "+z"
        print("Couldn't find the 'forward' and 'upward' direction in the directions dictionnary.\
               Defaulted to x as the forward axis and z as the upward axis")

    forward_pos_sign = _extract_sign(forward_ax)
    forward_ax  = forward_ax[1]
    up_pos_sign = _extract_sign(up_ax)
    up_ax  = up_ax[1]

    #xmin, xmax = np.min(frame_plotted[0::3]), np.max(frame_plotted[0::3])
    #ymin, ymax = np.min(frame_plotted[1::3]), np.max(frame_plotted[1::3])
    #zmin, zmax = np.min(frame_plotted[2::3]), np.max(frame_plotted[2::3])

    lim_min, lim_max = np.min(frame_plotted), np.max(frame_plotted)

    fig_limit = max(np.abs(lim_min), lim_max)

    elevs = [0,0,0]
    azims = [0,0,0]
    
    for i, ax in enumerate(axs):
        #ax.set_axis_off()
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')

        ax.set_xlim(-fig_limit, fig_limit)
        ax.set_ylim(-fig_limit, fig_limit)
        ax.set_zlim(-fig_limit, fig_limit)

        """
        #calculate the limits of the figure. 
        if figure_limit == None:
            lim_min = np.abs(np.min(local_pos[list(local_pos)[-1]]))
            lim_max = np.abs(np.max(local_pos[list(local_pos)[-1]]))
            lim = lim_min if lim_min > lim_max else lim_max
            figure_limit = lim
        """
        elev = elevs[i]
        azim = azims[i]
        if not up_pos_sign:
            # if the upward sign is of negative sign, for ex -z
            # the we turn the elevation by 180
            elev += 180 
        ax.view_init(elev=elev+20, azim=azim-20, vertical_axis=up_ax)
        #ax.view_init(vertical_axis=up_ax)

    return fig, axs

def plot_frame(bvh_object, frame):

    """ 
    NOTE : accept euler angle and automat convert to spatial
    """

    directions_dict = get_forw_up_axis(bvh_object, frame)
    fig, axs = setup_plt(frame, directions_dict=directions_dict)

    name2index = bvh_object.name2coord_idx

    for node in bvh_object.nodes[1:]:
        #if 'End Site' in node.name:
        #    continue
        col_idx = [name2index[f'{node.name}_X'], name2index[f'{node.name}_Y'], name2index[f'{node.name}_Z']]
        parent_col_idx = [name2index[f'{node.parent.name}_X'], name2index[f'{node.parent.name}_Y'], name2index[f'{node.parent.name}_Z']]
        coord = frame[col_idx]
        parent_coord =frame[parent_col_idx]
        axs[0].plot(xs = [parent_coord[0], coord[0]],
                ys = [parent_coord[1], coord[1]],
                zs = [parent_coord[2], coord[2]],
                c = 'blue',
                lw = 2.5)
        
    plt.tight_layout()
    plt.show()



def _get_main_direction(coord_array)->str:
    main_direction_idx = np.argmax(np.abs(coord_array))
    if coord_array[main_direction_idx] < 0:
        main_dir = "-"
    else:
        main_dir = "+"

    if main_direction_idx == 0:
        main_dir += "x"
    elif main_direction_idx == 1:
        main_dir += "y"
    elif main_direction_idx == 2:
        main_dir += "z"
    else:
        raise ValueError("Invalid index")
    
    return main_dir



def _extract_sign(ax:str)->bool:
    if ax[0] == '+':
        return True
    elif ax[0] == '-':
        return False
    else:
        raise ValueError("The sign of the axis should be either '+' or '-'.")