import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path





def plot_frame(bvh_object, frame):

    """ 
    NOTE : accept euler angle and automat convert to spatial ?
    """
    num_subplots = 1

    directions_dict = _get_forw_up_axis(bvh_object, frame)
    fig, axs = _setup_plt(frame, num_subplots=num_subplots, directions_dict=directions_dict)

    lines = [axs[0].plot([], [], [], c='blue', lw=2.5)[0] for _ in bvh_object.nodes[1:]]

    lines = _draw_skeleton(frame, bvh_object, lines)
        
    plt.tight_layout()
    #plt.show()
    if num_subplots == 1:
        return fig, axs[0]
    else:
        return fig, axs



def plot_animation(bvh_object, frames=-1,
                   local = True,
                   savefile = True,
                   filepath = Path('./anim.mp4'),
                   direction_ref = 'rest',
                   fps = -1):
    """
    Plot the bvh animation.
    Input:
    - bvh_object : A Bvh object.
    - frames : can be -1, a frame index, a list of frames (as contained in bvh.frames), or a list of frame indices
        - if -1 (default setting) : plot all the frames in the bvh object
        - if an int : plot the one frame indexed by the argument.
        - if a list of frames : plot all the frames passed as argument. 
              The frames line needs to be of same format as the frames line in the bvh object
        - if a list of frame indices : plot all the frames indexed by the argument
    - savefile : a bool, if true will save the file at the path in filepath
    -  filepath : a string or a Path object, path where to save the video file.
                only used when savefile == True.
    - direction_ref : either 'rest' or 'first'.
                    If 'res', the view direction of the plot is based
                         on the skeleton rest pose.
                    If 'first',  the view direction of the plot is based
                         on the skeleton in the first frame.
    - fps : if -1, then use the frame rate indicated in the bvh file.
            else, use 
    """
    if isinstance(frames, int):
        if frames == -1 :
            frames = bvh_object.get_spatial_coord(local=local)
        elif frames>= 0 and frames < bvh_object.frame_count:
            # we need an iterable for the animation so we need to put the frame into a list
            frames = [bvh_object.get_spatial_coord(frame_num=frames, local=local)]
        else : 
            raise ValueError("Index out of bounds for the frames.")
    else :
        try:
            _ = (f for f in frames)
        except:
            raise ValueError("frames should be either an int or an iterable.")
        #if we are here, then frames is an iterable
        if isinstance(frames[0], int):
            frames = bvh_object.get_spatial_coord(local=local)[frames]
        else:
            try:
                if len(frames[0]) != (len(bvh_object.nodes)) *3:
                    raise ValueError(f"The length of the frame inside the\
                                      iterable given as argument should be {(len(bvh_object.nodes)) *3}.")
                else:
                    #if everything is good, we can use the frames as is
                    frames = frames
            except:
                #if len(frames[0]) created a problem
                raise ValueError("frames should be either an int or an iterable.")
    
    if direction_ref not in ['rest', 'first']:
        raise  ValueError("direction_ref should be either 'rest' or 'first'.")
    
    if not isinstance(fps, int) or fps==0 or fps < -1 :
        raise  ValueError("fps should be -1 or a positiv int")
    
    ## -- end of prooftesting the parameters

    if direction_ref == 'rest':
        direction_pose = bvh_object.get_rest_pose(mode='coordinates')
    else:
        direction_pose = frames[0]

    directions_dict = _get_forw_up_axis(bvh_object, direction_pose)

    # if local is true, then we don't need to extend the limit of the axis
    # due to the skeleton moving around (origin = root)
    if local:
        fig, axs = _setup_plt(frames[0], directions_dict=directions_dict)
        ax = axs[0]
    else:
        # if local = False on the other hand, we do need to adjust
        fig, ax = _setup_plt_animation_world(frames, directions_dict)

    # Create lines initially without data
    lines = [ax.plot([], [], [], c='blue', lw=2.5)[0] for _ in bvh_object.nodes[1:]]

    #calculate the times between frames
    if fps == -1:
        fps = 1 / bvh_object.frame_frequency

    interval = int(1/fps*1000) #miliseconds
    # Creating the Animation object
    anim = animation.FuncAnimation(
        fig, _draw_skeleton, frames, fargs=(bvh_object, lines), interval=interval)
    
    if savefile:
        anim.save(Path(filepath))
    #plt.show()

    return fig, ax





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
    

def _get_forw_up_axis(bvh_object, frame)->dict:
    """ 
        Return the upward and forward axis of the bvh file in a dictionnary format. 
        ONLY WORKS WITH HUMAN SKELETON!
        The axis are based on the analysis of the frame given as an argument.
        It works assuming that the skeleton is in a mostly straight or sitting position
        in the frame. In case of lying down, or weird position, will return wrong result.
    """

    #we want to work with the local coordinates here:
    frame = np.copy(frame)
    for i in range(3):
        frame[i::3] = frame[i::3] - frame[i]

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



def _setup_plt(frame_plotted, num_subplots=1, directions_dict = {}):
    """
    setup the axis of the subplots.
    create up to 3 subplots, with different viewing angles  --- not yet, only 0ne
    this function serves to calculate the best viewing angles,
    based on the info contained in the argument directions_dict.
    Also determine the plot limits based on the displayed frame.
    """
    if num_subplots >= 4:
        raise ValueError("Too many subplots.")
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

    elev_list, azim_list = _angle_up_forward(forward_ax, up_ax)

    root_pos = frame_plotted[[0,1,2]]
    local_coord = np.copy(frame_plotted)
    for i in range(3):
        local_coord[i::3] = local_coord[i::3] - local_coord[i]

    lim_min, lim_max = np.min(local_coord), np.max(local_coord)

    ax2id = {'x':0, 'y':1, 'z':2}
    up_id = ax2id[up_ax[1]]
    non_up_id = [0,1,2]
    non_up_id.remove(up_id)

    x_min_max = [root_pos[0] + lim_min, root_pos[0] + lim_max]
    y_min_max = [root_pos[1] + lim_min, root_pos[1] + lim_max]
    z_min_max = [root_pos[2] + lim_min, root_pos[2] + lim_max]
    ax_lims = x_min_max + y_min_max + z_min_max #[xmin, xmax, ymin, ymax, zmin, zmax]
    
    for i, ax in enumerate(axs):
        ax.axis(ax_lims)

        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')

        elev = elev_list[i]
        azim = azim_list[i]

        ax.view_init(elev=elev, azim=azim, vertical_axis=up_ax[1])
        #ax.view_init(vertical_axis=up_ax)

    return fig, axs


def _draw_skeleton(frame, bvh_object, lines):
    """
    Function to draw the bvh skeleton in one frame
    by creating Line3D object
    lines needs to be of length len(bvh_object.nodes - 1) 
    """
    name2index = bvh_object.name2coord_idx

    for line, node in zip(lines, bvh_object.nodes[1:]): #skip the root
        col_idx = [name2index[f'{node.name}_X'], name2index[f'{node.name}_Y'], name2index[f'{node.name}_Z']]
        parent_col_idx = [name2index[f'{node.parent.name}_X'], name2index[f'{node.parent.name}_Y'], name2index[f'{node.parent.name}_Z']]
        coord = frame[col_idx]
        parent_coord =frame[parent_col_idx]
        line.set_data_3d([parent_coord[0], coord[0]],
                        [parent_coord[1], coord[1]],
                        [parent_coord[2], coord[2]])
    
    return lines


def _angle_up_forward(bvh_forward_ax, bvh_up_ax):
    """
    function to return the elevation and azimut necessary
    to have the correct axis pointing forward for the skeleton.
    bvh_forward_ax and bvh_up_ax needs to be of the form sign+axisletter:
    "+z", "-x" etc

    We assume that the vertical axis fof the view  will be determined with 
    ax.view_init(vertical_axis=bvh_up_ax), and so
    bvh upward axis = matplolib vertical axis, modulo sign.
    """
    elev, azim = np.array([20,0,0]), np.array([-20,0,0])

    bvh_forward_pos_sign = _extract_sign(bvh_forward_ax)
    bvh_forward_ax  = bvh_forward_ax[1]
    bvh_up_pos_sign = _extract_sign(bvh_up_ax)
    bvh_up_ax  = bvh_up_ax[1]
     # --- first we identify the front and turn to the front

    #if up is ..., then front is ... as default config in matplotlib view_init
    default_up2front = {'z':'x', 'y':'z', 'x':'y'}

    if bvh_forward_ax != default_up2front[bvh_up_ax]:
        azim += 90

    if not bvh_up_pos_sign:
        # if the upward axis sign is of negative sign, for ex -z
        # the we turn the elevation by 180
        elev += 180 
        azim += 180
    
    if bvh_forward_ax == default_up2front[bvh_up_ax] and not bvh_forward_pos_sign:
        #if the forward axis sign is of negative sign
        azim += 180

    return elev, azim


def _setup_plt_animation_world(frames, directions_dict={}):
    """ 
    Setup the plot with _setup_plt.
    Recalculate the axis dimension of the plot to take all the frames
    into account
    """
    fig, axs = _setup_plt(frames[0], directions_dict=directions_dict)
    ax = axs[0]
    # we have axis with the proper up and forward dir. 
    # Just need to recalculate the limits
    x_min, x_max= np.min(frames[:,0::3]),  np.max(frames[:,0::3])
    y_min, y_max= np.min(frames[:,1::3]),  np.max(frames[:,1::3])
    z_min, z_max= np.min(frames[:,2::3]),  np.max(frames[:,2::3])

    

    dist_x, dist_y, dist_z = x_max-x_min, y_max-y_min, z_max-z_min 
    middle_x, middle_y, middle_z = x_min + dist_x/2 , y_min + dist_y/2, z_min + dist_z/2
    # the middle_x etc are now the coordinates for the middle
    #  of the whoe range of movement following the given axis
    fig_lim = max(dist_x, dist_y, dist_z)

    axis_lim = [middle_x - fig_lim/2 , middle_x + fig_lim/2,
                middle_y - fig_lim/2 , middle_y + fig_lim/2,
                middle_z - fig_lim/2 , middle_z + fig_lim/2]


    ax.axis(axis_lim)
    return fig, ax