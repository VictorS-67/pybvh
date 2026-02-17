import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

from pathlib import Path





def plot_frame(bvh_object, frame, centered = "world"):
    """ 
    Plot a bvh frame.
    Input:
    - bvh_object : A Bvh object.
    - frame : can be an int (frame index) or a 2-D array of shape ``(N, 3)``
              with spatial coordinates.
    - centered : ``"skeleton"`` or ``"world"``.
    """
    num_subplots = 1

    if isinstance(frame, int):
        frame = bvh_object.get_spatial_coord(frame_num=frame, centered=centered)
    elif isinstance(frame, np.ndarray):
        pass
    else:
        raise ValueError("frame should be either an int or a numpy array.")

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



def plot_animation(
        bvh_object, 
        frames=-1,
        centered = "world",
        savefile = True,
        filepath = Path('./anim.mp4'),
        direction_ref = 'rest',
        fps = -1,
        show_axis = True,
        verbose = False
        ):
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
    - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton", the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0]).
                If "first", the first frame root position is considered to be [0, 0, 0]. From there,
                the skeleton moves in the space normally.
                If "world", the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root).
    - savefile : a bool, if true will save the file at the path in filepath
    - filepath : a string or a Path object, path where to save the video file.
                only used when savefile == True.
    - direction_ref : either 'rest' or 'first'.
                    If 'res', the view direction of the plot is based
                         on the skeleton rest pose.
                    If 'first',  the view direction of the plot is based
                         on the skeleton in the first frame.
    - fps : if -1, then use the frame rate indicated in the bvh file.
            else, use 
    """

    # --------------- prooftesting the parameters ------------
    frames, filepath, writer = _prooftest_plot_animation_parameters(
        bvh_object,
        frames,
        centered,
        savefile,
        filepath,
        direction_ref,
        fps
        )
    #frames is now an np.array of shape (*, N, 3)

    ## --------------- end of prooftesting the parameters------

    if direction_ref == 'rest':
        direction_pose = bvh_object.get_rest_pose(mode='coordinates')
    else:
        direction_pose = frames[0]

    directions_dict = _get_forw_up_axis(bvh_object, direction_pose)

    fig, ax = _setup_plt_animation_world(frames, directions_dict)

    if not show_axis:
        ax.axis('off')

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
        if writer == "jshtml":
            html_content = anim.to_jshtml() 
            with open(filepath, 'w') as f:
                f.write(html_content)
        else:
            anim.save(Path(filepath), writer=writer)
    #plt.show()

    return fig, ax


def _prooftest_plot_animation_parameters(
        bvh_object,
        frames,
        centered,
        savefile,
        filepath,
        direction_ref,
        fps
        ):
    # ---- test frames
    if isinstance(frames, int):
        # if frames is an int, we need to check if it is in the bounds or -1
        if frames == -1 :
            frames = bvh_object.get_spatial_coord(centered=centered)
        elif frames>= 0 and frames < bvh_object.frame_count:
            # we need an iterable for the animation so we need to put the one frame into a list
            frames = [bvh_object.get_spatial_coord(frame_num=frames, centered=centered)]
        else : 
            raise ValueError("Index out of bounds for the frames.")
    else :
        # if frames is not an int, we need to check if it is an iterable
        try:
            _ = (f for f in frames)
        except:
            raise ValueError("frames should be either an int or an iterable.")
        #if we are here, then frames is an iterable
        frames = np.asarray(frames)
        if frames.ndim == 1 and np.issubdtype(frames.dtype, np.integer):
            # list of frame indices
            frames = bvh_object.get_spatial_coord(centered=centered)[frames]
        elif frames.ndim == 2:
            # single (N, 3) frame
            frames = frames[np.newaxis, :]
        elif frames.ndim == 3:
            # (F, N, 3) already
            if frames.shape[1:] != (len(bvh_object.nodes), 3):
                raise ValueError(
                    f"Each frame should have shape ({len(bvh_object.nodes)}, 3).")
        else:
            raise ValueError("frames should be an int, list of ints, or array of spatial coords.")
    
    # ---- test savefile and savepath and available writers
    if savefile:
        output_path = Path(filepath)
        ext = output_path.suffix.lower()
        
        # 1. Check available writers
        has_ffmpeg = animation.writers.is_available('ffmpeg')
        
        # 2. Logic for Video (.mp4, .mov, .avi)
        if ext in ['.mp4', '.mov', '.avi']:
            if has_ffmpeg:
                writer='ffmpeg'
            else:
                # Smart Fallback: Change to GIF
                filepath = output_path.with_suffix('.gif')
                warnings.warn(
                    f"FFmpeg not found! Cannot save as {ext}.\n"
                    f"Falling back to PillowWriter and saving as '{filepath}' instead.\n"
                    f".webp and .html are also available."
                )
                writer='pillow'

        # 3. Logic for Image Formats (.gif, .webp, .apng)
        elif ext in ['.gif', '.webp', '.apng']:
            # Pillow handles all of these natively now
            writer='pillow'

        # 4. Logic for HTML
        elif ext == '.html':
            # to_jshtml() is superior to writer='html' because it embeds the data
            writer='jshtml'

        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
    # ---- test direction_ref
    if direction_ref not in ['rest', 'first']:
        raise  ValueError("direction_ref should be either 'rest' or 'first'.")
    
    # ---- test fps
    if not isinstance(fps, int) or fps==0 or fps < -1 :
        raise  ValueError("fps should be -1 or a positiv int")
    
    # ---- test centered_options
    centered_options = ['skeleton', 'first', 'world']
    if centered not in centered_options:
        raise ValueError(f'The value {centered} is not recognized for the centered argument.\
                            Currently recognized keywords are {centered_options}')
    
    return frames, filepath, writer


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
        frame shape is (N, 3).
    """

    # work with local coordinates (root at origin)
    local_coord = frame - frame[0]  # (N, 3) - (3,) broadcast

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
        up_joint_coord = local_coord[bvh_object.name2idx[upward_joint.name]]
    except:
        # if not, use the second node (index 1) as the upward joint
        up_joint_coord = local_coord[1]

    up_ax = _get_main_direction(up_joint_coord)

    # for the front, we suppose that the last joint in the dictionary is one of the feet,
    # so it will point forward.
    foot_vector = local_coord[-1] - local_coord[-2]
    forward_ax = _get_main_direction(foot_vector)

    return {'forward': forward_ax, 'upward': up_ax}



def _setup_plt(frame_plotted, num_subplots=1, directions_dict = {}):
    """
    setup the axis of the subplots.
    create up to 3 subplots, with different viewing angles  --- not yet, only one
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

    root_pos = frame_plotted[0]  # (3,)
    local_coord = frame_plotted - root_pos  # broadcast

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
    Function to draw the bvh skeleton in one frame.
    frame is shape (N, 3).  lines has length len(bvh_object.nodes) - 1.
    """
    name2index = bvh_object.name2idx

    for line, node in zip(lines, bvh_object.nodes[1:]): #skip the root
        coord = frame[name2index[node.name]]
        parent_coord = frame[name2index[node.parent.name]]
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
    x_min, x_max= np.min(frames[:,:,0]),  np.max(frames[:,:,0])
    y_min, y_max= np.min(frames[:,:,1]),  np.max(frames[:,:,1])
    z_min, z_max= np.min(frames[:,:,2]),  np.max(frames[:,:,2])

    

    dist_x, dist_y, dist_z = x_max-x_min, y_max-y_min, z_max-z_min 
    middle_x, middle_y, middle_z = x_min + dist_x/2 , y_min + dist_y/2, z_min + dist_z/2
    # the middle_x etc are now the coordinates for the middle
    #  of the wholle range of movement following the given axis
    fig_lim = max(dist_x, dist_y, dist_z)

    axis_lim = [middle_x - fig_lim/2 , middle_x + fig_lim/2,
                middle_y - fig_lim/2 , middle_y + fig_lim/2,
                middle_z - fig_lim/2 , middle_z + fig_lim/2]


    ax.axis(axis_lim)
    return fig, ax