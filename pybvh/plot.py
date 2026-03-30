from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .bvh import Bvh

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes




def plot_frame(bvh_object: Bvh, frame: int | np.ndarray, centered: str = "world") -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]:
    """Plot a single BVH frame as a 3D skeleton.

    Renders the skeleton for a given frame using matplotlib's 3D plotting.
    The viewing angle is automatically determined from the skeleton's
    orientation.

    Parameters
    ----------
    bvh_object : Bvh
        A Bvh object containing the skeleton hierarchy.
    frame : int or np.ndarray
        Frame to plot. If an int, it is used as a frame index into
        ``bvh_object``. If a 2-D array of shape ``(N, 3)``, it is
        treated as pre-computed spatial coordinates.
    centered : str, optional
        Coordinate centering mode, either ``"skeleton"`` or ``"world"``
        (default ``"world"``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : matplotlib.axes.Axes
        The 3D axes (single axes object when ``num_subplots == 1``),
        or a list of axes otherwise.
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
        bvh_object: Bvh,
        frames: int | np.ndarray = -1,
        centered: str = "world",
        savefile: bool = True,
        filepath: str | Path = Path('./anim.mp4'),
        direction_ref: str = 'rest',
        fps: int = -1,
        show_axis: bool = True,
        verbose: bool = False
        ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Animate a BVH skeleton and optionally save to a video file.

    Creates a 3D animation of the skeleton over the specified frames.
    The viewing angle is automatically determined from the skeleton's
    orientation, and the axis limits are computed to encompass all frames.

    Parameters
    ----------
    bvh_object : Bvh
        A Bvh object containing the skeleton hierarchy and motion data.
    frames : int or array-like, optional
        Frames to animate. If ``-1`` (default), all frames in the BVH
        object are used. If a single int, only that frame index is shown.
        If an array of ints, those frame indices are used. If a 2-D
        array of shape ``(N, 3)``, it is treated as a single frame of
        spatial coordinates. If a 3-D array of shape ``(F, N, 3)``, it
        is treated as pre-computed spatial coordinates for *F* frames.
    centered : str, optional
        Coordinate centering mode (default ``"world"``):

        - ``"skeleton"`` -- root is always at the origin.
        - ``"first"`` -- first frame root is at the origin; subsequent
          frames move normally.
        - ``"world"`` -- global coordinates are used as-is.
    savefile : bool, optional
        If ``True`` (default), save the animation to *filepath*.
    filepath : str or pathlib.Path, optional
        Output path for the saved animation (default ``'./anim.mp4'``).
        Supported formats include ``.mp4``, ``.mov``, ``.avi``, ``.gif``,
        ``.webp``, ``.apng``, and ``.html``. Only used when *savefile*
        is ``True``.
    direction_ref : str, optional
        Reference pose for determining the viewing direction (default
        ``'rest'``). ``'rest'`` uses the skeleton rest pose; ``'first'``
        uses the first animation frame.
    fps : int, optional
        Frames per second for the animation. If ``-1`` (default), the
        frame rate from the BVH file is used.
    show_axis : bool, optional
        If ``True`` (default), display the 3D axis. Set to ``False`` to
        hide it.
    verbose : bool, optional
        If ``True``, print additional information. Default is ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : matplotlib.axes.Axes
        The 3D axes used for the animation.
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

    directions_dict = _get_forw_up_axis(bvh_object, direction_pose)  # type: ignore[arg-type]

    fig, ax = _setup_plt_animation_world(frames, directions_dict)

    if not show_axis:
        ax.axis('off')

    # Create lines initially without data
    lines = [ax.plot([], [], [], c='blue', lw=2.5)[0] for _ in bvh_object.nodes[1:]]

    #calculate the times between frames
    if fps == -1:
        fps = 1 / bvh_object.frame_frequency  # type: ignore[assignment]

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
        bvh_object: Bvh,
        frames: int | np.ndarray,
        centered: str,
        savefile: bool,
        filepath: str | Path,
        direction_ref: str,
        fps: int
        ) -> tuple[np.ndarray, Path, str]:
    """Validate and normalise parameters for ``plot_animation``.

    Parameters
    ----------
    bvh_object : Bvh
        The BVH object.
    frames : int or array-like
        Raw *frames* argument from the caller.
    centered : str
        Coordinate centering mode.
    savefile : bool
        Whether to save the animation.
    filepath : str or pathlib.Path
        Output file path.
    direction_ref : str
        Reference pose identifier (``'rest'`` or ``'first'``).
    fps : int
        Frames per second (``-1`` for BVH default).

    Returns
    -------
    frames : np.ndarray
        Spatial coordinates array of shape ``(F, N, 3)``.
    filepath : pathlib.Path
        Validated (possibly adjusted) output path.
    writer : str
        Matplotlib animation writer name.
    """
    # ---- test frames
    if isinstance(frames, int):
        # if frames is an int, we need to check if it is in the bounds or -1
        if frames == -1 :
            frames = bvh_object.get_spatial_coord(centered=centered)
        elif frames>= 0 and frames < bvh_object.frame_count:
            # we need an iterable for the animation so we need to put the one frame into an array
            frames = np.array([bvh_object.get_spatial_coord(frame_num=frames, centered=centered)])
        else :
            raise ValueError("Index out of bounds for the frames.")
    else :
        # if frames is not an int, we need to check if it is an iterable
        try:
            _ = (f for f in frames)
        except:
            raise ValueError("frames should be either an int or an iterable.")
        #if we are here, then frames is an iterable
        frames_arr: np.ndarray = np.asarray(frames)
        if frames_arr.ndim == 1 and np.issubdtype(frames_arr.dtype, np.integer):
            # list of frame indices
            frames = bvh_object.get_spatial_coord(centered=centered)[frames_arr]
        elif frames_arr.ndim == 2:
            # single (N, 3) frame
            frames = frames_arr[np.newaxis, :]
        elif frames_arr.ndim == 3:
            # (F, N, 3) already
            if frames_arr.shape[1:] != (len(bvh_object.nodes), 3):
                raise ValueError(
                    f"Each frame should have shape ({len(bvh_object.nodes)}, 3).")
            frames = frames_arr
        else:
            raise ValueError("frames should be an int, list of ints, or array of spatial coords.")

    # ---- test savefile and savepath and available writers
    writer: str = 'pillow'  # default, overwritten below when savefile is True
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

    return frames, Path(filepath), writer  # type: ignore[return-value]


def _get_main_direction(coord_array: np.ndarray) -> str:
    """Return the signed axis string (e.g. ``'+y'``) for the dominant component.

    Parameters
    ----------
    coord_array : np.ndarray
        1-D array of length 3 representing an (x, y, z) vector.

    Returns
    -------
    main_dir : str
        Signed axis label such as ``'+x'``, ``'-z'``, etc.
    """
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


def _extract_sign(ax: str) -> bool:
    """Return ``True`` if the axis string has a ``'+'`` sign, ``False`` if ``'-'``.

    Parameters
    ----------
    ax : str
        Signed axis string, e.g. ``'+x'`` or ``'-z'``.

    Returns
    -------
    is_positive : bool
        ``True`` for positive, ``False`` for negative.
    """
    if ax[0] == '+':
        return True
    elif ax[0] == '-':
        return False
    else:
        raise ValueError("The sign of the axis should be either '+' or '-'.")


def _get_forw_up_axis(bvh_object: Bvh, frame: np.ndarray) -> dict[str, str]:
    """Infer the forward and upward axes from a skeleton frame (human only).

    Parameters
    ----------
    bvh_object : Bvh
        The BVH object containing the skeleton hierarchy.
    frame : np.ndarray
        Spatial coordinates of shape ``(N, 3)`` for a single frame.

    Returns
    -------
    directions : dict
        Dictionary with keys ``'forward'`` and ``'upward'``, each
        mapping to a signed axis string (e.g. ``'+y'``, ``'-z'``).
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
        up_joint_coord = local_coord[bvh_object.node_index[upward_joint.name]]
    except:
        # if not, use the second node (index 1) as the upward joint
        up_joint_coord = local_coord[1]

    up_ax = _get_main_direction(up_joint_coord)

    # for the front, we suppose that the last joint in the dictionary is one of the feet,
    # so it will point forward.
    foot_vector = local_coord[-1] - local_coord[-2]
    forward_ax = _get_main_direction(foot_vector)

    return {'forward': forward_ax, 'upward': up_ax}



def _setup_plt(frame_plotted: np.ndarray, num_subplots: int = 1, directions_dict: dict[str, str] = {}) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """Set up 3D subplot(s) with auto-determined viewing angles and axis limits.

    Parameters
    ----------
    frame_plotted : np.ndarray
        Spatial coordinates of shape ``(N, 3)`` used to compute axis limits.
    num_subplots : int, optional
        Number of subplots to create (max 3, default 1).
    directions_dict : dict, optional
        Dictionary with ``'forward'`` and ``'upward'`` signed axis strings.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    axs : list of matplotlib.axes.Axes
        List of 3D axes objects.
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


def _draw_skeleton(frame: np.ndarray, bvh_object: Bvh, lines: list[Any]) -> list[Any]:
    """Draw the skeleton bones for a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Spatial coordinates of shape ``(N, 3)``.
    bvh_object : Bvh
        The BVH object containing the skeleton hierarchy.
    lines : list of mpl_toolkits.mplot3d.art3d.Line3D
        Pre-allocated line artists (length ``len(bvh_object.nodes) - 1``).

    Returns
    -------
    lines : list of mpl_toolkits.mplot3d.art3d.Line3D
        The updated line artists.
    """
    name2index = bvh_object.node_index

    for line, node in zip(lines, bvh_object.nodes[1:]): #skip the root
        coord = frame[name2index[node.name]]
        parent_coord = frame[name2index[node.parent.name]]  # type: ignore[union-attr]
        line.set_data_3d([parent_coord[0], coord[0]],
                        [parent_coord[1], coord[1]],
                        [parent_coord[2], coord[2]])

    return lines


def _angle_up_forward(bvh_forward_ax: str, bvh_up_ax: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute elevation and azimuth arrays for the given forward/up axes.

    Parameters
    ----------
    bvh_forward_ax : str
        Signed forward axis, e.g. ``'+z'`` or ``'-x'``.
    bvh_up_ax : str
        Signed upward axis, e.g. ``'+y'`` or ``'-z'``.

    Returns
    -------
    elev : np.ndarray
        Elevation angles (one per subplot).
    azim : np.ndarray
        Azimuth angles (one per subplot).
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


def _setup_plt_animation_world(frames: np.ndarray, directions_dict: dict[str, str] = {}) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Set up the animation plot with axis limits spanning all frames.

    Parameters
    ----------
    frames : np.ndarray
        Spatial coordinates of shape ``(F, N, 3)``.
    directions_dict : dict, optional
        Dictionary with ``'forward'`` and ``'upward'`` signed axis strings.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : matplotlib.axes.Axes
        The 3D axes with limits adjusted for the full animation range.
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


    ax.axis(axis_lim)  # type: ignore[call-overload]
    return fig, ax
