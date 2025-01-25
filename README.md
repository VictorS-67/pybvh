## UPDATE : 
- 2025-01-25 : Change the parameter "local" in the functions to get spatial coordinates and to plot the bvh. The new parameter is "centered", with three modes:
    - "skeleton" : equivalent to previous local = True, root pos always 0, 0, 0:
    - "world" : equivalent to previous local = False, root at coordinates as in saved in bvh frames
    - "first" : the first frame root position is 0, 0, 0. From there, the skeleton moves in the space normally.

# pybvh
Python library to work with bvh files

The main point of this library is a Bvh class, which contains all the necesseray information found in a bvh file.
Through the use of this object, it is easy to read and write a bvh file, but also to convert it to a Pandas Dataframe, and conversely to transform a Dataframe into a bvh object.

See the jupyter file 'tutorial' for example of use.

### Curent functionality
- Bvh class
    - parameters : 
        - bvhobject.nodes : a list of all the nodes in the Hierarchy. The nodes are BvhRoot, BvhJoint or BvhNode objects for respectively the root, the joints and the end sites.
        - bvhobject.frames : the rotational data as a 2D numpy array.
        - bvhobject.frame_frequency : the frames frequency as can be found in a bvh file.
        - bvhobject.frame_template : the organized name of each column of the bvhobject.frames.
        - bvhobject.frame_count : the number of frames (=the number of lines of the bvhobject.frames parameter).
        - bvhobject.root : the root of the Hierarchy, aka bvhobject.nodes[0].
    - methods :
        - to_bvh_file(filepath, verbose=True) : save a bvh object to a bvh file at the location filepath (str or Path object).
        - get_spatial_coord(frame_num=-1, local=True) : get the spatial coordinates of every joints for all frames or only one.
        - get_df_constructor(mode = 'euler', local=True) : get a list of dictionnary that can be transmitted to a pd.Dataframe() constructor to directly obtain a Dataframe. Can construct a DataFrame with euler angles or spatial coordinates.
        - hierarchy_info_as_dict() : get a dictionnary describing the organisation of the Hierarchy in the bvh object.

- read_bvh_file(filepath) : read a .bvh file at the filepath location (str or Path object), and create a Bvh object

- df_to_bvh(hier, df) : df is a pandas DataFrame, containing joints rotational data. hier is either a dictionnary describing the hierarchy, or a list of nodes. Will create a bvh object based on those two arguments.

- plot.plot_frame(bvh_object, frame) : plot a matplotlib projection3d that shows the frame passed as a parameter.


### TODO:
- making direct changes to rot_channels and pos_channels impossible/regulated. Needs to go through a class method, so that we can also change the frames columns order (and value!) at the same time.
- obtaining graph dataset
- class method to transform euler angle directly (different Euler angle order, transformation to rotation matrix etc.)
- visualization of the bvh animation
- change docstrings to Numpy/Scipy standard
