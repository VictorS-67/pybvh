from pathlib import Path
import numpy as np
import copy

from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .spatial_coord import frames_to_spatial_coord
from . import rotations

class Bvh:
    """
    takes as initial data the path of a bvh file
    the hierarchy information are stored in a list of BvhNode objects, one object per joint.
    the frames are stored as a numpy 2D array.
    """
    def __init__(self, nodes=[BvhRoot()], frames = np.array([[]]), frame_frequency=0, frame_template=[]):
        self.nodes = nodes
        self.frames = frames
        self.frame_frequency = frame_frequency
        self.frame_count = len(self.frames)
        self.root = self.nodes[0]

        if frame_template != []:
            self.frame_template = frame_template
        else:
            self._create_frame_template()
        
        # create the self.name2coord_idx parameter
        # a dictionnary that will allow to easily access the space coordinates
        self._create_name2coord_idx()
        

    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, value):
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("nodes should be a list of BvhNode class/subclasse objects")
        self._nodes = value 
        
    @property
    def frames(self):
        return self._frames
    @frames.setter
    def frames(self, value):
        if isinstance(value, list):
            value = np.array(list)
        if not isinstance(value, np.ndarray):
            raise ValueError("frames should be a 2D list or numpy array")
        self._frames = value

    @property
    def frame_template(self):
        return self._frame_template
    @frame_template.setter
    def frame_template(self, value):
        if (not isinstance(value, list)) or len(value) != self.frames.shape[1] :
            raise ValueError("frame_template should be a list, and len(frame_template) == frames.shape[1]")
        self._frame_template = value

    @property
    def frame_frequency(self):
        return self._frame_frequency
    @frame_frequency.setter
    def frame_frequency(self, value):
        self._frame_frequency = value
        
    @property
    def frame_count(self):
        return self._frame_count
    @frame_count.setter
    def frame_count(self, value):
        if self._frames.shape[1] == 0:
            self._frame_count = 0
        else:
            self._frame_count = value

    @property
    def root(self):
        return self._root
    @root.setter
    def root(self, value):
        if not isinstance(value, BvhRoot):
            raise ValueError("The first element of nodes should be a BvhRoot object")
        self._root = value

            
    def __str__(self):
        count_joints =  0
        for node in self.nodes:
            if not node.is_end_site() : count_joints += 1 
        return f'{count_joints} elements in the Hierarchy, {self.frame_count} frames at a frequency of {self.frame_frequency:.6f}Hz'
        
    def __repr__(self):
        nodes_str = []
        for node in self.nodes:
            sep = node.__str__().split()
            if sep[0] == 'ROOT':
                nodes_str += [node.__str__()]
            elif sep[0] == 'JOINT':
                nodes_str += [sep[1]]
        nodes_str = ''.join(str(nodes_str).split("'"))

        frames_str = f'array(shape={self.frames.shape}, dtype={self.frames.dtype})'
        
        return f'Bvh(nodes={nodes_str}, frames={frames_str}, frame_frequency={self.frame_frequency:.6f})'
    
    def copy(self):
        return copy.deepcopy(self)

    
    def _create_frame_template(self):
        if self.frames.shape == (1,0):
            #if we are creating an empty object or object with empty frames
            self.frame_template = []
            return
        
        frame_template = []
        root = self.nodes[0]
        for ax in ['X', 'Y', 'Z']:
            frame_template += [f'{root.name}_{ax}_pos']
        for node in self.nodes:
            if node.is_end_site():
                continue
            for ax in node.rot_channels:
                frame_template += [f'{node.name}_{ax}_rot']

        self.frame_template = frame_template



    def to_bvh_file(self, new_filepath, verbose=True):
        """
        This function will write the bvh object into a bvh file, following the proper standard for this type of file.
        If verbose is true, will tell when the saving is succesful.
        """
        #first test the name of the filepath
        new_filepath = Path(new_filepath)
        if new_filepath.suffix != '.bvh':
            raise Exception(f'{new_filepath.name} is not a bvh file')
        elif not new_filepath.parent.exists() :
            raise Exception(f'{new_filepath.parent} is not a valid directory')

        #if everything is good with the place of the file, we go on to writing the file
        #first we define a few auxilliary functions
        def offset_to_str(node):
            offset_str = 'OFFSET'
            for num in node.offset:
                offset_str += ' ' + f'{num:.6f}'
            return offset_str
        
        def channels_to_str(node):
            chanels_str = 'CHANNELS'
            if node.parent == None:
                #if it's a root 6 channels (pos + rot) 
                chanels_str += ' 6'
                for pos_ax in node.pos_channels:
                    chanels_str += ' ' + pos_ax + 'position'
            else:
                chanels_str += ' 3'
                
            for rot_ax in node.rot_channels :
                chanels_str += ' ' + rot_ax + 'rotation'
                
            return chanels_str
        
        
        def rec_node_to_file(node, file = None, depth = 0):
            #end condition of the recurrence is reaching an End site
            if node.is_end_site():
                print('\t'*depth + 'End Site', file=file)
                print('\t'*depth + '{', file=file)
                print('\t'*(depth+1) + offset_to_str(node), file=file)
                print('\t'*depth + '}', file=file)
            else:
                #root or joint
                if node.parent == None:
                    type = 'ROOT'
                else:
                    type = 'JOINT'
                print('\t'*depth + type + ' ' + node.name, file=file)
                print('\t'*depth +'{', file=file)
                print('\t'*(depth+1) + offset_to_str(node), file=file)
                print('\t'*(depth+1) + channels_to_str(node), file=file)
                for child in node.children:
                    rec_node_to_file(child, file = file, depth = depth+1)
                print('\t'*depth +'}', file=file)

        #from there, we can write the actual bvh file
        with open(new_filepath, "w") as f:
            f.write('HIERARCHY\n')
            
            rec_node_to_file(self.root, file =  f)
        
            f.write('MOTION\n')
            f.write(f'Frames: {self.frame_count}\n')
            f.write(f'Frame Time: {self.frame_frequency:.6f}\n')

            for frame in self.frames:
                f.write(np.array2string(frame,
                                        formatter={'float_kind':lambda x: "%.6f" % x},
                                        max_line_width=10000000
                                       )[1:-1])
                f.write(f'\n')

        if verbose:
            print(f'Succesfully saved the file {new_filepath.name} at the location\n{new_filepath.parent.absolute()}')
        #-------------- end of the write function


    def get_spatial_coord(self, frame_num=-1, centered="world"):
        """
        Obtain the spatial coordinates of the joints.
        The coordinates are given in the form of a numpy array.
        
        Note: This method always recomputes spatial coordinates. If you need to call
        this repeatedly with the same data, cache the result yourself:
            coords = bvh.get_spatial_coord(-1)  # compute once, reuse coords
        
        Input :
        - frame_num : if -1 (default value) then all the frames will be returned
                        converted into spatial coordinates for the joints
                    Else if any int >= 0 is given, then return the spatial 
                    coordinates for the frame number corresponding to frame_num
        - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton" , the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0] in ALL frames).
                If "first", the first frame root position is considered to be [0, 0, 0]. From there,
                the skeleton moves in the space normally. If only one frame to return, then
                "skeleton" and "first" will give the same result.
                If "world", the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root saved in the bvh object in all frames).

        """
        centered_options = ['skeleton', 'first', 'world']
        if centered not in centered_options:
            raise ValueError(f'The value {centered} is not recognized for the centered argument.\
                             Currently recognized keywords are {centered_options}')

        if (frame_num >= 0) and (frame_num < self.frame_count):
            # Single frame requested
            return_one_frame = True
            frame = frames_to_spatial_coord(self, frames=self.frames[frame_num], centered="world")
        elif frame_num == -1:
            # All frames requested
            return_one_frame = False
            frames_array = frames_to_spatial_coord(self, frames=self.frames, centered="world")
        else:
            raise ValueError("frame_num needs to be -1 or a positive integer smaller than the total amount of frames in the bvh file.")
            
        # Apply centering transformation
        if centered == "world":
            return frame if return_one_frame else frames_array
        elif centered == "first":
            first_frame = frames_to_spatial_coord(self, frames=self.frames[0], centered="world")
            offset = np.tile(first_frame[:3], len(first_frame) // 3)
            if return_one_frame:
                return frame - offset
            else:
                return frames_array - offset
        elif centered == "skeleton":
            if return_one_frame:
                return frame - np.tile(frame[:3], len(frame) // 3)
            else:
                return frames_array - np.tile(frames_array[:, :3], frames_array.shape[1] // 3)

        

    def get_rest_pose(self, mode='coordinates'):
        """
        Return the rest pose of the skeleton.
        Input : - mode cam be 'euler' or 'coordinates'.
                If 'euler', the rest pose is returned as euler angles.
                If 'coordinates', the rest pose is returned as spatial coordinates
        """
        correct_modes = ['euler', 'coordinates']
        rest_angle = np.zeros_like(self.frames[0])
        if mode == 'euler':
            return rest_angle
        elif mode == 'coordinates':
            rest_coord = frames_to_spatial_coord(self, frames=rest_angle, centered="skeleton")
            return rest_coord
        else:
            raise ValueError(f'The value {mode} is not recognized for the mode argument.\
                             Currently recognized keywords are {correct_modes}')
        

    def get_df_constructor(self, mode = 'euler', centered="world"):
        """
        This function returns a list of dictionnary ready made to easily create a pandas DataFrame. 
        Input :
        - mode : can be 'euler' or 'coordinates'.
                If the mode is 'euler', the constructed DataFrame will contain
                the rotational data, as they are in a raw bvh file.
                If the mode is 'coordinates', the constructed DataFrame will contain
                the spatial coordinates of the joints (including the End sites)
        - centered : a string that can be either "skeleton", "first", or "world".
                If "skeleton" , the coordinates are local to the skeleton (meaning the root
                coordinates are considered to be [0, 0, 0] in ALL frames).
                If "first", the first frame root position is considered to be [0, 0, 0]. From there,
                the skeleton moves in the space normally. If only one frame to return, then
                "skeleton" and "first" will give the same result.
                If "world", the coordinates are global (meaning the root coordinates are
                the actual coordinates of the root saved in the bvh object in all frames).

        The purpose of this function is to be combined with pd.DataFrame this way:
        pd.Dataframe(bvh_object.get_df_constructor).
        Each line of the df represents a frame of the bvh animation.

        If the mode is 'euler', the colums will be of the form
        'name_ax_pos/rot' (ex: Neck_X_rot'). The End site will not appear,
        as they do not have rotational data

        If the mode is 'coordinates', the colums will be of the form
        'name_ax' (ex: Neck_X'). The End site will appear in the dataframe.
        Furthermore, in case the mode is 'coordinates', the argu
        """
        correct_modes = ['euler', 'coordinates']

        if mode == 'euler':
            return self._get_df_constructor_euler_angles()
        elif mode == 'coordinates':
            return self._get_df_constructor_spatial_coord(centered=centered)
        else : 
            raise ValueError(f'The value {mode} is not recognized for the mode argument.\
                             Currently recognized keywords are {correct_modes}')
        
    
    def _get_df_constructor_euler_angles(self):
        """
        internal function called by self.get_df_constructor()
        in case the mode is 'euler'
        """
        dictList = []
        
        for i, frame in enumerate(self.frames):
            tempo_dict = {}
            tempo_dict['time'] = i*self.frame_frequency
            for j, col_name in enumerate(self.frame_template):
                tempo_dict[col_name] = frame[j]

            dictList.append(tempo_dict)

        return dictList

    def _get_df_constructor_spatial_coord(self, centered):
        """
        internal function called by self.get_df_constructor() in case its mode is 'coordinates'
        """
        
        spatial_array = self.get_spatial_coord(centered=centered)

        dictList = []

        colum_names = []
        for node in self.nodes:
            for ax in ['X', 'Y', 'Z']:
                colum_names += [f'{node.name}_{ax}']

        for i, frame in enumerate(spatial_array):
            tempo_dict = {}
            tempo_dict['time'] = i*self.frame_frequency

            for j, col_name in enumerate(colum_names):
                tempo_dict[col_name] = frame[j]

            dictList.append(tempo_dict)

        return dictList



    
    def hierarchy_info_as_dict(self):
        """
        Return a dictionary containing the hierarchy information, with the format
        {'name1' : {
                  'pos_channels' = ['X', 'Y', 'Z'], #if root
                  'rot_channels' = ['X', 'Y', 'Z'],
                  'offset': [float, float, float],
                  'parent' : 'nameOfParent',
                  'children' : ['nameOfChild1',... ],
                  },
        'name2' : { ... }, 
        ...
        }
        """
        hier_dict = {}
        for node in self.nodes:
            hier_dict[node.name] = {'offset' : node.offset}
            if isinstance(node, BvhRoot):
                hier_dict[node.name]['pos_channels'] = node.pos_channels
            if isinstance(node, BvhJoint):
                hier_dict[node.name]['rot_channels'] = node.rot_channels
                hier_dict[node.name]['children'] = [child.name for child in node.children]
            hier_dict[node.name]['parent'] = None if node.parent is None else node.parent.name
            
        return copy.deepcopy(hier_dict)
    
    def _create_name2coord_idx(self):
        name2coord_idx = {}
        i=0
        for node in self.nodes:
            for ax in ['X', 'Y', 'Z']:
                name2coord_idx[f'{node.name}_{ax}'] = i
                i+=1
        self.name2coord_idx = name2coord_idx





    def change_skeleton(self, new_skeleton, inplace=False):
        """
        Create new nodes with the offset equals to ones in the "new_skeleton" bvh object.
        Input :
        - use_skeleton : a Bvh object
        - inplace : if True, the function will modify the nodes in place.
                    if False, the function will return a new Bvh objects
        Output : 
        If inplace is True, the function returns None.
        If inplace is False, the function returns a new Bvh object with the new skeleton
        """
        # need to check if the use_skeleton argument is a Bvh object first
        try:
            new_skel_nodes = new_skeleton.nodes
            new_skel_root = new_skeleton.root
            root_rot = new_skel_root.rot_channels
        except:
            raise ValueError('The argument use_skeleton needs to be a Bvh object')
        
        
        newnodes2idx = {}
        for i, new_skel_node in enumerate(new_skel_nodes):
            newnodes2idx[new_skel_node.name] = i

        if inplace:
            nodes = self.nodes
        else:
            new_bvh = self.copy()
            nodes = new_bvh.nodes
        
        for node in nodes:
            try:
                # we check if a node with the same name exists in the new skeleton 
                new_node_offset = new_skel_nodes[newnodes2idx[node.name]].offset
            except:
                raise ValueError(f"Could not find the node {node.name} in the provided new_skeleton bvh object")
            node.offset = new_node_offset

        if inplace:
            # reset the spatial coordinates if they were already calculated
            self._has_spatial = False
            self._spatial_coord = np.array([[]])
            return None
        else:
            new_bvh._has_spatial = False
            new_bvh._spatial_coord = np.array([[]])
            return new_bvh

    def scale_skeleton(self, scale, inplace=False):
        """
        Scale the offset of the nodes by a factor.
        Input :
        - nodes : a list of BvhNode objects
        - scale : a float, or a list/np array of 3 floats
        - inplace : if True, the function will modify the nodes in place.
                    if False, the function will return a new Bvh objects
                    with the scaled offset
        Output :
        If inplace is True, the function returns None.
        If inplace is False, the function returns a new Bvh object with the scaled offset

        We check if the scale argument is a float or a list/np array of 3 floats
        """
        if isinstance(scale, (int, float)):
            scale = np.array([scale, scale, scale])
        elif len(scale) == 3:
            scale = np.array(scale)
        else:
            raise ValueError('The scale argument should be a float, or a list/np array of 3 floats')
        
        
        if inplace:

            # reset the spatial coordinates if they were already calculated
            self._has_spatial = False
            self._spatial_coord = np.array([[]])

            for node in self.nodes:
                node.offset = node.offset * scale
            return None
        
        else:
            new_bvh = self.copy()

            # reset the spatial coordinates if they were already calculated
            new_bvh._has_spatial = False
            new_bvh._spatial_coord = np.array([[]])
            for node in new_bvh.nodes:
                node.offset = node.offset * scale
            return new_bvh
        

        

    def single_joint_euler_angle(self, joint_name, new_order, inplace=True):
        """
        Change the Euler angle order of a single joint for all frames.

        Converts the joint's rotation data via rotation matrices so the
        resulting Euler angles use the new order but represent the same
        physical rotations.  Updates frames, frame_template and the node's
        rot_channels atomically.

        Parameters
        ----------
        joint_name : str
            Name of the joint whose Euler order should be changed.
        new_order : str or list of 3 chars
            New rotation order, e.g. 'XYZ' or ['X', 'Y', 'Z'].
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        if isinstance(new_order, str):
            new_order = list(new_order.upper())
        else:
            new_order = [c.upper() for c in new_order]

        # Find the joint node
        joint = None
        for node in self.nodes:
            if not node.is_end_site() and node.name == joint_name:
                joint = node
                break
        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found among non-end-site nodes.")

        old_order = joint.rot_channels

        # If the order is already the same, nothing to do
        if old_order == new_order:
            return None if inplace else self.copy()

        target = self if inplace else self.copy()

        # Find the column index in frames for this joint
        col = 3  # skip root position columns
        target_joint = None
        for node in target.nodes:
            if node.is_end_site():
                continue
            if node.name == joint_name:
                target_joint = node
                break
            col += 3

        # Convert: old Euler → rotmat → new Euler
        angles_old = target.frames[:, col:col+3]  # (num_frames, 3) degrees
        R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
        angles_new = rotations.rotmat_to_euler(R, new_order, degrees=True)

        # Write new angles back
        target.frames[:, col:col+3] = angles_new

        # Update node's rot_channels
        target_joint.rot_channels = new_order

        # Rebuild frame_template to reflect the new channel order
        target._create_frame_template()

        if inplace:
            return None
        return target


    def change_all_euler_orders(self, new_order, inplace=True):
        """
        Change the Euler angle order of ALL joints to a single unified order.

        This is useful for ML pipelines that expect a consistent rotation
        order across all joints.

        Parameters
        ----------
        new_order : str or list of 3 chars
            New rotation order, e.g. 'XYZ' or ['X', 'Y', 'Z'].
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        if isinstance(new_order, str):
            new_order_list = list(new_order.upper())
        else:
            new_order_list = [c.upper() for c in new_order]

        target = self if inplace else self.copy()

        col = 3  # skip root position columns
        for node in target.nodes:
            if node.is_end_site():
                continue

            old_order = node.rot_channels
            if old_order != new_order_list:
                angles_old = target.frames[:, col:col+3]
                R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
                angles_new = rotations.rotmat_to_euler(R, new_order_list, degrees=True)
                target.frames[:, col:col+3] = angles_new
                node.rot_channels = new_order_list
            col += 3

        # Rebuild frame_template once at the end
        target._create_frame_template()

        if inplace:
            return None
        return target



    def get_frames_as_rotmat(self):
        """
        Convert all per-joint Euler angles in self.frames to rotation matrices.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_rotmats : ndarray, shape (num_frames, num_joints, 3, 3)
            Rotation matrix for each joint in each frame.
            Joint order follows self.nodes (end sites excluded).
        joints : list of BvhNode
            Joint  to the second axis of joint_rotmats.
        """
        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = self.frame_count

        root_pos = self.frames[:, :3].copy()

        joint_rotmats = np.empty((num_frames, num_joints, 3, 3), dtype=np.float64)

        col = 3  # skip root position columns
        for j_idx, joint in enumerate(joints):
            angles = self.frames[:, col:col+3]  # (num_frames, 3) in degrees
            order = joint.rot_channels
            joint_rotmats[:, j_idx] = rotations.euler_to_rotmat(angles, order, degrees=True)
            col += 3

        return root_pos, joint_rotmats, joints


    def get_frames_as_6d(self):
        """
        Convert all per-joint Euler angles to 6D rotation representation.

        The 6D representation (Zhou et al., CVPR 2019) is continuous and
        well-suited for neural network training.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_rot6d : ndarray, shape (num_frames, num_joints, 6)
            6D rotation for each joint in each frame.
            Joint order follows self.nodes (end sites excluded).
        joints : list of BvhNode
            Joint corresponding to the second axis of joint_rot6d.
        """
        root_pos, joint_rotmats, joints = self.get_frames_as_rotmat()
        # (num_frames, num_joints, 3, 3) -> (num_frames, num_joints, 6)
        joint_rot6d = rotations.rotmat_to_rot6d(joint_rotmats)
        return root_pos, joint_rot6d, joints


    def get_frames_as_quaternion(self):
        """
        Convert all per-joint Euler angles to quaternions.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_quats : ndarray, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) for each joint in each frame.
            Joint order follows self.nodes (end sites excluded).
        joint : list of BvhNode
            Joint corresponding to the second axis of joint_quats.
        """
        root_pos, joint_rotmats, joints = self.get_frames_as_rotmat()
        # (num_frames, num_joints, 3, 3) -> (num_frames, num_joints, 4)
        joint_quats = rotations.rotmat_to_quat(joint_rotmats)
        return root_pos, joint_quats, joints


    def get_frames_as_axisangle(self):
        """
        Convert all per-joint Euler angles to axis-angle vectors.

        The axis-angle representation is the unit rotation axis scaled
        by the rotation angle in radians.  Used in SMPL/SMPL-X body
        models and many pose estimation pipelines.

        Returns
        -------
        root_pos : ndarray, shape (num_frames, 3)
            Root position for each frame.
        joint_aa : ndarray, shape (num_frames, num_joints, 3)
            Axis-angle vector for each joint in each frame.
            Joint order follows self.nodes (end sites excluded).
        joints : list of BvhNode
            Joint corresponding to the second axis of joint_aa.
        """
        root_pos, joint_rotmats, joints = self.get_frames_as_rotmat()
        # (num_frames, num_joints, 3, 3) -> (num_frames, num_joints, 3)
        joint_aa = rotations.rotmat_to_axisangle(joint_rotmats)
        return root_pos, joint_aa, joints


    def set_frames_from_6d(self, root_pos, joint_rot6d):
        """
        Set self.frames from root positions and 6D rotation data.

        Converts 6D rotations back to Euler angles using each joint's
        rot_channels order, then writes into self.frames.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_rot6d : array_like, shape (num_frames, num_joints, 6)
            6D rotation per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        """
        root_pos = np.asarray(root_pos, dtype=np.float64)
        joint_rot6d = np.asarray(joint_rot6d, dtype=np.float64)

        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos.shape[0]

        if joint_rot6d.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_rot6d, "
                f"got {joint_rot6d.shape[1]}")

        # Convert 6D -> rotation matrices -> Euler angles per joint
        joint_rotmats = rotations.rot6d_to_rotmat(joint_rot6d)
        # joint_rotmats shape: (num_frames, num_joints, 3, 3)

        num_channels = 3 + num_joints * 3  # root_pos(3) + 3 per joint
        new_frames = np.empty((num_frames, num_channels), dtype=np.float64)
        new_frames[:, :3] = root_pos

        col = 3
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            euler = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)
            new_frames[:, col:col+3] = euler
            col += 3

        self.frames = new_frames
        self.frame_count = num_frames


    def set_frames_from_quaternion(self, root_pos, joint_quats):
        """
        Set self.frames from root positions and quaternion data.

        Converts quaternions back to Euler angles using each joint's
        rot_channels order, then writes into self.frames.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_quats : array_like, shape (num_frames, num_joints, 4)
            Quaternion (w, x, y, z) per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        """
        root_pos = np.asarray(root_pos, dtype=np.float64)
        joint_quats = np.asarray(joint_quats, dtype=np.float64)

        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos.shape[0]

        if joint_quats.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_quats, "
                f"got {joint_quats.shape[1]}")

        joint_rotmats = rotations.quat_to_rotmat(joint_quats)

        num_channels = 3 + num_joints * 3
        new_frames = np.empty((num_frames, num_channels), dtype=np.float64)
        new_frames[:, :3] = root_pos

        col = 3
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            euler = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)
            new_frames[:, col:col+3] = euler
            col += 3

        self.frames = new_frames
        self.frame_count = num_frames


    def set_frames_from_axisangle(self, root_pos, joint_aa):
        """
        Set self.frames from root positions and axis-angle data.

        Converts axis-angle vectors back to Euler angles using each joint's
        rot_channels order, then writes into self.frames.

        Parameters
        ----------
        root_pos : array_like, shape (num_frames, 3)
            Root position per frame.
        joint_aa : array_like, shape (num_frames, num_joints, 3)
            Axis-angle vector per joint per frame.
            Joint order must match self.nodes (end sites excluded).
        """
        root_pos = np.asarray(root_pos, dtype=np.float64)
        joint_aa = np.asarray(joint_aa, dtype=np.float64)

        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos.shape[0]

        if joint_aa.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_aa, "
                f"got {joint_aa.shape[1]}")

        joint_rotmats = rotations.axisangle_to_rotmat(joint_aa)

        num_channels = 3 + num_joints * 3
        new_frames = np.empty((num_frames, num_channels), dtype=np.float64)
        new_frames[:, :3] = root_pos

        col = 3
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            euler = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)
            new_frames[:, col:col+3] = euler
            col += 3

        self.frames = new_frames
        self.frame_count = num_frames

        
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


