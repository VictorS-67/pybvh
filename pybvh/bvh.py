from pathlib import Path
import numpy as np
import copy

from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .spatial_coord import frames_to_spatial_coord
from . import rotations

class Bvh:
    """
    Container for BVH motion-capture data.

    The hierarchy information is stored in a list of BvhNode objects,
    one per joint / end-site.

    Motion data is stored as two structured arrays:

    - ``root_pos``:     shape ``(F, 3)``      — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)``   — Euler angles (degrees) per joint per frame
    """
    def __init__(self, nodes=[BvhRoot()], root_pos=None, joint_angles=None,
                 frame_frequency=0):
        self.nodes = nodes
        self.frame_frequency = frame_frequency
        self.root = self.nodes[0]

        # Validate that root position channels are standard XYZ
        if self.root.pos_channels != ['X', 'Y', 'Z']:
            raise ValueError(
                f"Non-standard root position channel order "
                f"{self.root.pos_channels} is not supported. "
                f"Expected ['X', 'Y', 'Z'].")

        # ---------- Determine root_pos / joint_angles ----------
        if root_pos is not None and joint_angles is not None:
            self.root_pos = np.asarray(root_pos, dtype=np.float64)
            self.joint_angles = np.asarray(joint_angles, dtype=np.float64)
        else:
            # Empty object
            self.root_pos = np.empty((0, 3), dtype=np.float64)
            self.joint_angles = np.empty((0, 0, 3), dtype=np.float64)

        # node name → integer index into the spatial-coordinate array
        self._create_name2idx()
        

    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, value):
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("nodes should be a list of BvhNode class/subclasse objects")
        self._nodes = value 

    @property
    def frame_frequency(self):
        return self._frame_frequency
    @frame_frequency.setter
    def frame_frequency(self, value):
        self._frame_frequency = value

    @property
    def frame_count(self):
        """Number of frames (computed from root_pos)."""
        return len(self.root_pos)

    @property
    def root(self):
        return self._root
    @root.setter
    def root(self, value):
        if not isinstance(value, BvhRoot):
            raise ValueError("The first element of nodes should be a BvhRoot object")
        self._root = value

    @property
    def euler_column_names(self):
        """Column names describing root_pos + joint_angles in flat layout order.

        Useful for building DataFrames or inspecting the channel mapping.
        Generated on the fly from the node hierarchy.
        """
        names = []
        root = self.root
        for ax in root.pos_channels:
            names.append(f'{root.name}_{ax}_pos')
        for node in self.nodes:
            if node.is_end_site():
                continue
            for ax in node.rot_channels:
                names.append(f'{node.name}_{ax}_rot')
        return names

            
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

        frames_str = f'array(root_pos={self.root_pos.shape}, joint_angles={self.joint_angles.shape}, dtype={self.root_pos.dtype})'
        
        return f'Bvh(nodes={nodes_str}, frames={frames_str}, frame_frequency={self.frame_frequency:.6f})'
    
    def copy(self):
        return copy.deepcopy(self)

    
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

            for i in range(self.frame_count):
                frame_flat = np.concatenate([self.root_pos[i],
                                             self.joint_angles[i].ravel()])
                f.write(np.array2string(frame_flat,
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

        Returns an ndarray of shape ``(N, 3)`` for a single frame or
        ``(F, N, 3)`` for all frames, where *N* is the total number of
        nodes (joints + end sites).

        Parameters
        ----------
        frame_num : int
            Frame index to return.  ``-1`` (default) returns all frames.
        centered : str
            ``"world"`` – root at actual position.
            ``"skeleton"`` – root at origin for all frames.
            ``"first"`` – first-frame root at origin, then moves normally.
        """
        centered_options = ['skeleton', 'first', 'world']
        if centered not in centered_options:
            raise ValueError(
                f'The value {centered} is not recognized for the centered '
                f'argument. Currently recognized keywords are {centered_options}')

        if frame_num == -1:
            return frames_to_spatial_coord(
                self, root_pos=self.root_pos,
                joint_angles=self.joint_angles, centered=centered)
        elif 0 <= frame_num < self.frame_count:
            return frames_to_spatial_coord(
                self, root_pos=self.root_pos[frame_num],
                joint_angles=self.joint_angles[frame_num], centered=centered)
        else:
            raise ValueError(
                "frame_num needs to be -1 or a positive integer smaller "
                "than the total amount of frames in the bvh file.")

        

    def get_rest_pose(self, mode='coordinates'):
        """
        Return the rest pose of the skeleton (all angles zero, root at origin).

        Parameters
        ----------
        mode : str
            ``'euler'`` – return a tuple ``(root_pos, joint_angles)`` of zeros
            matching the structured shapes.
            ``'coordinates'`` – return spatial coordinates as ``(N, 3)``.
        """
        correct_modes = ['euler', 'coordinates']
        if mode == 'euler':
            return np.zeros(3, dtype=np.float64), np.zeros_like(self.joint_angles[0])
        elif mode == 'coordinates':
            return frames_to_spatial_coord(
                self,
                root_pos=np.zeros(3),
                joint_angles=np.zeros_like(self.joint_angles[0]),
                centered="skeleton")
        else:
            raise ValueError(
                f'The value {mode} is not recognized for the mode argument. '
                f'Currently recognized keywords are {correct_modes}')
        

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
        Return a dict of arrays ready for ``pd.DataFrame(result)``.

        Keys are column names (``'time'``, ``'Hips_X_pos'``, …).
        Values are 1-D numpy arrays of length *frame_count*.
        """
        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_frequency

        root = self.root
        for i, ax in enumerate(root.pos_channels):
            result[f'{root.name}_{ax}_pos'] = self.root_pos[:, i]

        j_idx = 0
        for node in self.nodes:
            if node.is_end_site():
                continue
            for i, ax in enumerate(node.rot_channels):
                result[f'{node.name}_{ax}_rot'] = self.joint_angles[:, j_idx, i]
            j_idx += 1

        return result

    def _get_df_constructor_spatial_coord(self, centered):
        """
        Return a dict of arrays with spatial coordinates
        ready for ``pd.DataFrame(result)``.
        """
        spatial_array = self.get_spatial_coord(centered=centered)  # (F, N, 3)

        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_frequency

        for n_idx, node in enumerate(self.nodes):
            for i, ax in enumerate(['X', 'Y', 'Z']):
                result[f'{node.name}_{ax}'] = spatial_array[:, n_idx, i]

        return result



    
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
    
    def _create_name2idx(self):
        """Build a dict mapping node name → integer index in the nodes list."""
        self.name2idx = {node.name: i for i, node in enumerate(self.nodes)}





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
            return None
        else:
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
            for node in self.nodes:
                node.offset = node.offset * scale
            return None
        
        else:
            new_bvh = self.copy()
            for node in new_bvh.nodes:
                node.offset = node.offset * scale
            return new_bvh
        

        

    def single_joint_euler_angle(self, joint, new_order, inplace=True):
        """
        Change the Euler angle order of a single joint for all frames.

        Converts the joint's rotation data via rotation matrices so the
        resulting Euler angles use the new order but represent the same
        physical rotations.  Updates frames, frame_template and the node's
        rot_channels atomically.

        Parameters
        ----------
        joint : str or BvhNode
            Name of the joint or the BvhNode object itself whose Euler order should be changed.
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
        if isinstance(joint, BvhNode):
            joint_name = joint.name
        elif isinstance(joint, str):
            joint_name = joint
        else:
            raise ValueError("joint should be a string (joint name) or a BvhNode object")

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

        # Find the joint index in joint_angles
        j_idx = 0
        target_joint = None
        for node in target.nodes:
            if node.is_end_site():
                continue
            if node.name == joint_name:
                target_joint = node
                break
            j_idx += 1

        # Convert: old Euler → rotmat → new Euler
        angles_old = target.joint_angles[:, j_idx]  # (num_frames, 3) degrees
        R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
        angles_new = rotations.rotmat_to_euler(R, new_order, degrees=True)

        # Write new angles back
        target.joint_angles[:, j_idx] = angles_new

        # Update node's rot_channels
        target_joint.rot_channels = new_order

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

        j_idx = 0
        for node in target.nodes:
            if node.is_end_site():
                continue

            old_order = node.rot_channels
            if old_order != new_order_list:
                angles_old = target.joint_angles[:, j_idx]
                R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
                angles_new = rotations.rotmat_to_euler(R, new_order_list, degrees=True)
                target.joint_angles[:, j_idx] = angles_new
                node.rot_channels = new_order_list
            j_idx += 1

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
            Joint corresponding to the second axis of joint_rotmats.
        """
        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = self.frame_count

        root_pos = self.root_pos.copy()

        joint_rotmats = np.empty((num_frames, num_joints, 3, 3), dtype=np.float64)

        for j_idx, joint in enumerate(joints):
            angles = self.joint_angles[:, j_idx]  # (num_frames, 3) in degrees
            order = joint.rot_channels
            joint_rotmats[:, j_idx] = rotations.euler_to_rotmat(angles, order, degrees=True)

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
        joints : list of BvhNode
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
        Set motion data from root positions and 6D rotation data.

        Converts 6D rotations back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

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

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        self.root_pos = root_pos
        self.joint_angles = new_angles


    def set_frames_from_quaternion(self, root_pos, joint_quats):
        """
        Set motion data from root positions and quaternion data.

        Converts quaternions back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

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

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        self.root_pos = root_pos
        self.joint_angles = new_angles


    def set_frames_from_axisangle(self, root_pos, joint_aa):
        """
        Set motion data from root positions and axis-angle data.

        Converts axis-angle vectors back to Euler angles using each joint's
        rot_channels order, then writes into root_pos and joint_angles.

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

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        self.root_pos = root_pos
        self.joint_angles = new_angles

        
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


