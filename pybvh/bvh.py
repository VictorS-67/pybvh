from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal, Sequence, Union, overload

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .spatial_coord import frames_to_spatial_coord
from . import rotations
from .tools import get_forw_up_axis, get_up_axis_index

class Bvh:
    """Container for BVH motion-capture data.

    The hierarchy is stored as a list of ``BvhNode`` objects (one per
    joint / end-site).  Motion data is stored as two structured arrays:

    - ``root_pos``:     shape ``(F, 3)``    — root translation per frame
    - ``joint_angles``: shape ``(F, J, 3)`` — Euler angles (degrees) per joint per frame

    Attributes
    ----------
    nodes : list of BvhNode
        Skeleton hierarchy in topological order.
    root : BvhRoot
        The root node (``nodes[0]``).
    root_pos : ndarray, shape (F, 3)
        Root position per frame.
    joint_angles : ndarray, shape (F, J, 3)
        Euler angles in degrees per joint per frame.
    frame_frequency : float
        Duration of one frame in seconds.
    frame_count : int
        Number of frames (read-only).
    node_index : dict
        Mapping from node name to its index in ``nodes``.
    joint_names : list of str
        Names of non-end-site joints in topological order (read-only).
    joint_count : int
        Number of non-end-site joints (read-only).
    """
    def __init__(
        self,
        nodes: list[BvhNode] = [BvhRoot()],
        root_pos: npt.ArrayLike | None = None,
        joint_angles: npt.ArrayLike | None = None,
        frame_frequency: float = 0,
    ) -> None:
        self.nodes = nodes
        self.frame_frequency = frame_frequency
        self.root = self.nodes[0]  # type: ignore[assignment]

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
        self._create_node_index()

        # Freeze channel attributes on all nodes to prevent
        # desynchronization with joint_angles
        for node in self.nodes:
            if hasattr(node, '_frozen'):
                node._frozen = True
        

    @property
    def nodes(self) -> list[BvhNode]:
        return self._nodes
    @nodes.setter
    def nodes(self, value: list[BvhNode]) -> None:
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("nodes should be a list of BvhNode class/subclasse objects")
        self._nodes = value 

    @property
    def frame_frequency(self) -> float:
        return self._frame_frequency
    @frame_frequency.setter
    def frame_frequency(self, value: float) -> None:
        self._frame_frequency = value

    @property
    def frame_count(self) -> int:
        """Number of frames (computed from root_pos)."""
        return len(self.root_pos)

    @property
    def root(self) -> BvhRoot:
        return self._root
    @root.setter
    def root(self, value: BvhRoot) -> None:
        if not isinstance(value, BvhRoot):
            raise ValueError("The first element of nodes should be a BvhRoot object")
        self._root = value

    @property
    def euler_column_names(self) -> list[str]:
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
            for ax in node.rot_channels:  # type: ignore[attr-defined]
                names.append(f'{node.name}_{ax}_rot')
        return names

            
    def __str__(self) -> str:
        count_joints =  0
        for node in self.nodes:
            if not node.is_end_site() : count_joints += 1 
        return f'{count_joints} elements in the Hierarchy, {self.frame_count} frames at a frequency of {self.frame_frequency:.6f}Hz'
        
    def __repr__(self) -> str:
        nodes_str = []
        for node in self.nodes:
            sep = node.__str__().split()
            if sep[0] == 'ROOT':
                nodes_str += [node.__str__()]
            elif sep[0] == 'JOINT':
                nodes_str += [sep[1]]
        nodes_repr = ''.join(str(nodes_str).split("'"))

        frames_str = f'array(root_pos={self.root_pos.shape}, joint_angles={self.joint_angles.shape}, dtype={self.root_pos.dtype})'

        return f'Bvh(nodes={nodes_repr}, frames={frames_str}, frame_frequency={self.frame_frequency:.6f})'
    
    def copy(self) -> Bvh:
        return copy.deepcopy(self)

    
    def to_bvh_file(self, new_filepath: str | Path, verbose: bool = True) -> None:
        """Write the Bvh object to a ``.bvh`` file.

        Parameters
        ----------
        new_filepath : str or Path
            Destination file path.  Must have a ``.bvh`` extension.
        verbose : bool, optional
            If True (default), print a confirmation message on success.

        Raises
        ------
        Exception
            If the file extension is not ``.bvh`` or the parent directory
            does not exist.
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


    def get_spatial_coord(self, frame_num: int = -1, centered: str = "world") -> npt.NDArray[np.float64]:
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

        

    def get_rest_pose(self, mode: str = 'coordinates') -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        

    def get_df_constructor(self, mode: str = 'euler', centered: str = "world") -> dict[str, npt.NDArray[np.float64]]:
        """Return a dict of arrays for ``pd.DataFrame(result)``.

        Each key is a column name, each value a 1-D NumPy array of
        length ``frame_count``.

        Parameters
        ----------
        mode : str, optional
            ``'euler'`` — columns are ``'JointName_X_rot'`` etc. (default).
            ``'coordinates'`` — columns are ``'JointName_X'`` etc.,
            including end sites.
        centered : str, optional
            ``"world"`` (default), ``"skeleton"``, or ``"first"``.
            Only used when ``mode='coordinates'``.

        Returns
        -------
        dict
            Column-name → 1-D array mapping, ready for ``pd.DataFrame()``.
        """
        correct_modes = ['euler', 'coordinates']

        if mode == 'euler':
            return self._get_df_constructor_euler_angles()
        elif mode == 'coordinates':
            return self._get_df_constructor_spatial_coord(centered=centered)
        else : 
            raise ValueError(f'The value {mode} is not recognized for the mode argument.\
                             Currently recognized keywords are {correct_modes}')
        
    
    def _get_df_constructor_euler_angles(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for Euler-angle DataFrame."""
        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_frequency

        root = self.root
        for i, ax in enumerate(root.pos_channels):
            result[f'{root.name}_{ax}_pos'] = self.root_pos[:, i]

        j_idx = 0
        for node in self.nodes:
            if node.is_end_site():
                continue
            for i, ax in enumerate(node.rot_channels):  # type: ignore[attr-defined]
                result[f'{node.name}_{ax}_rot'] = self.joint_angles[:, j_idx, i]
            j_idx += 1

        return result

    def _get_df_constructor_spatial_coord(self, centered: str) -> dict[str, npt.NDArray[np.float64]]:
        """Return column-name → array dict for spatial-coordinate DataFrame."""
        spatial_array = self.get_spatial_coord(centered=centered)  # (F, N, 3)

        result = {}
        result['time'] = np.arange(self.frame_count) * self.frame_frequency

        for n_idx, node in enumerate(self.nodes):
            for i, ax in enumerate(['X', 'Y', 'Z']):
                result[f'{node.name}_{ax}'] = spatial_array[:, n_idx, i]

        return result



    
    def hierarchy_info_as_dict(self) -> dict:
        """Return the skeleton hierarchy as a plain dictionary.

        Returns
        -------
        dict
            ``{name: {'offset': [...], 'parent': str|None,
            'rot_channels': [...], 'children': [...]}, ...}``.
            Root entries also include ``'pos_channels'``.
            The returned dict is a deep copy (safe to mutate).
        """
        hier_dict: dict[str, dict[str, object]] = {}
        for node in self.nodes:
            hier_dict[node.name] = {'offset' : node.offset}
            if isinstance(node, BvhRoot):
                hier_dict[node.name]['pos_channels'] = node.pos_channels
            if isinstance(node, BvhJoint):
                hier_dict[node.name]['rot_channels'] = node.rot_channels
                hier_dict[node.name]['children'] = [child.name for child in node.children]
            hier_dict[node.name]['parent'] = None if node.parent is None else node.parent.name
            
        return copy.deepcopy(hier_dict)
    
    def _create_node_index(self) -> None:
        """Build ``node_index`` mapping node name to its index in ``nodes``."""
        self._node_index = {node.name: i for i, node in enumerate(self.nodes)}

    @property
    def node_index(self) -> dict[str, int]:
        """Mapping from node name to its integer index in ``nodes``.

        Returns
        -------
        dict
            ``{node_name: int}`` for every node (joints and end sites).
        """
        return self._node_index

    @property
    def joint_names(self) -> list[str]:
        """Names of non-end-site joints in topological order.

        Returns
        -------
        list of str
        """
        return [n.name for n in self.nodes if not n.is_end_site()]

    @property
    def joint_count(self) -> int:
        """Number of non-end-site joints.

        Returns
        -------
        int
        """
        return len(self.joint_names)





    @overload
    def change_skeleton(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = ..., strict: bool = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def change_skeleton(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = ..., strict: bool = ..., inplace: Literal[False] = ...) -> Bvh: ...
    def change_skeleton(self, new_skeleton: Bvh, name_mapping: dict[str, str] | None = None,
                        strict: bool = False, inplace: bool = False) -> Bvh | None:
        """Copy joint offsets from a reference skeleton.

        Parameters
        ----------
        new_skeleton : Bvh
            Reference skeleton whose offsets will be copied.
        name_mapping : dict, optional
            Maps self's joint names to ``new_skeleton``'s joint names,
            e.g. ``{'Hips': 'mixamorig:Hips', ...}``.
            Joints not in the mapping are matched by identical name.
            If None (default), all joints are matched by name.
        strict : bool, optional
            If True, raise ``ValueError`` when a joint in self has no
            match in ``new_skeleton``.  If False (default), unmapped
            joints keep their original offsets.
        inplace : bool, optional
            If True, modify self and return None.
            If False (default), return a modified copy.

        Returns
        -------
        None or Bvh
        """
        try:
            new_skel_nodes = new_skeleton.nodes
        except AttributeError:
            raise ValueError('new_skeleton must be a Bvh object')

        # Build name → index lookup for the reference skeleton
        newnodes2idx = {n.name: i for i, n in enumerate(new_skel_nodes)}

        if inplace:
            nodes = self.nodes
        else:
            new_bvh = self.copy()
            nodes = new_bvh.nodes

        for node in nodes:
            # Determine the target name in new_skeleton
            if name_mapping and node.name in name_mapping:
                target_name = name_mapping[node.name]
            else:
                target_name = node.name

            if target_name in newnodes2idx:
                node.offset = new_skel_nodes[newnodes2idx[target_name]].offset
            elif strict:
                raise ValueError(
                    f"Node '{node.name}' (mapped to '{target_name}') not found "
                    f"in new_skeleton and strict=True.")
            # else: keep original offset (lenient mode)

        if inplace:
            return None
        return new_bvh

    @overload
    def scale_skeleton(self, scale: float | npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def scale_skeleton(self, scale: float | npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def scale_skeleton(self, scale: float | npt.ArrayLike, inplace: bool = False) -> Bvh | None:
        """Scale all node offsets by a factor.

        Parameters
        ----------
        scale : float or array_like of shape (3,)
            Uniform scalar or per-axis scale factors.
        inplace : bool, optional
            If True, modify self and return None.
            If False (default), return a modified copy.

        Returns
        -------
        None or Bvh
        """
        if isinstance(scale, (int, float)):
            scale_arr: npt.NDArray[np.float64] = np.array([scale, scale, scale], dtype=np.float64)
        else:
            scale_arr = np.asarray(scale, dtype=np.float64)
            if scale_arr.shape != (3,):
                raise ValueError('The scale argument should be a float, or a list/np array of 3 floats')


        if inplace:
            for node in self.nodes:
                node.offset = node.offset * scale_arr
            return None

        else:
            new_bvh = self.copy()
            for node in new_bvh.nodes:
                node.offset = node.offset * scale_arr
            return new_bvh
        

        

    @overload
    def single_joint_euler_angle(self, joint: str | BvhNode, new_order: Union[str, Sequence[str]], *, inplace: Literal[True]) -> None: ...
    @overload
    def single_joint_euler_angle(self, joint: str | BvhNode, new_order: Union[str, Sequence[str]], inplace: Literal[False] = ...) -> Bvh: ...
    def single_joint_euler_angle(self, joint: str | BvhNode, new_order: Union[str, Sequence[str]], inplace: bool = False) -> Bvh | None:
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

        found_joint: BvhNode | None = None
        for node in self.nodes:
            if not node.is_end_site() and node.name == joint_name:
                found_joint = node
                break
        if found_joint is None:
            raise ValueError(f"Joint '{joint_name}' not found among non-end-site nodes.")

        old_order = found_joint.rot_channels  # type: ignore[attr-defined]

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

        # Update node's rot_channels (bypass freeze check)
        target_joint._set_rot_channels_internal(new_order)  # type: ignore[union-attr]

        if inplace:
            return None
        return target


    @overload
    def change_all_euler_orders(self, new_order: Union[str, Sequence[str]], *, inplace: Literal[True]) -> None: ...
    @overload
    def change_all_euler_orders(self, new_order: Union[str, Sequence[str]], inplace: Literal[False] = ...) -> Bvh: ...
    def change_all_euler_orders(self, new_order: Union[str, Sequence[str]], inplace: bool = False) -> Bvh | None:
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

            old_order = node.rot_channels  # type: ignore[attr-defined]
            if old_order != new_order_list:
                angles_old = target.joint_angles[:, j_idx]
                R = rotations.euler_to_rotmat(angles_old, old_order, degrees=True)
                angles_new = rotations.rotmat_to_euler(R, new_order_list, degrees=True)
                target.joint_angles[:, j_idx] = angles_new
                node._set_rot_channels_internal(new_order_list)  # type: ignore[attr-defined]
            j_idx += 1

        if inplace:
            return None
        return target



    def get_frames_as_rotmat(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[BvhJoint]]:
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
            order = joint.rot_channels  # type: ignore[attr-defined]
            joint_rotmats[:, j_idx] = rotations.euler_to_rotmat(angles, order, degrees=True)

        return root_pos, joint_rotmats, joints  # type: ignore[return-value]


    def get_frames_as_6d(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[BvhJoint]]:
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


    def get_frames_as_quaternion(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[BvhJoint]]:
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


    def get_frames_as_axisangle(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[BvhJoint]]:
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


    @overload
    def set_frames_from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_frames_from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def set_frames_from_6d(self, root_pos: npt.ArrayLike, joint_rot6d: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_rot6d_arr: npt.NDArray[np.float64] = np.asarray(joint_rot6d, dtype=np.float64)

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_rot6d_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_rot6d, "
                f"got {joint_rot6d_arr.shape[1]}")

        # Convert 6D -> rotation matrices -> Euler angles per joint
        joint_rotmats = rotations.rot6d_to_rotmat(joint_rot6d_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

        if inplace:
            return None
        return target


    @overload
    def set_frames_from_quaternion(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_frames_from_quaternion(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def set_frames_from_quaternion(self, root_pos: npt.ArrayLike, joint_quats: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_quats_arr: npt.NDArray[np.float64] = np.asarray(joint_quats, dtype=np.float64)

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_quats_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_quats, "
                f"got {joint_quats_arr.shape[1]}")

        joint_rotmats = rotations.quat_to_rotmat(joint_quats_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

        if inplace:
            return None
        return target


    @overload
    def set_frames_from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_frames_from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: Literal[False] = ...) -> Bvh: ...
    def set_frames_from_axisangle(self, root_pos: npt.ArrayLike, joint_aa: npt.ArrayLike, inplace: bool = False) -> Bvh | None:
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
        inplace : bool
            If True, modify self and return None.
            If False, return a modified copy while leaving self unchanged.

        Returns
        -------
        None or Bvh
            None if inplace, otherwise a new Bvh object.
        """
        target = self if inplace else self.copy()

        root_pos_arr: npt.NDArray[np.float64] = np.asarray(root_pos, dtype=np.float64)
        joint_aa_arr: npt.NDArray[np.float64] = np.asarray(joint_aa, dtype=np.float64)

        joints = [n for n in target.nodes if not n.is_end_site()]
        num_joints = len(joints)
        num_frames = root_pos_arr.shape[0]

        if joint_aa_arr.shape[1] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints in joint_aa, "
                f"got {joint_aa_arr.shape[1]}")

        joint_rotmats = rotations.axisangle_to_rotmat(joint_aa_arr)

        new_angles = np.empty((num_frames, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                joint_rotmats[:, j_idx], order, degrees=True)

        target.root_pos = root_pos_arr
        target.joint_angles = new_angles

        if inplace:
            return None
        return target


    # ----------------------------------------------------------------
    # Frame slicing, concatenation, and resampling
    # ----------------------------------------------------------------

    def slice_frames(self, start: int | None = None, end: int | None = None, step: int | None = None) -> Bvh:
        """Return a new Bvh with a slice of frames.

        Parameters
        ----------
        start, end, step : int or None
            Slice parameters (same semantics as ``array[start:end:step]``).

        Returns
        -------
        Bvh
            New Bvh object with the sliced frames and same skeleton.
        """
        new_bvh = self.copy()
        s = slice(start, end, step)
        new_bvh.root_pos = self.root_pos[s].copy()
        new_bvh.joint_angles = self.joint_angles[s].copy()
        # Adjust frame frequency if step changes the sampling rate
        if step is not None and abs(step) != 1:
            new_bvh.frame_frequency = self.frame_frequency * abs(step)
        return new_bvh

    def concat(self, other: Bvh) -> Bvh:
        """Concatenate frames from another Bvh with the same skeleton.

        Parameters
        ----------
        other : Bvh
            Must have the same skeleton (same node names and rotation
            orders).

        Returns
        -------
        Bvh
            New Bvh with frames from ``self`` followed by ``other``.

        Raises
        ------
        ValueError
            If skeletons are incompatible (different node count, names,
            or rotation orders).
        """
        if len(self.nodes) != len(other.nodes):
            raise ValueError(
                f"Node count mismatch: {len(self.nodes)} vs {len(other.nodes)}")
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1.name != n2.name:
                raise ValueError(
                    f"Node name mismatch: '{n1.name}' vs '{n2.name}'")
            if not n1.is_end_site() and not n2.is_end_site():
                if n1.rot_channels != n2.rot_channels:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"Rotation order mismatch for '{n1.name}': "
                        f"{n1.rot_channels} vs {n2.rot_channels}")  # type: ignore[attr-defined]

        if self.frame_frequency != other.frame_frequency:
            import warnings
            warnings.warn(
                f"Frame frequency mismatch: {self.frame_frequency} vs "
                f"{other.frame_frequency}. Using self's frequency.")

        new_bvh = self.copy()
        new_bvh.root_pos = np.concatenate(
            [self.root_pos, other.root_pos], axis=0)
        new_bvh.joint_angles = np.concatenate(
            [self.joint_angles, other.joint_angles], axis=0)
        return new_bvh

    def resample(self, target_fps: float) -> Bvh:
        """Resample frames to a new frame rate via interpolation.

        Root position is linearly interpolated.  Joint rotations are
        converted to quaternions and interpolated with SLERP for
        smooth, gimbal-lock-free results.

        Parameters
        ----------
        target_fps : float
            Target frames per second.

        Returns
        -------
        Bvh
            New Bvh with resampled frames.
        """
        if self.frame_count < 2:
            return self.copy()

        # Original and target timestamps
        t_orig = np.arange(self.frame_count) * self.frame_frequency
        new_freq = 1.0 / target_fps
        t_new = np.arange(0, t_orig[-1] + 1e-12, new_freq)
        # Clip to avoid floating-point overshoot
        t_new = t_new[t_new <= t_orig[-1] + 1e-12]

        num_new = len(t_new)
        joints = [n for n in self.nodes if not n.is_end_site()]
        num_joints = len(joints)

        # --- Root position: linear interpolation ---
        new_root_pos = np.empty((num_new, 3), dtype=np.float64)
        for ax in range(3):
            new_root_pos[:, ax] = np.interp(t_new, t_orig, self.root_pos[:, ax])

        # --- Joint angles: quaternion SLERP ---
        # Convert all joints to quaternions: (F, J, 4)
        _, joint_quats, _ = self.get_frames_as_quaternion()

        # Find surrounding frame indices for each new timestamp
        idx_right = np.searchsorted(t_orig, t_new, side='right')
        idx_right = np.clip(idx_right, 1, self.frame_count - 1)
        idx_left = idx_right - 1

        # Compute interpolation parameter t in [0, 1]
        t_left = t_orig[idx_left]
        t_right = t_orig[idx_right]
        dt = t_right - t_left
        # Avoid division by zero for duplicate timestamps
        dt = np.where(dt < 1e-15, 1.0, dt)
        alpha = (t_new - t_left) / dt  # (num_new,)

        # SLERP for all joints at once: shape (num_new, J, 4)
        q_left = joint_quats[idx_left]    # (num_new, J, 4)
        q_right = joint_quats[idx_right]  # (num_new, J, 4)

        # Broadcast alpha to (num_new, J) for per-joint SLERP
        alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (num_new, num_joints))
        new_quats = rotations.quat_slerp(q_left, q_right, alpha_jt)

        # Convert back to Euler angles per joint
        new_angles = np.empty((num_new, num_joints, 3), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            order = joint.rot_channels  # type: ignore[attr-defined]
            new_angles[:, j_idx] = rotations.rotmat_to_euler(
                rotations.quat_to_rotmat(new_quats[:, j_idx]),
                order, degrees=True)

        new_bvh = self.copy()
        new_bvh.root_pos = new_root_pos
        new_bvh.joint_angles = new_angles
        new_bvh.frame_frequency = new_freq
        return new_bvh

    # ----------------------------------------------------------------
    # Joint subsetting
    # ----------------------------------------------------------------

    def extract_joints(self, joint_names: list[str]) -> Bvh:
        """Extract a subset of joints into a new Bvh.

        Removed joints' offsets are collapsed into their nearest kept
        descendant via vector addition (valid at rest pose).  Their
        rotation contribution during animation is lost.

        Parameters
        ----------
        joint_names : list of str
            Names of joints to keep.  The root must be included.
            End sites are handled automatically (kept if their parent
            is kept, otherwise removed).

        Returns
        -------
        Bvh
            New Bvh with the reduced skeleton and corresponding motion
            data.

        Raises
        ------
        ValueError
            If the root joint is not in ``joint_names``.
        """
        keep_set = set(joint_names)

        if self.root.name not in keep_set:
            raise ValueError(
                f"Root joint '{self.root.name}' must be in joint_names.")

        # --- Build old joint index for each non-end-site node ---
        old_j_idx = {}
        j = 0
        for node in self.nodes:
            if not node.is_end_site():
                old_j_idx[node.name] = j
                j += 1

        # --- For each kept joint, find nearest kept ancestor and
        #     accumulated offset (sum of intermediate offsets) ---
        # Also collect which old joint indices to keep.
        new_nodes = []
        kept_old_j_indices = []
        # Map old node name → new node object (for parent/children wiring)
        new_node_map: dict[str, BvhNode] = {}

        for node in self.nodes:
            if node.is_end_site():
                # Keep end site only if its parent is kept
                if node.parent is not None and node.parent.name in keep_set:
                    # Walk up from this end site accumulating offset
                    # (in case there were removed intermediates — though
                    # end sites are always direct children, just be safe)
                    acc_offset = node.offset.copy()
                    new_end = BvhNode(
                        node.name, offset=acc_offset,
                        parent=new_node_map[node.parent.name])
                    new_nodes.append(new_end)
                    new_node_map[node.name] = new_end
                continue

            if node.name not in keep_set:
                continue

            # This is a kept joint. Find its nearest kept ancestor.
            acc_offset = node.offset.copy()
            walker = node.parent
            while walker is not None and walker.name not in keep_set:
                acc_offset = walker.offset + acc_offset
                walker = walker.parent

            if walker is None:
                # This is the root (no parent)
                if isinstance(node, BvhRoot):
                    new_node = BvhRoot(
                        node.name, offset=acc_offset,
                        pos_channels=list(node.pos_channels),
                        rot_channels=list(node.rot_channels),
                        children=[])
                else:
                    raise ValueError(
                        f"Joint '{node.name}' has no kept ancestor and "
                        f"is not the root.")
            else:
                new_parent = new_node_map[walker.name]
                new_node = BvhJoint(  # type: ignore[assignment]
                    node.name, offset=acc_offset,
                    rot_channels=list(node.rot_channels),  # type: ignore[attr-defined]
                    children=[], parent=new_parent)
                new_parent.children = new_parent.children + [new_node]  # type: ignore[attr-defined]

            new_nodes.append(new_node)
            new_node_map[node.name] = new_node
            kept_old_j_indices.append(old_j_idx[node.name])

        # Wire end-site children into their parents
        for node in new_nodes:
            if node.is_end_site() and node.parent is not None:
                parent = node.parent
                if node not in parent.children:  # type: ignore[attr-defined]
                    parent.children = parent.children + [node]  # type: ignore[attr-defined]

        # If a kept joint has no children at all, create an end site
        # using the offset to its nearest original end-site descendant.
        for node in new_nodes:
            if not node.is_end_site() and not node.children:  # type: ignore[attr-defined]
                # Find original end-site descendant
                orig_node = None
                for n in self.nodes:
                    if n.name == node.name:
                        orig_node = n
                        break
                end_offset = self._find_end_site_offset(orig_node)  # type: ignore[arg-type]
                end_site = BvhNode(
                    f'End Site {node.name}', offset=end_offset, parent=node)
                node.children = [end_site]  # type: ignore[attr-defined]
                new_nodes.append(end_site)

        # --- Build new joint_angles by selecting kept columns ---
        new_joint_angles = self.joint_angles[:, kept_old_j_indices, :]

        return Bvh(
            nodes=new_nodes,
            root_pos=self.root_pos.copy(),
            joint_angles=new_joint_angles.copy(),
            frame_frequency=self.frame_frequency)

    def _find_end_site_offset(self, node: BvhNode) -> npt.NDArray[np.float64]:
        """Find accumulated offset to the nearest end-site descendant."""
        # BFS through children
        queue = [(child, child.offset.copy()) for child in node.children]  # type: ignore[attr-defined]
        while queue:
            child, acc = queue.pop(0)
            if child.is_end_site():
                return acc
            for grandchild in child.children:
                queue.append((grandchild, acc + grandchild.offset))
        # Fallback: zero offset
        return np.zeros(3, dtype=np.float64)


    # =========================================================================
    #  ML Pipeline Features
    # =========================================================================

    def get_joint_velocities(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint position velocities via finite differences.

        Parameters
        ----------
        centered : str, optional
            Coordinate centering mode (default ``"world"``).
            Ignored if *coords* is provided.
        in_frames : bool, optional
            If True, return velocity in units/frame.
            If False (default), return velocity in units/second.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates. If None, computed
            internally via :meth:`get_spatial_coord`.

        Returns
        -------
        ndarray, shape (F-1, N, 3)
            Velocity of each node between consecutive frames.

        Raises
        ------
        ValueError
            If fewer than 2 frames, or ``frame_frequency == 0``
            when ``in_frames=False``.
        """
        if self.frame_count < 2:
            raise ValueError(
                "At least 2 frames are required to compute velocities.")
        if not in_frames and self.frame_frequency == 0:
            raise ValueError(
                "frame_frequency is 0; cannot compute per-second velocity. "
                "Use in_frames=True for per-frame velocity.")

        if coords is None:
            coords = self.get_spatial_coord(centered=centered)

        vel = coords[1:] - coords[:-1]  # (F-1, N, 3)

        if not in_frames:
            vel = vel / self.frame_frequency

        return vel

    def get_joint_accelerations(
        self,
        centered: str = "world",
        in_frames: bool = False,
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint position accelerations via second-order finite differences.

        Parameters
        ----------
        centered : str, optional
            Coordinate centering mode (default ``"world"``).
            Ignored if *coords* is provided.
        in_frames : bool, optional
            If True, return acceleration in units/frame^2.
            If False (default), return in units/second^2.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates. If None, computed
            internally via :meth:`get_spatial_coord`.

        Returns
        -------
        ndarray, shape (F-2, N, 3)
            Acceleration of each node.

        Raises
        ------
        ValueError
            If fewer than 3 frames, or ``frame_frequency == 0``
            when ``in_frames=False``.
        """
        if self.frame_count < 3:
            raise ValueError(
                "At least 3 frames are required to compute accelerations.")
        if not in_frames and self.frame_frequency == 0:
            raise ValueError(
                "frame_frequency is 0; cannot compute per-second acceleration. "
                "Use in_frames=True for per-frame acceleration.")

        vel = self.get_joint_velocities(
            centered=centered, in_frames=True, coords=coords)
        acc = vel[1:] - vel[:-1]  # (F-2, N, 3)

        if not in_frames:
            acc = acc / (self.frame_frequency ** 2)

        return acc

    def get_angular_velocities(
        self,
        in_frames: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Compute per-joint angular velocities via rotation matrix log map.

        For each consecutive frame pair, computes the relative rotation
        ``R_rel = R_t^T @ R_{t+1}`` and converts to axis-angle, giving
        an angular velocity vector whose norm is the rotation angle.

        Parameters
        ----------
        in_frames : bool, optional
            If True, return angular velocity in radians/frame.
            If False (default), return in radians/second.

        Returns
        -------
        ndarray, shape (F-1, J, 3)
            Angular velocity vector per joint per frame transition.
            The direction is the rotation axis; the magnitude is the
            rotation angle (in radians or radians/second).

        Raises
        ------
        ValueError
            If fewer than 2 frames, or ``frame_frequency == 0``
            when ``in_frames=False``.
        """
        if self.frame_count < 2:
            raise ValueError(
                "At least 2 frames are required to compute angular velocities.")
        if not in_frames and self.frame_frequency == 0:
            raise ValueError(
                "frame_frequency is 0; cannot compute per-second angular velocity. "
                "Use in_frames=True for per-frame angular velocity.")

        _, joint_rotmats, _ = self.get_frames_as_rotmat()  # (F, J, 3, 3)

        # R_rel = R_t^T @ R_{t+1}  for each consecutive pair
        R_t = joint_rotmats[:-1]      # (F-1, J, 3, 3)
        R_t1 = joint_rotmats[1:]      # (F-1, J, 3, 3)
        R_rel = np.einsum('...ji,...jk->...ik', R_t, R_t1)  # transpose + matmul

        ang_vel = rotations.rotmat_to_axisangle(R_rel)  # (F-1, J, 3)

        if not in_frames:
            ang_vel = ang_vel / self.frame_frequency

        return ang_vel

    def get_root_relative_positions(
        self,
        centered: str = "world",
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute joint positions relative to the root at each frame.

        Unlike ``centered="skeleton"`` which places the root at the
        world origin for all frames, this subtracts the root's position
        per frame, preserving the relative pose of the skeleton.

        Parameters
        ----------
        centered : str, optional
            Coordinate centering mode (default ``"world"``).
            Ignored if *coords* is provided.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates.

        Returns
        -------
        ndarray, shape (F, N, 3)
            Positions of all nodes relative to the root each frame.
            The root node (index 0) will be ``(0, 0, 0)`` in every frame.
        """
        if coords is None:
            coords = self.get_spatial_coord(centered=centered)

        return coords - coords[:, 0:1, :]  # broadcast root position

    def get_root_trajectory(
        self,
        up_axis: str | None = None,
    ) -> npt.NDArray[np.float64]:
        """Extract root trajectory features commonly used in motion ML.

        Returns the root's ground-plane position, heading angle, and
        their velocities — the standard root representation in
        HumanML3D-style pipelines.

        Parameters
        ----------
        up_axis : str or None, optional
            Signed axis string (e.g. ``'+y'``, ``'+z'``). If None,
            auto-detected from the rest pose.

        Returns
        -------
        ndarray, shape (F, 4)
            Columns: ``[ground_pos_a, ground_pos_b, heading_sin, heading_cos]``
            where a and b are the two ground-plane axes (non-up axes
            in order x, y, z with up removed).

        Notes
        -----
        The heading angle is extracted from the root joint's rotation
        matrix projected onto the ground plane. The heading is defined
        as the rotation around the up axis.
        """
        # Determine up axis
        if up_axis is None:
            rest_pose: npt.NDArray[np.float64] = self.get_rest_pose(mode='coordinates')  # type: ignore[assignment]
            up_idx = get_up_axis_index(self, rest_pose)
        else:
            up_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis[1]]

        # Ground-plane axes (the two non-up axes, in order)
        ground_axes = [i for i in range(3) if i != up_idx]

        # Root ground-plane position: (F, 2)
        ground_pos = self.root_pos[:, ground_axes]

        # Root heading: extract from root rotation matrix
        root_joint = self.nodes[0]
        root_angles = self.joint_angles[:, 0]  # (F, 3) degrees
        root_order = root_joint.rot_channels  # type: ignore[attr-defined]
        R_root = rotations.euler_to_rotmat(root_angles, root_order, degrees=True)  # (F, 3, 3)

        # Heading = rotation around up axis
        # Project the forward direction through the rotation matrix
        # Use the column of R corresponding to the first ground axis
        # The heading angle is atan2 of the projected forward direction
        fwd_idx = ground_axes[0]
        # Forward vector after rotation: R @ e_fwd, projected onto ground plane
        fwd_rotated_a = R_root[:, ground_axes[0], fwd_idx]  # component along first ground axis
        fwd_rotated_b = R_root[:, ground_axes[1], fwd_idx]  # component along second ground axis
        heading = np.arctan2(fwd_rotated_b, fwd_rotated_a)  # (F,)

        return np.column_stack([
            ground_pos,           # (F, 2)
            np.sin(heading),      # (F,)
            np.cos(heading),      # (F,)
        ])

    def get_foot_contacts(
        self,
        foot_joints: list[str] | None = None,
        method: str = "velocity",
        threshold: float | None = None,
        centered: str = "world",
        coords: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Detect binary foot contact labels per frame.

        Parameters
        ----------
        foot_joints : list of str or None, optional
            Joint names to use as foot markers. If None,
            auto-detects by searching for joints with "foot" or
            "toe" in the name (case-insensitive).
        method : str, optional
            Detection method: ``"velocity"`` (default) thresholds
            the speed of the foot joints, ``"height"`` thresholds
            the foot joint height above the ground plane.
        threshold : float or None, optional
            Detection threshold. If None, defaults to ``0.5``
            for ``"velocity"`` (units/frame) or auto-calibrated
            for ``"height"`` (5th percentile of foot heights).
        centered : str, optional
            Coordinate centering mode (default ``"world"``).
            Ignored if *coords* is provided.
        coords : ndarray, shape (F, N, 3), optional
            Pre-computed spatial coordinates.

        Returns
        -------
        ndarray, shape (F, num_foot_joints)
            Binary contact labels (1.0 = contact, 0.0 = no contact).
            For ``"velocity"`` method, the first frame is always 1.0
            since velocity is undefined.

        Raises
        ------
        ValueError
            If no foot joints are found or specified, or if method
            is unknown.
        """
        if method not in ("velocity", "height"):
            raise ValueError(
                f"Unknown method '{method}'. Choose 'velocity' or 'height'.")

        if coords is None:
            coords = self.get_spatial_coord(centered=centered)

        # Auto-detect foot joints
        if foot_joints is None:
            foot_joints = []
            for node in self.nodes:
                name_lower = node.name.lower()
                if any(kw in name_lower for kw in ("foot", "toe")):
                    if not node.is_end_site():
                        foot_joints.append(node.name)
            if not foot_joints:
                raise ValueError(
                    "Could not auto-detect foot joints. Please provide "
                    "foot_joints explicitly (e.g. ['LeftFoot', 'RightFoot']).")

        # Get indices into the spatial coord array
        foot_indices = []
        for name in foot_joints:
            if name not in self.node_index:
                raise ValueError(f"Joint '{name}' not found in skeleton.")
            foot_indices.append(self.node_index[name])

        foot_coords = coords[:, foot_indices, :]  # (F, num_feet, 3)

        if method == "velocity":
            if threshold is None:
                threshold = 0.5  # units/frame

            if self.frame_count < 2:
                return np.ones((self.frame_count, len(foot_joints)),
                               dtype=np.float64)

            foot_vel = foot_coords[1:] - foot_coords[:-1]  # (F-1, num_feet, 3)
            speed = np.linalg.norm(foot_vel, axis=-1)  # (F-1, num_feet)
            contacts = (speed < threshold).astype(np.float64)

            # Prepend first frame as contact (no velocity info)
            first_frame = np.ones((1, len(foot_joints)), dtype=np.float64)
            contacts = np.concatenate([first_frame, contacts], axis=0)

        else:  # height method
            rest_pose_arr: npt.NDArray[np.float64] = self.get_rest_pose(mode='coordinates')  # type: ignore[assignment]
            up_idx = get_up_axis_index(self, rest_pose_arr)
            foot_heights = foot_coords[:, :, up_idx]  # (F, num_feet)

            if threshold is None:
                # Auto-calibrate: 5th percentile of foot heights
                threshold = float(np.percentile(foot_heights, 5)) + 0.5

            contacts = (foot_heights < threshold).astype(np.float64)

        return contacts

    def to_feature_array(
        self,
        representation: str = "6d",
        include_root_pos: bool = True,
        include_velocities: bool = False,
        include_foot_contacts: bool = False,
        centered: str = "world",
        foot_joints: list[str] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Export motion as a single flat feature array for ML pipelines.

        Composes root position, joint rotations, velocities, and foot
        contacts into a single ``(F, D)`` array ready for model input.

        Parameters
        ----------
        representation : str, optional
            Rotation representation: ``'euler'``, ``'6d'`` (default),
            ``'quaternion'``, ``'axisangle'``, or ``'rotmat'``.
        include_root_pos : bool, optional
            If True (default), include root position (3 columns).
        include_velocities : bool, optional
            If True, include joint velocity features. Reduces frame
            count by 1 (first frame has no velocity).
        include_foot_contacts : bool, optional
            If True, include foot contact labels.
        centered : str, optional
            Coordinate centering mode (default ``"world"``).
        foot_joints : list of str or None, optional
            Foot joints for contact detection. Only used when
            ``include_foot_contacts=True``.

        Returns
        -------
        ndarray, shape (F, D) or (F-1, D)
            Flat feature array. Frame count is F-1 when velocities
            are included (first frame is dropped for alignment).

        Raises
        ------
        ValueError
            If representation is unknown.
        """
        valid_reps = {"euler", "6d", "quaternion", "axisangle", "rotmat"}
        if representation not in valid_reps:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Choose from {sorted(valid_reps)}.")

        # Compute spatial coords once (shared by velocities and contacts)
        coords = None
        if include_velocities or include_foot_contacts:
            coords = self.get_spatial_coord(centered=centered)

        parts: list[npt.NDArray[np.float64]] = []

        # Root position
        if include_root_pos:
            parts.append(self.root_pos)

        # Joint rotations
        if representation == "euler":
            rot = self.joint_angles.reshape(self.frame_count, -1)
        elif representation == "6d":
            _, rot_raw, _ = self.get_frames_as_6d()
            rot = rot_raw.reshape(self.frame_count, -1)
        elif representation == "quaternion":
            _, rot_raw, _ = self.get_frames_as_quaternion()
            rot = rot_raw.reshape(self.frame_count, -1)
        elif representation == "axisangle":
            _, rot_raw, _ = self.get_frames_as_axisangle()
            rot = rot_raw.reshape(self.frame_count, -1)
        else:  # rotmat
            _, rot_raw, _ = self.get_frames_as_rotmat()
            rot = rot_raw.reshape(self.frame_count, -1)
        parts.append(rot)

        # Determine if we need to truncate for velocity alignment
        has_velocity_trim = include_velocities and self.frame_count >= 2

        # Velocities
        if include_velocities:
            vel = self.get_joint_velocities(
                centered=centered, in_frames=True, coords=coords)
            # vel is (F-1, N, 3), flatten to (F-1, N*3)
            vel_flat = vel.reshape(vel.shape[0], -1)
            parts.append(vel_flat)

        # Foot contacts
        if include_foot_contacts:
            contacts = self.get_foot_contacts(
                foot_joints=foot_joints, centered=centered, coords=coords)
            parts.append(contacts)

        # Align frames: if velocities are included, trim all to F-1
        if has_velocity_trim:
            aligned = []
            for p in parts:
                if p.shape[0] == self.frame_count:
                    aligned.append(p[1:])  # drop first frame
                else:
                    aligned.append(p)      # already F-1
            parts = aligned

        return np.concatenate(parts, axis=1)


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


