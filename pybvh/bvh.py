from pathlib import Path
import numpy as np
import copy

from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .spatial_coord import frame_to_spatial_coord, frames_to_spatial_coord

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

        self._has_spatial = False
        # the spatial coordinates stored are always in world centered coordinates
        # so the root position is the actual root position in the bvh file
        self._spatial_coord = np.array([[]])

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
            if 'End Site' not in node.name  : count_joints += 1 
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
            if 'End Site' in node.name:
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
            if 'End Site' in node.name :
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


    def get_spatial_coord(self, frame_num=-1, centered="world", change_skeleton=None):
        """
        Obtain the spatial coordinates of the joints.
        The coordinates are given in the form of a numpy array.
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
        
        return_one_frame = True

        if (frame_num < -1) or (frame_num >= self.frame_count):
            raise ValueError("frame_num needs to be -1 or a positive integer smaller than the total amount of frames in the bvh file.")

        elif frame_num >= 0:
            # the user wants only one frame
            # since it is only one frame, we don't care about the performance,
            # no need to check if the spatial coordinates are already calculated
            frame = frame_to_spatial_coord(self, self.frames[frame_num], change_skeleton=change_skeleton)

        elif (frame_num == -1) and (not self._has_spatial) and (change_skeleton == None):
            # the user wants all the frames, with the same skeleton as the one in the bvh object
            # we don't have already calculated and saved the spatial coordinates
            # then we calculate the frames in world coord, and we save them
            return_one_frame = False
            frames_array = frames_to_spatial_coord(self, frames = self.frames, change_skeleton=change_skeleton)
            self._spatial_coord = frames_array.copy()
            self._has_spatial = True
        
        elif (frame_num == -1) and (change_skeleton != None):
            # the user wants all the frames, with a different skeleton than the one in the bvh object
            # we need to calculate the frames in world coord, and we don't save them
            return_one_frame = False
            frames_array = frames_to_spatial_coord(self, frames = self.frames, change_skeleton=change_skeleton)
        
        elif (frame_num == -1) and self._has_spatial and (change_skeleton == None):
            # the user wants all the frames, with the same skeleton as the one in the bvh object
            # we have already calculated and saved all the spatial coordinates for the frames
            # (they are saved in world centered coordinates)
            return_one_frame = False
            frames_array = self._spatial_coord.copy()
            
        # we have the frame/frames, in world coordinates
        # now we need to center them as the user wants
            
        if centered == "world" :
            if return_one_frame:
                return frame
            else:
                return frames_array
        elif centered == "first":
            # for the first frame, we only use its root position and total length.
            # The change of the skeleton has no impact whatsoever
            first_frame = frame_to_spatial_coord(self, self.frames[0])
            if return_one_frame:
                return frame - np.tile(first_frame[:3], int(len(first_frame)/3))
            else:
                return frames_array - np.tile(first_frame[:3], int(len(first_frame)/3)) #broadcasting
        elif centered == "skeleton":
            if return_one_frame:
                return frame - np.tile(frame[:3], int(frame.shape[0]/3))
            else:
                return frames_array - np.tile(frames_array[:,:3], int(frames_array.shape[1]/3))

        

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
            rest_coord = frame_to_spatial_coord(self, rest_angle, skel_centered=True)
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
        if centered == "skeleton":
            #if skeleton centered coordinates
            message = f"The coordinates in the dataFrame are centered on the skeleton."
        elif centered == "first":
            #if first frame coordinates
            message = f"The coordinates in the dataFrame are centered on the first frame."
        elif centered == "world":  
            #if world coordinates
            message = f"The coordinates in the dataFrame are in world coordinates."
        
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
            hier_dict[node.name]['parent'] = 'None' if node.parent==None else node.parent.name
            
        return copy.deepcopy(hier_dict)
    
    def _create_name2coord_idx(self):
        name2coord_idx = {}
        i=0
        for node in self.nodes:
            for ax in ['X', 'Y', 'Z']:
                name2coord_idx[f'{node.name}_{ax}'] = i
                i+=1
        self.name2coord_idx = name2coord_idx


    

    def single_joint_euler_angle(self, joint_name, order):
        """
        This function takes as parameters the joint name to modify (str),
        and the new euler angle order (str eg.'XYZ' or list eg ['X', 'Y', 'Z']).
        It changes the euler angle of this joint for all frames.
        """
        

        
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#----------------------------- end of BVH class-----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


