from pathlib import Path
import numpy as np
import copy

from .bvhnode import BvhNode, BvhJoint, BvhRoot 

class Bvh:
    """
    takes as initial data the path of a bvh file
    the hierarchy information are stored in a list of BvhNode objects, one object per joint.
    the frames are stored as a numpy 2D array.
    """
    def __init__(self, nodes=[BvhRoot()], frames = np.array([[]]), frame_template=[], frame_frequency=0):
        self.nodes = nodes
        self.frames = frames
        self.frame_template = frame_template
        self.frame_frequency = frame_frequency
        self.frame_count = len(self.frames)
        self.root = self.nodes[0]
        

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
        return f'{count_joints} elements in the Hierarchy, {self.frame_count} frames'
        
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

    


    def to_bvh_file(self, new_filepath):
        """
        This function will write the bvh object into a bvh file, following the proper standard for this type of file.
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
                
        #-------------- end of the write function
    

    def get_df_constructor(self):
        """
        This function returns a list of dictionnary ready made to easily create a pandas DataFrame. 
        The purpose of this function is to be combined with pd.DataFrame this way:
        pd.Dataframe(bvh_object.get_df_constructor)
        The colums will be of the form 'name_ax_pos/rot' (ex: Neck_X_rot'), and each line represents a frame.
        """
        
        #pos_channels = self.root.pos_channels
        #rot_channels = self.root.rot_channels
        dictList = []
        
        for i, frame in enumerate(self.frames):
            tempo_dict = {}
            tempo_dict['time'] = i*self.frame_frequency
            for j, col_name in enumerate(self.frame_template):
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


