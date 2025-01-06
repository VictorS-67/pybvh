from pathlib import Path
import numpy as np

from .bvhnode import BvhNode, BvhJoint, BvhRoot 
from .bvh import Bvh
from .tools import test_file

def read_bvh_file(filepath):
    """
        This method construct a Bvh object based on the information
        given by the _extract_bvh_file_info() method
    """
    node_list, frame_array, frame_frequency, frame_template = _extract_bvh_file_info(filepath)
    return Bvh(nodes=node_list, frames=frame_array, frame_frequency=frame_frequency, frame_template=frame_template)

def _extract_bvh_file_info(filepath):
    """
    This function only need the filepath of a bvh file to work. 
    It returns a tuple of 4 objects: a list of BvhNode objects, a numpy array of the frames, the frame_template,
    and the frame frequency.
    """
    node_list = []
    frame_template = [] # this will help to construct the scond part after the hierarchy, the frame list

    filepath = test_file(filepath)
    
    with open(filepath, "r") as f:
        #---------- first, read the hierarchy in the file (first part of the file)
        
        parent_position = 0 # this will help when the parent of the joint we are working on is not directly the previous joint in the file
        parent_is_previous = True #this is a flag variable to telle us if the parent of the joint we are working on is the previous joint in the file, or not
        for line in f:
            line = line.split()
            
            # if the line starts with ROOT, then the next 3 lines are about the information of the root
            # we want to save them in a BvhRoot object
            if line[0] == 'ROOT':
                name = line[1]
                i = 0
                for line in f:
                    line = line.split()
                    if line[0] == 'OFFSET':
                        offset = [float(x) for x in line[1:]]
                    elif line[0] == 'CHANNELS':
                        pos_channels, rot_channels = [x[0] for x in line[2:5]], [x[0] for x in line[5:]]
                    #after 3 lines we get out of the subloop    
                    if i == 2:
                        break
                    i += 1
                node_list.append(BvhRoot(name, offset, pos_channels, rot_channels, [], None))
                for ax in pos_channels:
                    frame_template.append(name+'_'+ax+'_pos')
                for ax in rot_channels:
                    frame_template.append(name+'_'+ax+'_rot')
                    
            # if the line starts with JOINT, then the next 3 lines are about the information of this joint
            # we want to save them in a BvhJoint object
            elif line[0] == 'JOINT':
                name = line[1]
                i = 0
                for line in f:
                    line = line.split()
                    if line[0] == 'OFFSET':
                        offset = [float(x) for x in line[1:]]
                    elif line[0] == 'CHANNELS':
                        rot_channels = [x[0] for x in line[2:]]
                    #after 3 lines we get out of the subloop    
                    if i ==2:
                        break
                    i += 1
                if parent_is_previous:
                    #if this joint is direclty folowing the previous one in the file, its parent is the previous Joint in the list
                    parent = node_list[-1]
                else:
                    #if this joint is direclty folowing the previous one in the file and there are '}' between, we need to use the parent_position variable
                    parent = node_list[parent_position]
                node_list.append(BvhJoint(name, offset, rot_channels, [], parent))
                #let's not forget that, we gave this Joint a parent, but then we also need to tell the parent that this is its child
                parent.children = parent.children + [node_list[-1]]
                parent_position += 1
                parent_is_previous = True
                for ax in rot_channels:
                    frame_template.append(name+'_'+ax+'_rot')
                    
            elif line[0] == 'End':
                i = 0
                for line in f:
                    line = line.split()
                    if line[0] == 'OFFSET':
                        offset = [float(x) for x in line[1:]]
                    #after 2 lines we get out of the subloop    
                    if i ==1:
                        break
                    i += 1
                #for an end site, the parent is always just before in the list
                parent = node_list[-1]
                node_list.append(BvhNode('End Site '+parent.name, offset, parent))
                parent.children = parent.children + [node_list[-1]]
                parent_position += 1
                
            elif line[0] == '}':
                #finally, to corectly assign the parent, we need to regress in the parent position whenever the imbrication are closed
                parent_position -= 1
                parent_is_previous = False
            elif line[0] == "Frames:":
                frame_count = int(line[1])
            elif line[0] == "Frame" and line[1] == "Time:":
                frame_frequency = float(line[2])
                # we will modify a bit the frequency to have a higher precision than what is given in the file
                frame_frequency = 1/int(1/frame_frequency)
                break
        #small test to see if we reach the end of the hierarchy with no trouble.
        try:
            frame_count == 0
            frame_frequency == 0
        except: 
            print("Frame count or frame frequency is missing")

        #----------  End of the Hierarchy part. After the hierarchy comes the frames data.
        
        frame_number = 0
        for line in f:
            line = line.split()
            line = np.array([float(x) for x in line])

            if frame_number ==0:
                frame_array = [line] 
            else:
                frame_array = np.append(frame_array, [line], axis=0)

            frame_number+=1

            
    #-----------------end of reading the file
    # frame_template is a list we created of the form [jointName_ax_pos/rot].
    # ex : [Hips_X_pos, Hips_Y_pos, Hips_Z_pos, Hips_X_rot, ...]
    return (node_list, frame_array, frame_frequency, frame_template)
