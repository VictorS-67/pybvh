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
    #list of BvhNode objects in the file hierarchy
    node_list = [] 
    # this list will help to construct
    #  the second part after the hierarchy, the frame list
    frame_template = []
    # this is a flag variable to tell us if the parent of the joint
    # we are working on is the directly previous joint in the file, or not
    parent_is_previous = True
    # this is necessary when the parent of the joint we are working on
    # is not directly the previous joint in the file
    parent_depth = 1
    # line number if we need to report a problem in the file
    line_number = 0

    filepath = test_file(filepath)
    
    with open(filepath, "r") as f:
        #---------- first, read the hierarchy in the file (first part of the file)

        # read the file line by line
        for line in f:
            line_number += 1
            line = line.split()
            # if the line starts with ROOT, then the next 3 lines are about the information of the root
            # we want to save them in a BvhRoot object
            
            if line[0] == 'ROOT':
                name = line[1]
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('root', f, line_number)
                except:
                    raise Exception(f"Could not read the offset or channels of the root {name},\n at line {line_number} in the file {filepath}")
                
                node_list.append(BvhRoot(name, offset, pos_channels, rot_channels, [], None))
                for ax in pos_channels:
                    frame_template.append(name+'_'+ax+'_pos')
                for ax in rot_channels:
                    frame_template.append(name+'_'+ax+'_rot')
                    
            # if the line starts with JOINT,
            # then the next 3 lines are about the information of this joint
            # we want to save them in a BvhJoint object
            elif line[0] == 'JOINT':
                name = line[1]
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('joint', f, line_number)
                except:
                    raise Exception(f"Could not read the offset or channels of the joint {name},\n at line {line_number} in the file {filepath}")
                #since it is a joint, pos_channels == None

                if parent_is_previous:
                    # if this joint is direclty folowing the previous one in the file,
                    # its parent is the previous Joint in the list
                    parent_node = node_list[-1]
                    parent_depth = 1
                else:
                    # if this joint is not directly following its parent in the file,
                    # that means that there was and 'End Site' above it,
                    # with also a certain number of '}'. Those increased the parent_depth variable.
                    # To get its parent, we will walk backward the chain of parent-child links
                    # as many times as the parent_depth tell us to.
                    # the first node of the chain needs to be the previous end site
                    parent_node = node_list[-1]
                    parent_depth -= 1
                    for i in range(parent_depth):
                        parent_node = parent_node.parent

                node_list.append(BvhJoint(name, offset, rot_channels, [], parent_node))
                # let's not forget that, we gave this Joint a parent,
                # but then we also need to tell the parent that this is its child 
                # so we link its parent directly to the node we just added in the list
                parent_node.children = parent_node.children + [node_list[-1]]
                parent_is_previous = True
                for ax in rot_channels:
                    frame_template.append(name+'_'+ax+'_rot')
                    
            elif line[0] == 'End':
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('end_site', f, line_number)
                except:
                    raise Exception(f"Could not read the offset or channels of the End Site\nat line {line_number} in the file {filepath}")
                # since it is an end site, pos_channels == None and rot_channels == None

                #for an end site, the parent is always just before in the list
                parent_node = node_list[-1]
                parent_depth = 1

                node_list.append(BvhNode('End Site '+parent_node.name, offset, parent_node))
                parent_node.children = parent_node.children + [node_list[-1]]
                
            elif line[0] == '}':
                # to corectly assign the parent to a node,
                # we need to increase the parent_depth variable every time
                # we read a '}' in the file
                parent_depth += 1
                parent_is_previous = False

            elif line[0] == "Frames:":
                frame_count = int(line[1])

            elif line[0] == "Frame" and line[1] == "Time:":
                frame_frequency = float(line[2])
                # we will modify a bit the frequency to have a higher precision than what is given in the file
                frame_frequency = 1/int(1/frame_frequency)
                # --- we close the loop related to reading the hierarchy ---
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


def _get_offset_channels(node_type:str, f, line_number:int):
    """
    This function is used to extract the offset and channels information
    from the file. It is used in the _extract_bvh_file_info() function.
    It will also advance in the file until the end of the information
    (3 lines for root and joint, 2 for end sites)
    """
    offset = None
    rot_channels = None
    pos_channels = None

    # i is used to get out of the subloop after 3 or 2 lines
    # depending on the node type
    i=0

    line_number = line_number

    if node_type == 'root':
        for line in f:
            line_number += 1
            line = line.split()
            if line[0] == 'OFFSET':
                offset = [float(x) for x in line[1:]]
            elif line[0] == 'CHANNELS':
                pos_channels, rot_channels = [x[0] for x in line[2:5]], [x[0] for x in line[5:]]
            #after 3 lines we get out of the subloop    
            if i == 2:
                break
            i += 1
        #checking that the information is complete
        if len(offset) !=3 or len(pos_channels) !=3 or len(rot_channels) !=3:
            raise Exception()
    elif node_type == 'joint':
        for line in f:
            line_number += 1
            line = line.split()
            if line[0] == 'OFFSET':
                offset = [float(x) for x in line[1:]]
            elif line[0] == 'CHANNELS':
                rot_channels = [x[0] for x in line[2:]]
            #after 3 lines we get out of the subloop    
            if i ==2:
                break
            i += 1
        #checking that the information is complete
        if len(offset) !=3 or len(rot_channels) !=3:
            raise Exception()
    elif node_type == 'end_site':
        for line in f:
            line_number += 1
            line = line.split()
            if line[0] == 'OFFSET':
                offset = [float(x) for x in line[1:]]
            #after 2 lines we get out of the subloop    
            if i ==1:
                break
            i += 1
        #checking that the information is complete
        if len(offset) !=3:
            raise Exception()
    else:
        raise ValueError('node_type should be either root, joint or end_site')

    return (offset, pos_channels, rot_channels, line_number)
    
    