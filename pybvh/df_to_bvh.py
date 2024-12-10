import re
import copy

from .bvh import Bvh
from .bvhnode import BvhNode, BvhJoint, BvhRoot 

def _check_df_columns(df):
    """
    This function takes a pandas df as variable. It will return a copy of the df,
    with only the colums of the form 'name_ax_pos/rot' (ex: Neck_X_rot'), or 'time'.
    In case there is no columns following this naming convention, an error is raised.
    In case there is no time column, an error is raised.
    The first rotational data appearing in the DataFrane should be 3 columns for the root position followed by 3 columns for the root rotation
    """
    new_df = df.copy()
    # first we keep only the columns following the aforementioned format
    valid_pattern = r'(.+_[xyzXYZ]_pos|.+_[xyzXYZ]_rot)'
    for col_name in new_df.columns:
        pattern_ok = (re.fullmatch(valid_pattern, col_name) != None)
        if not pattern_ok:
            new_df = new_df.drop(col_name, axis=1) #that includes dropping the time col

    #check that root pos and rot appear first
    root_name = new_df.columns[0].split('_')
    root_name = root_name[0]
    root_er = 'The first rotational data appearing in the DataFrane should be 3 columns for the root position followed by 3 columns for the root rotation'
    for col_name in new_df.columns[0:3]:
        #those should be the root pos
        splitted_name = col_name.split('_')
        col_joint_name, ax, rotpos = splitted_name[0], splitted_name[1], splitted_name[2]
        if col_joint_name != root_name or rotpos != 'pos' :
            raise Exception(root_er)
    for col_name in new_df.columns[3:6]:
        #those should be the root rot
        splitted_name = col_name.split('_')
        col_joint_name, ax, rotpos = splitted_name[0], splitted_name[1], splitted_name[2]
        if col_joint_name != root_name or rotpos != 'rot' :
            raise Exception(root_er)
        
    #check if there is a time column in the original df, and add it back into the df
    has_time = False
    for col_name in df.columns:
        if col_name.lower() == 'time':
            has_time = True
            time_col = df[col_name]
            break
    if len(new_df.columns) == 0:
        raise Exception("No column found following the proper naming convention 'name_ax_pos/rot'")
    elif not has_time:
        raise Exception("No time column found")

    new_df.insert(0, 'time', time_col)

    return new_df

    
def _check_df_match_with_hier(hier, df):
    """
    take as input a DataFrame, and a list of nodes containing the hierarchy info.
    return the hierarchy list as well as the df
    
    check that the df contain all the info for the joints present in the hierarchy (here a list of nodes).
    The Dataframe given has already been checked and only contains columns with proper naming.
    The first column is time, the next 3 are the root position, and the next 3 the root rotation.
    If the column of the df are not in the same order as the list, priority to the hierarchy list. 
    We will change the column order to match the hierarchy
    """
    if any([not isinstance(x, BvhNode) for x in hier]):
        raise ValueError("The list given should only contain BvhNode class/subclasse objects")

    # Will create a list of column names based on the info in the hier list.
    # If the list match the df.columns, keep it this way.

    def node_to_names(node, rotpos = 'rot'):
        if rotpos == 'rot':
            return [node.name + '_' + ax + '_rot' for ax in node.rot_channels]
        elif rotpos == 'pos':
            return [node.name + '_' + ax + '_pos' for ax in node.pos_channels]
    
    correct_col_list = ['time']
    correct_col_list.extend(node_to_names(hier[0], rotpos= 'pos'))
    for node in hier:
        if 'End Site' in node.name:
            continue
        correct_col_list.extend(node_to_names(node))

    if list(df.columns) == correct_col_list:
        return hier, df

    # If the list we created doesn't match witht the df columns,
    # create a new df with correctly ordered column
    try:
        df = df[correct_col_list]
    except:
        raise Exception('Some columns in the DataFrame do not match with information in the list of nodes')

    return hier, df

def _complete_hier_dict(hier, df):
    """
    Take as input a dictionary containing the hierarchy info.
    Check the info in the dictionary and return one containing the hierarchy info
    with rot and pos channels if missing before.

    We won't assume that the dictionnary is ordered here.
    eg of dict
    {'name1' : {
                  'offset': [float, float, float]
                  'parent' : 'nameOfParent',
                  'children' : ['nameOfChild1',... ],
                  'rot_channels' : ['X', 'Y', 'Z'], #optional
                  'pos_channels' : ['X', 'Y', 'Z'] #optional
                  },
     'name2' : { ... }, 
     ...
     }
     If the dict misses rot or pos info for some categories, use df columns order to fill it
    """
    df_cols = df.columns
    
    for name, info_dict in hier.items():
        #all elt should have offset and parent component
        try:
            offset = info_dict['offset']
        except:
            raise Exception(f'no offset component found in the dictionary {name}')
        try:
            parent = info_dict['parent']
            if (parent == None) or (parent == 'None'):
                root = name
                #if it's the root, we want to test if has pos_channels
                try:
                    pos_channels = info_dict['pos_channels']
                    root_has_pos = True
                except:
                    root_has_pos = False
            else:
                try:
                    parent_info = hier[parent]
                except:
                    raise Exception(f'The parent {parent} of node {name} is not a node in the dictionnary')
        except:
            raise Exception(f'no parent component found in the dictionary {name}')
            
        #for end site, that's all there is to it
        if 'End Site' in name:
            continue
        
        try:
            children = info_dict['children']
        except:
            raise Exception(f'no children component found in the dictionary {name}')

        for child in children:
            try:
                child_info = hier[child]
            except:
                raise Exception(f'The child {child} of node {name} is not a node in the dictionnary')

        # we finished checking that the necesseray info are here
        # now we will check if rot and pos channels are present.
        # In case they are not, we will add them from the df
        try:
            rot_channels = info_dict['rot_channels']
        except:
            rot_channels = []
            corresponding_cols = df.columns[df.columns.str.contains(name)]
            for col_name in corresponding_cols:
                splitted_name = col_name.split('_')
                col_joint_name, ax, rotpos = splitted_name[0], splitted_name[1], splitted_name[2]
                if rotpos != 'rot':
                    continue
                rot_channels.append(ax)

        hier[name]['rot_channels'] = rot_channels
                

    #finally we add the root pos if needed
    if not root_has_pos:
        pos_channels = []
        for pos_col in df_cols[1:4]:
            pos_channels.append(pos_col.split('_')[1])
        hier[root]['pos_channels'] = pos_channels

    return hier
        
            
def _hier_dict_to_list(hier):
    """
    Take as input a dictionary containing the Complete hierarchy info.
    Return the hierarchy as an organized list of nodes
    
    ex of dictionary : 
    {'name1' : {
                  'offset': [float, float, float]
                  'parent' : 'nameOfParent',
                  'children' : ['nameOfChild1',... ],
                  'rot_channels' : ['X', 'Y', 'Z']
                  'pos_channels' : ['X', 'Y', 'Z']
                  },
     'name2' : { ... }, 
     ...
     }
    """
    er_str = 'Incorrect column name for the column '

    #first we create the list, without filling children or parent yet
    #we will use a recursive function for that going through the children of the nodes
    def create_list_rec(node_name, is_start=False):
        info_dict = hier[node_name]
        if 'End Site' in node_name:
            node = BvhNode(node_name, offset=info_dict['offset'])
            return [node]
        else:
            #We want only the node at the very beginning to be BvhRoot, the rest BvhJoint (or BvhNode if End Site)
            if is_start :
                node = BvhRoot(node_name, offset=info_dict['offset'],
                               rot_channels=info_dict['rot_channels'], pos_channels=info_dict['pos_channels'])
            else : 
                node = BvhJoint(node_name, offset=info_dict['offset'], rot_channels=info_dict['rot_channels'])
            list_node = [node]
            for child in info_dict['children']:
                list_node += create_list_rec(child)
            return list_node

    #let's find the root and call our rec fun on it
    for name, info_dict in hier.items():
        parent = info_dict['parent']
        if (parent == None) or (parent == 'None'): 
            root = name
            break
            
    list_nodes = create_list_rec(root, is_start=True) #we send the is_start to make sure the first elt is a BvhRoot

    # we now have the list in the correct order. We want to fill the 
    # parent and children of each node with a ref pointer to the correct object in the list

    #first pass, we will list the index of each node in the list
    name_to_index = {}
    for i, node in enumerate(list_nodes):
        name_to_index[node.name] = i

    #then we use that to rapidly fill the info
    for node in list_nodes:
        parent_name = hier[node.name]['parent']
        if (parent_name == None) or (parent_name == 'None'):
            #the node is the root, we don't fill the parent but we do fill the missng pos info
            node.pos_channels = hier[node.name]['pos_channels']
        else:
            node.parent = list_nodes[name_to_index[parent_name]]
            
        if 'End Site' in node.name:
            continue
            
        children_name_list = hier[node.name]['children']
        children_list = []
        for child_name in children_name_list:
            children_list.append(list_nodes[name_to_index[child_name]])
        node.children = children_list
    
    return list_nodes
        

def df_to_bvh(hier, df):
    """
    This function takes a pandas df as variable, as well as either a list of nodes, or a dictionnary representing the hierarchy.
    It returns a Bvh object, based on the dataframe and the hierarchy information..
    
    The df needs to have a time colums, as well as colums of the form 'name_ax_pos/rot' (ex: Neck_X_rot').
    A call is made to _check_df_columns in order to check the columns.
    The order of the column in the df is important, since the nodes order in the bvh object
    will reflect the order they appear in the df columns.

    hier can be a list of nodes (BvhNode, BvhJoint and BvhRoot objects),
    or a dictionary including the elements: 
    {'name1' : {
                  'offset': [float, float, float]
                  'parent' : 'nameOfParent',
                  'children' : ['nameOfChild1',... ]
                  },
     'name2' : { ... }, 
     ...
     }
    """
    
    df = _check_df_columns(df) # this creates a copy of the df

    hier = copy.deepcopy(hier)
    
    if isinstance(hier, list):
        #arrange the df correctly to fit with list of nodes info if possible
        hier_list, df = _check_df_match_with_hier(hier, df) #arrange the df correctly to fit with list of nodes info if possible
    elif isinstance(hier, dict):
        hier = _complete_hier_dict(hier, df) # check the info in the dict and fill them from df if possible
        hier_list = _hier_dict_to_list(hier) # create the hier list of nodes
        hier_list, df = _check_df_match_with_hier(hier_list, df)
    else:
        raise TypeError('variable hier should be either a list of nodes or a dictionary')

    time_series = df['time']
    frames = df.drop(['time'], axis=1)
    frame_template = list(frames.columns)
    frames = frames.to_numpy()
    frame_frequency = 1/int(1/(time_series.to_numpy()[-1] / (len(time_series)-1)))

    return Bvh(nodes=hier_list, frames=frames, frame_template=frame_template , frame_frequency=frame_frequency)

    