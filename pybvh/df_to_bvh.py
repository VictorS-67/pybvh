from __future__ import annotations

import re
import copy
import numpy as np
from typing import TYPE_CHECKING

from .bvh import Bvh
from .bvhnode import BvhNode, BvhJoint, BvhRoot

if TYPE_CHECKING:
    import pandas as pd

def _check_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and filter DataFrame columns to the expected naming convention.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Must contain a ``time`` column and motion columns
        following the ``name_ax_pos/rot`` pattern (e.g. ``Neck_X_rot``).
        The first six motion columns must be root position then root rotation.

    Returns
    -------
    new_df : pandas.DataFrame
        Copy of *df* containing only the ``time`` column and validly-named
        motion columns, with root position/rotation verified.

    Raises
    ------
    Exception
        If no columns match the naming convention, if no ``time`` column is
        found, or if the first six motion columns are not root position
        followed by root rotation.
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
    time_col: pd.Series | None = None  # type: ignore[type-arg]
    for col_name in df.columns:
        if col_name.lower() == 'time':
            has_time = True
            time_col = df[col_name]
            break
    if len(new_df.columns) == 0:
        raise Exception("No column found following the proper naming convention 'name_ax_pos/rot'")
    elif not has_time or time_col is None:
        raise Exception("No time column found")

    new_df.insert(0, 'time', time_col)

    return new_df


def _check_df_match_with_hier(hier: list[BvhNode], df: pd.DataFrame) -> tuple[list[BvhNode], pd.DataFrame]:
    """Reorder DataFrame columns to match the joint hierarchy.

    Parameters
    ----------
    hier : list of BvhNode
        Ordered hierarchy of joints/nodes.
    df : pandas.DataFrame
        Validated DataFrame whose columns follow the ``name_ax_pos/rot``
        convention.

    Returns
    -------
    hier : list of BvhNode
        The same hierarchy list, unchanged.
    df : pandas.DataFrame
        DataFrame with columns reordered to match *hier*.

    Raises
    ------
    ValueError
        If *hier* contains objects that are not ``BvhNode`` instances.
    Exception
        If the DataFrame is missing columns required by the hierarchy.
    """
    if any([not isinstance(x, BvhNode) for x in hier]):
        raise ValueError("The list given should only contain BvhNode class/subclasse objects")

    # Will create a list of column names based on the info in the hier list.
    # If the list match the df.columns, keep it this way.

    def node_to_names(node: BvhNode, rotpos: str = 'rot') -> list[str]:
        if rotpos == 'rot':
            assert isinstance(node, BvhJoint)
            return [node.name + '_' + ax + '_rot' for ax in node.rot_channels]
        elif rotpos == 'pos':
            assert isinstance(node, BvhRoot)
            return [node.name + '_' + ax + '_pos' for ax in node.pos_channels]
        else:
            raise ValueError(f"rotpos must be 'rot' or 'pos', got '{rotpos}'")

    correct_col_list: list[str] = ['time']
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

def _complete_hier_dict(hier: dict[str, dict], df: pd.DataFrame) -> dict[str, dict]:
    """Fill missing rotation/position channel info in a hierarchy dictionary.

    Parameters
    ----------
    hier : dict
        Hierarchy dictionary keyed by joint name. Each value is a dict with
        required keys ``'offset'`` and ``'parent'``, and optional keys
        ``'children'``, ``'rot_channels'``, and ``'pos_channels'``.
        Missing channel information is inferred from *df* column order.
    df : pandas.DataFrame
        Validated DataFrame whose columns follow the ``name_ax_pos/rot``
        convention.

    Returns
    -------
    hier : dict
        The same dictionary, updated in-place with ``'rot_channels'`` (and
        ``'pos_channels'`` for the root) filled from *df* where absent.

    Raises
    ------
    Exception
        If any node is missing ``'offset'``, ``'parent'``, or ``'children'``
        entries, or if a referenced parent/child is not present in the dict.
    """
    df_cols = df.columns
    root: str = ""
    root_has_pos: bool = False

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


def _hier_dict_to_list(hier: dict[str, dict]) -> list[BvhNode]:
    """Convert a complete hierarchy dictionary to an ordered list of nodes.

    Parameters
    ----------
    hier : dict
        Complete hierarchy dictionary (as returned by
        ``_complete_hier_dict``), keyed by joint name with ``'offset'``,
        ``'parent'``, ``'children'``, ``'rot_channels'``, and optionally
        ``'pos_channels'`` for each entry.

    Returns
    -------
    list_nodes : list of BvhNode
        Depth-first ordered list of ``BvhRoot``, ``BvhJoint``, and
        ``BvhNode`` (end-site) objects with parent/children references set.
    """
    er_str = 'Incorrect column name for the column '

    #first we create the list, without filling children or parent yet
    #we will use a recursive function for that going through the children of the nodes
    def create_list_rec(node_name: str, is_start: bool = False) -> list[BvhNode]:
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
                list_node += create_list_rec(child)  # type: ignore[arg-type]
            return list_node  # type: ignore[return-value]

    #let's find the root and call our rec fun on it
    root_name: str = ""
    for name, info_dict in hier.items():
        parent = info_dict['parent']
        if (parent == None) or (parent == 'None'):
            root_name = name
            break

    list_nodes = create_list_rec(root_name, is_start=True) #we send the is_start to make sure the first elt is a BvhRoot

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
            node.pos_channels = hier[node.name]['pos_channels']  # type: ignore[attr-defined]
        else:
            node.parent = list_nodes[name_to_index[parent_name]]

        if 'End Site' in node.name:
            continue

        children_name_list = hier[node.name]['children']
        children_list = []
        for child_name in children_name_list:
            children_list.append(list_nodes[name_to_index[child_name]])
        node.children = children_list  # type: ignore[attr-defined]

    return list_nodes


def df_to_bvh(hier: list[BvhNode] | dict[str, dict], df: pd.DataFrame) -> Bvh:
    """Create a Bvh object from a hierarchy description and a motion DataFrame.

    Build a complete BVH representation by combining skeletal hierarchy
    information with per-frame motion data stored in a pandas DataFrame.
    Column order in the DataFrame determines joint ordering in the resulting
    ``Bvh`` object unless a hierarchy list is provided (in which case the
    hierarchy order takes priority).

    Parameters
    ----------
    hier : list of BvhNode or dict
        Skeletal hierarchy, supplied as either:

        * A **list** of ``BvhRoot``, ``BvhJoint``, and ``BvhNode`` objects
          with parent/children already set.
        * A **dict** keyed by joint name, where each value contains at least
          ``'offset'`` (list of 3 floats), ``'parent'`` (str or None), and
          ``'children'`` (list of str).  Optional keys ``'rot_channels'``
          and ``'pos_channels'`` (each a list such as ``['X', 'Y', 'Z']``)
          will be inferred from *df* if absent.
    df : pandas.DataFrame
        Motion data.  Must include a ``time`` column and motion columns
        named ``<joint>_<axis>_pos`` or ``<joint>_<axis>_rot`` (e.g.
        ``Hips_X_pos``, ``Neck_Z_rot``).  The first three motion columns
        must be root position, followed by three root rotation columns.

    Returns
    -------
    bvh : Bvh
        Fully constructed ``Bvh`` instance containing the hierarchy, root
        positions, joint angles, and frame frequency derived from the time
        column.

    Raises
    ------
    TypeError
        If *hier* is neither a list nor a dict.
    Exception
        If *df* columns do not satisfy naming or ordering requirements (see
        ``_check_df_columns``), or if *df* and *hier* are inconsistent (see
        ``_check_df_match_with_hier``).
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
    frames = frames.to_numpy()
    frame_frequency = 1/int(1/(time_series.to_numpy()[-1] / (len(time_series)-1)))

    num_joints = len([n for n in hier_list if not n.is_end_site()])
    root_pos = frames[:, :3].astype(np.float64)
    joint_angles = frames[:, 3:].reshape(frames.shape[0], num_joints, 3).astype(np.float64)

    return Bvh(nodes=hier_list, root_pos=root_pos, joint_angles=joint_angles,
               frame_frequency=frame_frequency)
