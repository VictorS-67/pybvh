from __future__ import annotations

import re
import numpy as np
from typing import Any, Mapping, Sequence, TYPE_CHECKING, cast

from .bvh import Bvh
from .bvhnode import BvhNode, BvhJoint, BvhRoot
from .node_tree import nodes_from_table, nodes_to_table
from .io import _snap_frame_time

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
    # (axis letters are uppercase by convention — see Bvh.to_df_dict output)
    valid_pattern = r'.+_[XYZ]_(pos|rot)'
    for col_name in new_df.columns:
        pattern_ok = (re.fullmatch(valid_pattern, col_name) != None)
        if not pattern_ok:
            new_df = new_df.drop(col_name, axis=1) #that includes dropping the time col

    if len(new_df.columns) == 0:
        raise ValueError("No column found following the naming convention 'name_ax_pos' or 'name_ax_rot'")

    #check that root pos and rot appear first
    # rsplit from the right: joint names may themselves contain underscores
    # (e.g. 'Left_Hip_X_rot' -> joint 'Left_Hip', axis 'X', kind 'rot')
    root_name = new_df.columns[0].rsplit('_', 2)[0]
    root_er = 'The first rotational data appearing in the DataFrane should be 3 columns for the root position followed by 3 columns for the root rotation'
    for col_name in new_df.columns[0:3]:
        #those should be the root pos
        col_joint_name, ax, rotpos = col_name.rsplit('_', 2)
        if col_joint_name != root_name or rotpos != 'pos' :
            raise ValueError(root_er)
    for col_name in new_df.columns[3:6]:
        #those should be the root rot
        col_joint_name, ax, rotpos = col_name.rsplit('_', 2)
        if col_joint_name != root_name or rotpos != 'rot' :
            raise ValueError(root_er)

    #check if there is a time column in the original df, and add it back into the df
    has_time = False
    time_col: pd.Series | None = None  # type: ignore[type-arg]
    for col_name in df.columns:
        if col_name.lower() == 'time':
            has_time = True
            time_col = df[col_name]
            break
    if not has_time or time_col is None:
        raise ValueError("No 'time' column found in the DataFrame")

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
        if node.is_end_site():
            continue
        correct_col_list.extend(node_to_names(node))

    if list(df.columns) == correct_col_list:
        return hier, df

    # If the list we created doesn't match witht the df columns,
    # create a new df with correctly ordered column
    try:
        df = df[correct_col_list]
    except KeyError as e:
        missing = set(correct_col_list) - set(df.columns)
        raise ValueError(
            f"DataFrame is missing columns required by the hierarchy: "
            f"{sorted(missing) if missing else list(e.args)}") from e

    return hier, df

def _nodes_from_hier(
        hier: Sequence[BvhNode] | Sequence[Mapping[str, Any]]) -> list[BvhNode]:
    """Build fresh nodes from *hier*, a node table or a node list.

    The form is decided by the type of the first element: a ``BvhNode``
    means a node list, read through ``nodes_to_table``; a ``Mapping``
    means a node table. Both go through ``nodes_from_table``, which
    validates the tree and is the one place a node tree is built.
    """
    if isinstance(hier, Mapping):
        raise TypeError(
            "hier is a dict: the name-keyed hierarchy dict was removed in "
            "v0.10.0. Pass bvh.to_node_table(), a list with one entry per "
            "node in depth-first order and parent as the parent's index, or "
            "bvh.nodes. See the CHANGELOG for the migration.")
    if len(hier) == 0:
        raise ValueError(
            "hier is empty: pass bvh.to_node_table() or bvh.nodes, a node "
            "table or a node list with the root first.")
    # The first element decides the form; the casts state that choice,
    # which narrowing a union of sequences by one element cannot.
    first = hier[0]
    if isinstance(first, BvhNode):
        return nodes_from_table(nodes_to_table(cast(Sequence[BvhNode], hier)))
    if isinstance(first, Mapping):
        return nodes_from_table(cast(Sequence[Mapping[str, Any]], hier))
    raise TypeError(
        f"hier[0] is a {type(first).__name__}; hier must be a node table "
        f"(bvh.to_node_table(), one dict per node) or a node list "
        f"(bvh.nodes).")


def df_to_bvh(hier: Sequence[BvhNode] | Sequence[Mapping[str, Any]],
              df: pd.DataFrame) -> Bvh:
    """Create a Bvh object from a skeleton and a motion DataFrame.

    Build a complete BVH representation by combining a skeleton with
    per-frame motion data stored in a pandas DataFrame. The skeleton
    decides the node order of the resulting ``Bvh``; the DataFrame's
    motion columns are matched to it by name.

    Parameters
    ----------
    hier : list of dict or list of BvhNode
        The skeleton, supplied as either:

        * A **node table**, as :meth:`Bvh.to_node_table` returns: one
          ``dict`` per node in depth-first order with ``name``,
          ``parent`` (the index of the parent's entry, ``None`` on the
          root), ``offset``, ``rot_channels`` on the root and joints and
          ``pos_channels`` on the root; an entry without ``rot_channels``
          is an end site. :func:`~pybvh.nodes_from_table` defines the
          format, its two leniencies and what it rejects.
        * A **node list** of ``BvhRoot``, ``BvhJoint`` and ``BvhEndSite``
          objects in depth-first order, such as :attr:`Bvh.nodes`. It is
          read through :func:`~pybvh.nodes_to_table`, so only each node's
          ``parent`` is consulted, and nothing is shared with the list
          given.

        A table and a list are told apart by the type of the first
        element, and both build their nodes through
        :func:`~pybvh.nodes_from_table`, which returns fresh nodes and
        raises ``ValueError`` for a table or a list that is not one tree
        in depth-first order. Channel orders come from the skeleton and
        are never inferred from *df*: a table entry states its
        ``rot_channels`` or it is an end site. The name-keyed hierarchy
        dict of earlier releases is refused with ``TypeError``.
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
        If *hier* is the name-keyed hierarchy dict of earlier releases, or
        neither a node table nor a node list.
    ValueError
        If *hier* is empty, or is not one tree in depth-first order (see
        :func:`~pybvh.nodes_from_table`).
    Exception
        If *df* columns do not satisfy naming or ordering requirements (see
        ``_check_df_columns``), or if *df* and *hier* are inconsistent (see
        ``_check_df_match_with_hier``).

    Notes
    -----
    The DataFrame's ``_rot`` columns are in **degrees** — the human-readable convention used by :meth:`Bvh.to_df_dict` output. ``df_to_bvh`` converts them to the radians held on :attr:`Bvh.joint_angles`; feed this function degrees even though the rest of the pybvh API works in radians.
    """

    df = _check_df_columns(df) # this creates a copy of the df
    nodes = _nodes_from_hier(hier)
    nodes, df = _check_df_match_with_hier(nodes, df)

    time_series = df['time']
    frames = df.drop(['time'], axis=1)
    frames = frames.to_numpy()
    time_values = time_series.to_numpy()
    if len(time_values) < 2:
        raise ValueError(
            f"df must contain at least 2 rows to derive the frame time "
            f"from the 'time' column (got {len(time_values)})")
    # Elapsed time over frame intervals — robust to a nonzero first timestamp.
    frame_time = float((time_values[-1] - time_values[0]) / (len(time_values) - 1))
    frame_time = _snap_frame_time(frame_time)

    num_joints = len([n for n in nodes if not n.is_end_site()])
    root_pos = frames[:, :3].astype(np.float64)
    # DataFrame angles are in degrees (human-readable); pybvh holds radians.
    joint_angles_deg = frames[:, 3:].reshape(frames.shape[0], num_joints, 3).astype(np.float64)
    joint_angles = np.deg2rad(joint_angles_deg)

    return Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
               frame_time=frame_time)
