"""Build a :class:`~pybvh.bvh.Bvh` from a pandas DataFrame of motion.

The inverse of :meth:`Bvh.to_df_dict <pybvh.bvh.Bvh.to_df_dict>` in
``'euler'`` mode: :func:`df_to_bvh` pairs a skeleton with a DataFrame
holding a ``time`` column in seconds and one column per channel, root
positions in the skeleton's length unit and joint rotations in
**degrees**, the unit of a file and of ``to_df_dict``, converted here to
the radians of ``joint_angles``. pandas is never imported at run time:
this module only reads the DataFrame it is given.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .bvh import Bvh, _motion_column_names
from .bvhnode import BvhNode
from .io import _snap_frame_time
from .node_tree import nodes_from_table, nodes_to_table

if TYPE_CHECKING:
    import pandas as pd


_SUFFIX_RULE = (
    "pybvh labels the columns of a repeated node name X, X.1, X.2, ... in "
    "node order, the first keeping its name, as pandas labels repeated CSV "
    "headers")


def _time_label(df: pd.DataFrame) -> Hashable:
    """The label of *df*'s ``time`` column.

    The first label spelling ``time`` in any case is taken.
    """
    for label in df.columns:
        if str(label).lower() == 'time':
            return label
    raise ValueError("No 'time' column found in the DataFrame")


def _motion_columns(nodes: Sequence[BvhNode], df: pd.DataFrame) -> pd.DataFrame:
    """*df*'s motion columns in the flat layout order of *nodes*.

    The labels expected are those ``Bvh.to_df_dict(mode='euler')`` gives
    *nodes*, unique by construction, and they are selected from *df* by
    name in that order: the columns of *df* may come in any order, and
    columns outside the set, ``time`` among them, are left out. A
    missing label raises ``ValueError`` listing every missing one, and
    a label listed twice raises as well, since each label must name one
    column. Both messages state the suffix rule when it can be the
    cause: when *nodes* repeats a joint name, so that a frame labelled
    without the rule lacks the ``.1`` columns, or when *df* repeats a
    label.
    """
    expected = _motion_column_names(nodes, 'euler')
    present = set(df.columns)
    repeated = df.columns[df.columns.duplicated()].unique().tolist()
    missing = [name for name in expected if name not in present]
    ambiguous = [name for name in expected if name in repeated]
    if not missing and not ambiguous:
        return df[expected]

    joint_names = [node.name for node in nodes if not node.is_end_site()]
    hierarchy_repeats_a_name = len(set(joint_names)) < len(joint_names)
    problems = []
    if missing:
        problems.append(
            f"df is missing columns the hierarchy expects: {missing}")
    if ambiguous:
        problems.append(
            f"df lists expected columns more than once: {ambiguous}")
    if repeated:
        problems.append(f"df has repeated column labels {repeated}")
    if repeated or hierarchy_repeats_a_name:
        problems.append(_SUFFIX_RULE)
    raise ValueError(". ".join(problems))


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
        Motion data in the form :meth:`Bvh.to_df_dict` gives with
        ``mode='euler'``: a ``time`` column, matched without regard to
        case (the first such column when several differ only by case),
        and the hierarchy's flat layout, ``<root>_<axis>_pos`` per
        position channel of the root then ``<joint>_<axis>_rot`` per
        rotation channel of each joint in node order, a repeated node
        name labelled ``.1``, ``.2``, ... as ``to_df_dict`` labels it.
        The labels expected are derived from *hier*, and the columns
        bind by name: their order in *df* is free, and columns outside
        the expected set are ignored. The ``_rot`` columns are in
        degrees (see Notes).

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
        :func:`~pybvh.nodes_from_table`); if *df* has no ``time``
        column, lacks a column the hierarchy expects (the message lists
        every missing label and, when *df* has repeated labels, states
        how repeated node names are labelled) or lists an expected label
        more than once; if *df* has fewer than two rows, or the frame
        time derived from its ``time`` column is not finite.

    Notes
    -----
    The DataFrame's ``_rot`` columns are in **degrees** — the human-readable convention used by :meth:`Bvh.to_df_dict` output. ``df_to_bvh`` converts them to the radians held on :attr:`Bvh.joint_angles`; feed this function degrees even though the rest of the pybvh API works in radians.

    The frame time is the elapsed time of the ``time`` column divided
    by its number of intervals, so a first timestamp other than zero
    does not matter and uneven intervals are averaged rather than
    refused, and it is snapped to an exact ``1 / N`` when within 0.01%
    of one, as :func:`read_bvh_file` snaps a truncated ``Frame Time``:
    any integer rate comes back exact, so a ``time`` column written at
    six decimals for a 30 fps clip gives ``1 / 30`` exactly, and a
    non-integer rate such as 23.976 fps is kept as measured. A clip
    that goes out through :meth:`Bvh.to_df_dict` and back therefore
    keeps its skeleton exactly and its motion within float precision,
    the degrees conversion being the one step applied to the angles.

    Columns bind by label, not by position. The alternative, reading
    the motion columns in the order they come, would accept a frame
    whose columns are in the file's order under any labels, and would
    read a reordered or mislabelled one wrong without an error; by-label
    binding reads a reordered frame right and refuses a mislabelled
    one. The labels are those of :meth:`Bvh.to_df_dict`, which labels a
    repeated node name ``.1``, ``.2``, ... in node order, the first
    keeping its name, the way ``pandas.read_csv`` labels repeated
    headers; raising on a repeated name instead would refuse every
    hand whose two fingertips are end sites in coordinates mode. The
    suffix is counted over the nodes a mode exports, the joints here,
    so a joint that is ``X.1`` in a coordinates export because an
    earlier end site shares its name is plain ``X`` in the euler frame
    this function reads. A coordinates export is one-way: positions do
    not determine the joint angles, and this function reads euler-mode
    frames only.
    """

    nodes = _nodes_from_hier(hier)
    time_values = df[_time_label(df)].to_numpy()
    frames = _motion_columns(nodes, df).to_numpy()

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
