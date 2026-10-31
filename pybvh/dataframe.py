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

import math
from collections.abc import Hashable, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .bvh import Bvh, _motion_column_names
from .bvhnode import BvhNode
from .io import _rate_with_fewest_decimals
from .node_tree import _nodes_from_table_or_list

if TYPE_CHECKING:
    import pandas as pd


#: The window in which ``df_to_bvh`` looks for a table's rate: a measured
#: frame time is recovered as ``1 / rate`` when a rate lies within this
#: fraction of it. It stands in for the decimals a ``Frame Time`` prints,
#: which a table does not keep; ``df_to_bvh``'s Notes give the spans of a
#: 6-decimal table it serves. 5e-7 is half the gap between ``1 / 29.97``
#: and NTSC's ``1001 / 30000``, 1e-6 of the larger, so an NTSC table
#: measured within 5e-7 never holds ``1 / 29.97`` in its window: at 1e-6,
#: the gap itself, an NTSC table whose rounding raised its measured frame
#: time came back as ``1 / 29.97``, and one of another length kept its
#: measured value. For the same reason it is not
#: :data:`~pybvh.tools._SAME_RATE_RTOL`, 2e-6.
_RATE_RECOVERY_RTOL = 5e-7


_SUFFIX_RULE = (
    "pybvh labels the columns of a repeated node name X, X.1, X.2, ... in "
    "node order, the first keeping its name, as pandas labels repeated CSV "
    "headers"
)


def _frame_time_from_times(times: npt.NDArray[Any], label: Hashable) -> float:
    """The frame time the ``time`` column *times*, labelled *label*, means.

    A column of NaN on every row is an unset clip's and gives 0; a
    column whose first and last values are equal raises ``ValueError``.
    Otherwise the measured frame time goes through the reader's rule
    with :data:`_RATE_RECOVERY_RTOL` as the window, as
    :func:`df_to_bvh` states for the user.
    """
    # The table of an unset clip: Bvh.to_df_dict gives it NaN times.
    if np.isnan(times).all():
        return 0.0
    measured = float((times[-1] - times[0]) / (len(times) - 1))
    if measured == 0:
        raise ValueError(
            f"The {label!r} column gives no frame time: its first and last values "
            f"are both {times[0]}. A clip without a frame time has NaN times, as "
            "Bvh.to_df_dict gives an unset clip."
        )
    # Only a positive finite value has a rate to recover; a negative or
    # non-finite one is returned unchanged for the frame_time rule to reject.
    if not (math.isfinite(measured) and measured > 0):
        return measured
    exact = Fraction(measured)
    rate = _rate_with_fewest_decimals(
        exact * (1 - Fraction(_RATE_RECOVERY_RTOL)), exact * (1 + Fraction(_RATE_RECOVERY_RTOL))
    )
    if rate is None:
        return measured
    return 1 / float(rate)


def _time_label(df: pd.DataFrame) -> Hashable:
    """The label of *df*'s ``time`` column.

    The first label spelling ``time`` in any case is taken.
    """
    for label in df.columns:
        if str(label).lower() == "time":
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
    expected = _motion_column_names(nodes, "euler")
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
        problems.append(f"df is missing columns the hierarchy expects: {missing}")
    if ambiguous:
        problems.append(f"df lists expected columns more than once: {ambiguous}")
    if repeated:
        problems.append(f"df has repeated column labels {repeated}")
    if repeated or hierarchy_repeats_a_name:
        problems.append(_SUFFIX_RULE)
    raise ValueError(". ".join(problems))


def df_to_bvh(hier: Sequence[BvhNode] | Sequence[Mapping[str, Any]], df: pd.DataFrame) -> Bvh:
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
          taken over as :class:`Bvh` takes the nodes it is given: the
          clip holds those node objects and freezes them, so a node
          edited afterwards raises; nodes another clip owns are frozen
          already and are shared. A call that raises leaves the list
          editable. Pass ``nodes_to_table(nodes)`` to keep it editable.

        A table and a list are told apart by the type of the first
        element. A table is data: :func:`~pybvh.nodes_from_table` builds
        the clip's own nodes from it. Either raises ``ValueError`` when
        it is not one tree in depth-first order, a list's nodes wired
        both ways (each in its parent's ``children``, with that parent
        as its ``parent``). Channel orders come from the skeleton and
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
        :func:`~pybvh.nodes_from_table` for a table, :class:`Bvh` for a
        list); if *df* has no ``time``
        column, lacks a column the hierarchy expects (the message lists
        every missing label and, when *df* has repeated labels, states
        how repeated node names are labelled) or lists an expected label
        more than once; if *df* has fewer than two rows, if the first
        and last values of its ``time`` column are equal (the message
        names the column), or if the frame time derived from it is not
        finite.

    Notes
    -----
    The DataFrame's ``_rot`` columns are in **degrees** — the human-readable convention used by :meth:`Bvh.to_df_dict` output. ``df_to_bvh`` converts them to the radians held on :attr:`Bvh.joint_angles`; feed this function degrees even though the rest of the pybvh API works in radians.

    The frame time is the elapsed time of the ``time`` column divided
    by its number of intervals, so a first timestamp other than zero
    does not matter and uneven intervals are averaged rather than
    refused. A ``time`` column saved at a few decimals leaves that
    frame time a little off its clip's rate, so the rate is recovered
    by the rule :func:`~pybvh.read_bvh_file` applies to a truncated
    ``Frame Time``, with a relative 5e-7 standing in for the decimals
    a file prints: the frame time is ``1 / rate`` for the rate with the
    fewest decimals, from 0 to 3, whose frame time lies within 5e-7 of
    the measured one, relative, when it is the only rate with that
    many decimals; when several fit, or none, the measured value is
    kept. A column saved at 6 decimals measures the frame time within
    5e-7 when it spans two seconds or more, or one second when its
    first time is 0, as :meth:`Bvh.to_df_dict` writes it: only the last
    time then carries the rounding. A 30, 120 or 29.97 fps clip that
    long comes back at exactly ``1 / 30``, ``1 / 120`` or
    ``1 / 29.97``; a shorter one, or fewer decimals, may keep the
    measured value. NTSC's exact ``1001 / 30000`` has no rate with 3
    decimals, so a clip at that frame time comes back from a table
    saved at 6 decimals at its measured frame time: when that long,
    within 5e-7 of ``1001 / 30000``, so never as ``1 / 29.97``, which
    is 1.000001e-6 from it. A clip that goes out through
    :meth:`Bvh.to_df_dict` and back therefore keeps its skeleton
    exactly, its motion within float precision, the degrees conversion
    being the one step applied to the angles, and its rate within
    5e-7.

    A ``time`` column of NaN on every row reads back as an unset clip
    (``frame_time`` 0): it is what :meth:`Bvh.to_df_dict` gives a clip
    without a frame time. A column whose first and last values are
    equal is refused rather than read as unset, since it states times
    that do not move rather than no times.

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

    if isinstance(hier, Mapping):
        raise TypeError(
            "hier is a dict: the name-keyed hierarchy dict was removed in "
            "v0.10.0. Pass bvh.to_node_table(), a list with one entry per "
            "node in depth-first order and parent as the parent's index, or "
            "bvh.nodes. See the CHANGELOG for the migration."
        )
    nodes = _nodes_from_table_or_list(hier, "hier")
    time_label = _time_label(df)
    time_values = df[time_label].to_numpy(dtype=np.float64)
    frames = _motion_columns(nodes, df).to_numpy()

    if len(time_values) < 2:
        raise ValueError(
            f"df must contain at least 2 rows to derive the frame time "
            f"from the 'time' column (got {len(time_values)})"
        )
    frame_time = _frame_time_from_times(time_values, time_label)

    num_joints = len([n for n in nodes if not n.is_end_site()])
    root_pos = frames[:, :3].astype(np.float64)
    # DataFrame angles are in degrees (human-readable); pybvh holds radians.
    joint_angles_deg = frames[:, 3:].reshape(frames.shape[0], num_joints, 3).astype(np.float64)
    joint_angles = np.deg2rad(joint_angles_deg)

    return Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles, frame_time=frame_time)
