"""Feature export for ML pipelines.

Composes the per-frame motion descriptors from :mod:`pybvh.analysis`
(rotations, root position, velocities, foot contacts) into a single flat
``(F, D)`` array, plus the column-layout helper that describes it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .analysis import _validate_stencil_pad, joint_velocities
from .analysis import foot_contacts as _foot_contacts
from .bvh import Bvh, _check_centered
from .bvhnode import NodeSelector
from .rotations import REPRESENTATION_CHANNELS as _REPRESENTATION_WIDTHS
from .spatial_coord import _ground_plane_offset


def feature_array_layout(
    *,
    num_joints: int,
    num_feet: int = 0,
    num_translating_joints: int | None = None,
    representation: str = "6d",
    include_root_pos: bool = True,
    include_joint_translations: bool = False,
    include_velocities: bool = False,
    include_foot_contacts: bool = False,
) -> dict[str, slice]:
    """Column layout of the array returned by :func:`to_feature_array`.

    Returns a dict mapping block name to column slice so callers can
    write ``feat[:, layout['rotations']]`` without counting columns.
    Pure function — no :class:`~pybvh.bvh.Bvh` required; useful for
    model-shape setup before any data is loaded.

    Parameters
    ----------
    num_joints : int
        Number of joints (excluding end sites). Used for both the
        rotation and velocity blocks: velocities are per-joint (not
        per-node), aligning with the rotation block's joint axis.
    num_feet : int, optional
        Number of feet for contact detection.  Required (>0)
        when ``include_foot_contacts=True``.
    num_translating_joints : int, optional
        Number of translating joints, ``P`` of
        :attr:`Bvh.joint_translations <pybvh.bvh.Bvh.joint_translations>`.
        Required when ``include_joint_translations=True``, unused
        otherwise; with ``0`` the block is empty.
    representation : str, optional
        Rotation representation: ``'euler'``, ``'axisangle'`` (3 values
        per joint), ``'quat'`` (4), ``'6d'`` (6, default),
        ``'rotmat'`` (9).
    include_root_pos, include_joint_translations, include_velocities, include_foot_contacts : bool
        Mirror the flags of :func:`to_feature_array`.

    Returns
    -------
    dict
        ``{block_name: slice}`` with keys drawn from
        ``{"root_pos", "joint_translations", "rotations", "velocities",
        "foot_contacts"}`` depending on the flags, in that column order.

    Notes
    -----
    Within the ``rotations`` and ``velocities`` blocks joints are in
    **skeleton hierarchy order** — the file's declaration order, i.e.
    ``Bvh.joint_names``, end sites excluded — not alphabetical. For
    ``representation='euler'`` the three columns per joint follow *that
    joint's own* channel order from the file (which can differ per
    joint and between files), not a fixed XYZ; every other
    representation is order-independent. ``'rotmat'`` flattens each 3×3
    row-major (C order).

    Raises
    ------
    ValueError
        If ``representation`` is unknown, if
        ``include_foot_contacts=True`` but ``num_feet == 0``, or if
        ``include_joint_translations=True`` but
        ``num_translating_joints`` is ``None``.
    """
    if representation not in _REPRESENTATION_WIDTHS:
        raise ValueError(
            f"Unknown representation {representation!r}; "
            f"must be one of {sorted(_REPRESENTATION_WIDTHS)}."
        )
    if include_foot_contacts and num_feet <= 0:
        raise ValueError("num_feet must be > 0 when include_foot_contacts=True")
    if include_joint_translations and num_translating_joints is None:
        raise ValueError(
            "num_translating_joints must be given when include_joint_translations=True"
        )
    K = _REPRESENTATION_WIDTHS[representation]

    layout: dict[str, slice] = {}
    cursor = 0
    if include_root_pos:
        layout["root_pos"] = slice(cursor, cursor + 3)
        cursor += 3
    if include_joint_translations:
        assert num_translating_joints is not None  # refused above
        translation_width = num_translating_joints * 3
        layout["joint_translations"] = slice(cursor, cursor + translation_width)
        cursor += translation_width
    rot_width = num_joints * K
    layout["rotations"] = slice(cursor, cursor + rot_width)
    cursor += rot_width
    if include_velocities:
        vel_width = num_joints * 3
        layout["velocities"] = slice(cursor, cursor + vel_width)
        cursor += vel_width
    if include_foot_contacts:
        layout["foot_contacts"] = slice(cursor, cursor + num_feet)
        cursor += num_feet
    return layout


def to_feature_array(
    bvh: Bvh,
    representation: str = "6d",
    include_root_pos: bool = True,
    include_velocities: bool = False,
    include_foot_contacts: bool = False,
    centered: str = "world",
    feet: Sequence[NodeSelector] | None = None,
    stencil: str = "central",
    pad: str = "edge",
    include_joint_translations: bool = False,
) -> npt.NDArray[np.float64]:
    """Export motion as a single flat feature array for ML pipelines.

    Composes root position, joint translations, joint rotations,
    velocities, and foot contacts into a single ``(F, D)`` array ready
    for model input.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    representation : str, optional
        Rotation representation: ``'euler'``, ``'6d'`` (default),
        ``'quat'``, ``'axisangle'``, or ``'rotmat'`` (9 values
        per joint as a flattened 3×3). ``'quat'`` inherits the
        per-frame canonical form (``w >= 0``) of :meth:`Bvh.to_quat`,
        which is **not** temporally continuous — a joint passing
        through 180° flips sign between frames. Apply
        :func:`pybvh.rotations.quat_unwrap` to the array yourself if
        your model is sensitive to that; ``'6d'`` (the default) has no
        such discontinuity, which is the representation's whole point.
    include_root_pos : bool, optional
        If True (default), include root position (3 columns).
    include_velocities : bool, optional
        If True, include joint velocity features, in units per
        **frame** — deliberately not :func:`joint_velocities`' own
        units-per-second default, because a feature array is consumed
        per-frame by a model and a per-frame delta keeps the block on
        the same time base as the rest of the array. Multiply by
        ``1 / bvh.frame_time`` to recover units/second, and note the
        two disagree by exactly that factor if you cross-check this
        block against a direct ``joint_velocities(bvh)`` call.
    include_foot_contacts : bool, optional
        If True, include foot contact labels.  Contacts are always
        detected in world frame (see :func:`pybvh.analysis.foot_contacts`),
        whatever ``centered`` says.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
    feet : sequence of str or BvhNode, or None, optional
        The feet for contact detection, each a node name or one of
        ``bvh.nodes`` (see :data:`~pybvh.NodeSelector`), resolved as
        :func:`pybvh.analysis.foot_contacts` resolves its ``feet``; the
        nodes :func:`pybvh.analysis.auto_detect_feet` returns if None.
        Only used when ``include_foot_contacts=True``.
    stencil, pad : optional
        Only affect output when ``include_velocities=True``.  Same
        semantics as :func:`joint_velocities`.  The root-position,
        rotation, and foot-contact blocks are trimmed in time so all
        blocks align with the velocity shape: ``pad="none"`` drops one
        frame from each end for ``stencil="central"``, and the last
        frame for ``stencil="forward"`` (a forward difference labels
        frame ``i``, so the final frame has no velocity).
    include_joint_translations : bool, optional
        If True, include :attr:`Bvh.joint_translations
        <pybvh.bvh.Bvh.joint_translations>` flattened per frame
        (``3 * P`` columns, after the root position): per translating
        joint, in node order, its ``X Y Z`` position relative to its
        parent, the file's values. ``centered`` does not apply to them.
        A clip with no translating joint (``P = 0``) gets an empty
        block. Default False: the array is then the same whether or not
        the clip has translating joints.

    Returns
    -------
    ndarray, shape (F, D), (F-1, D), or (F-2, D)
        See :func:`feature_array_layout` for the column layout.
        Leading dimension depends on ``include_velocities`` and the
        ``stencil`` × ``pad`` combination.

    Raises
    ------
    ValueError
        If ``representation`` or ``centered`` is unknown, or ``stencil`` /
        ``pad`` is invalid. With ``include_foot_contacts=True``, also where
        :func:`pybvh.analysis.foot_contacts` raises it for ``feet``.
    TypeError
        With ``include_foot_contacts=True``, where
        :func:`pybvh.analysis.foot_contacts` raises it for ``feet``: a
        bare ``str`` or ``BvhNode`` in place of a sequence of them, or an
        entry that is neither.
    """
    if representation not in _REPRESENTATION_WIDTHS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {sorted(_REPRESENTATION_WIDTHS)}."
        )
    _check_centered(centered)
    if include_velocities:
        _validate_stencil_pad(stencil, pad)

    parts: list[npt.NDArray[np.float64]] = []

    # Root position — apply same ``centered`` semantics as node_positions.
    if include_root_pos:
        if centered == "skeleton":
            parts.append(np.zeros_like(bvh.root_pos))
        elif centered == "first":
            # Ground-plane centering: first-frame root subtracted in the
            # non-up axes only, matching node_positions(centered="first").
            parts.append(bvh.root_pos - _ground_plane_offset(bvh.root_pos[0], bvh.world_up))
        else:  # "world"
            parts.append(bvh.root_pos)

    # Joint translations — each relative to its parent, so ``centered``
    # leaves them alone.
    if include_joint_translations:
        parts.append(bvh.joint_translations.reshape(bvh.frame_count, -1))

    # Joint rotations
    if representation == "euler":
        rot = bvh.joint_angles.reshape(bvh.frame_count, -1)
    elif representation == "6d":
        rot_raw = bvh.to_6d()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "quat":
        rot_raw = bvh.to_quat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "axisangle":
        rot_raw = bvh.to_axisangle()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    else:  # rotmat
        rot_raw = bvh.to_rotmat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    parts.append(rot)

    # Velocities — call directly (same module)
    vel_shape: int | None = None
    if include_velocities:
        vel = joint_velocities(bvh, centered=centered, in_frames=True, stencil=stencil, pad=pad)
        vel_flat = vel.reshape(vel.shape[0], -1)
        parts.append(vel_flat)
        vel_shape = vel.shape[0]

    # Foot contacts — call directly (same module); always (F, num_feet).
    # foot_contacts reads the clip's world-frame FK (the Bvh cache) whatever
    # ``centered`` is: per-frame "skeleton" centering would distort the
    # foot-speed signal.
    if include_foot_contacts:
        contacts = _foot_contacts(bvh, feet=feet)
        parts.append(contacts)

    # Align frames: trim F-shaped blocks (root_pos, rot, contacts) to match
    # the velocity block's leading dimension.  The trim convention follows
    # the stencil: "central" cannot define both boundaries, so one frame is
    # dropped from each end; "forward" labels frame i with
    # (x[i+1] - x[i]) / dt, so the frame without a defined derivative is
    # the last one.
    if include_velocities and vel_shape is not None and vel_shape != bvh.frame_count:
        F = bvh.frame_count
        drop = F - vel_shape
        if stencil == "central":
            # the only case here is drop=2 (pad="none", central) → [1:-1]
            assert drop % 2 == 0, f"Unexpected central drop {drop}"
            slicer = slice(drop // 2, F - drop // 2)
        else:  # stencil == "forward"
            slicer = slice(0, F - drop)
        aligned = [p[slicer] if p.shape[0] == F else p for p in parts]
        parts = aligned

    return np.concatenate(parts, axis=1)
