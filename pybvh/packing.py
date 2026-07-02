"""Feature-array packing for ML pipelines.

Composes the per-frame motion descriptors from :mod:`pybvh.analysis`
(rotations, root position, velocities, foot contacts) into a single flat
``(F, D)`` array, plus the column-layout helper that describes it.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .bvh import Bvh
from .rotations import REPRESENTATION_CHANNELS as _REPRESENTATION_WIDTHS
from .spatial_coord import _ground_plane_offset
from .analysis import joint_velocities, foot_contacts, _validate_stencil_pad


def feature_array_layout(
    *,
    num_joints: int,
    num_feet: int = 0,
    representation: str = "6d",
    include_root_pos: bool = True,
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
        Number of foot joints for contact detection.  Required (>0)
        when ``include_foot_contacts=True``.
    representation : str, optional
        Rotation representation: ``'euler'``, ``'axisangle'`` (3 values
        per joint), ``'quaternion'`` (4), ``'6d'`` (6, default),
        ``'rotmat'`` (9).
    include_root_pos, include_velocities, include_foot_contacts : bool
        Mirror the flags of :func:`to_feature_array`.

    Returns
    -------
    dict
        ``{block_name: slice}`` with keys drawn from
        ``{"root_pos", "rotations", "velocities", "foot_contacts"}``
        depending on the flags.

    Raises
    ------
    ValueError
        If ``representation`` is unknown, or if
        ``include_foot_contacts=True`` but ``num_feet == 0``.
    """
    if representation not in _REPRESENTATION_WIDTHS:
        raise ValueError(
            f"Unknown representation {representation!r}; "
            f"must be one of {sorted(_REPRESENTATION_WIDTHS)}."
        )
    if include_foot_contacts and num_feet <= 0:
        raise ValueError(
            "num_feet must be > 0 when include_foot_contacts=True"
        )
    K = _REPRESENTATION_WIDTHS[representation]

    layout: dict[str, slice] = {}
    cursor = 0
    if include_root_pos:
        layout["root_pos"] = slice(cursor, cursor + 3)
        cursor += 3
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
    foot_joints: list[str] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Export motion as a single flat feature array for ML pipelines.

    Composes root position, joint rotations, velocities, and foot
    contacts into a single ``(F, D)`` array ready for model input.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    representation : str, optional
        Rotation representation: ``'euler'``, ``'6d'`` (default),
        ``'quaternion'``, ``'axisangle'``, or ``'rotmat'`` (9 values
        per joint as a flattened 3×3).
    include_root_pos : bool, optional
        If True (default), include root position (3 columns).
    include_velocities : bool, optional
        If True, include joint velocity features.
    include_foot_contacts : bool, optional
        If True, include foot contact labels.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
    foot_joints : list of str or None, optional
        Foot joints for contact detection. Only used when
        ``include_foot_contacts=True``.
    stencil, pad : optional
        Only affect output when ``include_velocities=True``.  Same
        semantics as :func:`joint_velocities`.  The root-position,
        rotation, and foot-contact blocks are trimmed in time so all
        blocks align with the velocity shape.

    Returns
    -------
    ndarray, shape (F, D), (F-1, D), or (F-2, D)
        See :func:`feature_array_layout` for the column layout.
        Leading dimension depends on ``include_velocities`` and the
        ``stencil`` × ``pad`` combination.

    Raises
    ------
    ValueError
        If ``representation`` is unknown, or ``stencil`` / ``pad`` is
        invalid.
    """
    if representation not in _REPRESENTATION_WIDTHS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {sorted(_REPRESENTATION_WIDTHS)}.")
    if include_velocities:
        _validate_stencil_pad(stencil, pad)

    # Compute spatial coords once (shared by velocities and contacts)
    coords = None
    if include_velocities or include_foot_contacts:
        coords = bvh.node_positions(centered=centered)

    parts: list[npt.NDArray[np.float64]] = []

    # Root position — apply same ``centered`` semantics as node_positions.
    if include_root_pos:
        if centered == "skeleton":
            parts.append(np.zeros_like(bvh.root_pos))
        elif centered == "first":
            # Ground-plane centering: first-frame root subtracted in the
            # non-up axes only, matching node_positions(centered="first").
            parts.append(
                bvh.root_pos - _ground_plane_offset(bvh.root_pos[0], bvh.world_up))
        else:  # "world"
            parts.append(bvh.root_pos)

    # Joint rotations
    if representation == "euler":
        rot = bvh.joint_angles.reshape(bvh.frame_count, -1)
    elif representation == "6d":
        _, rot_raw = bvh.to_6d()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "quaternion":
        _, rot_raw = bvh.to_quaternions()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "axisangle":
        _, rot_raw = bvh.to_axisangle()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    else:  # rotmat
        _, rot_raw = bvh.to_rotmat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    parts.append(rot)

    # Velocities — call directly (same module)
    vel_shape: int | None = None
    if include_velocities:
        vel = joint_velocities(
            bvh, centered=centered, in_frames=True, coords=coords,
            stencil=stencil, pad=pad)
        vel_flat = vel.reshape(vel.shape[0], -1)
        parts.append(vel_flat)
        vel_shape = vel.shape[0]

    # Foot contacts — call directly (same module); always (F, num_feet)
    if include_foot_contacts:
        contacts = foot_contacts(
            bvh, foot_joints=foot_joints, centered=centered, coords=coords)
        assert isinstance(contacts, np.ndarray)
        parts.append(contacts)

    # Align frames: trim F-shaped blocks (root_pos, rot, contacts) to match
    # the velocity block's leading dimension.  The trim convention depends
    # on stencil: "central" drops both boundaries symmetrically; "forward"
    # keeps the existing drop-first-frame convention for backward compat
    # with the pre-split API.
    if include_velocities and vel_shape is not None and vel_shape != bvh.frame_count:
        F = bvh.frame_count
        drop = F - vel_shape
        if stencil == "central":
            # Symmetric: drop (drop//2) from each side when drop is even;
            # the only case here is drop=2 (pad="none", central) → [1:-1]
            assert drop % 2 == 0, f"Unexpected central drop {drop}"
            left, right = drop // 2, drop // 2
            slicer = slice(left, F - right)
        else:  # stencil == "forward"
            # Existing convention: drop from the front
            slicer = slice(drop, F)
        aligned = [
            p[slicer] if p.shape[0] == F else p
            for p in parts
        ]
        parts = aligned

    return np.concatenate(parts, axis=1)

