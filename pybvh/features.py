"""ML pipeline feature extraction for BVH motion data.

All functions are standalone and take a :class:`~pybvh.bvh.Bvh` object as
their first argument.  Thin wrapper methods on the ``Bvh`` class delegate
to these functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .bvh import Bvh
from . import rotations
from .tools import get_up_axis_index


# ----------------------------------------------------------------
#  Joint velocities & accelerations
# ----------------------------------------------------------------

def get_joint_velocities(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute per-joint position velocities via finite differences.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        Ignored if *coords* is provided.
    in_frames : bool, optional
        If True, return velocity in units/frame.
        If False (default), return velocity in units/second.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates. If None, computed
        internally via :meth:`Bvh.get_spatial_coord`.

    Returns
    -------
    ndarray, shape (F-1, N, 3)
        Velocity of each node between consecutive frames.

    Raises
    ------
    ValueError
        If fewer than 2 frames, or ``frame_frequency == 0``
        when ``in_frames=False``.
    """
    if bvh.frame_count < 2:
        raise ValueError(
            "At least 2 frames are required to compute velocities.")
    if not in_frames and bvh.frame_frequency == 0:
        raise ValueError(
            "frame_frequency is 0; cannot compute per-second velocity. "
            "Use in_frames=True for per-frame velocity.")

    if coords is None:
        coords = bvh.get_spatial_coord(centered=centered)

    vel = coords[1:] - coords[:-1]  # (F-1, N, 3)

    if not in_frames:
        vel = vel / bvh.frame_frequency

    return vel


def get_joint_accelerations(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute per-joint position accelerations via second-order finite differences.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        Ignored if *coords* is provided.
    in_frames : bool, optional
        If True, return acceleration in units/frame^2.
        If False (default), return in units/second^2.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates. If None, computed
        internally via :meth:`Bvh.get_spatial_coord`.

    Returns
    -------
    ndarray, shape (F-2, N, 3)
        Acceleration of each node.

    Raises
    ------
    ValueError
        If fewer than 3 frames, or ``frame_frequency == 0``
        when ``in_frames=False``.
    """
    if bvh.frame_count < 3:
        raise ValueError(
            "At least 3 frames are required to compute accelerations.")
    if not in_frames and bvh.frame_frequency == 0:
        raise ValueError(
            "frame_frequency is 0; cannot compute per-second acceleration. "
            "Use in_frames=True for per-frame acceleration.")

    vel = get_joint_velocities(bvh, centered=centered, in_frames=True, coords=coords)
    acc = vel[1:] - vel[:-1]  # (F-2, N, 3)

    if not in_frames:
        acc = acc / (bvh.frame_frequency ** 2)

    return acc


# ----------------------------------------------------------------
#  Angular velocities
# ----------------------------------------------------------------

def get_angular_velocities(
    bvh: Bvh,
    in_frames: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute per-joint angular velocities via rotation matrix log map.

    For each consecutive frame pair, computes the relative rotation
    ``R_rel = R_t^T @ R_{t+1}`` and converts to axis-angle, giving
    an angular velocity vector whose norm is the rotation angle.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    in_frames : bool, optional
        If True, return angular velocity in radians/frame.
        If False (default), return in radians/second.

    Returns
    -------
    ndarray, shape (F-1, J, 3)
        Angular velocity vector per joint per frame transition.
        The direction is the rotation axis; the magnitude is the
        rotation angle (in radians or radians/second).

    Raises
    ------
    ValueError
        If fewer than 2 frames, or ``frame_frequency == 0``
        when ``in_frames=False``.
    """
    if bvh.frame_count < 2:
        raise ValueError(
            "At least 2 frames are required to compute angular velocities.")
    if not in_frames and bvh.frame_frequency == 0:
        raise ValueError(
            "frame_frequency is 0; cannot compute per-second angular velocity. "
            "Use in_frames=True for per-frame angular velocity.")

    _, joint_rotmats, _ = bvh.get_frames_as_rotmat()  # (F, J, 3, 3)

    # R_rel = R_t^T @ R_{t+1}  for each consecutive pair
    R_t = joint_rotmats[:-1]      # (F-1, J, 3, 3)
    R_t1 = joint_rotmats[1:]      # (F-1, J, 3, 3)
    R_rel = np.einsum('...ji,...jk->...ik', R_t, R_t1)  # transpose + matmul

    ang_vel = rotations.rotmat_to_axisangle(R_rel)  # (F-1, J, 3)

    if not in_frames:
        ang_vel = ang_vel / bvh.frame_frequency

    return ang_vel


# ----------------------------------------------------------------
#  Root-relative positions
# ----------------------------------------------------------------

def get_root_relative_positions(
    bvh: Bvh,
    centered: str = "world",
    coords: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute joint positions relative to the root at each frame.

    Unlike ``centered="skeleton"`` which places the root at the
    world origin for all frames, this subtracts the root's position
    per frame, preserving the relative pose of the skeleton.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        Ignored if *coords* is provided.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates.

    Returns
    -------
    ndarray, shape (F, N, 3)
        Positions of all nodes relative to the root each frame.
        The root node (index 0) will be ``(0, 0, 0)`` in every frame.
    """
    if coords is None:
        coords = bvh.get_spatial_coord(centered=centered)

    return coords - coords[:, 0:1, :]  # broadcast root position


# ----------------------------------------------------------------
#  Root trajectory
# ----------------------------------------------------------------

def get_root_trajectory(
    bvh: Bvh,
    up_axis: str | None = None,
) -> npt.NDArray[np.float64]:
    """Extract root trajectory features commonly used in motion ML.

    Returns the root's ground-plane position, heading angle, and
    their velocities — the standard root representation in
    HumanML3D-style pipelines.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    up_axis : str or None, optional
        Signed axis string (e.g. ``'+y'``, ``'+z'``). If None,
        auto-detected from the rest pose.

    Returns
    -------
    ndarray, shape (F, 4)
        Columns: ``[ground_pos_a, ground_pos_b, heading_sin, heading_cos]``
        where a and b are the two ground-plane axes (non-up axes
        in order x, y, z with up removed).

    Notes
    -----
    The heading angle is extracted from the root joint's rotation
    matrix projected onto the ground plane. The heading is defined
    as the rotation around the up axis.
    """
    # Determine up axis
    if up_axis is None:
        rest_pose: npt.NDArray[np.float64] = bvh.get_rest_pose(mode='coordinates')  # type: ignore[assignment]
        up_idx = get_up_axis_index(bvh, rest_pose)
    else:
        up_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis[1]]

    # Ground-plane axes (the two non-up axes, in order)
    ground_axes = [i for i in range(3) if i != up_idx]

    # Root ground-plane position: (F, 2)
    ground_pos = bvh.root_pos[:, ground_axes]

    # Root heading: extract from root rotation matrix
    root_joint = bvh.nodes[0]
    root_angles = bvh.joint_angles[:, 0]  # (F, 3) degrees
    root_order = root_joint.rot_channels  # type: ignore[attr-defined]
    R_root = rotations.euler_to_rotmat(root_angles, root_order, degrees=True)  # (F, 3, 3)

    # Heading = rotation around up axis
    # Project the forward direction through the rotation matrix
    # Use the column of R corresponding to the first ground axis
    # The heading angle is atan2 of the projected forward direction
    fwd_idx = ground_axes[0]
    # Forward vector after rotation: R @ e_fwd, projected onto ground plane
    fwd_rotated_a = R_root[:, ground_axes[0], fwd_idx]  # component along first ground axis
    fwd_rotated_b = R_root[:, ground_axes[1], fwd_idx]  # component along second ground axis
    heading = np.arctan2(fwd_rotated_b, fwd_rotated_a)  # (F,)

    return np.column_stack([
        ground_pos,           # (F, 2)
        np.sin(heading),      # (F,)
        np.cos(heading),      # (F,)
    ])


# ----------------------------------------------------------------
#  Foot contacts
# ----------------------------------------------------------------

def get_foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    method: str = "velocity",
    threshold: float | None = None,
    centered: str = "world",
    coords: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Detect binary foot contact labels per frame.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str or None, optional
        Joint names to use as foot markers. If None,
        auto-detects by searching for joints with "foot" or
        "toe" in the name (case-insensitive).
    method : str, optional
        Detection method: ``"velocity"`` (default) thresholds
        the speed of the foot joints, ``"height"`` thresholds
        the foot joint height above the ground plane.
    threshold : float or None, optional
        Detection threshold. If None, defaults to ``0.5``
        for ``"velocity"`` (units/frame) or auto-calibrated
        for ``"height"`` (5th percentile of foot heights).
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        Ignored if *coords* is provided.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates.

    Returns
    -------
    ndarray, shape (F, num_foot_joints)
        Binary contact labels (1.0 = contact, 0.0 = no contact).
        For ``"velocity"`` method, the first frame is always 1.0
        since velocity is undefined.

    Raises
    ------
    ValueError
        If no foot joints are found or specified, or if method
        is unknown.
    """
    if method not in ("velocity", "height"):
        raise ValueError(
            f"Unknown method '{method}'. Choose 'velocity' or 'height'.")

    if coords is None:
        coords = bvh.get_spatial_coord(centered=centered)

    # Auto-detect foot joints
    if foot_joints is None:
        foot_joints = []
        for node in bvh.nodes:
            name_lower = node.name.lower()
            if any(kw in name_lower for kw in ("foot", "toe")):
                if not node.is_end_site():
                    foot_joints.append(node.name)
        if not foot_joints:
            raise ValueError(
                "Could not auto-detect foot joints. Please provide "
                "foot_joints explicitly (e.g. ['LeftFoot', 'RightFoot']).")

    # Get indices into the spatial coord array
    foot_indices = []
    for name in foot_joints:
        if name not in bvh.node_index:
            raise ValueError(f"Joint '{name}' not found in skeleton.")
        foot_indices.append(bvh.node_index[name])

    foot_coords = coords[:, foot_indices, :]  # (F, num_feet, 3)

    if method == "velocity":
        if threshold is None:
            threshold = 0.5  # units/frame

        if bvh.frame_count < 2:
            return np.ones((bvh.frame_count, len(foot_joints)),
                           dtype=np.float64)

        foot_vel = foot_coords[1:] - foot_coords[:-1]  # (F-1, num_feet, 3)
        speed = np.linalg.norm(foot_vel, axis=-1)  # (F-1, num_feet)
        contacts = (speed < threshold).astype(np.float64)

        # Prepend first frame as contact (no velocity info)
        first_frame = np.ones((1, len(foot_joints)), dtype=np.float64)
        contacts = np.concatenate([first_frame, contacts], axis=0)

    else:  # height method
        rest_pose_arr: npt.NDArray[np.float64] = bvh.get_rest_pose(mode='coordinates')  # type: ignore[assignment]
        up_idx = get_up_axis_index(bvh, rest_pose_arr)
        foot_heights = foot_coords[:, :, up_idx]  # (F, num_feet)

        if threshold is None:
            # Auto-calibrate: 5th percentile of foot heights
            threshold = float(np.percentile(foot_heights, 5)) + 0.5

        contacts = (foot_heights < threshold).astype(np.float64)

    return contacts


# ----------------------------------------------------------------
#  Feature array export
# ----------------------------------------------------------------

def to_feature_array(
    bvh: Bvh,
    representation: str = "6d",
    include_root_pos: bool = True,
    include_velocities: bool = False,
    include_foot_contacts: bool = False,
    centered: str = "world",
    foot_joints: list[str] | None = None,
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
        ``'quaternion'``, ``'axisangle'``, or ``'rotmat'``.
    include_root_pos : bool, optional
        If True (default), include root position (3 columns).
    include_velocities : bool, optional
        If True, include joint velocity features. Reduces frame
        count by 1 (first frame has no velocity).
    include_foot_contacts : bool, optional
        If True, include foot contact labels.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
    foot_joints : list of str or None, optional
        Foot joints for contact detection. Only used when
        ``include_foot_contacts=True``.

    Returns
    -------
    ndarray, shape (F, D) or (F-1, D)
        Flat feature array. Frame count is F-1 when velocities
        are included (first frame is dropped for alignment).

    Raises
    ------
    ValueError
        If representation is unknown.
    """
    valid_reps = {"euler", "6d", "quaternion", "axisangle", "rotmat"}
    if representation not in valid_reps:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {sorted(valid_reps)}.")

    # Compute spatial coords once (shared by velocities and contacts)
    coords = None
    if include_velocities or include_foot_contacts:
        coords = bvh.get_spatial_coord(centered=centered)

    parts: list[npt.NDArray[np.float64]] = []

    # Root position
    if include_root_pos:
        parts.append(bvh.root_pos)

    # Joint rotations
    if representation == "euler":
        rot = bvh.joint_angles.reshape(bvh.frame_count, -1)
    elif representation == "6d":
        _, rot_raw, _ = bvh.get_frames_as_6d()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "quaternion":
        _, rot_raw, _ = bvh.get_frames_as_quaternion()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "axisangle":
        _, rot_raw, _ = bvh.get_frames_as_axisangle()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    else:  # rotmat
        _, rot_raw, _ = bvh.get_frames_as_rotmat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    parts.append(rot)

    # Determine if we need to truncate for velocity alignment
    has_velocity_trim = include_velocities and bvh.frame_count >= 2

    # Velocities — call directly (same module)
    if include_velocities:
        vel = get_joint_velocities(
            bvh, centered=centered, in_frames=True, coords=coords)
        # vel is (F-1, N, 3), flatten to (F-1, N*3)
        vel_flat = vel.reshape(vel.shape[0], -1)
        parts.append(vel_flat)

    # Foot contacts — call directly (same module)
    if include_foot_contacts:
        contacts = get_foot_contacts(
            bvh, foot_joints=foot_joints, centered=centered, coords=coords)
        parts.append(contacts)

    # Align frames: if velocities are included, trim all to F-1
    if has_velocity_trim:
        aligned = []
        for p in parts:
            if p.shape[0] == bvh.frame_count:
                aligned.append(p[1:])  # drop first frame
            else:
                aligned.append(p)      # already F-1
        parts = aligned

    return np.concatenate(parts, axis=1)
