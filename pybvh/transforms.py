"""Spatial augmentation transforms for BVH motion data.

**Bvh-level API** — All transforms operate on :class:`~pybvh.bvh.Bvh`
objects and follow the ``inplace=False`` convention: by default they
return a new object, leaving the original unchanged.

**NumPy-level API** — Lower-level functions (``mirror_angles``,
``rotate_angles_vertical``) accept raw arrays + minimal metadata for
users who work with pre-extracted arrays.
"""
from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from .bvh import Bvh
from .bvhnode import BvhNode, BvhJoint
from . import rotations
from .tools import (
    get_forw_up_axis, get_up_axis_index,
    rotX, rotY, rotZ,
)


# =========================================================================
# 6.5  Root Translation
# =========================================================================

@overload
def translate_root(
    bvh: Bvh, offset: npt.ArrayLike, *, inplace: Literal[True],
) -> None: ...
@overload
def translate_root(
    bvh: Bvh, offset: npt.ArrayLike, inplace: Literal[False] = ...,
) -> Bvh: ...
def translate_root(
    bvh: Bvh,
    offset: npt.ArrayLike,
    inplace: bool = False,
) -> Bvh | None:
    """Shift the root position by a constant 3-D offset.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    offset : array_like of shape (3,)
        Translation vector ``(dx, dy, dz)``.
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None
    """
    off = np.asarray(offset, dtype=np.float64)
    if off.shape != (3,):
        raise ValueError(f"offset must have shape (3,), got {off.shape}")

    target = bvh if inplace else bvh.copy()
    target.root_pos = target.root_pos + off  # broadcast (F,3)+(3,)
    if inplace:
        return None
    return target


def random_translate_root(
    bvh: Bvh,
    range_xyz: tuple[float, float] = (-100.0, 100.0),
    rng: np.random.Generator | None = None,
) -> Bvh:
    """Translate root by a random offset sampled uniformly per axis.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    range_xyz : tuple of (low, high)
        Uniform sampling range applied to each axis independently.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    Bvh
    """
    if rng is None:
        rng = np.random.default_rng()
    offset = rng.uniform(range_xyz[0], range_xyz[1], size=3)
    return translate_root(bvh, offset)  # type: ignore[return-value]


# =========================================================================
# 6.4  Joint Noise Injection
# =========================================================================

@overload
def add_joint_noise(
    bvh: Bvh, sigma_deg: float, *, sigma_pos: float = ...,
    rng: np.random.Generator | None = ..., inplace: Literal[True],
) -> None: ...
@overload
def add_joint_noise(
    bvh: Bvh, sigma_deg: float, sigma_pos: float = ...,
    rng: np.random.Generator | None = ..., inplace: Literal[False] = ...,
) -> Bvh: ...
def add_joint_noise(
    bvh: Bvh,
    sigma_deg: float,
    sigma_pos: float = 0.0,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
    wrap: bool = True,
) -> Bvh | None:
    """Add Gaussian noise to joint rotation angles.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    sigma_deg : float
        Standard deviation of noise in degrees, added to
        ``joint_angles``.
    sigma_pos : float, optional
        Standard deviation of noise added to ``root_pos``
        (default 0 — no position noise).
    rng : numpy.random.Generator or None
        Random generator for reproducibility.
    inplace : bool, optional
        If True, modify *bvh* and return None.
    wrap : bool, optional
        If True (default), wrap noised angles to [-180, 180] so
        downstream Euler-to-rotmat round-trips don't see discontinuities.
        Set to False if the consumer handles angle ranges itself.

    Returns
    -------
    Bvh or None
    """
    if rng is None:
        rng = np.random.default_rng()

    target = bvh if inplace else bvh.copy()
    if sigma_deg > 0:
        noised = (
            target.joint_angles
            + rng.normal(0.0, sigma_deg, target.joint_angles.shape)
        )
        if wrap:
            noised = (noised + 180.0) % 360.0 - 180.0
        target.joint_angles = noised
    if sigma_pos > 0:
        target.root_pos = (
            target.root_pos
            + rng.normal(0.0, sigma_pos, target.root_pos.shape)
        )
    if inplace:
        return None
    return target


# =========================================================================
# 6.3  Speed Perturbation
# =========================================================================

def speed_perturbation(bvh: Bvh, factor: float) -> Bvh:
    """Change motion speed by resampling.

    A *factor* of 2.0 makes the motion twice as fast (fewer frames);
    0.5 makes it half as fast (more frames).  Uses the existing
    :meth:`Bvh.resample` which performs quaternion SLERP for rotations.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    factor : float
        Speed multiplier (must be > 0).

    Returns
    -------
    Bvh
        New Bvh with adjusted frame count and ``frame_frequency``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    if bvh.frame_frequency <= 0:
        raise ValueError("Cannot resample: frame_frequency is 0.")
    original_fps = 1.0 / bvh.frame_frequency
    # Resample to fewer/more frames, then restore original frame rate.
    # factor > 1 → faster → fewer frames; factor < 1 → slower → more frames.
    result = bvh.resample(original_fps / factor)
    result.frame_frequency = bvh.frame_frequency
    return result


def random_speed_perturbation(
    bvh: Bvh,
    factor_range: tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> Bvh:
    """Apply a random speed change sampled uniformly from *factor_range*.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    factor_range : tuple of (low, high)
        Range for the speed factor.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    Bvh
    """
    if rng is None:
        rng = np.random.default_rng()
    factor = float(rng.uniform(factor_range[0], factor_range[1]))
    return speed_perturbation(bvh, factor)


# =========================================================================
# 6.6  Frame Dropout with Interpolation
# =========================================================================

@overload
def dropout_frames(
    bvh: Bvh, drop_rate: float, *, rng: np.random.Generator | None = ...,
    inplace: Literal[True],
) -> None: ...
@overload
def dropout_frames(
    bvh: Bvh, drop_rate: float, rng: np.random.Generator | None = ...,
    inplace: Literal[False] = ...,
) -> Bvh: ...
def dropout_frames(
    bvh: Bvh,
    drop_rate: float,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
) -> Bvh | None:
    """Replace randomly selected frames with SLERP-interpolated values.

    Dropped frames are filled by spherical linear interpolation
    (SLERP) of the nearest kept neighbours' quaternion rotations and
    linear interpolation of root positions.  The output has the **same**
    frame count as the input.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    drop_rate : float
        Fraction of frames to drop, in (0, 1).  First and last
        frames are always kept.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None
    """
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    if rng is None:
        rng = np.random.default_rng()

    F = bvh.frame_count
    if F < 2 or drop_rate == 0.0:
        if inplace:
            return None
        return bvh.copy()

    # Build keep mask — always keep first and last
    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[F - 1] = True
    kept_indices = np.where(keep_mask)[0]

    if len(kept_indices) < 2:
        # Very unlikely, but handle gracefully
        if inplace:
            return None
        return bvh.copy()

    # Get quaternion representation
    root_pos_orig, quats_orig, _ = bvh.get_frames_as_quaternion()
    # quats_orig: (F, J, 4),  root_pos_orig: (F, 3)

    # For every frame, find left and right kept-neighbour indices
    # searchsorted gives the insertion point in kept_indices
    ins = np.searchsorted(kept_indices, np.arange(F), side='right')
    left_idx = np.clip(ins - 1, 0, len(kept_indices) - 1)
    right_idx = np.clip(ins, 0, len(kept_indices) - 1)

    left_frames = kept_indices[left_idx]   # (F,)
    right_frames = kept_indices[right_idx]  # (F,)

    span = (right_frames - left_frames).astype(np.float64)
    span = np.where(span == 0, 1.0, span)
    alpha = (np.arange(F, dtype=np.float64) - left_frames) / span  # (F,)

    # Interpolate root position linearly
    new_root_pos = (
        root_pos_orig[left_frames] * (1.0 - alpha[:, None])
        + root_pos_orig[right_frames] * alpha[:, None]
    )

    # SLERP quaternions per joint
    J = quats_orig.shape[1]
    new_quats = np.empty_like(quats_orig)
    for j in range(J):
        q_left = quats_orig[left_frames, j]    # (F, 4)
        q_right = quats_orig[right_frames, j]  # (F, 4)
        new_quats[:, j] = rotations.quat_slerp(q_left, q_right, alpha)

    # Overwrite only dropped frames (keep originals for kept frames)
    new_root_pos[keep_mask] = root_pos_orig[keep_mask]
    new_quats[keep_mask] = quats_orig[keep_mask]

    # Convert back to Euler angles
    joints = [n for n in bvh.nodes if isinstance(n, BvhJoint)]
    new_angles = np.empty((F, len(joints), 3), dtype=np.float64)
    for j_idx, joint in enumerate(joints):
        order = "".join(joint.rot_channels)
        R = rotations.quat_to_rotmat(new_quats[:, j_idx])
        new_angles[:, j_idx] = rotations.rotmat_to_euler(R, order, degrees=True)

    target = bvh if inplace else bvh.copy()
    target.root_pos = new_root_pos
    target.joint_angles = new_angles
    if inplace:
        return None
    return target


# =========================================================================
# NumPy-level API — rotate_angles_vertical
# =========================================================================

def rotate_angles_vertical(
    joint_angles: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    angle_deg: float,
    up_idx: int,
    root_order: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate motion around the vertical axis (NumPy-level).

    Only modifies ``root_pos`` and the root joint's Euler angles
    (index 0 in ``joint_angles``).  Non-root joints are in parent-local
    coordinates and are unaffected.

    Parameters
    ----------
    joint_angles : ndarray of shape (F, J, 3)
        Euler angles in degrees.
    root_pos : ndarray of shape (F, 3)
        Root translation per frame.
    angle_deg : float
        Rotation angle in degrees.
    up_idx : int
        Index of the up axis (0=X, 1=Y, 2=Z).
    root_order : str
        Euler order of the root joint, e.g. ``'ZYX'``.

    Returns
    -------
    (new_joint_angles, new_root_pos)
        Copies with the rotation applied.

    See Also
    --------
    rotate_vertical : Bvh-level wrapper that auto-detects ``up_idx``
        and ``root_order`` from the skeleton.

    Examples
    --------
    >>> angles = bvh.joint_angles          # (F, J, 3) degrees
    >>> pos = bvh.root_pos                 # (F, 3)
    >>> up = tools.get_up_axis_index(bvh, bvh.get_spatial_coord(0))
    >>> order = ''.join(bvh.root.rot_channels)
    >>> new_angles, new_pos = rotate_angles_vertical(
    ...     angles, pos, 90.0, up, order)
    """
    angle_rad = np.radians(angle_deg)
    rot_funcs = {0: rotX, 1: rotY, 2: rotZ}
    R_vert: npt.NDArray[np.float64] = rot_funcs[up_idx](angle_rad)

    new_root_pos = (R_vert @ root_pos.T).T

    new_angles = joint_angles.copy()
    R_root = rotations.euler_to_rotmat(joint_angles[:, 0], root_order, degrees=True)
    R_new = R_vert[np.newaxis] @ R_root
    new_angles[:, 0] = rotations.rotmat_to_euler(R_new, root_order, degrees=True)

    return new_angles, new_root_pos


# =========================================================================
# 6.2  Vertical Rotation
# =========================================================================

@overload
def rotate_vertical(
    bvh: Bvh, angle_deg: float, *, up_axis: str | None = ...,
    inplace: Literal[True],
) -> None: ...
@overload
def rotate_vertical(
    bvh: Bvh, angle_deg: float, up_axis: str | None = ...,
    inplace: Literal[False] = ...,
) -> Bvh: ...
def rotate_vertical(
    bvh: Bvh,
    angle_deg: float,
    up_axis: str | None = None,
    inplace: bool = False,
) -> Bvh | None:
    """Rotate the entire motion around the vertical (up) axis.

    Only the root joint's world-space rotation and root position are
    modified.  Child joints are in parent-local coordinates and are
    unaffected.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    angle_deg : float
        Rotation angle in degrees (positive = counter-clockwise
        when viewed from above).
    up_axis : str or None
        Signed axis string (e.g. ``'+y'``).  Auto-detected if None.
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None
    """
    target = bvh if inplace else bvh.copy()

    # Determine up axis
    if up_axis is None:
        rest = target.get_rest_pose(mode='coordinates')
        up_idx = get_up_axis_index(target, rest)  # type: ignore[arg-type]
    else:
        up_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis[1]]

    root_order = "".join(target.root.rot_channels)
    new_angles, new_root_pos = rotate_angles_vertical(
        target.joint_angles, target.root_pos, angle_deg, up_idx, root_order,
    )
    target.joint_angles = new_angles
    target.root_pos = new_root_pos

    if inplace:
        return None
    return target


def random_rotate_vertical(
    bvh: Bvh,
    angle_range: tuple[float, float] = (-180.0, 180.0),
    up_axis: str | None = None,
    rng: np.random.Generator | None = None,
) -> Bvh:
    """Rotate motion by a random angle around the vertical axis.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    angle_range : tuple of (low, high)
        Angle sampling range in degrees.
    up_axis : str or None
        Signed axis string.  Auto-detected if None.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    Bvh
    """
    if rng is None:
        rng = np.random.default_rng()
    angle = float(rng.uniform(angle_range[0], angle_range[1]))
    return rotate_vertical(bvh, angle, up_axis=up_axis)  # type: ignore[return-value]


# =========================================================================
# NumPy-level API — mirror_angles
# =========================================================================

def mirror_angles(
    joint_angles: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    lr_joint_pairs: list[tuple[int, int]],
    lateral_idx: int,
    rot_channels: list[list[str]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mirror joint angles and root position (NumPy-level).

    Performs the array-level operations of mirroring: negating the
    lateral component of ``root_pos``, swapping L/R joint angle columns,
    and negating Euler angle components whose rotation axis is not the
    lateral axis.

    This function does **not** modify skeleton offsets (bone geometry).
    For a complete mirror that also adjusts the skeleton, use the
    Bvh-level :func:`mirror` function.

    Parameters
    ----------
    joint_angles : ndarray of shape (F, J, 3)
        Euler angles in degrees.
    root_pos : ndarray of shape (F, 3)
        Root translation per frame.
    lr_joint_pairs : list of (left_idx, right_idx)
        Index pairs into the joint axis of ``joint_angles``.
    lateral_idx : int
        Index of the lateral axis (0=X, 1=Y, 2=Z).
    rot_channels : list of list of str
        Per-joint Euler channel order, e.g. ``[['Z','Y','X'], ...]``.
        Length must equal ``J``.

    Returns
    -------
    (new_joint_angles, new_root_pos)
        Copies with the mirroring applied.

    See Also
    --------
    mirror : Bvh-level wrapper that also mirrors skeleton offsets and
        auto-detects ``lr_joint_pairs``, ``lateral_idx``, and
        ``rot_channels`` from the skeleton.

    Examples
    --------
    >>> angles = bvh.joint_angles
    >>> pos = bvh.root_pos
    >>> pairs = bvh.auto_detect_lr_pairs()
    >>> lat = tools.get_forw_up_axis(bvh, bvh.get_spatial_coord(0))
    >>> channels = [n.rot_channels for n in bvh.nodes if not n.is_end_site()]
    >>> new_angles, new_pos = mirror_angles(
    ...     angles, pos, pairs, lat_idx, channels)
    """
    new_angles = joint_angles.copy()
    new_root_pos = root_pos.copy()

    # Negate root_pos lateral component
    new_root_pos[:, lateral_idx] *= -1

    # Swap L/R joint angle columns
    for lj, rj in lr_joint_pairs:
        left_data = new_angles[:, lj].copy()
        new_angles[:, lj] = new_angles[:, rj]
        new_angles[:, rj] = left_data

    # Negate Euler components whose rotation axis is NOT the lateral axis
    lateral_upper = "XYZ"[lateral_idx]
    for j_idx, channels in enumerate(rot_channels):
        for ch_idx, ch in enumerate(channels):
            if ch != lateral_upper:
                new_angles[:, j_idx, ch_idx] *= -1

    return new_angles, new_root_pos


# =========================================================================
# 6.1  Left-Right Mirroring
# =========================================================================

def auto_detect_lr_mapping(bvh: Bvh) -> dict[str, str]:
    """Auto-detect left/right joint pairs by name heuristics.

    Tries "Left"↔"Right" substring replacement and "L"↔"R" prefix
    patterns.  Returns a dict mapping left joint names to right joint
    names (one direction only).

    Parameters
    ----------
    bvh : Bvh
        Input BVH with named joints.

    Returns
    -------
    dict
        ``{"LeftArm": "RightArm", ...}``.  Empty if no pairs found.
    """
    joint_names = set(bvh.joint_names)
    # Also include end-site names for completeness
    all_names = {n.name for n in bvh.nodes}
    mapping: dict[str, str] = {}
    seen: set[str] = set()

    for name in bvh.joint_names:
        if name in seen:
            continue

        partner: str | None = None

        # Strategy 1: "Left" <-> "Right" anywhere in name
        if "Left" in name:
            candidate = name.replace("Left", "Right", 1)
            if candidate in joint_names:
                partner = candidate
        elif "Right" in name:
            candidate = name.replace("Right", "Left", 1)
            if candidate in joint_names:
                partner = candidate

        # Strategy 2: "L" / "R" prefix followed by uppercase
        if partner is None and len(name) >= 2:
            if name[0] == "L" and name[1].isupper():
                candidate = "R" + name[1:]
                if candidate in joint_names:
                    partner = candidate
            elif name[0] == "R" and name[1].isupper():
                candidate = "L" + name[1:]
                if candidate in joint_names:
                    partner = candidate

        if partner is not None and partner not in seen:
            # Normalise: always store the "Left" / "L" version as key
            left, right = _order_lr_pair(name, partner)
            mapping[left] = right
            seen.add(left)
            seen.add(right)

    return mapping


def _order_lr_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(left_name, right_name)``."""
    for kw in ("Left", "left"):
        if kw in a:
            return (a, b)
        if kw in b:
            return (b, a)
    # Fallback: "L" prefix → left
    if a.startswith("L"):
        return (a, b)
    return (b, a)


def auto_detect_lr_pairs(bvh: Bvh) -> list[tuple[int, int]]:
    """Auto-detect left/right joint pairs as index tuples.

    Wraps :func:`auto_detect_lr_mapping` and converts joint name
    pairs to index pairs in ``joint_angles`` index space (axis 1
    of ``bvh.joint_angles``).

    Parameters
    ----------
    bvh : Bvh
        Input BVH with named joints.

    Returns
    -------
    list of (int, int)
        ``[(left_idx, right_idx), ...]`` in ``joint_angles`` index
        space.  Empty if no pairs found.
    """
    mapping = auto_detect_lr_mapping(bvh)
    j_name2idx = {name: i for i, name in enumerate(bvh.joint_names)}
    pairs: list[tuple[int, int]] = []
    for left_name, right_name in mapping.items():
        if left_name in j_name2idx and right_name in j_name2idx:
            pairs.append((j_name2idx[left_name], j_name2idx[right_name]))
    return pairs


@overload
def mirror(
    bvh: Bvh, *, left_right_mapping: dict[str, str] | None = ...,
    lateral_axis: str | None = ..., inplace: Literal[True],
) -> None: ...
@overload
def mirror(
    bvh: Bvh, left_right_mapping: dict[str, str] | None = ...,
    lateral_axis: str | None = ..., inplace: Literal[False] = ...,
) -> Bvh: ...
def mirror(
    bvh: Bvh,
    left_right_mapping: dict[str, str] | None = None,
    lateral_axis: str | None = None,
    inplace: bool = False,
) -> Bvh | None:
    """Mirror (reflect) the motion across the lateral plane.

    Swaps left/right joint data and negates the appropriate rotation
    and position components so that the skeleton appears as a mirror
    image.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    left_right_mapping : dict or None
        ``{"LeftArm": "RightArm", ...}``.  Auto-detected if None.
    lateral_axis : str or None
        Axis perpendicular to the mirror plane, e.g. ``'x'``.
        Auto-detected if None (the axis that is neither forward
        nor upward).
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None

    Raises
    ------
    ValueError
        If auto-detection finds no left/right pairs.
    """
    target = bvh if inplace else bvh.copy()

    # --- Detect lateral axis ---
    if lateral_axis is None:
        rest = target.get_rest_pose(mode='coordinates')
        dirs = get_forw_up_axis(target, rest)  # type: ignore[arg-type]
        used = {dirs['forward'][1], dirs['upward'][1]}
        lateral_char = ({"x", "y", "z"} - used).pop()
    else:
        lateral_char = lateral_axis.lower().lstrip("+-")
    lateral_idx = {"x": 0, "y": 1, "z": 2}[lateral_char]

    # --- Detect L/R pairs ---
    if left_right_mapping is None:
        left_right_mapping = auto_detect_lr_mapping(target)

    # Build joint-index pairs (indices into joint_angles axis 1)
    joints = [n for n in target.nodes if isinstance(n, BvhJoint)]
    j_name2idx = {j.name: i for i, j in enumerate(joints)}
    lr_j_pairs: list[tuple[int, int]] = []
    for left_name, right_name in left_right_mapping.items():
        if left_name in j_name2idx and right_name in j_name2idx:
            lr_j_pairs.append((j_name2idx[left_name], j_name2idx[right_name]))

    # Build node-index pairs for offset swapping (includes end sites)
    node_name2idx = target.node_index
    lr_node_pairs: list[tuple[int, int]] = []
    for left_name, right_name in left_right_mapping.items():
        if left_name in node_name2idx and right_name in node_name2idx:
            lr_node_pairs.append(
                (node_name2idx[left_name], node_name2idx[right_name])
            )
            # Also pair their end-site children
            left_node = target.nodes[node_name2idx[left_name]]
            right_node = target.nodes[node_name2idx[right_name]]
            if isinstance(left_node, BvhJoint) and isinstance(right_node, BvhJoint):
                left_ends = [c for c in left_node.children if c.is_end_site()]
                right_ends = [c for c in right_node.children if c.is_end_site()]
                for le, re in zip(left_ends, right_ends):
                    li = node_name2idx[le.name]
                    ri = node_name2idx[re.name]
                    lr_node_pairs.append((li, ri))

    # --- Steps 1, 4, 5: Mirror arrays via NumPy-level API ---
    rot_ch = [list(j.rot_channels) for j in joints]
    new_angles, new_root_pos = mirror_angles(
        target.joint_angles, target.root_pos,
        lr_j_pairs, lateral_idx, rot_ch,
    )
    target.joint_angles = new_angles
    target.root_pos = new_root_pos

    # --- Step 2: Negate node offset lateral component for ALL nodes ---
    for node in target.nodes:
        off = node.offset.copy()
        off[lateral_idx] *= -1
        node.offset = off

    # --- Step 3: Swap offsets for L/R paired nodes ---
    for li, ri in lr_node_pairs:
        left_off = target.nodes[li].offset.copy()
        right_off = target.nodes[ri].offset.copy()
        target.nodes[li].offset = right_off
        target.nodes[ri].offset = left_off

    if inplace:
        return None
    return target
