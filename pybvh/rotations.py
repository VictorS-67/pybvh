"""
Rotation & rigid-transform math for skeleton-based motion data.

All functions are batch-vectorized using NumPy and operate on arrays
where the leading dimensions are batch dimensions.

Supported representations:
- Euler angles: (*, 3) in degrees or radians
- Rotation matrices: (*, 3, 3)
- 6D rotation (Zhou et al., CVPR 2019): (*, 6) — continuous representation
- Quaternions: (*, 4) in (w, x, y, z) scalar-first convention
- Axis-angle: (*, 3) — rotation axis scaled by rotation angle in radians
- Rigid transforms (SE(3)): (*, 4, 4) homogeneous matrices, with the
  matching se(3) twist coordinates ``[ω(3), v(3)]`` (rotation-first,
  V-Jacobian-coupled) — see :func:`se3_exp` / :func:`se3_log`.

Convention note:
    Euler angles in BVH files use intrinsic rotations with pre-multiplication:
        R = R_first @ R_second @ R_third
    where the order comes from the joint's rot_channels (e.g., ['Z','Y','X']).
    Angles are in degrees in BVH files, but most functions here work in radians
    unless stated otherwise.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt


# Channel count per per-joint rotation representation. Handy for
# allocating output arrays or sizing model layers without hard-coding
# the numbers at each call site.
REPRESENTATION_CHANNELS = {
    "euler": 3,
    "axisangle": 3,
    "quaternion": 4,
    "6d": 6,
    "rotmat": 9,
}


# ============================================================================
# Euler angles <-> Rotation matrices
# ============================================================================

def _validate_order(order_str: str) -> None:
    if len(order_str) != 3 or not all(c in 'XYZ' for c in order_str):
        raise ValueError(f"order must be 3 characters from 'XYZ', got '{order_str}'")


def _parse_order(order: Union[str, Sequence[str]]) -> tuple[str | None, list[str] | None]:
    """Classify the ``order`` argument.

    Returns ``(single_order, per_joint_orders)`` — exactly one is not None.
    Accepted forms:

    - ``'ZYX'`` (string)                         → single order
    - ``['Z', 'Y', 'X']`` (3 single chars)       → single order
    - ``['ZYX', 'ZYX', 'ZXY', ...]`` (N strings) → per-joint orders
    """
    if isinstance(order, str):
        return order.upper(), None

    order_seq = [str(o) for o in order]
    if len(order_seq) == 3 and all(len(o) == 1 for o in order_seq):
        # Backward-compat: 3 single chars joined to form one global order
        return ''.join(order_seq).upper(), None

    per_joint = [o.upper() for o in order_seq]
    for o in per_joint:
        _validate_order(o)
    return None, per_joint


def euler_to_rotmat(
    angles: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert Euler angles to rotation matrices (batch).

    Parameters
    ----------
    angles : array_like, shape (*, 3) or (*, J, 3)
        Euler angles. Each row is (angle1, angle2, angle3) following the
        axis order given by `order`.  When `order` is a per-joint
        sequence of length J, the second-to-last axis is the joint axis.
    order : str or sequence of strings
        - ``'ZYX'`` or ``['Z', 'Y', 'X']`` — single global order applied
          to every entry.  R = R1 @ R2 @ R3 (intrinsic, pre-multiplied).
        - ``['ZYX', 'ZYX', 'ZXY', ...]`` — per-joint orders, one entry
          per joint along axis ``-2`` of ``angles``.  Input shape must
          satisfy ``angles.shape[-2] == len(order)``.  Joints sharing an
          order are grouped so the rotation math vectorizes inside each
          group.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    R : ndarray, shape (*, 3, 3) or (*, J, 3, 3)
        Rotation matrices, one per input entry.
    """
    angles_arr: npt.NDArray[np.float64] = np.asarray(angles, dtype=np.float64)
    if degrees:
        angles_arr = np.radians(angles_arr)

    single_order, per_joint = _parse_order(order)

    if single_order is not None:
        _validate_order(single_order)
        single = (angles_arr.ndim == 1)
        if single:
            angles_arr = angles_arr[np.newaxis, :]
        R = _euler_to_rotmat_rad(angles_arr, single_order)
        return R[0] if single else R

    # Per-joint mode
    assert per_joint is not None
    J = len(per_joint)
    if angles_arr.ndim < 2 or angles_arr.shape[-2] != J:
        raise ValueError(
            f"per-joint order of length {J} requires angles shape "
            f"(..., {J}, 3); got shape {angles_arr.shape}")

    out = np.empty(angles_arr.shape + (3,), dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for j, o in enumerate(per_joint):
        groups.setdefault(o, []).append(j)
    for order_str, idxs in groups.items():
        idx_arr = np.asarray(idxs)
        block = angles_arr[..., idx_arr, :]              # (*, |group|, 3)
        out[..., idx_arr, :, :] = _euler_to_rotmat_rad(block, order_str)
    return out


def _euler_to_rotmat_rad(
    angles_rad: npt.NDArray[np.float64],
    order_str: str,
) -> npt.NDArray[np.float64]:
    """Core multiplication: angles_rad shape (..., 3) → (..., 3, 3)."""
    orig_shape = angles_rad.shape[:-1]
    flat = angles_rad.reshape(-1, 3)
    R = _elementary_rotmat(flat[:, 0], order_str[0])
    R = R @ _elementary_rotmat(flat[:, 1], order_str[1])
    R = R @ _elementary_rotmat(flat[:, 2], order_str[2])
    return R.reshape(orig_shape + (3, 3))


def rotmat_to_euler(
    R: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrices to Euler angles (batch).

    Uses the convention of intrinsic rotations with pre-multiplication.
    Handles gimbal lock by setting the third angle to 0.

    Parameters
    ----------
    R : array_like, shape (*, 3, 3) or (*, J, 3, 3)
        Rotation matrices.  When `order` is a per-joint sequence of
        length J, the third-to-last axis is the joint axis.
    order : str or sequence of strings
        - ``'ZYX'`` or ``['Z', 'Y', 'X']`` — single global order.
        - ``['ZYX', 'ZYX', ...]`` — per-joint orders, one entry per
          joint along axis ``-3`` of ``R``.  Must satisfy
          ``R.shape[-3] == len(order)``.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    angles : ndarray, shape (*, 3) or (*, J, 3)
        Euler angles in the specified order.
    """
    R_arr: npt.NDArray[np.float64] = np.asarray(R, dtype=np.float64)

    single_order, per_joint = _parse_order(order)

    if single_order is not None:
        _validate_order(single_order)
        single = (R_arr.ndim == 2)
        if single:
            R_arr = R_arr[np.newaxis, :, :]
        out = _rotmat_to_euler_rad(R_arr, single_order)
        if degrees:
            out = np.degrees(out)
        return out[0] if single else out

    # Per-joint mode
    assert per_joint is not None
    J = len(per_joint)
    if R_arr.ndim < 3 or R_arr.shape[-3] != J:
        raise ValueError(
            f"per-joint order of length {J} requires R shape "
            f"(..., {J}, 3, 3); got shape {R_arr.shape}")

    out = np.empty(R_arr.shape[:-2] + (3,), dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for j, o in enumerate(per_joint):
        groups.setdefault(o, []).append(j)
    for order_str, idxs in groups.items():
        idx_arr = np.asarray(idxs)
        block = R_arr[..., idx_arr, :, :]                # (*, |group|, 3, 3)
        out[..., idx_arr, :] = _rotmat_to_euler_rad(block, order_str)
    if degrees:
        out = np.degrees(out)
    return out


def _rotmat_to_euler_rad(
    R_arr: npt.NDArray[np.float64],
    order_str: str,
) -> npt.NDArray[np.float64]:
    """Core extraction: R_arr shape (..., 3, 3) → (..., 3) radians."""
    ax2idx = {'X': 0, 'Y': 1, 'Z': 2}
    i = ax2idx[order_str[0]]
    j = ax2idx[order_str[1]]
    k = ax2idx[order_str[2]]
    orig_shape = R_arr.shape[:-2]
    flat = R_arr.reshape(-1, 3, 3)
    out = _extract_euler(flat, i, j, k)
    return out.reshape(orig_shape + (3,))


# ============================================================================
# Rotation matrices <-> 6D representation (Zhou et al., CVPR 2019)
# ============================================================================

def rotmat_to_rot6d(R: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrices to 6D representation.

    The 6D representation consists of the first two columns of the
    rotation matrix, concatenated into a 6-vector.

    Parameters
    ----------
    R : array_like, shape (*, 3, 3)
        Rotation matrices.

    Returns
    -------
    rot6d : ndarray, shape (*, 6)
        6D rotation vectors [col0 | col1].
    """
    R_arr: npt.NDArray[np.float64] = np.asarray(R, dtype=np.float64)
    # Take first two columns: R[..., :, 0] and R[..., :, 1]
    return np.concatenate([R_arr[..., :, 0], R_arr[..., :, 1]], axis=-1)


def rot6d_to_rotmat(rot6d: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert 6D rotation representation to rotation matrices using
    Gram-Schmidt orthogonalization (Zhou et al., CVPR 2019).

    Parameters
    ----------
    rot6d : array_like, shape (*, 6)
        6D rotation vectors [a1 | a2] where a1 and a2 are 3-vectors.

    Returns
    -------
    R : ndarray, shape (*, 3, 3)
        Rotation matrices (proper rotations, det = +1).
    """
    rot6d_arr: npt.NDArray[np.float64] = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d_arr[..., :3]
    a2 = rot6d_arr[..., 3:]

    # Gram-Schmidt: orthonormalize
    b1 = _normalize(a1)
    # b2 = normalize(a2 - (a2 . b1) * b1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1)
    # b3 = b1 x b2
    b3 = np.cross(b1, b2, axis=-1)

    # Stack columns into rotation matrix
    return np.stack([b1, b2, b3], axis=-1)


# ============================================================================
# Euler angles <-> 6D (convenience wrappers)
# ============================================================================

def euler_to_rot6d(
    angles: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert Euler angles to 6D rotation representation.

    Parameters
    ----------
    angles : array_like, shape (*, 3)
        Euler angles.
    order : str or list
        Rotation axis order, e.g. 'ZYX'.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    rot6d : ndarray, shape (*, 6)
    """
    return rotmat_to_rot6d(euler_to_rotmat(angles, order, degrees=degrees))


def rot6d_to_euler(
    rot6d: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert 6D rotation representation to Euler angles.

    Parameters
    ----------
    rot6d : array_like, shape (*, 6)
        6D rotation vectors.
    order : str or list
        Rotation axis order, e.g. 'ZYX'.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    angles : ndarray, shape (*, 3)
    """
    return rotmat_to_euler(rot6d_to_rotmat(rot6d), order, degrees=degrees)


# ============================================================================
# Rotation matrices <-> Quaternions
# ============================================================================

def rotmat_to_quat(R: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrices to quaternions (batch).

    Uses the Shepperd method for numerical stability.

    Parameters
    ----------
    R : array_like, shape (*, 3, 3)
        Rotation matrices.

    Returns
    -------
    q : ndarray, shape (*, 4)
        Unit quaternions in (w, x, y, z) scalar-first convention.
    """
    R_arr: npt.NDArray[np.float64] = np.asarray(R, dtype=np.float64)
    single = (R_arr.ndim == 2)
    if single:
        R_arr = R_arr[np.newaxis, :, :]

    batch_shape = R_arr.shape[:-2]
    R_flat = R_arr.reshape(-1, 3, 3)
    N = R_flat.shape[0]

    q = np.empty((N, 4), dtype=np.float64)

    # Shepperd's method: choose the largest diagonal element to avoid
    # division by near-zero.
    # trace = R00 + R11 + R22
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    # Case 0: trace > 0  =>  w is largest
    s = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2  # s = 4w
    mask0 = trace > 0
    if np.any(mask0):
        q[mask0, 0] = 0.25 * s[mask0]
        q[mask0, 1] = (R_flat[mask0, 2, 1] - R_flat[mask0, 1, 2]) / s[mask0]
        q[mask0, 2] = (R_flat[mask0, 0, 2] - R_flat[mask0, 2, 0]) / s[mask0]
        q[mask0, 3] = (R_flat[mask0, 1, 0] - R_flat[mask0, 0, 1]) / s[mask0]

    # Case 1: R00 is largest diagonal
    mask1 = (~mask0) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if np.any(mask1):
        s1 = np.sqrt(np.maximum(1.0 + R_flat[mask1, 0, 0] - R_flat[mask1, 1, 1] - R_flat[mask1, 2, 2], 0.0)) * 2
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s1
        q[mask1, 1] = 0.25 * s1
        q[mask1, 2] = (R_flat[mask1, 0, 1] + R_flat[mask1, 1, 0]) / s1
        q[mask1, 3] = (R_flat[mask1, 0, 2] + R_flat[mask1, 2, 0]) / s1

    # Case 2: R11 is largest diagonal
    mask2 = (~mask0) & (~mask1) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if np.any(mask2):
        s2 = np.sqrt(np.maximum(1.0 + R_flat[mask2, 1, 1] - R_flat[mask2, 0, 0] - R_flat[mask2, 2, 2], 0.0)) * 2
        q[mask2, 0] = (R_flat[mask2, 0, 2] - R_flat[mask2, 2, 0]) / s2
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s2
        q[mask2, 2] = 0.25 * s2
        q[mask2, 3] = (R_flat[mask2, 1, 2] + R_flat[mask2, 2, 1]) / s2

    # Case 3: R22 is largest diagonal
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if np.any(mask3):
        s3 = np.sqrt(np.maximum(1.0 + R_flat[mask3, 2, 2] - R_flat[mask3, 0, 0] - R_flat[mask3, 1, 1], 0.0)) * 2
        q[mask3, 0] = (R_flat[mask3, 1, 0] - R_flat[mask3, 0, 1]) / s3
        q[mask3, 1] = (R_flat[mask3, 0, 2] + R_flat[mask3, 2, 0]) / s3
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s3
        q[mask3, 3] = 0.25 * s3

    # Normalize to unit quaternion
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    # Enforce canonical form: w >= 0
    neg_w = q[:, 0] < 0
    q[neg_w] *= -1

    q = q.reshape(batch_shape + (4,))
    if single:
        return q[0]
    return q


def quat_to_rotmat(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert quaternions to rotation matrices (batch).

    Parameters
    ----------
    q : array_like, shape (*, 4)
        Quaternions in (w, x, y, z) scalar-first convention.
        Need not be unit quaternions (will be normalized).

    Returns
    -------
    R : ndarray, shape (*, 3, 3)
        Rotation matrices.
    """
    q_arr: npt.NDArray[np.float64] = np.asarray(q, dtype=np.float64)
    single = (q_arr.ndim == 1)
    if single:
        q_arr = q_arr[np.newaxis, :]

    # Normalize
    q_arr = q_arr / np.linalg.norm(q_arr, axis=-1, keepdims=True)

    w, x, y, z = q_arr[..., 0], q_arr[..., 1], q_arr[..., 2], q_arr[..., 3]

    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(q_arr.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)

    if single:
        return R[0]
    return R


# ============================================================================
# Euler angles <-> Quaternions (convenience wrappers)
# ============================================================================

def euler_to_quat(
    angles: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert Euler angles to quaternions.

    Parameters
    ----------
    angles : array_like, shape (*, 3)
        Euler angles.
    order : str or list, e.g. 'ZYX'
        Rotation axis order.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    q : ndarray, shape (*, 4)
        Quaternions (w, x, y, z).
    """
    return rotmat_to_quat(euler_to_rotmat(angles, order, degrees=degrees))


def quat_to_euler(
    q: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert quaternions to Euler angles.

    Parameters
    ----------
    q : array_like, shape (*, 4)
        Quaternions (w, x, y, z).
    order : str or list, e.g. 'ZYX'
        Rotation axis order.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    angles : ndarray, shape (*, 3)
        Euler angles.
    """
    return rotmat_to_euler(quat_to_rotmat(q), order, degrees=degrees)




# ============================================================================
# Rotation matrices <-> Axis-angle
# ============================================================================

def rotmat_to_axisangle(R: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrices to axis-angle representation (batch).

    The axis-angle vector is the unit rotation axis scaled by the rotation
    angle (in radians).  For the identity rotation the zero vector is returned.

    Uses the logarithmic map: angle = arccos((trace(R)-1)/2), axis from the
    skew-symmetric part of R.  The 180° case is handled via the eigenvector
    of R corresponding to eigenvalue 1.

    Parameters
    ----------
    R : array_like, shape (*, 3, 3)
        Rotation matrices.

    Returns
    -------
    aa : ndarray, shape (*, 3)
        Axis-angle vectors (axis × angle_radians).
    """
    R_arr: npt.NDArray[np.float64] = np.asarray(R, dtype=np.float64)
    single = (R_arr.ndim == 2)
    if single:
        R_arr = R_arr[np.newaxis, :, :]

    batch_shape = R_arr.shape[:-2]
    R_flat = R_arr.reshape(-1, 3, 3)
    N = R_flat.shape[0]

    # angle = arccos( clamp( (trace - 1) / 2 ) )
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)  # in [0, π]

    aa = np.zeros((N, 3), dtype=np.float64)

    # ---- General case: sin(angle) is not near zero ----
    sin_angle = np.sin(angle)
    general = sin_angle > 1e-7
    if np.any(general):
        # axis from skew-symmetric part:  (R - R^T) / (2 sin θ)
        # r = [R32-R23, R13-R31, R21-R12]
        r = np.empty((N, 3), dtype=np.float64)
        r[:, 0] = R_flat[:, 2, 1] - R_flat[:, 1, 2]
        r[:, 1] = R_flat[:, 0, 2] - R_flat[:, 2, 0]
        r[:, 2] = R_flat[:, 1, 0] - R_flat[:, 0, 1]

        idx = np.where(general)[0]
        aa[idx] = (r[idx] / (2.0 * sin_angle[idx, np.newaxis])) * angle[idx, np.newaxis]

    # ---- Near 180° case: sin(angle) ≈ 0 but angle ≈ π ----
    near_pi = (~general) & (angle > 1e-7)  # angle > 0 but sin≈0 ⟹ near π
    if np.any(near_pi):
        idx = np.where(near_pi)[0]
        for i in idx:
            # R ≈ 2 * (n n^T) - I  ⟹  n n^T = (R + I) / 2
            # Pick the column of (R+I) with the largest norm as n
            M = (R_flat[i] + np.eye(3)) / 2.0
            col_norms = np.sum(M ** 2, axis=0)
            best = np.argmax(col_norms)
            axis = M[:, best]
            axis = axis / np.linalg.norm(axis)
            aa[i] = axis * angle[i]

    # Near-zero angle case: aa stays at 0 (identity rotation)

    aa = aa.reshape(batch_shape + (3,))
    if single:
        return aa[0]
    return aa


def axisangle_to_rotmat(aa: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues' formula (batch).

    Parameters
    ----------
    aa : array_like, shape (*, 3)
        Axis-angle vectors (axis × angle_radians).  Zero vector maps to identity.

    Returns
    -------
    R : ndarray, shape (*, 3, 3)
        Rotation matrices.
    """
    aa_arr: npt.NDArray[np.float64] = np.asarray(aa, dtype=np.float64)
    single = (aa_arr.ndim == 1)
    if single:
        aa_arr = aa_arr[np.newaxis, :]

    batch_shape = aa_arr.shape[:-1]
    aa_flat = aa_arr.reshape(-1, 3)
    N = aa_flat.shape[0]

    angle = np.linalg.norm(aa_flat, axis=-1)  # (N,)

    # Normalise axis (safe against zero length)
    safe = angle > 1e-12
    axis = np.zeros_like(aa_flat)
    axis[safe] = aa_flat[safe] / angle[safe, np.newaxis]

    # Rodrigues: R = I + sin(θ) [k]× + (1 - cos θ) [k]×²
    # where [k]× is the skew-symmetric matrix of the unit axis k
    K = np.zeros((N, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] =  axis[:, 0]

    sin_a = np.sin(angle)[:, np.newaxis, np.newaxis]
    cos_a = np.cos(angle)[:, np.newaxis, np.newaxis]

    I = np.eye(3, dtype=np.float64)[np.newaxis, :, :]  # (1, 3, 3)
    R = I + sin_a * K + (1.0 - cos_a) * (K @ K)

    R = R.reshape(batch_shape + (3, 3))
    if single:
        return R[0]
    return R


# ============================================================================
# Euler angles <-> Axis-angle (convenience wrappers)
# ============================================================================

def euler_to_axisangle(
    angles: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert Euler angles to axis-angle vectors.

    Parameters
    ----------
    angles : array_like, shape (*, 3)
        Euler angles.
    order : str or list, e.g. 'ZYX'
        Rotation axis order.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    aa : ndarray, shape (*, 3)
        Axis-angle vectors (axis × angle_radians).
    """
    return rotmat_to_axisangle(euler_to_rotmat(angles, order, degrees=degrees))


def axisangle_to_euler(
    aa: npt.ArrayLike,
    order: Union[str, Sequence[str]],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Convert axis-angle vectors to Euler angles.

    Parameters
    ----------
    aa : array_like, shape (*, 3)
        Axis-angle vectors (axis × angle_radians).
    order : str or list, e.g. 'ZYX'
        Rotation axis order.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    angles : ndarray, shape (*, 3)
        Euler angles.
    """
    return rotmat_to_euler(axisangle_to_rotmat(aa), order, degrees=degrees)


# ============================================================================
# Quaternion SLERP
# ============================================================================

def quat_slerp(
    q1: npt.ArrayLike,
    q2: npt.ArrayLike,
    t: float | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation between quaternions.

    Parameters
    ----------
    q1 : array_like, shape (*, 4)
        Start quaternions (w, x, y, z).
    q2 : array_like, shape (*, 4)
        End quaternions (w, x, y, z).
    t : float or array_like
        Interpolation parameter(s) in [0, 1].

    Returns
    -------
    q : ndarray, shape (*, 4)
        Interpolated unit quaternions.
    """
    q1_arr: npt.NDArray[np.float64] = np.asarray(q1, dtype=np.float64)
    q2_arr: npt.NDArray[np.float64] = np.asarray(q2, dtype=np.float64)
    t_arr: npt.NDArray[np.float64] = np.asarray(t, dtype=np.float64)

    # Ensure shortest path: flip q2 if dot product is negative
    dot = np.sum(q1_arr * q2_arr, axis=-1, keepdims=True)
    q2_arr = np.where(dot < 0, -q2_arr, q2_arr)
    dot = np.abs(dot)

    # Clamp for numerical safety
    dot = np.clip(dot, 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # Replace zeros with 1.0 to avoid division warnings — the result
    # is discarded via np.where for these entries anyway.
    safe_sin = np.where(sin_theta < 1e-7, 1.0, sin_theta)

    # Near-identical quaternions: fall back to normalized lerp
    near_zero = (sin_theta.squeeze(-1) < 1e-7) if sin_theta.ndim > 1 else (sin_theta < 1e-7)
    near_zero = np.expand_dims(near_zero, -1) if near_zero.ndim < q1_arr.ndim else near_zero

    # Reshape t for broadcasting
    if t_arr.ndim == 0:
        t_val = float(t_arr)
        s1 = np.where(near_zero, 1.0 - t_val, np.sin((1.0 - t_val) * theta) / safe_sin)
        s2 = np.where(near_zero, t_val, np.sin(t_val * theta) / safe_sin)
    else:
        t_arr = t_arr[..., np.newaxis]  # add quaternion dim
        s1 = np.where(near_zero, 1.0 - t_arr, np.sin((1.0 - t_arr) * theta) / safe_sin)
        s2 = np.where(near_zero, t_arr, np.sin(t_arr * theta) / safe_sin)

    result = s1 * q1_arr + s2 * q2_arr
    # Normalize result
    result = result / np.linalg.norm(result, axis=-1, keepdims=True)
    return result


# ============================================================================
# String-dispatch converter
# ============================================================================

_VALID_REPRS = ("euler", "rotmat", "6d", "quaternion", "axisangle")


def convert(
    data: npt.ArrayLike,
    from_repr: str,
    to_repr: str,
    *,
    order: Union[str, Sequence[str], None] = None,
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Convert rotation data between representations via a string alias.

    Pivots through rotation matrices internally, so every pair of
    representations is reachable.  When ``from_repr`` or ``to_repr`` is
    ``"euler"``, the ``order`` argument is required (and accepts the
    same single-string / per-joint-sequence forms as
    :func:`euler_to_rotmat`).

    Parameters
    ----------
    data : array_like
        Input data. Shape depends on ``from_repr`` — see
        :data:`REPRESENTATION_CHANNELS` for the channel count.
    from_repr, to_repr : str
        One of ``"euler"``, ``"rotmat"``, ``"6d"``, ``"quaternion"``,
        ``"axisangle"``.
    order : str or sequence of strings, optional
        Euler rotation order(s).  Required when ``from_repr == "euler"``
        or ``to_repr == "euler"``; ignored otherwise.
    degrees : bool, optional
        Interpret/emit Euler angles in degrees (default ``False``).
        Ignored when neither side is Euler.

    Returns
    -------
    ndarray
        Converted data.  Shape depends on ``to_repr``.
    """
    if from_repr not in _VALID_REPRS:
        raise ValueError(f"from_repr {from_repr!r} not in {_VALID_REPRS}")
    if to_repr not in _VALID_REPRS:
        raise ValueError(f"to_repr {to_repr!r} not in {_VALID_REPRS}")

    if (from_repr == "euler" or to_repr == "euler") and order is None:
        raise ValueError("order= is required when from_repr or to_repr is 'euler'")

    # Identity
    if from_repr == to_repr:
        return np.asarray(data, dtype=np.float64).copy()

    # Step 1: lift to rotmat
    if from_repr == "rotmat":
        R = np.asarray(data, dtype=np.float64)
    elif from_repr == "euler":
        R = euler_to_rotmat(data, order, degrees=degrees)  # type: ignore[arg-type]
    elif from_repr == "6d":
        R = rot6d_to_rotmat(data)
    elif from_repr == "quaternion":
        R = quat_to_rotmat(data)
    else:  # "axisangle"
        R = axisangle_to_rotmat(data)

    # Step 2: project from rotmat
    if to_repr == "rotmat":
        return R
    if to_repr == "euler":
        return rotmat_to_euler(R, order, degrees=degrees)  # type: ignore[arg-type]
    if to_repr == "6d":
        return rotmat_to_rot6d(R)
    if to_repr == "quaternion":
        return rotmat_to_quat(R)
    return rotmat_to_axisangle(R)  # "axisangle"


# ============================================================================
# Internal helpers
# ============================================================================

def _normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize vectors along the last axis. Safe against zero-length."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


def _elementary_rotmat(angle: npt.NDArray[np.float64], axis: str) -> npt.NDArray[np.float64]:
    """
    Build elementary (single-axis) rotation matrices (batch).
    
    It constructs the standard Rx, Ry, or Rz matrices for a given angle.

    Parameters
    ----------
    angle : ndarray, shape (N,)
        Rotation angles in radians.
    axis : str
        One of 'X', 'Y', 'Z'.

    Returns
    -------
    R : ndarray, shape (*, 3, 3)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == 'X':
        R = np.stack([
            np.stack([one,  zero, zero], axis=-1),
            np.stack([zero, c,   -s],    axis=-1),
            np.stack([zero, s,    c],    axis=-1),
        ], axis=-2)
    elif axis == 'Y':
        R = np.stack([
            np.stack([c,    zero, s],    axis=-1),
            np.stack([zero, one,  zero], axis=-1),
            np.stack([-s,   zero, c],    axis=-1),
        ], axis=-2)
    elif axis == 'Z':
        R = np.stack([
            np.stack([c,   -s,   zero], axis=-1),
            np.stack([s,    c,   zero], axis=-1),
            np.stack([zero, zero, one],  axis=-1),
        ], axis=-2)
    else:
        raise ValueError(f"axis must be 'X', 'Y', or 'Z', got '{axis}'")

    return R


def _extract_euler(R: npt.NDArray[np.float64], i: int, j: int, k: int) -> npt.NDArray[np.float64]:
    """
    Extract Euler angles from rotation matrices for axes (i, j, k).

    Handles both Tait-Bryan (i != k) and proper Euler (i == k) sequences.

    Parameters
    ----------
    R : ndarray, shape (N, 3, 3)
    i, j, k : int
        Axis indices (0=X, 1=Y, 2=Z).

    Returns
    -------
    angles : ndarray, shape (N, 3)
    """
    N = R.shape[0]
    angles = np.empty((N, 3), dtype=np.float64)

    if i == k:
        # Proper Euler angles (e.g., ZYZ, XYX, ...)
        # Find the third axis: the one that is not i or j
        k_actual = 3 - i - j  # since {0,1,2} and we know i,j
        # But the user specified i==k, so the actual third axis in the
        # decomposition is i again. We use the proper Euler formula.
        # Sign factor for the cross-product parity
        sign = 1.0 if (j - i) % 3 == 2 else -1.0
        c2 = R[:, i, i]
        c2 = np.clip(c2, -1.0, 1.0)
        angles[:, 1] = np.arccos(c2)

        # Check for gimbal lock
        safe = np.abs(np.sin(angles[:, 1])) > 1e-7

        # Safe case
        angles[:, 0] = np.where(safe,
            np.arctan2(R[:, j, i], sign * R[:, k_actual, i]),
            0.0)
        angles[:, 2] = np.where(safe,
            np.arctan2(R[:, i, j], -sign * R[:, i, k_actual]),
            np.arctan2(sign * R[:, j, k_actual], R[:, j, j]))

        # The above k_actual is 3 - i - j; for proper Euler this is the
        # third distinct axis
    else:
        # Tait-Bryan angles (e.g., ZYX, XYZ, ...)
        # Sign factor: +1 if (i,j,k) is an even permutation of (0,1,2), else -1
        sign = 1.0 if (j - i) % 3 == 1 else -1.0

        # Middle angle from arcsin
        s2 = sign * R[:, i, k]
        s2 = np.clip(s2, -1.0, 1.0)
        angles[:, 1] = np.arcsin(s2)

        # Check for gimbal lock (cos(angle2) ≈ 0)
        safe = np.abs(np.cos(angles[:, 1])) > 1e-7

        # Safe case
        angles[:, 0] = np.where(safe,
            np.arctan2(-sign * R[:, j, k], R[:, k, k]),
            0.0)
        angles[:, 2] = np.where(safe,
            np.arctan2(-sign * R[:, i, j], R[:, i, i]),
            np.arctan2(sign * R[:, j, i], R[:, j, j]))

    return angles


# ============================================================================
# SE(3) rigid transforms — exp / log, screw interpolation, geodesic distance
# ============================================================================
#
# A rigid transform is a 4x4 homogeneous matrix ``T = [[R, d], [0, 1]]``.
# Its se(3) twist coordinates are ``ξ = [ω(3), v(3)]`` — **rotation-first**,
# with the translation part **V-Jacobian-coupled**: ``d = V(ω) · v`` (so ``v``
# is the screw's linear velocity, NOT the raw translation ``d`` unless ω = 0).
# This matches Modern Robotics / Vemulapalli 2014 / pytransform3d. The exp/log
# maps reuse the existing SO(3) Rodrigues (:func:`axisangle_to_rotmat`) and log
# (:func:`rotmat_to_axisangle`, which already handles the θ≈π branch).

_SE3_SMALL = 1e-3  # below this angle, use Taylor series for the V coefficients


def _skew(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Skew-symmetric matrices ``[w]×`` of vectors ``w`` — shape (*, 3, 3)."""
    K = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -w[..., 2]
    K[..., 0, 2] = w[..., 1]
    K[..., 1, 0] = w[..., 2]
    K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]
    K[..., 2, 1] = w[..., 0]
    return K


def _v_coeffs(theta: npt.NDArray[np.float64]) -> tuple[npt.NDArray, npt.NDArray]:
    """Coefficients ``b, c`` of the left-Jacobian ``V = I + b[ω]× + c[ω]×²``.

    ``b = (1−cosθ)/θ²``, ``c = (θ−sinθ)/θ³`` — Taylor-expanded below
    ``_SE3_SMALL`` so θ→0 stays finite and accurate.
    """
    small = theta < _SE3_SMALL
    safe = np.where(small, 1.0, theta)
    b_exact = (1.0 - np.cos(safe)) / safe ** 2
    c_exact = (safe - np.sin(safe)) / safe ** 3
    t2 = theta ** 2
    b_series = 0.5 - t2 / 24.0 + t2 ** 2 / 720.0
    c_series = 1.0 / 6.0 - t2 / 120.0 + t2 ** 2 / 5040.0
    return np.where(small, b_series, b_exact), np.where(small, c_series, c_exact)


def _vinv_coeff(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Coefficient ``e`` of ``V⁻¹ = I − ½[ω]× + e[ω]×²``.

    ``e = 1/θ² − cot(θ/2)/(2θ)``. Written via ``cot(θ/2)`` it is finite at
    θ = π (where ``cot(π/2) = 0``); Taylor-expanded below ``_SE3_SMALL``.
    """
    small = theta < _SE3_SMALL
    safe = np.where(small, 1.0, theta)
    half = safe / 2.0
    e_exact = 1.0 / safe ** 2 - np.cos(half) / (2.0 * safe * np.sin(half))
    t2 = theta ** 2
    e_series = 1.0 / 12.0 + t2 / 720.0 + t2 ** 2 / 30240.0
    return np.where(small, e_series, e_exact)


def se3_exp(twist: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Exponential map se(3) → SE(3): twist ``[ω, v]`` → 4×4 transform (batch).

    ``R = exp([ω]×)`` and ``d = V(ω) · v``, where ``V`` is the SO(3) left
    Jacobian. The linear part ``v`` is therefore screw-coupled, not the raw
    translation (they coincide only when ``ω = 0``).

    Parameters
    ----------
    twist : array_like, shape (*, 6)
        se(3) coordinates ``[ω(3), v(3)]``, rotation-first.

    Returns
    -------
    T : ndarray, shape (*, 4, 4)
        Homogeneous rigid transforms.

    See Also
    --------
    se3_log : Inverse map. screw_interpolate : SE(3) geodesic blend.

    Notes
    -----
    Source: Modern Robotics (Lynch & Park); Vemulapalli et al. 2014.
    """
    xi = np.asarray(twist, dtype=np.float64)
    single = xi.ndim == 1
    if single:
        xi = xi[np.newaxis, :]

    omega = xi[..., :3]
    v = xi[..., 3:]
    theta = np.linalg.norm(omega, axis=-1)
    R = axisangle_to_rotmat(omega)
    K = _skew(omega)
    b, c = _v_coeffs(theta)
    eye = np.eye(3, dtype=np.float64)
    V = eye + b[..., None, None] * K + c[..., None, None] * (K @ K)
    d = (V @ v[..., None])[..., 0]

    T = np.zeros(xi.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = d
    T[..., 3, 3] = 1.0
    return T[0] if single else T


def _so3_log(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotation-matrix log as axis-angle, via the quaternion (robust at θ≈π).

    The ``arccos((trace−1)/2)`` angle and the eigenvector axis used by
    :func:`rotmat_to_axisangle` are ill-conditioned near π (errors ~1e-4 at
    θ = π − 1e-6); the quaternion route stays machine-precise everywhere, so
    the SE(3) log and geodesic build on this. Returns ``axis × angle`` with
    angle in ``[0, π]``.
    """
    q = rotmat_to_quat(R)
    q = np.where(q[..., :1] < 0.0, -q, q)  # canonical w >= 0 -> angle in [0, π]
    vec = q[..., 1:]
    vec_norm = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, q[..., 0])
    # aa = (vec / ‖vec‖) · angle; the ratio tends to 2 at identity (bounded).
    safe = np.where(vec_norm > 1e-12, vec_norm, 1.0)
    scale = np.where(vec_norm > 1e-12, angle / safe, 0.0)
    return vec * scale[..., None]


def se3_log(transform: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Logarithm map SE(3) → se(3): 4×4 transform → twist ``[ω, v]`` (batch).

    Inverse of :func:`se3_exp`: ``ω = log(R)`` and ``v = V⁻¹(ω) · d``.

    Parameters
    ----------
    transform : array_like, shape (*, 4, 4)
        Homogeneous rigid transforms.

    Returns
    -------
    twist : ndarray, shape (*, 6)
        se(3) coordinates ``[ω(3), v(3)]``, rotation-first.

    Notes
    -----
    At a rotation angle of exactly π the axis sign is ambiguous (as for any
    SO(3) log); the round-trip ``se3_exp(se3_log(T)) == T`` holds regardless
    because ``v`` is recomputed consistently with whichever ω is chosen.

    Source: Modern Robotics; Vemulapalli et al. 2014.
    """
    T = np.asarray(transform, dtype=np.float64)
    single = T.ndim == 2
    if single:
        T = T[np.newaxis, ...]

    R = T[..., :3, :3]
    d = T[..., :3, 3]
    omega = _so3_log(R)
    theta = np.linalg.norm(omega, axis=-1)
    K = _skew(omega)
    e = _vinv_coeff(theta)
    eye = np.eye(3, dtype=np.float64)
    V_inv = eye - 0.5 * K + e[..., None, None] * (K @ K)
    v = (V_inv @ d[..., None])[..., 0]

    twist = np.concatenate([omega, v], axis=-1)
    return twist[0] if single else twist


def screw_interpolate(
    T0: npt.ArrayLike,
    T1: npt.ArrayLike,
    t: float,
) -> npt.NDArray[np.float64]:
    """Screw-motion interpolation between two rigid transforms.

    ``T0 · exp(t · log(T0⁻¹ T1))`` — the SE(3) geodesic, the rigid-transform
    analogue of quaternion SLERP. Rotation and translation advance together
    along a constant screw axis. ``t = 0`` returns ``T0``; ``t = 1`` returns
    ``T1``.

    Parameters
    ----------
    T0, T1 : array_like, shape (*, 4, 4)
        Endpoint transforms.
    t : float
        Interpolation parameter (typically in ``[0, 1]``, extrapolates
        outside).

    Returns
    -------
    ndarray, shape (*, 4, 4)
        The interpolated transform.

    Notes
    -----
    Source: Vemulapalli et al. 2014 (Lie-group skeletal features).
    """
    T0 = np.asarray(T0, dtype=np.float64)
    T1 = np.asarray(T1, dtype=np.float64)
    relative = se3_log(np.linalg.inv(T0) @ T1)
    return T0 @ se3_exp(t * relative)


def relative_transform(
    seg_m: npt.ArrayLike,
    seg_n: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Rigid transform of segment ``n`` in segment ``m``'s local frame.

    The geometry→SE(3) bridge: each segment (a pair of endpoint positions)
    defines a local coordinate frame — origin at its start, x-axis along the
    segment, the remaining axes completed orthonormally — and the result is
    ``T_m⁻¹ · T_n``, the pose of segment ``n`` relative to segment ``m``.
    Feed the resulting transforms to :func:`se3_log` for Lie-group features.

    Parameters
    ----------
    seg_m, seg_n : array_like, shape (*, 2, 3)
        Segment endpoint pairs ``[start, end]``; each must have nonzero
        length (coincident endpoints have no frame and yield ``nan``).

    Returns
    -------
    ndarray, shape (*, 4, 4)
        The relative rigid transform.

    Notes
    -----
    Source: Vemulapalli et al. 2014.
    """
    Tm = _segment_frame(np.asarray(seg_m, dtype=np.float64))
    Tn = _segment_frame(np.asarray(seg_n, dtype=np.float64))
    return np.linalg.inv(Tm) @ Tn


def _segment_frame(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """World←segment transform: origin at start, x along the segment.

    A zero-length segment (coincident endpoints) has no direction, hence no
    frame — those entries are filled with ``nan`` (guarded so no divide
    warning is raised).
    """
    start = seg[..., 0, :]
    end = seg[..., 1, :]
    x = end - start
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    defined = x_norm[..., 0] > 1e-12
    x = x / np.where(x_norm > 1e-12, x_norm, 1.0)
    # Reference axis = the world axis least aligned with x (stable cross).
    least = np.argmin(np.abs(x), axis=-1)
    ref = np.eye(3, dtype=np.float64)[least]
    y = np.cross(ref, x)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    y = y / np.where(y_norm > 1e-12, y_norm, 1.0)
    z = np.cross(x, y)

    T = np.zeros(seg.shape[:-2] + (4, 4), dtype=np.float64)
    T[..., :3, 0] = x
    T[..., :3, 1] = y
    T[..., :3, 2] = z
    T[..., :3, 3] = start
    T[..., 3, 3] = 1.0
    return np.where(defined[..., None, None], T, np.nan)


def rotation_geodesic_distance(
    R1: npt.ArrayLike,
    R2: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Geodesic (angular) distance between rotations, in radians (batch).

    ``‖log(R1ᵀ R2)‖`` — the angle of the relative rotation, the shortest
    arc on SO(3). Equivalent to ``2·arccos(|⟨q1, q2⟩|)`` on quaternions.
    Result is in ``[0, π]``.

    Parameters
    ----------
    R1, R2 : array_like, shape (*, 3, 3)
        Rotation matrices.

    Returns
    -------
    ndarray, shape (*)
        Geodesic distance in radians.

    Notes
    -----
    Source: Aristidou et al. 2017/2018 (orientation-space metrics).
    """
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    relative = np.swapaxes(R1, -1, -2) @ R2
    return np.linalg.norm(_so3_log(relative), axis=-1)
