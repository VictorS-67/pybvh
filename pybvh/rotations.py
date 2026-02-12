"""
Rotation representation conversions for skeleton-based motion data.

All functions are batch-vectorized using NumPy and operate on arrays
where the leading dimensions are batch dimensions.

Supported representations:
- Euler angles: (*, 3) in degrees or radians
- Rotation matrices: (*, 3, 3)
- 6D rotation (Zhou et al., CVPR 2019): (*, 6) — continuous representation
- Quaternions: (*, 4) in (w, x, y, z) scalar-first convention
- Axis-angle: (*, 3) — rotation axis scaled by rotation angle in radians

Convention note:
    Euler angles in BVH files use intrinsic rotations with pre-multiplication:
        R = R_first @ R_second @ R_third
    where the order comes from the joint's rot_channels (e.g., ['Z','Y','X']).
    Angles are in degrees in BVH files, but most functions here work in radians
    unless stated otherwise.
"""

import numpy as np


# ============================================================================
# Euler angles <-> Rotation matrices
# ============================================================================

def euler_to_rotmat(angles, order, degrees=False):
    """
    Convert Euler angles to rotation matrices (batch).

    Parameters
    ----------
    angles : array_like, shape (*, 3)
        Euler angles. Each row is (angle1, angle2, angle3) following the
        axis order given by `order`.
    order : str or list of 3 chars
        Rotation order, e.g. 'ZYX' or ['Z', 'Y', 'X'].
        Uses intrinsic rotation with pre-multiplication: R = R1 @ R2 @ R3.
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    R : ndarray, shape (*, 3, 3)
        Rotation matrices.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if degrees:
        angles = np.radians(angles)

    single = (angles.ndim == 1)
    if single:
        angles = angles[np.newaxis, :]  # (1, 3)

    order_str = ''.join(order).upper()
    if len(order_str) != 3 or not all(c in 'XYZ' for c in order_str):
        raise ValueError(f"order must be 3 characters from 'XYZ', got '{order_str}'")

    R = _elementary_rotmat(angles[:, 0], order_str[0])
    R = R @ _elementary_rotmat(angles[:, 1], order_str[1])
    R = R @ _elementary_rotmat(angles[:, 2], order_str[2])

    if single:
        return R[0]
    return R


def rotmat_to_euler(R, order, degrees=False):
    """
    Convert rotation matrices to Euler angles (batch).

    Uses the convention of intrinsic rotations with pre-multiplication.
    Handles gimbal lock by setting the third angle to 0.

    Parameters
    ----------
    R : array_like, shape (*, 3, 3)
        Rotation matrices.
    order : str or list of 3 chars
        Rotation order, e.g. 'ZYX' or ['Z', 'Y', 'X'].
    degrees : bool
        If True, Euler angles are in degrees. Default False (radians).

    Returns
    -------
    angles : ndarray, shape (*, 3)
        Euler angles in the specified order.
    """
    R = np.asarray(R, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, :, :]  # (1, 3, 3)

    order_str = ''.join(order).upper()
    if len(order_str) != 3 or not all(c in 'XYZ' for c in order_str):
        raise ValueError(f"order must be 3 characters from 'XYZ', got '{order_str}'")

    # Strategy: decompose R = R1(a1) @ R2(a2) @ R3(a3)
    # We invert this by finding a1, a2, a3 from R.
    # This is done by mapping axis letters to indices and using
    # the known structure of the composite matrix.
    ax2idx = {'X': 0, 'Y': 1, 'Z': 2}
    i = ax2idx[order_str[0]]
    j = ax2idx[order_str[1]]
    k = ax2idx[order_str[2]]

    angles = _extract_euler(R, i, j, k)

    if degrees:
        angles = np.degrees(angles)

    if single:
        return angles[0]
    return angles


# ============================================================================
# Rotation matrices <-> 6D representation (Zhou et al., CVPR 2019)
# ============================================================================

def rotmat_to_rot6d(R):
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
    R = np.asarray(R, dtype=np.float64)
    # Take first two columns: R[..., :, 0] and R[..., :, 1]
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_rotmat(rot6d):
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
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]

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

def euler_to_rot6d(angles, order, degrees=False):
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


def rot6d_to_euler(rot6d, order, degrees=False):
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

def rotmat_to_quat(R):
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
    R = np.asarray(R, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, :, :]

    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
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


def quat_to_rotmat(q):
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
    q = np.asarray(q, dtype=np.float64)
    single = (q.ndim == 1)
    if single:
        q = q[np.newaxis, :]

    # Normalize
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
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

def euler_to_quat(angles, order, degrees=False):
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


def quat_to_euler(q, order, degrees=False):
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

def rotmat_to_axisangle(R):
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
    R = np.asarray(R, dtype=np.float64)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, :, :]

    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
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


def axisangle_to_rotmat(aa):
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
    aa = np.asarray(aa, dtype=np.float64)
    single = (aa.ndim == 1)
    if single:
        aa = aa[np.newaxis, :]

    batch_shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)
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

def euler_to_axisangle(angles, order, degrees=False):
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


def axisangle_to_euler(aa, order, degrees=False):
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
# Internal helpers
# ============================================================================

def _normalize(v):
    """Normalize vectors along the last axis. Safe against zero-length."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


def _elementary_rotmat(angle, axis):
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
    R : ndarray, shape (N, 3, 3)
    """
    N = angle.shape[0]
    c = np.cos(angle)
    s = np.sin(angle)
    one = np.ones(N, dtype=np.float64)
    zero = np.zeros(N, dtype=np.float64)

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


def _extract_euler(R, i, j, k):
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
        sign = 1.0 if (j - i) % 3 == 1 else -1.0
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
            np.arctan2(-sign * R[:, j, k_actual], R[:, j, j]))

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
