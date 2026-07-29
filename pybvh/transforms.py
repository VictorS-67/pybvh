"""Spatial augmentation transforms for BVH motion data.

**Bvh-level API** — All transforms operate on :class:`~pybvh.bvh.Bvh`
objects and follow the ``inplace=False`` convention: by default they
return a new object, leaving the original unchanged.

**NumPy-level API** — Lower-level functions (``mirror_angles``,
``rotate_angles_vertical``) accept raw arrays + minimal metadata for
users who work with pre-extracted arrays.

All angle parameters are in **radians** (pybvh's internal convention);
functions that rotate accept a ``degrees=True`` flag matching the
convention in :mod:`~pybvh.rotations`.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING, overload

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhJoint
from . import rotations
from .tools import (
    _axis_aligned_rotation,
    _axis_index_sign,
    _compute_forward_at,
    _resolve_lr_pairs,
    _rest_leftward,
    _rest_upward,
    _validate_axis_string,
    _AXIS_CHAR_TO_IDX,
)

if TYPE_CHECKING:
    from .bvh import Bvh


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
    offset_range: tuple[float, float] = (-100.0, 100.0),
    rng: np.random.Generator | None = None,
) -> Bvh:
    """Translate root by a random offset sampled uniformly per axis.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    offset_range : tuple of (low, high)
        Uniform sampling range applied to each axis independently.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    Bvh
    """
    if rng is None:
        rng = np.random.default_rng()
    offset = rng.uniform(offset_range[0], offset_range[1], size=3)
    return translate_root(bvh, offset)  # type: ignore[return-value]


# =========================================================================
# 6.4  Joint Noise Injection
# =========================================================================

# Rotation noise and position noise are separate functions because their
# sigmas are in different units — radians and the skeleton's length unit.
# A single call taking both meant `degrees=` had to apply to one argument
# and not the other, which is the shape of a unit bug waiting to happen.

@overload
def add_rotation_noise(
    bvh: Bvh, sigma: float, *,
    rng: np.random.Generator | None = ..., inplace: Literal[True],
    wrap: bool = ..., degrees: bool = ...,
) -> None: ...
@overload
def add_rotation_noise(
    bvh: Bvh, sigma: float,
    rng: np.random.Generator | None = ..., inplace: Literal[False] = ...,
    wrap: bool = ..., degrees: bool = ...,
) -> Bvh: ...
def add_rotation_noise(
    bvh: Bvh,
    sigma: float,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
    wrap: bool = False,
    degrees: bool = False,
) -> Bvh | None:
    """Add zero-mean Gaussian noise to joint rotation angles.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    sigma : float
        Standard deviation of the rotation noise in **radians** (or
        degrees if ``degrees=True``), added to ``joint_angles``.
        ``0`` is a no-op and draws nothing from ``rng``.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.
    inplace : bool, optional
        If True, modify *bvh* and return None.
    wrap : bool, optional
        If True, wrap noised angles to ``[-π, π]``. Default False:
        BVH channels can legitimately hold values outside that range
        (accumulated rotations spanning multiple turns), and wrapping
        those would corrupt them.
    degrees : bool, optional
        If True, interpret ``sigma`` in degrees. Default False (radians).

    Returns
    -------
    Bvh or None

    See Also
    --------
    add_position_noise : The root-translation counterpart.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    sigma_rad = np.radians(sigma) if degrees else sigma
    if rng is None:
        rng = np.random.default_rng()

    target = bvh if inplace else bvh.copy()
    if sigma_rad > 0:
        noised = (
            target.joint_angles
            + rng.normal(0.0, sigma_rad, target.joint_angles.shape)
        )
        if wrap:
            noised = (noised + np.pi) % (2.0 * np.pi) - np.pi
        target.joint_angles = noised
    if inplace:
        return None
    return target


@overload
def add_position_noise(
    bvh: Bvh, sigma: float, *,
    rng: np.random.Generator | None = ..., inplace: Literal[True],
) -> None: ...
@overload
def add_position_noise(
    bvh: Bvh, sigma: float,
    rng: np.random.Generator | None = ..., inplace: Literal[False] = ...,
) -> Bvh: ...
def add_position_noise(
    bvh: Bvh,
    sigma: float,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
) -> Bvh | None:
    """Add zero-mean Gaussian noise to the root translation.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    sigma : float
        Standard deviation of the noise added to ``root_pos``, in the
        skeleton's length unit — the same unit as the bone offsets, so
        it scales with the file rather than being absolute. There is no
        ``degrees=`` here because this is a length, not an angle.
        ``0`` is a no-op and draws nothing from ``rng``.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None

    See Also
    --------
    add_rotation_noise : The joint-angle counterpart.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if rng is None:
        rng = np.random.default_rng()

    target = bvh if inplace else bvh.copy()
    if sigma > 0:
        target.root_pos = (
            target.root_pos + rng.normal(0.0, sigma, target.root_pos.shape)
        )
    if inplace:
        return None
    return target


# =========================================================================
# 6.3  Speed Perturbation
# =========================================================================

def perturb_speed(bvh: Bvh, factor: float) -> Bvh:
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
        New Bvh with an adjusted frame count and the **original**
        ``frame_time`` — the clip's duration changes, its playback rate
        does not.

    Notes
    -----
    This resamples: the motion is re-interpolated onto a new time grid
    (see :meth:`Bvh.resample`), so joint rotations pass through SLERP
    and the frame count changes. The alternative convention — scale
    ``frame_time`` by ``1 / factor`` and leave the samples untouched —
    is lossless and changes duration too, but yields a non-standard
    frame rate rather than a resampled clip. Use that directly when you
    want playback-rate change without interpolation.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    if bvh.frame_time <= 0:
        raise ValueError("Cannot resample: frame_time is 0.")
    original_fps = 1.0 / bvh.frame_time
    # Resample to fewer/more frames, then restore original frame rate.
    # factor > 1 → faster → fewer frames; factor < 1 → slower → more frames.
    result = bvh.resample(original_fps / factor)
    result.frame_time = bvh.frame_time
    return result


def random_perturb_speed(
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
    return perturb_speed(bvh, factor)


# =========================================================================
# 6.6  Frame Dropout with Interpolation
# =========================================================================

@overload
def drop_frames(
    bvh: Bvh, drop_rate: float, *, rng: np.random.Generator | None = ...,
    inplace: Literal[True],
) -> None: ...
@overload
def drop_frames(
    bvh: Bvh, drop_rate: float, rng: np.random.Generator | None = ...,
    inplace: Literal[False] = ...,
) -> Bvh: ...
def drop_frames(
    bvh: Bvh,
    drop_rate: float,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
) -> Bvh | None:
    """Replace randomly selected frames with SLERP-interpolated values.

    Dropped frames are filled by spherical linear interpolation
    (SLERP) of the nearest kept neighbours' quaternion rotations and
    linear interpolation of root positions.  The output has the **same**
    frame count as the input, and kept frames are preserved exactly
    (bit-for-bit) — only the dropped frames are re-synthesized.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    drop_rate : float
        Fraction of frames to drop, in ``[0, 1)``.  First and last
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
    drop_mask = ~keep_mask

    if not drop_mask.any():
        if inplace:
            return None
        return bvh.copy()

    # For every dropped frame, find left and right kept-neighbour indices.
    # searchsorted gives the insertion point in kept_indices.
    dropped = np.where(drop_mask)[0]
    ins = np.searchsorted(kept_indices, dropped, side='right')
    left_frames = kept_indices[ins - 1]   # (D,)
    right_frames = kept_indices[ins]      # (D,)
    alpha = (dropped - left_frames) / (right_frames - left_frames)  # (D,)

    # Interpolate root position linearly (dropped rows only)
    new_root_pos = bvh.root_pos.copy()
    new_root_pos[drop_mask] = (
        bvh.root_pos[left_frames] * (1.0 - alpha[:, None])
        + bvh.root_pos[right_frames] * alpha[:, None]
    )

    # SLERP all joints of the dropped frames in one broadcast call:
    # (D, J, 4) endpoints with (D, 1) t broadcasting over the joint axis.
    _, quats = bvh.to_quat()  # (F, J, 4)
    dropped_quats = rotations.quat_slerp(
        quats[left_frames], quats[right_frames], alpha[:, None])

    # Convert only the dropped frames back to Euler — kept frames never
    # round-trip through quaternions, so their angle values are untouched.
    new_angles = bvh.joint_angles.copy()
    new_angles[drop_mask] = rotations.quat_to_euler(
        dropped_quats, bvh.euler_orders)

    target = bvh if inplace else bvh.copy()
    target.root_pos = new_root_pos
    target.joint_angles = new_angles
    if inplace:
        return None
    return target


# =========================================================================
# NumPy-level API — rotate_angles_vertical
# =========================================================================

def _resolve_vertical_pivot(
    pivot: str | npt.ArrayLike,
    root_pos: npt.NDArray[np.float64],
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Resolve a ``pivot=`` argument to a ground-plane point ``(3,)``.

    The up-axis component is dropped from every form, including an
    explicit point: a vertical rotation about any point on a vertical
    line is the same rotation, so the component carries no information —
    and zeroing it keeps heights bit-identical through the
    translate-rotate-translate round trip instead of subtracting and
    re-adding an offset.
    """
    if isinstance(pivot, str):
        if pivot == "origin":
            return np.zeros(3)
        if pivot != "root":
            raise ValueError(
                f"pivot must be 'origin', 'root', or a (3,) point, "
                f"got {pivot!r}")
        if len(root_pos) == 0:
            raise ValueError(
                "pivot='root' needs at least one frame to read the root "
                "position from")
        point = np.array(root_pos[0], dtype=np.float64)
    else:
        point = np.array(pivot, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(
                f"an explicit pivot must have shape (3,), got {point.shape}")

    point[up_idx] = 0.0
    return point


def rotate_angles_vertical(
    joint_angles: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    angle: float,
    up_idx: int,
    root_order: str,
    degrees: bool = False,
    pivot: str | npt.ArrayLike = "origin",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate motion around the vertical axis (NumPy-level).

    Only modifies ``root_pos`` and the root joint's Euler angles
    (index 0 in ``joint_angles``).  Non-root joints are in parent-local
    coordinates and are unaffected.

    Parameters
    ----------
    joint_angles : ndarray of shape (F, J, 3)
        Euler angles **in radians** (pybvh's internal convention).
    root_pos : ndarray of shape (F, 3)
        Root translation per frame.
    angle : float
        Rotation angle in radians (or degrees if ``degrees=True``).
    up_idx : int
        Index of the up axis (0=X, 1=Y, 2=Z).
    root_order : str
        Euler order of the root joint, e.g. ``'ZYX'``.
    degrees : bool, optional
        If True, interpret ``angle`` in degrees. Default False (radians).
    pivot : {"origin", "root"} or array-like of shape (3,), optional
        World-space point the motion turns about. ``"origin"`` (default)
        turns about the world origin, so a motion standing away from it
        sweeps through space along an arc; ``"root"`` turns about the
        first frame's root position, i.e. turn-in-place; an explicit
        ``(3,)`` point turns about a fixed landmark. Only the two
        horizontal components are read — every point on a vertical line
        spans the same rotation axis — so ``"root"`` is equivalently the
        first-frame root projected to the ground plane.

    Returns
    -------
    (new_joint_angles, new_root_pos)
        Copies with the rotation applied. Angles in radians.

    Notes
    -----
    The pivot only translates the root trajectory: rotating about ``p``
    is the same as ``root_pos - p`` → rotate about the origin →
    ``+ p``, and the root's world rotation is identical either way. So a
    pipeline that already centers its clips on the first-frame root gets
    turn-in-place from the default ``"origin"`` and needs no ``pivot=``.

    See Also
    --------
    rotate_vertical : Bvh-level wrapper that auto-detects ``up_idx``
        and ``root_order`` from the skeleton.

    Examples
    --------
    >>> angles = bvh.joint_angles          # (F, J, 3) radians
    >>> pos = bvh.root_pos                 # (F, 3)
    >>> up = {'x': 0, 'y': 1, 'z': 2}[bvh.world_up[1]]
    >>> order = ''.join(bvh.root.rot_channels)
    >>> new_angles, new_pos = rotate_angles_vertical(
    ...     angles, pos, np.pi / 2, up, order)
    """
    angle_rad = np.radians(angle) if degrees else angle
    pivot_point = _resolve_vertical_pivot(pivot, root_pos, up_idx)
    R_vert: npt.NDArray[np.float64] = rotations._elementary_rotmat(
        angle_rad, 'XYZ'[up_idx])

    new_root_pos = (R_vert @ (root_pos - pivot_point).T).T + pivot_point

    new_angles = joint_angles.copy()
    R_root = rotations.euler_to_rotmat(joint_angles[:, 0], root_order)
    R_new = R_vert[np.newaxis] @ R_root
    new_angles[:, 0] = rotations.rotmat_to_euler(R_new, root_order)

    return new_angles, new_root_pos


# =========================================================================
# 6.2  Vertical Rotation
# =========================================================================

@overload
def rotate_vertical(
    bvh: Bvh, angle: float, *, up_axis: str | None = ...,
    degrees: bool = ..., pivot: str | npt.ArrayLike = ...,
    inplace: Literal[True],
) -> None: ...
@overload
def rotate_vertical(
    bvh: Bvh, angle: float, up_axis: str | None = ...,
    degrees: bool = ..., pivot: str | npt.ArrayLike = ...,
    inplace: Literal[False] = ...,
) -> Bvh: ...
def rotate_vertical(
    bvh: Bvh,
    angle: float,
    up_axis: str | None = None,
    degrees: bool = False,
    pivot: str | npt.ArrayLike = "origin",
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
    angle : float
        Rotation angle in radians (positive = counter-clockwise
        when viewed from above). Pass ``degrees=True`` for degrees.
    up_axis : str or None
        Signed axis string (e.g. ``'+y'``).  Auto-detected if None.
    degrees : bool, optional
        If True, interpret ``angle`` in degrees. Default False (radians).
    pivot : {"origin", "root"} or array-like of shape (3,), optional
        World-space point the motion turns about. Default ``"origin"``,
        the world origin: a character standing away from it sweeps
        through space along an arc rather than turning where it stands.
        ``"root"`` is that turn-in-place — it pivots about the first
        frame's root position, projected to the ground plane. An
        explicit ``(3,)`` point pivots about a fixed landmark (a dataset
        capture centre, a stage mark). Only the two horizontal
        components of a point are read: every point on a vertical line
        spans the same rotation axis, so the up-axis component is
        dropped and heights come through untouched.
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None

    Notes
    -----
    The pivot moves only the root trajectory — the root's world
    rotation, and every child joint's parent-local angles, are identical
    for any pivot. Equivalently, ``pivot="root"`` is *center on the
    first-frame root → rotate about the origin → un-center*: a pipeline
    that already centers its clips gets turn-in-place from the default
    and needs no ``pivot=`` at all.

    Randomized pivots are deliberately not offered here (unlike
    ``angle``, which has :func:`random_rotate_vertical`). A random pivot
    is a rotation composed with a translation, so it is already
    expressible as ``rotate_vertical`` + :func:`translate_root`, and
    which distribution to draw from is a pipeline's decision, not a
    motion's.
    """
    target = bvh if inplace else bvh.copy()

    # Determine up axis — use the Bvh's world_up property so manual
    # overrides are respected.
    if up_axis is None:
        axis_str = target.world_up
    else:
        axis_str = _validate_axis_string(up_axis)
    up_idx, up_sign = _axis_index_sign(axis_str)

    root_order = "".join(target.root.rot_channels)
    new_angles, new_root_pos = rotate_angles_vertical(
        target.joint_angles, target.root_pos, angle * up_sign, up_idx,
        root_order, degrees=degrees, pivot=pivot,
    )
    target.joint_angles = new_angles
    target.root_pos = new_root_pos

    if inplace:
        return None
    return target


def random_rotate_vertical(
    bvh: Bvh,
    angle_range: tuple[float, float] = (-np.pi, np.pi),
    up_axis: str | None = None,
    degrees: bool = False,
    pivot: str | npt.ArrayLike = "origin",
    rng: np.random.Generator | None = None,
) -> Bvh:
    """Rotate motion by a random angle around the vertical axis.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    angle_range : tuple of (low, high)
        Angle sampling range in radians (default full circle,
        ``(-π, π)``), or in degrees if ``degrees=True``.
    up_axis : str or None
        Signed axis string.  Auto-detected if None.
    degrees : bool, optional
        If True, interpret ``angle_range`` in degrees. Default False
        (radians).
    pivot : {"origin", "root"} or array-like of shape (3,), optional
        World-space point to turn about, as in :func:`rotate_vertical`.
        Only the *angle* is random; the pivot is fixed for the call.
    rng : numpy.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    Bvh
    """
    if rng is None:
        rng = np.random.default_rng()
    angle = float(rng.uniform(angle_range[0], angle_range[1]))
    return rotate_vertical(  # type: ignore[return-value]
        bvh, angle, up_axis=up_axis, degrees=degrees, pivot=pivot)


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
        Euler angles **in radians** (pybvh's internal convention).
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
    >>> pairs = transforms.auto_detect_lr_pairs(bvh)
    >>> lat_idx = {'x': 0, 'y': 1, 'z': 2}[bvh.left_at(frame=0)[1]]
    >>> channels = [n.rot_channels for n in bvh.nodes if not n.is_end_site()]
    >>> new_angles, new_pos = mirror_angles(
    ...     angles, pos, pairs, lat_idx, channels)
    """
    new_angles = joint_angles.copy()
    new_root_pos = root_pos.copy()

    # Negate root_pos lateral component
    new_root_pos[:, lateral_idx] *= -1

    # Negate Euler components whose rotation axis is NOT the lateral axis.
    # This MUST happen before the L/R swap so each joint's own Euler order
    # is used for the negation (not the swapped destination's order).
    lateral_upper = "XYZ"[lateral_idx]
    for j_idx, channels in enumerate(rot_channels):
        for ch_idx, ch in enumerate(channels):
            if ch != lateral_upper:
                new_angles[:, j_idx, ch_idx] *= -1

    # Swap L/R joint angle columns
    for lj, rj in lr_joint_pairs:
        left_data = new_angles[:, lj].copy()
        new_angles[:, lj] = new_angles[:, rj]
        new_angles[:, rj] = left_data

    return new_angles, new_root_pos


# =========================================================================
# 6.1  Left-Right Mirroring
# =========================================================================

def auto_detect_lr_pairs(bvh: Bvh) -> list[tuple[int, int]]:
    """Auto-detect left/right joint pairs as index tuples.

    Converts the joint name pairs of :attr:`Bvh.lr_mapping` to index
    pairs in ``joint_angles`` index space (axis 1 of
    ``bvh.joint_angles``).

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
    return _resolve_lr_pairs(bvh.lr_mapping, bvh.joint_index)


@overload
def mirror(
    bvh: Bvh, *, lr_mapping: dict[str, str] | None = ...,
    lateral_axis: str | None = ..., inplace: Literal[True],
) -> None: ...
@overload
def mirror(
    bvh: Bvh, lr_mapping: dict[str, str] | None = ...,
    lateral_axis: str | None = ..., inplace: Literal[False] = ...,
) -> Bvh: ...
def mirror(
    bvh: Bvh,
    lr_mapping: dict[str, str] | None = None,
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
    lr_mapping : dict or None
        ``{"LeftArm": "RightArm", ...}``.  Defaults to
        :attr:`Bvh.lr_mapping` (auto-detected or user-set).
    lateral_axis : str or None
        Axis perpendicular to the mirror plane, e.g. ``'x'`` or
        ``'+x'`` (the sign is irrelevant for mirroring).
        **Auto-detected if None**, and the detection is usually the
        right thing to use: it averages the left-minus-right rest-pose
        offsets over the L/R joint pairs and takes the dominant axis, so
        it measures how *this* skeleton is actually built rather than
        assuming a convention. Pass an explicit axis only to override
        that, or when the detection raises because the L/R offsets are
        degenerate (parallel to the up axis, or zero).
    inplace : bool, optional
        If True, modify *bvh* and return None.

    Returns
    -------
    Bvh or None

    Raises
    ------
    ValueError
        If no left/right pairs are available, or if an explicitly
        passed ``lr_mapping`` names joints that don't exist.
    """
    target = bvh if inplace else bvh.copy()

    # --- Detect L/R pairs (first, so we can fail fast with a clear error) ---
    explicit_mapping = lr_mapping is not None
    if lr_mapping is None:
        lr_mapping = target.lr_mapping

    if not lr_mapping:
        raise ValueError(
            "No L/R joint pairs available for mirror(). The extended "
            "name heuristic did not recognize this skeleton's convention. "
            "Provide a mapping explicitly — either set `bvh.lr_mapping = "
            "{...}` after loading, or pass `lr_mapping=` at load time "
            "(`read_bvh_file(..., lr_mapping=...)`). If your skeleton is "
            "not bilaterally symmetric, mirroring does not apply.")

    # Build joint-index pairs (indices into joint_angles axis 1).
    # An explicitly passed mapping is resolved strictly so typos raise
    # instead of producing silently half-mirrored output; the cached
    # `Bvh.lr_mapping` was already validated on assignment.
    joints = [n for n in target.nodes if isinstance(n, BvhJoint)]
    j_name2idx = {j.name: i for i, j in enumerate(joints)}
    lr_j_pairs = _resolve_lr_pairs(
        lr_mapping, j_name2idx, strict=explicit_mapping)

    # --- Detect lateral axis ---
    # Mirror is a topology operation (swap L/R joints, negate the lateral
    # component), so we use the rest-pose leftward axis derived from the
    # L/R mapping above. Pass the explicit mapping so we use the same
    # pairs. Only the axis letter is consumed; the sign doesn't matter
    # for mirroring (reflecting across the plane perpendicular to the
    # axis is the same operation whether the axis points left or right).
    if lateral_axis is None:
        rest_left = _rest_leftward(target, mapping=lr_mapping)
        if rest_left is None:
            raise ValueError(
                "Cannot infer lateral axis from L/R mapping: averaged "
                "left-to-right offsets are degenerate (parallel to up "
                "axis or zero). Pass `lateral_axis=` explicitly.")
        lateral_char = rest_left[1]
    else:
        lateral_char = _validate_axis_string(
            lateral_axis, allow_unsigned=True)[1]
    lateral_idx = _AXIS_CHAR_TO_IDX[lateral_char]

    # Build node-index pairs for offset swapping (includes end sites)
    node_name2idx = target.node_index
    lr_node_pairs = _resolve_lr_pairs(lr_mapping, node_name2idx)
    end_site_pairs: list[tuple[int, int]] = []
    for li, ri in lr_node_pairs:
        left_node = target.nodes[li]
        right_node = target.nodes[ri]
        if isinstance(left_node, BvhJoint) and isinstance(right_node, BvhJoint):
            left_ends = [c for c in left_node.children if c.is_end_site()]
            right_ends = [c for c in right_node.children if c.is_end_site()]
            if len(left_ends) != len(right_ends):
                raise ValueError(
                    f"Cannot pair end sites for mirror(): joint "
                    f"{left_node.name!r} has {len(left_ends)} end site(s) "
                    f"but its L/R partner {right_node.name!r} has "
                    f"{len(right_ends)}.")
            for left_end, right_end in zip(left_ends, right_ends):
                end_site_pairs.append((node_name2idx[left_end.name],
                                       node_name2idx[right_end.name]))
    lr_node_pairs.extend(end_site_pairs)

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


# =========================================================================
# 8.  Coordinate-frame reorientation
# =========================================================================

def _apply_similarity_to_joints(
    angles: npt.NDArray[np.float64],
    joints: list,
    R_left: npt.NDArray[np.float64],
    R_right: npt.NDArray[np.float64],
    joint_indices=None,
) -> None:
    """Apply ``R_j' = R_left @ R_j @ R_right`` to selected joints, in place.

    Uses the per-joint-order overload of ``euler_to_rotmat`` /
    ``rotmat_to_euler`` so both conversions vectorize across the
    selected joint axis in one call.
    """
    if joint_indices is None:
        joint_indices = list(range(len(joints)))
    else:
        joint_indices = list(joint_indices)

    sel = np.asarray(joint_indices)
    per_joint = ["".join(joints[j].rot_channels) for j in joint_indices]  # type: ignore[attr-defined]

    block = angles[:, sel]                                              # (F, G, 3)
    R_j = rotations.euler_to_rotmat(block, per_joint)     # (F, G, 3, 3)
    R_j_new = R_left @ R_j @ R_right
    angles[:, sel] = rotations.rotmat_to_euler(R_j_new, per_joint)


def _reorient_rest(target: Bvh, R_fix: npt.NDArray[np.float64]) -> None:
    """Rotate rest-pose offsets by ``R_fix``, compensating rotations in place.

    Shared core of :func:`reorient_rest_up` and
    :func:`reorient_rest_forward`.  FK positions are unchanged:
    the root rotation absorbs the inverse (``R_root' = R_root @ R_fixᵀ``)
    and every other joint gets the similarity transform
    ``R_j' = R_fix @ R_j @ R_fixᵀ``.  ``root_pos`` and ``world_up`` are
    not touched (the world frame is unchanged).
    """
    R_fix_inv = R_fix.T  # orthogonal → inverse = transpose

    # 1. Rotate all offsets
    for node in target.nodes:
        node.offset = R_fix @ node.offset

    # 2. Compensate joint rotations so FK positions are unchanged
    joints = [n for n in target.nodes if not n.is_end_site()]
    angles_copy = target.joint_angles.copy()

    # Root (index 0): right-multiply by R_fix_inv only
    root_order = "".join(joints[0].rot_channels)  # type: ignore[attr-defined]
    R_root = rotations.euler_to_rotmat(angles_copy[:, 0], root_order)
    angles_copy[:, 0] = rotations.rotmat_to_euler(
        R_root @ R_fix_inv, root_order)

    # All other joints: full similarity
    _apply_similarity_to_joints(
        angles_copy, joints, R_fix, R_fix_inv,
        joint_indices=range(1, len(joints)))

    target.joint_angles = angles_copy


@overload
def reorient_world_up(bvh: Bvh, new_up: str, *, inplace: Literal[True]) -> None: ...
@overload
def reorient_world_up(bvh: Bvh, new_up: str, inplace: Literal[False] = ...) -> Bvh: ...
def reorient_world_up(bvh: Bvh, new_up: str, inplace: bool = False) -> Bvh | None:
    """Change the world coordinate system's vertical axis.

    Applies a global rotation to the entire animation (root translation,
    skeleton offsets, root joint rotations) so the world vertical axis
    changes from the current ``bvh.world_up`` to ``new_up``.  The character
    looks visually identical; only the coordinate system changes.

    Restricted to axis-aligned rotations (multiples of 90 degrees) for
    lossless transformation.

    Parameters
    ----------
    bvh : Bvh
    new_up : str
        Target up axis, e.g. ``'+y'``.
    inplace : bool

    Returns
    -------
    Bvh or None
    """
    old_up = _validate_axis_string(bvh.world_up)
    new_up = _validate_axis_string(new_up)

    target = bvh if inplace else bvh.copy()

    if old_up == new_up:
        return None if inplace else target

    R = _axis_aligned_rotation(old_up, new_up)

    # 1. Rotate root positions
    target.root_pos = (R @ target.root_pos.T).T

    # 2. Rotate all node offsets
    for node in target.nodes:
        node.offset = R @ node.offset

    # 3. Conjugate ALL joint rotations: R_j' = R @ R_j @ R^T
    #    This is the correct formula for a global scene rotation applied to
    #    the BVH hierarchy (both offsets and rotations change together).
    joints = [n for n in target.nodes if not n.is_end_site()]
    angles_copy = target.joint_angles.copy()
    _apply_similarity_to_joints(angles_copy, joints, R, R.T)
    target.joint_angles = angles_copy

    # 4. Update metadata
    target.world_up = new_up

    return None if inplace else target


@overload
def reorient_rest_up(bvh: Bvh, new_up: str, *, inplace: Literal[True]) -> None: ...
@overload
def reorient_rest_up(bvh: Bvh, new_up: str, inplace: Literal[False] = ...) -> Bvh: ...
def reorient_rest_up(bvh: Bvh, new_up: str, inplace: bool = False) -> Bvh | None:
    """Rotate the skeleton's rest-pose offsets so its topological up aligns
    with ``new_up``, compensating all joint rotations so that FK positions
    are unchanged.

    This fixes files where the rest pose and animation disagree on the up
    axis (e.g. rest pose authored in Y-up but animation plays in Z-up).
    After this call, the disagreement warning disappears.

    The world coordinate system is unchanged: ``root_pos`` and ``world_up``
    are NOT modified.

    Parameters
    ----------
    bvh : Bvh
    new_up : str
        Target rest-pose up axis, e.g. ``'+y'``.
    inplace : bool

    Returns
    -------
    Bvh or None

    Raises
    ------
    ValueError
        If the rest pose is degenerate (no up axis can be inferred).
    """
    current_rest_up = _rest_upward(bvh)
    if current_rest_up is None:
        raise ValueError(
            "Cannot infer the skeleton's rest-pose up axis: the rest "
            "pose is degenerate (all joint offsets are zero or there "
            "are too few joints).")
    new_up = _validate_axis_string(new_up)

    target = bvh if inplace else bvh.copy()

    if current_rest_up == new_up:
        return None if inplace else target

    _reorient_rest(target, _axis_aligned_rotation(current_rest_up, new_up))
    return None if inplace else target


@overload
def reorient_rest_forward(bvh: Bvh, new_forward: str, *, inplace: Literal[True]) -> None: ...
@overload
def reorient_rest_forward(bvh: Bvh, new_forward: str, inplace: Literal[False] = ...) -> Bvh: ...
def reorient_rest_forward(bvh: Bvh, new_forward: str, inplace: bool = False) -> Bvh | None:
    """Rotate the skeleton's rest-pose offsets so the character faces
    ``new_forward``, compensating all joint rotations so FK positions are
    unchanged.

    The rotation is around the world up axis (a rotation in the ground
    plane).  ``new_forward`` must not be parallel to ``world_up``.

    Parameters
    ----------
    bvh : Bvh
    new_forward : str
        Target rest-pose forward axis, e.g. ``'+y'`` or ``'-z'``.
    inplace : bool

    Returns
    -------
    Bvh or None

    Raises
    ------
    ValueError
        If ``new_forward`` is parallel to ``world_up``.
    """
    new_forward = _validate_axis_string(new_forward)
    up_idx = bvh.up_axis.index
    fwd_idx = _AXIS_CHAR_TO_IDX[new_forward[1]]
    if up_idx == fwd_idx:
        raise ValueError(
            f"new_forward ({new_forward}) cannot be parallel to "
            f"world_up ({bvh.world_up})")

    # Determine current rest-pose forward from topology
    rest_coords = bvh.rest_pose_positions()
    current_fwd = _compute_forward_at(bvh, rest_coords, bvh.world_up)

    target = bvh if inplace else bvh.copy()

    if current_fwd == new_forward:
        return None if inplace else target

    _reorient_rest(target, _axis_aligned_rotation(current_fwd, new_forward))
    return None if inplace else target
