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
from .bvhnode import BvhJoint
from . import rotations
from .tools import (
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
def add_noise(
    bvh: Bvh, sigma_deg: float, *, sigma_pos: float = ...,
    rng: np.random.Generator | None = ..., inplace: Literal[True],
) -> None: ...
@overload
def add_noise(
    bvh: Bvh, sigma_deg: float, sigma_pos: float = ...,
    rng: np.random.Generator | None = ..., inplace: Literal[False] = ...,
) -> Bvh: ...
def add_noise(
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
        If True (default), wrap noised angles to [-π, π] (radians) so
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
        # sigma_deg is a user-facing degrees-of-noise contract; convert
        # to radians once because joint_angles is internally radians.
        sigma_rad = np.deg2rad(sigma_deg)
        noised = (
            target.joint_angles
            + rng.normal(0.0, sigma_rad, target.joint_angles.shape)
        )
        if wrap:
            # Wrap to [-π, π] (radian equivalent of the old [-180°, 180°]).
            noised = (noised + np.pi) % (2.0 * np.pi) - np.pi
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
        New Bvh with adjusted frame count and ``frame_time``.
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
    root_pos_orig, quats_orig = bvh.to_quaternions()
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
        new_angles[:, j_idx] = rotations.rotmat_to_euler(R, order)

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
        Euler angles **in radians** (pybvh's internal convention).
    root_pos : ndarray of shape (F, 3)
        Root translation per frame.
    angle_deg : float
        Rotation angle in degrees (user-facing — converted to radians
        internally).
    up_idx : int
        Index of the up axis (0=X, 1=Y, 2=Z).
    root_order : str
        Euler order of the root joint, e.g. ``'ZYX'``.

    Returns
    -------
    (new_joint_angles, new_root_pos)
        Copies with the rotation applied. Angles in radians.

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
    ...     angles, pos, 90.0, up, order)
    """
    angle_rad = np.radians(angle_deg)
    rot_funcs = {0: rotX, 1: rotY, 2: rotZ}
    R_vert: npt.NDArray[np.float64] = rot_funcs[up_idx](angle_rad)

    new_root_pos = (R_vert @ root_pos.T).T

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

    # Determine up axis — use the Bvh's world_up property so manual
    # overrides are respected.
    if up_axis is None:
        axis_str = target.world_up
    else:
        axis_str = up_axis
    up_sign = 1 if axis_str[0] == '+' else -1
    up_idx = {'x': 0, 'y': 1, 'z': 2}[axis_str[1]]

    root_order = "".join(target.root.rot_channels)
    new_angles, new_root_pos = rotate_angles_vertical(
        target.joint_angles, target.root_pos, angle_deg * up_sign, up_idx, root_order,
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

_NUMBER_SUFFIX_RE = __import__('re').compile(r'\.\d+$')


def _strip_number_suffix(name: str) -> str:
    """Strip trailing ``.NNN`` suffix like Blender's ``.001`` duplicates."""
    return _NUMBER_SUFFIX_RE.sub('', name)


def _strip_namespace_prefix(name: str) -> tuple[str, str]:
    """Strip a ``foo:`` namespace prefix (Mixamo and similar).

    Returns ``(prefix, base)``. If no namespace, prefix is empty.
    """
    if ':' in name:
        prefix, _, base = name.rpartition(':')
        return prefix + ':', base
    return '', name


_SUFFIX_LR_TABLE = [
    # (left_suffix, right_suffix) — matched against the trailing substring
    # of the base name (after namespace and number-suffix strip). Ordered
    # most-specific-first so e.g. ".Left" / ".Right" wins over ".L" / ".R"
    # if both would parse.
    ('.Left', '.Right'), ('_Left', '_Right'),
    ('.left', '.right'), ('_left', '_right'),
    ('.L', '.R'), ('_L', '_R'),
    ('.l', '.r'), ('_l', '_r'),
]


def _try_suffix_partner(base: str, joint_names: set[str], prefix: str,
                        number_suffix: str) -> str | None:
    """If ``base`` matches a known L/R suffix, return the partner's full name."""
    for left_suf, right_suf in _SUFFIX_LR_TABLE:
        if base.endswith(left_suf):
            partner_base = base[: -len(left_suf)] + right_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
        elif base.endswith(right_suf):
            partner_base = base[: -len(right_suf)] + left_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
    return None


def _detect_lr_mapping_by_names(bvh: Bvh) -> dict[str, str]:
    """Internal: detect L/R joint pairs by extended name heuristics.

    Called at ``Bvh.__init__`` to populate ``bvh.lr_mapping`` eagerly.
    Rules tried most-specific-first so delimited suffixes win over bare
    substring matches:

    1. Delimited suffix — ``arm.L`` / ``arm.R``, ``arm_L`` / ``arm_R``,
       and their lowercase and ``.Left`` / ``_Left`` variants. Names are
       normalized by stripping a ``foo:`` namespace prefix (Mixamo's
       ``mixamorig:``) and a trailing ``.NNN`` number suffix (Blender's
       ``.001`` duplicates) before the suffix match; the partner is
       rebuilt with the same prefix/number suffix.
    2. Substring ``left`` / ``right`` (case-insensitive) — handles
       ``LeftArm`` / ``RightArm``, ``leftArm`` / ``rightArm``, and bare
       cases like ``LeftEye`` / ``RightEye`` with no delimiter. Namespace
       prefix is stripped before matching.
    3. Prefix ``L*`` / ``R*`` followed by an uppercase letter — handles
       ``LArm`` / ``RArm``.

    Mutual-match requirement: both halves must exist as joint names.
    Singletons are not paired.

    Returns ``{}`` if no pairs detected. See ``bvh.lr_mapping`` for the
    public API that wraps this function.
    """
    joint_names = set(bvh.joint_names)
    mapping: dict[str, str] = {}
    seen: set[str] = set()

    for name in bvh.joint_names:
        if name in seen:
            continue

        partner: str | None = None

        # Decompose name for suffix rule: strip namespace and number suffix
        ns_prefix, after_ns = _strip_namespace_prefix(name)
        base = _strip_number_suffix(after_ns)
        number_suffix = after_ns[len(base):]  # e.g. '.001' or ''

        # Strategy 1: delimited suffix (most specific)
        partner = _try_suffix_partner(base, joint_names, ns_prefix, number_suffix)

        # Strategy 2: "left"/"right" substring (case-insensitive)
        # Work on the post-namespace base (no number suffix) so Mixamo
        # names like "mixamorig:LeftArm" match against "LeftArm".
        if partner is None:
            lower = base.lower()
            if "left" in lower:
                partner_base = (base
                                .replace("Left", "Right")
                                .replace("left", "right")
                                .replace("LEFT", "RIGHT"))
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate
            elif "right" in lower:
                partner_base = (base
                                .replace("Right", "Left")
                                .replace("right", "left")
                                .replace("RIGHT", "LEFT"))
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate

        # Strategy 3: "L" / "R" prefix followed by uppercase
        if partner is None and len(base) >= 2:
            if base[0] == "L" and base[1].isupper():
                partner_base = "R" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate
            elif base[0] == "R" and base[1].isupper():
                partner_base = "L" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate

        if partner is not None and partner not in seen:
            # Normalise: always store the "Left" / "L" version as key
            left, right = _order_lr_pair(name, partner)
            mapping[left] = right
            seen.add(left)
            seen.add(right)

    return mapping


def auto_detect_lr_mapping(bvh: Bvh) -> dict[str, str]:
    """Return the L/R joint-name mapping for this skeleton.

    Thin wrapper around :attr:`Bvh.lr_mapping` that returns an empty
    dict (instead of ``None``) when no pairs are available — back-compat
    shape for code that expected a dict.

    Parameters
    ----------
    bvh : Bvh
        Input BVH with named joints.

    Returns
    -------
    dict
        ``{"LeftArm": "RightArm", ...}``.  Empty if no pairs available.
    """
    return bvh.lr_mapping if bvh.lr_mapping is not None else {}


def _order_lr_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(left_name, right_name)``."""
    # Prefer substring form first (most common)
    for kw in ("Left", "left", "LEFT"):
        if kw in a:
            return (a, b)
        if kw in b:
            return (b, a)
    # Normalize for suffix check: strip namespace + number suffix
    _, a_after = _strip_namespace_prefix(a)
    _, b_after = _strip_namespace_prefix(b)
    a_base = _strip_number_suffix(a_after)
    b_base = _strip_number_suffix(b_after)
    for left_suf, _right_suf in _SUFFIX_LR_TABLE:
        if a_base.endswith(left_suf):
            return (a, b)
        if b_base.endswith(left_suf):
            return (b, a)
    # Fallback: "L" prefix → left
    a_prefix_check = _strip_number_suffix(a_after)
    b_prefix_check = _strip_number_suffix(b_after)
    if a_prefix_check.startswith("L") and (len(a_prefix_check) < 2 or a_prefix_check[1].isupper()):
        return (a, b)
    if b_prefix_check.startswith("L") and (len(b_prefix_check) < 2 or b_prefix_check[1].isupper()):
        return (b, a)
    return (a, b)


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
    from .tools import _iter_unique_lr_pairs
    mapping = auto_detect_lr_mapping(bvh)
    j_name2idx = {name: i for i, name in enumerate(bvh.joint_names)}
    pairs: list[tuple[int, int]] = []
    for left_name, right_name in _iter_unique_lr_pairs(mapping):
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

    # --- Detect L/R pairs (first, so we can fail fast with a clear error) ---
    if left_right_mapping is None:
        left_right_mapping = target.lr_mapping

    if not left_right_mapping:
        raise ValueError(
            "No L/R joint pairs available for mirror(). The extended "
            "name heuristic did not recognize this skeleton's convention. "
            "Provide a mapping explicitly — either set `bvh.lr_mapping = "
            "{...}` after loading, or pass `lr_mapping=` at load time "
            "(`read_bvh_file(..., lr_mapping=...)`). If your skeleton is "
            "not bilaterally symmetric, mirroring does not apply.")

    # --- Detect lateral axis ---
    # Mirror is a topology operation (swap L/R joints, negate the lateral
    # component), so we use the rest-pose leftward axis derived from the
    # L/R mapping above. Pass the explicit mapping so we use the same
    # pairs. Only the axis letter is consumed; the sign doesn't matter
    # for mirroring (reflecting across the plane perpendicular to the
    # axis is the same operation whether the axis points left or right).
    if lateral_axis is None:
        from .tools import _rest_leftward
        rest_left = _rest_leftward(target, mapping=left_right_mapping)
        if rest_left is None:
            raise ValueError(
                "Cannot infer lateral axis from L/R mapping: averaged "
                "left-to-right offsets are degenerate (parallel to up "
                "axis or zero). Pass `lateral_axis=` explicitly.")
        lateral_char = rest_left[1]
    else:
        lateral_char = lateral_axis.lower().lstrip("+-")
    lateral_idx = {"x": 0, "y": 1, "z": 2}[lateral_char]

    from .tools import _iter_unique_lr_pairs

    # Build joint-index pairs (indices into joint_angles axis 1)
    joints = [n for n in target.nodes if isinstance(n, BvhJoint)]
    j_name2idx = {j.name: i for i, j in enumerate(joints)}
    lr_j_pairs: list[tuple[int, int]] = []
    for left_name, right_name in _iter_unique_lr_pairs(left_right_mapping):
        if left_name in j_name2idx and right_name in j_name2idx:
            lr_j_pairs.append((j_name2idx[left_name], j_name2idx[right_name]))

    # Build node-index pairs for offset swapping (includes end sites)
    node_name2idx = target.node_index
    lr_node_pairs: list[tuple[int, int]] = []
    for left_name, right_name in _iter_unique_lr_pairs(left_right_mapping):
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
    from .tools import _validate_axis_string, _axis_aligned_rotation

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
    """
    from .tools import _validate_axis_string, _rest_upward, _axis_aligned_rotation

    current_rest_up = _rest_upward(bvh)
    new_up = _validate_axis_string(new_up)

    target = bvh if inplace else bvh.copy()

    if current_rest_up == new_up:
        return None if inplace else target

    R_fix = _axis_aligned_rotation(current_rest_up, new_up)
    R_fix_inv = R_fix.T  # orthogonal → inverse = transpose

    # 1. Rotate all offsets
    for node in target.nodes:
        node.offset = R_fix @ node.offset

    # 2. Compensate joint rotations so FK positions are unchanged.
    #    Root:     R_root' = R_root @ R_fix_inv
    #    Non-root: R_j'    = R_fix @ R_j @ R_fix_inv  (similarity transform)
    joints = [n for n in target.nodes if not n.is_end_site()]
    angles_copy = target.joint_angles.copy()

    # Root (index 0): right-multiply by R_fix_inv only
    root_order = "".join(joints[0].rot_channels)  # type: ignore[attr-defined]
    R_root = rotations.euler_to_rotmat(angles_copy[:, 0], root_order)
    angles_copy[:, 0] = rotations.rotmat_to_euler(
        R_root @ R_fix_inv, root_order)

    # All other joints: full similarity
    _apply_similarity_to_joints(
        angles_copy, joints, R_fix, R_fix_inv, joint_indices=range(1, len(joints)))

    target.joint_angles = angles_copy

    # root_pos unchanged (world frame unchanged)
    # world_up unchanged (world frame unchanged)
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
    from .tools import (_validate_axis_string, _axis_aligned_rotation,
                        _compute_forward_at, _AXIS_CHAR_TO_IDX)

    new_forward = _validate_axis_string(new_forward)
    up_idx = _AXIS_CHAR_TO_IDX[bvh.world_up[1]]
    fwd_idx = _AXIS_CHAR_TO_IDX[new_forward[1]]
    if up_idx == fwd_idx:
        raise ValueError(
            f"new_forward ({new_forward}) cannot be parallel to "
            f"world_up ({bvh.world_up})")

    # Determine current rest-pose forward from topology
    rest_coords = bvh.rest_pose_coords()
    current_fwd = _compute_forward_at(bvh, rest_coords, bvh.world_up)

    target = bvh if inplace else bvh.copy()

    if current_fwd == new_forward:
        return None if inplace else target

    R_fix = _axis_aligned_rotation(current_fwd, new_forward)
    R_fix_inv = R_fix.T

    # Same compensation pattern as reorient_rest_up:
    # 1. Rotate all offsets
    for node in target.nodes:
        node.offset = R_fix @ node.offset

    # 2. Compensate all joints
    joints = [n for n in target.nodes if not n.is_end_site()]
    angles_copy = target.joint_angles.copy()

    root_order = "".join(joints[0].rot_channels)  # type: ignore[attr-defined]
    R_root = rotations.euler_to_rotmat(angles_copy[:, 0], root_order)
    angles_copy[:, 0] = rotations.rotmat_to_euler(
        R_root @ R_fix_inv, root_order)

    _apply_similarity_to_joints(
        angles_copy, joints, R_fix, R_fix_inv, joint_indices=range(1, len(joints)))

    target.joint_angles = angles_copy

    return None if inplace else target
