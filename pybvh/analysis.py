"""Motion analysis for BVH data.

Velocities, accelerations, angular velocities, root trajectory, and foot
contacts. Every function takes a :class:`~pybvh.bvh.Bvh` object as its
first argument; thin wrapper methods on the ``Bvh`` class delegate here.

Feature-array packing for ML pipelines lives in :mod:`pybvh.packing`.
"""
from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Callable

import numpy as np
import numpy.typing as npt

from .bvh import Bvh
from . import rotations
from . import geometry

_EPS = 1e-12


# ----------------------------------------------------------------
#  Joint velocities & accelerations
# ----------------------------------------------------------------

def _validate_stencil_pad(stencil: str, pad: str) -> None:
    if stencil not in ("central", "forward"):
        raise ValueError(
            f"stencil must be 'central' or 'forward', got {stencil!r}")
    if pad not in ("edge", "none"):
        raise ValueError(f"pad must be 'edge' or 'none', got {pad!r}")


def _non_end_site_indices(bvh: Bvh) -> list[int]:
    """Indices in ``nodes`` order that correspond to non-end-site joints.

    The same indices select the joint-axis subset of any per-node array
    (e.g. ``node_positions()`` output of shape ``(F, N, 3)``) to produce
    a joint-aligned ``(F, J, 3)``.
    """
    return [i for i, n in enumerate(bvh.nodes) if not n.is_end_site()]


def node_velocities(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-node position velocities (joints + end sites).

    Two orthogonal choices:

    * ``stencil`` picks the finite-difference method — central
      (second-order accurate, symmetric) or forward (first-order,
      causal).
    * ``pad`` picks the boundary-handling convention — ``"edge"``
      fills the boundary so the output has the same shape as the
      input; ``"none"`` drops the boundary frames that the stencil
      can't define.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        **Ignored if `coords` is provided** — `coords` takes precedence.
        Note: ``"world"`` and ``"first"`` produce identical velocities
        (constant offsets vanish under differentiation); only
        ``"skeleton"`` is meaningfully different here.
    in_frames : bool, optional
        If True, return velocity in units/frame.
        If False (default), return velocity in units/second.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates. If None, computed
        internally via :meth:`Bvh.node_positions`.
    stencil : {"central", "forward"}, optional
        ``"central"`` (default): ``v[i] = (pos[i+1] - pos[i-1]) / (2·dt)``.
        Second-order accurate at interior frames.  ``"forward"``:
        ``v[i] = (pos[i+1] - pos[i]) / dt``.  First-order accurate;
        matches the convention common in many ML papers.
    pad : {"edge", "none"}, optional
        ``"edge"`` (default): output shape equals input shape
        ``(F, N, 3)``.  For ``stencil="central"`` the first/last
        frames use a one-sided difference (``np.gradient`` template);
        for ``stencil="forward"`` the trailing frame replicates the
        last valid forward-diff value.
        ``"none"``: drop boundary frames where the stencil is
        undefined — shape ``(F-2, N, 3)`` for central,
        ``(F-1, N, 3)`` for forward.

    Returns
    -------
    ndarray
        Shape depends on ``stencil`` × ``pad``:

        =========  ======  ================
        stencil    pad     shape
        =========  ======  ================
        central    edge    ``(F, N, 3)``
        central    none    ``(F-2, N, 3)``
        forward    edge    ``(F, N, 3)``
        forward    none    ``(F-1, N, 3)``
        =========  ======  ================

    See Also
    --------
    joint_velocities : Same data restricted to non-end-site joints
        (``(F, J, 3)``). Use that when the output should index-align
        with :attr:`Bvh.joint_angles` / :func:`angular_velocities`.

    Raises
    ------
    ValueError
        If the clip is too short for the chosen combination,
        ``frame_time == 0`` when ``in_frames=False``, or either
        parameter is invalid.  ``stencil="central"`` requires at
        least 3 frames; ``stencil="forward"`` requires at least 2.
    """
    _validate_stencil_pad(stencil, pad)
    min_frames = 3 if stencil == "central" else 2
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r} requires at least {min_frames} frames "
            f"to compute velocities (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second velocity. "
            "Use in_frames=True for per-frame velocity.")

    if coords is None:
        coords = bvh.node_positions(centered=centered)

    dt = 1.0 if in_frames else bvh.frame_time

    if stencil == "central":
        central = np.gradient(coords, dt, axis=0)  # (F, N, 3)
        return central if pad == "edge" else central[1:-1]  # (F-2, N, 3)

    # stencil == "forward"
    fd = (coords[1:] - coords[:-1]) / dt  # (F-1, N, 3)
    if pad == "edge":
        # Replicate the last forward value (equivalent to backward
        # diff at frame F-1 under the same stencil assumption).
        return np.concatenate([fd, fd[-1:]], axis=0)  # (F, N, 3)
    return fd  # (F-1, N, 3)


def joint_velocities(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-joint position velocities (end sites excluded).

    Returns the joint-axis subset of :func:`node_velocities` — same
    finite-difference math, but restricted to non-end-site joints so
    the output indexes match :attr:`Bvh.joint_angles` and
    :func:`angular_velocities`. Output shape is ``(F, J, 3)`` (or the
    appropriate trimmed variant per ``stencil`` × ``pad``).

    See :func:`node_velocities` for the full parameter / shape docs.
    """
    nv = node_velocities(
        bvh, centered=centered, in_frames=in_frames, coords=coords,
        stencil=stencil, pad=pad)
    return nv[:, _non_end_site_indices(bvh), :]


def node_accelerations(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-node position accelerations (joints + end sites).

    Applies the chosen ``stencil`` twice to the input positions.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        **Ignored if `coords` is provided** — `coords` takes precedence.
        Note: ``"world"`` and ``"first"`` produce identical accelerations
        (constant offsets vanish under differentiation); only
        ``"skeleton"`` is meaningfully different here.
    in_frames : bool, optional
        If True, return acceleration in units/frame^2.
        If False (default), return in units/second^2.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates. If None, computed
        internally via :meth:`Bvh.node_positions`.
    stencil : {"central", "forward"}, optional
        Finite-difference method applied twice.  Default ``"central"``.
    pad : {"edge", "none"}, optional
        Boundary handling.  ``"edge"`` (default): output shape equals
        input shape ``(F, N, 3)``.  ``"none"``: drop boundary frames
        the stencil can't define — central drops 4 frames total
        ``(F-4, N, 3)``; forward drops 2 ``(F-2, N, 3)``.

    Returns
    -------
    ndarray
        Shape depends on ``stencil`` × ``pad``:

        =========  ======  ================
        stencil    pad     shape
        =========  ======  ================
        central    edge    ``(F, N, 3)``
        central    none    ``(F-4, N, 3)``
        forward    edge    ``(F, N, 3)``
        forward    none    ``(F-2, N, 3)``
        =========  ======  ================

    Composition identity: ``np.gradient(node_velocities(), dt)`` equals
    ``node_accelerations()`` exactly under the defaults
    (``stencil="central"``, ``pad="edge"``).  Not guaranteed for other
    combinations.

    See Also
    --------
    joint_accelerations : Same data restricted to non-end-site joints
        (``(F, J, 3)``).

    Raises
    ------
    ValueError
        If the clip is too short for the chosen combination,
        ``frame_time == 0`` when ``in_frames=False``, or either
        parameter is invalid.  Minimum frames: 3 for
        ``central``+``edge``, ``forward``+``edge``, and
        ``forward``+``none``; 5 for ``central``+``none``.
    """
    _validate_stencil_pad(stencil, pad)
    min_frames = 5 if (stencil == "central" and pad == "none") else 3
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r}, pad={pad!r} requires at least "
            f"{min_frames} frames (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second acceleration. "
            "Use in_frames=True for per-frame acceleration.")

    if coords is None:
        coords = bvh.node_positions(centered=centered)

    dt = 1.0 if in_frames else bvh.frame_time

    if stencil == "central":
        # np.gradient twice — preserves the composition identity with
        # node_velocities(stencil="central", pad="edge").
        central = np.gradient(
            np.gradient(coords, dt, axis=0), dt, axis=0)  # (F, N, 3)
        return central if pad == "edge" else central[2:-2]  # (F-4, N, 3)

    # stencil == "forward": forward-forward
    vel = (coords[1:] - coords[:-1]) / dt      # (F-1, N, 3)
    acc = (vel[1:] - vel[:-1]) / dt            # (F-2, N, 3)
    if pad == "edge":
        # Replicate last value twice to reach F
        return np.concatenate([acc, acc[-1:], acc[-1:]], axis=0)
    return acc


def joint_accelerations(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-joint position accelerations (end sites excluded).

    Returns the joint-axis subset of :func:`node_accelerations` — same
    twice-applied finite-difference math, restricted to non-end-site
    joints so output indexes match :attr:`Bvh.joint_angles`. Output
    shape is ``(F, J, 3)`` (or the appropriate trimmed variant per
    ``stencil`` × ``pad``).

    See :func:`node_accelerations` for the full parameter / shape docs.
    """
    na = node_accelerations(
        bvh, centered=centered, in_frames=in_frames, coords=coords,
        stencil=stencil, pad=pad)
    return na[:, _non_end_site_indices(bvh), :]


# ----------------------------------------------------------------
#  Angular velocities
# ----------------------------------------------------------------

def angular_velocities(
    bvh: Bvh,
    in_frames: bool = False,
    stencil: str = "central",
    pad: str = "edge",
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute per-joint angular velocities via rotation matrix log map.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    in_frames : bool, optional
        If True, return angular velocity in radians/frame (or
        degrees/frame if ``degrees=True``).
        If False (default), return in radians/second (or degrees/
        second if ``degrees=True``).
    degrees : bool, optional
        If True, convert the final output from radians to degrees.
        Default False (radians).  Consistent with the ``degrees=``
        flag on :mod:`pybvh.rotations` functions.
    stencil : {"central", "forward"}, optional
        ``"central"`` (default): two-step relative rotation
        ``R_rel = R_{i-1}^T @ R_{i+1}``, ``ω[i] = log(R_rel) / 2``.
        Spans ``2·dt`` so the short-way angle cap is 360°/frame.
        ``"forward"``: one-step ``ω[i] = log(R_i^T @ R_{i+1})``.
        Spans ``dt`` so the cap is 180°/frame; matches the common
        one-step-rotation convention in motion capture literature.
    pad : {"edge", "none"}, optional
        ``"edge"`` (default): output shape ``(F, J, 3)``.  For
        ``stencil="central"`` the first/last frames use a one-sided
        one-step forward/backward rotation (same template as
        ``np.gradient``); for ``stencil="forward"`` the trailing
        frame replicates the last valid forward value.
        ``"none"``: drop boundary frames the stencil can't define —
        central returns ``(F-2, J, 3)``, forward returns
        ``(F-1, J, 3)``.

    Returns
    -------
    ndarray
        Shape depends on ``stencil`` × ``pad``:

        =========  ======  ================
        stencil    pad     shape
        =========  ======  ================
        central    edge    ``(F, J, 3)``
        central    none    ``(F-2, J, 3)``
        forward    edge    ``(F, J, 3)``
        forward    none    ``(F-1, J, 3)``
        =========  ======  ================

        Direction is the rotation axis; magnitude is the rotation
        angle (radians or radians/second).  Angles are clamped to
        ``[0, π]`` — rotations exceeding the short-way angle wrap.

    Raises
    ------
    ValueError
        If fewer than 2 frames (``stencil="forward"``) or 3 frames
        (``stencil="central"``), ``frame_time == 0`` when
        ``in_frames=False``, or either parameter is invalid.
    """
    _validate_stencil_pad(stencil, pad)
    min_frames = 3 if stencil == "central" else 2
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r} requires at least {min_frames} frames "
            f"(have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second angular velocity. "
            "Use in_frames=True for per-frame angular velocity.")

    _, R = bvh.to_rotmat()  # (F, J, 3, 3)
    F = R.shape[0]

    if stencil == "forward":
        # ω[i] = log(R_i^T @ R_{i+1})
        R_rel = np.einsum('...ji,...jk->...ik', R[:-1], R[1:])  # (F-1, J, 3, 3)
        ang_vel = rotations.rotmat_to_axisangle(R_rel)          # radians/frame
        if pad == "edge":
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]], axis=0)  # (F, J, 3)
        if not in_frames:
            ang_vel = ang_vel / bvh.frame_time
        if degrees:
            ang_vel = np.degrees(ang_vel)
        return ang_vel

    # stencil == "central": two-step ω[i] = log(R_{i-1}^T R_{i+1}) / 2
    R_rel_central = np.einsum('...ji,...jk->...ik', R[:-2], R[2:])  # (F-2, J, 3, 3)
    omega_central = rotations.rotmat_to_axisangle(R_rel_central) / 2.0  # rad/frame

    if pad == "none":
        if not in_frames:
            omega_central = omega_central / bvh.frame_time
        if degrees:
            omega_central = np.degrees(omega_central)
        return omega_central  # (F-2, J, 3)

    # pad == "edge": one-sided forward/backward at boundaries
    omega = np.empty((F,) + R.shape[1:-2] + (3,), dtype=np.float64)
    omega[1:-1] = omega_central
    R_rel_first = np.einsum('...ji,...jk->...ik', R[0:1], R[1:2])
    omega[0:1] = rotations.rotmat_to_axisangle(R_rel_first)
    R_rel_last = np.einsum('...ji,...jk->...ik', R[-2:-1], R[-1:])
    omega[-1:] = rotations.rotmat_to_axisangle(R_rel_last)
    if not in_frames:
        omega = omega / bvh.frame_time
    if degrees:
        omega = np.degrees(omega)
    return omega  # (F, J, 3)


# ----------------------------------------------------------------
#  Root trajectory
# ----------------------------------------------------------------

def root_trajectory(
    bvh: Bvh,
    up_axis: str | None = None,
    include_velocities: bool = False,
    stencil: str = "central",
    pad: str = "edge",
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Extract root trajectory features commonly used in motion ML.

    Returns the root's ground-plane position and heading angle (as
    ``sin``/``cos`` pair).  Optionally appends ground-plane and
    heading velocities.

    The heading reference is the **rest-pose forward direction** —
    derived from the skeleton's L/R lateral geometry crossed with
    ``world_up`` (see :func:`pybvh.tools._compute_forward_at`).  This
    means "heading = rest-pose forward" at any frame whose root
    rotation is identity, regardless of what rotation the clip
    starts with.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    up_axis : str or None, optional
        Signed axis string (e.g. ``'+y'``, ``'+z'``). If None,
        uses ``bvh.world_up``.
    include_velocities : bool, optional
        If True, append ``[ground_a_vel, ground_b_vel, heading_vel]``
        to the output. Velocities are in coordinate-units/second and
        radians/second (heading is unwrapped before differentiating
        to avoid ±π jumps).
    stencil, pad : optional
        Only used with ``include_velocities=True``.  Same semantics
        as :func:`joint_velocities` — see that docstring for the full
        matrix.  Default ``stencil="central", pad="edge"`` returns
        shape ``(F, 7)``; ``stencil="forward", pad="none"`` returns
        ``(F-1, 7)``; ``stencil="central", pad="none"`` returns
        ``(F-2, 7)``.
    degrees : bool, optional
        If True, convert the ``heading_vel`` column from
        radians/second to degrees/second.  Default False (radians).
        ``ground_*_vel`` columns are linear positions per second and
        are unaffected.  Only used when ``include_velocities=True``.

    Returns
    -------
    ndarray
        Shape ``(F, 4)`` when ``include_velocities=False``.  When
        ``include_velocities=True`` the trailing 3 columns are
        ``[ground_a_vel, ground_b_vel, heading_vel]`` and the leading
        4-column base is trimmed to match the chosen ``stencil`` ×
        ``pad`` shape.

        Columns: ``[ground_pos_a, ground_pos_b, heading_sin, heading_cos]``,
        optionally followed by ``[ground_a_vel, ground_b_vel, heading_vel]``.
        ``a`` and ``b`` are the two ground-plane axes (non-up axes in
        the natural ``x, y, z`` order with the up axis removed).
    """
    from .tools import _compute_forward_at, _axis_to_vector

    # Resolve up axis (honors bvh.world_up by default)
    up_str = bvh.world_up if up_axis is None else up_axis
    up_idx = {'x': 0, 'y': 1, 'z': 2}[up_str[1]]
    ground_axes = [i for i in range(3) if i != up_idx]

    # Rest-pose forward direction — independent of animation start
    rest_coords = bvh.rest_pose_coords()
    fwd_axis = _compute_forward_at(bvh, rest_coords, up_str)
    fwd_rest = _axis_to_vector(fwd_axis)  # (3,) unit vector

    # Root rotation over time
    root_joint = bvh.nodes[0]
    root_angles = bvh.joint_angles[:, 0]
    root_order = root_joint.rot_channels  # type: ignore[attr-defined]
    R_root = rotations.euler_to_rotmat(root_angles, root_order)

    # World-space forward at each frame = R_root @ rest_forward
    fwd_world = np.einsum('fij,j->fi', R_root, fwd_rest)  # (F, 3)

    ground_pos = bvh.root_pos[:, ground_axes]  # (F, 2)
    heading = np.arctan2(fwd_world[:, ground_axes[1]],
                         fwd_world[:, ground_axes[0]])  # (F,)

    base = np.column_stack([
        ground_pos,
        np.sin(heading),
        np.cos(heading),
    ])  # (F, 4)

    if not include_velocities:
        return base

    _validate_stencil_pad(stencil, pad)
    if bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute root_trajectory velocities. "
            "Set bvh.frame_time to a non-zero value first."
        )
    min_frames = 3 if stencil == "central" else 2
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r} requires at least {min_frames} frames "
            f"(have {bvh.frame_count})."
        )

    dt = bvh.frame_time
    heading_unwrapped = np.unwrap(heading)

    if stencil == "central":
        if pad == "edge":
            ground_vel = np.gradient(ground_pos, dt, axis=0)         # (F, 2)
            heading_vel = np.gradient(heading_unwrapped, dt)         # (F,)
            base_aligned = base                                       # (F, 4)
        else:  # pad == "none": strict central, drop first and last
            ground_vel = (ground_pos[2:] - ground_pos[:-2]) / (2.0 * dt)          # (F-2, 2)
            heading_vel = (heading_unwrapped[2:] - heading_unwrapped[:-2]) / (2.0 * dt)  # (F-2,)
            base_aligned = base[1:-1]                                              # (F-2, 4)
    else:  # stencil == "forward"
        ground_fd = (ground_pos[1:] - ground_pos[:-1]) / dt                        # (F-1, 2)
        heading_fd = (heading_unwrapped[1:] - heading_unwrapped[:-1]) / dt         # (F-1,)
        if pad == "edge":
            ground_vel = np.concatenate([ground_fd, ground_fd[-1:]], axis=0)       # (F, 2)
            heading_vel = np.concatenate([heading_fd, heading_fd[-1:]])            # (F,)
            base_aligned = base                                                    # (F, 4)
        else:  # pad == "none": existing "drop first frame" convention for forward
            ground_vel = ground_fd                                                 # (F-1, 2)
            heading_vel = heading_fd                                               # (F-1,)
            base_aligned = base[1:]                                                # (F-1, 4)

    if degrees:
        heading_vel = np.degrees(heading_vel)
    return np.column_stack([base_aligned, ground_vel, heading_vel])


# ----------------------------------------------------------------
#  Foot contacts
# ----------------------------------------------------------------

def foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    method: str = "combined",
    centered: str = "world",
    coords: npt.NDArray[np.float64] | None = None,
    *,
    vel_threshold: float | None = None,
    height_threshold: float | None = None,
    floor: float | str = "auto",
    min_contact_duration: float = 0.1,
    min_gap_duration: float = 0.1,
    return_info: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], dict]:
    """Detect binary foot contact labels per frame.

    The default combines a velocity check (foot not moving) **and** a
    height check (foot near the ground) following the HuMoR heuristic
    — each signal catches a different failure mode of the other.
    ``method="velocity"`` and ``method="height"`` remain available as
    single-signal escape hatches.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str or None, optional
        Explicit foot joints (recommended). If None, falls back to
        :func:`auto_detect_foot_joints` which matches ``"foot"``/
        ``"toe"`` substrings then filters by skeletal topology.
    method : {"combined", "velocity", "height"}, optional
        ``"combined"`` (default): foot is in contact when speed is
        below ``vel_threshold`` **and** height above floor is below
        ``height_threshold``.  ``"velocity"`` / ``"height"``: single
        signal.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        **Ignored if `coords` is provided** — `coords` takes precedence.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates.
    vel_threshold : float or None, keyword-only, optional
        Speed threshold in world units per frame.  Defaults to
        ``0.004 × skeleton_scale`` where ``skeleton_scale`` is the
        mean rest-pose distance from the root to the foot joints.
        Scale-invariant across cm- and m-scale skeletons, and
        unaffected by finger/spine subdivision (unlike a
        median-bone-length reference, which shrinks when a skeleton
        has many short finger bones).
    height_threshold : float or None, keyword-only, optional
        Clearance above the estimated floor, in world units.  Defaults
        to ``0.013 × skeleton_scale``.  A foot is "low enough" when
        ``foot_height − floor < height_threshold``.
    floor : float or ``"auto"``, keyword-only, optional
        Floor height along the raw ``world_up`` axis.  ``"auto"``
        (default) estimates it as the 2nd percentile of the per-frame
        minimum foot height.  Pass a float to pin the floor explicitly
        (e.g. ``floor=0.0`` when the rig is already ground-aligned).
    min_contact_duration : float, keyword-only, optional
        Morphological open: contact runs shorter than this many
        **seconds** are set to 0.  Default ``0.1`` s (3 frames at
        30 fps) — removes contact flickers shorter than 100 ms, which
        are physically implausible.  Set to ``0.0`` to disable.
        Internally converted to frames via
        ``max(1, round(duration / frame_time))``.
    min_gap_duration : float, keyword-only, optional
        Morphological close: non-contact gaps shorter than this many
        **seconds** are filled (set to 1).  Default ``0.1`` s —
        bridges short interruptions in an otherwise continuous contact
        phase, catching pivot-foot artefacts where the joint briefly
        exceeds the velocity threshold even though the physical foot
        is planted.  Set to ``0.0`` to disable.
    return_info : bool, keyword-only, optional
        If True, return ``(contacts, info)`` where ``info`` is a dict
        holding the detected joints, the thresholds actually applied,
        the estimated floor, the skeleton scale used for auto-
        calibration, and the method used.  Useful for debugging and
        for downstream pipelines that need to record the detection
        parameters.  ``"skeleton_scale"`` is only present when
        auto-calibration ran (i.e., at least one threshold was left
        at its default).

    Returns
    -------
    ndarray of shape ``(F, num_foot_joints)``, or
    tuple ``(ndarray, dict)`` when ``return_info=True``.
        Binary contact labels (1.0 = contact, 0.0 = no contact).
        For ``"velocity"``/``"combined"``, frame 0 is propagated from
        frame 1 because velocity is undefined at frame 0.  Column
        order matches ``foot_joints`` (or, for auto-detection, the
        order returned by :func:`auto_detect_foot_joints`).

    Raises
    ------
    ValueError
        - If ``method`` is unknown.
        - If ``floor`` is a string other than ``"auto"``.
        - If no foot joints can be found or any named joint is
          missing from the skeleton.
        - When the height signal is involved and ``bvh.world_up`` is
          inconsistent with rest-pose geometry (feet above hips).
    """
    if method not in ("velocity", "height", "combined"):
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Choose 'combined', 'velocity', or 'height'.")

    if isinstance(floor, str) and floor != "auto":
        raise ValueError(
            f"floor must be 'auto' or a float, got {floor!r}")

    if coords is None:
        coords = bvh.node_positions(centered=centered)

    up_sign = 1 if bvh.world_up[0] == '+' else -1
    up_idx = {'x': 0, 'y': 1, 'z': 2}[bvh.world_up[1]]

    # Rest-pose coords are used by auto-detect and by the height-signal
    # sanity check; compute once, reuse.
    rest_coords: npt.NDArray[np.float64] | None = None

    if foot_joints is None:
        rest_coords = bvh.rest_pose_coords()
        foot_joints = auto_detect_foot_joints(bvh, _rest_coords=rest_coords)
        if not foot_joints:
            raise ValueError(
                "Could not auto-detect foot joints. Please provide "
                "foot_joints explicitly (e.g. ['LeftFoot', 'RightFoot']).")

    foot_indices: list[int] = []
    for name in foot_joints:
        if name not in bvh.node_index:
            raise ValueError(f"Joint {name!r} not found in skeleton.")
        foot_indices.append(bvh.node_index[name])

    foot_coords = coords[:, foot_indices, :]  # (F, num_feet, 3)
    num_feet = len(foot_joints)
    F = bvh.frame_count

    needs_vel = method in ("velocity", "combined")
    needs_height = method in ("height", "combined")

    # Rest-pose coords drive both the skeleton-scale estimate and the
    # height-signal sanity check.  Compute once, reuse.
    needs_scale = (
        (needs_vel and vel_threshold is None)
        or (needs_height and height_threshold is None)
    )
    if (needs_height or needs_scale) and rest_coords is None:
        rest_coords = bvh.rest_pose_coords()

    scale: float | None = None
    if needs_scale:
        assert rest_coords is not None
        scale = _skeleton_scale(rest_coords, foot_indices)

    # ---- Sanity check for the height signal ----
    if needs_height:
        assert rest_coords is not None
        rest_foot_height = (
            rest_coords[foot_indices, up_idx].mean() * up_sign
        )
        rest_hip_height = rest_coords[0, up_idx] * up_sign
        if rest_foot_height > rest_hip_height:
            raise ValueError(
                f"world_up={bvh.world_up!r} is inconsistent with rest-pose "
                f"geometry: feet are above hips along the declared up axis "
                f"({rest_foot_height:.3f} vs {rest_hip_height:.3f}). "
                f"`bvh.world_up` was likely auto-detected incorrectly; set "
                f"it manually with `bvh.world_up = '<axis>'`."
            )

    # ---- Velocity signal ----
    vel_mask: npt.NDArray[np.bool_] | None = None
    vel_thr_used: float | None = None
    if needs_vel:
        if vel_threshold is None:
            assert scale is not None
            # 0.4% of the root-to-foot rest distance per frame.
            # Reverse-engineered from HuMoR-equivalent + pivot-foot tolerance
            # on real BVH clips.
            vel_threshold = 0.004 * scale
        vel_thr_used = float(vel_threshold)
        if F < 2:
            # No motion info available — treat as "no velocity evidence
            # against contact" so combined falls back to the height signal.
            vel_mask = np.ones((F, num_feet), dtype=bool)
        else:
            foot_vel = foot_coords[1:] - foot_coords[:-1]  # (F-1, nf, 3)
            speed = np.linalg.norm(foot_vel, axis=-1)       # (F-1, nf)
            inner = speed < vel_threshold
            # Propagate frame 0 from frame 1 (velocity undefined at frame 0)
            vel_mask = np.concatenate([inner[0:1], inner], axis=0)

    # ---- Height signal ----
    height_mask: npt.NDArray[np.bool_] | None = None
    height_thr_used: float | None = None
    floor_raw: float | None = None
    if needs_height:
        heights_signed = foot_coords[:, :, up_idx] * up_sign  # up-positive
        if isinstance(floor, str):
            floor_signed = _estimate_floor(heights_signed)
        else:
            floor_signed = float(floor) * up_sign
        floor_raw = float(floor_signed * up_sign)
        if height_threshold is None:
            assert scale is not None
            # ~1.3% of root-to-foot rest distance above floor.
            height_threshold = 0.013 * scale
        height_thr_used = float(height_threshold)
        height_mask = (heights_signed - floor_signed) < height_threshold

    # ---- Combine ----
    if method == "velocity":
        assert vel_mask is not None
        mask = vel_mask
    elif method == "height":
        assert height_mask is not None
        mask = height_mask
    else:  # combined
        assert vel_mask is not None and height_mask is not None
        mask = vel_mask & height_mask

    # ---- Morphological duration filters (time → frames) ----
    dt = bvh.frame_time if bvh.frame_time > 0 else 1.0
    min_contact_frames = max(1, round(min_contact_duration / dt)) if min_contact_duration > 0 else 1
    min_gap_frames = max(1, round(min_gap_duration / dt)) if min_gap_duration > 0 else 1

    if min_contact_frames > 1:
        mask = _filter_short_runs(mask, min_contact_frames, value=True)
    if min_gap_frames > 1:
        mask = _filter_short_runs(mask, min_gap_frames, value=False)

    contacts = mask.astype(np.float64)

    if not return_info:
        return contacts

    info: dict = {
        "joints": list(foot_joints),
        "method": method,
        "min_contact_duration": float(min_contact_duration),
        "min_gap_duration": float(min_gap_duration),
    }
    if scale is not None:
        info["skeleton_scale"] = float(scale)
    if vel_thr_used is not None:
        info["vel_threshold"] = vel_thr_used
    if height_thr_used is not None:
        info["height_threshold"] = height_thr_used
        info["floor"] = floor_raw
    return contacts, info


def auto_detect_foot_joints(
    bvh: Bvh,
    _rest_coords: npt.NDArray[np.float64] | None = None,
) -> list[str]:
    """Auto-detect foot joint names by topology.

    Algorithm:

    1. Substring match: candidates are joints whose names contain
       ``"foot"`` or ``"toe"`` (case-insensitive).
    2. Tip-descendant filter: keep only candidates that have an end
       site or a toe-named child.  This drops IK helpers, which
       typically have no children.
    3. Most-distal filter: drop candidates whose subtree (any depth)
       contains another candidate.  On a rig with
       ``Foot → ToeBase → EndSite``, this keeps only ``ToeBase`` —
       the more distal, ground-contacting joint.
    4. Deterministic order: sort by rest-pose height along
       ``bvh.world_up`` (lowest first); alphabetical name within
       ties so the output is stable across runs.

    If step 1 produces matches but step 2 drops all of them, the
    tip filter is skipped with a ``UserWarning`` (better than
    returning nothing for unusual rigs).

    Parameters
    ----------
    bvh : Bvh
        Input skeleton.

    Returns
    -------
    list of str
        Joint names in deterministic order.  Empty if no candidates.

    Notes
    -----
    This is the same detection used internally by
    :func:`foot_contacts` when ``foot_joints=None``.  Call it
    directly to preview the detection or to feed an explicit list
    back in.
    """
    up_sign = 1 if bvh.world_up[0] == '+' else -1
    up_idx = {'x': 0, 'y': 1, 'z': 2}[bvh.world_up[1]]

    # Step 1: substring match
    matched = [
        n for n in bvh.nodes
        if not n.is_end_site()
        and any(kw in n.name.lower() for kw in ("foot", "toe"))
    ]
    if not matched:
        return []

    # Step 2: has-tip filter
    from .bvhnode import BvhNode

    def _has_tip(node: BvhNode) -> bool:
        return any(
            c.is_end_site() or "toe" in c.name.lower()
            for c in (getattr(node, "children", None) or [])
        )

    with_tip = [n for n in matched if _has_tip(n)]
    if not with_tip:
        warnings.warn(
            "auto_detect_foot_joints: no candidates have tip descendants; "
            "falling back to all substring matches. Pass foot joints "
            "explicitly if auto-detection looks wrong.",
            UserWarning,
            stacklevel=3,
        )
        with_tip = matched

    # Step 3: most-distal — drop candidates whose subtree contains another candidate
    candidate_names = {n.name for n in with_tip}

    def _has_candidate_descendant(node: BvhNode) -> bool:
        for child in (getattr(node, "children", None) or []):
            if child.name in candidate_names:
                return True
            if not child.is_end_site() and _has_candidate_descendant(child):
                return True
        return False

    most_distal = [n for n in with_tip if not _has_candidate_descendant(n)]

    # Step 4: stable sort — height (ascending) then name (alphabetical)
    rest_coords = _rest_coords if _rest_coords is not None else bvh.rest_pose_coords()
    assert isinstance(rest_coords, np.ndarray)  # mypy narrowing

    def _sort_key(node: BvhNode) -> tuple[float, str]:
        idx = bvh.node_index[node.name]
        height = float(rest_coords[idx, up_idx] * up_sign)
        return (height, node.name)

    most_distal.sort(key=_sort_key)
    return [n.name for n in most_distal]


def _skeleton_scale(
    rest_coords: npt.NDArray[np.float64],
    foot_indices: list[int],
) -> float:
    """Scale reference for ``foot_contacts`` thresholds.

    Defined as the mean Euclidean rest-pose distance from the root
    (index 0) to the foot joints.  Independent of rest-pose
    orientation (no axis projection) and unaffected by finger/spine
    subdivision, because only the leg chain contributes.

    For a humanoid this is roughly half of skeleton height; the
    absolute value matters less than the fact that it scales linearly
    with the whole skeleton.
    """
    root = rest_coords[0]
    dists = [float(np.linalg.norm(rest_coords[i] - root)) for i in foot_indices]
    # Fallback: if all feet sit on the root (degenerate rig), use 1.0 so
    # thresholds don't collapse to zero.  Callers should not hit this in
    # practice — auto-detection requires tip descendants.
    return float(np.mean(dists)) if dists and max(dists) > 0 else 1.0


def _estimate_floor(
    heights_signed: npt.NDArray[np.float64],
    percentile: float = 2.0,
) -> float:
    """Estimate the floor height from a window of foot samples.

    Works in the up-positive (sign-corrected) coordinate. Floor is the
    low percentile of the per-frame *minimum* foot height — robust to
    the case where feet never plant simultaneously (one foot always
    up) and to occasional spurious low frames.
    """
    min_per_frame = heights_signed.min(axis=1)
    return float(np.percentile(min_per_frame, percentile))


def _filter_short_runs(
    mask: npt.NDArray[np.bool_],
    min_run: int,
    value: bool = True,
) -> npt.NDArray[np.bool_]:
    """Remove runs of ``value`` shorter than ``min_run`` frames, per column.

    With ``value=True`` (default): zeroes out short True runs
    (morphological open — removes contact jitter).  With
    ``value=False``: fills in short False runs (morphological close —
    bridges short gaps in an otherwise continuous contact phase).

    Returns the input unchanged when ``min_run <= 1``.
    """
    if min_run <= 1:
        return mask
    if mask.ndim != 2:
        raise ValueError("_filter_short_runs expects a 2-D boolean array")

    F = mask.shape[0]
    # Search for runs of True in `m`; flip up front to also cover the
    # value=False case (filling short False gaps in `mask`).
    m = mask if value else ~mask

    # Pad with False top/bottom so a run starting at row 0 or ending at
    # row F-1 still produces ±1 transitions in `diffs`.
    padded = np.zeros((F + 2, mask.shape[1]), dtype=np.int8)
    padded[1:-1] = m
    diffs = np.diff(padded, axis=0)  # (F+1, M); +1 at run starts, -1 just past run ends

    pos_col = np.arange(F + 1)[:, None]
    # Most-recent run start position at or before each row (inclusive).
    start_idx = np.where(diffs == 1, pos_col, -1)
    start_pos = np.maximum.accumulate(start_idx, axis=0)
    # Next run end position strictly after each row.
    end_idx = np.where(diffs == -1, pos_col, F + 2)
    end_pos = np.minimum.accumulate(end_idx[::-1], axis=0)[::-1]

    run_length = end_pos[1:F + 1] - start_pos[:F]
    short_run_mask = m & (run_length < min_run)

    if value:
        return mask & ~short_run_mask
    return mask | short_run_mask


# ----------------------------------------------------------------
#  Jerk (third derivative of position) — [Bvh]
# ----------------------------------------------------------------

def node_jerk(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-node position jerk (third derivative) — ``(F, N, 3)``.

    Applies the chosen ``stencil`` three times to the positions, the
    next rung of the velocity → acceleration → jerk ladder. The jerk
    *magnitude* ``np.linalg.norm(node_jerk(...), axis=-1)`` is the usual
    smoothness signal.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``). Ignored if
        ``coords`` is given.
    in_frames : bool, optional
        If True, units/frame³; else units/second³ (default).
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed positions; computed via :meth:`Bvh.node_positions`
        if None.
    stencil : {"central", "forward"}, optional
        Finite-difference method applied three times. Default
        ``"central"``.
    pad : {"edge", "none"}, optional
        ``"edge"`` (default): output shape ``(F, N, 3)``. ``"none"``:
        drop boundary frames the stencil can't define — central drops 6
        ``(F-6, N, 3)``; forward drops 3 ``(F-3, N, 3)``.

    Returns
    -------
    ndarray
        Per-node jerk. Composition identity:
        ``np.gradient(node_accelerations(), dt)`` equals ``node_jerk()``
        exactly under the defaults (``stencil="central"``, ``pad="edge"``).

    Raises
    ------
    ValueError
        Too-short clip, ``frame_time == 0`` with ``in_frames=False``, or
        invalid parameter. Minimum frames: 7 for ``central``+``none``, 4
        for ``forward``, 3 otherwise.
    """
    _validate_stencil_pad(stencil, pad)
    if stencil == "central" and pad == "none":
        min_frames = 7
    elif stencil == "forward":
        min_frames = 4
    else:
        min_frames = 3
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r}, pad={pad!r} requires at least "
            f"{min_frames} frames (have {bvh.frame_count}).")
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second jerk. "
            "Use in_frames=True for per-frame jerk.")

    if coords is None:
        coords = bvh.node_positions(centered=centered)
    dt = 1.0 if in_frames else bvh.frame_time

    if stencil == "central":
        # np.gradient three times — preserves the composition identity
        # with node_accelerations(stencil="central", pad="edge").
        jerk = np.gradient(np.gradient(
            np.gradient(coords, dt, axis=0), dt, axis=0), dt, axis=0)
        return jerk if pad == "edge" else jerk[3:-3]  # (F-6, N, 3)

    # stencil == "forward": forward applied three times
    vel = (coords[1:] - coords[:-1]) / dt   # (F-1, N, 3)
    acc = (vel[1:] - vel[:-1]) / dt          # (F-2, N, 3)
    jerk = (acc[1:] - acc[:-1]) / dt         # (F-3, N, 3)
    if pad == "edge":
        return np.concatenate([jerk, jerk[-1:], jerk[-1:], jerk[-1:]], axis=0)
    return jerk


def joint_jerk(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Per-joint position jerk (end sites excluded) — ``(F, J, 3)``.

    The joint-axis subset of :func:`node_jerk`, index-aligned with
    :attr:`Bvh.joint_angles`. See :func:`node_jerk` for full docs.
    """
    nj = node_jerk(bvh, centered=centered, in_frames=in_frames, coords=coords,
                   stencil=stencil, pad=pad)
    return nj[:, _non_end_site_indices(bvh), :]


# ----------------------------------------------------------------
#  Smoothness — array-pure kernels on a 1-D speed profile
# ----------------------------------------------------------------

def sparc(
    speed: npt.NDArray[np.float64],
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Spectral arc length (SPARC) smoothness of a speed profile.

    The negative arc length of the normalized Fourier magnitude spectrum
    over ``[0, fc]`` Hz — a smoothness measure that is robust to noise and
    invariant to amplitude/duration. Values are ``≤ 0``; closer to ``0``
    is smoother.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed (magnitude) profile.
    fs : float
        Sampling rate in Hz.
    padlevel : int, optional
        Zero-padding exponent: ``nfft = 2**(ceil(log2(T)) + padlevel)``
        (default 4).
    fc : float, optional
        Upper cutoff frequency in Hz (default 10.0).
    amp_th : float, optional
        Normalized amplitude threshold selecting the spectral band
        (default 0.05).

    Returns
    -------
    float
        The spectral arc length (SAL). ``nan`` for a zero speed profile
        (a perfectly still joint), whose spectrum carries no energy and
        whose smoothness is therefore undefined.

    Notes
    -----
    Source: Balasubramanian et al. 2015, "On the analysis of movement
    smoothness." Reimplemented in NumPy; validated against the authors'
    reference output (see ``tests/test_smoothness_golden.py``).
    """
    speed = np.asarray(speed, dtype=np.float64)
    n = speed.shape[0]
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    freq = fs * np.arange(nfft) / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    peak = mag.max()
    if peak == 0:
        return float("nan")  # flat/zero speed: spectrum is degenerate
    mag = mag / peak

    in_band = freq <= fc
    freq_sel = freq[in_band]
    mag_sel = mag[in_band]

    above = np.nonzero(mag_sel >= amp_th)[0]
    if above.size == 0:
        return float("nan")  # no component above the amplitude threshold
    lo, hi = int(above[0]), int(above[-1])
    freq_sel = freq_sel[lo:hi + 1]
    mag_sel = mag_sel[lo:hi + 1]
    if freq_sel.size < 2:
        return float("nan")  # single-point band (constant speed): no arc to trace

    arc = np.sqrt((np.diff(freq_sel) / (freq_sel[-1] - freq_sel[0])) ** 2
                  + np.diff(mag_sel) ** 2)
    return float(-arc.sum())


def dimensionless_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> float:
    """Dimensionless jerk (DLJ) smoothness of a speed profile.

    ``-(duration³ / peak²) · ∫ (d²v/dt²)² dt`` — scale-invariant integrated
    squared jerk. More negative is less smooth.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    float
        The dimensionless jerk (``≤ 0``).

    Notes
    -----
    Source: Hogan & Sternad 2009; Balasubramanian et al. Validated against
    the reference output.
    """
    speed = np.asarray(speed, dtype=np.float64)
    dt = 1.0 / fs
    duration = speed.shape[0] * dt
    peak = np.abs(speed).max()
    jerk = np.diff(speed, 2) / dt ** 2
    scale = duration ** 3 / peak ** 2
    return float(-scale * np.sum(jerk ** 2) * dt)


def log_dimensionless_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> float:
    """Log dimensionless jerk (LDLJ) — ``-ln|DLJ|``.

    The log transform of :func:`dimensionless_jerk`, the form most used in
    practice. More negative is less smooth.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    float
        ``-ln|DLJ|``. A zero-jerk (constant-speed) profile is perfectly
        smooth and returns ``+inf``.

    Notes
    -----
    Source: Balasubramanian et al. Validated against the reference output.
    """
    dlj = dimensionless_jerk(speed, fs)
    if dlj == 0:
        return float("inf")  # zero jerk -> perfectly smooth
    return float(-np.log(np.abs(dlj)))


def number_of_peaks(speed: npt.NDArray[np.float64]) -> int:
    """Number of local maxima in a speed profile.

    A simple smoothness proxy — a single smooth movement has one velocity
    peak; more peaks mean more sub-movements.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.

    Returns
    -------
    int
        Count of strict interior local maxima.

    Notes
    -----
    Source: Balasubramanian et al. (number-of-peaks metric).
    """
    speed = np.asarray(speed, dtype=np.float64)
    interior = speed[1:-1]
    return int(np.sum((interior > speed[:-2]) & (interior > speed[2:])))


def speed_metric(speed: npt.NDArray[np.float64]) -> float:
    """Mean-to-peak speed ratio — ``mean(v) / max(v)``, in ``[0, 1]``.

    A bell-shaped (smooth) speed profile has a low ratio; a flat plateau
    approaches 1.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.

    Returns
    -------
    float
        The mean/peak ratio.

    Notes
    -----
    Source: Balasubramanian et al. (speed-metric); Flash & Hogan.
    """
    speed = np.asarray(speed, dtype=np.float64)
    peak = np.abs(speed).max()
    return float(speed.mean() / peak) if peak > 0 else float("nan")


def integrated_squared_jerk(speed: npt.NDArray[np.float64], fs: float) -> float:
    """Integrated squared jerk — ``∫ (d²v/dt²)² dt`` (dimensional)."""
    speed = np.asarray(speed, dtype=np.float64)
    dt = 1.0 / fs
    jerk = np.diff(speed, 2) / dt ** 2
    return float(np.sum(jerk ** 2) * dt)


def mean_squared_jerk(speed: npt.NDArray[np.float64], fs: float) -> float:
    """Mean squared jerk — ``mean((d²v/dt²)²)``."""
    speed = np.asarray(speed, dtype=np.float64)
    dt = 1.0 / fs
    jerk = np.diff(speed, 2) / dt ** 2
    return float(np.mean(jerk ** 2))


def rms_squared_jerk(speed: npt.NDArray[np.float64], fs: float) -> float:
    """Root-mean-square jerk — ``sqrt(mean((d²v/dt²)²))``."""
    return float(np.sqrt(mean_squared_jerk(speed, fs)))


_SMOOTHNESS_FS_METRICS: dict[str, Callable[..., float]] = {
    "sparc": sparc,
    "dimensionless_jerk": dimensionless_jerk,
    "log_dimensionless_jerk": log_dimensionless_jerk,
    "integrated_squared_jerk": integrated_squared_jerk,
    "mean_squared_jerk": mean_squared_jerk,
    "rms_squared_jerk": rms_squared_jerk,
}
_SMOOTHNESS_PLAIN_METRICS: dict[str, Callable[..., float]] = {
    "number_of_peaks": number_of_peaks,
    "speed_metric": speed_metric,
}


def smoothness(
    speed: npt.NDArray[np.float64],
    fs: float,
    metric: str = "sparc",
    **kwargs: float,
) -> float:
    """Dispatch to a named smoothness metric on a 1-D speed profile.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.
    fs : float
        Sampling rate in Hz.
    metric : str, optional
        One of ``"sparc"`` (default), ``"dimensionless_jerk"``,
        ``"log_dimensionless_jerk"``, ``"integrated_squared_jerk"``,
        ``"mean_squared_jerk"``, ``"rms_squared_jerk"``,
        ``"number_of_peaks"``, ``"speed_metric"``.
    **kwargs
        Metric-specific options (e.g. ``padlevel`` / ``fc`` / ``amp_th``
        for ``"sparc"``).

    Returns
    -------
    float
        The selected smoothness scalar.

    Raises
    ------
    ValueError
        If ``metric`` is unknown.
    """
    if metric in _SMOOTHNESS_FS_METRICS:
        return _SMOOTHNESS_FS_METRICS[metric](speed, fs, **kwargs)
    if metric in _SMOOTHNESS_PLAIN_METRICS:
        return _SMOOTHNESS_PLAIN_METRICS[metric](speed, **kwargs)
    known = sorted(_SMOOTHNESS_FS_METRICS) + sorted(_SMOOTHNESS_PLAIN_METRICS)
    raise ValueError(f"Unknown smoothness metric {metric!r}; choose from {known}.")


# ----------------------------------------------------------------
#  Signal reductions — array-pure
# ----------------------------------------------------------------

VelocityReductions = namedtuple(
    "VelocityReductions", ["peak", "mean", "peak_to_mean", "peak_deceleration"])


def velocity_reductions(
    speed: npt.NDArray[np.float64],
    fs: float = 1.0,
) -> VelocityReductions:
    """Scalar reductions of a speed profile.

    Parameters
    ----------
    speed : ndarray, shape (T,)
        1-D speed profile.
    fs : float, optional
        Sampling rate in Hz (default 1.0); scales ``peak_deceleration``
        into units/second.

    Returns
    -------
    VelocityReductions
        Named tuple ``(peak, mean, peak_to_mean, peak_deceleration)`` —
        ``peak_deceleration`` is the largest instantaneous rate of speed
        decrease (``≥ 0``).

    Notes
    -----
    Source: Pollick et al., Halovic & Kroos, Samadani et al.
    """
    speed = np.asarray(speed, dtype=np.float64)
    peak = float(speed.max())
    mean = float(speed.mean())
    rate = np.diff(speed) * fs
    peak_deceleration = float(-rate.min()) if rate.size else 0.0
    peak_to_mean = peak / mean if abs(mean) > _EPS else float("nan")
    return VelocityReductions(peak, mean, peak_to_mean, peak_deceleration)


def zero_crossings(
    signal: npt.NDArray[np.float64],
    axis: int = 0,
) -> npt.NDArray[np.int_]:
    """Count sign changes of a signal along an axis.

    Strict crossings only — consecutive samples with a product ``< 0``;
    exact zeros are not counted as crossings.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    axis : int, optional
        Axis along which to count (default 0).

    Returns
    -------
    ndarray
        Crossing counts with ``axis`` removed (a scalar for 1-D input).

    Notes
    -----
    Source: Zhao & Badler (motion feature counts).
    """
    signal = np.asarray(signal, dtype=np.float64)
    lo = np.take(signal, np.arange(signal.shape[axis] - 1), axis=axis)
    hi = np.take(signal, np.arange(1, signal.shape[axis]), axis=axis)
    return np.sum((lo * hi) < 0, axis=axis)


def active_segments(
    speed: npt.NDArray[np.float64],
    threshold: float,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of "active" (above-threshold) samples.

    Parameters
    ----------
    speed : ndarray
        Speed (or any non-negative activity) signal.
    threshold : float
        Activity threshold; samples strictly above it are active. No
        hidden default — the caller picks the threshold, keeping this a
        theory-neutral primitive.

    Returns
    -------
    ndarray of bool
        ``speed > threshold``.

    Notes
    -----
    Source: Pollick et al., Bernhardt & Robinson.
    """
    return np.asarray(speed, dtype=np.float64) > threshold


def active_duration(
    speed: npt.NDArray[np.float64],
    threshold: float,
    frame_time: float = 1.0,
) -> float:
    """Total time spent active — active sample count × ``frame_time``.

    Parameters
    ----------
    speed : ndarray
        Speed signal.
    threshold : float
        Activity threshold (see :func:`active_segments`).
    frame_time : float, optional
        Seconds per sample (default 1.0 → returns a sample count).

    Returns
    -------
    float
        Active duration.
    """
    return float(np.count_nonzero(active_segments(speed, threshold)) * frame_time)


# ----------------------------------------------------------------
#  Kinetic energy & gait — [Bvh]
# ----------------------------------------------------------------

def kinetic_energy(
    bvh: Bvh,
    masses: npt.NDArray[np.float64] | None = None,
    centered: str = "world",
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Per-frame kinetic energy summed over joints.

    With ``masses``, ``Σ_j ½ m_j ‖v_j‖²`` (true kinetic energy). Without,
    ``Σ_j ‖v_j‖²`` (unit-mass energy proxy) — pybvh ships no segment-mass
    model, so pass anatomical masses for physical energy.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    masses : ndarray, shape (J,), optional
        Per-joint masses (joint-axis order, end sites excluded). Default
        None → unit-mass proxy.
    centered : str, optional
        Centering mode for the velocities (default ``"world"``).
    stencil, pad : optional
        Velocity finite-difference convention (see
        :func:`joint_velocities`).

    Returns
    -------
    ndarray
        Per-frame energy; leading length follows the velocity
        ``stencil`` × ``pad`` shape.

    Notes
    -----
    Source: Głowinski et al., Piana et al., Lu et al. 2025.
    """
    vel = joint_velocities(bvh, centered=centered, stencil=stencil, pad=pad)
    speed_sq = np.sum(vel ** 2, axis=-1)  # (F, J)
    if masses is None:
        return speed_sq.sum(axis=-1)
    m = np.asarray(masses, dtype=np.float64)
    return 0.5 * np.sum(m * speed_sq, axis=-1)


def _root_horizontal_distance(bvh: Bvh) -> float:
    """Path length of the root projected onto the ground plane."""
    from .tools import _axis_to_vector
    up = _axis_to_vector(bvh.world_up)
    root = bvh.root_pos
    height = root @ up
    horizontal = root - height[:, None] * up
    return float(geometry.path_length(horizontal))


def _contact_onsets(bvh: Bvh, foot_joints: list[str] | None) -> int:
    """Number of foot-contact onsets (0→1 transitions) across all feet."""
    contacts = foot_contacts(bvh, foot_joints=foot_joints)
    assert isinstance(contacts, np.ndarray)
    in_contact = contacts > 0.5
    onsets = in_contact[1:] & ~in_contact[:-1]
    return int(onsets.sum())


def cadence(bvh: Bvh, foot_joints: list[str] | None = None) -> float:
    """Step rate — foot-contact onsets per second.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; auto-detected if None.

    Returns
    -------
    float
        Steps per second (0.0 if the clip has no duration).

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012, Karg et al. 2010.
    """
    duration = (bvh.frame_count - 1) * bvh.frame_time
    if duration <= 0:
        return 0.0
    return _contact_onsets(bvh, foot_joints) / duration


def stride_length(bvh: Bvh, foot_joints: list[str] | None = None) -> float:
    """Mean stride length — horizontal distance travelled per stride.

    One stride is two steps (a full gait cycle), so this is the root's
    horizontal path length divided by ``onsets / 2``.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; auto-detected if None.

    Returns
    -------
    float
        Distance per stride in skeleton units (``nan`` if no strides
        were detected).

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012, Karg et al. 2010.
    """
    n_strides = _contact_onsets(bvh, foot_joints) / 2.0
    if n_strides <= 0:
        return float("nan")
    return _root_horizontal_distance(bvh) / n_strides


def walking_pace(bvh: Bvh, foot_joints: list[str] | None = None) -> float:
    """Mean horizontal speed — root ground-path length per second.

    Equals ``stride_length × cadence / 2`` by construction.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; unused for the distance itself, accepted for API
        symmetry with the other gait metrics.

    Returns
    -------
    float
        Horizontal units per second (0.0 if the clip has no duration).

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012.
    """
    duration = (bvh.frame_count - 1) * bvh.frame_time
    if duration <= 0:
        return 0.0
    return _root_horizontal_distance(bvh) / duration


def range_of_motion(
    signal: npt.NDArray[np.float64],
    axis: int = 0,
) -> npt.NDArray[np.float64]:
    """Peak-to-peak range of a signal — ``max − min`` along an axis.

    For a joint-angle channel this is its range of motion over the clip.

    Parameters
    ----------
    signal : ndarray
        Input signal (e.g. a joint-angle time series).
    axis : int, optional
        Axis to reduce over (default 0, the frame axis).

    Returns
    -------
    ndarray
        The peak-to-peak range with ``axis`` removed.

    Notes
    -----
    Source: gait / biomechanics range-of-motion descriptors.
    """
    return np.ptp(np.asarray(signal, dtype=np.float64), axis=axis)


# ----------------------------------------------------------------
#  Covariance descriptors — array-pure
# ----------------------------------------------------------------

def cov3dj(pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Covariance of 3D joint positions over time (Cov3DJ).

    Flattens each frame's joints to a ``3N`` vector and returns the
    ``(3N, 3N)`` sample covariance across frames — a fixed-size pose
    descriptor independent of sequence length.

    Parameters
    ----------
    pos : ndarray, shape (F, N, 3)
        Per-frame joint positions.

    Returns
    -------
    ndarray, shape (3N, 3N)
        The sample covariance matrix.

    Notes
    -----
    Source: Hussein et al. (Cov3DJ).
    """
    pos = np.asarray(pos, dtype=np.float64)
    flat = pos.reshape(pos.shape[0], -1)  # (F, 3N)
    centered = flat - flat.mean(axis=0)
    return (centered.T @ centered) / flat.shape[0]


def lagged_correlation(
    signal: npt.NDArray[np.float64],
    lag: int,
) -> npt.NDArray[np.float64]:
    """Lagged covariance matrix — ``M(l) = (1/(T−l)) Vᵀ[l:] V[:-l]``.

    Captures temporal structure between channels at a fixed lag, averaged
    over the ``T − l`` overlapping sample pairs (so every entry is a mean,
    independent of the lag).

    Parameters
    ----------
    signal : ndarray, shape (T, D)
        Multichannel signal (time × channels).
    lag : int
        Non-negative lag in samples.

    Returns
    -------
    ndarray, shape (D, D)
        The lagged covariance.

    Raises
    ------
    ValueError
        If ``lag`` is negative or ``>= T``.

    Notes
    -----
    Source: Venture et al. (lagged covariance descriptors).
    """
    signal = np.asarray(signal, dtype=np.float64)
    t = signal.shape[0]
    if lag < 0 or lag >= t:
        raise ValueError(f"lag must be in [0, {t}), got {lag}.")
    if lag == 0:
        ahead, behind = signal, signal
    else:
        ahead, behind = signal[lag:], signal[:-lag]
    return (ahead.T @ behind) / ahead.shape[0]
