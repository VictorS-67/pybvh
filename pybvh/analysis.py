"""Motion analysis for BVH data.

Velocities, accelerations, angular velocities, root trajectory, and foot
contacts. Every function takes a :class:`~pybvh.bvh.Bvh` object as its
first argument; thin wrapper methods on the ``Bvh`` class delegate here.

Feature-array export for ML pipelines lives in :mod:`pybvh.features`.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt

from . import _contacts, geometry, rotations
from ._orientation import _root_heading_units
from ._warnings import user_stacklevel
from .bvh import Bvh
from .bvhnode import BvhNode
from .signal import finite_difference as _finite_difference
from .tools import _axis_to_vector

_EPS = 1e-12


# ----------------------------------------------------------------
#  Joint velocities & accelerations
# ----------------------------------------------------------------


def _validate_stencil_pad(stencil: str, pad: str) -> None:
    if stencil not in ("central", "forward"):
        raise ValueError(f"stencil must be 'central' or 'forward', got {stencil!r}")
    if pad not in ("edge", "none"):
        raise ValueError(f"pad must be 'edge' or 'none', got {pad!r}")


def _finite_difference_min_frames(order: int, stencil: str, pad: str) -> int:
    """Fewest frames a clip needs for an ``order``-th derivative.

    A policy about clips, stricter than what
    :func:`pybvh.signal.finite_difference` needs to compute: every
    combination must leave at least one frame the stencil defines, and
    ``"central"`` with ``pad="edge"`` at least one central rather than
    one-sided difference.
    """
    if stencil == "forward":
        return order + 1
    return 2 * order + 1 if pad == "none" else 3


def _non_end_site_indices(bvh: Bvh) -> list[int]:
    """Indices in ``nodes`` order that correspond to non-end-site joints.

    The same indices select the joint-axis subset of any per-node array
    (e.g. ``node_positions()`` output of shape ``(F, N, 3)``) to produce
    a joint-aligned ``(F, J, 3)``.
    """
    return [i for i, n in enumerate(bvh.nodes) if not n.is_end_site()]


def _validate_node_coords(bvh: Bvh, coords: npt.NDArray[np.float64] | None) -> None:
    """Reject ``coords`` whose joint axis is not node-shaped.

    The ``joint_*`` kinematics functions subset the node axis down to
    non-end-site joints; a joint-shaped ``(F, J, 3)`` input would be
    silently mis-indexed.
    """
    n_nodes = len(bvh.nodes)
    if coords is not None and coords.shape[1] != n_nodes:
        raise ValueError(
            f"coords must be node-shaped (F, N, 3) with N = {n_nodes} nodes "
            f"(joints + end sites, as returned by Bvh.node_positions()); got "
            f"{coords.shape}."
        )


def node_velocities(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Compute per-node position velocities (joints + end sites).

    Differentiates through :func:`pybvh.signal.finite_difference`, with
    two orthogonal choices:

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
    min_frames = _finite_difference_min_frames(1, stencil, pad)
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r} requires at least {min_frames} frames "
            f"to compute velocities (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second velocity. "
            "Use in_frames=True for per-frame velocity."
        )

    if coords is None:
        coords = bvh.node_positions(centered=centered)

    dt = 1.0 if in_frames else bvh.frame_time
    return _finite_difference(coords, dt, stencil=stencil, pad=pad)


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

    Raises
    ------
    ValueError
        If ``coords`` is not node-shaped ``(F, N, 3)`` (in addition to
        the :func:`node_velocities` conditions).
    """
    _validate_node_coords(bvh, coords)
    nv = node_velocities(
        bvh, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
    )
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

    Applies the chosen ``stencil`` twice to the input positions and pads
    or crops once, at the end:
    :func:`pybvh.signal.finite_difference` with ``order=2``.

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
    min_frames = _finite_difference_min_frames(2, stencil, pad)
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r}, pad={pad!r} requires at least "
            f"{min_frames} frames (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second acceleration. "
            "Use in_frames=True for per-frame acceleration."
        )

    if coords is None:
        coords = bvh.node_positions(centered=centered)

    dt = 1.0 if in_frames else bvh.frame_time
    return _finite_difference(coords, dt, order=2, stencil=stencil, pad=pad)


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

    Raises
    ------
    ValueError
        If ``coords`` is not node-shaped ``(F, N, 3)`` (in addition to
        the :func:`node_accelerations` conditions).
    """
    _validate_node_coords(bvh, coords)
    na = node_accelerations(
        bvh, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
    )
    return na[:, _non_end_site_indices(bvh), :]


def node_speed_derivative(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Per-node rate of change of speed ``d‖v‖/dt`` (joints + end sites).

    Computes the per-node speed ``‖v‖`` from :func:`node_velocities`, then
    applies the same finite-difference stencil once more to that scalar
    series.  Positive values mean the node is *speeding up*, negative
    values mean it is *slowing down* — the natural "is the movement
    accelerating or braking" signal, independent of direction changes.

    This is the **tangential acceleration** ``a_t = d‖v‖/dt`` of the
    node's trajectory.  The complementary normal (centripetal) component
    is ``a_n = ‖v‖² · κ`` with ``κ`` the trajectory curvature
    (:func:`pybvh.geometry.curvature`), and the two decompose the full
    acceleration vector: ``a_t² + a_n² = ‖a‖²``.

    It is **not recoverable from** :func:`node_accelerations`: this is
    the difference of the norm (``Δ‖v⃗‖``), not the norm of the
    difference (``‖Δv⃗‖ = ‖a⃗‖``).  A direction change at constant speed
    gives ``‖a⃗‖ > 0`` but ``d‖v‖/dt = 0``.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    centered : str, optional
        Coordinate centering mode (default ``"world"``).
        **Ignored if `coords` is provided** — `coords` takes precedence.
    in_frames : bool, optional
        If True, return the rate in units/frame^2.
        If False (default), return in units/second^2.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates. If None, computed
        internally via :meth:`Bvh.node_positions`.
    stencil : {"central", "forward"}, optional
        Finite-difference method, applied at both stages (positions →
        velocities, speed → its derivative).  Default ``"central"``.
    pad : {"edge", "none"}, optional
        Boundary handling, applied at both stages.  ``"edge"``
        (default): output shape equals input length ``(F, N)``.
        ``"none"``: drop boundary frames the repeated stencil can't
        define — central drops 4 frames total ``(F-4, N)``; forward
        drops 2 ``(F-2, N)``.

    Returns
    -------
    ndarray
        Shape depends on ``stencil`` × ``pad`` — the frame trimming of
        :func:`node_accelerations`, without the trailing 3-axis:

        =========  ======  ================
        stencil    pad     shape
        =========  ======  ================
        central    edge    ``(F, N)``
        central    none    ``(F-4, N)``
        forward    edge    ``(F, N)``
        forward    none    ``(F-2, N)``
        =========  ======  ================

    See Also
    --------
    joint_speed_derivative : Same data restricted to non-end-site
        joints (``(F, J)``).
    node_accelerations : The vector second derivative — its norm is the
        full ``‖a‖``, of which this is the tangential component.
    velocity_reductions : ``peak_acceleration`` / ``peak_deceleration``
        are the extrema of this series (``stencil="forward"``,
        ``pad="none"``).

    Raises
    ------
    ValueError
        If the clip is too short for the chosen combination,
        ``frame_time == 0`` when ``in_frames=False``, or either
        parameter is invalid.  Minimum frames match
        :func:`node_accelerations` (two stencil applications): 3 for
        ``central``+``edge``, ``forward``+``edge``, and
        ``forward``+``none``; 5 for ``central``+``none``.

    Notes
    -----
    Source: Hachimura et al. 2005 (Time Effort).
    """
    _validate_stencil_pad(stencil, pad)
    min_frames = _finite_difference_min_frames(2, stencil, pad)
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r}, pad={pad!r} requires at least "
            f"{min_frames} frames (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute a per-second speed derivative. "
            "Use in_frames=True for per-frame units."
        )

    vel = node_velocities(
        bvh, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
    )
    speed = np.linalg.norm(vel, axis=-1)
    dt = 1.0 if in_frames else bvh.frame_time
    return _finite_difference(speed, dt, stencil=stencil, pad=pad)


def joint_speed_derivative(
    bvh: Bvh,
    centered: str = "world",
    in_frames: bool = False,
    coords: npt.NDArray[np.float64] | None = None,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Per-joint rate of change of speed ``d‖v‖/dt`` (end sites excluded).

    Returns the joint-axis subset of :func:`node_speed_derivative` —
    same two-stage finite-difference math, restricted to non-end-site
    joints so output indexes match :attr:`Bvh.joint_angles`. Output
    shape is ``(F, J)`` (or the appropriate trimmed variant per
    ``stencil`` × ``pad``).

    See :func:`node_speed_derivative` for the full parameter / shape /
    sign-semantics docs.

    Raises
    ------
    ValueError
        If ``coords`` is not node-shaped ``(F, N, 3)`` (in addition to
        the :func:`node_speed_derivative` conditions).
    """
    _validate_node_coords(bvh, coords)
    sd = node_speed_derivative(
        bvh, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
    )
    return sd[:, _non_end_site_indices(bvh)]


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

    These are the rates of each joint's **parent-relative (local)
    rotation** — the BVH channels themselves — expressed in the joint's
    own (body) frame: ``ω[i] = log(R_i^T @ R_{i+1}) / dt`` on the local
    rotation matrices, no forward kinematics involved. Two published
    alternatives differ: the **world-frame angular velocity of a
    segment** (the biomechanics convention) composes rotations down the
    chain first and diverges for every joint whose parent is itself
    rotating; and the **spatial-frame** rate ``log(R_{i+1} @ R_i^T)``
    differs from the body-frame rate by conjugation with ``R`` — same
    magnitude, rotated axis. The local convention is what BVH data
    natively parameterizes; compose ``Bvh.to_rotmat`` output through
    the hierarchy yourself if you need segment rates.

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
            f"stencil={stencil!r} requires at least {min_frames} frames (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second angular velocity. "
            "Use in_frames=True for per-frame angular velocity."
        )

    _, R = bvh.to_rotmat()  # (F, J, 3, 3)
    F = R.shape[0]

    def _finalize(omega: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Shared radians/frame → requested-units post-processing."""
        if not in_frames:
            omega = omega / bvh.frame_time
        if degrees:
            omega = np.degrees(omega)
        return omega

    if stencil == "forward":
        # ω[i] = log(R_i^T @ R_{i+1})
        R_rel = np.einsum("...ji,...jk->...ik", R[:-1], R[1:])  # (F-1, J, 3, 3)
        ang_vel = rotations.rotmat_to_axisangle(R_rel)  # radians/frame
        if pad == "edge":
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]], axis=0)  # (F, J, 3)
        return _finalize(ang_vel)

    # stencil == "central": two-step ω[i] = log(R_{i-1}^T R_{i+1}) / 2
    R_rel_central = np.einsum("...ji,...jk->...ik", R[:-2], R[2:])  # (F-2, J, 3, 3)
    omega_central = rotations.rotmat_to_axisangle(R_rel_central) / 2.0  # rad/frame

    if pad == "none":
        return _finalize(omega_central)  # (F-2, J, 3)

    # pad == "edge": one-sided forward/backward at boundaries
    omega = np.empty((F,) + R.shape[1:-2] + (3,), dtype=np.float64)
    omega[1:-1] = omega_central
    R_rel_first = np.einsum("...ji,...jk->...ik", R[0:1], R[1:2])
    omega[0:1] = rotations.rotmat_to_axisangle(R_rel_first)
    R_rel_last = np.einsum("...ji,...jk->...ik", R[-2:-1], R[-1:])
    omega[-1:] = rotations.rotmat_to_axisangle(R_rel_last)
    return _finalize(omega)  # (F, J, 3)


# ----------------------------------------------------------------
#  Root trajectory
# ----------------------------------------------------------------


def root_trajectory(
    bvh: Bvh,
    world_up: str | None = None,
    include_velocities: bool = False,
    stencil: str = "central",
    pad: str = "edge",
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Extract root trajectory features commonly used in motion ML.

    Returns the root's ground-plane position and heading angle (as
    ``sin``/``cos`` pair).  Optionally appends ground-plane and
    heading velocities.

    The heading reference is :attr:`~pybvh.bvh.Bvh.rest_forward`, the
    direction the rest pose faces in the skeleton's own frame (rest left
    × rest up, read from the rest pose alone). The root rotation of each
    frame carries it into the world, and the heading is its projection
    on the ground. At a frame whose root rotation is identity the
    heading is therefore rest forward's own ground direction, whatever
    rotation the clip starts with. When rest forward is the house
    default (no usable left/right geometry; reading it warns), the
    heading is still built on it, and both the heading and its rate of
    turn (``heading_vel``) depend on that guessed reference whenever
    the root pitches or rolls: a tilted root's forward projects on the
    ground at an angle, and turns there at a rate, that differ with the
    forward it starts from. On a defaulted rest forward they are
    reliable only for a root that turns about world up alone.
    :func:`facing_frame` chooses otherwise for its fallback, which takes
    no heading from a defaulted rest forward.

    The heading is **orientation-derived** — the root bone's rotation
    applied to the rest forward, projected to the ground plane — not
    the direction of travel (a side-stepping character keeps its
    heading). It therefore inherits pelvis twist, and its ground-plane
    projection shrinks when the character bends far forward;
    :func:`facing_frame` is the whole-body, yaw-only alternative that
    avoids both. For the direction of motion, differentiate the ground
    position (the ``ground_*_vel`` columns).

    On a frame where that projection is too short to read (the root's
    forward stands along world up, norm below ``1e-6``) the heading is
    undefined, and its ``heading_sin`` and ``heading_cos`` are ``NaN``,
    the :mod:`pybvh.geometry` policy for values the motion makes
    undefined, rather than an arbitrary angle. With
    ``include_velocities=True`` the ``NaN`` spreads into the
    ``heading_vel`` samples whose finite-difference stencil reads that
    frame (with the default central stencil, the frames either side of
    it); the heading is unwrapped over the readable frames only, so no
    other sample is affected. The ground columns are always finite.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    world_up : str or None, optional
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
    # Resolve up axis (honors bvh.world_up by default)
    up_str = bvh.world_up if world_up is None else world_up
    up_idx = {"x": 0, "y": 1, "z": 2}[up_str[1]]
    ground_axes = [i for i in range(3) if i != up_idx]

    # Rest forward, in the skeleton's own frame: the root rotation is what
    # carries it into the world, frame by frame.
    # (F, 3), NaN rows
    fwd_ground = _root_heading_units(bvh._root_rotmats(), bvh.rest_forward, up_str)

    ground_pos = bvh.root_pos[:, ground_axes]  # (F, 2)
    heading = np.arctan2(fwd_ground[:, ground_axes[1]], fwd_ground[:, ground_axes[0]])  # (F,)

    base = np.column_stack(
        [
            ground_pos,
            np.sin(heading),
            np.cos(heading),
        ]
    )  # (F, 4)

    if not include_velocities:
        return base

    _validate_stencil_pad(stencil, pad)
    if bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute root_trajectory velocities. "
            "Set bvh.frame_time to a non-zero value first."
        )
    min_frames = _finite_difference_min_frames(1, stencil, pad)
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r} requires at least {min_frames} frames (have {bvh.frame_count})."
        )

    dt = bvh.frame_time
    # Unwrapped over the readable frames only: np.unwrap carries a NaN
    # into every later sample, and a NaN heading should reach only the
    # velocity samples whose stencil reads it.
    readable = ~np.isnan(heading)
    heading_unwrapped = heading.copy()
    heading_unwrapped[readable] = np.unwrap(heading[readable])
    ground_vel = _finite_difference(ground_pos, dt, stencil=stencil, pad=pad)
    heading_vel = _finite_difference(heading_unwrapped, dt, stencil=stencil, pad=pad)
    if pad == "edge":
        base_aligned = base  # (F, 4)
    elif stencil == "central":
        base_aligned = base[1:-1]  # (F-2, 4)
    else:  # forward + none: "drop first frame" alignment convention
        base_aligned = base[1:]  # (F-1, 4)

    if degrees:
        heading_vel = np.degrees(heading_vel)
    return np.column_stack([base_aligned, ground_vel, heading_vel])


# ----------------------------------------------------------------
#  Facing frame
# ----------------------------------------------------------------

FacingFrame = namedtuple("FacingFrame", ["forward", "left", "up", "valid"])


def facing_frame(
    bvh: Bvh,
    coords: npt.NDArray[np.float64] | None = None,
) -> FacingFrame:
    """Per-frame facing basis of the character, as continuous unit vectors.

    Returns the orthonormal right-handed triple ``(forward, left, up)``
    for every frame — the continuous form of the axis-label pair
    :meth:`Bvh.forward_at` / :meth:`Bvh.left_at`, which snap exactly
    this basis to the nearest signed world axis. Use the labels for
    rest-pose canonicalization and dataset-convention checks; use this
    function whenever the actual facing direction matters — the snapped
    label stays constant while a character turns through less than 90°,
    the vectors track the rotation frame by frame.

    This is a **yaw-only, gravity-aligned facing frame**: ``up`` is the
    exact :attr:`Bvh.world_up` unit vector on every frame, so the basis
    only ever rotates about the vertical — deliberately *not* a pelvis
    orientation (no roll or pitch; a character bending forward keeps
    their facing frame level). It is also distinct from
    :func:`root_trajectory`'s heading, which is a *second facing
    estimate built differently* — the root bone's rotation applied to
    the rest forward — not a direction of motion: that heading inherits
    pelvis twist and collapses toward numerical ambiguity when the
    character bends far forward, while this basis measures the
    whole-body L/R geometry and stays yaw-only. The two diverge
    whenever the pelvis turns or tilts relative to the body. Neither
    is velocity-based; for the direction of *travel*, differentiate
    ground position (``root_trajectory``'s ``ground_*_vel`` columns).

    Construction (per frame, all in world space): the leftward
    direction is the average of ``(left_pos - right_pos)`` over the
    L/R joint pairs in :attr:`Bvh.lr_mapping`, projected onto the
    horizontal plane and normalized; ``forward = leftward × up``;
    ``left = up × forward`` (re-orthogonalized). The triple satisfies
    ``forward × left = up``.

    Fallback policy (shared by :meth:`Bvh.forward_at` and
    :meth:`Bvh.left_at`): a frame whose horizontal ``(left - right)``
    average nearly vanishes or lies along ``world_up`` (norm below
    ``1e-6``, the body on its side) takes the root's heading at that
    frame (the one :func:`root_trajectory` returns) as its forward, when
    :attr:`Bvh.rest_forward` is measured and that heading is readable
    (:func:`root_trajectory` builds a heading on a defaulted rest
    forward too, for its rate of turn; as a facing it would offer a
    guessed direction as a measured one); otherwise the arbitrary-but-stable per-up-axis default (``'+z'`` for
    a ``y``-up world), with a ``UserWarning`` that names the frames. A
    skeleton with no ``lr_mapping`` takes that default on every frame,
    with a warning that names the skeleton. Every row keeps
    ``forward = left × up``, so the triple stays orthonormal and
    right-handed, and ``left`` is never ``world_up``. The ``valid``
    field reports exactly those frames: ``False`` on every fallback row,
    heading or default (all-``False`` when the skeleton has no L/R pairs
    at all). The fallback rows still carry a usable basis rather than
    ``nan``, so rendering-style consumers can ignore ``valid``; anything
    *measuring* facing should mask by it. The bvhplot follow camera
    does: it takes no heading from such frames, and interpolates across
    them from the measured frames around them.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates for all frames (as returned
        by ``Bvh.node_positions()``): one row per clip frame, since the
        fallback reads the root's rotation at each. When provided,
        skips the per-call forward kinematics.

    Returns
    -------
    FacingFrame
        Named tuple ``(forward, left, up, valid)``: three ``(F, 3)``
        float64 arrays of per-frame unit vectors forming a
        right-handed orthonormal basis, and a ``(F,)`` bool array,
        ``False`` on frames whose basis is a fallback rather than a
        measurement (see the fallback policy above).

    Raises
    ------
    ValueError
        If ``coords`` is not node-shaped, or does not hold one row per
        clip frame.

    See Also
    --------
    Bvh.forward_at, Bvh.left_at : The categorical (snapped axis label)
        form of the same construction.
    root_trajectory : Root-bone-orientation heading (a different facing
        estimate — see Notes above; not a travel direction).
    """
    if coords is None:
        coords = bvh.node_positions()
    else:
        coords = np.asarray(coords, dtype=np.float64)
    _validate_node_coords(bvh, coords)
    if coords.shape[0] != bvh.frame_count:
        raise ValueError(
            f"coords must hold one row per clip frame ({bvh.frame_count}), as "
            f"node_positions() returns them, since a frame whose left/right "
            f"axis cannot be read falls back to the root's heading at that "
            f"frame; got {coords.shape[0]} rows."
        )
    forward, left, up, valid = bvh._facing_at(coords, slice(None))
    return FacingFrame(forward, left, up, valid)


# ----------------------------------------------------------------
#  Foot contacts
# ----------------------------------------------------------------


@overload
def foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = ...,
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: Literal[False] = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = ...,
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: Literal[True],
) -> tuple[npt.NDArray[np.float64], _contacts.ContactReport]: ...
@overload
def foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = ...,
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: bool,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], _contacts.ContactReport]: ...
def foot_contacts(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    method: str = "combined",
    coords: npt.NDArray[np.float64] | None = None,
    *,
    vel_threshold: float | None = None,
    vel_smooth_duration: float | None = None,
    height_threshold: float | None = None,
    floor: float | str | None = None,
    min_contact_duration: float = 0.1,
    min_gap_duration: float = 0.1,
    hysteresis: float = 0.25,
    adaptive: bool = False,
    height_reference: str | None = None,
    return_info: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], _contacts.ContactReport]:
    """Detect binary foot contact labels per frame.

    The default combines a velocity check (foot not moving) **and** a
    height check (foot near the ground) following the HuMoR heuristic
    — each signal catches a different failure mode of the other.
    ``method="velocity"`` and ``method="height"`` remain available as
    single-signal escape hatches.

    Each method reads its own settings, and a setting given that the chosen method does not read raises a ``ValueError`` instead of being ignored: ``method="velocity"`` reads ``vel_threshold`` and ``vel_smooth_duration``; ``method="height"`` reads ``height_threshold`` and ``floor``; ``method="combined"`` reads all four, and ``height_reference`` when ``height_threshold`` is not given. Every method reads ``min_contact_duration``, ``min_gap_duration`` and ``hysteresis``, and ``adaptive`` where it leaves a threshold to derive. The settings some method does not read default to ``None``, which stands for the default named under each.

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
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed spatial coordinates.  Must be **world-frame**
        positions or a constant translation thereof (e.g.
        ``centered="first"`` output) — per-frame centerings such as
        ``centered="skeleton"`` distort the foot-speed signal.  If
        None, world-frame positions are computed internally.
    vel_threshold : float or None, keyword-only, optional
        Speed threshold in world units **per second**.  Defaults to
        ``0.12 × skeleton_scale`` u/s where ``skeleton_scale`` is the
        mean rest-pose distance from the root to the foot joints
        (equivalent to the pre-v0.8 ``0.004 × skeleton_scale`` per
        frame at 30 fps).  Scale-invariant across cm- and m-scale
        skeletons, and unaffected by finger/spine subdivision (unlike
        a median-bone-length reference, which shrinks when a skeleton
        has many short finger bones).
    vel_smooth_duration : float or None, keyword-only, optional
        Physical time span (seconds) the foot-speed estimator is
        conditioned over before thresholding.  Displacement vectors are
        box-averaged over ``max(1, round(vel_smooth_duration /
        frame_time))`` frames (capped at ``F - 1``) before taking the
        norm — for interior frames this equals differencing positions
        ``~vel_smooth_duration`` apart, making the *signal* (not just
        the threshold units) frame-rate independent: adjacent-frame
        differencing at 120 fps picks up high-frequency jitter that
        30 fps differencing averages out, splitting genuine stance
        phases.  Averaging the vectors (norm-of-mean, not
        mean-of-norms) lets oscillatory jitter cancel.  ``None``
        (default) means ``1/30`` s — a 1-frame no-op at ≤ 30 fps (labels identical to
        the raw signal there), 4 frames at 120 fps.  Set ``0.0`` to
        disable (raw adjacent-frame differencing).  Trade-off: a
        single-frame glitch smears over up to one window (~33 ms), the
        same timescale the duration filters already treat as noise.
        Applies to the velocity signal only; the height signal is
        position-level and is never smoothed.
    height_threshold : float or None, keyword-only, optional
        Clearance above the estimated floor, in world units.  Defaults
        to ``0.013 × skeleton_scale``.  A foot is "low enough" when
        ``foot_height − floor < height_threshold``.
    floor : float, ``"auto"``, ``"min"`` or None, keyword-only, optional
        Height of the reference the clearance is measured against, along
        the raw ``world_up`` axis.  ``"auto"`` (what ``None``, the default, means) estimates it as
        the 2nd percentile of the per-frame minimum foot height — robust
        to occasional spurious low frames — over the ``coords`` and the
        feet actually in use.  ``"min"`` uses the true minimum instead —
        exact, but a single glitched-low frame drags it down; the two
        diverge on clips with marker noise or long airborne phases.
        Pass a float to pin it explicitly (e.g. ``floor=0.0`` when the
        rig is already ground-aligned).

        This reference is **not** :attr:`Bvh.floor_height`, and passing
        that in is usually wrong here. The scene ground is measured over
        *all* nodes, so on a rig with toe end sites it sits a toe-length
        below the foot *joints* whose centres this function compares
        against — feeding it in makes every foot read as permanently
        airborne. The reference belongs to the joints being tested; the
        scene ground belongs to the clip.
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
    hysteresis : float, keyword-only, optional
        Schmitt-trigger band fraction (default ``0.25``). A frame enters
        swing only when its signal rises above ``threshold*(1+hysteresis)``
        and a contact is kept only if it ever drops below
        ``threshold*(1-hysteresis)`` — so an isolated dip near the
        threshold no longer flips the label. Strictly suppresses boundary
        flicker (it cannot invent contacts). Set to ``0.0`` for the plain
        single-threshold behaviour.
    adaptive : bool, keyword-only, optional
        If True, derive each foot's thresholds from its own signal
        distribution (Otsu's bimodal split between the stance and swing
        clusters) instead of the fixed scale fraction, falling back to the
        fixed default for any foot whose signal is not convincingly
        bimodal (e.g. standing, or a foot that never clearly swings).
        Default ``False``; recommended for known-locomotion clips where the
        fixed threshold under- or over-detects.  :func:`gait_parameters`
        enables this by default — calling it declares the clip is
        locomotion.  ``adaptive=True`` raises where it has no threshold to
        derive: every threshold the method reads given, or, under
        ``"combined"`` with ``height_reference="velocity"``, a
        ``vel_threshold`` given (the height threshold is then calibrated on
        the stance frames, not split).
    height_reference : {"velocity", "floor"} or None, keyword-only, optional
        How the default ``height_threshold`` is anchored (only used with ``method="combined"`` when ``height_threshold`` is None).
        ``"velocity"`` (what ``None``, the default, means): per-foot stance-median calibration — each foot's threshold is set from the median of its clearance over the frames where that foot is slow (speed below ``vel_threshold``, default ``0.12 × skeleton_scale`` u/s), so retargeted mocap whose feet hover above the estimated floor still detects stance; on rigs whose feet reach the floor this reduces to the ``"floor"`` margin.
        ``"floor"``: the fixed ``0.013 × skeleton_scale`` margin above the estimated floor — no per-foot calibration.
        The stance calibration presumes the joints contact the ground regularly; for arbitrary (non-foot) joint sets, :func:`ground_contacts` defaults to ``"floor"`` for exactly that reason.
    return_info : bool, keyword-only, optional
        If True, return ``(contacts, report)``, where ``report`` is a
        :class:`~pybvh.ContactReport`: the joints, the method and the
        settings applied, each joint's thresholds, the skeleton scale, the
        contact reference (``floor``), the ``speed`` and ``clearance``
        signals the thresholds were compared against, each joint's
        ``confidence`` and the unsupervised quality diagnostics
        (``foot_skate_mean`` / ``foot_skate_max``, ``airborne_fraction``,
        ``height_at_contact``). Every field is present, ``None`` where the
        method does not use it; a per-joint value is ``(n_joints,)``, one
        joint or several.

    Returns
    -------
    ndarray of shape ``(F, num_foot_joints)``, or
    tuple ``(ndarray, ContactReport)`` when ``return_info=True``.
        Binary contact labels (1.0 = contact, 0.0 = no contact).
        For ``"velocity"``/``"combined"``, frame 0 is propagated from
        frame 1 because velocity is undefined at frame 0.  Column
        order matches ``foot_joints`` (or, for auto-detection, the
        order returned by :func:`auto_detect_foot_joints`).

    Raises
    ------
    ValueError
        - If ``method`` or ``height_reference`` is unknown.
        - If a setting is given that ``method`` does not read: ``height_threshold``, ``floor`` or ``height_reference`` under ``"velocity"``; ``vel_threshold``, ``vel_smooth_duration`` or ``height_reference`` under ``"height"``; ``height_reference`` with a ``height_threshold`` given; ``adaptive=True`` with no threshold to derive.
        - If ``floor`` is a string other than ``"auto"`` or ``"min"``.
        - If no foot joints can be found or any named joint is
          missing from the skeleton.
        - If ``foot_joints`` is explicitly an empty list — contact
          detection needs at least one joint.
        - If ``vel_smooth_duration`` is negative.
        - When the height signal is involved and ``bvh.world_up`` is
          inconsistent with rest-pose geometry (feet above hips).
        - If ``frame_time == 0`` and the velocity signal (units/second)
          or a nonzero duration filter needs a time base.

    See Also
    --------
    ground_contacts : The same detection engine for arbitrary (non-foot) joint sets — hands, knees, props.
    auto_detect_foot_joints : The detection used when ``foot_joints`` is None.
    """
    joint_names, joint_indices = _resolve_contact_joints(bvh, foot_joints)
    settings = _contacts._ContactSettings.gather(
        method=method,
        vel_threshold=vel_threshold,
        vel_smooth_duration=vel_smooth_duration,
        height_threshold=height_threshold,
        floor=floor,
        min_contact_duration=min_contact_duration,
        min_gap_duration=min_gap_duration,
        hysteresis=hysteresis,
        adaptive=adaptive,
        height_reference=height_reference,
        default_height_reference="velocity",
    )
    return _contacts_on_clip(
        bvh,
        joint_names,
        joint_indices,
        settings,
        coords=coords,
        return_info=return_info,
        check_rest_height=True,
    )


@overload
def ground_contacts(
    bvh: Bvh,
    joints: Sequence[str | int],
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: Literal[False] = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def ground_contacts(
    bvh: Bvh,
    joints: Sequence[str | int],
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: Literal[True],
) -> tuple[npt.NDArray[np.float64], _contacts.ContactReport]: ...
@overload
def ground_contacts(
    bvh: Bvh,
    joints: Sequence[str | int],
    method: str = ...,
    coords: npt.NDArray[np.float64] | None = ...,
    *,
    vel_threshold: float | None = ...,
    vel_smooth_duration: float | None = ...,
    height_threshold: float | None = ...,
    floor: float | str | None = ...,
    min_contact_duration: float = ...,
    min_gap_duration: float = ...,
    hysteresis: float = ...,
    adaptive: bool = ...,
    height_reference: str | None = ...,
    return_info: bool,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], _contacts.ContactReport]: ...
def ground_contacts(
    bvh: Bvh,
    joints: Sequence[str | int],
    method: str = "combined",
    coords: npt.NDArray[np.float64] | None = None,
    *,
    vel_threshold: float | None = None,
    vel_smooth_duration: float | None = None,
    height_threshold: float | None = None,
    floor: float | str | None = None,
    min_contact_duration: float = 0.1,
    min_gap_duration: float = 0.1,
    hysteresis: float = 0.25,
    adaptive: bool = False,
    height_reference: str | None = None,
    return_info: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], _contacts.ContactReport]:
    """Detect ground-contact labels for an arbitrary set of joints.

    The same detection engine as :func:`foot_contacts` — velocity and/or clearance-above-floor thresholding with hysteresis and duration filters — but for any joint set: hands during floor work, knees in a crawl, a prop bone. Because the joints are not assumed to be feet, two foot-specific behaviors are dropped: there is no rest-pose "feet below hips" sanity check, and ``height_reference`` defaults to ``"floor"`` instead of :func:`foot_contacts`' ``"velocity"`` — the per-joint stance calibration behind ``"velocity"`` presumes regular ground contact, which feet in locomotion have and a hand that touches the floor twice does not.

    .. warning::
        With the default ``floor="auto"`` the floor is estimated **from the given joints' own trajectories** (2nd percentile of their per-frame minimum height). A joint set that never actually grounds makes its lowest hover point the "floor" and fabricates contacts there. Unless the joints genuinely reach the ground for a meaningful fraction of the clip, pass an explicit float. :attr:`Bvh.floor_height` is the scene ground and reads low for this purpose — it is measured over all nodes, including end sites that reach below any joint centre — so a joint compared against it needs a ``height_threshold`` widened by that offset.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    joints : sequence of str or int
        The joints to test, as node names and/or **node**-space indices (rows of :meth:`Bvh.node_positions`; NumPy-style negative indices allowed). End sites are legal — fingertips and toe tips are often exactly the grounding points. Output column order matches this sequence.
    method : {"combined", "velocity", "height"}, optional
        Same meaning as in :func:`foot_contacts`.
    coords : ndarray, shape (F, N, 3), optional
        Pre-computed **world-frame** positions (or a constant translation thereof), as in :func:`foot_contacts`.
    vel_threshold, vel_smooth_duration, height_threshold, floor, min_contact_duration, min_gap_duration, hysteresis, adaptive : keyword-only, optional
        Same meaning and defaults as in :func:`foot_contacts`, read by the same methods (a setting given that the method does not read raises). The auto thresholds scale by the mean rest-pose distance from the root to the **given** joints (the same skeleton-scale rule, applied to this joint set) — so the defaults stay proportionate for short chains like hands. Mind the ``floor="auto"`` warning above.
    height_reference : {"floor", "velocity"} or None, keyword-only, optional
        ``None`` (default) means ``"floor"``: the fixed ``0.013 × skeleton_scale`` margin above the floor. ``"velocity"`` enables :func:`foot_contacts`' per-joint stance calibration; only use it when the joints ground regularly enough to have stance statistics.
    return_info : bool, keyword-only, optional
        As in :func:`foot_contacts`; the report's ``joints`` always holds the resolved **names** (indices are mapped back through ``bvh.nodes``).

    Returns
    -------
    ndarray of shape ``(F, num_joints)``, or tuple ``(ndarray, ContactReport)`` when ``return_info=True``.
        Binary contact labels (1.0 = contact); columns follow ``joints`` order.

    Raises
    ------
    ValueError
        - If ``method`` or ``height_reference`` is unknown.
        - If a setting is given that ``method`` does not read, as in :func:`foot_contacts`.
        - If ``floor`` is a string other than ``"auto"`` or ``"min"``.
        - If ``joints`` is empty — contact detection needs at least one joint.
        - If a joint name is not in the skeleton.
        - If ``vel_smooth_duration`` is negative.
        - If ``frame_time == 0`` and the velocity signal (units/second) or a nonzero duration filter needs a time base.
    TypeError
        If a ``joints`` entry is neither a str nor an int (``bool`` is rejected explicitly — ``True``/``False`` silently indexing nodes 1/0 would be a trap).
    IndexError
        If a node index is out of range for ``bvh.nodes``.

    See Also
    --------
    foot_contacts : The foot-specialized entry point (auto-detection, rest-pose sanity check, ``height_reference="velocity"`` default).
    """
    joint_names, joint_indices = _resolve_contact_joints(bvh, joints)
    settings = _contacts._ContactSettings.gather(
        method=method,
        vel_threshold=vel_threshold,
        vel_smooth_duration=vel_smooth_duration,
        height_threshold=height_threshold,
        floor=floor,
        min_contact_duration=min_contact_duration,
        min_gap_duration=min_gap_duration,
        hysteresis=hysteresis,
        adaptive=adaptive,
        height_reference=height_reference,
        default_height_reference="floor",
    )
    return _contacts_on_clip(
        bvh,
        joint_names,
        joint_indices,
        settings,
        coords=coords,
        return_info=return_info,
        check_rest_height=False,
    )


def _resolve_contact_joints(
    bvh: Bvh, joints: Sequence[str | int] | None
) -> tuple[list[str], list[int]]:
    """Resolve a detector's joint argument into parallel node names and node indices.

    ``None`` auto-detects the feet (:func:`foot_contacts`' default). Entries are node names or node indices, negative indices counting from the end as in NumPy; the names returned for indices are the nodes' own. An empty list passes through: the detector refuses it.
    """
    if joints is None:
        joints = auto_detect_foot_joints(bvh)
        if not joints:
            raise ValueError(
                "Could not auto-detect foot joints. Please provide "
                "foot_joints explicitly (e.g. ['LeftFoot', 'RightFoot'])."
            )

    num_nodes = len(bvh.nodes)
    joint_names: list[str] = []
    joint_indices: list[int] = []
    for joint in joints:
        # bool is an int subclass; catch it before the int branch.
        if isinstance(joint, bool):
            raise TypeError(
                f"joints entries must be node names (str) or node indices (int), got bool {joint!r}"
            )
        if isinstance(joint, (int, np.integer)):
            index = int(joint)
            if not -num_nodes <= index < num_nodes:
                raise IndexError(
                    f"node index {index} out of range for a skeleton with {num_nodes} nodes"
                )
            if index < 0:
                index += num_nodes
            joint_indices.append(index)
            joint_names.append(bvh.nodes[index].name)
        elif isinstance(joint, str):
            if joint not in bvh.node_index:
                raise ValueError(f"Joint {joint!r} not found in skeleton.")
            joint_indices.append(bvh.node_index[joint])
            joint_names.append(joint)
        else:
            raise TypeError(
                "joints entries must be node names (str) or node indices "
                f"(int), got {type(joint).__name__}"
            )
    return joint_names, joint_indices


def _contacts_on_clip(
    bvh: Bvh,
    joint_names: list[str],
    joint_indices: list[int],
    settings: _contacts._ContactSettings,
    *,
    coords: npt.NDArray[np.float64] | None,
    return_info: bool,
    check_rest_height: bool,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], _contacts.ContactReport]:
    """The clip's part of :func:`foot_contacts` and :func:`ground_contacts`, around the detector on plain arrays.

    Both entry points resolve their joint arguments with :func:`_resolve_contact_joints` into the parallel ``joint_names`` / ``joint_indices`` (node-space) lists, gather their settings into a :class:`~pybvh._contacts._ContactSettings`, and delegate here. This reads from the clip what the detector takes (the positions, the rest pose, the frame time, the up axis), runs the checks whose messages name the clip, and hands the arrays to :func:`pybvh._contacts._contacts_from_positions`. ``check_rest_height`` enables the rest-pose "joints below hips" sanity check, which presumes the joints are feet.
    """
    if bvh.frame_time == 0:
        if settings.reads_velocity:
            raise ValueError(
                "frame_time is 0; cannot compute the units/second foot "
                "speed. Set bvh.frame_time to the clip's real sampling "
                "period."
            )
        if settings.min_contact_duration > 0 or settings.min_gap_duration > 0:
            raise ValueError(
                "frame_time is 0; cannot convert the duration filters "
                "(seconds) to frames. Set bvh.frame_time, or disable the "
                "filters with min_contact_duration=0.0 and "
                "min_gap_duration=0.0."
            )

    if not joint_indices:
        raise ValueError(
            "contact detection needs at least one joint; got an empty "
            "joint list. (auto_detect_foot_joints returns [] on footless "
            "rigs — pass joints explicitly.)"
        )

    if coords is None:
        coords = bvh.node_positions()

    up_idx, up_sign, _ = bvh.world_up_axis
    rest_coords = bvh._rest_pose()

    # ---- Sanity check for the height signal (feet-presuming entry only) ----
    if settings.reads_height and check_rest_height:
        rest_foot_height = rest_coords[joint_indices, up_idx].mean() * up_sign
        rest_hip_height = rest_coords[0, up_idx] * up_sign
        if rest_foot_height > rest_hip_height:
            raise ValueError(
                f"world_up={bvh.world_up!r} is inconsistent with rest-pose "
                f"geometry: feet are above hips along the declared up axis "
                f"({rest_foot_height:.3f} vs {rest_hip_height:.3f}). "
                f"`bvh.world_up` was likely auto-detected incorrectly; set "
                f"it manually with `bvh.world_up = '<axis>'`."
            )

    return _contacts._contacts_from_positions(
        coords[:, joint_indices, :],
        joint_names,
        rest_coords[joint_indices],
        rest_coords[0],
        frame_time=bvh.frame_time,
        up_idx=up_idx,
        up_sign=up_sign,
        settings=settings,
        return_info=return_info,
    )


def auto_detect_foot_joints(bvh: Bvh) -> list[str]:
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
        Returning ``[]`` — never raising — is the contract for footless
        rigs (no ``"foot"``/``"toe"`` names, or matches with no tip
        descendants that step 2's fallback also rejects).

    Notes
    -----
    This is the same detection used internally by
    :func:`foot_contacts` when ``foot_joints=None``.  Call it
    directly to preview the detection or to feed an explicit list
    back in.

    An empty result is a *report*, not an error: this function never raises on footless rigs. The contact detectors are where emptiness becomes fatal — :func:`foot_contacts` and :func:`ground_contacts` both raise ``ValueError`` when handed an empty joint list, so a silent ``[]`` cannot flow into a zero-column contact array.
    """
    up_idx, up_sign, _ = bvh.world_up_axis

    # Step 1: substring match
    matched = [
        n
        for n in bvh.nodes
        if not n.is_end_site() and any(kw in n.name.lower() for kw in ("foot", "toe"))
    ]
    if not matched:
        return []

    # Step 2: has-tip filter
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
            stacklevel=user_stacklevel(),
        )
        with_tip = matched

    # Step 3: most-distal — drop candidates whose subtree contains another candidate
    candidate_names = {n.name for n in with_tip}

    def _has_candidate_descendant(node: BvhNode) -> bool:
        for child in getattr(node, "children", None) or []:
            if child.name in candidate_names:
                return True
            if not child.is_end_site() and _has_candidate_descendant(child):
                return True
        return False

    most_distal = [n for n in with_tip if not _has_candidate_descendant(n)]

    # Step 4: stable sort — height (ascending) then name (alphabetical)
    rest_coords = bvh._rest_pose()

    # Identity-keyed: node names are not unique in general, and a name
    # lookup here would read another node's height into the sort.
    node_position = {id(node): i for i, node in enumerate(bvh.nodes)}

    def _sort_key(node: BvhNode) -> tuple[float, str]:
        idx = node_position[id(node)]
        height = float(rest_coords[idx, up_idx] * up_sign)
        return (height, node.name)

    most_distal.sort(key=_sort_key)
    return [n.name for n in most_distal]


def skeleton_size(bvh: Bvh, foot_joints: list[str] | None = None) -> float:
    """Absolute skeleton scale — mean rest-pose root-to-foot distance.

    A size proxy that scales linearly with the whole skeleton (≈ half the
    standing height for a humanoid). Only the leg chain contributes, so
    finger/spine subdivision does not affect it. This is the public name for
    the scale ``foot_contacts`` uses internally to set its thresholds; use it
    for size normalization. For the *relative* scale between two skeletons,
    see :func:`relative_scale_factor`.

    A skeleton whose size cannot be measured — auto-detection finds no
    feet, or every foot sits exactly on the root — raises rather than
    returning a substitute: any fabricated number (``1.0`` reads as a
    plausible metre-scale humanoid, ``nan`` poisons everything scaled by
    it) would be indistinguishable from, or worse than, a measurement.
    Catch the error and choose your own scale if a total function is
    needed. (``foot_contacts``' *internal* threshold scale keeps a
    private ``1.0`` fallback for degenerate rigs — a threshold must
    exist even where a measurement does not.)

    Parameters
    ----------
    bvh : Bvh
        Input skeleton.
    foot_joints : list of str, optional
        Foot joints; auto-detected from topology if None. Explicitly
        passed names must exist in the skeleton (``ValueError``
        otherwise).

    Returns
    -------
    float
        Mean rest-pose distance from the root to the foot joints
        (always ``> 0``).

    Raises
    ------
    ValueError
        If an explicitly passed foot joint name is not in the skeleton;
        if auto-detection finds no foot joints; or if all foot joints
        coincide with the root in the rest pose (no measurable size).

    Notes
    -----
    Source: gait/biomech normalization (Troje, Karg et al.).
    """
    if foot_joints is None:
        foot_joints = auto_detect_foot_joints(bvh)
        if not foot_joints:
            raise ValueError(
                "skeleton_size: no foot joints auto-detected on this "
                "skeleton, so its size cannot be measured. Pass foot_joints= "
                "explicitly (the deepest leg-chain joints), or catch this "
                "error and choose your own scale."
            )
    else:
        unknown = [n for n in foot_joints if n not in bvh.node_index]
        if unknown:
            raise ValueError(f"skeleton_size: joint names {unknown} not found in skeleton.")
    rest_coords = bvh._rest_pose()
    foot_indices = [bvh.node_index[name] for name in foot_joints]
    distance = _contacts._rest_distance_to_root(rest_coords[foot_indices], rest_coords[0])
    if distance is None:
        raise ValueError(
            "skeleton_size: every foot joint coincides with the root in the "
            "rest pose, so this skeleton has no measurable size. Pass "
            "different foot_joints=, or catch this error and choose your "
            "own scale."
        )
    return distance


def relative_scale_factor(
    reference: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    centered: bool = False,
) -> float:
    """Least-squares uniform scale matching ``target`` to ``reference``.

    The scalar ``s`` minimizing ``‖reference − s·target‖²`` over all
    coordinates — i.e. ``s = ⟨reference, target⟩ / ⟨target, target⟩`` (Troje-
    style size normalization between two skeletons or poses). Both arrays must
    share shape (e.g. two ``(N, 3)`` rest poses, or ``(F, N, 3)`` sequences).

    This is the *relative* scale between skeletons; for a single skeleton's
    absolute size, see :func:`skeleton_size`.

    By default the fit is taken **about the coordinate origin** — neither
    array is mean-centered first. The Procrustes/Umeyama convention
    centers both point sets on their centroids before fitting scale; the
    two agree only when the poses are already centered, and diverge in
    proportion to the centroid offset. That makes the origin form right
    for rest poses (root at the origin) and wrong for world-frame
    sequences with a translated root — pass ``centered=True`` for those.
    Note ``centered=True`` is the *scale-only* Procrustes fit: no
    rotation is estimated, so it is not the full Umeyama similarity
    estimate, whose scale differs once a rotation is jointly fitted.

    Parameters
    ----------
    reference : ndarray
        The pose/sequence to match.
    target : ndarray
        The pose/sequence being scaled. Same shape as ``reference``.
    centered : bool, keyword-only, optional
        If True, subtract each array's centroid — the mean over every
        axis except the last (i.e. over all points, and all frames for a
        sequence) — before fitting. Default False (fit about the origin).

    Returns
    -------
    float
        The optimal scale ``s`` (``nan`` if ``target`` is all-zero — or,
        with ``centered=True``, constant, since a centered constant array
        is all-zero).

    Notes
    -----
    Source: Troje 2002 (pose normalization); Umeyama 1991 for the
    centered convention.
    """
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape != target.shape:
        raise ValueError(
            f"reference and target must share shape, got {reference.shape} and {target.shape}"
        )
    if centered:
        point_axes = tuple(range(reference.ndim - 1))
        reference = reference - reference.mean(axis=point_axes, keepdims=True)
        target = target - target.mean(axis=point_axes, keepdims=True)
    denom = float(np.sum(target * target))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(reference * target) / denom)


def _compute_floor_height(bvh: Bvh) -> float:
    """Scene ground level in raw world coordinates.

    Backs the cached :attr:`Bvh.floor_height`: the 2nd-percentile of the
    per-frame minimum height over **all** nodes (see
    :func:`pybvh.geometry._floor_level`), signed back into the raw up axis. All
    nodes, not the feet — the ground is where the motion's geometry
    bottoms out, whichever body part gets there. See
    :attr:`Bvh.floor_height` for the convention and the alternatives it
    was chosen over.
    """
    up_idx, up_sign, _ = bvh.world_up_axis
    coords = bvh.node_positions()  # (F, N, 3), world
    return geometry._floor_level(coords, up_idx, up_sign)


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

    Applies the chosen ``stencil`` three times to the positions and pads
    or crops once, at the end (:func:`pybvh.signal.finite_difference`
    with ``order=3``), the next rung of the velocity → acceleration →
    jerk ladder. The jerk *magnitude*
    ``np.linalg.norm(node_jerk(...), axis=-1)`` is the usual smoothness
    signal.

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
    min_frames = _finite_difference_min_frames(3, stencil, pad)
    if bvh.frame_count < min_frames:
        raise ValueError(
            f"stencil={stencil!r}, pad={pad!r} requires at least "
            f"{min_frames} frames (have {bvh.frame_count})."
        )
    if not in_frames and bvh.frame_time == 0:
        raise ValueError(
            "frame_time is 0; cannot compute per-second jerk. "
            "Use in_frames=True for per-frame jerk."
        )

    if coords is None:
        coords = bvh.node_positions(centered=centered)
    dt = 1.0 if in_frames else bvh.frame_time
    return _finite_difference(coords, dt, order=3, stencil=stencil, pad=pad)


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
    Raises ``ValueError`` if ``coords`` is not node-shaped ``(F, N, 3)``.
    """
    _validate_node_coords(bvh, coords)
    nj = node_jerk(
        bvh, centered=centered, in_frames=in_frames, coords=coords, stencil=stencil, pad=pad
    )
    return nj[:, _non_end_site_indices(bvh), :]


# ----------------------------------------------------------------
#  Smoothness — array-pure kernels on a speed profile
# ----------------------------------------------------------------
#  Every kernel accepts a 1-D ``(T,)`` profile (scalar out) or a 2-D
#  ``(T, K)`` stack of profiles reduced independently per column
#  (``(K,)`` out). ``_validate_speed_profile`` rejects anything else up
#  front — silently reducing a higher-rank array globally produced
#  plausible-looking garbage.
#
#  WHAT ``speed`` MAY CONTAIN, and why it is not restricted to magnitudes.
#  Two input kinds are valid, and both appear in the source literature:
#
#    * a speed profile ``‖v‖ >= 0`` — what ``Bvh.smoothness`` feeds, from
#      ``np.linalg.norm`` of a joint velocity;
#    * a *signed* scalar velocity ``ẋ`` — Hogan & Sternad define these
#      measures on ``x(t)``, "any scalar coordinate", so its derivative
#      carries a sign. The pinned Balasubramanian reference is built for
#      it (``movement_peak = max(abs(movement))``), and one of the four
#      golden fixture signals genuinely goes negative.
#
#  So sign is not an error, and rejecting it would break the formulation
#  these kernels come from. What each kernel does with it is stated in
#  its own docstring; the rule of thumb is that *normalizers* take the
#  magnitude (``max|v|``, ``mean|v|``) while the *signal being
#  differentiated or transformed* is used as given.


def _validate_speed_profile(speed: npt.NDArray[np.float64]) -> None:
    if speed.ndim not in (1, 2):
        raise ValueError(
            f"speed must have shape (T,) — one profile — or (T, K) — K "
            f"profiles reduced per column; got shape {speed.shape}."
        )


_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_PyScalarT = TypeVar("_PyScalarT", int, float)


def _reduce_like(
    speed: npt.NDArray[np.float64],
    result: npt.NDArray[_ScalarT],
    to_scalar: Callable[[npt.NDArray[_ScalarT]], _PyScalarT],
) -> _PyScalarT | npt.NDArray[_ScalarT]:
    """Shape a reduced result to match its input: scalar out for ``(T,)``.

    The single implementation of the ``(T,) -> scalar`` /
    ``(T, K) -> (K,)`` contract every kernel in this section shares.
    ``to_scalar`` turns the 0-d result of a ``(T,)`` input into the
    Python scalar the kernel returns (``float``, or ``int`` for a count).
    """
    return to_scalar(result) if speed.ndim == 1 else result


def _sparc_from_spectrum(
    freq: npt.NDArray[np.float64],
    mag: npt.NDArray[np.float64],
    fc: float,
    amp_th: float,
) -> float:
    """SPARC of one magnitude spectrum (the original 1-D code path)."""
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
    freq_sel = freq_sel[lo : hi + 1]
    mag_sel = mag_sel[lo : hi + 1]
    if freq_sel.size < 2:
        return float("nan")  # single-point band (constant speed): no arc to trace

    arc = np.sqrt((np.diff(freq_sel) / (freq_sel[-1] - freq_sel[0])) ** 2 + np.diff(mag_sel) ** 2)
    return float(-arc.sum())


def sparc(
    speed: npt.NDArray[np.float64],
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float | npt.NDArray[np.float64]:
    """Spectral arc length (SPARC) smoothness of a speed profile.

    The negative arc length of the normalized Fourier magnitude spectrum
    over ``[0, fc]`` Hz — a smoothness measure that is robust to noise and
    invariant to amplitude/duration. Values are ``≤ 0``; closer to ``0``
    is smoother.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile, or a signed scalar velocity — the spectrum is
        taken of the values as given, not their magnitude, so the two
        are genuinely different inputs here (matching the reference
        implementation). A 2-D input is ``K`` independent profiles in
        columns, reduced per column (the FFT is batched along the time
        axis; each column's result is identical to the 1-D call on that
        column).
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
    float or ndarray
        The spectral arc length (SAL) — a scalar for ``(T,)`` input, a
        ``(K,)`` array for ``(T, K)``. ``nan`` for a zero speed profile
        (a perfectly still joint), whose spectrum carries no energy and
        whose smoothness is therefore undefined.

    Notes
    -----
    Source: Balasubramanian et al. 2015, "On the analysis of movement
    smoothness." Reimplemented in NumPy; validated against the authors'
    reference output (see ``tests/test_smoothness_golden.py``).
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    n = speed.shape[0]
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    freq = fs * np.arange(nfft) / nfft
    mag = np.abs(np.fft.fft(speed, nfft, axis=0))
    if speed.ndim == 1:
        return _sparc_from_spectrum(freq, mag, fc, amp_th)
    # the band bounds (lo, hi) are data-dependent per column, so the
    # arc-length tail stays a per-column pass over the batched spectrum
    return np.array(
        [_sparc_from_spectrum(freq, mag[:, k], fc, amp_th) for k in range(speed.shape[1])]
    )


DLJ_NORMALIZERS = ("peak_speed", "mean_speed", "amplitude")


def _dlj_scale(
    speed: npt.NDArray[np.float64],
    duration: float,
    normalize: str,
    amplitude: float | npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The DLJ normalization factor, and the extent it divides by.

    The extent is returned alongside so the caller can apply the shared
    ``extent == 0 -> nan`` guard on the quantity that actually degenerates.
    """
    if normalize not in DLJ_NORMALIZERS:
        raise ValueError(f"Unknown normalize {normalize!r}; choose from {list(DLJ_NORMALIZERS)}.")

    if normalize == "amplitude":
        if amplitude is None:
            raise ValueError(
                "normalize='amplitude' needs the movement extent A, which "
                "cannot be recovered from a speed profile alone — pass "
                "amplitude=<float or (K,) array>."
            )
        extent = np.asarray(amplitude, dtype=np.float64)
        valid_shapes: set[tuple[int, ...]] = {()} if speed.ndim == 1 else {(), (speed.shape[1],)}
        wanted = "a scalar" if speed.ndim == 1 else f"a scalar or shape ({speed.shape[1]},)"
        if extent.shape not in valid_shapes:
            raise ValueError(
                f"amplitude must be {wanted} to match speed of shape "
                f"{speed.shape}; got shape {extent.shape}."
            )
        if np.any(extent < 0):
            raise ValueError("amplitude is a movement extent and must be >= 0.")
        with np.errstate(divide="ignore", invalid="ignore"):
            return duration**5 / extent**2, extent

    # An ignored amplitude= is how a convention mismatch turns into an
    # unnoticed wrong number, so reject it instead of dropping it silently.
    if amplitude is not None:
        raise ValueError(
            f"amplitude= applies only to normalize='amplitude'; got normalize={normalize!r}."
        )
    # magnitudes: the normalizer is a scale, and a signed velocity's mean
    # would otherwise cancel toward zero on an out-and-back movement
    extent = np.abs(speed).max(axis=0) if normalize == "peak_speed" else np.abs(speed).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return duration**3 / extent**2, extent


def dimensionless_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
    normalize: str = "peak_speed",
    amplitude: float | npt.NDArray[np.float64] | None = None,
) -> float | npt.NDArray[np.float64]:
    """Dimensionless jerk (DLJ) smoothness of a speed profile.

    Integrated squared jerk made scale-invariant by dividing out movement
    duration and size — ``-(duration³ / peak²) · ∫ (d²v/dt²)² dt`` at the
    default. More negative is less smooth.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile, or a signed scalar velocity (Hogan & Sternad
        define the measure on ``x(t)``, "any scalar coordinate", whose
        derivative carries a sign). The signal is differentiated as
        given; only the normalizer takes magnitudes. A 2-D input is
        ``K`` independent profiles in columns, reduced per column.
    fs : float
        Sampling rate in Hz.
    normalize : {"peak_speed", "mean_speed", "amplitude"}, optional
        Which published normalizer to apply to the jerk integral
        (default ``"peak_speed"``). All three are dimensionless and keep
        the same negative-is-less-smooth sign; they differ in magnitude
        only. The two speed-based ones take magnitudes (``max|v|``,
        ``mean|v|``), so a signed velocity normalizes by its excursion
        rather than by a mean that cancels on an out-and-back. See the
        Notes for how to choose.
    amplitude : float or ndarray, optional
        The movement extent ``A`` — required by ``normalize="amplitude"``
        and rejected by the other two. Scalar, or shape ``(K,)`` to match
        a ``(T, K)`` speed input.

    Returns
    -------
    float or ndarray
        The dimensionless jerk (``≤ 0``) — a scalar for ``(T,)`` input,
        a ``(K,)`` array for ``(T, K)``. ``nan`` when the normalizing
        extent is zero (an all-zero speed profile, or ``amplitude=0``),
        matching the degenerate-input convention of :func:`sparc` and
        :func:`speed_metric`.

    Raises
    ------
    ValueError
        If ``normalize`` is unknown, if ``normalize="amplitude"`` is used
        without ``amplitude=``, or if ``amplitude=`` is passed with one of
        the speed-based normalizers.

    Notes
    -----
    Hogan & Sternad 2009 give the measure in three forms::

        ∫(d²v/dt²)² dt · D⁵/A²   ≡   · D³/v_mean²   |   variant: · D³/v_peak²

    The first two are the same measure — ``A = v_mean · D`` — so
    ``"mean_speed"`` is exactly ``"amplitude"`` evaluated at the arc
    length the speed profile itself implies, ``A = ∫|v| dt``. Passing
    ``amplitude=`` is only a *different* measure when the extent comes
    from somewhere else, most usefully the endpoint displacement
    ``‖p_T − p_0‖``, which is smaller than the arc length for any path
    that is not straight. (For the arc length of a trajectory, prefer
    ``"mean_speed"`` over ``amplitude=geometry.path_length(traj)``: the
    chord sum uses a different quadrature than this function's integral
    and will not agree exactly.)

    ``"peak_speed"`` — the default, and what the widely used reference
    implementation computes — is genuinely distinct from the other two.
    It differs by a factor of ``(v_mean / v_peak)²``, which depends on the
    *shape* of the speed profile rather than being constant, so a
    bell-shaped reach and a plateaued sustained movement of equal extent
    and duration rank differently under it. That factor is
    :func:`speed_metric`, so ``dimensionless_jerk(v, fs, "mean_speed")``
    equals ``dimensionless_jerk(v, fs) / speed_metric(v)**2``.

    Reproducing a published DLJ figure therefore means matching its
    convention; the default is unchanged from earlier pybvh releases.

    Source: Hogan & Sternad 2009; Balasubramanian et al. (the
    ``"peak_speed"`` form). Validated against the reference output.
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    dt = 1.0 / fs
    duration = speed.shape[0] * dt
    scale, extent = _dlj_scale(speed, duration, normalize, amplitude)
    jerk = np.diff(speed, 2, axis=0) / dt**2
    with np.errstate(invalid="ignore"):
        dlj = -scale * np.sum(jerk**2, axis=0) * dt
    # zero extent: the normalization is undefined
    dlj = np.where(extent == 0, np.nan, dlj)
    return _reduce_like(speed, dlj, float)


def log_dimensionless_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
    normalize: str = "peak_speed",
    amplitude: float | npt.NDArray[np.float64] | None = None,
) -> float | npt.NDArray[np.float64]:
    """Log dimensionless jerk (LDLJ) — ``-ln|DLJ|``.

    The log transform of :func:`dimensionless_jerk`, the form most used in
    practice. More negative is less smooth.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile; a 2-D input is ``K`` independent profiles in
        columns, reduced per column.
    fs : float
        Sampling rate in Hz.
    normalize : {"peak_speed", "mean_speed", "amplitude"}, optional
        Normalizer forwarded to :func:`dimensionless_jerk` (default
        ``"peak_speed"``). Under the log transform the choice becomes an
        additive offset of ``-ln`` of the normalizer ratio.
    amplitude : float or ndarray, optional
        Movement extent forwarded to :func:`dimensionless_jerk`; required
        by ``normalize="amplitude"`` and rejected by the other two.

    Returns
    -------
    float or ndarray
        ``-ln|DLJ|`` — a scalar for ``(T,)`` input, a ``(K,)`` array for
        ``(T, K)``. A zero-jerk (constant-speed) profile is perfectly
        smooth and returns ``+inf``.

    Notes
    -----
    Source: Balasubramanian et al. Validated against the reference output.
    """
    speed = np.asarray(speed, dtype=np.float64)
    dlj = dimensionless_jerk(speed, fs, normalize=normalize, amplitude=amplitude)
    with np.errstate(divide="ignore"):
        ldlj = np.where(
            dlj == 0,
            np.inf,  # zero jerk -> perfectly smooth
            -np.log(np.abs(dlj)),
        )
    return _reduce_like(speed, ldlj, float)


def number_of_peaks(
    speed: npt.NDArray[np.float64],
    min_height: float | None = None,
) -> int | npt.NDArray[np.int_]:
    """Number of local maxima in a speed profile.

    A simple smoothness proxy — a single smooth movement has one velocity
    peak; more peaks mean more sub-movements.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile; a 2-D input is ``K`` independent profiles in
        columns, counted per column.
    min_height : float or None, optional
        Minimum height a maximum must reach to be counted, in the same
        units as ``speed``. ``None`` (default) counts every strict local
        maximum, however small. See Notes — the default makes this
        metric sensitive to noise, and the literature says so.

    Returns
    -------
    int or ndarray
        Count of qualifying strict interior local maxima — a scalar for
        ``(T,)`` input, a ``(K,)`` array for ``(T, K)``.

    Notes
    -----
    **Strictness.** A sample counts when it is strictly greater than
    *both* neighbours. The alternative, allowing ties, would count every
    sample of a flat-topped peak separately, which is worse; the cost of
    strictness is that an exactly flat maximum — two or more equal
    adjacent samples — counts as **zero** peaks rather than one. Exact
    ties are rare in float data but reachable after quantization or
    box-filter smoothing.

    **Sign.** The comparison is on the values as given, not their
    magnitude, so for a signed velocity this counts maxima of the signed
    signal — velocity peaks in one direction — and a trough at ``-5``
    beside neighbours at ``-6`` is a maximum. That is the right reading
    for a scalar velocity and a no-op for a speed profile; use
    ``min_height`` if you want only the positive-going peaks.

    **Height threshold.** With no threshold, every micro-fluctuation is a
    "sub-movement", so on noisy data the count tracks the noise rather
    than the movement — Hogan & Sternad 2009 warn about exactly this
    (peak counting is "prone to spurious peaks", and separately blind to
    arrests). The default is unthresholded because that is what the
    metric means in its source literature, not because it is the better
    choice for real data; set ``min_height`` when the profile is noisy.
    It is absolute rather than a fraction of the peak so it composes with
    whatever normalization you have already applied — for a relative
    threshold, pass ``min_height=0.05 * speed.max()``.

    Source: Balasubramanian et al. (number-of-peaks metric); Hogan &
    Sternad 2009 (its noise sensitivity). Note the pinned SPARC reference
    implementation ships no peak-counting function, so unlike
    :func:`sparc` and :func:`dimensionless_jerk` this metric has no
    golden-reference test — only the properties asserted in
    ``tests/test_analysis_primitives.py``.
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    interior = speed[1:-1]
    is_peak = (interior > speed[:-2]) & (interior > speed[2:])
    if min_height is not None:
        is_peak &= interior >= min_height
    peaks = np.sum(is_peak, axis=0)
    return _reduce_like(speed, peaks, int)


def speed_metric(
    speed: npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Mean-to-peak speed ratio — ``mean|v| / max|v|``, in ``[0, 1]``.

    A bell-shaped (smooth) speed profile has a low ratio; a flat plateau
    approaches 1.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile, or a signed scalar velocity; a 2-D input is ``K``
        independent profiles in columns, reduced per column.

    Returns
    -------
    float or ndarray
        The mean/peak ratio — a scalar for ``(T,)`` input, a ``(K,)``
        array for ``(T, K)``. ``nan`` for an all-zero profile.

    Notes
    -----
    Both reductions take the magnitude. For a non-negative speed profile
    that is simply ``mean(v) / max(v)``; for a signed velocity it is what
    keeps the documented ``[0, 1]`` range true, since a raw mean cancels
    toward zero on an out-and-back movement and would yield a ratio at or
    below zero against a positive peak.

    The same magnitude convention is used by
    :func:`dimensionless_jerk`'s ``"mean_speed"`` normalizer, which makes
    this the exact conversion factor between its two distinct
    normalizers: ``dimensionless_jerk(v, fs, "mean_speed") ==
    dimensionless_jerk(v, fs) / speed_metric(v)**2``.

    Source: Balasubramanian et al. (speed-metric); Flash & Hogan.
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    magnitude = np.abs(speed)
    peak = magnitude.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(peak > 0, magnitude.mean(axis=0) / peak, np.nan)
    return _reduce_like(speed, ratio, float)


def integrated_squared_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> float | npt.NDArray[np.float64]:
    """Integrated squared jerk — ``∫ (d²v/dt²)² dt`` (dimensional).

    Accepts ``(T,)`` (scalar out) or ``(T, K)`` (``(K,)`` out, reduced
    per column).
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    dt = 1.0 / fs
    jerk = np.diff(speed, 2, axis=0) / dt**2
    isj = np.sum(jerk**2, axis=0) * dt
    return _reduce_like(speed, isj, float)


def mean_squared_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> float | npt.NDArray[np.float64]:
    """Mean squared jerk — ``mean((d²v/dt²)²)``.

    Accepts ``(T,)`` (scalar out) or ``(T, K)`` (``(K,)`` out, reduced
    per column).
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    dt = 1.0 / fs
    jerk = np.diff(speed, 2, axis=0) / dt**2
    msj = np.mean(jerk**2, axis=0)
    return _reduce_like(speed, msj, float)


def rms_squared_jerk(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> float | npt.NDArray[np.float64]:
    """Root-mean-square jerk — ``sqrt(mean((d²v/dt²)²))``.

    Accepts ``(T,)`` (scalar out) or ``(T, K)`` (``(K,)`` out, reduced
    per column).
    """
    speed = np.asarray(speed, dtype=np.float64)
    rms = np.sqrt(mean_squared_jerk(speed, fs))
    return _reduce_like(speed, rms, float)


# A smoothness metric's value: a scalar for a (T,) profile, one value per
# column for (T, K); number_of_peaks counts, so its array holds integers.
_SmoothnessValue = float | npt.NDArray[np.float64] | npt.NDArray[np.int_]

_SMOOTHNESS_FS_METRICS: dict[str, Callable[..., _SmoothnessValue]] = {
    "sparc": sparc,
    "dimensionless_jerk": dimensionless_jerk,
    "log_dimensionless_jerk": log_dimensionless_jerk,
    "integrated_squared_jerk": integrated_squared_jerk,
    "mean_squared_jerk": mean_squared_jerk,
    "rms_squared_jerk": rms_squared_jerk,
}
_SMOOTHNESS_PLAIN_METRICS: dict[str, Callable[..., _SmoothnessValue]] = {
    "number_of_peaks": number_of_peaks,
    "speed_metric": speed_metric,
}


def smoothness(
    speed: npt.NDArray[np.float64],
    fs: float,
    metric: str = "sparc",
    **kwargs: Any,
) -> float | npt.NDArray[np.float64] | npt.NDArray[np.int_]:
    """Dispatch to a named smoothness metric on a speed profile.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile; a 2-D input is ``K`` independent profiles in
        columns, reduced per column.
    fs : float
        Sampling rate in Hz.
    metric : str, optional
        One of ``"sparc"`` (default), ``"dimensionless_jerk"``,
        ``"log_dimensionless_jerk"``, ``"integrated_squared_jerk"``,
        ``"mean_squared_jerk"``, ``"rms_squared_jerk"``,
        ``"number_of_peaks"``, ``"speed_metric"``.
    **kwargs
        Metric-specific options: ``padlevel`` / ``fc`` / ``amp_th`` for
        ``"sparc"``; ``normalize`` / ``amplitude`` for
        ``"dimensionless_jerk"`` and ``"log_dimensionless_jerk"``;
        ``min_height`` for ``"number_of_peaks"``. The remaining metrics
        take none. See each kernel's own docstring for defaults.

    Returns
    -------
    float, int or ndarray
        The selected smoothness value — a scalar for ``(T,)`` input, a
        ``(K,)`` array for ``(T, K)``. ``"number_of_peaks"`` counts, so
        it gives an ``int`` or an integer array.

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
    "VelocityReductions", ["peak", "mean", "peak_to_mean", "peak_acceleration", "peak_deceleration"]
)


def velocity_reductions(
    speed: npt.NDArray[np.float64],
    fs: float,
) -> VelocityReductions:
    """Scalar reductions of a speed profile.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed profile; a 2-D input is ``K`` independent profiles in
        columns, reduced per column (every field of the result becomes
        a ``(K,)`` array).
    fs : float
        Sampling rate in Hz; scales ``peak_acceleration`` and
        ``peak_deceleration`` into units/second².  Required — like the
        other array-pure kernels (:func:`sparc`, :func:`smoothness`),
        the time base must be stated explicitly.

    Returns
    -------
    VelocityReductions
        Named tuple ``(peak, mean, peak_to_mean, peak_acceleration,
        peak_deceleration)`` — floats for ``(T,)`` input, ``(K,)``
        arrays for ``(T, K)``. ``peak_acceleration`` is the largest
        instantaneous rate of speed *increase* and ``peak_deceleration``
        the largest rate of speed *decrease*; both are ``>= 0`` (``0`` when
        the speed never rises / never falls).

    Notes
    -----
    All fields reduce the **values as given** — none takes a magnitude
    first. For a genuine (non-negative) speed profile that is the only
    reading; for a signed scalar velocity it means ``peak`` is the
    signed maximum (not the excursion ``max|v|``) and ``peak_to_mean``
    is unbounded (a raw mean cancels toward zero on an out-and-back
    movement, going ``nan`` at exactly zero). This differs from
    :func:`speed_metric`, whose documented ``[0, 1]`` range forces the
    magnitude convention ``mean|v| / max|v|`` — so for signed input
    ``peak_to_mean`` is *not* ``1 / speed_metric``. Take ``np.abs``
    first if you want magnitude reductions of a signed velocity.

    ``peak_acceleration`` and ``peak_deceleration`` are the positive and
    negative extrema (clamped at 0) of the per-frame speed-derivative
    series ``d‖v‖/dt``: when ``speed`` is the norm of
    :func:`node_velocities` with ``stencil="forward"``, ``pad="none"``,
    they equal the extrema of :func:`node_speed_derivative` under the
    same convention (asserted in the test suite so the two can't drift).

    Source: Pollick et al., Halovic & Kroos, Samadani et al.
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)
    peak = speed.max(axis=0)
    mean = speed.mean(axis=0)
    rate = np.diff(speed, axis=0) * fs
    if rate.shape[0]:
        peak_acceleration = np.maximum(rate.max(axis=0), 0.0)
        peak_deceleration = np.maximum(-rate.min(axis=0), 0.0)
    else:  # single-sample profile: no rate is measurable
        peak_acceleration = np.zeros_like(peak)
        peak_deceleration = np.zeros_like(peak)
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_to_mean = np.where(np.abs(mean) > _EPS, peak / mean, np.nan)
    if speed.ndim == 1:
        return VelocityReductions(
            float(peak),
            float(mean),
            float(peak_to_mean),
            float(peak_acceleration),
            float(peak_deceleration),
        )
    return VelocityReductions(peak, mean, peak_to_mean, peak_acceleration, peak_deceleration)


def zero_crossings(
    signal: npt.NDArray[np.float64],
    axis: int = 0,
) -> npt.NDArray[np.int_]:
    """Count sign changes of a signal along an axis.

    Strict crossings only — consecutive samples with a product ``< 0``;
    exact zeros are not counted as crossings. The consequence: a sign
    change that passes *through* an exact zero sample (``+1, 0, -1``)
    counts **zero** crossings, not one. The alternative convention —
    sign-change counting, ``np.diff(np.sign(x)) != 0`` — counts it
    (twice, unless zeros are carried forward). Exact zeros are rare in
    float data but reachable after quantization, rectification, or
    box-filter smoothing.

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
    fs: float,
) -> float | npt.NDArray[np.float64]:
    """Total time spent active — active sample count / ``fs``.

    Parameters
    ----------
    speed : ndarray, shape (T,) or (T, K)
        Speed signal; a 2-D input is ``K`` independent signals in
        columns, reduced per column.
    threshold : float
        Activity threshold (see :func:`active_segments`).
    fs : float
        Sampling rate in Hz. Required — the time base must be stated
        explicitly, matching the other array-pure kernels.

    Returns
    -------
    float or ndarray
        Active duration in seconds — a scalar for ``(T,)`` input, a
        ``(K,)`` array for ``(T, K)``.
    """
    speed = np.asarray(speed, dtype=np.float64)
    _validate_speed_profile(speed)  # the input, not the derived boolean mask
    duration = np.count_nonzero(active_segments(speed, threshold), axis=0) / fs
    return _reduce_like(speed, duration, float)


# ----------------------------------------------------------------
#  Kinetic energy & gait — [Bvh]
# ----------------------------------------------------------------


def _resolve_masses(
    masses: npt.NDArray[np.float64] | Mapping[str, float],
    joint_names: list[str],
) -> npt.NDArray[np.float64]:
    """Resolve a ``masses`` argument to a ``(J,)`` vector in joint-axis order.

    Accepts either a ``{joint_name: mass}`` mapping (validated for exact
    coverage — every joint once, no strangers) or an already-ordered ``(J,)``
    array (length-checked). Raises a clear ``ValueError`` otherwise, rather
    than letting a mismatch surface as a cryptic broadcast error downstream.
    Either form must sum to a positive total — an all-zero (or
    negative-total) mass vector silently zeroes every energy it multiplies.
    """
    names = list(joint_names)
    if isinstance(masses, Mapping):
        known = set(names)
        missing = [n for n in names if n not in masses]
        unknown = [k for k in masses if k not in known]
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"missing masses for {missing}")
            if unknown:
                problems.append(f"unknown joint names {unknown}")
            raise ValueError(
                f"masses dict must map every joint exactly once ({'; '.join(problems)})"
            )
        m = np.array([float(masses[n]) for n in names], dtype=np.float64)
    else:
        m = np.asarray(masses, dtype=np.float64)
        if m.shape != (len(names),):
            raise ValueError(
                f"masses must have shape ({len(names)},) in joint-axis order "
                f"(see Bvh.joint_names), got {m.shape}"
            )
    total = float(m.sum())
    # Strict > 0 with no epsilon — deliberately looser than geometry's
    # _EPS guard: masses only ever multiply (never divide), so any true
    # positive total is usable; only zero/negative/NaN totals are traps.
    if not total > 0.0:
        raise ValueError(
            f"masses must have a positive total, got sum {total!r} — a "
            f"zero/negative/NaN total silently zeroes the energy"
        )
    return m


def kinetic_energy(
    bvh: Bvh,
    masses: npt.NDArray[np.float64] | Mapping[str, float] | None = None,
    centered: str = "world",
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Per-frame kinetic energy summed over joints.

    With ``masses``, ``Σ_j ½ m_j ‖v_j‖²`` (true kinetic energy). Without,
    ``Σ_j ‖v_j‖²`` (unit-mass energy proxy) — pybvh ships no segment-mass
    model, so pass anatomical masses for physical energy. This is a
    point-mass-at-joints model; rigid-body energy (segment-CoM masses,
    rotational inertia) is not supported.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    masses : ndarray of shape (J,), or mapping {joint_name: mass}, optional
        Per-joint masses (end sites excluded). A mapping keyed by joint name
        is validated for exact coverage and is the safer form — an array
        relies on matching ``Bvh.joint_names`` order. Default None →
        unit-mass proxy.
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

    Raises
    ------
    ValueError
        If a ``masses`` mapping does not cover the joints exactly, a
        ``masses`` array has the wrong length, or the masses do not sum
        to a positive total (an all-zero mass vector would silently
        zero the energy). The unit-mass ``masses=None`` default is
        unaffected.

    Notes
    -----
    Source: Głowinski et al., Piana et al., Lu et al. 2025.
    """
    vel = joint_velocities(bvh, centered=centered, stencil=stencil, pad=pad)
    speed_sq = np.sum(vel**2, axis=-1)  # (F, J)
    if masses is None:
        return speed_sq.sum(axis=-1)
    m = _resolve_masses(masses, bvh.joint_names)
    return 0.5 * np.sum(m * speed_sq, axis=-1)


def _root_horizontal_distance(bvh: Bvh) -> float:
    """Path length of the root projected onto the ground plane."""
    up = _axis_to_vector(bvh.world_up)
    root = bvh.root_pos
    height = root @ up
    horizontal = root - height[:, None] * up
    return float(geometry.path_length(horizontal))


def _foot_contact_events(
    contacts: npt.NDArray[np.float64],
) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Per foot column, the onset and offset frame indices.

    Onsets are 0→1 transitions (touchdown), offsets 1→0 (lift-off); the
    frame index is the first frame of the new state. Returns a list over
    feet of ``(onset_frames, offset_frames)``.
    """
    planted = np.asarray(contacts) > 0.5
    events = []
    for fi in range(planted.shape[1]):
        c = planted[:, fi]
        onsets = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        offsets = np.flatnonzero(~c[1:] & c[:-1]) + 1
        events.append((onsets, offsets))
    return events


def cadence(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    *,
    contacts: npt.NDArray[np.float64] | None = None,
) -> float:
    """Step rate — foot-contact onsets per second.

    A projection of :func:`gait_parameters` (the single definition of
    every gait scalar).

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; auto-detected if None.
    contacts : ndarray, shape (F, n_feet), optional
        Pre-computed contact labels (see :func:`gait_parameters`).

    Returns
    -------
    float
        Steps per second (``nan`` if the clip has no duration — the
        rate is undefined, not zero).

    Notes
    -----
    **Unit.** Steps per *second*, consistent with every other rate in
    pybvh (Hz, units/s). The clinical gait literature — including the
    sources below — reports cadence in steps per **minute**; multiply
    by 60 to compare against published figures. Onsets are pooled over
    *all* contact columns, so passing more than two contact-bearing
    joints per foot inflates the count proportionally.

    Source: Crane & Gross, Gross et al. 2012, Karg et al. 2010.
    """
    return gait_parameters(bvh, foot_joints=foot_joints, contacts=contacts).cadence


def stride_length(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    *,
    contacts: npt.NDArray[np.float64] | None = None,
) -> float:
    """Mean stride length — distance between successive same-foot landings.

    The standard, foot-measured stride: for each foot, the horizontal
    distance between its position at consecutive contact onsets, pooled
    over feet and averaged. A projection of :func:`gait_parameters` —
    see that for the full spatiotemporal set (variability, step length,
    stance, double-support, asymmetry) computed in one pass.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; auto-detected if None.
    contacts : ndarray, shape (F, n_feet), optional
        Pre-computed contact labels (see :func:`gait_parameters`).

    Returns
    -------
    float
        Mean stride length in skeleton units (``nan`` if no foot completes
        a stride — fewer than two contacts).

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012, Karg et al. 2010.
    """
    return gait_parameters(bvh, foot_joints=foot_joints, contacts=contacts).stride_length


def walking_pace(bvh: Bvh) -> float:
    """Mean horizontal speed — root ground-path length per second.

    Same definition as the ``walking_pace`` field of
    :func:`gait_parameters`, but computed directly from the root path so
    it needs no foot joints or contact detection.  Note it only
    *approximates* ``stride_length × cadence / 2`` (exact for straight,
    steady, symmetric gait; it diverges on curved or irregular walking
    because the root path and the foot landings measure different
    things).

    Parameters
    ----------
    bvh : Bvh
        Input motion.

    Returns
    -------
    float
        Horizontal units per second (``nan`` if the clip has no
        duration — the rate is undefined, not zero).

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012.
    """
    duration = (bvh.frame_count - 1) * bvh.frame_time
    if duration <= 0:
        return float("nan")
    return _root_horizontal_distance(bvh) / duration


GaitParameters = namedtuple(
    "GaitParameters",
    [
        "cadence",
        "walking_pace",
        "stride_length",
        "stride_cv",
        "step_length",
        "stance_fraction",
        "double_support_fraction",
        "asymmetry",
    ],
)


def _compute_gait_parameters(contacts, foot_h, foot_names, frame_time, root_distance, progression):
    """Spatiotemporal gait parameters from contacts + foot ground positions.

    Pure array core (no ``Bvh``) so it is directly testable. ``foot_h`` is the
    feet's horizontal (ground-projected) positions, shape ``(F, n_feet, 3)``,
    column order matching ``foot_names`` and the ``contacts`` columns.
    ``progression`` is the net horizontal travel vector (e.g. the root's
    first→last displacement); ``step_length`` is measured along it.
    """
    n_frames = contacts.shape[0]
    duration = (n_frames - 1) * frame_time
    events = _foot_contact_events(contacts)

    total_onsets = sum(len(onsets) for onsets, _ in events)
    # nan, not 0.0: a zero-duration clip leaves the rates *undefined* —
    # "not stepping" would be a measurement — matching the other fields.
    cadence = total_onsets / duration if duration > 0 else float("nan")
    pace = root_distance / duration if duration > 0 else float("nan")

    # stride length: per foot, distance between consecutive landing positions
    per_foot_strides: dict[str, npt.NDArray[np.float64]] = {}
    pooled = []
    for fi, (onsets, _) in enumerate(events):
        if len(onsets) >= 2:
            landings = foot_h[onsets, fi, :]
            d = np.linalg.norm(np.diff(landings, axis=0), axis=1)
            per_foot_strides[foot_names[fi]] = d
            pooled.append(d)
    pooled = np.concatenate(pooled) if pooled else np.empty(0)
    stride = float(pooled.mean()) if pooled.size else float("nan")
    # stride variability: within-foot CV (so left/right *asymmetry* does not leak
    # in as "variability"). Undefined until some foot has >= 2 strides.
    resid = [d - d.mean() for d in per_foot_strides.values() if len(d) >= 2]
    if resid and abs(stride) > _EPS:
        stride_cv = float(np.concatenate(resid).std() / stride)
    else:
        stride_cv = float("nan")

    # step length: forward advance between consecutive landings (any foot),
    # projected onto the progression direction so lateral step *width* is
    # excluded. nan when there is no net travel to define a direction.
    prog_norm = float(np.linalg.norm(progression))
    if total_onsets >= 2 and prog_norm > _EPS:
        direction = np.asarray(progression, dtype=np.float64) / prog_norm
        frames = np.concatenate([onsets for onsets, _ in events])
        positions = np.concatenate(
            [foot_h[onsets, fi, :] for fi, (onsets, _) in enumerate(events)], axis=0
        )
        pts = positions[np.argsort(frames, kind="stable")]
        advances = np.abs(np.diff(pts, axis=0) @ direction)
        step = float(advances.mean())
    else:
        step = float("nan")

    # stance fraction: stance time / gait-cycle time, per cycle, meaned
    stance_fracs = []
    for onsets, offsets in events:
        for k in range(len(onsets) - 1):
            o0, o1 = onsets[k], onsets[k + 1]
            after = offsets[(offsets > o0) & (offsets <= o1)]
            if after.size:
                stance_fracs.append((after[0] - o0) / (o1 - o0))
    stance = float(np.mean(stance_fracs)) if stance_fracs else float("nan")

    # double support: fraction of frames with >= 2 feet planted
    if contacts.shape[1] >= 2:
        n_planted = (np.asarray(contacts) > 0.5).sum(axis=1)
        double_support = float((n_planted >= 2).mean())
    else:
        double_support = float("nan")

    # asymmetry: |L − R| mean stride / their average; needs one L and one R foot
    lefts = [n for n in per_foot_strides if "left" in n.lower()]
    rights = [n for n in per_foot_strides if "right" in n.lower()]
    if len(lefts) == 1 and len(rights) == 1:
        mL = per_foot_strides[lefts[0]].mean()
        mR = per_foot_strides[rights[0]].mean()
        denom = 0.5 * (mL + mR)
        asymmetry = float(abs(mL - mR) / denom) if abs(denom) > _EPS else float("nan")
    else:
        asymmetry = float("nan")

    return GaitParameters(cadence, pace, stride, stride_cv, step, stance, double_support, asymmetry)


def gait_parameters(
    bvh: Bvh,
    foot_joints: list[str] | None = None,
    *,
    contacts: npt.NDArray[np.float64] | None = None,
) -> GaitParameters:
    """Spatiotemporal gait parameters in one pass.

    Bundles the foot-measured gait analysis: ``cadence`` (onsets/s),
    ``walking_pace`` (root ground speed), ``stride_length`` and its
    coefficient of variation ``stride_cv`` (landing→next-same-foot-landing),
    ``step_length`` (forward advance between successive any-foot landings,
    measured along the direction of travel so step *width* is excluded),
    ``stance_fraction`` (mean fraction of a cycle a foot is planted),
    ``double_support_fraction`` (fraction of frames with ≥2 feet planted), and
    ``asymmetry`` (left/right stride difference). Underdetermined fields are
    ``nan`` — uniformly, across all eight (e.g. ``asymmetry`` without one
    identifiable left and right foot, ``stride_length`` if no foot completes
    two contacts, ``step_length`` if there is no net travel, ``cadence`` and
    ``walking_pace`` on a zero-duration clip).

    These are *kinematic* — computed from foot positions and contact timing
    alone. Dynamic gait analysis (joint torques, ground-reaction force,
    mechanical work) needs a physical model and is out of scope.

    Conventions worth naming, since published gait figures differ on all
    three:

    - ``stride_cv`` pools each foot's deviations from that **same foot's**
      mean before taking the standard deviation, then divides by the
      overall mean stride. The plainer alternative — the CV of the pooled
      stride sample — folds left/right asymmetry into "variability"; here
      the two are separate fields, and they diverge exactly in proportion
      to ``asymmetry``.
    - ``asymmetry`` is ``|mean_L − mean_R| / (½(mean_L + mean_R))``:
      **unsigned and unitless**. Robinson's Symmetry Index, standard in
      the gait literature, is the same ratio kept **signed** and ×100 —
      multiply by 100 for SI magnitude; the direction of the asymmetry is
      not recoverable from this field.
    - ``step_length`` projects onto a **single whole-clip progression
      chord** (the root's first→last horizontal displacement), not a
      per-step heading. On a curved walk this systematically shortens the
      value, and a closed loop has no net travel and returns ``nan``.
      ``stride_length`` by contrast is the full **Euclidean** distance
      between same-foot landings, so it *includes* lateral step width
      while ``step_length`` excludes it — the two are not the 2:1 pair a
      clinical report would use on curved or wide-stance gait.

    Parameters
    ----------
    bvh : Bvh
        Input motion.
    foot_joints : list of str, optional
        Foot joints; auto-detected if None.
    contacts : ndarray, shape (F, n_feet), optional
        Pre-computed contact labels (column order matching ``foot_joints``);
        otherwise :func:`foot_contacts` is run with ``adaptive=True`` —
        gait input is locomotion by definition, which is the documented
        precondition for the adaptive per-foot thresholds (the fixed
        defaults under-detect stance on retargeted mocap whose feet hover
        above the estimated floor).  Pass explicit ``contacts`` for full
        control over the detection.

    Returns
    -------
    GaitParameters
        Named tuple of the eight parameters above.

    Raises
    ------
    ValueError
        If no foot joints can be found.

    Notes
    -----
    Source: Crane & Gross, Gross et al. 2012, Karg et al. 2010.
    """
    if foot_joints is None:
        foot_joints = auto_detect_foot_joints(bvh)
    if not foot_joints:
        raise ValueError("gait_parameters: no foot joints found; pass foot_joints explicitly")

    node_pos = bvh.node_positions()
    if contacts is None:
        # adaptive=True: calling gait_parameters *is* the declaration that the
        # clip is locomotion — exactly the precondition foot_contacts documents
        # for its adaptive thresholds.  The fixed thresholds under-detect
        # stance on retargeted mocap whose feet hover above the floor, which
        # yields impossible gait numbers (double_support = 0 on a plain walk).
        contacts = foot_contacts(bvh, foot_joints=foot_joints, coords=node_pos, adaptive=True)
    contacts = np.asarray(contacts, dtype=np.float64)

    up = _axis_to_vector(bvh.world_up)
    foot_idx = [bvh.index(name, space="node") for name in foot_joints]
    foot_xyz = node_pos[:, foot_idx, :]  # (F, n_feet, 3)
    foot_h = foot_xyz - (foot_xyz @ up)[..., None] * up  # project to ground

    root = bvh.root_pos
    root_h = root - (root @ up)[:, None] * up  # (F, 3) on the ground
    progression = root_h[-1] - root_h[0]  # net travel direction

    return _compute_gait_parameters(
        contacts,
        foot_h,
        list(foot_joints),
        bvh.frame_time,
        _root_horizontal_distance(bvh),
        progression,
    )


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
    ``(3N, 3N)`` population covariance across frames (divides by ``F``,
    not ``F − 1``) — a fixed-size pose descriptor independent of
    sequence length.

    Parameters
    ----------
    pos : ndarray, shape (F, N, 3)
        Per-frame joint positions.

    Returns
    -------
    ndarray, shape (3N, 3N)
        The population covariance matrix.

    Notes
    -----
    Source: Hussein et al. (Cov3DJ).
    """
    pos = np.asarray(pos, dtype=np.float64)
    flat = pos.reshape(pos.shape[0], -1)  # (F, 3N)
    centered = flat - flat.mean(axis=0)
    return (centered.T @ centered) / flat.shape[0]


def lagged_covariance(
    signal: npt.NDArray[np.float64],
    lag: int,
) -> npt.NDArray[np.float64]:
    """Lagged covariance matrix — ``M(l) = (1/(T−l)) Σ_t (v_{t+l} − v̄)(v_t − v̄)ᵀ``.

    Captures temporal structure between channels at a fixed lag: the
    covariance between the signal and itself ``lag`` samples earlier,
    averaged over the ``T − l`` overlapping sample pairs (so every entry
    is a mean, independent of the lag).  The signal is centered on its
    temporal mean ``v̄`` first — a true covariance, so a constant offset
    contributes nothing.  ``lag=0`` reduces to the ordinary population
    covariance of the channels (cf. :func:`cov3dj`).

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
    centered = signal - signal.mean(axis=0)
    if lag == 0:
        ahead, behind = centered, centered
    else:
        ahead, behind = centered[lag:], centered[:-lag]
    return (ahead.T @ behind) / ahead.shape[0]
