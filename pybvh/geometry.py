"""Position descriptors for BVH motion — points in R³.

The position half of pybvh's geometry surface (the orientation half lives in
:mod:`pybvh.rotations`). Every function here is **array-pure**: it takes plain
NumPy point arrays and returns NumPy arrays, with no :class:`~pybvh.bvh.Bvh`
dependency, so downstream libraries can build on these kernels directly.

Two shape conventions run through the module:

* **Point-set** kernels (``bounding_box``, ``bounding_sphere``,
  ``bounding_ellipsoid``, ``center_of_mass``, ``verticality``) take ``pts``
  shaped ``(..., P, 3)`` and reduce over the point axis ``P``, keeping any
  leading batch axes (e.g. a frame axis ``F``) — so they vectorize over time
  with no Python frame loop.
* **Trajectory** kernels (``path_length``, ``directness``, ``curvature``,
  ``torsion``, ``movement_phase``, ``ground_path``) take ``traj`` shaped
  ``(F, 3)`` or ``(F, N, 3)`` — the first axis ``F`` is time.

Derivatives (``curvature``, ``torsion``, ``movement_phase``) route through
:func:`pybvh.signal.finite_difference`, the same convention used by the
kinematics ladder, so geometry and velocity derivatives stay consistent.

**Zero-denominator policy.** Every ratio kernel (``curvature``,
``directness``, ``verticality``) returns ``np.nan`` at samples where its
denominator vanishes (a stationary joint, a perfectly vertical pose). ``nan``
is used deliberately over ``0.0`` so an *undefined* value is never confused
with a genuine zero (e.g. the real zero curvature of a straight segment).
The ``nan`` policy covers *data* degeneracy — values the motion itself made undefined. Invalid *arguments* (e.g. a ``weights`` vector with no positive total in ``center_of_mass``) are caller mistakes and raise ``ValueError`` instead of silently propagating ``nan``.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import numpy.typing as npt

from .signal import finite_difference

_EPS = 1e-12


def _safe_ratio(
    numerator: npt.NDArray[np.float64],
    denominator: npt.NDArray[np.float64],
    fill: float = np.nan,
) -> npt.NDArray[np.float64]:
    """``numerator / denominator``, returning ``fill`` where ``denominator``
    is ~0. The division is computed against a guarded denominator (never
    zero), so no divide-by-zero warning is ever raised — the zero-denominator
    policy is enforced before the divide, not patched up after it."""
    valid = denominator > _EPS
    guarded = np.where(valid, denominator, 1.0)
    return np.where(valid, numerator / guarded, fill)


BoundingBox = namedtuple("BoundingBox", ["min", "max", "extent", "volume"])
BoundingSphere = namedtuple("BoundingSphere", ["center", "radius"])
BoundingEllipsoid = namedtuple("BoundingEllipsoid", ["center", "radii", "axes"])
GroundPath = namedtuple("GroundPath", ["distance", "area"])


# ----------------------------------------------------------------
#  Inter-point relations
# ----------------------------------------------------------------

def inter_joint_distance(
    pos: npt.NDArray[np.float64],
    pairs: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Euclidean distance between pairs of points — ``‖p_a − p_b‖``.

    Parameters
    ----------
    pos : ndarray, shape (..., P, 3)
        Point positions (e.g. ``node_positions`` output ``(F, N, 3)``).
    pairs : array_like, shape (Q, 2)
        Integer index pairs into the point axis ``P``.

    Returns
    -------
    ndarray, shape (..., Q)
        Distance for each pair, vectorized over the leading axes.
    """
    pos = np.asarray(pos, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.intp)
    a = pos[..., pairs[:, 0], :]
    b = pos[..., pairs[:, 1], :]
    return np.linalg.norm(a - b, axis=-1)


def joint_angle(
    a: npt.NDArray[np.float64],
    vertex: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Angle at ``vertex`` in the triangle ``a–vertex–b``.

    Uses the numerically stable form ``atan2(‖u×v‖, u·v)`` with
    ``u = a − vertex``, ``v = b − vertex`` — accurate across the whole
    ``[0, π]`` range (unlike ``arccos`` of a normalized dot, which loses
    precision near 0 and π). Symmetric: ``joint_angle(a, v, b)`` equals
    ``joint_angle(b, v, a)``.

    Parameters
    ----------
    a, vertex, b : ndarray, shape (..., 3)
        The two outer points and the shared vertex.
    degrees : bool, optional
        Return degrees instead of radians (default radians).

    Returns
    -------
    ndarray, shape (...)
        The angle at ``vertex``.

    Notes
    -----
    Source: ubiquitous; see Saha et al., Crenn et al. 2016, Basak et al.
    """
    u = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v = np.asarray(b, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    ang = np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.sum(u * v, axis=-1))
    return np.degrees(ang) if degrees else ang


def segment_axis_angle(
    seg: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    """Angle between a segment vector and a reference axis, in ``[0, π]``.

    ``atan2(‖seg×axis‖, seg·axis)`` — e.g. the inclination of a bone
    relative to ``world_up``.

    Parameters
    ----------
    seg : ndarray, shape (..., 3)
        Segment / bone direction vectors (need not be unit length).
    axis : ndarray, shape (3,) or (..., 3)
        Reference axis (need not be unit length).
    degrees : bool, optional
        Return degrees instead of radians (default radians).

    Returns
    -------
    ndarray, shape (...)
        The angle between ``seg`` and ``axis``.

    Notes
    -----
    Source: Barliya et al., Gross et al., Truong et al.
    """
    seg = np.asarray(seg, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    ang = np.arctan2(
        np.linalg.norm(np.cross(seg, axis), axis=-1), np.sum(seg * axis, axis=-1))
    return np.degrees(ang) if degrees else ang


def triangle_area(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Area of triangle ``(a, b, c)`` — ``½‖(b−a)×(c−a)‖``.

    Parameters
    ----------
    a, b, c : ndarray, shape (..., 3)
        Triangle vertices.

    Returns
    -------
    ndarray, shape (...)
        Triangle area, vectorized over the leading axes.

    Notes
    -----
    Source: Bhattacharya et al. (walk descriptors), Crenn et al. 2016.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


def point_to_plane_distance(
    point: npt.NDArray[np.float64],
    plane_point: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    signed: bool = True,
) -> npt.NDArray[np.float64]:
    """Distance from ``point`` to the plane through ``plane_point``.

    ``(point − plane_point) · n̂``, where ``n̂`` is the unit ``normal``.

    Parameters
    ----------
    point, plane_point : ndarray, shape (..., 3)
        Query point(s) and a point on the plane.
    normal : ndarray, shape (..., 3)
        Plane normal (need not be unit length).
    signed : bool, optional
        If True (default), the sign encodes which side of the plane the
        point is on; if False, return the absolute distance.

    Returns
    -------
    ndarray, shape (...)
        Signed (or absolute) distance.

    Notes
    -----
    Source: Müller et al. (motion templates), Kapadia et al.
    """
    point = np.asarray(point, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    d = np.sum((point - plane_point) * n, axis=-1)
    return d if signed else np.abs(d)


def point_to_segment_distance(
    point: npt.NDArray[np.float64],
    seg_a: npt.NDArray[np.float64],
    seg_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Shortest distance from ``point`` to the segment ``[seg_a, seg_b]``.

    The projection parameter is clamped to ``[0, 1]`` so the nearest
    point is on the segment, not its infinite line. A degenerate segment
    (``seg_a == seg_b``) reduces to the point-to-point distance.

    Parameters
    ----------
    point, seg_a, seg_b : ndarray, shape (..., 3)
        Query point and the two segment endpoints.

    Returns
    -------
    ndarray, shape (...)
        Distance to the segment.

    Notes
    -----
    Source: Müller et al., Kapadia et al.
    """
    point = np.asarray(point, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64)
    seg_b = np.asarray(seg_b, dtype=np.float64)
    ab = seg_b - seg_a
    ab_sq = np.sum(ab * ab, axis=-1)
    t = _safe_ratio(np.sum((point - seg_a) * ab, axis=-1), ab_sq, fill=0.0)
    t = np.clip(t, 0.0, 1.0)
    nearest = seg_a + t[..., None] * ab
    return np.linalg.norm(point - nearest, axis=-1)


# ----------------------------------------------------------------
#  Bounding volumes & center of mass
# ----------------------------------------------------------------

def bounding_box(pts: npt.NDArray[np.float64]) -> BoundingBox:
    """Axis-aligned bounding box of a point set.

    Parameters
    ----------
    pts : ndarray, shape (..., P, 3)
        Points; reduced over the point axis ``P``.

    Returns
    -------
    BoundingBox
        Named tuple ``(min, max, extent, volume)`` — ``min``/``max``/
        ``extent`` shaped ``(..., 3)``, ``volume`` shaped ``(...)``.
        Vectorizes over the leading axes (no per-frame loop).

    Notes
    -----
    Source: ubiquitous (gesture/gait bounding-region descriptors).
    """
    pts = np.asarray(pts, dtype=np.float64)
    lower = pts.min(axis=-2)
    upper = pts.max(axis=-2)
    extent = upper - lower
    volume = np.prod(extent, axis=-1)
    return BoundingBox(lower, upper, extent, volume)


def bounding_sphere(pts: npt.NDArray[np.float64]) -> BoundingSphere:
    """Approximate enclosing sphere via Ritter's two-pass heuristic.

    Pass 1 finds a near-diameter pair (farthest point from an arbitrary
    seed, then farthest from that) to seat the centre; pass 2 grows the
    radius to the maximum distance from that centre, guaranteeing all
    points are enclosed. The result is **approximate** (not the minimal
    enclosing sphere) but fully vectorized over the leading axes — exact
    Welzl is recursive/randomized and would force a Python per-frame
    loop, which the library avoids.

    Parameters
    ----------
    pts : ndarray, shape (..., P, 3)
        Points; reduced over the point axis ``P``.

    Returns
    -------
    BoundingSphere
        Named tuple ``(center, radius)`` — ``center`` shaped ``(..., 3)``,
        ``radius`` shaped ``(...)``.

    Notes
    -----
    Source: Ritter (1990); Larboulette & Gibet, Noroozi et al.
    """
    pts = np.asarray(pts, dtype=np.float64)
    seed = pts[..., 0, :]
    far1 = _farthest_point(pts, seed)
    far2 = _farthest_point(pts, far1)
    center = 0.5 * (far1 + far2)
    radius = np.linalg.norm(pts - center[..., None, :], axis=-1).max(axis=-1)
    return BoundingSphere(center, radius)


def _farthest_point(
    pts: npt.NDArray[np.float64],
    query: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The point in ``pts`` (over axis ``-2``) farthest from ``query``."""
    d = np.linalg.norm(pts - query[..., None, :], axis=-1)  # (..., P)
    idx = np.argmax(d, axis=-1)  # (...)
    return np.take_along_axis(pts, idx[..., None, None], axis=-2)[..., 0, :]


def bounding_ellipsoid(pts: npt.NDArray[np.float64]) -> BoundingEllipsoid:
    """PCA-aligned bounding ellipsoid of a point set.

    The principal axes are the eigenvectors of the point covariance
    (via batched :func:`numpy.linalg.eigh`). The semi-axis radii start
    from the maximum absolute projection of the centred points onto each
    axis and are then grown by one shared factor — the worst point's
    ellipsoidal norm — so **every point satisfies**
    ``Σ_k (x_k / r_k)² ≤ 1``: the per-axis maxima alone only bound the
    points' box, and a point projecting strongly onto two axes at once
    would sit outside that inscribed ellipsoid. Approximate (not the
    minimal-volume Löwner–John ellipsoid), but vectorized over the
    leading axes.

    Parameters
    ----------
    pts : ndarray, shape (..., P, 3)
        Points; reduced over the point axis ``P``.

    Returns
    -------
    BoundingEllipsoid
        Named tuple ``(center, radii, axes)`` — ``center`` ``(..., 3)``,
        ``radii`` ``(..., 3)`` (semi-axis lengths, ascending eigenvalue
        order), ``axes`` ``(..., 3, 3)`` (principal directions as
        columns).

    Notes
    -----
    Source: Larboulette & Gibet (motion descriptors).
    """
    pts = np.asarray(pts, dtype=np.float64)
    p = pts.shape[-2]
    center = pts.mean(axis=-2)
    centered = pts - center[..., None, :]
    cov = np.einsum("...pi,...pj->...ij", centered, centered) / p
    _evals, evecs = np.linalg.eigh(cov)
    coords = np.einsum("...pi,...ij->...pj", centered, evecs)
    radii = np.abs(coords).max(axis=-2)
    # Per-axis max projections fit the points' box, not the ellipsoid: a
    # point near a box corner has Σ(x_k/r_k)² up to 3. Grow all radii by
    # the worst point's ellipsoidal norm so the ellipsoid encloses every
    # point while keeping the PCA orientation and aspect ratio. A ~zero
    # radius means zero extent on that axis (its projections are all ~0),
    # so the guarded ratio contributes nothing there.
    guarded = np.where(radii > _EPS, radii, 1.0)
    norm_sq = np.sum((coords / guarded[..., None, :]) ** 2, axis=-1)
    radii = radii * np.sqrt(norm_sq.max(axis=-1))[..., None]
    return BoundingEllipsoid(center, radii, evecs)


def center_of_mass(
    pts: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Centre of mass of a point set — ``Σ wₖ pₖ / Σ wₖ``.

    Parameters
    ----------
    pts : ndarray, shape (..., P, 3)
        Points; reduced over the point axis ``P``.
    weights : ndarray, shape (P,), optional
        Per-point weights. Default is uniform (the plain centroid) —
        pybvh ships no body-segment mass model; pass anatomical masses
        explicitly for a true centre of mass.

    Returns
    -------
    ndarray, shape (..., 3)
        The (weighted) centre of mass.

    Raises
    ------
    ValueError
        If ``weights`` has no positive total (zero, sub-epsilon, negative, or NaN sum) — the weighted mean would be all-NaN (or sign-flipped) for every frame. Individual negative weights are allowed as long as the total stays positive.

    Notes
    -----
    Source: Larboulette & Gibet, Kapadia et al., Piana et al.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if weights is None:
        return pts.mean(axis=-2)
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    # One predicate rejects zero, sub-epsilon, negative, AND NaN totals
    # (NaN comparisons are False). This is an argument error, not data
    # degeneracy, so it raises instead of following the module's nan policy.
    if not total > _EPS:
        raise ValueError(
            f"center_of_mass weights must have a positive total, got sum "
            f"{total!r} — a zero/negative/NaN total makes the weighted "
            f"mean undefined for every frame")
    return np.sum(pts * w[:, None], axis=-2) / total


def com_displacement(
    com: npt.NDArray[np.float64],
    com_ref: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Distance of a centre of mass from a reference — ``‖com − com_ref‖``.

    Parameters
    ----------
    com : ndarray, shape (..., 3)
        Centre-of-mass position(s) (e.g. per-frame, ``(F, 3)``).
    com_ref : ndarray, shape (3,) or (..., 3)
        Reference centre of mass (e.g. the first-frame or mean CoM). Must be
        in the same coordinate frame as ``com``.

    Returns
    -------
    ndarray, shape (...)
        Displacement magnitude.

    Notes
    -----
    Source: Larboulette & Gibet, Kapadia et al.
    """
    com = np.asarray(com, dtype=np.float64)
    com_ref = np.asarray(com_ref, dtype=np.float64)
    return np.linalg.norm(com - com_ref, axis=-1)


def verticality(
    pts: npt.NDArray[np.float64],
    up: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Height-to-width ratio of a point set along ``up``.

    The vertical extent (spread along ``up``) divided by the horizontal
    extent (the diagonal of the bounding box in the plane orthogonal to
    ``up``). ``> 1`` is a tall/upright posture, ``< 1`` a wide/crouched
    one. Returns ``np.nan`` when the horizontal extent is ~0 (a perfectly
    vertical configuration).

    Parameters
    ----------
    pts : ndarray, shape (..., P, 3)
        Points; reduced over the point axis ``P``.
    up : ndarray, shape (3,)
        Up axis (need not be unit length).

    Returns
    -------
    ndarray, shape (...)
        The height/width ratio.

    Notes
    -----
    Source: Larboulette & Gibet.
    """
    pts = np.asarray(pts, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    height_coord = np.sum(pts * up, axis=-1)  # (..., P)
    height = height_coord.max(axis=-1) - height_coord.min(axis=-1)
    horizontal = pts - height_coord[..., None] * up
    width = np.linalg.norm(
        horizontal.max(axis=-2) - horizontal.min(axis=-2), axis=-1)
    return _safe_ratio(height, width)


# ----------------------------------------------------------------
#  Trajectory descriptors
# ----------------------------------------------------------------

def path_length(traj: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Arc length travelled — ``Σ ‖p_{t+1} − p_t‖`` over the frame axis.

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.

    Returns
    -------
    ndarray
        Scalar for ``(F, 3)``; shape ``(N,)`` for ``(F, N, 3)``.

    Notes
    -----
    Source: ubiquitous (trajectory / effort descriptors).
    """
    traj = np.asarray(traj, dtype=np.float64)
    return np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum(axis=0)


def directness(traj: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Directness — ``‖p_T − p_0‖ / path_length``, in ``[0, 1]``.

    The net start→end displacement as a fraction of the total distance
    travelled: ``1`` for a path straight to its destination, approaching
    ``0`` for one that nets little progress — note an out-and-back returns
    ``0`` (zero net displacement), since this measures *directness of
    travel*, not per-segment straightness. Returns ``np.nan`` for a
    stationary trajectory (zero path length).

    Also known as the straightness index (Camurri's "Directness Index").

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.

    Returns
    -------
    ndarray
        Scalar for ``(F, 3)``; shape ``(N,)`` for ``(F, N, 3)``.

    Notes
    -----
    Source: Camurri et al., Samadani et al., Ajili et al.
    """
    traj = np.asarray(traj, dtype=np.float64)
    length = path_length(traj)
    displacement = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    return _safe_ratio(displacement, length)


def _aligned_derivatives(
    traj: npt.NDArray[np.float64],
    frame_time: float,
    order: int,
    stencil: str,
    pad: str,
) -> list[npt.NDArray[np.float64]]:
    """Successive derivatives ``[ṗ, …]`` up to ``order``, frame-aligned.

    Each is differenced at ``pad="edge"`` (full length, so all orders
    share a frame axis); for ``pad="none"`` the boundary frames the
    repeated stencil cannot cleanly define are then trimmed equally from
    every order so they stay aligned — ``order`` frames from each end for
    a central stencil, ``order`` from the tail for forward.
    """
    # Validate up front: every internal finite_difference call below uses
    # pad="edge", so a typo'd `pad` would otherwise silently mean "edge".
    if stencil not in ("central", "forward"):
        raise ValueError(
            f"stencil must be 'central' or 'forward', got {stencil!r}")
    if pad not in ("edge", "none"):
        raise ValueError(f"pad must be 'edge' or 'none', got {pad!r}")

    derivs: list[npt.NDArray[np.float64]] = []
    current = np.asarray(traj, dtype=np.float64)
    for _ in range(order):
        current = finite_difference(
            current, frame_time, stencil=stencil, pad="edge", axis=0)
        derivs.append(current)
    if pad == "none":
        if stencil == "central":
            derivs = [d[order:-order] for d in derivs]
        else:  # forward
            derivs = [d[:-order] for d in derivs]
    return derivs


def _speed_and_swept(
    traj: npt.NDArray[np.float64],
    frame_time: float,
    stencil: str,
    pad: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Speed ``‖ṗ‖`` and swept magnitude ``‖ṗ × p̈‖`` per frame.

    The two ingredients shared by :func:`curvature`
    (``swept / speed³``) and :func:`movement_phase` (``swept / speed²``).
    """
    d1, d2 = _aligned_derivatives(traj, frame_time, 2, stencil, pad)
    speed = np.linalg.norm(d1, axis=-1)
    swept = np.linalg.norm(np.cross(d1, d2), axis=-1)
    return speed, swept


def curvature(
    traj: npt.NDArray[np.float64],
    frame_time: float,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Trajectory curvature ``κ = ‖ṗ × p̈‖ / ‖ṗ‖³`` per frame.

    The radius of curvature is ``1 / κ``. Returns ``np.nan`` where the
    speed ``‖ṗ‖`` is ~0 (a momentarily stationary joint, where curvature
    is undefined). Note this is distinct from the genuine ``κ = 0`` of a
    straight segment.

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.
    frame_time : float
        Seconds between frames.
    stencil, pad : optional
        Finite-difference convention, shared with the kinematics ladder
        (see :func:`pybvh.signal.finite_difference`). Default
        ``"central"`` / ``"edge"`` keeps the output length ``F``.

    Returns
    -------
    ndarray
        Curvature per frame: ``(F,)`` for ``(F, 3)`` input, ``(F, N)``
        for ``(F, N, 3)`` (trimmed along the frame axis when
        ``pad="none"``).

    Notes
    -----
    Source: Larboulette & Gibet, Gibet et al.
    """
    speed, swept = _speed_and_swept(traj, frame_time, stencil, pad)
    return _safe_ratio(swept, speed ** 3)


def torsion(
    traj: npt.NDArray[np.float64],
    frame_time: float,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Trajectory torsion ``τ = (ṗ × p̈) · p⃛ / ‖ṗ × p̈‖²`` per frame.

    Torsion measures how sharply the trajectory twists out of its
    instantaneous plane; it is ~0 for a planar curve. Returns ``np.nan``
    where ``‖ṗ × p̈‖`` is ~0 (straight or stationary, where torsion is
    undefined).

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.
    frame_time : float
        Seconds between frames.
    stencil, pad : optional
        Finite-difference convention (see :func:`curvature`).

    Returns
    -------
    ndarray
        Torsion per frame (shape as in :func:`curvature`).

    Notes
    -----
    Source: Bouchard & Badler, Zhao & Badler.
    """
    d1, d2, d3 = _aligned_derivatives(traj, frame_time, 3, stencil, pad)
    swept = np.cross(d1, d2)
    twist = np.sum(swept * d3, axis=-1)
    swept_sq = np.sum(swept * swept, axis=-1)
    return _safe_ratio(twist, swept_sq)


def movement_phase(
    traj: npt.NDArray[np.float64],
    frame_time: float,
    stencil: str = "central",
    pad: str = "edge",
) -> npt.NDArray[np.float64]:
    """Movement-phase signal ``speed · curvature = ‖ṗ × p̈‖ / ‖ṗ‖²`` per frame.

    Peaks mark the fast, sharply-turning instants that segment a
    trajectory into ballistic phases. ``np.nan`` where speed is ~0
    (curvature is undefined there), matching :func:`curvature`.

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.
    frame_time : float
        Seconds between frames.
    stencil, pad : optional
        Finite-difference convention (see :func:`curvature`).

    Returns
    -------
    ndarray
        The ``speed · curvature`` signal (shape as in :func:`curvature`).

    Notes
    -----
    Source: Larboulette & Gibet, Gibet et al.
    """
    speed, swept = _speed_and_swept(traj, frame_time, stencil, pad)
    return _safe_ratio(swept, speed ** 2)


def ground_path(
    traj: npt.NDArray[np.float64],
    up: npt.NDArray[np.float64],
) -> GroundPath:
    """Trajectory projected onto the ground plane (orthogonal to ``up``).

    Returns the projected path length and the signed-area magnitude of
    the projected polygon (via the shoelace formula — not a convex hull),
    a compact measure of how much ground a joint sweeps over.

    Parameters
    ----------
    traj : ndarray, shape (F, 3) or (F, N, 3)
        Trajectory; the first axis is time.
    up : ndarray, shape (3,)
        Up axis (need not be unit length).

    Returns
    -------
    GroundPath
        Named tuple ``(distance, area)`` — scalars for ``(F, 3)`` input,
        shape ``(N,)`` for ``(F, N, 3)``.

    Notes
    -----
    Source: Aristidou et al., Larboulette & Gibet.
    """
    traj = np.asarray(traj, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    height = np.sum(traj * up, axis=-1)
    ground = traj - height[..., None] * up

    reference = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = np.cross(up, reference)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    x = np.sum(ground * e1, axis=-1)  # (F, ...)
    y = np.sum(ground * e2, axis=-1)
    shoelace = x * np.roll(y, -1, axis=0) - np.roll(x, -1, axis=0) * y
    area = 0.5 * np.abs(shoelace.sum(axis=0))
    return GroundPath(path_length(ground), area)


# ----------------------------------------------------------------
#  Pose-level operations
# ----------------------------------------------------------------

def pose_distance(
    pose_a: npt.NDArray[np.float64],
    pose_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Euclidean distance between two poses — ``‖X₁ − X₂‖``.

    The root of the summed squared differences over the joint and
    coordinate axes (a pose-similarity kernel for nearest-neighbour /
    alignment work). A true metric — square it if a squared-distance
    kernel is wanted.

    Parameters
    ----------
    pose_a, pose_b : ndarray, shape (..., N, 3)
        Poses (e.g. ``(N, 3)`` single poses, or ``(F, N, 3)`` sequences).

    Returns
    -------
    ndarray, shape (...)
        Euclidean distance, reduced over the trailing ``(N, 3)`` axes.

    Notes
    -----
    Source: trajectory-basis pose models (Torresani-era).
    """
    diff = np.asarray(pose_a, dtype=np.float64) - np.asarray(pose_b, dtype=np.float64)
    return np.sqrt(np.sum(diff * diff, axis=(-2, -1)))


def mean_pose_subtract(seq: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Centre a sequence on its mean pose — ``p − mean_t p``.

    Removes the per-joint temporal mean, leaving only motion about the
    average posture.

    Parameters
    ----------
    seq : ndarray, shape (F, N, 3)
        Pose sequence; the first axis is time.

    Returns
    -------
    ndarray, shape (F, N, 3)
        The mean-subtracted sequence.

    Notes
    -----
    Source: frame-operation primitive (PCA / trajectory-basis prep).
    """
    seq = np.asarray(seq, dtype=np.float64)
    return seq - seq.mean(axis=0, keepdims=True)
