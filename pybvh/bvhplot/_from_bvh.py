"""From a :class:`~pybvh.bvh.Bvh` to a Scene.

Input normalization, Scene construction, skeleton topology, camera
presets and bone chain classification. Together with the router, which
is handed the user's ``Bvh`` and prepares it (resampling, rest pose,
world-up checks), this is the Bvh-facing layer of bvhplot. The router
imports this module; nothing else in the package does.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._scene import UP_AXIS_INDEX, Scene, SkeletonView

if TYPE_CHECKING:
    from ..bvh import Bvh


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def as_clip_list(bvh: Bvh | list[Bvh]) -> list[Bvh]:
    """The clips an entry point was given, as a non-empty list.

    Every bvhplot entry point that accepts one ``Bvh`` or a list calls
    this first, so an empty list is rejected with one message before
    any other work.

    Raises
    ------
    ValueError
        If *bvh* is an empty list.
    """
    bvh_list = bvh if isinstance(bvh, list) else [bvh]
    if len(bvh_list) == 0:
        raise ValueError("At least one Bvh object is required.")
    return bvh_list


def normalize_input(
    bvh: Bvh | list[Bvh],
    frames: int | npt.NDArray[np.floating] | None,
    centered: str,
) -> tuple[list[Bvh], list[npt.NDArray[np.float64]]]:
    """Normalize single/list Bvh + frame spec into parallel lists.

    Parameters
    ----------
    bvh : Bvh or list[Bvh]
        One or more BVH objects to visualize.
    frames : int, ndarray, or None
        - ``None``: all frames (spatial coords for entire motion).
        - int: single frame index (NumPy semantics — negative counts
          from the end).
        - 2-D array ``(N, 3)``: single frame of spatial coordinates
          (only valid when *bvh* is a single Bvh).
        - 3-D array ``(F, N, 3)``: pre-computed spatial coordinates,
          of which one frame is kept, the first, as an int keeps one
          (only valid when *bvh* is a single Bvh).
    centered : str
        Centering mode passed to ``bvh.node_positions()``.

    Returns
    -------
    bvh_list : list[Bvh]
        Always a list (length >= 1).
    coords_list : list[ndarray]
        Parallel list of spatial coordinates, each ``(F, N, 3)``.
    """
    bvh_list = as_clip_list(bvh)

    coords_list: list[npt.NDArray[np.float64]] = []

    if frames is None:
        # All frames for each Bvh
        for b in bvh_list:
            coords = b.node_positions(centered=centered)
            if coords.ndim == 2:
                coords = coords[np.newaxis]  # (N, 3) -> (1, N, 3)
            coords_list.append(coords)

    elif isinstance(frames, int):
        # Single frame index
        for b in bvh_list:
            coords = b.node_positions(frame=frames, centered=centered)
            coords_list.append(coords[np.newaxis])  # (N, 3) -> (1, N, 3)

    elif isinstance(frames, np.ndarray):
        if len(bvh_list) != 1:
            raise ValueError(
                "Pre-computed coordinate arrays can only be passed with a "
                "single Bvh object, not a list.")
        arr = np.asarray(frames, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (N, 3) -> (1, N, 3)
        elif arr.ndim != 3:
            raise ValueError(
                f"Expected frames array with 2 or 3 dimensions, got {arr.ndim}.")
        # An array becomes one still frame, its first: the other frames
        # must not size the picture or place the floor.
        coords_list.append(arr[:1])

    else:
        raise TypeError(
            f"frames must be int, ndarray, or None, got {type(frames).__name__}.")

    return bvh_list, coords_list


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def make_scene(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    camera: str | tuple[float, float],
    labels: list[str] | None,
    *,
    canonical_floor: bool = True,
    clip_frames: int | slice | None = None,
) -> Scene:
    """Assemble a :class:`Scene` from parallel per-skeleton data.

    The Bvh -> Scene adapter. Together with :func:`normalize_input`,
    which turns a Bvh into coords, it is where bvhplot reads a
    :class:`Bvh`; backends only ever see the Scene. Computes, per
    view, the topology, camera angles, floor height
    and every skeleton fact the backends draw from (timing, node names,
    rest pose, L/R pairs, orientation, chain classification, root
    heading), so the views are complete once this returns.

    ``clip_frames`` says which frames of the clip the coords are, so
    the clip-derived, frame-indexed facts can be aligned to them. Root
    heading (the ``[sin, cos]`` columns of
    :func:`~pybvh.analysis.root_trajectory`) is the only one today. An
    ``int`` names the single clip frame the coords hold (NumPy
    semantics, negative from the end) and requires one-row coords; a
    ``slice`` selects the frames, and the result is truncated or
    last-row padded to the coords' frame count, mirroring
    :func:`align_frame_counts`, so ``slice(None)`` is the whole clip.
    The default ``None`` means the coords are not the clip's (a rest
    pose, a caller-supplied array, a transformed copy) and no clip frame
    corresponds to them; ``root_heading`` is then ``None`` rather than
    a number that would be wrong. The default is the safe one on
    purpose: a caller has to vouch for the coords' provenance to get
    frame-indexed clip facts attached to them. Coords vouched for as
    the clip's frames were posed from the skeleton's rest pose, so the
    view also states that they share its unit
    (:attr:`~._scene.SkeletonView.coords_in_rest_unit`); for the
    default the unit ratio is measured from the coords.

    Floor convention: with ``canonical_floor=True`` the floor is the
    cached :attr:`Bvh.floor_height` — the robust 2nd percentile over all
    nodes, end sites included, so the plane sits under the whole
    skeleton rather than at a joint centre. Valid for world-frame
    coords, and for ``centered="first"`` coords because first-centering
    is ground-plane-only (heights stay in world units). An outlier-low
    frame can therefore dip a node slightly below the drawn plane; the
    alternative (the true minimum) would let a single glitched frame
    sink the plane for the whole clip. With ``canonical_floor=False``
    (root-relative or caller-supplied coords) the floor is the minimum
    up-coordinate of the coords in use — the clip-wide estimate does not
    apply to coords in another frame of reference, and the true minimum
    is the safe choice for a pose whose own extent is all we have.

    This is the scene ground, not the reference ``foot_contacts``
    measures clearance against; see :attr:`Bvh.floor_height`.
    """
    from ..analysis import root_trajectory
    from ..tools import _facing_lr_pairs

    views: list[SkeletonView] = []
    for i, (b, coords) in enumerate(zip(bvh_list, coords_list)):
        azimuth, elevation, up_axis, forward_axis = _camera_angles_and_forward(
            b, coords[0], camera)
        up_idx = UP_AXIS_INDEX.get(up_axis, 2)
        up_sign = float(b.up_axis.sign)
        bones = get_skeleton_lines(b)

        if clip_frames is None:
            root_heading = None
        else:
            heading = root_trajectory(b)[:, 2:4]
            if isinstance(clip_frames, slice):
                root_heading = _align_rows(heading[clip_frames],
                                           coords.shape[0])
            else:
                root_heading = heading[clip_frames][np.newaxis]
            if root_heading.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"clip_frames={clip_frames!r} names one clip frame but "
                    f"the coords hold {coords.shape[0]} frames; pass a slice "
                    f"for multi-frame coords.")
        if canonical_floor:
            floor_height = float(b.floor_height)
        else:
            # The ground is the signed-lowest point: the coordinate
            # minimum for a positive up axis, the MAXIMUM for a
            # negative one (where larger values point downward).
            extreme = coords[..., up_idx]
            floor_height = float(extreme.min() if up_sign > 0
                                 else extreme.max())
        views.append(SkeletonView(
            coords=coords,
            bones=bones,
            label=labels[i] if labels and i < len(labels) else None,
            azimuth=azimuth,
            elevation=elevation,
            up=b.world_up,
            floor_height=floor_height,
            frame_time=float(b.frame_time),
            node_names=[node.name for node in b.nodes],
            rest_coords=b.rest_pose_positions(),
            rest_up=b.rest_up,
            lr_pairs=_facing_lr_pairs(b),
            forward_axis=forward_axis,
            bone_chains=_chain_per_bone(get_bone_chains(b), len(bones)),
            root_heading=root_heading,
            coords_in_rest_unit=clip_frames is not None,
        ))
    return Scene(views=views)


def _align_rows(
    arr: npt.NDArray[np.float64], num_frames: int,
) -> npt.NDArray[np.float64]:
    """Truncate or last-row pad ``arr`` along axis 0 to ``num_frames``.

    The same rule :func:`align_frame_counts` applies to coords, so a
    frame-indexed field follows its coords through either alignment.
    """
    if arr.shape[0] >= num_frames:
        return arr[:num_frames]
    pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _chain_per_bone(chains: dict[str, list[int]], n_bones: int) -> list[str]:
    """Invert :func:`get_bone_chains`' chain -> bones map into one name per bone.

    Bones no chain claims read as ``"spine"``, the documented fallback.
    """
    per_bone = ["spine"] * n_bones
    for chain_name, bone_indices in chains.items():
        for i in bone_indices:
            per_bone[i] = chain_name
    return per_bone


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

def get_skeleton_lines(bvh: Bvh) -> list[tuple[int, int]]:
    """Precompute (parent_node_idx, child_node_idx) pairs for bone drawing.

    Computed once per skeleton, reused every frame by all backends.

    Parameters
    ----------
    bvh : Bvh
        The BVH object containing the skeleton hierarchy.

    Returns
    -------
    lines : list of (int, int)
        Each tuple is ``(parent_index, child_index)`` into the flat
        ``nodes`` list (i.e. the same indexing used by spatial coordinates).

    Notes
    -----
    The parent-first view of :attr:`Bvh.node_edges`, so the drawn bones
    are the same topology forward kinematics posed — including on
    skeletons whose node names repeat, where a name-keyed lookup would
    draw one bone twice and omit another.
    """
    return [(parent, child) for child, parent in bvh.node_edges]


# ---------------------------------------------------------------------------
# Camera angles
# ---------------------------------------------------------------------------

def get_camera_angles(
    bvh: Bvh,
    ref_frame: npt.NDArray[np.float64],
    camera: str | tuple[float, float] = "front",
) -> tuple[float, float, str]:
    """Resolve a camera specification to (azimuth, elevation, up_axis).

    Thin view of :func:`_camera_angles_and_forward`, which also returns
    the forward axis the azimuth was derived from.

    Parameters
    ----------
    bvh : Bvh
        The BVH object (used for axis detection).
    ref_frame : ndarray of shape (N, 3)
        A reference frame of spatial coordinates for axis heuristics.
    camera : str or (float, float)
        - ``"front"`` — auto-detected front view (default).
        - ``"side"`` — 90 degrees from front.
        - ``"top"`` — bird's-eye view looking down the up axis.
        - ``(azimuth_deg, elevation_deg)`` — custom angles.

    Returns
    -------
    azimuth : float
        Azimuth angle in degrees.
    elevation : float
        Elevation angle in degrees.
    up_axis : str
        Single character: ``'x'``, ``'y'``, or ``'z'``.
    """
    azimuth, elevation, up_axis, _ = _camera_angles_and_forward(
        bvh, ref_frame, camera)
    return azimuth, elevation, up_axis


def _camera_angles_and_forward(
    bvh: Bvh,
    ref_frame: npt.NDArray[np.float64],
    camera: str | tuple[float, float],
) -> tuple[float, float, str, str]:
    """:func:`get_camera_angles` plus the signed forward axis string.

    ``forward_axis`` (e.g. ``'+y'``) is the character's facing at
    *ref_frame* snapped to the dominant signed world axis
    (:func:`~pybvh.tools._compute_forward_at`: leftward from the L/R
    joint geometry, crossed with up, then snapped). When that geometry
    is degenerate (leftward parallel to up, or no L/R pairs) it falls
    back to the rest-pose leftward and then to a fixed per-up-axis
    default, so the string is always one of the six axes, never the
    continuous direction. It is the fact the ``"front"`` preset turns
    into an azimuth; :func:`make_scene` keeps it on the view so lateral
    spacing and the camera never disagree about which way the skeleton
    faces.
    """
    from ..tools import _compute_forward_at, extract_sign

    # World up comes from the Bvh property (auto-detected with manual
    # override). Forward is computed from the given reference frame's
    # actual joint positions, so it tracks the character's orientation
    # as the animation plays (not just the rest-pose topology).
    up_ax = bvh.world_up
    forward_ax = _compute_forward_at(bvh, ref_frame, up_ax)
    up_char = up_ax[1]                   # 'y'
    up_positive = extract_sign(up_ax)
    fwd_char = forward_ax[1]
    fwd_positive = extract_sign(forward_ax)

    if isinstance(camera, tuple):
        return float(camera[0]), float(camera[1]), up_char, forward_ax

    # Compute base azimuth/elevation for the "front" view.
    # The logic: determine which matplotlib azimuth faces the skeleton's
    # forward axis, accounting for which axis is up.

    # Matplotlib's default front-facing axis given the vertical axis
    # Matplotlib's default front-facing axis at azim=0 for each vertical_axis:
    # vertical_axis='z': azim=0 looks along -x, so default front is 'x'
    # vertical_axis='y': azim=0 looks along -z, so default front is 'z'
    # vertical_axis='x': azim=0 looks along -y, so default front is 'y'
    default_up2front = {'z': 'x', 'y': 'z', 'x': 'y'}

    base_azim = -20.0
    base_elev = 20.0

    if fwd_char != default_up2front[up_char]:
        base_azim += 90.0

    if not up_positive:
        base_elev += 180.0
        base_azim += 180.0

    # A negative forward (e.g. '-y' instead of '+y') flips the camera to
    # the opposite side of the skeleton. Apply regardless of whether the
    # forward axis matches the up's default front axis.
    if not fwd_positive:
        base_azim += 180.0

    if camera == "front":
        return base_azim, base_elev, up_char, forward_ax
    elif camera == "side":
        return base_azim + 90.0, base_elev, up_char, forward_ax
    elif camera == "top":
        return base_azim, 90.0, up_char, forward_ax
    else:
        raise ValueError(
            f"Unknown camera preset {camera!r}. "
            f"Use 'front', 'side', 'top', or (azimuth, elevation).")


# ---------------------------------------------------------------------------
# Bone chain classification (for per-chain coloring)
# ---------------------------------------------------------------------------

def get_bone_chains(bvh: Bvh) -> dict[str, list[int]]:
    """Classify each drawn bone into a kinematic chain for coloring.

    Chains are ``"spine"`` (every unpaired node — torso, neck, head),
    ``"l_arm"`` / ``"r_arm"``, and ``"l_leg"`` / ``"r_leg"``. Sides come
    from :attr:`Bvh.node_lr_pairs` (the same left/right detection that
    powers ``mirror``); legs are told apart from arms by walking the
    topology from the auto-detected foot joints (paired ancestors of a
    foot, the foot itself, and everything below it), so no joint-name
    heuristics beyond what those two detectors already encode.

    Returns
    -------
    chains : dict[str, list[int]]
        Maps chain name to indices into :func:`get_skeleton_lines`'s
        bone list. Every bone index appears in exactly one chain.

    Notes
    -----
    A limb is the maximal run of *paired* nodes: a bone belongs to a
    limb chain only when both its endpoints are paired, so the
    junction bones connecting the torso to a limb (an unpaired parent
    like ``Spine3`` or ``Hips`` to a paired child like a shoulder or
    hip joint) stay in ``"spine"`` — limbs start at the shoulder ball
    and hip socket, the TEMOS/MDM capsule-figure convention. The
    alternative (a bone belongs to its child node's chain, coloring
    those connectors as limbs) diverges visibly on rigs whose shoulder
    joints sit on the spine axis: the connector is then a vertical
    segment lying *on* the spine, and limb-coloring it paints a stray
    arm-colored "vertebra" onto the torso.

    Fallbacks: with no left/right pairs (``node_lr_pairs is None``)
    every bone lands in ``"spine"`` — the caller should draw a single
    color. With pairs but no detectable feet, every paired-to-paired
    bone is classified as an arm: side coloring survives, the arm/leg
    shade distinction does not.
    """
    bones = get_skeleton_lines(bvh)
    pairs = bvh.node_lr_pairs
    if pairs is None:
        return {"spine": list(range(len(bones)))}

    left_nodes = {left for left, _ in pairs}
    right_nodes = {right for _, right in pairs}

    parent_of = {child: parent for child, parent in bvh.node_edges}
    children_of: dict[int, list[int]] = {}
    for child, parent in bvh.node_edges:
        children_of.setdefault(parent, []).append(child)

    # Leg nodes: from each detected foot, everything below it plus the
    # paired ancestors above it (stopping at the first unpaired node,
    # i.e. where the leg meets the torso).
    leg_nodes: set[int] = set()
    for foot_name in bvh.auto_detect_foot_joints():
        foot_idx = bvh.node_index.get(foot_name)
        if foot_idx is None:
            continue
        stack = [foot_idx]
        while stack:
            n = stack.pop()
            leg_nodes.add(n)
            stack.extend(children_of.get(n, []))
        n2 = parent_of.get(foot_idx)
        while n2 is not None and (n2 in left_nodes or n2 in right_nodes):
            leg_nodes.add(n2)
            n2 = parent_of.get(n2)

    chains: dict[str, list[int]] = {
        "spine": [], "l_arm": [], "l_leg": [], "r_arm": [], "r_leg": []}
    paired = left_nodes | right_nodes
    for i, (parent, child) in enumerate(bones):
        if child in left_nodes:
            side = "l"
        elif child in right_nodes:
            side = "r"
        else:
            chains["spine"].append(i)
            continue
        if parent not in paired:
            # Junction bone (torso -> limb): stays spine-colored, the
            # limb starts at the first fully-paired bone (see Notes).
            chains["spine"].append(i)
            continue
        limb = "leg" if child in leg_nodes else "arm"
        chains[f"{side}_{limb}"].append(i)
    return {name: idxs for name, idxs in chains.items() if idxs}
