"""Shared helpers for all visualization backends.

Pure-data operations: skeleton topology, bounding boxes, camera math,
orthographic projection, and the :class:`Scene` container every backend
consumes. No plotting library imports.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bvh import Bvh

from ._scene import Scene, SkeletonView, UP_AXIS_INDEX, compute_unified_limits


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
    view, the topology, cubic bounding box, camera angles, floor height
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
    frame-indexed clip facts attached to them.

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
        center, half_span = compute_unified_limits([coords])
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
            center=center,
            half_span=half_span,
            azimuth=azimuth,
            elevation=elevation,
            up_axis=up_axis,
            floor_height=floor_height,
            frame_time=float(b.frame_time),
            node_names=[node.name for node in b.nodes],
            rest_coords=b.rest_pose_positions(),
            lr_pairs=_facing_lr_pairs(b),
            up_vector=np.asarray(b.up_axis.vector, dtype=np.float64),
            forward_axis=forward_axis,
            bone_chains=_chain_per_bone(get_bone_chains(b), len(bones)),
            root_heading=root_heading,
            up_sign=up_sign,
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
# Input normalization
# ---------------------------------------------------------------------------

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
        - 3-D array ``(F, N, 3)``: pre-computed spatial coordinates
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
    # Wrap single Bvh
    if not isinstance(bvh, list):
        bvh_list = [bvh]
    else:
        bvh_list = bvh

    if len(bvh_list) == 0:
        raise ValueError("At least one Bvh object is required.")

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
        coords_list.append(arr)

    else:
        raise TypeError(
            f"frames must be int, ndarray, or None, got {type(frames).__name__}.")

    return bvh_list, coords_list


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
# Ghost trails and trajectory traces (shared backend conventions)
# ---------------------------------------------------------------------------

# The trace and ghost look must be identical across backends: one
# render() call, different sinks. These are the single definitions.
TRACE_COLOR = "#7A8090"
TRACE_BLEND = 0.9          # blended toward the background at this weight
GHOST_WIDTH_FACTOR = 0.75  # ghosts draw thinner than the live skeleton


def ghost_schedule(
    style: Style,
    frame_time: float,
    n_ghosts: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    """Frame lag and fade weights for a ghost trail.

    Ghost slot ``j`` trails the live pose by ``(j+1) * lag`` frames;
    weights fade from 0.32 (nearest, darkest) to 0.15 (oldest).
    """
    lag = max(1, round(style.ghost_spacing / frame_time))
    weights = np.linspace(0.32, 0.15, n_ghosts)
    return lag, weights


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# RGB is the canonical channel order everywhere in bvhplot; the OpenCV
# backend converts to BGR at its own border (its channel-order quirk
# stays its own concern).

# Per-skeleton comparison palette (multi-skeleton figures)
PALETTE_RGB = [
    (50, 120, 255),   # blue
    (220, 50, 50),    # red
    (50, 180, 50),    # green
    (50, 130, 200),   # teal
    (200, 100, 50),   # orange
    (200, 50, 200),   # magenta
]
PALETTE_MPL = [(r / 255, g / 255, b / 255) for (r, g, b) in PALETTE_RGB]

# Per-chain palette (single-skeleton figures): left = warm, right = cool,
# spine dark — side is encoded by temperature, chain by shade. Derived
# from the Okabe-Ito colorblind-safe palette.
CHAIN_COLORS = {
    "spine": "#3A3F4A",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}
# Dark-background variant: same warm/cool encoding, spine and joints
# lightened so they read against a near-black ground.
CHAIN_COLORS_DARK = {
    "spine": "#C8CCD6",
    "l_arm": "#E69F00",
    "l_leg": "#D55E00",
    "r_arm": "#56B4E9",
    "r_leg": "#0072B2",
}


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

    left_nodes = {l for l, _ in pairs}
    right_nodes = {r for _, r in pairs}

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


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Preset field values. "paper" is the publication-grade default;
# "debug" reproduces the pre-0.9.0 output exactly (single blue, full
# axes, no floor); "dark" is the paper look on a near-black ground.
_STYLE_PRESETS: dict[str, dict[str, object]] = {
    "paper": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#23262E",
        floor="solid",
        floor_alpha=0.85,
        background="white",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
    "debug": dict(
        bone_width=2.5,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="single",
        chain_colors=CHAIN_COLORS,
        joint_markers=False,
        joint_size=9.0,
        joint_color="#23262E",
        floor=None,
        floor_alpha=0.85,
        background="white",
        axes="full",
        projection="persp",
        dpi=None,
        supersample=1,
        shadow=False,
        ghost_spacing=0.3,
    ),
    "dark": dict(
        bone_width=3.0,
        bone_color=(0.1, 0.2, 0.8),
        color_mode="auto",
        chain_colors=CHAIN_COLORS_DARK,
        joint_markers=True,
        joint_size=9.0,
        joint_color="#E8E8EC",
        floor="solid",
        floor_alpha=0.85,
        background="#16181D",
        axes="off",
        projection="persp",
        dpi=None,
        supersample=2,
        shadow=True,
        ghost_spacing=0.3,
    ),
}

_VALID_COLOR_MODES = {"auto", "chains", "skeleton", "single"}
_VALID_FLOORS = {None, "solid", "grid", "checker"}
_VALID_AXES = {"off", "full"}
_VALID_PROJECTIONS = {"persp", "ortho"}


@dataclass(frozen=True, init=False)
class Style:
    """Visual styling for every bvhplot function.

    Construct from a preset name plus any field overrides::

        Style("paper")                    # the defaults
        Style("paper", floor=None)        # paper look, no ground plane
        Style("dark", bone_width=4.0)

    Presets: ``"paper"`` (publication-grade default: ground plane,
    per-chain colors, joint markers, axes off), ``"debug"`` (the
    pre-0.9.0 look: single blue skeleton, full axes and ticks, no
    floor — for coordinate inspection), ``"dark"`` (paper on a
    near-black ground, for slides and project pages).

    Fields split into two documented groups. **Look** fields apply to
    every figure/video output (matplotlib, OpenCV, vedo offscreen):
    ``bone_width``, ``bone_color``, ``color_mode``, ``chain_colors``,
    ``joint_markers``, ``joint_size``, ``joint_color``, ``floor``,
    ``floor_alpha``, ``background``, ``axes``, ``projection``,
    ``ghost_spacing`` (seconds between the faded trailing poses that
    ``render(ghost=...)`` draws). The two *interactive viewers* apply
    the subset that has meaning in a live window: background, bone
    width, and single-skeleton chain colors in both; floor kind in the
    vedo viewer, where ``"checker"`` falls back to ``"grid"``. Fields
    outside that subset (``axes``, ``projection``, ``joint_markers``,
    ...) do not alter the viewers. **Output** fields apply only where
    raster output is produced: ``dpi`` (matplotlib figures),
    ``supersample`` (OpenCV export), ``shadow`` (vedo offscreen
    renders).

    ``color_mode``: ``"auto"`` uses per-chain colors for a single
    skeleton and flat per-skeleton palette colors for multi-skeleton
    comparisons (the GT-vs-generated convention); ``"chains"`` forces
    chain colors everywhere; ``"skeleton"`` forces the flat palette;
    ``"single"`` draws one skeleton in ``bone_color`` (multi-skeleton
    still uses the palette — the pre-0.9.0 behavior).

    ``axes`` and ``floor`` are independent, so ``Style("paper",
    axes="full")`` keeps the ground plane — and on the matplotlib
    backend that plane paints over the axis panes and grid lines,
    because any floor forces manual draw order (mplot3d's computed
    z-order would wash the skeleton out under the semi-transparent
    plane). For clean coordinate-inspection axes use ``"debug"``, or
    add ``floor=None``.
    """

    bone_width: float
    bone_color: tuple[float, float, float]
    color_mode: str
    chain_colors: dict[str, str]
    joint_markers: bool
    joint_size: float
    joint_color: str
    floor: str | None
    floor_alpha: float
    background: str
    axes: str
    projection: str
    dpi: int | None
    supersample: int
    shadow: bool
    ghost_spacing: float

    def __init__(self, preset: str = "paper", **overrides: object) -> None:
        if preset not in _STYLE_PRESETS:
            raise ValueError(
                f"Unknown style preset {preset!r}. "
                f"Choose from: {sorted(_STYLE_PRESETS)}")
        self._assign_fields(dict(_STYLE_PRESETS[preset]), overrides)

    def _assign_fields(
        self,
        fields: dict[str, object],
        overrides: dict[str, object],
    ) -> None:
        """Shared construction path for __init__ and replace: unknown-
        field check, defensive copies, assignment, validation."""
        unknown = set(overrides) - set(fields)
        if unknown:
            raise TypeError(
                f"Unknown Style field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(fields)}")
        fields.update(overrides)
        for name, value in fields.items():
            # Copy mutable field values (chain_colors) so no instance
            # aliases the module-level preset dicts — mutating one
            # Style must never restyle every other figure.
            if isinstance(value, dict):
                value = dict(value)
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {sorted(_VALID_COLOR_MODES)}, "
                f"got {self.color_mode!r}")
        if self.floor not in _VALID_FLOORS:
            raise ValueError(
                f"floor must be one of "
                f"{sorted(f for f in _VALID_FLOORS if f)} or None, "
                f"got {self.floor!r}")
        if self.axes not in _VALID_AXES:
            raise ValueError(
                f"axes must be one of {sorted(_VALID_AXES)}, "
                f"got {self.axes!r}")
        if self.projection not in _VALID_PROJECTIONS:
            raise ValueError(
                f"projection must be one of {sorted(_VALID_PROJECTIONS)}, "
                f"got {self.projection!r}")
        if not self.bone_width > 0:
            raise ValueError(
                f"bone_width must be positive, got {self.bone_width}")
        if not (isinstance(self.supersample, int) and self.supersample >= 1):
            raise ValueError(
                f"supersample must be an integer >= 1, "
                f"got {self.supersample!r}")
        if not self.ghost_spacing > 0:
            raise ValueError(
                f"ghost_spacing must be positive (seconds), "
                f"got {self.ghost_spacing!r}")

    def replace(self, **overrides: object) -> Style:
        """A new Style with the given fields changed."""
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)}
        new = object.__new__(Style)
        new._assign_fields(fields, overrides)
        return new


def resolve_style(style: Style | str) -> Style:
    """Accept a preset name or a Style instance; return a Style."""
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        return Style(style)
    raise TypeError(
        f"style must be a Style or a preset name string, "
        f"got {type(style).__name__}")


def effective_color_mode(style: Style, n_skeletons: int) -> str:
    """Resolve ``"auto"``/``"single"`` to the concrete mode for n panels.

    The multi-skeleton auto-switch rule: comparisons get flat
    per-skeleton palette colors unless chains are forced explicitly.
    """
    if style.color_mode == "auto":
        return "chains" if n_skeletons == 1 else "skeleton"
    if style.color_mode == "single":
        return "single" if n_skeletons == 1 else "skeleton"
    return style.color_mode


def bone_colors_for_view(
    view: SkeletonView,
    style: Style,
    view_index: int,
    n_skeletons: int,
) -> list:
    """Per-bone colors for one view, in matplotlib-friendly form.

    Each entry is a hex string or an RGB float tuple, parallel to
    ``view.bones``. The OpenCV backend converts these at its border.
    """
    n_bones = len(view.bones)
    mode = effective_color_mode(style, n_skeletons)
    if mode == "skeleton":
        return [PALETTE_MPL[view_index % len(PALETTE_MPL)]] * n_bones
    if mode == "single":
        return [style.bone_color] * n_bones
    # chains — a skeleton with no L/R pairs is all-"spine" in
    # view.bone_chains, i.e. a single dark color (the documented fallback).
    spine_color = style.chain_colors.get("spine", "#3A3F4A")
    return [style.chain_colors.get(chain_name, spine_color)
            for chain_name in view.bone_chains]
