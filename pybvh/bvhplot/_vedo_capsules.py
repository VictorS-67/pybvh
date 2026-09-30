"""Capsule-skeleton geometry for the vedo backends.

One skeleton = two merged vedo meshes (all bone tubes, all joint
spheres — 2 VTK actors instead of ~120) with canonical vertices kept
around so per-frame posing is a single vectorized ``einsum``. Shared
by the interactive viewer (`_vedo.py`) and the offscreen renderer
(`_vedo_offscreen.py`) so the two can never drift apart.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING, Sequence

from ._colors import rgb255
from ._style import bone_width_scale
from ._viewport import STANDING_STILL_HALF_SPAN

if TYPE_CHECKING:
    from ._scene import SkeletonView
    from ._viewport import Viewport

# VTK has a depth buffer, so surfaces that share a plane flicker. The
# viewport puts the ground plane exactly at the scene ground; this
# toolkit draws it a hair below, and the projected shadows between the
# plane and the ground, so plane, shadow and whatever lies on the
# ground (the root trail) have a fixed order. Fractions of the
# half-span. These are z-fighting epsilons and nothing more: sinking
# the plane by the capsule radius was considered and rejected (ADR
# 0002).
FLOOR_EPSILON = 0.004
SHADOW_EPSILON = 0.002


# The offscreen capsules' specular highlight color (#AAAAAA).
_HIGHLIGHT_GRAY = (170 / 255, 170 / 255, 170 / 255)

LENGTH_BOOST = (1.0, 1.5)   # long bones get plumper; short ones never thinner
# Share of the gap to its nearest crowder that a bone may take. Two facing
# crowders nominally sum past the gap here, but a bone is at full radius only
# at its parent end (tubes taper to half) and CHAIN_TAPER slims each link
# after the first, so fingers stay separate along their length while their
# thick ends meet at the knuckles — which is what reads as a palm. Above
# ~0.7 that fusion spreads down the fingers on tightly-packed rigs.
CROWD_FRACTION = 0.60
SAME_DIR_COS = 0.5          # "same direction" = within 60 degrees
MIN_OVERLAP = 0.25          # side-by-side run, as a fraction of the shorter bone
# Each link inside a crowded run is slimmer than the last. This compounds
# along the chain, so a harsh value eventually bites loosely-crowded limbs
# several links down; 0.85 shapes hands while leaving plain rigs untouched.
CHAIN_TAPER = 0.85
STUB_CAP_FACTOR = 2.0       # a stub is at most 2x the thinnest bone it joins
MIN_RADIUS_FRACTION = 0.10  # visibility floor

# Base capsule radius as a fraction of the body size, at the paper
# style's bone width: the v0.9.0 radius of a standing still, 2.6% of
# its half-span.
BASE_RADIUS_FRACTION = 0.026 * STANDING_STILL_HALF_SPAN


def vedo_rgb(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """A 0-255 RGB color in the form vedo reads unambiguously: floats
    in [0, 1].

    vedo reads a CSS ``"rgb(r,g,b)"`` string as black, and scales an
    integer triple by 1/255 only when a component exceeds 1, so a
    0-255 ``(0, 0, 1)`` would come out full blue.
    """
    r, g, b = rgb
    return (r / 255, g / 255, b / 255)


def vedo_color(color: object) -> tuple[float, float, float]:
    """A style color (any form matplotlib parses) in vedo's form.

    Style colors are read by matplotlib's parser in every backend.
    vedo's own parser differs: it rejects short hex such as
    ``"#fff"`` and reads matplotlib-only names such as ``"C0"`` as
    gray. The color goes through :func:`~._colors.rgb255`, the 0-255
    conversion every other bvhplot color takes, then
    :func:`vedo_rgb`.
    """
    return vedo_rgb(rgb255(color))


def floor_placement(
    viewport: Viewport,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Where a vedo ``Plane`` or ``Grid`` for the viewport's ground
    plane goes: ``(position, normal, side)``.

    The position is the plane's centre, ``FLOOR_EPSILON`` half-spans
    below the scene ground; the side is the full side length vedo's
    ``s=`` takes, twice the viewport's floor reach."""
    up = viewport.up_index
    position = viewport.center.copy()
    position[up] = viewport.below_floor(FLOOR_EPSILON * viewport.half_span)
    normal = np.zeros(3)
    normal[up] = 1.0
    return position, normal, 2.0 * viewport.floor_reach


def shadow_height(viewport: Viewport) -> float:
    """The height projected shadows are flattened to: between the
    ground plane as drawn and the scene ground."""
    return viewport.below_floor(SHADOW_EPSILON * viewport.half_span)


def base_radius(body_size: float, bone_width: float) -> float:
    """Base bone radius in scene units, the single sizing formula of
    the viewer and the offscreen renderer.

    ``BASE_RADIUS_FRACTION`` of the body size
    (:attr:`~._scene.SkeletonView.body_size`, the rest pose's height),
    scaled by the style's bone width
    (:func:`~._style.bone_width_scale`: the paper default is the 1:1
    anchor). Taken from the body and not from the viewport's
    half-span, which grows with the distance the clip travels: a still
    and the whole clip draw a body at the same proportions, and each
    skeleton of a scene is sized from its own. Hand bones draw at half
    this base (see :func:`adaptive_radii`).
    """
    return BASE_RADIUS_FRACTION * body_size * bone_width_scale(bone_width)


def _segment_frames(pose, bone_array):
    """Endpoints, lengths and unit directions for every bone."""
    starts, ends = pose[bone_array[:, 0]], pose[bone_array[:, 1]]
    delta = ends - starts
    lengths = np.linalg.norm(delta, axis=1)
    directions = delta / np.maximum(lengths, 1e-9)[:, None]
    return starts, ends, lengths, directions


def crowding_clearance(
    rest_pose: npt.NDArray[np.float64],
    bones: list[tuple[int, int]],
) -> npt.NDArray[np.float64]:
    """Per-bone lateral room before the nearest bone that crowds it.

    A bone is *crowded* only by one that (1) runs in the same direction
    (signed parent-to-child, within ``SAME_DIR_COS``), (2) overlaps it
    along that shared direction, and (3) is laterally close; the return
    value is that lateral offset, or ``inf`` for a bone nothing crowds.
    All three conjuncts are load-bearing, and each rules out a case the
    others accept: unsigned parallelism would read two clavicles
    diverging from the spine as crowding each other; without the
    overlap test a bone's own chain continuation (the link two joints
    up a neck) counts as a crowder, which is what a plain
    nearest-bone-distance measure gets wrong; and the lateral offset,
    not the raw segment distance, is what merges two tubes visually.

    Measured on the **rest pose**, so the result describes the
    skeleton's structure rather than a transient pose (arms crossed in
    frame 0 must not thin the arms); this also makes it invariant to
    the rest/animation convention mismatches pybvh warns about, since
    only relative geometry is used. The alternative — a low percentile
    over sampled animation frames — tracks poses that actually occur
    but thins limbs that merely pass close by, and costs a pass over
    the motion.
    """
    bone_array = np.asarray(bones, dtype=int)
    starts, ends, lengths, directions = _segment_frames(rest_pose, bone_array)
    centers = (starts + ends) / 2

    shares_node = (bone_array[:, None, :, None]
                   == bone_array[None, :, None, :]).any(axis=(2, 3))
    same_direction = (directions @ directions.T) > SAME_DIR_COS

    shared_axis = directions[:, None, :] + directions[None, :, :]
    shared_axis /= np.maximum(
        np.linalg.norm(shared_axis, axis=-1, keepdims=True), 1e-9)
    proj_start_i = (starts[:, None, :] * shared_axis).sum(-1)
    proj_end_i = (ends[:, None, :] * shared_axis).sum(-1)
    proj_start_j = (starts[None, :, :] * shared_axis).sum(-1)
    proj_end_j = (ends[None, :, :] * shared_axis).sum(-1)
    overlap = (np.minimum(np.maximum(proj_start_i, proj_end_i),
                          np.maximum(proj_start_j, proj_end_j))
               - np.maximum(np.minimum(proj_start_i, proj_end_i),
                            np.minimum(proj_start_j, proj_end_j)))
    side_by_side = overlap > MIN_OVERLAP * np.minimum(
        lengths[:, None], lengths[None, :])

    offset = centers[None, :, :] - centers[:, None, :]
    axial = (offset * shared_axis).sum(-1)
    lateral = np.linalg.norm(offset - axial[..., None] * shared_axis, axis=-1)

    degenerate = lengths < 1e-8
    crowds = (same_direction & side_by_side & ~shares_node
              & ~degenerate[None, :] & ~degenerate[:, None])
    np.fill_diagonal(crowds, False)
    return np.where(crowds, lateral, np.inf).min(axis=1)


def adaptive_radii(
    frame0: npt.NDArray[np.float64],
    bones: list[tuple[int, int]],
    r_base: float,
    rest_pose: npt.NDArray[np.float64] | None = None,
) -> tuple[dict[tuple[int, int], float], npt.NDArray[np.float64]]:
    """Per-bone and per-joint capsule radii, from geometry alone.

    Four local rules, in order. **Length** scales a bone up with its
    length relative to the skeleton median, clipped to
    ``LENGTH_BOOST`` — upward only, so long limbs read as plump
    capsules while short links keep the base radius. (Scaling *down*
    with length is the obvious alternative and is what this used to
    do; it fails on short-but-isolated bones, thinning a two-link neck
    to a third of the bone below it.) **Crowding** then caps a bone at
    ``CROWD_FRACTION`` of its lateral room (:func:`crowding_clearance`)
    — the rule that keeps fingers legible without knowing they are
    fingers. **Propagation** carries each bone's crowding cap down the
    chain, shrinking it by ``CHAIN_TAPER`` per link, so a crowded run
    tapers from its base outward: fingertips that splay apart in the
    rest pose stay slimmer than the knuckles feeding them. Without the
    taper the caps *grow* distally (splayed tips have more room than
    packed metacarpals), which reads as a hand thin at the wrist and
    fattest at the tips — backwards. ``inf`` is taper-invariant, so
    uncrowded chains such as a spine are untouched. The cap is what propagates, not the radius
    — a short clavicle feeding a long humerus means a parent is
    legitimately thinner than its child, so propagating the radius
    itself would spindle every limb. **Stub capping** finally limits a
    bone *too short to read as a tube* (length below its own radius) to
    ``STUB_CAP_FACTOR`` times the thinnest bone it joins, so the
    transverse stubs inside a palm follow their surroundings; longer
    bones are exempt, or a wrist joining five thin metacarpals would be
    dragged down to finger width.

    Crowding is measured on *rest_pose* (structure, not a transient
    pose), which must be in *frame0*'s unit: the caller converts it
    (:class:`CapsuleSkeleton` scales the view's rest pose by
    :attr:`~._scene.SkeletonView.coords_per_rest_unit`), so the one
    unit conversion of a view is the one its body size uses. Without
    a rest pose in that unit (``None``), crowding is measured on
    *frame0*: the alternative, skipping the crowding rule, would draw
    side-by-side bones at full radius, fused into one. A joint's
    radius is the **minimum** of the bones it joins, never the mean:
    at a hub where one thick bone meets several thin ones, the mean
    bulges a sphere out past the thin tubes.

    Returns ``(bone_radii, joint_radii)``.
    """
    if not bones:
        return {}, np.full(len(frame0), r_base * 0.5)

    bone_array = np.asarray(bones, dtype=int)
    _, _, lengths, _ = _segment_frames(frame0, bone_array)
    median_length = float(np.median(lengths)) if len(lengths) else 1.0
    radii = r_base * np.clip(
        lengths / median_length if median_length > 0 else 1.0, *LENGTH_BOOST)

    if rest_pose is None:
        rest_pose = frame0
    cap = CROWD_FRACTION * crowding_clearance(rest_pose, bones)

    parent_bone = _parent_bone_indices(bones)
    for index in _root_first_order(parent_bone):
        if parent_bone[index] is not None:
            cap[index] = min(cap[index],
                             cap[parent_bone[index]] * CHAIN_TAPER)
    radii = np.minimum(radii, cap)

    neighbours = _neighbour_indices(parent_bone)
    uncapped = radii.copy()
    for index in range(len(bones)):
        if lengths[index] >= uncapped[index] or not neighbours[index]:
            continue
        thinnest = min(uncapped[j] for j in neighbours[index])
        radii[index] = min(radii[index], STUB_CAP_FACTOR * thinnest)

    radii = np.maximum(radii, MIN_RADIUS_FRACTION * r_base)
    bone_radii = {bone: float(radii[k]) for k, bone in enumerate(bones)}

    joint_radii = np.full(len(frame0), r_base * 0.5)
    connected: list[list[float]] = [[] for _ in range(len(frame0))]
    for (p, c), radius in bone_radii.items():
        connected[p].append(radius)
        connected[c].append(radius)
    for j in range(len(frame0)):
        if connected[j]:
            joint_radii[j] = float(min(connected[j]))
    return bone_radii, joint_radii


def _parent_bone_indices(bones):
    """For each bone, the index of the bone ending at its parent node."""
    bone_of_child = {child: k for k, (_p, child) in enumerate(bones)}
    return [bone_of_child.get(parent) for parent, _c in bones]


def _root_first_order(parent_bone):
    """Bone indices ordered parents before children (cycle-safe)."""
    children: dict[int, list[int]] = {}
    roots = []
    for k, parent in enumerate(parent_bone):
        if parent is None:
            roots.append(k)
        else:
            children.setdefault(parent, []).append(k)
    order, stack, seen = [], list(reversed(roots)), set()
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        order.append(k)
        stack.extend(reversed(children.get(k, [])))
    order.extend(k for k in range(len(parent_bone)) if k not in seen)
    return order


def _neighbour_indices(parent_bone):
    """Bones sharing a node with each bone (its parent and its children)."""
    neighbours: list[list[int]] = [[] for _ in parent_bone]
    for k, parent in enumerate(parent_bone):
        if parent is not None:
            neighbours[k].append(parent)
            neighbours[parent].append(k)
    return neighbours


class CapsuleSkeleton:
    """Merged tube+sphere actors for one skeleton, posable per frame.

    Parameters
    ----------
    view : SkeletonView
        Supplies frame-0 coords (for adaptive radii), the bone list,
        the rest pose the crowding is measured on (in the coords' unit,
        by :attr:`~._scene.SkeletonView.coords_per_rest_unit`; frame 0
        stands in for it when the view has no such ratio) and the body
        size the radii are fractions of.
    bone_width : float
        The style's bone width; the base radius is
        :func:`base_radius` of the view's body size and this width,
        and is kept as :attr:`base_radius`.
    bone_rgb : sequence of (int, int, int)
        Per-bone RGB (0-255), parallel to ``view.bones``
        (:func:`~._colors.bone_colors_255`).
    joint_rgb : (N, 3) uint8 array
        Per-node RGB (0-255) (:func:`~._colors.node_colors_255`).
        Both are baked as per-point colors, so the coloring survives
        the merge into one actor and no mesh is left to VTK's default
        scalar map (a tube carries its radius as point data).
    flat_lighting : bool
        ``True`` (viewer): ambient-only so colors stay stable across
        frames. ``False`` (offscreen renders): default VTK diffuse
        shading — capsules read as 3D.
    """

    def __init__(
        self,
        view: SkeletonView,
        bone_width: float,
        bone_rgb: Sequence[tuple[int, int, int]],
        joint_rgb: npt.NDArray[np.uint8],
        *,
        flat_lighting: bool = True,
    ) -> None:
        from vedo import Tube, Sphere, merge  # type: ignore[import-untyped]

        frame0 = view.coords[0]
        bones = view.bones
        r_base = base_radius(view.body_size, bone_width)
        self.base_radius = r_base
        self.bone_parent_idx = np.array([b[0] for b in bones], dtype=int)
        self.bone_child_idx = np.array([b[1] for b in bones], dtype=int)

        coords_per_rest_unit = view.coords_per_rest_unit
        rest_pose = (None if coords_per_rest_unit is None
                     else view.rest_coords * coords_per_rest_unit)
        bone_radii, joint_radii = adaptive_radii(
            frame0, bones, r_base, rest_pose)

        # --- canonical bone tubes ---
        bone_meshes = []
        bone_verts = []
        for k, (p_i, c_i) in enumerate(bones):
            r = bone_radii.get((p_i, c_i), r_base)
            tube = Tube([[0, 0, 0], [0, 0, 1]], r=[r, r / 2], res=12)
            tube.pointcolors = np.tile(
                np.array(bone_rgb[k], dtype=np.uint8), (tube.npoints, 1))
            bone_verts.append(tube.vertices.copy())
            bone_meshes.append(tube)

        if bone_meshes:
            self.bones_mesh = merge(bone_meshes)
            self.canonical_bone_verts = np.array(bone_verts)
        else:
            self.bones_mesh = None
            self.canonical_bone_verts = np.empty((0, 0, 3))

        # --- canonical joint spheres ---
        joint_meshes = []
        joint_verts = []
        for j in range(frame0.shape[0]):
            sph = Sphere(pos=(0, 0, 0), r=joint_radii[j], res=12)
            sph.pointcolors = np.tile(
                np.asarray(joint_rgb[j], dtype=np.uint8), (sph.npoints, 1))
            joint_verts.append(sph.vertices.copy())
            joint_meshes.append(sph)
        self.joints_mesh = merge(joint_meshes)
        self.canonical_joint_verts = np.array(joint_verts)

        for mesh in (self.bones_mesh, self.joints_mesh):
            if mesh is None:
                continue
            prop = mesh.actor.GetProperty()
            if flat_lighting:
                # Viewer: ambient-only so colors stay stable across
                # frames as bones rotate.
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
            else:
                # Offscreen renders: near-full diffuse so the tubes
                # shade on both sides and read as round 3D capsules,
                # with just enough ambient that shadow-side faces keep
                # their hue instead of going near-black. The point
                # colors replace the ambient and diffuse colors only,
                # so the highlight's color is set here.
                prop.SetAmbient(0.2)
                prop.SetDiffuse(0.8)
                prop.SetSpecular(0.1)
                prop.SetSpecularColor(_HIGHLIGHT_GRAY)

    @property
    def actors(self) -> list:
        return [m for m in (self.bones_mesh, self.joints_mesh)
                if m is not None]

    def update(self, frame_data: npt.NDArray[np.float64]) -> None:
        """Pose both merged meshes to *frame_data* via vectorized numpy."""
        p_idx = self.bone_parent_idx
        c_idx = self.bone_child_idx

        if len(p_idx) > 0 and self.bones_mesh is not None:
            starts = frame_data[p_idx]                     # (n_bones, 3)
            ends = frame_data[c_idx]                       # (n_bones, 3)
            diffs = ends - starts
            lengths = np.linalg.norm(diffs, axis=1)        # (n_bones,)

            # Vectorized rotation+scale matrices
            safe_len = np.where(lengths < 1e-8, 1.0, lengths)
            z_ax = diffs / safe_len[:, np.newaxis]
            refs = np.tile(np.array([1., 0, 0]), (len(p_idx), 1))
            refs[np.abs(z_ax[:, 0]) >= 0.9] = [0., 1, 0]
            x_ax = np.cross(refs, z_ax)
            x_ax /= np.linalg.norm(x_ax, axis=1, keepdims=True).clip(1e-10)
            y_ax = np.cross(z_ax, x_ax)

            # (n_bones, 3, 3): columns are [x, y, z*length]
            rotscale = np.stack(
                [x_ax, y_ax, z_ax * lengths[:, np.newaxis]], axis=2)

            # Single einsum: R @ v for all bones at once
            transformed = (
                np.einsum('bij,bvj->bvi', rotscale,
                          self.canonical_bone_verts)
                + starts[:, np.newaxis, :])

            # Collapse zero-length bones (degenerate triangles)
            zero = np.where(lengths < 1e-8)[0]
            if len(zero):
                for zi in zero:
                    transformed[zi] = starts[zi]

            self.bones_mesh.vertices = transformed.reshape(-1, 3)

        # Joints: vectorized translation (single operation)
        self.joints_mesh.vertices = (
            self.canonical_joint_verts + frame_data[:, np.newaxis, :]
        ).reshape(-1, 3)
