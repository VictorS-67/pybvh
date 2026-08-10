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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._common import SkeletonView


def adaptive_radii(
    frame0: npt.NDArray[np.float64],
    bones: list[tuple[int, int]],
    r_base: float,
) -> tuple[dict[tuple[int, int], float], npt.NDArray[np.float64]]:
    """Per-bone and per-joint radii scaled by bone length.

    Each bone's radius is proportional to its length relative to the
    median (clipped to [0.3, 2]x): short finger bones get thin tubes,
    long limb bones stay thick. A joint's radius is the mean of its
    connected bones' radii.
    """
    lengths = {
        (p, c): float(np.linalg.norm(frame0[c] - frame0[p]))
        for p, c in bones
    }
    med = float(np.median(list(lengths.values()))) if lengths else 1.0
    bone_radii: dict[tuple[int, int], float] = {}
    for (p, c), length in lengths.items():
        ratio = np.clip(length / med, 0.3, 2.0) if med > 0 else 1.0
        bone_radii[(p, c)] = r_base * ratio

    joint_radii = np.full(len(frame0), r_base * 0.5)
    connected: list[list[float]] = [[] for _ in range(len(frame0))]
    for (p, c), rad in bone_radii.items():
        connected[p].append(rad)
        connected[c].append(rad)
    for j in range(len(frame0)):
        if connected[j]:
            joint_radii[j] = float(np.mean(connected[j]))
    return bone_radii, joint_radii


class CapsuleSkeleton:
    """Merged tube+sphere actors for one skeleton, posable per frame.

    Parameters
    ----------
    view : SkeletonView
        Supplies frame-0 coords (for adaptive radii) and the bone list.
    r_base : float
        Base bone radius in scene units.
    color : object
        Uniform actor color (any vedo-parseable form).
    chain_rgb : list[(int, int, int)] or None
        Optional per-bone RGB (0-255). Baked as per-point colors so the
        coloring survives the merge into one actor; joints take the
        color of the bone whose child they are (``spine_rgb`` for the
        root).
    spine_rgb : (int, int, int)
        Fallback joint color under chain coloring.
    flat_lighting : bool
        ``True`` (viewer): ambient-only so colors stay stable across
        frames. ``False`` (offscreen renders): default VTK diffuse
        shading — capsules read as 3D.
    """

    def __init__(
        self,
        view: SkeletonView,
        r_base: float,
        color: object,
        *,
        chain_rgb: list[tuple[int, int, int]] | None = None,
        spine_rgb: tuple[int, int, int] = (58, 63, 74),
        flat_lighting: bool = True,
    ) -> None:
        from vedo import Tube, Sphere, merge  # type: ignore[import-untyped]

        frame0 = view.coords[0]
        bones = view.bones
        self.bone_parent_idx = np.array([b[0] for b in bones], dtype=int)
        self.bone_child_idx = np.array([b[1] for b in bones], dtype=int)

        bone_radii, joint_radii = adaptive_radii(frame0, bones, r_base)

        # --- canonical bone tubes ---
        bone_meshes = []
        bone_verts = []
        for k, (p_i, c_i) in enumerate(bones):
            r = bone_radii.get((p_i, c_i), r_base)
            tube = Tube([[0, 0, 0], [0, 0, 1]], r=[r, r / 2],
                        res=12, c=color)
            if chain_rgb is not None:
                tube.pointcolors = np.tile(
                    np.array(chain_rgb[k], dtype=np.uint8),
                    (tube.npoints, 1))
            bone_verts.append(tube.vertices.copy())
            bone_meshes.append(tube)

        if bone_meshes:
            self.bones_mesh = merge(bone_meshes)
            self.canonical_bone_verts = np.array(bone_verts)
        else:
            self.bones_mesh = None
            self.canonical_bone_verts = np.empty((0, 0, 3))

        # --- canonical joint spheres ---
        joint_rgb_by_node: dict[int, tuple[int, int, int]] = {}
        if chain_rgb is not None:
            for k, (_p, c_i) in enumerate(bones):
                joint_rgb_by_node[c_i] = chain_rgb[k]
        joint_meshes = []
        joint_verts = []
        for j in range(frame0.shape[0]):
            sph = Sphere(pos=(0, 0, 0), r=joint_radii[j], res=12, c=color)
            if chain_rgb is not None:
                sph.pointcolors = np.tile(
                    np.array(joint_rgb_by_node.get(j, spine_rgb),
                             dtype=np.uint8),
                    (sph.npoints, 1))
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
                # Offscreen renders: diffuse shading for 3D depth, with
                # enough ambient that shadow-side faces keep their hue
                # instead of going near-black under the headlight.
                prop.SetAmbient(0.45)
                prop.SetDiffuse(0.6)
                prop.SetSpecular(0.05)

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
