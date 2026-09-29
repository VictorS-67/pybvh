"""A Scene built from plain arrays, with no Bvh anywhere.

The proof that the Scene is a real seam: every backend must draw this
exactly as it draws a Scene that :func:`make_scene` produced from a
:class:`Bvh`. The module imports only NumPy and the Scene container;
``tests/test_scene_backends.py`` pins that.

The figure is a nine-node stick person, y up, facing +z, walking along
+z with a small bob and a hand swing: enough motion for ghosts, floor
traces, follow cameras and facing arrows to have something to show.
"""
from __future__ import annotations

import numpy as np

from pybvh.bvhplot._scene import Scene, SkeletonView, compute_unified_limits

NODE_NAMES = ["Hips", "Spine", "Head", "LeftArm", "LeftHand",
              "RightArm", "RightHand", "LeftFoot", "RightFoot"]
BONES = [(0, 1), (1, 2), (1, 3), (3, 4), (1, 5), (5, 6), (0, 7), (0, 8)]
BONE_CHAINS = ["spine", "spine", "l_arm", "l_arm", "r_arm", "r_arm",
               "l_leg", "r_leg"]
# Joint L/R pairs in node index space, the facing geometry's pairs.
LR_PAIRS = np.array([[3, 5], [4, 6], [7, 8]], dtype=np.intp)

# Rest pose: y up, facing +z, root at the origin. With up = +y and
# forward = +z the character's left is up x forward = +x.
REST_COORDS = np.array([
    [0.00, 1.0, 0.0],   # Hips
    [0.00, 1.5, 0.0],   # Spine
    [0.00, 1.8, 0.0],   # Head
    [0.30, 1.5, 0.0],   # LeftArm
    [0.60, 1.2, 0.0],   # LeftHand
    [-0.30, 1.5, 0.0],  # RightArm
    [-0.60, 1.2, 0.0],  # RightHand
    [0.15, 0.0, 0.0],   # LeftFoot
    [-0.15, 0.0, 0.0],  # RightFoot
])


def make_array_view(
    n_frames: int = 12,
    frame_time: float = 1 / 30,
    label: str | None = None,
    walk_speed: float = 0.05,
    lateral_shift: float = 0.0,
) -> SkeletonView:
    """One walking stick person as a complete SkeletonView."""
    t = np.arange(n_frames, dtype=np.float64)
    coords = np.repeat(REST_COORDS[np.newaxis], n_frames, axis=0)
    coords[:, :, 2] += (walk_speed * t)[:, np.newaxis]        # walk along +z
    coords[:, :, 1] += (0.02 * np.sin(0.8 * t))[:, np.newaxis]  # a little bob
    swing = 0.1 * np.sin(0.8 * t)
    coords[:, 4, 2] += swing                                   # hands swing
    coords[:, 6, 2] -= swing
    coords[:, :, 0] += lateral_shift

    center, half_span = compute_unified_limits([coords])
    # root_trajectory's [sin, cos] heading for a character facing +z
    # with y up: the ground basis is (x, z), cos along x, sin along z.
    root_heading = np.tile([1.0, 0.0], (n_frames, 1))

    return SkeletonView(
        coords=coords,
        bones=list(BONES),
        label=label,
        center=center,
        half_span=half_span,
        azimuth=-20.0,
        elevation=20.0,
        up_axis="y",
        floor_height=float(coords[..., 1].min()),
        frame_time=frame_time,
        node_names=list(NODE_NAMES),
        rest_coords=REST_COORDS.copy(),
        lr_pairs=LR_PAIRS.copy(),
        up_vector=np.array([0.0, 1.0, 0.0]),
        forward_axis="+z",
        bone_chains=list(BONE_CHAINS),
        root_heading=root_heading,
        up_sign=1.0,
    )


def make_array_scene(
    n_frames: int = 12,
    n_skeletons: int = 1,
    labels: list[str] | None = None,
) -> Scene:
    """A Scene of ``n_skeletons`` array-built views, side by side in x."""
    views = [
        make_array_view(
            n_frames,
            label=labels[i] if labels else None,
            lateral_shift=1.5 * i)
        for i in range(n_skeletons)]
    return Scene(views=views)
