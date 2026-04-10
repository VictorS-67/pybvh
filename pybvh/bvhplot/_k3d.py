"""k3d interactive backend for Jupyter notebooks.

Provides interactive 3D skeleton playback with camera rotation/zoom
and a frame scrubber widget.

Requires ``k3d >= 2.14`` and ``ipywidgets``.
"""
from __future__ import annotations

import warnings
import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._common import PALETTE_RGB, build_view_matrix, _UP_AXIS_INDEX

if TYPE_CHECKING:
    from ..bvh import Bvh


def play_k3d(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
) -> object:
    """Interactive skeleton playback in a Jupyter notebook via k3d.

    Parameters
    ----------
    bvh_list : list[Bvh]
        Skeleton objects.
    coords_list : list[ndarray]
        Spatial coordinates per skeleton, each ``(F, N, 3)``.
    fps : float
        Frames per second.
    labels : list[str] or None
        Labels per skeleton.
    skeleton_lines_list : list
        Precomputed bone index pairs per skeleton.
    center : ndarray (3,)
        Bounding box center.
    half_span : float
        Half side of cubic bounding box.
    azimuth : float
        Camera azimuth in degrees.
    elevation : float
        Camera elevation in degrees.
    up_axis : str
        Up axis (``'x'``, ``'y'``, or ``'z'``).

    Returns
    -------
    plot : k3d.Plot
        The k3d plot widget.
    """
    # Suppress traittypes dtype coercion warning (k3d passes uint32 indices
    # to a trait that validates as float32; the coercion is harmless).
    warnings.filterwarnings(
        "ignore", message=".*dtype.*does not match required type.*",
        module="traittypes")

    import k3d
    from IPython.display import display  # type: ignore[import-untyped]
    from ipywidgets import Play, IntSlider, jslink, HBox, VBox, Label  # type: ignore[import-untyped]

    # Pre-convert all coordinates to float32 once (k3d requires float32)
    coords_f32 = [c.astype(np.float32) for c in coords_list]

    num_frames = coords_f32[0].shape[0]
    n_skeletons = len(bvh_list)

    plot = k3d.plot(name='pybvh skeleton viewer')

    # Build k3d objects for each skeleton
    skeleton_objects: list[tuple[k3d.objects.Lines, k3d.objects.Points]] = []

    for s, (coords, bones) in enumerate(
            zip(coords_f32, skeleton_lines_list)):
        frame0 = coords[0]

        # Build indices array for k3d.lines: pairs of [start, end]
        indices = np.array(bones, dtype=np.uint32)  # (num_bones, 2)

        # Color as hex int (0xRRGGBB)
        r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
        color = int(r) << 16 | int(g) << 8 | int(b)

        lines = k3d.lines(
            frame0, indices,
            indices_type='segment',
            color=color,
            width=0.02 * half_span,
            name=labels[s] if labels and s < len(labels) else f"Skeleton {s}",
        )
        points = k3d.points(
            frame0,
            color=color,
            point_size=0.03 * half_span,
            name=f"Joints {s}",
        )

        plot += lines
        plot += points
        skeleton_objects.append((lines, points))

    # --- Root trajectory projected on the floor ---
    # Animated trail: vertices [0:current_frame] show the actual past
    # path, the remaining vertices collapse to the current frame so the
    # trail "grows" as the animation plays.
    # Snap the trail to the grid bottom (center - half_span on the up axis)
    # rather than to the lowest joint, because the k3d bbox is cubic and
    # extends below the lowest joint. Otherwise the trail floats above the
    # visible grid floor and parallax makes it appear offset from its true
    # XY position when viewed from an oblique angle.
    up_idx = _UP_AXIS_INDEX.get(up_axis, 1)
    floor_level = float(center[up_idx] - half_span)
    trail_objects: list[k3d.objects.Line] = []
    trail_full_paths: list[npt.NDArray[np.float32]] = []
    for s, coords in enumerate(coords_f32):
        root_path = coords[:, 0, :].copy()  # (F, 3)
        root_path[:, up_idx] = floor_level
        trail_full_paths.append(root_path)

        # Initial trail: all vertices collapsed at frame 0
        initial = np.tile(root_path[0], (num_frames, 1)).astype(np.float32)

        r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
        color = int(r) << 16 | int(g) << 8 | int(b)
        trail = k3d.line(
            initial,
            color=color,
            width=0.015 * half_span,
            opacity=0.6,
            shader='thick',
            name=f"Trajectory {s}",
        )
        plot += trail
        trail_objects.append(trail)

    # Set grid to cover the full motion extent
    grid_min = center - half_span
    grid_max = center + half_span
    plot.grid = [
        float(grid_min[0]), float(grid_min[1]), float(grid_min[2]),
        float(grid_max[0]), float(grid_max[1]), float(grid_max[2]),
    ]
    plot.grid_auto_fit = False
    plot.camera_auto_fit = False

    # Set camera explicitly using the same convention as matplotlib /
    # opencv / vedo backends so all backends produce identical views
    # for the same (azimuth, elevation, up_axis) parameters.
    # k3d's camera is a 9-element list:
    # [eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z]
    view_mat = build_view_matrix(azimuth, elevation, up_axis)
    eye_dir = view_mat[2]  # toward viewer
    cam_up = view_mat[1]
    cam_dist = half_span * 4.0
    cam_pos = center + eye_dir * cam_dist
    plot.camera = [
        float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
        float(center[0]), float(center[1]), float(center[2]),
        float(cam_up[0]), float(cam_up[1]), float(cam_up[2]),
    ]

    # Animation controls
    play_widget = Play(
        value=0,
        min=0,
        max=num_frames - 1,
        step=1,
        interval=int(1000.0 / fps),
        description='',
    )
    slider = IntSlider(
        value=0,
        min=0,
        max=num_frames - 1,
        step=1,
        description='Frame',
        layout={'width': '500px'},
    )
    frame_label = Label(value=f'0 / {num_frames - 1}')

    jslink((play_widget, 'value'), (slider, 'value'))

    def on_frame_change(change: dict) -> None:
        f = change['new']
        frame_label.value = f'{f} / {num_frames - 1}'
        for s, (lines_obj, pts_obj) in enumerate(skeleton_objects):
            frame_data = coords_f32[s][f]
            lines_obj.vertices = frame_data
            pts_obj.positions = frame_data
        # Update trails: vertices [0..f] are the real past path, the rest
        # collapse to position f so the trail grows as the animation plays.
        for s, trail_obj in enumerate(trail_objects):
            full_path = trail_full_paths[s]
            verts = np.empty_like(full_path)
            verts[:f + 1] = full_path[:f + 1]
            verts[f + 1:] = full_path[f]
            trail_obj.vertices = verts

    slider.observe(on_frame_change, names='value')

    controls = HBox([play_widget, slider, frame_label])
    display(VBox([plot, controls]))

    # Return None — the plot is already displayed above.
    # Returning the k3d.Plot would cause Jupyter to auto-display it
    # a second time with different axis ranges.
    return None
