"""k3d interactive backend for Jupyter notebooks.

Provides interactive 3D skeleton playback with camera rotation/zoom
and a frame scrubber widget.

Requires ``k3d >= 2.14`` and ``ipywidgets``.
"""
from __future__ import annotations

import warnings
import numpy as np
import numpy.typing as npt


from ._style import PALETTE_RGB, Style, effective_color_mode
from ._viewport import make_viewport
from ._scene import Scene, UP_AXIS_INDEX
from ._colors import node_colors_255



def _node_colors_uint32(
    scene: Scene,
    style: Style,
    s: int,
) -> npt.NDArray[np.uint32] | None:
    """Per-node chain colors as k3d 0xRRGGBB ints, or None when chain
    coloring does not apply (multi-skeleton / non-chain modes).

    The node-coloring rule itself lives in
    :func:`~._colors.node_colors_255`; this only packs the uint32s.
    """
    if effective_color_mode(style, scene.num_skeletons) != "chains":
        return None
    rgb = node_colors_255(
        scene.views[s], style, s, scene.num_skeletons).astype(np.uint32)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def play_k3d(
    scene: Scene,
    style: Style,
    fps: float,
) -> None:
    """Interactive skeleton playback in a Jupyter notebook via k3d.

    A single-scene backend: all skeletons share one camera (taken from
    the first view) and one viewport over the — possibly laterally
    spread — views.

    Style application (look fields): background, bone width, and chain
    colors for single-skeleton sessions (per-vertex colors — segments
    blend at chain boundaries, a k3d rendering artifact).

    Parameters
    ----------
    scene : Scene
        Prepared visualization.
    style : Style
        Visual styling (look fields).
    fps : float
        Frames per second.

    Returns
    -------
    None
        The plot and its controls are displayed as a side effect.
    """
    import k3d
    from IPython.display import display  # type: ignore[import-untyped]
    from ipywidgets import Play, IntSlider, jslink, HBox, VBox, Label  # type: ignore[import-untyped]
    from matplotlib.colors import to_rgb

    # k3d draws in perspective whatever the style asks.
    viewport = make_viewport(scene.views, projection="persp")
    center, half_span = viewport.center, viewport.half_span
    up_axis = viewport.up_axis
    labels = scene.labels
    skeleton_lines_list = [v.bones for v in scene.views]

    bg_r, bg_g, bg_b = (int(c * 255) for c in to_rgb(style.background))
    background_color = (bg_r << 16) | (bg_g << 8) | bg_b
    width_factor = style.bone_width / 3.0

    # Pre-convert all coordinates to float32 once (k3d requires float32)
    coords_f32 = [v.coords.astype(np.float32) for v in scene.views]

    num_frames = coords_f32[0].shape[0]
    n_skeletons = scene.num_skeletons

    # k3d passes uint32 indices to a trait that traittypes validates as
    # float32; the coercion is harmless but noisy. Scoped to object
    # construction (the only place the warning fires) so the filter
    # doesn't leak into the caller's process-wide warning state.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*dtype.*does not match required type.*",
            module="traittypes")

        plot = k3d.plot(name='pybvh skeleton viewer',
                        background_color=background_color)

        # Build k3d objects for each skeleton
        skeleton_objects: list[tuple[k3d.objects.Lines, k3d.objects.Points]] = []

        for s, (coords, bones) in enumerate(
                zip(coords_f32, skeleton_lines_list)):
            frame0 = coords[0]

            # Build indices array for k3d.lines: pairs of [start, end]
            indices = np.array(bones, dtype=np.uint32)  # (num_bones, 2)

            # Color as hex int (0xRRGGBB); chain coloring (when it
            # applies) rides as per-vertex colors instead.
            r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
            color = int(r) << 16 | int(g) << 8 | int(b)
            node_colors = _node_colors_uint32(scene, style, s)

            lines = k3d.lines(
                frame0, indices,
                indices_type='segment',
                color=color,
                colors=node_colors if node_colors is not None else [],
                width=0.02 * half_span * width_factor,
                name=(labels[s] if labels and labels[s] is not None
                      else f"Skeleton {s}"),
            )
            points = k3d.points(
                frame0,
                color=color,
                colors=node_colors if node_colors is not None else [],
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
        up_idx = UP_AXIS_INDEX.get(up_axis, 2)
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

    # The viewport's camera, the one every backend aims from the same
    # (azimuth, elevation, up) angles. k3d's camera is a 9-element list:
    # [eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z]
    eye, target, up = viewport.camera()
    plot.camera = [float(value) for value in (*eye, *target, *up)]

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
    # No return value — the plot is already displayed above. Returning
    # the k3d.Plot would cause Jupyter to auto-display it a second time
    # with different axis ranges.
