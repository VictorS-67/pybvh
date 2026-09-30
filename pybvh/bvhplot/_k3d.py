"""k3d interactive backend for Jupyter notebooks.

Provides interactive 3D skeleton playback with camera rotation/zoom
and a frame scrubber widget.

Requires ``k3d >= 2.14`` and ``ipywidgets``.
"""
from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._style import (
    PALETTE_RGB, Style, bone_width_scale, effective_color_mode)
from ._viewport import STANDING_STILL_HALF_SPAN, Viewport, make_viewport
from ._scene import Scene
from ._colors import (
    floor_palette, grid_box_colors, node_colors_255, rgb255)

# The floor is drawn this far below the scene ground, in half-spans, so
# the root trail, which lies exactly on the ground, has a fixed order
# with it. A z-fighting epsilon and nothing more (ADR 0002).
FLOOR_EPSILON = 0.004
# Lines per ground direction of a "grid" floor.
FLOOR_GRID_LINES = 21

# What is drawn on a body is sized from it, in these fractions of its
# body size: the v0.9.0 sizes of a standing still, which were 2%, 3%
# and 1.5% of its half-span. The bones' width is at the paper style's
# bone width.
BONE_WIDTH_FRACTION = 0.02 * STANDING_STILL_HALF_SPAN
JOINT_SIZE_FRACTION = 0.03 * STANDING_STILL_HALF_SPAN
TRAIL_WIDTH_FRACTION = 0.015 * STANDING_STILL_HALF_SPAN


def _packed(rgb: tuple[int, int, int]) -> int:
    """0-255 RGB as the 0xRRGGBB int k3d takes for a color."""
    r, g, b = rgb
    return (r << 16) | (g << 8) | b


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


class _Plot(NamedTuple):
    """A built k3d plot and the handles its frame callback updates."""

    plot: Any                        # k3d.plot.Plot
    viewport: Viewport
    coords: list[npt.NDArray[np.float32]]        # per skeleton, (F, N, 3)
    skeletons: list[tuple[Any, Any]]             # (k3d Lines, k3d Points)
    trails: list[Any]                            # k3d Line per skeleton
    trail_paths: list[npt.NDArray[np.float32]]   # per skeleton, (F, 3)
    floor: Any | None                # k3d Mesh or Lines, None without a floor


def _build_plot(
    scene: Scene,
    style: Style,
) -> _Plot:
    """Build the k3d plot for *scene*, posed to frame 0.

    Everything :func:`play_k3d` shows except the playback widgets, so
    that what is drawn, and where, can be read back without a notebook.
    """
    import k3d

    # k3d draws in perspective whatever the style asks.
    viewport = make_viewport(scene.views, projection="persp")
    labels = scene.labels

    grid_line_rgb, grid_label_rgb = grid_box_colors(style)
    width_factor = bone_width_scale(style.bone_width)

    # Pre-convert all coordinates to float32 once (k3d requires float32)
    coords_f32 = [v.coords.astype(np.float32) for v in scene.views]

    num_frames = coords_f32[0].shape[0]
    # k3d passes uint32 indices to a trait that traittypes validates as
    # float32; the coercion is harmless but noisy. Scoped to object
    # construction (the only place the warning fires) so the filter
    # doesn't leak into the caller's process-wide warning state.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*dtype.*does not match required type.*",
            module="traittypes")

        plot = k3d.plot(name='pybvh skeleton viewer',
                        background_color=_packed(rgb255(style.background)),
                        grid_color=_packed(grid_line_rgb),
                        label_color=_packed(grid_label_rgb))

        # Build k3d objects for each skeleton
        skeleton_objects: list[tuple[Any, Any]] = []

        for s, (view, coords) in enumerate(zip(scene.views, coords_f32)):
            frame0 = coords[0]

            # Build indices array for k3d.lines: pairs of [start, end]
            indices = np.array(view.bones, dtype=np.uint32)  # (num_bones, 2)

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
                width=BONE_WIDTH_FRACTION * view.body_size * width_factor,
                shader='thick',
                name=(labels[s] if labels and labels[s] is not None
                      else f"Skeleton {s}"),
            )
            points = k3d.points(
                frame0,
                color=color,
                colors=node_colors if node_colors is not None else [],
                point_size=JOINT_SIZE_FRACTION * view.body_size,
                shader='3dSpecular',
                name=f"Joints {s}",
            )

            plot += lines
            plot += points
            skeleton_objects.append((lines, points))

        # --- Root trajectory projected on the floor ---
        # Animated trail: vertices [0:current_frame] show the actual past
        # path, the remaining vertices collapse to the current frame so the
        # trail "grows" as the animation plays. It lies on the scene
        # ground, and the grid's bottom face is put just under the
        # ground (below), so it is seen on that face rather than
        # floating inside the box.
        trail_objects: list[Any] = []
        trail_full_paths: list[npt.NDArray[np.float32]] = []
        for s, view in enumerate(scene.views):
            root_path = viewport.ground_path(
                view.coords[:, 0, :]).astype(np.float32)  # (F, 3)
            trail_full_paths.append(root_path)

            # Initial trail: all vertices collapsed at frame 0
            initial = np.tile(root_path[0], (num_frames, 1)).astype(np.float32)

            r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
            color = int(r) << 16 | int(g) << 8 | int(b)
            trail = k3d.line(
                initial,
                color=color,
                width=TRAIL_WIDTH_FRACTION * view.body_size,
                opacity=0.6,
                shader='thick',
                name=f"Trajectory {s}",
            )
            plot += trail
            trail_objects.append(trail)

    floor = _build_floor(viewport, style)
    if floor is not None:
        plot += floor

    # The grid covers the full motion extent, from the ground up.
    grid_min, grid_max = viewport.grounded_box()
    plot.grid = [
        float(grid_min[0]), float(grid_min[1]), float(grid_min[2]),
        float(grid_max[0]), float(grid_max[1]), float(grid_max[2]),
    ]
    plot.grid_auto_fit = False
    plot.camera_auto_fit = False

    # The viewport's camera, fitted to the height alone (see
    # play_k3d). The view angle is read from the plot, not written
    # down, so the fit follows k3d's. k3d's camera is a 9-element list:
    # [eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z]
    eye, target, up = viewport.camera(view_angle=plot.camera_fov)
    plot.camera = [float(value) for value in (*eye, *target, *up)]

    return _Plot(plot, viewport, coords_f32, skeleton_objects,
                 trail_objects, trail_full_paths, floor)


def _build_floor(viewport: Viewport, style: Style) -> Any | None:
    """The viewport's ground plane as a k3d object, or ``None`` when
    the style draws no floor.

    ``"solid"`` is a quad; ``"grid"`` is lines, and so is
    ``"checker"``, which this backend does not draw (the vedo viewer
    falls back the same way). The plane sits ``FLOOR_EPSILON``
    half-spans below the ground, under the trail.
    """
    if style.floor is None:
        return None
    import k3d

    corners = viewport.floor_quad()
    corners[:, viewport.up_index] = viewport.below_floor(
        FLOOR_EPSILON * viewport.half_span)
    palette = floor_palette(style)

    if style.floor == "solid":
        return k3d.mesh(
            corners.astype(np.float32),
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
            color=_packed(rgb255(palette["face"])), opacity=style.floor_alpha,
            side='double', flat_shading=True, name="Floor")

    # Lines across the quad in both ground directions: from one edge
    # to the opposite one, at equal steps.
    steps = np.linspace(0.0, 1.0, FLOOR_GRID_LINES)[:, np.newaxis]
    along_first = (corners[0] + steps * (corners[1] - corners[0]),
                   corners[3] + steps * (corners[2] - corners[3]))
    along_second = (corners[0] + steps * (corners[3] - corners[0]),
                    corners[1] + steps * (corners[2] - corners[1]))
    starts = np.concatenate([along_first[0], along_second[0]])
    ends = np.concatenate([along_first[1], along_second[1]])
    vertices = np.concatenate([starts, ends]).astype(np.float32)
    count = len(starts)
    indices = np.stack(
        [np.arange(count), np.arange(count) + count], axis=1)
    return k3d.lines(
        vertices, indices.astype(np.uint32), indices_type='segment',
        color=_packed(rgb255(palette["grid"])),
        width=0.004 * viewport.half_span, opacity=0.8, name="Floor")


def play_k3d(
    scene: Scene,
    style: Style,
    fps: float,
) -> None:
    """Interactive skeleton playback in a Jupyter notebook via k3d.

    A single-scene backend: all skeletons share one camera (taken from
    the first view) and one viewport over the — possibly laterally
    spread — views.

    Camera. The viewport's camera (:meth:`~._viewport.Viewport.camera`)
    at the distance fitted to k3d's vertical view angle, read from the
    plot's ``camera_fov`` (60 degrees by default): every joint and end
    site of every frame lands inside ``FIT_FRACTION`` of the widget's
    height. The width is not fitted (``aspect=None``): the notebook
    sets the widget's width, which k3d does not report to Python. The
    alternative, fitting an assumed aspect ratio as the vedo renderer
    fits its resolution's, would stand the camera further back for a
    scene wider than that ratio and waste the height of a wider
    widget; the two agree whenever the scene is tighter vertically at
    the widget's width, as a single bundled clip is at any width
    beyond about 0.7 of the height. Several skeletons spread side by
    side fill less of the height, since the eye stands no nearer than
    the corner of the viewport's cube, which their spread widens (see
    :meth:`~._viewport.Viewport.eye_distance`), and can reach past the
    sides of a narrow widget (two standing figures need about 1.5
    times its height); the mouse wheel zooms out.

    Style application (look fields): background, bone width, the
    floor (``"checker"`` draws as a grid), and chain colors for
    single-skeleton sessions (per-vertex colors — segments blend at
    chain boundaries, a k3d rendering artifact). k3d's own grid box is
    kept as the toolkit's frame, with its bottom face just under the
    ground and its lines and labels colored from the background (see
    :class:`~pybvh.bvhplot.Style`).

    Sizes. What is drawn on a skeleton is sized from that skeleton's
    own body (:attr:`~._scene.SkeletonView.body_size`), never from the
    viewport, whose cube grows with the distance a clip travels: the
    bones' line width is ``BONE_WIDTH_FRACTION`` of the body size,
    scaled by ``bone_width`` (:func:`~._style.bone_width_scale`: the
    paper default is the 1:1 anchor), the joints' point size
    ``JOINT_SIZE_FRACTION`` and the root trail's width
    ``TRAIL_WIDTH_FRACTION``, the v0.9.0 sizes of a standing still. A
    still and the whole clip draw a body at the same proportions, and
    each skeleton of a scene gets its own. The sizes are scene lengths
    because that is what k3d takes here, with the shaders this backend
    asks for: its ``thick`` line shader draws
    ``width`` in scene units to within a few percent (``1.8 *
    tan(fov / 2)`` times it, 1.04 at k3d's default 60-degree field of
    view) and never thinner than about one pixel, and its
    ``3dSpecular`` points take ``point_size`` as a ball's diameter in
    the scene. They therefore follow the zoom, as the body does. The
    alternative, fixed pixel widths as the vedo viewer's fast mode and
    OpenCV draw, would keep a body's lines as wide when the camera is
    far from it as when it is close; the two agree only at one camera
    distance. The floor's grid lines belong to the floor and stay a
    fraction of the viewport's half-span.

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
    from IPython.display import display  # type: ignore[import-untyped]
    from ipywidgets import Play, IntSlider, jslink, HBox, VBox, Label  # type: ignore[import-untyped]

    built = _build_plot(scene, style)
    plot = built.plot
    coords_f32 = built.coords
    skeleton_objects = built.skeletons
    trail_objects = built.trails
    trail_full_paths = built.trail_paths
    num_frames = coords_f32[0].shape[0]

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
