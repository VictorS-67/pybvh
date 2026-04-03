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

from ._common import PALETTE_RGB

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

    # Set grid and camera to cover the full motion extent
    grid_min = center - half_span
    grid_max = center + half_span
    plot.grid = [
        float(grid_min[0]), float(grid_min[1]), float(grid_min[2]),
        float(grid_max[0]), float(grid_max[1]), float(grid_max[2]),
    ]
    plot.grid_auto_fit = False
    plot.camera_auto_fit = False

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

    slider.observe(on_frame_change, names='value')

    controls = HBox([play_widget, slider, frame_label])
    display(VBox([plot, controls]))

    # Return None — the plot is already displayed above.
    # Returning the k3d.Plot would cause Jupyter to auto-display it
    # a second time with different axis ranges.
    return None
