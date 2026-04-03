"""vedo interactive backend for desktop viewers.

Provides interactive 3D skeleton playback with camera rotation/zoom,
play/pause, and frame scrubbing in a desktop window.

Requires ``vedo >= 2023.5``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._common import PALETTE_RGB

if TYPE_CHECKING:
    from ..bvh import Bvh


def play_vedo(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
) -> object:
    """Interactive skeleton playback in a desktop window via vedo.

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
    plt : vedo.Plotter
        The vedo plotter instance.
    """
    from vedo import Plotter, Lines, Points, Text2D  # type: ignore[import-untyped]

    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)

    plt = Plotter(title="pybvh skeleton viewer", size=(1200, 800))

    # Build initial geometry for each skeleton
    bone_actors: list[Lines] = []
    point_actors: list[Points] = []

    for s, (coords, bones) in enumerate(
            zip(coords_list, skeleton_lines_list)):
        frame0 = coords[0]

        start_pts = frame0[[b[0] for b in bones]]
        end_pts = frame0[[b[1] for b in bones]]

        r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
        color_str = f"rgb({r},{g},{b})"

        lines_actor = Lines(start_pts, end_pts, lw=4, c=color_str)
        points_actor = Points(frame0, r=6, c=color_str)

        plt += lines_actor
        plt += points_actor
        bone_actors.append(lines_actor)
        point_actors.append(points_actor)

        if labels and s < len(labels):
            label = Text2D(
                labels[s],
                pos=(0.02 + s * 0.15, 0.95),
                c=color_str,
                s=1.2,
            )
            plt += label

    # Frame counter display
    frame_text = Text2D(f"Frame 0/{num_frames - 1}", pos=(0.75, 0.02), s=0.8)
    plt += frame_text

    # Instructions
    help_text = Text2D(
        "Space: play/pause | Left/Right: step | +/-: speed",
        pos=(0.02, 0.02),
        s=0.6,
        c='gray',
    )
    plt += help_text

    # Animation state
    state = {
        'frame': 0,
        'playing': True,
        'interval': int(1000.0 / fps),
        'timer_id': None,
    }

    def update_frame(f: int) -> None:
        """Update all skeleton geometries to frame f."""
        for s, (bones, lines_actor, points_actor) in enumerate(
                zip(skeleton_lines_list, bone_actors, point_actors)):
            frame_data = coords_list[s][f]
            start_pts = frame_data[[b[0] for b in bones]]
            end_pts = frame_data[[b[1] for b in bones]]
            lines_actor.points(np.vstack([start_pts, end_pts]))
            points_actor.points(frame_data)

        frame_text.text(f"Frame {f}/{num_frames - 1}")
        plt.render()

    def timer_callback(event: object) -> None:
        if not state['playing']:
            return
        state['frame'] = (state['frame'] + 1) % num_frames
        update_frame(state['frame'])

    def key_callback(event: object) -> None:
        key = plt.last_event.keypress  # type: ignore[attr-defined]
        if key == 'space':
            state['playing'] = not state['playing']
        elif key == 'Right':
            state['playing'] = False
            state['frame'] = min(state['frame'] + 1, num_frames - 1)
            update_frame(state['frame'])
        elif key == 'Left':
            state['playing'] = False
            state['frame'] = max(state['frame'] - 1, 0)
            update_frame(state['frame'])
        elif key == 'plus' or key == 'equal':
            state['interval'] = max(state['interval'] // 2, 8)
            if state['timer_id'] is not None:
                plt.timer_callback('destroy', state['timer_id'])
            state['timer_id'] = plt.timer_callback(
                'create', dt=state['interval'])
        elif key == 'minus':
            state['interval'] = min(state['interval'] * 2, 2000)
            if state['timer_id'] is not None:
                plt.timer_callback('destroy', state['timer_id'])
            state['timer_id'] = plt.timer_callback(
                'create', dt=state['interval'])

    plt.add_callback('timer', timer_callback)
    plt.add_callback('key press', key_callback)
    state['timer_id'] = plt.timer_callback('create', dt=state['interval'])

    plt.show()
    return plt
