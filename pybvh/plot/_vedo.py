"""vedo interactive backend for desktop viewers.

Provides interactive 3D skeleton playback with camera rotation/zoom,
playback controls, and frame scrubbing in a desktop window.

Two quality modes:

- ``"high"`` (default): 3D tapered tubes for bones, spheres for joints,
  floor grid, and VTK lighting. Professional look.
- ``"fast"``: Flat lines and points. Maximum performance for large files.

Requires ``vedo >= 2024.5``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._common import PALETTE_RGB

if TYPE_CHECKING:
    from ..bvh import Bvh

# Rich gold for single-skeleton "high" mode (aitviewer-inspired)
_WARM_AMBER = (230, 175, 50)
_UP_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _interleave(
    starts: npt.NDArray[np.float64],
    ends: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interleave start/end points as [s0, e0, s1, e1, ...]."""
    n = len(starts)
    out = np.empty((2 * n, 3), dtype=starts.dtype)
    out[0::2] = starts
    out[1::2] = ends
    return out


def play_vedo(
    bvh_list: list[Bvh],
    coords_list: list[npt.NDArray[np.float64]],
    fps: float,
    labels: list[str] | None,
    skeleton_lines_list: list[list[tuple[int, int]]],
    center: npt.NDArray[np.float64],
    half_span: float,
    *,
    up_axis: str = "y",
    azimuth: float = -20.0,
    elevation: float = 20.0,
    quality: str = "high",
) -> None:
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
    up_axis : str
        ``'x'``, ``'y'``, or ``'z'``.
    azimuth : float
        Azimuth angle in degrees (same convention as matplotlib).
    elevation : float
        Elevation angle in degrees (same convention as matplotlib).
    quality : str
        ``"high"`` for 3D geometry, ``"fast"`` for flat wireframe.
    """
    from vedo import (  # type: ignore[import-untyped]
        Plotter, Lines, Points, Tube, Sphere, Grid, Text2D,
    )

    num_frames = coords_list[0].shape[0]
    n_skeletons = len(bvh_list)
    use_high = quality == "high"

    # Sizing — base radius, then adapted per-bone by length
    r_bone_base = half_span * 0.013

    # Precompute adaptive radii: scale each bone's radius proportional
    # to its length relative to the median.  Short finger bones get thin
    # tubes; long limb bones stay thick.
    _bone_radii: list[dict[tuple[int, int], float]] = []
    _joint_radii: list[npt.NDArray[np.float64]] = []
    for _s in range(n_skeletons):
        _frame0 = coords_list[_s][0]
        _bones = skeleton_lines_list[_s]
        _lengths = {
            (p, c): float(np.linalg.norm(_frame0[c] - _frame0[p]))
            for p, c in _bones
        }
        _med = float(np.median(list(_lengths.values()))) if _lengths else 1.0
        _br: dict[tuple[int, int], float] = {}
        for (p, c), length in _lengths.items():
            ratio = np.clip(length / _med, 0.3, 2.0) if _med > 0 else 1.0
            _br[(p, c)] = r_bone_base * ratio
        # Joint radius = mean of connected bone radii
        _jr = np.full(len(_frame0), r_bone_base * 0.5)
        _conn: list[list[float]] = [[] for _ in range(len(_frame0))]
        for (p, c), rad in _br.items():
            _conn[p].append(rad)
            _conn[c].append(rad)
        for _j in range(len(_frame0)):
            if _conn[_j]:
                _jr[_j] = float(np.mean(_conn[_j]))
        _bone_radii.append(_br)
        _joint_radii.append(_jr)

    # Colors
    def _color(s: int) -> str:
        if n_skeletons == 1 and use_high:
            r, g, b = _WARM_AMBER
        else:
            r, g, b = PALETTE_RGB[s % len(PALETTE_RGB)]
        return f"rgb({r},{g},{b})"

    # --- Plotter setup ---
    plt = Plotter(
        title="pybvh skeleton viewer",
        size=(1400, 900),
        bg='#d4d4dc',        # soft cool gray at bottom (behind controls)
        bg2='lightskyblue',  # sky at top
    )

    # --- Floor grid (high quality only) ---
    if use_high:
        up_idx = _UP_AXIS_INDEX.get(up_axis, 1)
        # Place floor at the lowest point of all skeletons across all frames
        floor_y = min(c[:, :, up_idx].min() for c in coords_list)
        floor_pos = center.copy()
        floor_pos[up_idx] = floor_y
        floor = Grid(
            pos=tuple(floor_pos),
            s=[half_span * 2.5, half_span * 2.5],
            res=(30, 30),
        )
        if up_axis == 'y':
            floor.rotate_x(90)
        elif up_axis == 'x':
            floor.rotate_y(90)
        floor.lw(1).alpha(0.6).c('#555555').lighting('off')
        plt += floor

    # --- Build initial skeleton geometry ---
    # Each skeleton stores its actors for later removal/update
    skeleton_actors: list[list] = []  # list of actor lists per skeleton

    def _build_skeleton(s: int, frame_data: npt.NDArray) -> list:
        """Create actors for skeleton s at the given frame."""
        bones = skeleton_lines_list[s]
        color = _color(s)
        actors = []

        if use_high:
            br = _bone_radii[s]
            jr = _joint_radii[s]
            for p_idx, c_idx in bones:
                p, c = frame_data[p_idx], frame_data[c_idx]
                if np.linalg.norm(c - p) < 1e-8:
                    continue
                r = br.get((p_idx, c_idx), r_bone_base)
                tube = Tube([p, c], r=[r, r / 2],
                            res=12, c=color)
                tube.lighting('glossy')
                actors.append(tube)
            for j in range(len(frame_data)):
                sph = Sphere(pos=frame_data[j], r=jr[j],
                             res=12, c=color)
                sph.lighting('glossy')
                actors.append(sph)
        else:
            start_pts = frame_data[[b[0] for b in bones]]
            end_pts = frame_data[[b[1] for b in bones]]
            lines = Lines(start_pts, end_pts, lw=5, c=color)
            lines.lighting('off')
            points = Points(frame_data, r=8, c=color, alpha=0.9)
            actors.extend([lines, points])

        return actors

    for s in range(n_skeletons):
        actors = _build_skeleton(s, coords_list[s][0])
        for a in actors:
            plt += a
        skeleton_actors.append(actors)

    # --- Labels ---
    if labels:
        for s in range(min(len(labels), n_skeletons)):
            label = Text2D(
                labels[s],
                pos=(0.02 + s * 0.15, 0.95),
                c=_color(s), s=1.4, font='Calco',
            )
            plt += label

    # --- Camera setup (same convention as matplotlib / opencv backends) ---
    from ._common import build_view_matrix
    view_mat = build_view_matrix(azimuth, elevation, up_axis)
    # view_mat rows: [right, up, eye_direction (toward viewer)]
    eye_dir = view_mat[2]
    cam_dist = half_span * 4.0
    cam_pos = center + eye_dir * cam_dist
    cam_up = view_mat[1]

    def _set_camera() -> None:
        """Apply the initial camera position."""
        plt.camera.SetPosition(*cam_pos)
        plt.camera.SetFocalPoint(*center)
        plt.camera.SetViewUp(*cam_up)
        plt.reset_camera()

    _set_camera()

    # --- Animation state ---
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30
    state = {
        'frame': 0,
        'playing': True,
        'interval': int(1000.0 / fps),
        'timer_id': None,
        'speed': 1.0,
        'loop': True,
        '_slider_updating': False,
    }

    # =================================================================
    # UI LAYOUT
    # =================================================================
    #
    # Top-right:    Frame info (large, visible)
    # Left panel:   Speed control, loop toggle, reset camera
    # Bottom:       Slider + play/pause + step buttons
    # Bottom-left:  Keyboard shortcut help
    #

    # --- Top-right: frame info ---
    info_text = Text2D(
        f"Frame 0/{num_frames - 1} | t=0.00s",
        pos=(0.55, 0.96), s=1.3, c='#2c3e50', font='Calco',
    )
    plt += info_text

    # --- Left panel ---
    _PANEL_X = 0.01
    _PANEL_S = 1.4

    panel_title = Text2D(
        " Controls ",
        pos=(_PANEL_X, 0.92), s=_PANEL_S, c='white', bg='#2c3e50',
        font='Calco',
    )
    plt += panel_title

    speed_label = Text2D(
        "Speed:", pos=(_PANEL_X, 0.85), s=_PANEL_S,
        c='#2c3e50', font='Calco',
    )
    plt += speed_label

    speed_down_btn = Text2D(
        " < ", pos=(_PANEL_X, 0.79), s=_PANEL_S,
        c='white', bg='dodgerblue', font='Calco',
    )
    plt += speed_down_btn

    speed_text = Text2D(
        " 1.0x ", pos=(0.06, 0.79), s=_PANEL_S,
        c='#2c3e50', bg='#c8c8d4', font='Calco',
    )
    plt += speed_text

    speed_up_btn = Text2D(
        " > ", pos=(0.12, 0.79), s=_PANEL_S,
        c='white', bg='dodgerblue', font='Calco',
    )
    plt += speed_up_btn

    loop_btn = Text2D(
        " Loop: ON ", pos=(_PANEL_X, 0.72), s=_PANEL_S,
        c='white', bg='green4', font='Calco',
    )
    plt += loop_btn

    reset_btn = Text2D(
        " Reset Cam ", pos=(_PANEL_X, 0.65), s=_PANEL_S,
        c='white', bg='dodgerblue', font='Calco',
    )
    plt += reset_btn

    # --- Bottom: transport bar ────────────────────────────────────────────
    # Single source of truth: _SL_X0/_SL_X1 drive both the slider and the
    # button layout.  Change them and everything stays aligned automatically.
    _SL_X0, _SL_X1 = 0.15, 0.85   # slider / button-row x extents
    _BTN_Y   = 0.04                # button row baseline y
    _BTN_S   = 1.8                 # large, comfortable button text
    _BTN_GAP = 0.010               # normalized gap between adjacent buttons
    _N_BTNS  = 5
    # Divide the full slider span evenly: N equal cells separated by (N-1) gaps
    _BTN_W   = (_SL_X1 - _SL_X0 - (_N_BTNS - 1) * _BTN_GAP) / _N_BTNS
    _BTN_X   = [_SL_X0 + i * (_BTN_W + _BTN_GAP) for i in range(_N_BTNS)]
    # Cell centers for visually centering buttons (justify='bottom-center').
    # Hit regions still use _BTN_X (cell left edges).
    _BTN_XC  = [x + _BTN_W / 2 for x in _BTN_X]
    # Hit band y: generous lower bound (0.01) accounts for the ~0.018 systematic
    # offset between viewport y and GetEventPosition() y observed in practice.
    # Upper bound (0.08) stays safely below the slider baseline at 0.10.
    _BTN_HY0 = 0.01
    _BTN_HY1 = 0.08

    # ASCII text labels — Calco font lacks Unicode transport symbols (◀▶⏸).
    # All labels are 9 chars (padded) for generous sizing and consistent
    # background widths when play/pause toggles.
    _L_FIRST = "  Start  "   # skip to start
    _L_BACK  = "  Prev   "   # step back
    _L_PAUSE = "  Pause  "   # shown while playing
    _L_PLAY  = "  Play   "   # shown while paused
    _L_FWD   = "  Next   "   # step forward
    _L_LAST  = "   End   "   # skip to end

    _jst = 'bottom-center'
    btn_first = Text2D(_L_FIRST, pos=(_BTN_XC[0], _BTN_Y), s=_BTN_S,
                       c='white', bg='dodgerblue', font='Calco', justify=_jst)
    btn_back  = Text2D(_L_BACK,  pos=(_BTN_XC[1], _BTN_Y), s=_BTN_S,
                       c='white', bg='dodgerblue', font='Calco', justify=_jst)
    btn_play  = Text2D(_L_PAUSE, pos=(_BTN_XC[2], _BTN_Y), s=_BTN_S,
                       c='white', bg='tomato', font='Calco', justify=_jst)
    btn_fwd   = Text2D(_L_FWD,   pos=(_BTN_XC[3], _BTN_Y), s=_BTN_S,
                       c='white', bg='dodgerblue', font='Calco', justify=_jst)
    btn_last  = Text2D(_L_LAST,  pos=(_BTN_XC[4], _BTN_Y), s=_BTN_S,
                       c='white', bg='dodgerblue', font='Calco', justify=_jst)

    for btn in (btn_first, btn_back, btn_play, btn_fwd, btn_last):
        plt += btn

    # --- Bottom-left: keyboard shortcuts (below button row) ---
    help_text = Text2D(
        "Keys: Space  +/-  Left/Right  L  R  Home  End",
        pos=(0.02, 0.01), s=0.8, c='#666666', font='Calco',
    )
    plt += help_text

    # --- Frame scrubber slider ---
    def slider_callback(widget, event):
        if state['_slider_updating']:
            return
        f = int(round(widget.value))
        f = max(0, min(f, num_frames - 1))
        state['frame'] = f
        state['playing'] = False
        _sync_all()
        _update_frame_display(f)

    slider = plt.add_slider(
        slider_callback,
        xmin=0, xmax=num_frames - 1,
        value=0,
        pos=[(_SL_X0, 0.12), (_SL_X1, 0.12)],   # matches button row extents
        title='',
        show_value=False,
    )

    # =================================================================
    # UI SYNC HELPERS
    # =================================================================

    def _sync_all() -> None:
        """Sync all UI elements to match current state."""
        # Play/pause button
        if state['playing']:
            btn_play.text(_L_PAUSE)
            btn_play.background('tomato')
        else:
            btn_play.text(_L_PLAY)
            btn_play.background('green4')
        # Loop button
        if state['loop']:
            loop_btn.text(" Loop: ON ")
            loop_btn.background('green4')
        else:
            loop_btn.text(" Loop: OFF ")
            loop_btn.background('gray')
        # Speed display
        spd = state['speed']
        speed_text.text(f" {spd:.1f}x " if spd != int(spd) else f" {int(spd)}x ")

    def _update_frame_display(f: int) -> None:
        """Update frame info text."""
        t = f * frame_time
        info_text.text(f"Frame {f}/{num_frames - 1} | t={t:.2f}s")

    def _set_speed(new_speed: float) -> None:
        """Change playback speed and restart timer.

        For speeds < 1x the timer interval is stretched (fewer ticks).
        For speeds >= 1x the timer fires at the base rate and the
        callback skips frames to achieve the target speed.
        """
        state['speed'] = new_speed
        base_interval = max(int(1000.0 / fps), 8)
        if new_speed < 1.0:
            state['interval'] = max(int(base_interval / new_speed), 8)
        else:
            state['interval'] = base_interval
        _sync_all()
        if state['timer_id'] is not None:
            plt.timer_callback('destroy', state['timer_id'])
        state['timer_id'] = plt.timer_callback(
            'create', dt=state['interval'])

    def _jump_to(f: int) -> None:
        """Jump to frame f, pause, and sync UI."""
        state['playing'] = False
        state['frame'] = f
        state['_slider_updating'] = True
        slider.value = f
        state['_slider_updating'] = False
        _sync_all()
        _update_frame_display(f)

    # =================================================================
    # FRAME UPDATE
    # =================================================================

    def update_frame(f: int) -> None:
        for s in range(n_skeletons):
            for a in skeleton_actors[s]:
                plt.remove(a)
            new_actors = _build_skeleton(s, coords_list[s][f])
            for a in new_actors:
                plt.add(a)
            skeleton_actors[s] = new_actors
        _update_frame_display(f)
        plt.render()

    # =================================================================
    # CLICK HANDLER (manual hit-testing)
    # =================================================================

    def _on_click(event: object) -> None:
        # Use raw interactor position (actual cursor) rather than picked2d,
        # which sticks to the last-picked 2D actor position after any click.
        x, y = plt.interactor.GetEventPosition()
        w, h = plt.window.GetSize()
        nx, ny = x / w, y / h
        # --- Bottom transport buttons ---
        # Hit region = the computed cell for each button: x in [_BTN_X[i], _BTN_X[i]+_BTN_W]
        # y band covers the button baseline to approx top of font at _BTN_S.
        if _BTN_HY0 < ny < _BTN_HY1:
            for i, x0 in enumerate(_BTN_X):
                if x0 < nx < x0 + _BTN_W:
                    if i == 0:
                        _jump_to(0)
                        update_frame(0)
                    elif i == 1:
                        _jump_to(max(state['frame'] - 1, 0))
                        update_frame(state['frame'])
                    elif i == 2:
                        state['playing'] = not state['playing']
                        _sync_all()
                    elif i == 3:
                        _jump_to(min(state['frame'] + 1, num_frames - 1))
                        update_frame(state['frame'])
                    elif i == 4:
                        _jump_to(num_frames - 1)
                        update_frame(num_frames - 1)
                    plt.render()
                    return

        # --- Left panel buttons ---
        # speed_down at (0.01, 0.79), speed_up at (0.12, 0.79),
        # loop at (0.01, 0.72), reset at (0.01, 0.65)
        if nx < 0.20:
            if 0.76 < ny < 0.83:
                # Speed down: 0.01–0.05, speed up: 0.12–0.16
                if 0.01 < nx < 0.05:
                    _set_speed(max(state['speed'] / 2, 0.125))
                    plt.render()
                elif 0.12 < nx < 0.16:
                    _set_speed(min(state['speed'] * 2, 16.0))
                    plt.render()
            elif 0.69 < ny < 0.76:
                # Loop toggle
                state['loop'] = not state['loop']
                _sync_all()
                plt.render()
            elif 0.62 < ny < 0.69:
                # Reset camera to initial view
                _set_camera()
                plt.render()

    plt.add_callback('LeftButtonPress', _on_click)

    # =================================================================
    # TIMER CALLBACK
    # =================================================================

    def timer_callback(event: object) -> None:
        if not state['playing']:
            return
        step = max(1, round(state['speed']))
        next_f = state['frame'] + step
        if next_f >= num_frames:
            if state['loop']:
                next_f = 0
            else:
                state['playing'] = False
                _sync_all()
                return
        state['frame'] = next_f
        state['_slider_updating'] = True
        slider.value = next_f
        state['_slider_updating'] = False
        update_frame(next_f)

    # =================================================================
    # KEYBOARD CALLBACK
    # =================================================================

    def key_callback(event: object) -> None:
        key = plt.last_event.keypress  # type: ignore[attr-defined]

        if key == 'space':
            state['playing'] = not state['playing']
            _sync_all()
            plt.render()

        elif key == 'Right':
            _jump_to(min(state['frame'] + 1, num_frames - 1))
            update_frame(state['frame'])

        elif key == 'Left':
            _jump_to(max(state['frame'] - 1, 0))
            update_frame(state['frame'])

        elif key in ('plus', 'equal'):
            _set_speed(min(state['speed'] * 2, 16.0))
            plt.render()

        elif key == 'minus':
            _set_speed(max(state['speed'] / 2, 0.125))
            plt.render()

        elif key == 'l':
            state['loop'] = not state['loop']
            _sync_all()
            plt.render()

        elif key == 'r':
            _set_camera()
            plt.render()

        elif key == 'Home':
            _jump_to(0)
            update_frame(0)

        elif key == 'End':
            _jump_to(num_frames - 1)
            update_frame(num_frames - 1)

    # --- Register callbacks and start ---
    plt.add_callback('timer', timer_callback)
    plt.add_callback('key press', key_callback)
    state['timer_id'] = plt.timer_callback('create', dt=state['interval'])

    plt.show()
    return None
