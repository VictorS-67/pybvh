"""vedo interactive backend for desktop viewers.

Provides interactive 3D skeleton playback with camera rotation/zoom,
playback controls, and frame scrubbing in a desktop window.

Two quality modes:

- ``"high"`` (default): 3D tapered tubes for bones, spheres for joints,
  floor grid, flat ambient lighting.
- ``"fast"``: Flat lines and points. Maximum performance for large files.

Requires ``vedo >= 2024.5``.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING

from ._common import PALETTE_RGB, build_view_matrix

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
        Plotter, Lines, Points, Tube, Sphere, Grid, Text2D, merge,
    )
    import vtk  # type: ignore[import-untyped]

    num_frames = coords_list[0].shape[0]
    if num_frames < 1:
        return
    n_skeletons = len(bvh_list)
    use_high = quality == "high"

    # Sizing — base radius, then adapted per-bone by length
    r_bone_base = half_span * 0.013

    # Precompute adaptive radii: scale each bone's radius proportional
    # to its length relative to the median.  Short finger bones get thin
    # tubes; long limb bones stay thick.
    _bone_radii: list[dict[tuple[int, int], float]] = []
    _joint_radii: list[npt.NDArray[np.float64]] = []
    for s in range(n_skeletons):
        frame0 = coords_list[s][0]
        bones = skeleton_lines_list[s]
        lengths = {
            (p, c): float(np.linalg.norm(frame0[c] - frame0[p]))
            for p, c in bones
        }
        med = float(np.median(list(lengths.values()))) if lengths else 1.0
        br: dict[tuple[int, int], float] = {}
        for (p, c), length in lengths.items():
            ratio = np.clip(length / med, 0.3, 2.0) if med > 0 else 1.0
            br[(p, c)] = r_bone_base * ratio
        # Joint radius = mean of connected bone radii
        jr = np.full(len(frame0), r_bone_base * 0.5)
        conn: list[list[float]] = [[] for _ in range(len(frame0))]
        for (p, c), rad in br.items():
            conn[p].append(rad)
            conn[c].append(rad)
        for j in range(len(frame0)):
            if conn[j]:
                jr[j] = float(np.mean(conn[j]))
        _bone_radii.append(br)
        _joint_radii.append(jr)

    # Colors
    def _color_rgb(s: int) -> tuple[int, int, int]:
        if n_skeletons == 1 and use_high:
            return _WARM_AMBER
        return PALETTE_RGB[s % len(PALETTE_RGB)]

    def _color(s: int) -> str:
        r, g, b = _color_rgb(s)
        return f"rgb({r},{g},{b})"

    def _apply_flat_lighting(mesh: object) -> None:
        """Set flat ambient-only lighting so color is stable across frames.

        Keeps VTK's default scalar coloring (blue-to-green gradient on
        tubes) but removes normal-dependent shading that shifts as bones
        rotate.
        """
        prop = mesh.actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)

    # --- Plotter setup ---
    plt = Plotter(
        title="pybvh viewer",
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
        # up_axis='z': Grid defaults to XY plane, no rotation needed
        floor.lw(1).alpha(0.6).c('#555555').lighting('off')
        plt += floor

    # --- Build persistent skeleton geometry (created once, updated in-place) ---

    # High mode: 2 merged meshes per skeleton (all bones + all joints)
    _bones_mesh: list = []              # merged Mesh per skeleton
    _joints_mesh: list = []             # merged Mesh per skeleton
    _canonical_bone_verts: list = []    # ndarray (n_bones, V_bone, 3)
    _canonical_joint_verts: list = []   # ndarray (n_joints, V_joint, 3)
    # Pre-computed bone index arrays for vectorized frame lookups
    _bone_parent_idx: list = []         # ndarray (n_bones,) per skeleton
    _bone_child_idx: list = []          # ndarray (n_bones,) per skeleton

    # Fast mode: Lines + Points per skeleton
    _lines_actors: list = []
    _points_actors: list = []

    def _update_skeleton_high(s: int, frame_data: npt.NDArray) -> None:
        """Update merged bone and joint meshes via vectorized numpy."""
        p_idx = _bone_parent_idx[s]
        c_idx = _bone_child_idx[s]
        canonical_bones = _canonical_bone_verts[s]

        if len(p_idx) > 0 and _bones_mesh[s] is not None:
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
                np.einsum('bij,bvj->bvi', rotscale, canonical_bones)
                + starts[:, np.newaxis, :])

            # Collapse zero-length bones (degenerate triangles)
            zero = np.where(lengths < 1e-8)[0]
            if len(zero):
                for zi in zero:
                    transformed[zi] = starts[zi]

            _bones_mesh[s].vertices = transformed.reshape(-1, 3)

        # Joints: vectorized translation (single operation)
        canonical_joints = _canonical_joint_verts[s]
        _joints_mesh[s].vertices = (
            canonical_joints + frame_data[:, np.newaxis, :]
        ).reshape(-1, 3)

    def _update_skeleton_fast(s: int, frame_data: npt.NDArray) -> None:
        """Update Lines/Points vertex data in-place for skeleton *s*."""
        p_idx = _bone_parent_idx[s]
        c_idx = _bone_child_idx[s]
        _lines_actors[s].vertices = _interleave(
            frame_data[p_idx], frame_data[c_idx])
        _points_actors[s].vertices = frame_data

    # --- Create actors once and position to frame 0 ---
    for s in range(n_skeletons):
        color = _color(s)
        bones = skeleton_lines_list[s]
        _bone_parent_idx.append(np.array([b[0] for b in bones]))
        _bone_child_idx.append(np.array([b[1] for b in bones]))

        if use_high:
            br = _bone_radii[s]
            jr = _joint_radii[s]

            # Create canonical bone tubes and collect vertices
            bone_meshes = []
            bone_verts_list = []
            for p_i, c_i in bones:
                r = br.get((p_i, c_i), r_bone_base)
                tube = Tube([[0, 0, 0], [0, 0, 1]], r=[r, r / 2],
                            res=12, c=color)
                bone_verts_list.append(tube.vertices.copy())
                bone_meshes.append(tube)

            if bone_meshes:
                bones_merged = merge(bone_meshes)
                _apply_flat_lighting(bones_merged)
                _canonical_bone_verts.append(
                    np.array(bone_verts_list))  # (n_bones, V, 3)
            else:
                bones_merged = None
                _canonical_bone_verts.append(np.empty((0, 0, 3)))
            _bones_mesh.append(bones_merged)

            # Create canonical joint spheres and collect vertices
            joint_meshes = []
            joint_verts_list = []
            for j in range(coords_list[s].shape[1]):
                sph = Sphere(pos=(0, 0, 0), r=jr[j], res=12, c=color)
                joint_verts_list.append(sph.vertices.copy())
                joint_meshes.append(sph)
            joints_merged = merge(joint_meshes)
            _apply_flat_lighting(joints_merged)
            _canonical_joint_verts.append(
                np.array(joint_verts_list))   # (n_joints, V, 3)
            _joints_mesh.append(joints_merged)

            # Add only 2 merged actors to plotter
            if bones_merged is not None:
                plt += bones_merged
            plt += joints_merged

            # Position to frame 0
            _update_skeleton_high(s, coords_list[s][0])
        else:
            frame0 = coords_list[s][0]
            _lw = max(1, int(half_span * 0.04))
            _pr = max(1, int(half_span * 0.05))
            lines = Lines(
                frame0[_bone_parent_idx[s]],
                frame0[_bone_child_idx[s]],
                lw=_lw, c=color)
            lines.lighting('off')
            points = Points(frame0, r=_pr, c=color, alpha=0.9)
            _lines_actors.append(lines)
            _points_actors.append(points)
            plt += lines
            plt += points

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

    # --- Joint name labels (toggle with J key) ---
    # Use vtkBillboardTextActor3D so labels always face the camera.
    _label_actors: list[list] = []   # _label_actors[s][j] = vtkBillboardTextActor3D
    _label_offset = np.array([0, half_span * 0.02, 0])
    _label_fontsize = max(12, int(half_span * 0.4))
    for s in range(n_skeletons):
        lbl_list: list = []
        joint_names = [node.name for node in bvh_list[s].nodes]
        for j, name in enumerate(joint_names):
            pos0 = coords_list[s][0][j] + _label_offset
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetInput(name)
            actor.SetPosition(*pos0)
            actor.GetTextProperty().SetFontSize(_label_fontsize)
            actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.3)
            actor.GetTextProperty().SetBackgroundOpacity(0.7)
            actor.GetTextProperty().SetJustificationToCentered()
            actor.SetVisibility(0)
            lbl_list.append(actor)
            plt.renderer.AddActor(actor)
        _label_actors.append(lbl_list)

    # Keep full-rate data for FPS resampling and ghost mode
    _coords_full = [c.copy() for c in coords_list]

    # --- Ghost / onion-skin actors (toggle with N key) ---
    _GHOST_LEVELS = [0, 3, 5]    # number of ghost frames: off, 3, 5
    _ghost_lines: list[list] = []   # _ghost_lines[s] = list of Lines actors
    _ghost_points: list[list] = []  # _ghost_points[s] = list of Points actors
    _MAX_GHOSTS = max(_GHOST_LEVELS)
    _ghost_last_alpha: list[list[float]] = []  # cached alpha per ghost actor
    for s in range(n_skeletons):
        color = _color(s)
        g_lines: list = []
        g_points: list = []
        frame0 = _coords_full[s][0]
        start_pts = frame0[_bone_parent_idx[s]]
        end_pts = frame0[_bone_child_idx[s]]
        _lw_g = max(1, int(half_span * 0.02))
        _pr_g = max(1, int(half_span * 0.025))
        for _gi in range(_MAX_GHOSTS):
            gl = Lines(start_pts, end_pts, lw=_lw_g, c=color, alpha=0.15)
            gl.lighting('off')
            gl.actor.SetVisibility(0)
            gp = Points(frame0, r=_pr_g, c=color, alpha=0.15)
            gp.lighting('off')
            gp.actor.SetVisibility(0)
            g_lines.append(gl)
            g_points.append(gp)
            plt += gl
            plt += gp
        _ghost_lines.append(g_lines)
        _ghost_points.append(g_points)
        _ghost_last_alpha.append([-1.0] * _MAX_GHOSTS)

    # --- Root trajectory trail (toggle with T key) ---
    # Pre-compute full root path; pre-allocate Lines with collapsed segments.
    # Each frame, expand segments up to current frame (fast vertex update).
    _trail_actors: list = []
    _trail_full: list[npt.NDArray] = []       # pre-computed root paths
    _trail_collapsed: list[npt.NDArray] = []  # pre-allocated collapsed buffers
    _floor_y = min(c[:, :, _UP_AXIS_INDEX.get(up_axis, 1)].min()
                   for c in _coords_full) if use_high else 0.0
    for s in range(n_skeletons):
        root_all = _coords_full[s][:, 0, :].copy()  # (F, 3)
        root_all[:, _UP_AXIS_INDEX.get(up_axis, 1)] = _floor_y
        _trail_full.append(root_all)
        # Pre-allocate collapsed buffer (reused every frame via .copy())
        collapsed = np.tile(root_all[0], (2 * (len(root_all) - 1), 1))
        _trail_collapsed.append(collapsed)
        trail = Lines(collapsed[::2], collapsed[1::2],
                      lw=2, c=_color(s), alpha=0.6)
        trail.lighting('off')
        trail.actor.SetVisibility(0)
        _trail_actors.append(trail)
        plt += trail

    # --- Animation state ---
    state = {
        'frame': 0,
        'playing': True,
        'interval': max(int(1000.0 / max(fps, 1)), 8),
        'timer_id': None,
        'speed': 1.0,
        'loop': True,
        '_slider_updating': False,
        'show_labels': False,
        'skeleton_visible': [True] * n_skeletons,
        'ghost_level_idx': 0,     # index into _GHOST_LEVELS
        'show_trail': False,
        'ghost_history': deque(maxlen=_MAX_GHOSTS * 5 + 1),
        '_rendering': False,
        '_play_start_time': None,
        '_play_start_frame': 0,
    }

    # =================================================================
    # UI LAYOUT
    # =================================================================
    #
    # Title bar:    Frame/time/fps/speed info
    # Left panel:   Speed, FPS, loop toggle, reset camera
    # Bottom:       Slider + transport buttons (Start/Prev/Play/Next/End)
    # Bottom-left:  Help hint (press H for full overlay)
    #

    # Frame info is shown in the window title bar (not a 2D overlay)

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

    # --- FPS selector ---
    _native_fps = fps
    _fps_presets = [15, 30, 60, 120, int(round(_native_fps))]
    # De-duplicate and sort, ensure native fps is included
    _fps_presets = sorted(set(_fps_presets))
    # Default to 30fps if native is higher (comfortable playback),
    # otherwise use native fps.
    _default_fps = 30 if _native_fps > 30 else _native_fps
    _fps_idx = _fps_presets.index(
        min(_fps_presets, key=lambda x: abs(x - _default_fps)))

    fps_label = Text2D(
        "FPS:", pos=(_PANEL_X, 0.58), s=_PANEL_S,
        c='#2c3e50', font='Calco',
    )
    plt += fps_label

    fps_down_btn = Text2D(
        " < ", pos=(_PANEL_X, 0.52), s=_PANEL_S,
        c='white', bg='dodgerblue', font='Calco',
    )
    plt += fps_down_btn

    fps_text = Text2D(
        f" {_fps_presets[_fps_idx]} ", pos=(0.06, 0.52), s=_PANEL_S,
        c='#2c3e50', bg='#c8c8d4', font='Calco',
    )
    plt += fps_text

    fps_up_btn = Text2D(
        " > ", pos=(0.12, 0.52), s=_PANEL_S,
        c='white', bg='dodgerblue', font='Calco',
    )
    plt += fps_up_btn

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

    # --- Bottom-left: hint to open help ---
    help_hint = Text2D(
        "Press H for help",
        pos=(0.02, 0.01), s=0.8, c='#666666', font='Calco',
    )
    plt += help_hint

    # --- Help overlay (toggled with H key) ---
    _help_lines = [
        " Keyboard Shortcuts ",
        "",
        " Space       Play / Pause",
        " Left/Right  Step 1 frame",
        " Home / End  Jump to start / end",
        " +/-         Speed x2 / /2",
        " L           Toggle loop",
        " R           Reset camera",
        " F           Cycle FPS (15/30/60/120)",
        " J           Toggle joint labels",
        " S           Save screenshot",
        " N           Cycle ghost mode (off/3/5)",
        " T           Toggle trajectory trail",
        " 1-9         Toggle skeleton visibility",
        " H           Toggle this help",
    ]
    _help_overlay = Text2D(
        "\n".join(_help_lines),
        pos='center', s=1.2,
        c='white', bg='#1a1a2e', font='Calco',
    )
    _help_overlay.actor.SetVisibility(0)
    plt += _help_overlay

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

    def _current_step() -> int:
        """Subsampling step for the current FPS preset."""
        return max(1, math.ceil(_native_fps / _fps_presets[_fps_idx]))

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
        """Update frame info in the window title bar."""
        t = f * _current_step() / _native_fps
        plt.window.SetWindowName(
            f"pybvh viewer  |  Frame {f}/{num_frames - 1}"
            f"  |  t={t:.2f}s  |  {_fps_presets[_fps_idx]}fps"
            f"  |  {state['speed']:.3g}x")

    def _set_speed(new_speed: float) -> None:
        """Change playback speed and restart timer.

        For speeds < 1x the timer interval is stretched (fewer ticks).
        For speeds >= 1x the timer fires at the base rate and the
        callback skips frames to achieve the target speed.
        """
        state['speed'] = new_speed
        _reset_play_clock()
        effective_fps = _fps_presets[_fps_idx]
        base_interval = max(int(1000.0 / effective_fps), 8)
        if new_speed < 1.0:
            state['interval'] = max(int(base_interval / new_speed), 8)
        else:
            state['interval'] = base_interval
        _sync_all()
        if state['timer_id'] is not None:
            plt.timer_callback('destroy', state['timer_id'])
        state['timer_id'] = plt.timer_callback(
            'create', dt=state['interval'])

    def _set_fps(idx: int) -> None:
        """Change FPS preset and resample coordinate data."""
        nonlocal coords_list, num_frames, _fps_idx
        _fps_idx = idx
        target_fps = _fps_presets[idx]
        step = max(1, math.ceil(_native_fps / target_fps))
        coords_list = [c[::step] for c in _coords_full]
        num_frames = coords_list[0].shape[0]
        fps_text.text(f" {target_fps} ")
        # Reset playback to frame 0 (frame indices changed)
        state['playing'] = False
        state['frame'] = 0
        state['ghost_history'].clear()
        for s in range(n_skeletons):
            for gi in range(_MAX_GHOSTS):
                _ghost_lines[s][gi].actor.SetVisibility(0)
                _ghost_points[s][gi].actor.SetVisibility(0)
        state['_slider_updating'] = True
        slider.GetRepresentation().SetMinimumValue(0)
        slider.GetRepresentation().SetMaximumValue(num_frames - 1)
        slider.value = 0
        state['_slider_updating'] = False
        # Restart timer at new rate
        actual_fps = _native_fps / step
        base_interval = max(int(1000.0 / actual_fps), 8)
        state['interval'] = base_interval
        if state['timer_id'] is not None:
            plt.timer_callback('destroy', state['timer_id'])
        state['timer_id'] = plt.timer_callback(
            'create', dt=state['interval'])
        _sync_all()
        update_frame(0)

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
        n_ghosts = _GHOST_LEVELS[state['ghost_level_idx']]

        for s in range(n_skeletons):
            frame_data = coords_list[s][f]
            if use_high:
                _update_skeleton_high(s, frame_data)
            else:
                _update_skeleton_fast(s, frame_data)
            # Update joint labels when visible
            if state['show_labels']:
                for j in range(len(frame_data)):
                    _label_actors[s][j].SetPosition(
                        *(frame_data[j] + _label_offset))
            # Show trail [0:current_frame], collapse the rest
            if state['show_trail']:
                root_pts = _trail_full[s]
                full_f = min(f * _current_step(), len(root_pts) - 1)
                # Fast copy from pre-allocated collapsed buffer
                verts = _trail_collapsed[s].copy()
                if full_f > 0:
                    visible = _interleave(
                        root_pts[:full_f], root_pts[1:full_f + 1])
                    verts[:len(visible)] = visible
                _trail_actors[s].vertices = verts

        # Update ghost frames (shared across skeletons)
        if n_ghosts > 0:
            # Push current frame data into history BEFORE rendering ghosts
            state['ghost_history'].append(
                [coords_list[s][f].copy() for s in range(n_skeletons)])
            history = list(state['ghost_history'])
            _ghost_skip = 5  # show every 5th frame for wider separation
            for gi in range(_MAX_GHOSTS):
                # gi=0 → 5 frames ago, gi=1 → 10 frames ago, etc.
                hist_idx = (gi + 1) * _ghost_skip
                if hist_idx <= len(history) and gi < n_ghosts:
                    alpha = 0.3 * (1.0 - gi / n_ghosts)
                    for s in range(n_skeletons):
                        h_data = history[-hist_idx]
                        p_i = _bone_parent_idx[s]
                        c_i = _bone_child_idx[s]
                        sp = h_data[s][p_i]
                        ep = h_data[s][c_i]
                        _ghost_lines[s][gi].vertices = _interleave(sp, ep)
                        _ghost_lines[s][gi].actor.SetVisibility(1)
                        _ghost_points[s][gi].vertices = h_data[s]
                        _ghost_points[s][gi].actor.SetVisibility(1)
                        # Only update alpha when it changes
                        if alpha != _ghost_last_alpha[s][gi]:
                            _ghost_lines[s][gi].alpha(alpha)
                            _ghost_points[s][gi].alpha(alpha)
                            _ghost_last_alpha[s][gi] = alpha
                else:
                    for s in range(n_skeletons):
                        _ghost_lines[s][gi].actor.SetVisibility(0)
                        _ghost_points[s][gi].actor.SetVisibility(0)

        _update_frame_display(f)
        plt.render()

    # Apply default FPS if it differs from native (must be after update_frame)
    if _fps_presets[_fps_idx] != int(round(_native_fps)):
        _set_fps(_fps_idx)

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
                        _reset_play_clock()
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
            elif 0.49 < ny < 0.56:
                # FPS down: 0.01–0.05, FPS up: 0.12–0.16
                if 0.01 < nx < 0.05 and _fps_idx > 0:
                    _set_fps(_fps_idx - 1)
                elif 0.12 < nx < 0.16 and _fps_idx < len(_fps_presets) - 1:
                    _set_fps(_fps_idx + 1)

    plt.add_callback('LeftButtonPress', _on_click)

    # =================================================================
    # TIMER CALLBACK
    # =================================================================

    def _reset_play_clock() -> None:
        """Reset the wall-clock reference for time-based frame advancement."""
        state['_play_start_time'] = None

    def timer_callback(event: object) -> None:
        # Skip if not playing or if a previous render is still in progress.
        if not state['playing'] or state.get('_rendering'):
            return
        state['_rendering'] = True
        try:
            # Use wall-clock time to determine the correct frame.
            # This keeps animation speed accurate even when timer events
            # are dropped (e.g., when VTK overhead > timer interval).
            now = time.perf_counter()
            if state.get('_play_start_time') is None:
                state['_play_start_time'] = now
                state['_play_start_frame'] = state['frame']

            elapsed = now - state['_play_start_time']
            effective_fps = _fps_presets[_fps_idx]
            target_f = state['_play_start_frame'] + int(
                elapsed * effective_fps * state['speed'])

            if target_f >= num_frames:
                if state['loop']:
                    # Reset clock for seamless loop
                    target_f = target_f % num_frames
                    state['_play_start_time'] = now
                    state['_play_start_frame'] = target_f
                else:
                    target_f = num_frames - 1
                    state['playing'] = False
                    _sync_all()

            if target_f != state['frame']:
                state['frame'] = target_f
                state['_slider_updating'] = True
                slider.value = target_f
                state['_slider_updating'] = False
                update_frame(target_f)
        finally:
            state['_rendering'] = False

    # =================================================================
    # KEYBOARD CALLBACK
    # =================================================================

    def key_callback(event: object) -> None:
        key = plt.last_event.keypress  # type: ignore[attr-defined]

        if key == 'space':
            state['playing'] = not state['playing']
            _reset_play_clock()
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

        elif key == 'n':
            # Cycle ghost / onion-skin levels
            idx = (state['ghost_level_idx'] + 1) % len(_GHOST_LEVELS)
            state['ghost_level_idx'] = idx
            n_g = _GHOST_LEVELS[idx]
            print(f"Ghost mode: {n_g} frames"
                  f" (level {idx}/{len(_GHOST_LEVELS)-1})")
            state['ghost_history'].clear()
            # Hide all ghosts when turning off
            if n_g == 0:
                for s in range(n_skeletons):
                    for gi in range(_MAX_GHOSTS):
                        _ghost_lines[s][gi].actor.SetVisibility(0)
                        _ghost_points[s][gi].actor.SetVisibility(0)
            plt.render()

        elif key == 't':
            # Toggle root trajectory trail (pre-computed, just show/hide)
            state['show_trail'] = not state['show_trail']
            vis = 1 if state['show_trail'] else 0
            for s in range(n_skeletons):
                _trail_actors[s].actor.SetVisibility(vis)
            plt.render()

        elif key == 'f':
            # Cycle FPS presets
            _set_fps((_fps_idx + 1) % len(_fps_presets))

        elif key == 'j':
            # Toggle joint name labels
            state['show_labels'] = not state['show_labels']
            vis = 1 if state['show_labels'] else 0
            for s in range(n_skeletons):
                frame_data = coords_list[s][state['frame']]
                for j in range(len(frame_data)):
                    _label_actors[s][j].SetVisibility(vis)
                    if vis:
                        _label_actors[s][j].SetPosition(
                            *(frame_data[j] + _label_offset))
            plt.render()

        elif key == 's':
            # Screenshot current view
            fname = f"pybvh_frame_{state['frame']}.png"
            plt.screenshot(fname)
            print(f"Screenshot saved: {fname}")

        elif key in [str(d) for d in range(1, 10)]:
            # Toggle skeleton visibility (keys 1-9)
            idx = int(key) - 1
            if idx < n_skeletons:
                vis = state['skeleton_visible']
                vis[idx] = not vis[idx]
                v = 1 if vis[idx] else 0
                if use_high:
                    if _bones_mesh[idx] is not None:
                        _bones_mesh[idx].actor.SetVisibility(v)
                    _joints_mesh[idx].actor.SetVisibility(v)
                else:
                    _lines_actors[idx].actor.SetVisibility(v)
                    _points_actors[idx].actor.SetVisibility(v)
                # Also toggle labels for this skeleton
                for a in _label_actors[idx]:
                    a.SetVisibility(
                        v if state['show_labels'] else 0)
                plt.render()

        elif key == 'h':
            # Toggle help overlay
            vis = _help_overlay.actor.GetVisibility()
            _help_overlay.actor.SetVisibility(0 if vis else 1)
            plt.render()

    # --- Register callbacks and start ---
    plt.add_callback('timer', timer_callback)
    plt.add_callback('key press', key_callback)
    if state['timer_id'] is None:
        state['timer_id'] = plt.timer_callback('create', dt=state['interval'])

    plt.show()
    return None
