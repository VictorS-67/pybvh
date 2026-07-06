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

import numpy as np
import numpy.typing as npt

from typing import Callable, TYPE_CHECKING, TypedDict

from ._common import PALETTE_RGB, build_view_matrix, UP_AXIS_INDEX

if TYPE_CHECKING:
    from ..bvh import Bvh

# Rich gold for single-skeleton "high" mode (aitviewer-inspired)
_WARM_AMBER = (230, 175, 50)


class _PlayerState(TypedDict, total=False):
    frame: int
    playing: bool
    interval: int
    timer_id: int | None
    speed: float
    loop_mode: str
    play_direction: int
    _slider_updating: bool
    show_labels: bool
    skeleton_visible: list[bool]
    show_trail: bool
    _rendering: bool
    _play_start_time: float | None
    _play_start_frame: int
    _screenshot_hide_at: float | None


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


def _apply_flat_lighting(mesh: object) -> None:
    """Set flat ambient-only lighting so color is stable across frames.

    Keeps VTK's default scalar coloring (blue-to-green gradient on
    tubes) but removes normal-dependent shading that shifts as bones
    rotate.
    """
    prop = mesh.actor.GetProperty()  # type: ignore[attr-defined]
    prop.SetAmbient(1.0)
    prop.SetDiffuse(0.0)
    prop.SetSpecular(0.0)


# =====================================================================
# UI layout
# =====================================================================
#
# Title bar:    Frame/time/fps/speed info
# Left panel:   Speed, FPS, loop toggle, reset camera
# Bottom:       Slider + transport buttons (Start/Prev/Play/Next/End)
# Right panel:  Help overlay (toggled with the H key)
#
# Every clickable region lives in the ``_VedoPlayer._buttons`` registry:
# each entry places its Text2D *and* defines its click hit-box, so the
# layout and the hit-testing can never drift apart.

_PANEL_X = 0.01     # left-panel x
_PANEL_S = 1.4      # left-panel text scale
_RPANEL_X = 0.85    # right (help) panel x

# Bottom transport bar: _SL_X0/_SL_X1 drive both the slider and the
# button layout.  Change them and everything stays aligned automatically.
_SL_X0, _SL_X1 = 0.15, 0.85   # slider / button-row x extents
_BTN_S = 1.8                  # large, comfortable button text
_BTN_GAP = 0.010              # normalized gap between adjacent buttons
_N_BTNS = 5
# Divide the full slider span evenly: N equal cells separated by (N-1) gaps
_BTN_W = (_SL_X1 - _SL_X0 - (_N_BTNS - 1) * _BTN_GAP) / _N_BTNS
_BTN_X = [_SL_X0 + i * (_BTN_W + _BTN_GAP) for i in range(_N_BTNS)]
# Transport hit band: generous lower bound (0.01) accounts for the ~0.018
# systematic offset between viewport y and GetEventPosition() y observed
# in practice.  The top (0.08) stays safely below the slider baseline at 0.10.
_BTN_Y0 = 0.01
_BTN_H = 0.07
# Vertical offset from a button's hit-box bottom edge to its Text2D baseline.
_TEXT_RAISE = 0.03

# Transport button labels — ASCII words (symbols don't render well in Calco).
# All 9 chars padded for consistent background widths.
_L_FIRST = "  Start  "
_L_BACK = "  Prev   "
_L_PAUSE = "  Pause  "
_L_PLAY = "  Play   "
_L_FWD = "  Next   "
_L_LAST = "   End   "

_HELP_ENTRIES = [
    "Space  Play/Pause",
    "+/-    Speed",
    "Arrows Step frame",
    "L      Loop mode",
    "R      Reset camera",
    "F      Cycle FPS",
    "J      Joint labels",
    "S      Screenshot",
    "T      Trajectory",
    "1-9    Skeletons",
    "",
    "Drag       Orbit",
    "Shift+Drag Pan",
    "Scroll     Zoom",
]


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
    import vedo  # type: ignore[import-untyped]

    if coords_list[0].shape[0] < 1:
        return

    # Disable vedo's default key bindings (L=lighting, arrows=transparency)
    # to avoid conflicts with our playback controls.  Saved and restored in
    # try/finally so a viewer session doesn't permanently mutate vedo's
    # process-wide settings for the caller.
    saved_callbacks = (
        vedo.settings.enable_default_keyboard_callbacks,
        vedo.settings.enable_default_mouse_callbacks,
    )
    vedo.settings.enable_default_keyboard_callbacks = False
    vedo.settings.enable_default_mouse_callbacks = False
    try:
        player = _VedoPlayer(
            bvh_list, coords_list, fps, labels, skeleton_lines_list,
            center, half_span, up_axis=up_axis, azimuth=azimuth,
            elevation=elevation, quality=quality)
        player.show()
    finally:
        (vedo.settings.enable_default_keyboard_callbacks,
         vedo.settings.enable_default_mouse_callbacks) = saved_callbacks


class _VedoPlayer:
    """Interactive vedo skeleton player (one instance per ``play_vedo`` call).

    Construction stages: plotter → geometry (:meth:`_build_geometry`) →
    UI (:meth:`_build_ui`) → event callbacks.  The button registry
    ``self._buttons`` holds one ``(x0, y0, w, h, callback)`` entry per
    clickable region and is the single source of truth for both Text2D
    placement and click hit-testing (see :meth:`_add_button`).
    """

    def __init__(
        self,
        bvh_list: list[Bvh],
        coords_list: list[npt.NDArray[np.float64]],
        fps: float,
        labels: list[str] | None,
        skeleton_lines_list: list[list[tuple[int, int]]],
        center: npt.NDArray[np.float64],
        half_span: float,
        *,
        up_axis: str,
        azimuth: float,
        elevation: float,
        quality: str,
    ) -> None:
        from vedo import Plotter  # type: ignore[import-untyped]

        self.bvh_list = bvh_list
        self.coords_list = coords_list
        self.labels = labels
        self.skeleton_lines_list = skeleton_lines_list
        self.center = center
        self.half_span = half_span
        self.up_axis = up_axis
        self.azimuth = azimuth
        self.elevation = elevation
        self.use_high = quality == "high"

        self.num_frames = coords_list[0].shape[0]
        self.n_skeletons = len(bvh_list)
        # Keep full-rate data for FPS resampling
        self._coords_full = [c.copy() for c in coords_list]

        # --- FPS presets ---
        self._native_fps = fps
        self._fps_presets = sorted(set([15, 30, 60, 120, int(round(fps))]))
        default_fps = 30 if fps > 30 else fps
        self._fps_idx = self._fps_presets.index(
            min(self._fps_presets, key=lambda x: abs(x - default_fps)))

        # --- Animation state ---
        self.state: _PlayerState = {
            'frame': 0,
            'playing': True,
            'interval': max(int(1000.0 / max(fps, 1)), 8),
            'timer_id': None,
            'speed': 1.0,
            'loop_mode': 'loop',       # 'loop', 'ping-pong', 'off'
            'play_direction': 1,       # 1 = forward, -1 = backward
            '_slider_updating': False,
            'show_labels': False,
            'skeleton_visible': [True] * self.n_skeletons,
            'show_trail': False,
            '_rendering': False,
            '_play_start_time': None,
            '_play_start_frame': 0,
        }

        self.plt = Plotter(
            title="pybvh viewer",
            size=(1400, 900),
            bg='#d4d4dc',        # soft cool gray at bottom (behind controls)
            bg2='lightskyblue',  # sky at top
        )

        # Button registry: (x0, y0, w, h, callback) per clickable region.
        self._buttons: list[
            tuple[float, float, float, float, Callable[[], None]]] = []

        self._build_geometry()
        self._build_ui()

        # Apply default FPS if it differs from native (needs the slider,
        # so this runs after _build_ui).
        if self._fps_presets[self._fps_idx] != int(round(self._native_fps)):
            self._set_fps(self._fps_idx)

        self.plt.add_callback('LeftButtonPress', self._on_click)
        self.plt.add_callback('timer', self._on_timer)
        self.plt.add_callback('key press', self._on_key)
        if self.state['timer_id'] is None:
            self.state['timer_id'] = self.plt.timer_callback(
                'create', dt=self.state['interval'])

    def show(self) -> None:
        self.plt.show()

    # =================================================================
    # GEOMETRY
    # =================================================================

    def _color_rgb(self, s: int) -> tuple[int, int, int]:
        if self.n_skeletons == 1 and self.use_high:
            return _WARM_AMBER
        return PALETTE_RGB[s % len(PALETTE_RGB)]

    def _color(self, s: int) -> str:
        r, g, b = self._color_rgb(s)
        return f"rgb({r},{g},{b})"

    def _build_geometry(self) -> None:
        """Create the floor, skeleton actors, labels, camera, and trails."""
        from vedo import (  # type: ignore[import-untyped]
            Lines, Points, Tube, Sphere, Grid, Text2D, merge,
        )
        import vtk  # type: ignore[import-untyped]

        coords_list = self.coords_list
        n_skeletons = self.n_skeletons
        half_span = self.half_span
        up_idx = UP_AXIS_INDEX.get(self.up_axis, 2)

        # Sizing — base radius, then adapted per-bone by length
        r_bone_base = half_span * 0.013

        # Precompute adaptive radii: scale each bone's radius proportional
        # to its length relative to the median.  Short finger bones get thin
        # tubes; long limb bones stay thick.
        _bone_radii: list[dict[tuple[int, int], float]] = []
        _joint_radii: list[npt.NDArray[np.float64]] = []
        for s in range(n_skeletons):
            frame0 = coords_list[s][0]
            bones = self.skeleton_lines_list[s]
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

        # --- Floor grid (high quality only) ---
        if self.use_high:
            # Place floor at the lowest point of all skeletons across all frames
            floor_y = min(c[:, :, up_idx].min() for c in coords_list)
            floor_pos = self.center.copy()
            floor_pos[up_idx] = floor_y
            floor = Grid(
                pos=tuple(floor_pos),
                s=[half_span * 2.5, half_span * 2.5],
                res=(30, 30),
            )
            if self.up_axis == 'y':
                floor.rotate_x(90)
            elif self.up_axis == 'x':
                floor.rotate_y(90)
            # up_axis='z': Grid defaults to XY plane, no rotation needed
            floor.lw(1).alpha(0.6).c('#555555').lighting('off')
            self.plt += floor

        # --- Build persistent skeleton geometry (created once, updated in-place) ---

        # High mode: 2 merged meshes per skeleton (all bones + all joints)
        self._bones_mesh: list = []              # merged Mesh per skeleton
        self._joints_mesh: list = []             # merged Mesh per skeleton
        self._canonical_bone_verts: list = []    # ndarray (n_bones, V_bone, 3)
        self._canonical_joint_verts: list = []   # ndarray (n_joints, V_joint, 3)
        # Pre-computed bone index arrays for vectorized frame lookups
        self._bone_parent_idx: list = []         # ndarray (n_bones,) per skeleton
        self._bone_child_idx: list = []          # ndarray (n_bones,) per skeleton

        # Fast mode: Lines + Points per skeleton
        self._lines_actors: list = []
        self._points_actors: list = []

        # --- Create actors once and position to frame 0 ---
        for s in range(n_skeletons):
            color = self._color(s)
            bones = self.skeleton_lines_list[s]
            self._bone_parent_idx.append(np.array([b[0] for b in bones]))
            self._bone_child_idx.append(np.array([b[1] for b in bones]))

            if self.use_high:
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
                    self._canonical_bone_verts.append(
                        np.array(bone_verts_list))  # (n_bones, V, 3)
                else:
                    bones_merged = None
                    self._canonical_bone_verts.append(np.empty((0, 0, 3)))
                self._bones_mesh.append(bones_merged)

                # Create canonical joint spheres and collect vertices
                joint_meshes = []
                joint_verts_list = []
                for j in range(coords_list[s].shape[1]):
                    sph = Sphere(pos=(0, 0, 0), r=jr[j], res=12, c=color)
                    joint_verts_list.append(sph.vertices.copy())
                    joint_meshes.append(sph)
                joints_merged = merge(joint_meshes)
                _apply_flat_lighting(joints_merged)
                self._canonical_joint_verts.append(
                    np.array(joint_verts_list))   # (n_joints, V, 3)
                self._joints_mesh.append(joints_merged)

                # Add only 2 merged actors to plotter
                if bones_merged is not None:
                    self.plt += bones_merged
                self.plt += joints_merged

                # Position to frame 0
                self._update_skeleton_high(s, coords_list[s][0])
            else:
                frame0 = coords_list[s][0]
                _lw = max(1, int(half_span * 0.04))
                _pr = max(1, int(half_span * 0.05))
                lines = Lines(
                    frame0[self._bone_parent_idx[s]],
                    frame0[self._bone_child_idx[s]],
                    lw=_lw, c=color)
                lines.lighting('off')
                points = Points(frame0, r=_pr, c=color, alpha=0.9)
                self._lines_actors.append(lines)
                self._points_actors.append(points)
                self.plt += lines
                self.plt += points

        # --- Labels ---
        if self.labels:
            for s in range(min(len(self.labels), n_skeletons)):
                label = Text2D(
                    self.labels[s],
                    pos=(0.02 + s * 0.15, 0.95),
                    c=self._color(s), s=1.4, font='Calco',
                )
                self.plt += label

        # --- Camera setup (same convention as matplotlib / opencv backends) ---
        view_mat = build_view_matrix(self.azimuth, self.elevation, self.up_axis)
        # view_mat rows: [right, up, eye_direction (toward viewer)]
        eye_dir = view_mat[2]
        cam_dist = half_span * 4.0
        self._cam_pos = self.center + eye_dir * cam_dist
        self._cam_up = view_mat[1]
        self._set_camera()

        # --- Joint name labels (toggle with J key) ---
        # Use vtkBillboardTextActor3D so labels always face the camera.
        # _label_actors[s][j] = vtkBillboardTextActor3D
        self._label_actors: list[list] = []
        self._label_offset = np.zeros(3)
        self._label_offset[up_idx] = half_span * 0.02
        label_fontsize = max(12, int(half_span * 0.4))
        for s in range(n_skeletons):
            lbl_list: list = []
            joint_names = [node.name for node in self.bvh_list[s].nodes]
            for j, name in enumerate(joint_names):
                pos0 = coords_list[s][0][j] + self._label_offset
                actor = vtk.vtkBillboardTextActor3D()
                actor.SetInput(name)
                actor.SetPosition(*pos0)
                actor.GetTextProperty().SetFontSize(label_fontsize)
                actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                actor.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.3)
                actor.GetTextProperty().SetBackgroundOpacity(0.7)
                actor.GetTextProperty().SetJustificationToCentered()
                actor.SetVisibility(0)
                lbl_list.append(actor)
                self.plt.renderer.AddActor(actor)
            self._label_actors.append(lbl_list)

        # --- Root trajectory trail (toggle with T key) ---
        # Pre-compute full root path; pre-allocate Lines with collapsed
        # segments.  Each frame, expand segments up to the current frame
        # (fast vertex update).  The trail sits at the lowest joint level
        # across all skeletons and frames, in both quality modes.
        self._trail_actors: list = []
        self._trail_full: list[npt.NDArray] = []       # pre-computed root paths
        self._trail_collapsed: list[npt.NDArray] = []  # pre-allocated collapsed buffers
        trail_floor = min(c[:, :, up_idx].min() for c in self._coords_full)
        for s in range(n_skeletons):
            root_all = self._coords_full[s][:, 0, :].copy()  # (F, 3)
            root_all[:, up_idx] = trail_floor
            self._trail_full.append(root_all)
            # Pre-allocate collapsed buffer (reused every frame via .copy())
            collapsed = np.tile(root_all[0], (2 * (len(root_all) - 1), 1))
            self._trail_collapsed.append(collapsed)
            trail = Lines(collapsed[::2], collapsed[1::2],
                          lw=2, c=self._color(s), alpha=0.6)
            trail.lighting('off')
            trail.actor.SetVisibility(0)
            self._trail_actors.append(trail)
            self.plt += trail

    def _update_skeleton_high(self, s: int, frame_data: npt.NDArray) -> None:
        """Update merged bone and joint meshes via vectorized numpy."""
        p_idx = self._bone_parent_idx[s]
        c_idx = self._bone_child_idx[s]
        canonical_bones = self._canonical_bone_verts[s]

        if len(p_idx) > 0 and self._bones_mesh[s] is not None:
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

            self._bones_mesh[s].vertices = transformed.reshape(-1, 3)

        # Joints: vectorized translation (single operation)
        canonical_joints = self._canonical_joint_verts[s]
        self._joints_mesh[s].vertices = (
            canonical_joints + frame_data[:, np.newaxis, :]
        ).reshape(-1, 3)

    def _update_skeleton_fast(self, s: int, frame_data: npt.NDArray) -> None:
        """Update Lines/Points vertex data in-place for skeleton *s*."""
        p_idx = self._bone_parent_idx[s]
        c_idx = self._bone_child_idx[s]
        self._lines_actors[s].vertices = _interleave(
            frame_data[p_idx], frame_data[c_idx])
        self._points_actors[s].vertices = frame_data

    def _set_camera(self) -> None:
        """Apply the initial camera position."""
        self.plt.camera.SetPosition(*self._cam_pos)
        self.plt.camera.SetFocalPoint(*self.center)
        self.plt.camera.SetViewUp(*self._cam_up)
        self.plt.reset_camera()

    # =================================================================
    # UI
    # =================================================================

    def _add_button(
        self,
        text: str,
        x0: float,
        y0: float,
        w: float,
        h: float,
        callback: Callable[[], None],
        *,
        s: float = _PANEL_S,
        bg: str = 'dodgerblue',
        c: str = 'white',
        centered: bool = False,
    ):
        """Place a clickable Text2D and register its hit-box.

        The registry entry and the Text2D placement are derived from the
        same ``(x0, y0, w, h)`` rectangle: the text baseline sits
        ``_TEXT_RAISE`` above the hit-box bottom, left-aligned at ``x0``
        (or centered in the cell for transport buttons).
        """
        from vedo import Text2D  # type: ignore[import-untyped]

        if centered:
            t2d = Text2D(text, pos=(x0 + w / 2, y0 + _TEXT_RAISE), s=s,
                         c=c, bg=bg, font='Calco', justify='bottom-center')
        else:
            t2d = Text2D(text, pos=(x0, y0 + _TEXT_RAISE), s=s,
                         c=c, bg=bg, font='Calco')
        self.plt += t2d
        self._buttons.append((x0, y0, w, h, callback))
        return t2d

    def _build_ui(self) -> None:
        """Create the control panels, help overlay, and frame slider."""
        from vedo import Text2D  # type: ignore[import-untyped]

        # Frame info is shown in the window title bar (not a 2D overlay)

        # --- Left panel (compact: label + < value > on same line) ---
        self.speed_label = Text2D(
            "Spd", pos=(_PANEL_X, 0.92), s=_PANEL_S,
            c='#2c3e50', font='Calco',
        )
        self.plt += self.speed_label
        self._add_button(" < ", 0.05, 0.89, 0.03, 0.07, self._on_speed_down)
        self.speed_text = Text2D(
            " 1x ", pos=(0.08, 0.92), s=_PANEL_S,
            c='#2c3e50', bg='#c8c8d4', font='Calco',
        )
        self.plt += self.speed_text
        self._add_button(" > ", 0.12, 0.89, 0.04, 0.07, self._on_speed_up)

        # --- FPS selector ---
        self.fps_label = Text2D(
            "FPS", pos=(_PANEL_X, 0.86), s=_PANEL_S,
            c='#2c3e50', font='Calco',
        )
        self.plt += self.fps_label
        self._add_button(" < ", 0.05, 0.83, 0.03, 0.06, self._on_fps_down)
        self.fps_text = Text2D(
            f" {self._fps_presets[self._fps_idx]} ", pos=(0.08, 0.86),
            s=_PANEL_S, c='#2c3e50', bg='#c8c8d4', font='Calco',
        )
        self.plt += self.fps_text
        self._add_button(" > ", 0.12, 0.83, 0.04, 0.06, self._on_fps_up)

        self.loop_btn = self._add_button(
            " Loop ", _PANEL_X, 0.77, 0.19, 0.06, self._on_cycle_loop,
            bg='green4')
        self.reset_btn = self._add_button(
            " Reset Cam ", _PANEL_X, 0.71, 0.19, 0.06, self._on_reset_camera)

        # --- Bottom: transport bar ---
        self.btn_first = self._add_button(
            _L_FIRST, _BTN_X[0], _BTN_Y0, _BTN_W, _BTN_H, self._on_first,
            s=_BTN_S, centered=True)
        self.btn_back = self._add_button(
            _L_BACK, _BTN_X[1], _BTN_Y0, _BTN_W, _BTN_H, self._on_prev,
            s=_BTN_S, centered=True)
        self.btn_play = self._add_button(
            _L_PAUSE, _BTN_X[2], _BTN_Y0, _BTN_W, _BTN_H, self._toggle_play,
            s=_BTN_S, bg='tomato', centered=True)
        self.btn_fwd = self._add_button(
            _L_FWD, _BTN_X[3], _BTN_Y0, _BTN_W, _BTN_H, self._on_next,
            s=_BTN_S, centered=True)
        self.btn_last = self._add_button(
            _L_LAST, _BTN_X[4], _BTN_Y0, _BTN_W, _BTN_H, self._on_last,
            s=_BTN_S, centered=True)

        # --- Right panel: help (toggled with H key) ---
        self._help_header = Text2D(
            " Help (H) ", pos=(_RPANEL_X, 0.92), s=_PANEL_S,
            c='white', bg='#2c3e50', font='Calco',
        )
        self.plt += self._help_header

        self._help_items: list = []
        for i, txt in enumerate(_HELP_ENTRIES):
            t = Text2D(txt, pos=(_RPANEL_X, 0.86 - i * 0.045), s=1.1,
                       c='#2c3e50', font='Calco')
            t.actor.SetVisibility(0)
            self._help_items.append(t)
            self.plt += t

        # --- Screenshot feedback overlay (center-top, hidden by default) ---
        self._screenshot_text = Text2D(
            "", pos=(0.35, 0.92), s=1.2,
            c='white', bg='green4', font='Calco',
        )
        self._screenshot_text.actor.SetVisibility(0)
        self.plt += self._screenshot_text

        # --- Frame scrubber slider ---
        self.slider = self.plt.add_slider(
            self._on_slider,
            xmin=0, xmax=self.num_frames - 1,
            value=0,
            pos=[(_SL_X0, 0.12), (_SL_X1, 0.12)],   # matches button row extents
            title='',
            show_value=False,
        )

    # =================================================================
    # UI SYNC HELPERS
    # =================================================================

    def _current_step(self) -> int:
        """Subsampling step for the current FPS preset."""
        return max(1, math.ceil(
            self._native_fps / self._fps_presets[self._fps_idx]))

    def _sync_all(self) -> None:
        """Sync all UI elements to match current state."""
        # Play/pause button
        if self.state['playing']:
            self.btn_play.text(_L_PAUSE)
            self.btn_play.background('tomato')
        else:
            self.btn_play.text(_L_PLAY)
            self.btn_play.background('green4')
        # Loop / ping-pong button
        mode = self.state['loop_mode']
        if mode == 'loop':
            self.loop_btn.text(" Loop ")
            self.loop_btn.background('green4')
        elif mode == 'ping-pong':
            self.loop_btn.text(" Ping ")
            self.loop_btn.background('dodgerblue')
        else:
            self.loop_btn.text(" ---  ")
            self.loop_btn.background('gray')
        # Speed display
        spd = self.state['speed']
        self.speed_text.text(
            f" {spd:.1f}x " if spd != int(spd) else f" {int(spd)}x ")

    def _update_frame_display(self, f: int) -> None:
        """Update frame info in the window title bar."""
        t = f * self._current_step() / self._native_fps
        self.plt.window.SetWindowName(
            f"pybvh viewer  |  Frame {f}/{self.num_frames - 1}"
            f"  |  t={t:.2f}s  |  {self._fps_presets[self._fps_idx]}fps"
            f"  |  {self.state['speed']:.3g}x")

    def _reset_play_clock(self) -> None:
        """Reset the wall-clock reference for time-based frame advancement."""
        self.state['_play_start_time'] = None

    def _set_speed(self, new_speed: float) -> None:
        """Change playback speed and restart timer.

        For speeds < 1x the timer interval is stretched (fewer ticks).
        For speeds >= 1x the timer fires at the base rate and the
        callback skips frames to achieve the target speed.
        """
        self.state['speed'] = new_speed
        self._reset_play_clock()
        effective_fps = self._fps_presets[self._fps_idx]
        base_interval = max(int(1000.0 / effective_fps), 8)
        if new_speed < 1.0:
            self.state['interval'] = max(int(base_interval / new_speed), 8)
        else:
            self.state['interval'] = base_interval
        self._sync_all()
        if self.state['timer_id'] is not None:
            self.plt.timer_callback('destroy', self.state['timer_id'])
        self.state['timer_id'] = self.plt.timer_callback(
            'create', dt=self.state['interval'])

    def _set_fps(self, idx: int) -> None:
        """Change FPS preset and resample coordinate data."""
        self._fps_idx = idx
        target_fps = self._fps_presets[idx]
        step = max(1, math.ceil(self._native_fps / target_fps))
        self.coords_list = [c[::step] for c in self._coords_full]
        self.num_frames = self.coords_list[0].shape[0]
        self.fps_text.text(f" {target_fps} ")
        # Reset playback to frame 0 (frame indices changed)
        self.state['playing'] = False
        self.state['frame'] = 0
        self.state['_slider_updating'] = True
        self.slider.GetRepresentation().SetMinimumValue(0)
        self.slider.GetRepresentation().SetMaximumValue(self.num_frames - 1)
        self.slider.value = 0
        self.state['_slider_updating'] = False
        # Restart timer at new rate
        actual_fps = self._native_fps / step
        base_interval = max(int(1000.0 / actual_fps), 8)
        self.state['interval'] = base_interval
        if self.state['timer_id'] is not None:
            self.plt.timer_callback('destroy', self.state['timer_id'])
        self.state['timer_id'] = self.plt.timer_callback(
            'create', dt=self.state['interval'])
        self._sync_all()
        self._update_frame(0)

    def _jump_to(self, f: int) -> None:
        """Jump to frame f, pause, and sync UI."""
        self.state['playing'] = False
        self.state['frame'] = f
        self.state['_slider_updating'] = True
        self.slider.value = f
        self.state['_slider_updating'] = False
        self._sync_all()
        self._update_frame_display(f)

    # =================================================================
    # FRAME UPDATE
    # =================================================================

    def _update_frame(self, f: int) -> None:
        for s in range(self.n_skeletons):
            frame_data = self.coords_list[s][f]
            if self.use_high:
                self._update_skeleton_high(s, frame_data)
            else:
                self._update_skeleton_fast(s, frame_data)
            # Update joint labels when visible
            if self.state['show_labels']:
                for j in range(len(frame_data)):
                    self._label_actors[s][j].SetPosition(
                        *(frame_data[j] + self._label_offset))
            # Show trail [0:current_frame], collapse the rest
            if self.state['show_trail']:
                root_pts = self._trail_full[s]
                full_f = min(f * self._current_step(), len(root_pts) - 1)
                verts = self._trail_collapsed[s].copy()
                if full_f > 0:
                    visible = _interleave(
                        root_pts[:full_f], root_pts[1:full_f + 1])
                    verts[:len(visible)] = visible
                self._trail_actors[s].vertices = verts

        # Hide screenshot feedback after timeout
        hide_at = self.state.get('_screenshot_hide_at')
        if hide_at and time.perf_counter() > hide_at:
            self._screenshot_text.actor.SetVisibility(0)
            self.state['_screenshot_hide_at'] = None

        self._update_frame_display(f)
        self.plt.render()

    # =================================================================
    # BUTTON / KEY ACTIONS
    # =================================================================

    def _on_first(self) -> None:
        self._jump_to(0)
        self._update_frame(0)

    def _on_prev(self) -> None:
        self._jump_to(max(self.state['frame'] - 1, 0))
        self._update_frame(self.state['frame'])

    def _toggle_play(self) -> None:
        self.state['playing'] = not self.state['playing']
        self._reset_play_clock()
        self._sync_all()
        self.plt.render()

    def _on_next(self) -> None:
        self._jump_to(min(self.state['frame'] + 1, self.num_frames - 1))
        self._update_frame(self.state['frame'])

    def _on_last(self) -> None:
        self._jump_to(self.num_frames - 1)
        self._update_frame(self.num_frames - 1)

    def _on_speed_down(self) -> None:
        self._set_speed(max(self.state['speed'] / 2, 0.125))
        self.plt.render()

    def _on_speed_up(self) -> None:
        self._set_speed(min(self.state['speed'] * 2, 16.0))
        self.plt.render()

    def _on_fps_down(self) -> None:
        if self._fps_idx > 0:
            self._set_fps(self._fps_idx - 1)

    def _on_fps_up(self) -> None:
        if self._fps_idx < len(self._fps_presets) - 1:
            self._set_fps(self._fps_idx + 1)

    def _on_cycle_loop(self) -> None:
        # Cycle: loop → ping-pong → off → loop
        _cycle = {'loop': 'ping-pong', 'ping-pong': 'off', 'off': 'loop'}
        self.state['loop_mode'] = _cycle[self.state['loop_mode']]
        self.state['play_direction'] = 1
        self._reset_play_clock()
        self._sync_all()
        self.plt.render()

    def _on_reset_camera(self) -> None:
        self._set_camera()
        self.plt.render()

    # =================================================================
    # EVENT CALLBACKS
    # =================================================================

    def _on_click(self, event: object) -> None:
        """Hit-test the click against the button registry."""
        # Use raw interactor position (actual cursor) rather than picked2d,
        # which sticks to the last-picked 2D actor position after any click.
        x, y = self.plt.interactor.GetEventPosition()
        w, h = self.plt.window.GetSize()
        nx, ny = x / w, y / h
        for x0, y0, bw, bh, callback in self._buttons:
            if x0 < nx < x0 + bw and y0 < ny < y0 + bh:
                callback()
                return

    def _on_timer(self, event: object) -> None:
        # Skip if not playing or if a previous render is still in progress.
        state = self.state
        if not state['playing'] or state.get('_rendering'):
            return
        state['_rendering'] = True
        try:
            # Use wall-clock time to determine the correct frame.
            # This keeps animation speed accurate even when timer events
            # are dropped (e.g., when VTK overhead > timer interval).
            now = time.perf_counter()
            start_time = state.get('_play_start_time')
            if start_time is None:
                start_time = now
                state['_play_start_time'] = now
                state['_play_start_frame'] = state['frame']

            elapsed = now - start_time
            effective_fps = self._fps_presets[self._fps_idx]
            d = state['play_direction']
            target_f = state['_play_start_frame'] + d * int(
                elapsed * effective_fps * state['speed'])

            if target_f >= self.num_frames:
                if state['loop_mode'] == 'loop':
                    target_f = target_f % self.num_frames
                    state['_play_start_time'] = now
                    state['_play_start_frame'] = target_f
                elif state['loop_mode'] == 'ping-pong':
                    state['play_direction'] *= -1
                    target_f = self.num_frames - 1
                    self._reset_play_clock()
                else:
                    target_f = self.num_frames - 1
                    state['playing'] = False
                    self._sync_all()
            elif target_f < 0:
                if state['loop_mode'] == 'ping-pong':
                    state['play_direction'] *= -1
                    target_f = 0
                    self._reset_play_clock()
                else:
                    target_f = 0
                    state['playing'] = False
                    self._sync_all()

            if target_f != state['frame']:
                state['frame'] = target_f
                state['_slider_updating'] = True
                self.slider.value = target_f
                state['_slider_updating'] = False
                self._update_frame(target_f)
        finally:
            state['_rendering'] = False

    def _on_slider(self, widget: object, event: object) -> None:
        if self.state['_slider_updating']:
            return
        f = int(round(widget.value))  # type: ignore[attr-defined]
        f = max(0, min(f, self.num_frames - 1))
        self.state['frame'] = f
        self.state['playing'] = False
        self._sync_all()
        self._update_frame_display(f)

    def _on_key(self, event: object) -> None:
        key = self.plt.last_event.keypress  # type: ignore[attr-defined]
        state = self.state

        if key == 'space':
            self._toggle_play()

        elif key == 'Right':
            self._on_next()

        elif key == 'Left':
            self._on_prev()

        elif key in ('plus', 'equal'):
            self._on_speed_up()

        elif key == 'minus':
            self._on_speed_down()

        elif key == 'l':
            self._on_cycle_loop()

        elif key == 'r':
            self._on_reset_camera()

        elif key == 'Home':
            self._on_first()

        elif key == 'End':
            self._on_last()

        elif key == 't':
            # Toggle root trajectory trail (pre-computed, just show/hide)
            state['show_trail'] = not state['show_trail']
            vis = 1 if state['show_trail'] else 0
            for s in range(self.n_skeletons):
                self._trail_actors[s].actor.SetVisibility(vis)
            self.plt.render()

        elif key == 'f':
            # Cycle FPS presets
            self._set_fps((self._fps_idx + 1) % len(self._fps_presets))

        elif key == 'j':
            # Toggle joint name labels
            state['show_labels'] = not state['show_labels']
            vis = 1 if state['show_labels'] else 0
            for s in range(self.n_skeletons):
                frame_data = self.coords_list[s][state['frame']]
                for j in range(len(frame_data)):
                    self._label_actors[s][j].SetVisibility(vis)
                    if vis:
                        self._label_actors[s][j].SetPosition(
                            *(frame_data[j] + self._label_offset))
            self.plt.render()

        elif key == 's':
            # Screenshot with feedback overlay
            fname = f"pybvh_frame_{state['frame']}.png"
            self.plt.screenshot(fname)
            print(f"Screenshot saved: {fname}")
            self._screenshot_text.text(f" Saved: {fname} ")
            self._screenshot_text.actor.SetVisibility(1)
            state['_screenshot_hide_at'] = time.perf_counter() + 1.5
            self.plt.render()

        elif key in [str(d) for d in range(1, 10)]:
            # Toggle skeleton visibility (keys 1-9)
            idx = int(key) - 1
            if idx < self.n_skeletons:
                vis_list = state['skeleton_visible']
                vis_list[idx] = not vis_list[idx]
                v = 1 if vis_list[idx] else 0
                if self.use_high:
                    if self._bones_mesh[idx] is not None:
                        self._bones_mesh[idx].actor.SetVisibility(v)
                    self._joints_mesh[idx].actor.SetVisibility(v)
                else:
                    self._lines_actors[idx].actor.SetVisibility(v)
                    self._points_actors[idx].actor.SetVisibility(v)
                # Also toggle labels for this skeleton
                for a in self._label_actors[idx]:
                    a.SetVisibility(
                        v if state['show_labels'] else 0)
                self.plt.render()

        elif key == 'h':
            # Toggle right-side help panel
            vis = 0 if self._help_items[0].actor.GetVisibility() else 1
            for item in self._help_items:
                item.actor.SetVisibility(vis)
            self.plt.render()
