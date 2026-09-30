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

import time

import numpy as np
import numpy.typing as npt

from typing import Callable, TypedDict

from ._style import Style
from ._viewport import make_viewport
from ._scene import Scene, UP_AXIS_INDEX
from ._colors import (
    bone_colors_255, floor_palette, node_colors_255, skeleton_color_255,
)
from ._playback import PlaybackClock
from ._vedo_capsules import CapsuleSkeleton, floor_placement, vedo_rgb


# Test seam: forces the player's Plotter offscreen so construction,
# geometry, and the screenshot path can run without a display.
_FORCE_OFFSCREEN = False


class _UiState(TypedDict, total=False):
    """UI-only flags — playback bookkeeping lives in PlaybackClock."""
    timer_id: int | None
    _slider_updating: bool
    show_labels: bool
    skeleton_visible: list[bool]
    show_trail: bool
    _rendering: bool
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
    scene: Scene,
    style: Style,
    fps: float,
    *,
    quality: str = "high",
) -> None:
    """Interactive skeleton playback in a desktop window via vedo.

    A single-scene backend: all skeletons share one camera (taken from
    the first view) and one viewport over the — possibly laterally
    spread — views.

    Style application (look fields): background, floor kind
    (``"checker"`` falls back to ``"grid"`` here), bone width, and
    the bone colors ``color_mode`` resolves to, in both quality modes
    and from the same rule as the offscreen renderer
    (:func:`~._colors.bone_colors_255`). Joint spheres are structural
    in this viewer and always drawn.

    Parameters
    ----------
    scene : Scene
        Prepared visualization.
    style : Style
        Visual styling (look fields).
    fps : float
        Frames per second.
    quality : str
        ``"high"`` for 3D geometry, ``"fast"`` for flat wireframe.
    """
    import vedo  # type: ignore[import-untyped]

    if scene.num_frames < 1:
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
        player = _VedoPlayer(scene, style, fps, quality=quality)
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
        scene: Scene,
        style: Style,
        fps: float,
        *,
        quality: str,
    ) -> None:
        from vedo import Plotter  # type: ignore[import-untyped]

        # vedo draws in perspective whatever the style asks.
        self.viewport = make_viewport(scene.views, projection="persp")
        center, half_span = self.viewport.center, self.viewport.half_span
        self.scene = scene
        self.style = style
        self.coords_list = [v.coords for v in scene.views]
        self.labels = scene.labels
        self.skeleton_lines_list = [v.bones for v in scene.views]
        self.center = center
        self.half_span = half_span
        self.up_axis = self.viewport.up_axis
        self.use_high = quality == "high"

        self.n_skeletons = scene.num_skeletons
        # Keep full-rate data for FPS resampling
        self._coords_full = [v.coords.copy() for v in scene.views]

        # Playback bookkeeping lives in the pure state machine; this
        # class owns rendering, UI, and event dispatch only.
        # self.num_frames is a property reading it — one owner.
        self.clock = PlaybackClock(scene.num_frames, fps)

        # --- UI state ---
        self.state: _UiState = {
            'timer_id': None,
            '_slider_updating': False,
            'show_labels': False,
            'skeleton_visible': [True] * self.n_skeletons,
            'show_trail': False,
            '_rendering': False,
            '_screenshot_hide_at': None,
        }

        self.plt = Plotter(
            title="pybvh viewer",
            size=(1400, 900),
            bg=style.background,
            offscreen=_FORCE_OFFSCREEN,
        )

        # Button registry: (x0, y0, w, h, callback) per clickable region.
        self._buttons: list[
            tuple[float, float, float, float, Callable[[], None]]] = []
        # Every UI overlay actor registers here so the clean-screenshot
        # mode can hide the lot in one pass.
        self._ui_actors: list = []

        self._build_geometry()
        self._build_ui()

        # Apply default FPS if it differs from native (needs the slider,
        # so this runs after _build_ui).
        if self.clock.target_fps != int(round(self.clock.native_fps)):
            self._set_fps(self.clock.fps_idx)

        # An offscreen plotter has no interactor: no events, no timers
        # (the offscreen path renders stills; playback needs a window).
        if self.plt.interactor is not None:
            self.plt.add_callback('LeftButtonPress', self._on_click)
            self.plt.add_callback('timer', self._on_timer)
            self.plt.add_callback('key press', self._on_key)
            if self.state['timer_id'] is None:
                self.state['timer_id'] = self.plt.timer_callback(
                    'create', dt=self.clock.interval_ms)

    @property
    def num_frames(self) -> int:
        """Frame count at the current FPS preset (owned by the clock)."""
        return self.clock.num_frames

    def show(self) -> None:
        # resetcam=False: the camera is the viewport's. vedo's default
        # would hand its distance and target back to VTK, which refits
        # them to everything in the scene, floor plane included.
        self.plt.show(resetcam=False)

    # =================================================================
    # GEOMETRY
    # =================================================================

    def _skeleton_color(self, s: int) -> tuple[float, float, float]:
        """Skeleton *s*'s label and trail color, in vedo's form."""
        return vedo_rgb(
            skeleton_color_255(self.style, s, self.n_skeletons))

    def _build_geometry(self) -> None:
        """Create the floor, skeleton actors, labels, camera, and trails."""
        from vedo import (  # type: ignore[import-untyped]
            Lines, Points, Grid, Text2D,
        )
        import vtk  # type: ignore[import-untyped]

        coords_list = self.coords_list
        n_skeletons = self.n_skeletons
        half_span = self.half_span
        up_idx = UP_AXIS_INDEX.get(self.up_axis, 2)

        # Base radius from the shared sizing formula, then adapted
        # per-bone by length inside CapsuleSkeleton.
        r_bone_base = CapsuleSkeleton.base_radius(
            half_span, self.style.bone_width)

        # --- Floor (high quality only; kind from the style) ---
        if self.use_high and self.style.floor is not None:
            from vedo import Plane  # type: ignore[import-untyped]

            # One floor for the whole scene, the viewport's: the same
            # plane the offscreen renderer draws, so viewer and render
            # agree.
            floor_pos, normal, side = floor_placement(self.viewport)
            palette = floor_palette(self.style)
            if self.style.floor == "solid":
                floor = Plane(
                    pos=tuple(floor_pos), normal=tuple(normal),
                    s=(side, side))
                floor.alpha(self.style.floor_alpha)
                floor.c(palette["face"]).lighting('off')
            else:
                # "grid" — and "checker", which falls back to grid in
                # this viewer (no cheap checker primitive in vedo).
                # Built at the origin, turned to face up, then moved:
                # vedo rotates about the world origin, so a grid that
                # is placed first swings away from where it was put.
                floor = Grid(s=[side, side], res=(30, 30))
                if self.up_axis == 'y':
                    floor.rotate_x(90)
                elif self.up_axis == 'x':
                    floor.rotate_y(90)
                # up_axis='z': Grid defaults to XY plane, no rotation
                floor.pos(*floor_pos)
                floor.lw(1).alpha(0.6)
                floor.c(palette["grid"]).lighting('off')
            self.plt += floor

        # --- Build persistent skeleton geometry (created once, updated in-place) ---

        # High mode: one CapsuleSkeleton (2 merged actors) per skeleton
        self._capsules: list[CapsuleSkeleton | None] = []

        # Fast mode: Lines + Points per skeleton
        self._lines_actors: list = []
        self._points_actors: list = []
        # Bone index arrays for the fast-mode vertex updates
        self._bone_parent_idx: list = []
        self._bone_child_idx: list = []

        # --- Create actors once and position to frame 0 ---
        for s in range(n_skeletons):
            view = self.scene.views[s]
            bone_rgb = bone_colors_255(view, self.style, s, n_skeletons)
            joint_rgb = node_colors_255(
                view, self.style, s, n_skeletons, bone_rgb)
            bones = self.skeleton_lines_list[s]
            self._bone_parent_idx.append(np.array([b[0] for b in bones]))
            self._bone_child_idx.append(np.array([b[1] for b in bones]))

            if self.use_high:
                capsule = CapsuleSkeleton(
                    view, r_bone_base, bone_rgb, joint_rgb,
                    flat_lighting=True)
                self._capsules.append(capsule)
                for actor_mesh in capsule.actors:
                    self.plt += actor_mesh

                # Position to frame 0
                capsule.update(coords_list[s][0])
            else:
                self._capsules.append(None)
                frame0 = coords_list[s][0]
                _lw = max(1, int(half_span * 0.04))
                _pr = max(1, int(half_span * 0.05))
                lines = Lines(
                    frame0[self._bone_parent_idx[s]],
                    frame0[self._bone_child_idx[s]],
                    lw=_lw)
                lines.cellcolors = np.asarray(bone_rgb, dtype=np.uint8)
                lines.lighting('off')
                points = Points(frame0, r=_pr, alpha=0.9)
                points.pointcolors = joint_rgb
                self._lines_actors.append(lines)
                self._points_actors.append(points)
                self.plt += lines
                self.plt += points

        # --- Labels ---
        if self.labels:
            for s in range(min(len(self.labels), n_skeletons)):
                if self.labels[s] is None:
                    continue
                label = Text2D(
                    self.labels[s],
                    pos=(0.02 + s * 0.15, 0.95),
                    c=self._skeleton_color(s), s=1.4, font='Calco',
                )
                self.plt += label

        # --- Camera: the viewport's, the one the offscreen renderer uses ---
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
            joint_names = self.scene.views[s].node_names
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
        # (fast vertex update).  The trail lies on the scene ground, in
        # both quality modes; the floor plane is drawn a hair below it.
        self._trail_actors: list = []
        self._trail_full: list[npt.NDArray] = []       # pre-computed root paths
        self._trail_collapsed: list[npt.NDArray] = []  # pre-allocated collapsed buffers
        for s in range(n_skeletons):
            root_all = self.viewport.ground_path(
                self._coords_full[s][:, 0, :])  # (F, 3)
            self._trail_full.append(root_all)
            # Pre-allocate collapsed buffer (reused every frame via .copy())
            collapsed = np.tile(root_all[0], (2 * (len(root_all) - 1), 1))
            self._trail_collapsed.append(collapsed)
            trail = Lines(collapsed[::2], collapsed[1::2],
                          lw=2, c=self._skeleton_color(s), alpha=0.6)
            trail.lighting('off')
            trail.actor.SetVisibility(0)
            self._trail_actors.append(trail)
            self.plt += trail

    def _update_skeleton_fast(self, s: int, frame_data: npt.NDArray) -> None:
        """Update Lines/Points vertex data in-place for skeleton *s*."""
        p_idx = self._bone_parent_idx[s]
        c_idx = self._bone_child_idx[s]
        self._lines_actors[s].vertices = _interleave(
            frame_data[p_idx], frame_data[c_idx])
        self._points_actors[s].vertices = frame_data

    def _set_camera(self) -> None:
        """Put the camera where the viewport says, exactly.

        Only the clipping planes are left to VTK: they decide what is
        cut off in depth, not what is framed. VTK fits them to what is
        in the scene at the moment, so :meth:`_update_frame` refits
        them whenever the skeletons move."""
        eye, target, up = self.viewport.camera()
        self.plt.camera.SetPosition(*eye)
        self.plt.camera.SetFocalPoint(*target)
        self.plt.camera.SetViewUp(*up)
        self.plt.renderer.ResetCameraClippingRange()

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
        self._ui_actors.append(t2d)
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
        self._ui_actors.append(self.speed_label)
        self._add_button(" < ", 0.05, 0.89, 0.03, 0.07, self._on_speed_down)
        self.speed_text = Text2D(
            " 1x ", pos=(0.08, 0.92), s=_PANEL_S,
            c='#2c3e50', bg='#c8c8d4', font='Calco',
        )
        self.plt += self.speed_text
        self._ui_actors.append(self.speed_text)
        self._add_button(" > ", 0.12, 0.89, 0.04, 0.07, self._on_speed_up)

        # --- FPS selector ---
        self.fps_label = Text2D(
            "FPS", pos=(_PANEL_X, 0.86), s=_PANEL_S,
            c='#2c3e50', font='Calco',
        )
        self.plt += self.fps_label
        self._ui_actors.append(self.fps_label)
        self._add_button(" < ", 0.05, 0.83, 0.03, 0.06, self._on_fps_down)
        self.fps_text = Text2D(
            f" {self.clock.target_fps} ", pos=(0.08, 0.86),
            s=_PANEL_S, c='#2c3e50', bg='#c8c8d4', font='Calco',
        )
        self.plt += self.fps_text
        self._ui_actors.append(self.fps_text)
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
        self._ui_actors.append(self._help_header)

        self._help_items: list = []
        for i, txt in enumerate(_HELP_ENTRIES):
            t = Text2D(txt, pos=(_RPANEL_X, 0.86 - i * 0.045), s=1.1,
                       c='#2c3e50', font='Calco')
            t.actor.SetVisibility(0)
            self._help_items.append(t)
            self.plt += t
            self._ui_actors.append(t)

        # --- Screenshot feedback overlay (center-top, hidden by default) ---
        self._screenshot_text = Text2D(
            "", pos=(0.35, 0.92), s=1.2,
            c='white', bg='green4', font='Calco',
        )
        self._screenshot_text.actor.SetVisibility(0)
        self.plt += self._screenshot_text
        self._ui_actors.append(self._screenshot_text)

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

    def _sync_all(self) -> None:
        """Sync all UI elements to match the playback clock."""
        # Play/pause button
        if self.clock.playing:
            self.btn_play.text(_L_PAUSE)
            self.btn_play.background('tomato')
        else:
            self.btn_play.text(_L_PLAY)
            self.btn_play.background('green4')
        # Loop / ping-pong button
        mode = self.clock.loop_mode
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
        spd = self.clock.speed
        self.speed_text.text(
            f" {spd:.1f}x " if spd != int(spd) else f" {int(spd)}x ")

    def _update_frame_display(self, f: int) -> None:
        """Update frame info in the window title bar."""
        t = f * self.clock.step / self.clock.native_fps
        self.plt.window.SetWindowName(
            f"pybvh viewer  |  Frame {f}/{self.num_frames - 1}"
            f"  |  t={t:.2f}s  |  {self.clock.target_fps}fps"
            f"  |  {self.clock.speed:.3g}x")

    def _restart_timer(self) -> None:
        """(Re)create the render timer at the clock's current interval."""
        if self.plt.interactor is None:
            return
        if self.state['timer_id'] is not None:
            self.plt.timer_callback('destroy', self.state['timer_id'])
        self.state['timer_id'] = self.plt.timer_callback(
            'create', dt=self.clock.interval_ms)

    def _set_speed(self, new_speed: float) -> None:
        """Change playback speed and restart the timer."""
        self.clock.set_speed(new_speed)
        self._sync_all()
        self._restart_timer()

    def _set_fps(self, idx: int) -> None:
        """Change FPS preset and resample coordinate data.

        The clock owns the fps-to-step formula; the coords are sliced
        with clock.step so the two can never disagree.
        """
        self.clock.set_fps_index(idx, 1)   # frame count set below
        self.coords_list = [c[::self.clock.step]
                            for c in self._coords_full]
        self.clock.num_frames = self.coords_list[0].shape[0]
        self.fps_text.text(f" {self.clock.target_fps} ")
        self.state['_slider_updating'] = True
        self.slider.GetRepresentation().SetMinimumValue(0)
        self.slider.GetRepresentation().SetMaximumValue(self.num_frames - 1)
        self.slider.value = 0
        self.state['_slider_updating'] = False
        self._restart_timer()
        self._sync_all()
        self._update_frame(0)

    def _jump_to(self, f: int) -> None:
        """Jump to frame f, pause, and sync UI."""
        f = self.clock.jump_to(f)
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
                self._capsules[s].update(frame_data)
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
                full_f = min(f * self.clock.step, len(root_pts) - 1)
                verts = self._trail_collapsed[s].copy()
                if full_f > 0:
                    visible = _interleave(
                        root_pts[:full_f], root_pts[1:full_f + 1])
                    verts[:len(visible)] = visible
                self._trail_actors[s].vertices = verts
        # The clipping planes were fitted to the previous pose; a
        # skeleton that walked toward or away from the camera since
        # would be cut off in depth.
        self.plt.renderer.ResetCameraClippingRange()

        # Hide screenshot feedback after timeout
        hide_at = self.state.get('_screenshot_hide_at')
        if hide_at and time.perf_counter() > hide_at:
            self._screenshot_text.actor.SetVisibility(0)
            self.state['_screenshot_hide_at'] = None

        self._update_frame_display(f)
        self.plt.render()

    # =================================================================
    # SCREENSHOT
    # =================================================================

    def screenshot(
        self,
        fname: str | None = None,
        *,
        clean: bool = True,
        scale: int = 2,
    ) -> str:
        """Save a screenshot of the current frame.

        With ``clean=True`` (default) every control-panel overlay —
        buttons, help, speed/FPS readouts, and the frame slider — is
        hidden for the capture and restored afterwards, and the image
        is rendered at ``scale`` x the window resolution. Skeleton
        labels stay visible (they are content, not chrome).
        """
        if fname is None:
            fname = f"pybvh_frame_{self.clock.frame}.png"
        if not clean:
            self.plt.screenshot(fname)
            return fname

        hidden = []
        for actor_obj in self._ui_actors:
            vtk_actor = getattr(actor_obj, 'actor', actor_obj)
            if vtk_actor.GetVisibility():
                vtk_actor.SetVisibility(0)
                hidden.append(vtk_actor)
        slider_was_on = bool(self.slider.GetEnabled())
        if slider_was_on:
            self.slider.EnabledOff()
        try:
            self.plt.render()
            self.plt.screenshot(fname, scale=scale)
        finally:
            for vtk_actor in hidden:
                vtk_actor.SetVisibility(1)
            if slider_was_on:
                self.slider.EnabledOn()
            self.plt.render()
        return fname

    # =================================================================
    # BUTTON / KEY ACTIONS
    # =================================================================

    def _on_first(self) -> None:
        self._jump_to(0)
        self._update_frame(0)

    def _on_prev(self) -> None:
        self._jump_to(self.clock.frame - 1)
        self._update_frame(self.clock.frame)

    def _toggle_play(self) -> None:
        self.clock.toggle_play()
        self._sync_all()
        self.plt.render()

    def _on_next(self) -> None:
        self._jump_to(self.clock.frame + 1)
        self._update_frame(self.clock.frame)

    def _on_last(self) -> None:
        self._jump_to(self.num_frames - 1)
        self._update_frame(self.num_frames - 1)

    def _on_speed_down(self) -> None:
        self._set_speed(self.clock.speed / 2)
        self.plt.render()

    def _on_speed_up(self) -> None:
        self._set_speed(self.clock.speed * 2)
        self.plt.render()

    def _on_fps_down(self) -> None:
        if self.clock.fps_idx > 0:
            self._set_fps(self.clock.fps_idx - 1)

    def _on_fps_up(self) -> None:
        if self.clock.fps_idx < len(self.clock.fps_presets) - 1:
            self._set_fps(self.clock.fps_idx + 1)

    def _on_cycle_loop(self) -> None:
        self.clock.cycle_loop()
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
        # Skip if a previous render is still in progress. Wall-clock
        # frame advancement (accurate under dropped timer events) lives
        # in PlaybackClock.advance.
        state = self.state
        if state.get('_rendering'):
            return
        state['_rendering'] = True
        try:
            was_playing = self.clock.playing
            target_f = self.clock.advance(time.perf_counter())
            if was_playing and not self.clock.playing:
                self._sync_all()  # 'off' mode reached an end
            if target_f is not None:
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
        self.clock.jump_to(f)
        self._sync_all()
        self._update_frame(self.clock.frame)

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
            self._set_fps((self.clock.fps_idx + 1)
                          % len(self.clock.fps_presets))

        elif key == 'j':
            # Toggle joint name labels
            state['show_labels'] = not state['show_labels']
            vis = 1 if state['show_labels'] else 0
            for s in range(self.n_skeletons):
                frame_data = self.coords_list[s][self.clock.frame]
                for j in range(len(frame_data)):
                    self._label_actors[s][j].SetVisibility(vis)
                    if vis:
                        self._label_actors[s][j].SetPosition(
                            *(frame_data[j] + self._label_offset))
            self.plt.render()

        elif key == 's':
            # Clean screenshot (UI hidden, 2x resolution) with feedback
            fname = self.screenshot()
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
                    capsule = self._capsules[idx]
                    assert capsule is not None
                    for mesh in capsule.actors:
                        mesh.actor.SetVisibility(v)
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
