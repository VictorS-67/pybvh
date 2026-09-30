"""Every backend draws a Scene built from arrays, with no Bvh in sight.

The interface is the test surface: if a backend needed anything a
SkeletonView does not carry, these are the tests that would fail. The
Scene comes from ``tests/synthetic_scene.py``, which imports nothing
that knows what a Bvh is.
"""
from __future__ import annotations

import ast
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import synthetic_scene
from pybvh.bvhplot._style import Style
from synthetic_scene import make_array_scene


def test_factory_knows_nothing_about_bvh():
    tree = ast.parse(pathlib.Path(synthetic_scene.__file__).read_text())
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"__future__", "numpy", "pybvh.bvhplot._scene"}


@pytest.fixture(scope="module")
def scene():
    return make_array_scene(n_frames=12)


@pytest.fixture(scope="module")
def pair():
    return make_array_scene(n_frames=12, n_skeletons=2, labels=["a", "b"])


class TestMatplotlib:
    def test_frame(self, scene):
        from pybvh.bvhplot._matplotlib import frame_mpl
        fig, ax = frame_mpl(scene, Style("paper"))
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_frame_pair_with_labels(self, pair):
        from pybvh.bvhplot._matplotlib import frame_mpl
        fig, axes = frame_mpl(pair, Style("debug"))
        assert len(axes) == 2
        plt.close(fig)

    def test_sequence(self, scene):
        from pybvh.bvhplot._matplotlib import sequence_mpl
        samples = np.array([0, 5, 11], dtype=np.intp)
        fig, ax = sequence_mpl(scene, Style("paper"), samples, "offset")
        plt.close(fig)

    def test_render_with_ghosts_trace_and_follow(self, scene, tmp_path):
        from pybvh.bvhplot._matplotlib import render_mpl
        out = render_mpl(scene, Style("paper"), tmp_path / "walk.gif", 10.0,
                         motion="follow", ghost=2, trajectory=True,
                         resolution=(320, 240))
        assert out.exists() and out.stat().st_size > 0

    def test_trajectory_with_facing_arrows(self, pair):
        from pybvh.bvhplot._matplotlib import trajectory_mpl
        fig, ax = trajectory_mpl(pair, Style("paper"), facing_arrows=True)
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_each_facing_arrow_points_where_the_heading_says(self):
        """A y-up view is drawn on the (x, z) plane, and its heading is
        [sin, cos] with cos along x and sin along z. Over four frames
        the heading turns a quarter turn each frame, independently of
        the walk, so an arrow drawn from any other frame's heading, or
        with sin and cos swapped, points the wrong way."""
        import dataclasses
        from matplotlib.quiver import Quiver
        from pybvh.bvhplot._matplotlib import trajectory_mpl
        from pybvh.bvhplot._scene import Scene
        view = synthetic_scene.make_array_view(n_frames=4)
        quarter_turns = np.array([[1.0, 0.0],    # faces +z
                                  [0.0, -1.0],   # faces -x
                                  [-1.0, 0.0],   # faces -z
                                  [0.0, 1.0]])   # faces +x
        plot_directions = np.array([[0.0, 1.0],  # (x, z) on the plot
                                    [-1.0, 0.0],
                                    [0.0, -1.0],
                                    [1.0, 0.0]])
        turning = dataclasses.replace(view, root_heading=quarter_turns)
        fig, ax = trajectory_mpl(
            Scene(views=[turning]), Style("paper"), facing_arrows=True)
        try:
            (arrows,) = [c for c in ax.collections if isinstance(c, Quiver)]
            roots = view.coords[:, 0][:, [0, 2]]
            directions = np.stack([arrows.U, arrows.V], axis=1)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            assert len(directions) > 1
            for start, direction in zip(arrows.get_offsets(), directions):
                (drawn_at,) = np.flatnonzero(
                    np.all(np.isclose(roots, start), axis=1))
                np.testing.assert_allclose(
                    direction, plot_directions[drawn_at], atol=1e-12)
        finally:
            plt.close(fig)


class TestOpenCV:
    def test_render_with_every_option(self, scene, tmp_path):
        pytest.importorskip("cv2")
        from pybvh.bvhplot._opencv import render_opencv
        out = render_opencv(scene, Style("paper"), tmp_path / "walk.mp4",
                            10.0, (320, 240), motion="follow", ghost=1,
                            trajectory=True, frame_counter=True)
        assert out.exists() and out.stat().st_size > 0


def _gif_frames(path):
    from PIL import Image, ImageSequence
    with Image.open(path) as gif:
        return [np.asarray(frame.convert("RGB"))
                for frame in ImageSequence.Iterator(gif)]


class TestEveryPassIsDrawnAsTheFirst:
    """A looped Scene replays its clip: under a fixed camera, a frame
    of a later pass shows exactly what the same frame of the first pass
    showed, so no ghost, root-trace segment or frame count reaches back
    across the seam to the end of the previous pass."""

    CLIP = 12
    LOOPED = 30  # two whole passes and a partial third
    # 0.1 s at 30 fps: ghosts trail by 3 and 6 frames, so the first
    # frames of a pass would show the previous pass's end.
    STYLE = Style("paper", ghost_spacing=0.1, supersample=1)

    @pytest.fixture
    def looped(self):
        return make_array_scene(n_frames=self.CLIP).looped(self.LOOPED)

    def test_opencv(self, looped):
        pytest.importorskip("cv2")
        from pybvh.bvhplot._opencv import _generate_frames
        frames = list(_generate_frames(
            looped, self.STYLE, (160, 120), ghost=2, trajectory=True,
            frame_counter=True))
        assert len(frames) == self.LOOPED
        for f in range(self.CLIP, self.LOOPED):
            np.testing.assert_array_equal(
                frames[f], frames[f % self.CLIP], err_msg=f"frame {f}")

    def test_matplotlib(self, looped, tmp_path):
        from pybvh.bvhplot._matplotlib import render_mpl
        out = render_mpl(looped, self.STYLE, tmp_path / "loop.gif", 10.0,
                         ghost=2, trajectory=True, resolution=(160, 120))
        frames = _gif_frames(out)
        assert len(frames) == self.LOOPED
        for f in range(self.CLIP, self.LOOPED):
            np.testing.assert_array_equal(
                frames[f], frames[f % self.CLIP], err_msg=f"frame {f}")


class TestVedo:
    def test_offscreen_frame(self, scene):
        pytest.importorskip("vedo")
        from pybvh.bvhplot._vedo_offscreen import frame_vedo
        img = frame_vedo(scene, Style("paper"), resolution=(200, 200))
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_offscreen_render(self, scene, tmp_path):
        pytest.importorskip("vedo")
        from pybvh.bvhplot._vedo_offscreen import render_vedo
        out = render_vedo(scene, Style("paper"), tmp_path / "walk.mp4",
                          10.0, resolution=(200, 200))
        assert out.exists() and out.stat().st_size > 0

    def test_the_viewers_trails_lie_on_the_scene_ground(
            self, pair, monkeypatch):
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        scene = pair.spread("auto")
        player = _vedo._VedoPlayer(scene, Style("paper"), 30.0,
                                   quality="high")
        try:
            up = player.viewport.up_index
            assert len(player._trail_full) == 2
            for path in player._trail_full:
                np.testing.assert_array_equal(
                    path[:, up], player.viewport.floor_height)
        finally:
            player.plt.close()

    def test_viewer_shell_builds_from_the_pair(self, pair, monkeypatch):
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        player = _vedo._VedoPlayer(pair.spread("auto"), Style("paper"), 30.0,
                                   quality="high")
        try:
            assert player.scene.num_skeletons == 2
        finally:
            player.plt.close()

    @pytest.fixture
    def open_viewer(self, monkeypatch):
        """Open a headless viewer on a scene; closed after the test."""
        pytest.importorskip("vedo")
        players = []

        def open_(scene, quality="high"):
            players.append(_viewer(scene, monkeypatch, quality))
            return players[-1]

        yield open_
        for player in players:
            player.plt.close()

    @staticmethod
    def _scrub_to(player, frame):
        """Move the frame slider as a user's drag does: set its value,
        then fire the event its callback listens to."""
        player.slider.value = frame
        player.slider.InvokeEvent("InteractionEvent")

    @staticmethod
    def _drawn_joints(player, s):
        """Skeleton *s*'s joint positions as the viewer drew them."""
        if player.use_high:
            # one sphere per joint, merged; a sphere's bounding-box
            # midpoint is its center
            n_nodes = len(player.scene.views[s].node_names)
            spheres = np.asarray(
                player._capsules[s].joints_mesh.vertices).reshape(
                    n_nodes, -1, 3)
            return (spheres.min(axis=1) + spheres.max(axis=1)) / 2
        return np.asarray(player._points_actors[s].vertices)

    @pytest.mark.parametrize("quality", ["high", "fast"])
    def test_the_frame_slider_poses_each_skeleton_at_its_frame(
            self, pair, quality, open_viewer):
        scene = pair.spread("auto")
        player = open_viewer(scene, quality)
        self._scrub_to(player, 7)
        assert player.clock.frame == 7
        for s, view in enumerate(scene.views):
            # VTK keeps float32 vertices
            np.testing.assert_allclose(
                self._drawn_joints(player, s), view.coords[7], atol=1e-5)

    def test_at_half_the_clip_rate_the_slider_steps_two_clip_frames(
            self, scene, open_viewer):
        """The 30 fps clip played at the 15 fps preset: slider position
        5 is clip frame 10."""
        player = open_viewer(scene, "fast")
        player._set_fps(player.clock.fps_presets.index(15))
        self._scrub_to(player, 5)
        np.testing.assert_allclose(
            self._drawn_joints(player, 0), scene.views[0].coords[10],
            atol=1e-5)


class TestK3d:
    def test_play_builds_the_plot(self, pair, capsys):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import play_k3d
        # Outside a notebook IPython's display() prints the widget's repr;
        # the point here is only that the backend needs nothing beyond
        # the Scene.
        play_k3d(pair.spread("auto"), Style("paper"), 30.0)
        capsys.readouterr()

    def test_the_camera_is_the_viewports(self, pair):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        from pybvh.bvhplot._viewport import make_viewport
        scene = pair.spread("auto")
        built = _build_plot(scene, Style("paper"))
        eye, target, up = make_viewport(scene.views).camera()
        np.testing.assert_allclose(
            built.plot.camera, [*eye, *target, *up], rtol=1e-6)
        assert built.plot.camera_auto_fit is False

    def test_the_trails_lie_on_the_scene_ground(self, pair):
        """... and the grid's bottom face is put just under it, so the
        trail is seen on that face, not floating inside the box."""
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        from pybvh.bvhplot._viewport import FLOOR_INSET
        built = _build_plot(pair.spread("auto"), Style("paper"))
        viewport = built.viewport
        up = viewport.up_index
        for path, view in zip(built.trail_paths, pair.spread("auto").views):
            np.testing.assert_allclose(path[:, up], viewport.floor_height)
            ground = list(viewport.ground_axes)
            np.testing.assert_allclose(
                path[:, ground], view.coords[:, 0][:, ground], rtol=1e-6)
        grid = np.asarray(built.plot.grid).reshape(2, 3)
        assert grid[0, up] == pytest.approx(
            viewport.floor_height - FLOOR_INSET * viewport.half_span)
        assert grid[1, up] == pytest.approx(
            viewport.center[up] + viewport.half_span)
        assert built.plot.grid_auto_fit is False

    def test_a_negative_up_axis_keeps_the_trail_under_the_feet(self):
        pytest.importorskip("k3d")
        import dataclasses
        from pybvh.bvhplot._k3d import _build_plot
        from pybvh.bvhplot._scene import Scene
        view = synthetic_scene.make_array_view(n_frames=12)
        flipped = view.coords * np.array([1.0, -1.0, 1.0])
        negative = dataclasses.replace(
            view, coords=flipped, up="-y",
            rest_coords=view.rest_coords * np.array([1.0, -1.0, 1.0]),
            floor_height=float(flipped[..., 1].max()))
        built = _build_plot(Scene(views=[negative]), Style("paper"))
        # "under the feet" of a -y-up rig is the coordinate maximum
        assert np.all(built.trail_paths[0][:, 1] >= flipped[..., 1].max() - 1e-6)

    def test_the_floor_is_the_viewports_plane(self, pair):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import FLOOR_EPSILON, _build_plot
        built = _build_plot(pair.spread("auto"), Style("paper"))
        viewport = built.viewport
        up = viewport.up_index
        corners = np.asarray(built.floor.vertices).reshape(-1, 3)
        expected = viewport.floor_quad()
        ground = list(viewport.ground_axes)
        np.testing.assert_allclose(
            corners[:, ground], expected[:, ground], rtol=1e-6)
        # a hair below the ground, and so below the trail
        below = viewport.floor_height - FLOOR_EPSILON * viewport.half_span
        np.testing.assert_allclose(corners[:, up], below, rtol=1e-6)
        assert np.all(built.trail_paths[0][:, up] > corners[:, up].max())
        assert built.floor.opacity == pytest.approx(Style("paper").floor_alpha)

    @pytest.mark.parametrize("kind", ["grid", "checker"])
    def test_a_grid_floor_spans_the_same_plane(self, pair, kind):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import FLOOR_GRID_LINES, _build_plot
        built = _build_plot(
            pair.spread("auto"), Style("paper", floor=kind))
        viewport = built.viewport
        vertices = np.asarray(built.floor.vertices).reshape(-1, 3)
        # k3d keeps indices in a float32 trait
        indices = np.asarray(built.floor.indices).reshape(-1, 2).astype(int)
        assert len(indices) == 2 * FLOOR_GRID_LINES
        expected = viewport.floor_quad()
        for axis in viewport.ground_axes:
            assert vertices[:, axis].min() == pytest.approx(
                expected[:, axis].min(), rel=1e-6)
            assert vertices[:, axis].max() == pytest.approx(
                expected[:, axis].max(), rel=1e-6)
        # every line runs from one edge of the plane to the opposite one
        lengths = np.linalg.norm(
            vertices[indices[:, 1]] - vertices[indices[:, 0]], axis=1)
        np.testing.assert_allclose(
            lengths, 2 * viewport.floor_reach, rtol=1e-5)

    def test_a_style_without_a_floor_draws_none(self, pair):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        built = _build_plot(pair.spread("auto"), Style("debug"))
        assert built.floor is None

    @pytest.mark.parametrize("preset, grid, label", [
        # k3d's own defaults, which the paper look has always shown
        ("paper", 0xE6E6E6, 0x444444),
        # the same steps away from #16181D, toward white
        ("dark", 0x2D2F33, 0xC1C1C3),
    ])
    def test_the_grid_box_takes_its_colors_from_the_style(
            self, pair, preset, grid, label):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        built = _build_plot(pair.spread("auto"), Style(preset))
        assert f"{built.plot.grid_color:06X}" == f"{grid:06X}"
        assert f"{built.plot.label_color:06X}" == f"{label:06X}"

    def test_one_skeleton_and_one_trail_per_view(self, pair):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        built = _build_plot(pair.spread("auto"), Style("paper"))
        assert len(built.skeletons) == 2
        assert len(built.trails) == 2
        for path, view in zip(built.trail_paths, pair.spread("auto").views):
            assert path.shape == (view.coords.shape[0], 3)

    @staticmethod
    def _play_and_grab_slider(scene, monkeypatch):
        """Play *scene* with k3d outside a notebook: the plot as built
        and the frame slider of the widget that would be displayed."""
        pytest.importorskip("k3d")
        import IPython.display
        from pybvh.bvhplot import _k3d
        build_plot = _k3d._build_plot
        shown, built = [], []

        def build_and_keep(*args):
            built.append(build_plot(*args))
            return built[-1]

        monkeypatch.setattr(_k3d, "_build_plot", build_and_keep)
        monkeypatch.setattr(IPython.display, "display", shown.append)
        _k3d.play_k3d(scene, Style("paper"), 30.0)
        (widget,) = shown
        _plot, controls = widget.children
        (slider,) = [w for w in controls.children
                     if type(w).__name__ == "IntSlider"]
        return built[0], slider

    def test_the_frame_slider_poses_each_skeleton_at_its_frame(
            self, pair, monkeypatch):
        scene = pair.spread("auto")
        built, slider = self._play_and_grab_slider(scene, monkeypatch)
        slider.value = 7
        for (lines, points), view in zip(built.skeletons, scene.views):
            expected = view.coords[7].astype(np.float32)
            np.testing.assert_array_equal(lines.vertices, expected)
            np.testing.assert_array_equal(points.positions, expected)

    def test_the_frame_slider_grows_each_trail_to_its_frame(
            self, pair, monkeypatch):
        """The trail is the root's ground path up to the frame, and its
        later vertices wait at the frame's point."""
        scene = pair.spread("auto")
        built, slider = self._play_and_grab_slider(scene, monkeypatch)
        slider.value = 7
        for trail, path in zip(built.trails, built.trail_paths):
            drawn = np.asarray(trail.vertices)
            np.testing.assert_array_equal(drawn[:8], path[:8])
            np.testing.assert_array_equal(
                drawn[8:], np.broadcast_to(path[7], drawn[8:].shape))

    @pytest.mark.parametrize("preset, spine", [
        ("paper", 0x3A3F4A),
        ("dark", 0xC8CCD6),    # lightened to read on the dark ground
    ])
    def test_one_skeleton_colors_each_node_by_its_chain(
            self, scene, preset, spine):
        """A node takes its parent bone's chain color, the root the
        spine's: left warm, right cool (Okabe-Ito)."""
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        l_arm, r_arm, l_leg, r_leg = 0xE69F00, 0x56B4E9, 0xD55E00, 0x0072B2
        expected = [
            spine,           # Hips, the root
            spine, spine,    # Spine, Head
            l_arm, l_arm,    # LeftArm, LeftHand
            r_arm, r_arm,    # RightArm, RightHand
            l_leg, r_leg,    # LeftFoot, RightFoot
        ]
        built = _build_plot(scene, Style(preset))
        (lines, points), = built.skeletons
        for drawn in (lines.colors, points.colors):
            assert [f"{c:06X}" for c in drawn] == [
                f"{c:06X}" for c in expected]


# ---------------------------------------------------------------------------
# Every backend draws the viewport
# ---------------------------------------------------------------------------
# One thin check per backend that the adapter puts things where the
# viewport says. The viewport's own numbers are tested in
# tests/test_viewport.py; what is tested here is the translation into
# each toolkit, which is where the backends used to drift apart.

def _viewer(scene, monkeypatch, quality="high"):
    from pybvh.bvhplot import _vedo
    monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
    return _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality=quality)


def _vedo_plane(plotter):
    planes = [o for o in plotter.objects if type(o).__name__ == "Plane"]
    assert len(planes) == 1, f"expected one floor plane, got {len(planes)}"
    return np.asarray(planes[0].vertices)


def _floor_corners(backend, scene, monkeypatch):
    """World-space corners of the floor *backend* draws for *scene*."""
    if backend == "matplotlib":
        from mpl_toolkits.mplot3d import art3d
        from pybvh.bvhplot._matplotlib import frame_mpl

        # matplotlib keeps a 3-D collection's world-space vertices only
        # in private attributes, renamed between releases (`_vec` in
        # 3.9, gone in 3.11). Record the vertices the backend hands it
        # instead: those are the floor the backend computed.
        class RecordingPoly3DCollection(art3d.Poly3DCollection):
            def __init__(self, verts, *args, **kwargs):
                self.world_verts = np.asarray(verts, dtype=np.float64)
                super().__init__(verts, *args, **kwargs)

        monkeypatch.setattr(art3d, "Poly3DCollection", RecordingPoly3DCollection)
        fig, ax = frame_mpl(scene, Style("paper"), show=False)
        floors = [c for c in ax.collections if c.get_zorder() == 0.5]
        assert len(floors) == 1
        corners = floors[0].world_verts.reshape(-1, 3)
        plt.close(fig)
        return corners
    if backend == "vedo offscreen":
        pytest.importorskip("vedo")
        from pybvh.bvhplot._vedo_offscreen import _build_offscreen
        plotter, _, _ = _build_offscreen(scene, Style("paper"), (200, 200))
        corners = _vedo_plane(plotter)
        plotter.close()
        return corners
    if backend == "vedo viewer":
        pytest.importorskip("vedo")
        player = _viewer(scene, monkeypatch)
        corners = _vedo_plane(player.plt)
        player.plt.close()
        return corners
    assert backend == "k3d"
    pytest.importorskip("k3d")
    from pybvh.bvhplot._k3d import _build_plot
    built = _build_plot(scene, Style("paper"))
    return np.asarray(built.floor.vertices, dtype=np.float64).reshape(-1, 3)


# The z-fighting epsilon each toolkit declares, in half-spans: how far
# below the scene ground it draws the plane.
def _declared_epsilon(backend):
    if backend == "matplotlib":
        return 0.0
    if backend == "k3d":
        from pybvh.bvhplot._k3d import FLOOR_EPSILON
        return FLOOR_EPSILON
    from pybvh.bvhplot._vedo_capsules import FLOOR_EPSILON
    return FLOOR_EPSILON


_FLOOR_BACKENDS = ["matplotlib", "vedo offscreen", "vedo viewer", "k3d"]


class TestEveryBackendDrawsTheViewportsFloor:
    @pytest.mark.parametrize("backend", _FLOOR_BACKENDS)
    def test_the_plane_is_where_the_viewport_puts_it(
            self, backend, scene, monkeypatch):
        from pybvh.bvhplot._viewport import make_viewport
        viewport = make_viewport(scene.views, framing="still")
        corners = _floor_corners(backend, scene, monkeypatch)
        expected = viewport.floor_quad()
        up = viewport.up_index

        for axis in viewport.ground_axes:
            assert corners[:, axis].min() == pytest.approx(
                expected[:, axis].min(), rel=1e-5), backend
            assert corners[:, axis].max() == pytest.approx(
                expected[:, axis].max(), rel=1e-5), backend
        below = viewport.below_floor(
            _declared_epsilon(backend) * viewport.half_span)
        np.testing.assert_allclose(
            corners[:, up], below, rtol=1e-5, atol=1e-7, err_msg=backend)

    @pytest.mark.parametrize("backend", _FLOOR_BACKENDS)
    def test_moving_the_scene_ground_moves_the_plane(
            self, backend, scene, monkeypatch):
        """The property a floor re-derived inside a backend breaks."""
        import dataclasses
        from pybvh.bvhplot._scene import Scene
        view = scene.views[0]
        moved = Scene(views=[dataclasses.replace(
            view, floor_height=view.floor_height - 0.75)])
        before = _floor_corners(backend, scene, monkeypatch)
        after = _floor_corners(backend, moved, monkeypatch)
        up = view.up_index
        assert before[:, up].mean() - after[:, up].mean() == pytest.approx(
            0.75, abs=1e-5), backend

    def test_opencv_projects_the_viewports_plane(self, scene, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        from pybvh.bvhplot import _opencv
        from pybvh.bvhplot._viewport import make_viewport
        drawn = []
        real = cv2.fillPoly

        def capture(img, polygons, *args, **kwargs):
            drawn.append(np.asarray(polygons[0]).copy())
            return real(img, polygons, *args, **kwargs)

        monkeypatch.setattr(cv2, "fillPoly", capture)
        style = Style("paper", supersample=1)
        next(_opencv._generate_frames(scene, style, (320, 240)))

        viewport = make_viewport(scene.views, framing="clip")
        expected = viewport.project(viewport.floor_quad(), (320, 240), 0)
        assert len(drawn) == 1
        np.testing.assert_array_equal(drawn[0], expected)


class TestEveryBackendFramesTheViewportsBox:
    @pytest.mark.parametrize("floor", ["solid", None])
    def test_a_matplotlib_still_is_fitted_to_the_still_box(self, scene, floor):
        from pybvh.bvhplot._matplotlib import frame_mpl
        from pybvh.bvhplot._viewport import make_viewport
        fig, ax = frame_mpl(scene, Style("paper", floor=floor), show=False)
        viewport = make_viewport(
            scene.views, framing="still", include_floor=floor is not None)
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        np.testing.assert_array_equal(limits[:, 0], viewport.lo)
        np.testing.assert_array_equal(limits[:, 1], viewport.hi)
        plt.close(fig)

    @pytest.mark.parametrize("motion", ["fixed", "turntable"])
    def test_a_matplotlib_clip_is_fitted_to_the_clip_box(self, scene, motion):
        from pybvh.bvhplot._matplotlib import _setup_animated_panel
        from pybvh.bvhplot._viewport import make_viewport
        view = scene.views[0]
        viewport = make_viewport([view], framing="clip", motion=motion)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        _setup_animated_panel(ax, view, viewport, Style("paper"),
                              np.asarray(view.bones, dtype=int), 0, 1)
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        np.testing.assert_array_equal(limits[:, 0], viewport.lo)
        np.testing.assert_array_equal(limits[:, 1], viewport.hi)
        plt.close(fig)


def _drawn_projection(ax):
    """What a 3D axes is drawn with. mplot3d has no getter; its focal
    length is infinite for an orthographic projection (private, like
    ``_vec`` above: a rename fails these tests loudly)."""
    return "ortho" if np.isinf(ax._focal_length) else "persp"


class TestEveryBackendStatesItsProjection:
    """``viewport.projection`` describes the picture: the adapter that
    draws says what it draws with."""

    @pytest.mark.parametrize("asked", ["persp", "ortho"])
    def test_matplotlib_draws_what_the_style_asks(self, scene, asked):
        from pybvh.bvhplot._matplotlib import frame_mpl, sequence_mpl
        style = Style("paper", projection=asked)
        fig, ax = frame_mpl(scene, style, show=False)
        assert _drawn_projection(ax) == asked
        plt.close(fig)
        samples = np.array([0, 5, 11], dtype=np.intp)
        fig, ax = sequence_mpl(scene, style, samples, "overlay")
        assert _drawn_projection(ax) == asked
        plt.close(fig)

    @pytest.mark.parametrize("asked", ["persp", "ortho"])
    def test_the_offset_sequence_is_always_orthographic(self, scene, asked):
        from pybvh.bvhplot._matplotlib import sequence_mpl
        samples = np.array([0, 5, 11], dtype=np.intp)
        fig, ax = sequence_mpl(
            scene, Style("paper", projection=asked), samples, "offset")
        assert _drawn_projection(ax) == "ortho"
        plt.close(fig)

    @pytest.mark.parametrize("asked", ["persp", "ortho"])
    def test_a_matplotlib_clip_draws_its_viewports_projection(
            self, scene, asked):
        from pybvh.bvhplot._matplotlib import _setup_animated_panel
        from pybvh.bvhplot._viewport import make_viewport
        view = scene.views[0]
        viewport = make_viewport([view], framing="clip", projection=asked)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # the style asks for the opposite: the viewport is what is drawn
        other = "ortho" if asked == "persp" else "persp"
        _setup_animated_panel(ax, view, viewport,
                              Style("paper", projection=other),
                              np.asarray(view.bones, dtype=int), 0, 1)
        assert _drawn_projection(ax) == asked
        plt.close(fig)

    @pytest.mark.parametrize("make_axes, asked", [
        (dict(proj_type="ortho"), "persp"),
        (dict(proj_type="persp"), "ortho"),
        (dict(), "ortho"),
    ])
    def test_supplied_axes_are_drawn_with_the_styles_projection(
            self, scene, make_axes, asked):
        """``ax=`` hands over axes that already have a projection; the
        style still decides, so the viewport describes the picture."""
        from pybvh.bvhplot._matplotlib import frame_mpl, sequence_mpl
        style = Style("paper", projection=asked)
        samples = np.array([0, 5, 11], dtype=np.intp)
        for draw in (lambda ax: frame_mpl(scene, style, show=False, ax=ax),
                     lambda ax: sequence_mpl(
                         scene, style, samples, "overlay", ax=ax)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", **make_axes)
            draw(ax)
            assert _drawn_projection(ax) == asked
            plt.close(fig)

    def test_axes_reused_after_an_orthographic_figure_follow_the_style(
            self, scene):
        from pybvh.bvhplot._matplotlib import frame_mpl
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        frame_mpl(scene, Style("paper", projection="ortho"), show=False, ax=ax)
        assert _drawn_projection(ax) == "ortho"
        ax.clear()
        frame_mpl(scene, Style("paper"), show=False, ax=ax)
        assert _drawn_projection(ax) == "persp"
        plt.close(fig)

    @pytest.mark.parametrize("backend, library, expected", [
        ("_opencv", "cv2", "ortho"),
        ("_k3d", "k3d", "persp"),
        ("_vedo_offscreen", "vedo", "persp"),
        ("_vedo", "vedo", "persp"),
    ])
    def test_the_other_backends_say_what_they_draw_with(
            self, scene, monkeypatch, backend, library, expected):
        """OpenCV is orthographic, k3d and vedo perspective, whatever
        the style asks for."""
        pytest.importorskip(library)
        import importlib
        from pybvh.bvhplot import _viewport
        module = importlib.import_module(f"pybvh.bvhplot.{backend}")
        stated = []
        real = _viewport.make_viewport

        def record(views, **options):
            viewport = real(views, **options)
            stated.append(viewport.projection)
            return viewport

        monkeypatch.setattr(_viewport, "make_viewport", record)
        if hasattr(module, "make_viewport"):
            monkeypatch.setattr(module, "make_viewport", record)
        style = Style("paper", projection="ortho", supersample=1)
        if backend == "_opencv":
            next(module._generate_frames(scene, style, (160, 120)))
        elif backend == "_k3d":
            module._build_plot(scene, style)
        elif backend == "_vedo_offscreen":
            module._build_offscreen(scene, style, (120, 120))[0].close()
        else:
            monkeypatch.setattr(module, "_FORCE_OFFSCREEN", True)
            module._VedoPlayer(scene, style, 30.0, quality="high").plt.close()
        assert stated and set(stated) == {expected}


class TestEveryPerspectiveBackendUsesTheViewportsCamera:
    def test_vedo_offscreen_renders_from_it(self, scene, monkeypatch):
        """Checked on the camera VTK ends up with when a still is
        rendered, not on the numbers handed to it."""
        vedo = pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo_offscreen
        from pybvh.bvhplot._viewport import EYE_DISTANCE, make_viewport
        seen = {}
        real_screenshot = vedo.Plotter.screenshot

        def capture(plotter, *args, **kwargs):
            camera = plotter.camera
            seen.update(
                position=np.array(camera.GetPosition()),
                focal_point=np.array(camera.GetFocalPoint()),
                viewup=np.array(camera.GetViewUp()),
                distance=camera.GetDistance())
            return real_screenshot(plotter, *args, **kwargs)

        monkeypatch.setattr(vedo.Plotter, "screenshot", capture)
        _vedo_offscreen.frame_vedo(scene, Style("paper"), resolution=(200, 200))

        viewport = make_viewport(scene.views)
        eye, target, up = viewport.camera()
        np.testing.assert_allclose(seen["position"], eye)
        np.testing.assert_allclose(seen["focal_point"], target)
        np.testing.assert_allclose(seen["viewup"], up, atol=1e-12)
        assert seen["distance"] == pytest.approx(
            EYE_DISTANCE * viewport.half_span)

    # The vedo viewer's camera is tested in tests/test_vedo_player.py,
    # and k3d's in TestK3d above, each with the change that made it.


# ---------------------------------------------------------------------------
# Every backend sizes a body from that body
# ---------------------------------------------------------------------------
# What is drawn on a body (a capsule's radius) is a fraction of the
# body's size, never of the viewport's cube: the cube grows with the
# distance a clip travels, the body does not.

_BODY_SIZED_BACKENDS = ["vedo offscreen", "vedo viewer"]


def _travelling_view(body_lengths=5.0, n_frames=24):
    """The stick person (1.8 tall) walking *body_lengths* of its height."""
    from synthetic_scene import make_array_view
    return make_array_view(
        n_frames, walk_speed=body_lengths * 1.8 / (n_frames - 1))


def _still_of(view):
    import dataclasses
    return dataclasses.replace(
        view, coords=view.coords[:1], root_heading=view.root_heading[:1])


def _scaled(view, factor, lateral_shift):
    """*view* grown by *factor*, rest pose included, moved aside."""
    import dataclasses
    coords = view.coords * factor
    coords[..., 0] += lateral_shift
    return dataclasses.replace(
        view, coords=coords, rest_coords=view.rest_coords * factor,
        floor_height=float(coords[..., 1].min()))


def _body_sizes(backend, scene, monkeypatch):
    """What *backend* draws on each skeleton of *scene* with a size in
    the scene, one array per skeleton: the base capsule radius for the
    vedo backends."""
    return [np.array([capsule.base_radius])
            for capsule in _capsules(backend, scene, monkeypatch)]


def _capsules(backend, scene, monkeypatch):
    """The capsule skeletons a vedo *backend* builds for *scene*."""
    pytest.importorskip("vedo")
    if backend == "vedo offscreen":
        from pybvh.bvhplot._vedo_offscreen import _build_offscreen
        plotter, capsules, _ = _build_offscreen(
            scene, Style("paper"), (120, 120))
    else:
        assert backend == "vedo viewer"
        player = _viewer(scene, monkeypatch)
        plotter, capsules = player.plt, player._capsules
    plotter.close()
    return capsules


def _drawn_capsule_radii(backend, scene, monkeypatch):
    """The radius of every tube and sphere a vedo *backend* draws for
    the one skeleton of *scene*, read off the capsules' geometry: the
    tubes are built along z from their parent end, where they are
    widest, and the spheres at the origin."""
    [capsule] = _capsules(backend, scene, monkeypatch)
    tube_radii = np.linalg.norm(
        capsule.canonical_bone_verts[..., :2], axis=-1).max(axis=1)
    sphere_radii = np.linalg.norm(
        capsule.canonical_joint_verts, axis=-1).max(axis=1)
    return np.concatenate([tube_radii, sphere_radii])


def _one_node_walking(distance=10.0, n_frames=12):
    """A skeleton of one node carried *distance* along z."""
    from synthetic_scene import make_bare_view
    coords = np.zeros((n_frames, 1, 3))
    coords[:, 0, 2] = np.linspace(0.0, distance, n_frames)
    return make_bare_view(coords, np.zeros((1, 3)), [])


def _coincident_nodes(distance=10.0, n_frames=12):
    """Three nodes at one point, rest pose included, carried
    *distance* along z."""
    from synthetic_scene import make_bare_view
    coords = np.zeros((n_frames, 3, 3))
    coords[..., 2] = np.linspace(0.0, distance, n_frames)[:, np.newaxis]
    return make_bare_view(coords, np.zeros((3, 3)), [(0, 1), (1, 2)])


def _screenshot(plotter):
    plotter.render()
    return np.asarray(plotter.screenshot(asarray=True)).copy()


def _draws_the_body(backend, scene, monkeypatch):
    """Whether *backend* draws the one skeleton of *scene*: the
    offscreen still, with no floor to fill it, is not one flat color;
    the viewer's picture changes when the skeleton's capsules are taken
    out of it."""
    pytest.importorskip("vedo")
    if backend == "vedo offscreen":
        from pybvh.bvhplot._vedo_offscreen import frame_vedo
        image = frame_vedo(
            scene, Style("paper", floor=None), resolution=(200, 160))
        return len(np.unique(image.reshape(-1, 3), axis=0)) > 1
    assert backend == "vedo viewer"
    player = _viewer(scene, monkeypatch)
    try:
        with_body = _screenshot(player.plt)
        for capsule in player._capsules:
            player.plt.remove(*capsule.actors)
        return bool((_screenshot(player.plt) != with_body).any())
    finally:
        player.plt.close()


class TestEveryBackendSizesTheBodyFromTheBody:
    @pytest.mark.parametrize("backend", _BODY_SIZED_BACKENDS)
    def test_a_still_and_the_whole_clip_draw_the_same_body(
            self, backend, monkeypatch):
        from pybvh.bvhplot._scene import Scene
        clip = _travelling_view()
        [still] = _body_sizes(
            backend, Scene(views=[_still_of(clip)]), monkeypatch)
        [whole] = _body_sizes(backend, Scene(views=[clip]), monkeypatch)
        assert whole == pytest.approx(still)

    @pytest.mark.parametrize("backend", _BODY_SIZED_BACKENDS)
    def test_each_skeleton_is_sized_from_its_own_body(
            self, backend, monkeypatch):
        from pybvh.bvhplot._scene import Scene
        small = _travelling_view()
        big = _scaled(small, 3.0, lateral_shift=4.0)
        small_sizes, big_sizes = _body_sizes(
            backend, Scene(views=[small, big]), monkeypatch)
        assert big_sizes == pytest.approx(3.0 * small_sizes)

    @pytest.mark.parametrize("backend", ["vedo offscreen", "vedo viewer"])
    def test_capsules_follow_the_coords_unit_past_zero_length_bones(
            self, backend, monkeypatch):
        """Coordinates a caller hands in centimetres against a rest pose
        in metres draw every tube and sphere 100 times as wide, crowded
        ones included, although most of the bones have zero length (the
        median rest length is 0)."""
        from synthetic_scene import make_bare_view
        from pybvh.bvhplot._scene import Scene
        # Two parallel unit bones 0.01 apart (0-1 and 2-3), linked at
        # their base (0-2), with zero-length helpers on their tips.
        rest = np.array([
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.01, 0.0, 0.0], [0.01, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.01, 1.0, 0.0], [0.01, 1.0, 0.0]])
        bones = [(0, 1), (0, 2), (2, 3), (1, 4), (1, 5), (3, 6), (3, 7)]
        coords = np.repeat(rest[np.newaxis], 2, axis=0)
        in_metres = _drawn_capsule_radii(
            backend, Scene(views=[make_bare_view(coords, rest, bones)]),
            monkeypatch)
        in_centimetres = _drawn_capsule_radii(
            backend,
            Scene(views=[make_bare_view(100.0 * coords, rest, bones)]),
            monkeypatch)
        # The parallel bones are held to 60% of their 0.01 gap.
        assert in_metres[[0, 2]] == pytest.approx([0.006, 0.006], rel=1e-3)
        assert in_centimetres == pytest.approx(100.0 * in_metres, rel=1e-3)

    @pytest.mark.parametrize("make_view", [
        _one_node_walking, _coincident_nodes])
    @pytest.mark.parametrize("backend", _BODY_SIZED_BACKENDS)
    def test_a_body_with_no_rest_extent_is_still_drawn(
            self, backend, make_view, monkeypatch):
        """A rest pose with no extent has no height to size from; a
        clip that moves it is still drawn (see SkeletonView.body_size
        for the measure it falls back to)."""
        from pybvh.bvhplot._scene import Scene
        assert _draws_the_body(
            backend, Scene(views=[make_view()]), monkeypatch)


def _viewer_sizes(scene, monkeypatch, quality):
    """What the vedo viewer draws on each skeleton of *scene*, in
    *quality*: the fast mode's line width and point size (pixels, 0 in
    high quality), the joint labels' font size (pixels) and their lift
    above the joint (scene units)."""
    pytest.importorskip("vedo")
    from pybvh.bvhplot import _vedo
    monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
    player = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality=quality)
    try:
        sizes = []
        for s, view in enumerate(scene.views):
            label = player._label_actors[s][0]
            lift = np.asarray(label.GetPosition()) - view.coords[0, 0]
            if quality == "fast":
                line_width = player._lines_actors[s].properties.GetLineWidth()
                point_size = player._points_actors[s].properties.GetPointSize()
            else:
                line_width = point_size = 0.0
            sizes.append(dict(
                line_width=line_width, point_size=point_size,
                font_size=label.GetTextProperty().GetFontSize(),
                lift=float(np.linalg.norm(lift))))
        return sizes
    finally:
        player.plt.close()


class TestTheViewersOtherSizesIgnoreTheDistanceTravelled:
    @pytest.mark.parametrize("quality", ["fast", "high"])
    def test_a_still_and_the_whole_clip(self, quality, monkeypatch):
        from pybvh.bvhplot._scene import Scene
        clip = _travelling_view()
        [still] = _viewer_sizes(
            Scene(views=[_still_of(clip)]), monkeypatch, quality)
        [whole] = _viewer_sizes(Scene(views=[clip]), monkeypatch, quality)
        assert whole == pytest.approx(still)

    def test_pixel_sizes_do_not_follow_the_files_unit(self, monkeypatch):
        """Line width, point size and font size are pixels: the same
        body in centimetres draws them as in metres."""
        from pybvh.bvhplot._scene import Scene
        clip = _travelling_view()
        [metres] = _viewer_sizes(Scene(views=[clip]), monkeypatch, "fast")
        [centimetres] = _viewer_sizes(
            Scene(views=[_scaled(clip, 100.0, lateral_shift=0.0)]),
            monkeypatch, "fast")
        for pixels in ("line_width", "point_size", "font_size"):
            assert centimetres[pixels] == metres[pixels], pixels
        assert centimetres["lift"] == pytest.approx(100.0 * metres["lift"])

    def test_each_label_is_lifted_by_its_own_body(self, monkeypatch):
        from pybvh.bvhplot._scene import Scene
        small = _travelling_view()
        big = _scaled(small, 3.0, lateral_shift=4.0)
        small_sizes, big_sizes = _viewer_sizes(
            Scene(views=[small, big]), monkeypatch, "fast")
        assert big_sizes["lift"] == pytest.approx(3.0 * small_sizes["lift"])

    def test_fast_mode_draws_bones_as_wide_as_opencv_at_1080p(
            self, monkeypatch):
        """At the debug style's 2.5, OpenCV draws 3-pixel bones and
        joint discs 2 pixels wider in radius, 10 pixels across; the
        viewer's fast mode draws the same, not 2 pixels from rounding
        half to even."""
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo
        from pybvh.bvhplot._scene import Scene
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        player = _vedo._VedoPlayer(
            Scene(views=[_travelling_view()]), Style("debug"), 30.0,
            quality="fast")
        try:
            assert player._lines_actors[0].properties.GetLineWidth() == 3
            assert player._points_actors[0].properties.GetPointSize() == 10
        finally:
            player.plt.close()
