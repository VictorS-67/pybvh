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


class TestOpenCV:
    def test_render_with_every_option(self, scene, tmp_path):
        pytest.importorskip("cv2")
        from pybvh.bvhplot._opencv import render_opencv
        out = render_opencv(scene, Style("paper"), tmp_path / "walk.mp4",
                            10.0, (320, 240), motion="follow", ghost=1,
                            trajectory=True, frame_counter=True)
        assert out.exists() and out.stat().st_size > 0


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

    def test_one_skeleton_and_one_trail_per_view(self, pair):
        pytest.importorskip("k3d")
        from pybvh.bvhplot._k3d import _build_plot
        built = _build_plot(pair.spread("auto"), Style("paper"))
        assert len(built.skeletons) == 2
        assert len(built.trails) == 2
        for path, view in zip(built.trail_paths, pair.spread("auto").views):
            assert path.shape == (view.coords.shape[0], 3)
