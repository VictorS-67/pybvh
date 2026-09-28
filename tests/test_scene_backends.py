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
from pybvh.bvhplot._common import Style
from synthetic_scene import make_array_scene


def test_factory_knows_nothing_about_bvh():
    tree = ast.parse(pathlib.Path(synthetic_scene.__file__).read_text())
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"__future__", "numpy", "pybvh.bvhplot._common"}


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
                         follow=True, ghost=2, trajectory=True,
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
                            10.0, (320, 240), follow=True, ghost=1,
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
