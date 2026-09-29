"""Headless smoke tests for the vedo viewer shell (viewer cluster).

The playback *semantics* are covered in test_playback.py against the
pure PlaybackClock; these tests only prove the rendering/UI shell
constructs, delegates to the clock, and takes clean screenshots —
offscreen, no display needed.
"""
from __future__ import annotations

import numpy as np
import pytest

vedo = pytest.importorskip("vedo")

from pybvh import read_bvh_file
from pybvh.bvhplot._from_bvh import make_scene
from pybvh.bvhplot._style import Style
from pybvh.bvhplot import _vedo

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"


@pytest.fixture(scope="module")
def scene():
    bvh = read_bvh_file(BVH_PATH)
    coords = bvh.node_positions()[:40]
    return make_scene([bvh], [coords], "front", None)


@pytest.fixture
def player(scene, monkeypatch):
    monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
    p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
    yield p
    p.plt.close()


class TestCamera:
    """The viewer's camera is the viewport's, and stays so."""

    @staticmethod
    def _walk_toward_the_camera():
        import dataclasses
        from pybvh.bvhplot._scene import Scene
        from synthetic_scene import make_array_view
        view = make_array_view(n_frames=24)
        # twenty times the stride: the walk covers several body lengths
        far = view.coords.copy()
        far[..., 2] += 20 * (view.coords[:, :1, 2] - view.coords[:1, :1, 2])
        return Scene(views=[dataclasses.replace(view, coords=far)])

    def test_kept_through_show_and_reset(self, scene, monkeypatch):
        """VTK used to refit the distance and the target to everything
        in the scene, floor plane included."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend
        from pybvh.bvhplot._viewport import EYE_DISTANCE
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
            try:
                eye, target, up = p.viewport.camera()

                def check():
                    camera = p.plt.camera
                    np.testing.assert_allclose(camera.GetPosition(), eye)
                    np.testing.assert_allclose(camera.GetFocalPoint(), target)
                    np.testing.assert_allclose(
                        camera.GetViewUp(), up, atol=1e-12)
                    assert camera.GetDistance() == pytest.approx(
                        EYE_DISTANCE * p.viewport.half_span)

                check()
                p.show()
                check()
                p.plt.camera.SetPosition(*(eye * 3.0))
                p._on_reset_camera()
                check()
            finally:
                p.plt.close()

    @pytest.mark.parametrize("style", ["paper", "debug"])
    def test_the_skeleton_is_never_clipped_in_depth(self, monkeypatch, style):
        """VTK fits the clipping planes to what is in the scene at the
        moment. A skeleton that walks toward the camera must still be
        inside them on the last frame, and back on the first one after
        a reset there. Without a floor (debug) nothing else widens the
        planes, so that is the case that fails when they are not
        refitted."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend
        walk = self._walk_toward_the_camera()

        def depths_inside(p, frame):
            camera = p.plt.camera
            eye = np.array(camera.GetPosition())
            direction = np.array(camera.GetDirectionOfProjection())
            depth = (walk.views[0].coords[frame] - eye) @ direction
            near, far_plane = camera.GetClippingRange()
            return near <= depth.min() and depth.max() <= far_plane

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(walk, Style(style), 30.0, quality="high")
            try:
                p.show()
                assert depths_inside(p, 0)
                p._update_frame(23)
                assert depths_inside(p, 23)
                p._on_reset_camera()
                assert depths_inside(p, 23)
                p._update_frame(0)
                assert depths_inside(p, 0)
            finally:
                p.plt.close()


class TestPlayerShell:
    def test_constructs_with_chain_colors(self, player):
        # merged bone mesh exists and carries per-point colors
        capsule = player._capsules[0]
        assert capsule is not None and capsule.bones_mesh is not None
        assert capsule.bones_mesh.pointcolors is not None
        assert len(capsule.bones_mesh.pointcolors) > 0

    def test_delegates_to_clock(self, player):
        assert player.clock.playing
        player._toggle_play()
        assert not player.clock.playing
        player._jump_to(10)
        assert player.clock.frame == 10
        player._on_next()
        assert player.clock.frame == 11

    def test_clean_screenshot_hides_and_restores_ui(self, player, tmp_path):
        out = tmp_path / "shot.png"
        visible_before = [
            getattr(a, "actor", a).GetVisibility()
            for a in player._ui_actors]
        fname = player.screenshot(str(out), scale=1)
        assert fname == str(out)
        assert out.exists() and out.stat().st_size > 0
        visible_after = [
            getattr(a, "actor", a).GetVisibility()
            for a in player._ui_actors]
        assert visible_before == visible_after

    def test_fps_switch_resamples(self, player):
        idx15 = player.clock.fps_presets.index(15)
        player._set_fps(idx15)
        assert player.clock.target_fps == 15
        assert player.num_frames == len(player.coords_list[0])
        assert player.clock.frame == 0


class TestPlayerDarkStyle:
    def test_dark_background(self, scene, monkeypatch, tmp_path):
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(scene, Style("dark"), 30.0, quality="high")
        try:
            out = tmp_path / "dark.png"
            p.screenshot(str(out), scale=1)
            assert out.exists()
        finally:
            p.plt.close()


class TestGridFloor:
    """The grid floor ("grid", and "checker" which falls back to it) is
    built flat in vedo's XY plane and turned to face up. vedo turns
    about the world origin, so the grid must be turned first and moved
    after: placed first, it swung away from under the skeleton on any
    clip not centred on the origin."""

    @staticmethod
    def _scene_with_up(up):
        import dataclasses
        from pybvh.bvhplot._scene import Scene
        from synthetic_scene import make_array_view
        view = make_array_view(n_frames=12)
        if up == "+y":
            return Scene(views=[view])
        order = {"+z": [0, 2, 1], "+x": [1, 0, 2]}[up]
        forward = {"+z": "+y", "+x": "+z"}[up]
        heights = view.coords[..., order][..., "xyz".index(up[1])]
        return Scene(views=[dataclasses.replace(
            view, coords=view.coords[..., order],
            rest_coords=view.rest_coords[..., order], up=up,
            forward_axis=forward, root_heading=None,
            floor_height=float(heights.min()))])

    @pytest.mark.parametrize("up", ["+y", "+z", "+x"])
    @pytest.mark.parametrize("kind", ["grid", "checker"])
    def test_the_grid_lies_where_the_plane_goes(self, up, kind, monkeypatch):
        from pybvh.bvhplot._vedo_capsules import FLOOR_EPSILON
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(self._scene_with_up(up),
                              Style("paper", floor=kind), 30.0,
                              quality="high")
        try:
            grids = [o for o in p.plt.objects if type(o).__name__ == "Grid"]
            assert len(grids) == 1
            vertices = np.asarray(grids[0].vertices)
            viewport = p.viewport
            # the scene is not centred on the origin, or this proves nothing
            assert np.abs(viewport.center).max() > 0.2
            expected = viewport.floor_quad()
            for axis in viewport.ground_axes:
                assert vertices[:, axis].min() == pytest.approx(
                    expected[:, axis].min(), rel=1e-5)
                assert vertices[:, axis].max() == pytest.approx(
                    expected[:, axis].max(), rel=1e-5)
            np.testing.assert_allclose(
                vertices[:, viewport.up_index],
                viewport.below_floor(FLOOR_EPSILON * viewport.half_span),
                rtol=1e-5, atol=1e-6)
        finally:
            p.plt.close()
