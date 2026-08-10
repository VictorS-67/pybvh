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
from pybvh.bvhplot._common import Style, make_scene
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


class TestPlayerShell:
    def test_constructs_with_chain_colors(self, player):
        # merged bone mesh exists and carries per-point colors
        assert player._bones_mesh[0] is not None
        assert player._bones_mesh[0].pointcolors is not None
        assert len(player._bones_mesh[0].pointcolors) > 0

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
