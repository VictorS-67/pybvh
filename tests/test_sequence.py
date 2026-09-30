"""Tests for sequence(), ghost=, and trajectory traces (Phase 2)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import read_bvh_file, bvhplot
from pybvh.bvhplot import _resolve_sample_frames

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"


@pytest.fixture(scope="module")
def bvh():
    return read_bvh_file(BVH_PATH)


class TestResolveSampleFrames:
    def test_full_clip(self):
        idx = _resolve_sample_frames(100, 5, None)
        assert idx[0] == 0 and idx[-1] == 99
        assert len(idx) == 5

    def test_tuple_range(self):
        idx = _resolve_sample_frames(100, 4, (10, 50))
        assert idx[0] == 10 and idx[-1] == 49

    def test_negative_tuple(self):
        idx = _resolve_sample_frames(100, 3, (-20, -1))
        assert idx[0] == 80 and idx[-1] == 98

    def test_slice(self):
        idx = _resolve_sample_frames(100, 3, slice(10, 20))
        assert idx[0] == 10 and idx[-1] == 19

    def test_stepped_slice_raises(self):
        with pytest.raises(ValueError, match="step"):
            _resolve_sample_frames(100, 3, slice(0, 50, 2))

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _resolve_sample_frames(100, 3, (50, 50))

    def test_bad_n_poses_raises(self):
        with pytest.raises(ValueError, match="n_poses"):
            _resolve_sample_frames(100, 0, None)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="frames"):
            _resolve_sample_frames(100, 3, [1, 2])  # type: ignore[arg-type]

    def test_dedup_when_range_smaller_than_n(self):
        idx = _resolve_sample_frames(3, 8, None)
        assert list(idx) == [0, 1, 2]


class TestSequence:
    def test_offset_layout(self, bvh):
        fig, ax = bvhplot.sequence(bvh)
        assert fig is not None
        plt.close(fig)

    def test_overlay_layout(self, bvh):
        fig, ax = bvhplot.sequence(bvh, layout="overlay", n_poses=4)
        plt.close(fig)

    def test_frames_restriction(self, bvh):
        fig, ax = bvhplot.sequence(bvh, frames=(100, 300), n_poses=4)
        plt.close(fig)

    def test_dark_style(self, bvh):
        fig, ax = bvhplot.sequence(bvh, style="dark", n_poses=3)
        plt.close(fig)

    def test_list_raises(self, bvh):
        with pytest.raises(ValueError, match="single"):
            bvhplot.sequence([bvh, bvh])  # type: ignore[arg-type]

    def test_bad_layout_raises(self, bvh):
        with pytest.raises(ValueError, match="layout"):
            bvhplot.sequence(bvh, layout="diagonal")

    def test_bvh_wrapper(self, bvh):
        fig, ax = bvh.plot_sequence(n_poses=3)
        plt.close(fig)

    def test_offset_uses_noncubic_box(self, bvh):
        """The walk travels far: the box aspect must follow the data."""
        fig, ax = bvhplot.sequence(bvh)
        # matplotlib normalizes _box_aspect internally, so only the
        # RATIO between the axes is meaningful.
        aspect = np.asarray(ax._box_aspect)  # type: ignore[attr-defined]
        assert max(aspect) / min(aspect) > 1.5  # genuinely non-cubic
        plt.close(fig)


class TestGhostAndTrajectory:
    def test_render_mpl_ghost_trajectory(self, bvh, tmp_path):
        short = bvh[0:120]
        path = bvhplot.render(
            short, tmp_path / "g.gif", backend="matplotlib",
            resolution=(320, 240), ghost=2, trajectory=True)
        assert path.exists() and path.stat().st_size > 0

    def test_render_opencv_ghost_trajectory(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        short = bvh[0:120]
        path = bvhplot.render(
            short, tmp_path / "g.mp4", backend="opencv",
            resolution=(320, 240), ghost=2, trajectory=True)
        assert path.exists() and path.stat().st_size > 0

    def test_bad_ghost_raises(self, bvh, tmp_path):
        with pytest.raises(ValueError, match="ghost"):
            bvhplot.render(bvh, tmp_path / "x.mp4", ghost=-1)


def _gif_frames(path):
    from PIL import Image, ImageSequence
    with Image.open(path) as gif:
        return [np.asarray(frame.convert("RGB"))
                for frame in ImageSequence.Iterator(gif)]


def _still(bvh, num_frames):
    """The clip's first pose held for *num_frames* frames: two frames of
    a turntable over it look alike exactly when the camera does."""
    still = bvh[0:num_frames].copy()
    still.root_pos = np.repeat(bvh.root_pos[:1], num_frames, axis=0)
    still.joint_angles = np.repeat(bvh.joint_angles[:1], num_frames, axis=0)
    return still


class TestTurntablePeriod:
    """``turntable_period`` is seconds of the video per revolution,
    at the rate the render plays (``fps``, 10 here)."""

    def _render(self, clip, tmp_path, **options):
        pytest.importorskip("cv2")
        path = bvhplot.render(
            clip, tmp_path / "tt.gif", backend="opencv", fps=10,
            camera="turntable", resolution=(160, 120), **options)
        return _gif_frames(path)

    def test_a_period_shorter_than_the_clip_makes_several_orbits(
            self, bvh, tmp_path):
        # 0.4 s at 10 fps: a revolution every 4 frames, two over 8.
        frames = self._render(_still(bvh, 8), tmp_path, turntable_period=0.4)
        assert len(frames) == 8
        for f in range(4):
            np.testing.assert_array_equal(frames[f], frames[f + 4])
        assert not np.array_equal(frames[0], frames[2])

    def test_a_period_longer_than_the_clip_loops_it_until_the_orbit_ends(
            self, bvh, tmp_path):
        # 1.2 s at 10 fps: 12 frames, the 4-frame clip three times over,
        # while the camera keeps turning across the seams.
        frames = self._render(_still(bvh, 4), tmp_path, turntable_period=1.2)
        assert len(frames) == 12
        assert not np.array_equal(frames[0], frames[4])

    def test_the_loop_ends_on_the_frame_nearest_the_period(
            self, bvh, tmp_path):
        frames = self._render(bvh[0:4], tmp_path, turntable_period=1.26)
        assert len(frames) == 13

    def test_a_period_of_the_clips_length_is_the_default_orbit(
            self, bvh, tmp_path):
        clip = bvh[0:6]
        default = self._render(clip, tmp_path)
        timed = self._render(clip, tmp_path, turntable_period=0.6)
        assert len(timed) == len(default) == 6
        for mine, theirs in zip(timed, default):
            np.testing.assert_array_equal(mine, theirs)

    def test_a_clip_without_a_rate_is_timed_by_fps(self, bvh, tmp_path):
        clip = bvh[0:4].copy()
        clip.frame_time = 0
        frames = self._render(clip, tmp_path, turntable_period=1.2)
        assert len(frames) == 12

    def test_the_matplotlib_backend_loops_too(self, bvh, tmp_path):
        path = bvhplot.render(
            bvh[0:4], tmp_path / "tt.gif", backend="matplotlib", fps=10,
            camera="turntable", turntable_period=0.8, resolution=(160, 120))
        assert len(_gif_frames(path)) == 8

    @pytest.mark.parametrize("period", [0, -1.0, float("nan"), float("inf")])
    def test_the_period_is_a_positive_number_of_seconds(
            self, bvh, tmp_path, period):
        with pytest.raises(ValueError, match="turntable_period"):
            bvhplot.render(bvh[0:4], tmp_path / "x.gif", camera="turntable",
                           turntable_period=period)

    def test_the_period_needs_the_turntable_camera(self, bvh, tmp_path):
        with pytest.raises(ValueError, match="turntable_period"):
            bvhplot.render(bvh[0:4], tmp_path / "x.gif", camera="side",
                           turntable_period=2.0)

    def test_the_vedo_backend_has_no_turntable(self, bvh, tmp_path):
        pytest.importorskip("vedo")
        with pytest.raises(ValueError, match="turntable"):
            bvhplot.render(bvh[0:4], tmp_path / "x.gif", backend="vedo",
                           camera="turntable", turntable_period=2.0)


class TestPhase3Export:
    def test_turntable_opencv(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        path = bvhplot.render(
            bvh[0:30], tmp_path / "tt.mp4", backend="opencv",
            camera="turntable", resolution=(320, 240))
        assert path.exists() and path.stat().st_size > 0

    def test_turntable_mpl(self, bvh, tmp_path):
        path = bvhplot.render(
            bvh[0:6], tmp_path / "tt.gif", backend="matplotlib",
            camera="turntable", resolution=(320, 240))
        assert path.exists() and path.stat().st_size > 0

    def test_frame_counter_opt_in(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        path = bvhplot.render(
            bvh[0:10], tmp_path / "fc.mp4", backend="opencv",
            resolution=(320, 240), frame_counter=True)
        assert path.exists()

    def test_supersample_one_disables(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        from pybvh.bvhplot import Style
        path = bvhplot.render(
            bvh[0:10], tmp_path / "ss1.mp4", backend="opencv",
            resolution=(320, 240), style=Style("paper", supersample=1))
        assert path.exists()

    def test_play_opencv_backend_nameable_but_notebook_only(self, bvh):
        pytest.importorskip("cv2")
        # Everything auto can choose must be nameable; outside a
        # notebook the error must say why it can't run, not "unknown".
        with pytest.raises(ValueError, match="notebook"):
            bvhplot.play(bvh, backend="opencv")

    def test_play_unknown_backend_lists_opencv(self, bvh):
        with pytest.raises(ValueError, match="opencv"):
            bvhplot.play(bvh, backend="opencvv")


class TestReviewFixes:
    """Regression tests for the v0.9.0 code-review findings."""

    def test_sequence_trace_respects_frames_range(self, bvh):
        fig, ax = bvhplot.sequence(bvh, frames=(100, 300), n_poses=4)
        # the dashed trace is the only Line3D on the axes; the range is
        # stop-exclusive, so the trace spans frames 100..299
        (trace,) = ax.lines
        assert len(trace.get_xdata()) == 200
        plt.close(fig)

    def test_sequence_honors_floor_kind(self, bvh):
        from mpl_toolkits.mplot3d.art3d import (
            Line3DCollection, Poly3DCollection)
        from pybvh.bvhplot import Style
        fig, ax = bvhplot.sequence(
            bvh, n_poses=3, style=Style("paper", floor="grid"))
        # a grid floor is a Line3DCollection at floor zorder, not a quad
        floor_artists = [c for c in ax.collections
                        if c.get_zorder() == 0.5]
        assert floor_artists
        assert all(isinstance(c, Line3DCollection) for c in floor_artists)
        plt.close(fig)

    def test_mpl_ghosts_render_under_live_skeleton(self, bvh, tmp_path):
        """Ghost collections carry zorder 1.5, below the live bones (2)."""
        from pybvh.bvhplot._from_bvh import make_scene
        from pybvh.bvhplot import _matplotlib as m
        import matplotlib.pyplot as mplt
        coords = bvh.node_positions()[:50]
        scene = make_scene([bvh], [coords], "front", None)
        fig = mplt.figure()
        ax = fig.add_subplot(111, projection="3d")
        from pybvh.bvhplot._viewport import panel_viewports
        viewports = panel_viewports(scene.views, framing="clip")
        ghost_slots, traces = m._setup_render_extras(
            scene, viewports, bvhplot.Style("paper"), [ax], 2, True)
        for collection, _lag in ghost_slots[0]:
            assert collection.get_zorder() == 1.5
        assert traces[0].get_zorder() == 0.8
        mplt.close(fig)

    def test_opencv_panels_do_not_overdraw(self, bvh):
        """A panel's floor must not bleed into its neighbor: the left
        panel of a 2-up render equals the same view rendered alone."""
        cv2 = pytest.importorskip("cv2")
        from pybvh.bvhplot._from_bvh import make_scene
        from pybvh.bvhplot._opencv import _generate_frames
        from pybvh.bvhplot import Style
        # force chains so the multi-skeleton auto-switch can't recolor
        # the left panel relative to the solo render
        style = Style("paper", supersample=1, color_mode="chains")
        coords = bvh.node_positions()[:2]
        far = coords + np.array([500.0, 0.0, 0.0])
        pair = make_scene([bvh, bvh], [coords, far], "front", None)
        solo = make_scene([bvh], [coords], "front", None)
        pair_img = next(_generate_frames(pair, style, (400, 200)))
        solo_img = next(_generate_frames(solo, style, (200, 200)))
        # ignore the 1px divider column at x=200
        assert np.array_equal(pair_img[:, :199], solo_img[:, :199])
