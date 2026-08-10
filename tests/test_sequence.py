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
