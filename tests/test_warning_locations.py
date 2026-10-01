"""Every pybvh warning names the line of the user's code that led to it.

Python attributes a warning to the frame ``stacklevel`` levels above the
``warnings.warn`` call. A warning raised deep inside pybvh is reached
from several public calls through different depths, so each test here
reaches it through one public path and checks that the warning is
attributed to this file, the caller's, rather than to a pybvh module.
"""
from __future__ import annotations

import inspect
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import pybvh
from pybvh import analysis, bvhplot, read_bvh_directory, read_bvh_file
from pybvh.bvhplot import _matplotlib

BVH_DIR = Path(__file__).parent.parent / "bvh_data"
TEST3 = BVH_DIR / "bvh_test3.bvh"   # rest pose and first frame disagree on up
DISAGREEMENT = "Rest pose suggests world up"
NO_UP = "Could not infer a world up axis"
NO_FACING = "No usable left/right geometry"

# Every node at the origin and no head, neck, chest or spine joint:
# neither the first frame nor the rest pose says which way is up.
NO_UP_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Tail
  {
    OFFSET 0 0 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 0 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
"""

# Upright, but with no left/right joint pairs to measure a facing from.
NO_FACING_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    JOINT Head
    {
      OFFSET 0 10 0
      CHANNELS 3 Zrotation Yrotation Xrotation
      End Site
      {
        OFFSET 0 5 0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 90 0 0 0 0 0 0 0 0 0 0
0 90 0 0 0 0 0 0 0 0 0 0
"""


def _line_after_this_one() -> int:
    """The line number following the caller's line."""
    caller = inspect.currentframe().f_back  # type: ignore[union-attr]
    return caller.f_lineno + 1  # type: ignore[union-attr]


def _the_warning(caught, text):
    """The one warning in *caught* whose message contains *text*."""
    (found,) = [w for w in caught if text in str(w.message)]
    return found


def _files_named(caught, text):
    """The files the warnings in *caught* whose message contains *text*
    are attributed to: every one of them, for a call that warns more
    than once."""
    return {w.filename for w in caught if text in str(w.message)}


def test_these_tests_import_the_pybvh_they_test():
    """The editable install may point at another checkout: the
    locations below are only meaningful against this tree's package."""
    tree = Path(__file__).resolve().parent.parent
    assert Path(pybvh.__file__).resolve().parent == tree / "pybvh"


class TestWorldUpDisagreement:
    """The rest pose and the first frame of ``bvh_test3`` disagree on
    which way is up, which every inference of ``world_up`` reports."""

    def test_reading_the_file_names_the_read_call(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            read_bvh_file(TEST3)
        warning = _the_warning(caught, DISAGREEMENT)
        assert (warning.filename, warning.lineno) == (__file__, line)

    def test_a_slice_names_the_line_that_asks_for_its_up_axis(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = read_bvh_file(TEST3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            clip[0:10].world_up
        warning = _the_warning(caught, DISAGREEMENT)
        assert (warning.filename, warning.lineno) == (__file__, line)

    def test_world_up_inferred_names_its_caller(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = read_bvh_file(TEST3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clip.world_up_inferred
        assert _the_warning(caught, DISAGREEMENT).filename == __file__

    def test_an_edited_clip_names_the_line_that_reads_its_up_axis(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = read_bvh_file(TEST3)
        clip.translate_root(np.array([1.0, 0.0, 0.0]), inplace=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clip.world_up
        assert _the_warning(caught, DISAGREEMENT).filename == __file__

    @pytest.mark.parametrize("parallel", [False, True])
    def test_reading_a_directory_names_the_read_call(self, tmp_path, parallel):
        """In parallel the files are read on worker threads, whose
        stacks hold no line of the caller's: the warning must still be
        raised where the caller can be named."""
        shutil.copy(TEST3, tmp_path / "test3.bvh")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_bvh_directory(tmp_path, parallel=parallel)
        assert _the_warning(caught, DISAGREEMENT).filename == __file__


class TestReadingADirectory:

    @pytest.mark.parametrize("parallel", [False, True])
    def test_a_skipped_file_names_the_read_call(self, tmp_path, parallel):
        shutil.copy(BVH_DIR / "bvh_test1.bvh", tmp_path / "good.bvh")
        (tmp_path / "broken.bvh").write_text("not a BVH file")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            clips = read_bvh_directory(tmp_path, parallel=parallel,
                                       skip_errors=True)
        warning = _the_warning(caught, "skipping")
        assert (warning.filename, warning.lineno) == (__file__, line)
        assert [Path(c.source_path).name for c in clips] == ["good.bvh"]

    @pytest.mark.parametrize("parallel", [False, True])
    def test_without_skip_errors_the_first_failure_propagates(
            self, tmp_path, parallel):
        (tmp_path / "a_broken.bvh").write_text("not a BVH file")
        shutil.copy(BVH_DIR / "bvh_test1.bvh", tmp_path / "b_good.bvh")
        with pytest.raises(ValueError, match="a_broken"):
            read_bvh_directory(tmp_path, parallel=parallel)


class TestWorldUpFallback:
    """A skeleton that shows no up axis falls back to '+y' and warns
    on each inference."""

    @pytest.fixture
    def path(self, tmp_path):
        path = tmp_path / "no_up.bvh"
        path.write_text(NO_UP_BVH)
        return path

    def test_reading_the_file_names_the_read_call(self, path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            read_bvh_file(path)
        warning = _the_warning(caught, NO_UP)
        assert (warning.filename, warning.lineno) == (__file__, line)

    def test_an_edited_clip_names_the_line_that_reads_its_up_axis(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = read_bvh_file(path)
        clip.root_pos = clip.root_pos + 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clip.world_up
        assert _the_warning(caught, NO_UP).filename == __file__

    def test_world_up_inferred_names_its_caller(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = read_bvh_file(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clip.world_up_inferred
        assert _the_warning(caught, NO_UP).filename == __file__


class TestFacingFallback:
    """A skeleton with no left/right pairs falls back to a fixed forward
    axis, reached through every facing query."""

    @pytest.fixture
    def clip(self, tmp_path):
        path = tmp_path / "no_facing.bvh"
        path.write_text(NO_FACING_BVH)
        return read_bvh_file(path)

    @pytest.mark.parametrize("query", [
        lambda clip: clip.rest_forward,
        lambda clip: clip.forward_axis,
        lambda clip: clip.forward_at(0),
        lambda clip: clip.left_at(0),
        lambda clip: clip.facing_frame(),
    ], ids=["rest_forward", "forward_axis", "forward_at", "left_at",
            "facing_frame"])
    def test_each_facing_query_names_its_caller(self, clip, query):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            query(clip)
        assert _the_warning(caught, NO_FACING).filename == __file__

    @pytest.mark.parametrize("draw", [
        lambda clip, path: bvhplot.frame(clip),
        lambda clip, path: clip.plot_frame(),
        lambda clip, path: bvhplot.render(clip, path, backend="matplotlib"),
        lambda clip, path: clip.render(path, backend="matplotlib"),
        lambda clip, path: bvhplot.play(clip, backend="matplotlib"),
        lambda clip, path: clip.play(backend="matplotlib"),
    ], ids=["bvhplot.frame", "Bvh.plot_frame", "bvhplot.render",
            "Bvh.render", "bvhplot.play", "Bvh.play"])
    def test_drawing_names_the_draw_call(
            self, clip, draw, monkeypatch, tmp_path):
        """The "front" camera stands in front of the skeleton's
        forward axis, the fallback one here."""
        for backend in ("frame_mpl", "render_mpl", "play_mpl"):
            monkeypatch.setattr(_matplotlib, backend,
                                lambda *args, **kwargs: (None, None))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draw(clip, tmp_path / "clip.gif")
        assert _files_named(caught, NO_FACING) == {__file__}


# A foot joint whose only child is another joint, neither an end site nor
# a toe: the foot detector finds no candidate with a tip.
TIPLESS_FOOT_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Head
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 5 0
    }
  }
  JOINT LeftFoot
  {
    OFFSET 2 -10 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    JOINT LeftHeel
    {
      OFFSET 0 -1 -1
      CHANNELS 3 Zrotation Yrotation Xrotation
      End Site
      {
        OFFSET 0 0 -1
      }
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.033333
0 11 0 0 0 0 0 0 0 0 0 0 0 0 0
0 11 1 0 0 0 0 0 0 0 0 0 0 0 0
0 11 2 0 0 0 0 0 0 0 0 0 0 0 0
"""
TIPLESS = "no candidates have tip descendants"


class TestFootDetectionFallback:

    @pytest.fixture
    def clip(self, tmp_path):
        path = tmp_path / "tipless_foot.bvh"
        path.write_text(TIPLESS_FOOT_BVH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return read_bvh_file(path)

    @pytest.mark.parametrize("call", [
        lambda clip: analysis.auto_detect_foot_joints(clip),
        lambda clip: clip.auto_detect_foot_joints(),
        lambda clip: analysis.foot_contacts(clip),
        lambda clip: clip.foot_contacts(),
        lambda clip: analysis.skeleton_size(clip),
    ], ids=["analysis.auto_detect_foot_joints", "Bvh.auto_detect_foot_joints",
            "analysis.foot_contacts", "Bvh.foot_contacts",
            "analysis.skeleton_size"])
    def test_each_detection_names_its_caller(self, clip, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(clip)
        assert _files_named(caught, TIPLESS) == {__file__}

    def test_called_directly_names_the_line_of_the_call(self, clip):
        """A fixed level right for foot_contacts overshoots a direct
        call, past the user's line to the one that called it."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            analysis.auto_detect_foot_joints(clip)
        warning = _the_warning(caught, TIPLESS)
        assert (warning.filename, warning.lineno) == (__file__, line)


class TestConcatenatingClipsOfTwoRates:

    @pytest.fixture
    def clips(self):
        first = read_bvh_file(BVH_DIR / "bvh_test1.bvh")
        second = first.copy()
        second.frame_time = first.frame_time * 2
        return first, second

    def test_adding_names_the_line_of_the_sum(self, clips):
        first, second = clips
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            first + second
        warning = _the_warning(caught, "Frame time mismatch")
        assert (warning.filename, warning.lineno) == (__file__, line)

    def test_adding_in_place_names_the_line_of_the_sum(self, clips):
        first, second = clips
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = _line_after_this_one()
            first += second
        warning = _the_warning(caught, "Frame time mismatch")
        assert (warning.filename, warning.lineno) == (__file__, line)


def test_harmonize_names_the_line_of_its_call():
    from pybvh.batch import harmonize
    reference = read_bvh_file(BVH_DIR / "bvh_example.bvh")
    other_skeleton = read_bvh_file(BVH_DIR / "bvh_test2.bvh")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = _line_after_this_one()
        harmonize([reference, other_skeleton], reference=reference,
                  verbose=True)
    warning = _the_warning(caught, "harmonize:")
    assert (warning.filename, warning.lineno) == (__file__, line)


class TestBvhplot:
    """bvhplot's warnings, reached through its functions and through
    the Bvh methods that wrap them. The backends that would draw are
    stubbed out: only the warnings before them are under test."""

    @pytest.fixture
    def walk(self):
        return read_bvh_file(BVH_DIR / "bvh_test1.bvh")[0:5]

    @pytest.fixture
    def stubbed(self, monkeypatch):
        monkeypatch.setattr(_matplotlib, "render_mpl",
                            lambda *args, **kwargs: None)
        monkeypatch.setattr(_matplotlib, "play_mpl",
                            lambda *args, **kwargs: None)

    @pytest.fixture
    def without(self, monkeypatch):
        """Make the packages named importable no more."""
        def remove(*names):
            monkeypatch.setattr(
                bvhplot, "_module_importable",
                lambda name: name not in names)
        return remove

    @pytest.mark.parametrize("draw", [
        lambda clips, path: bvhplot.render(clips, path, backend="matplotlib"),
        lambda clips, path: bvhplot.play(clips, backend="matplotlib"),
    ], ids=["render", "play"])
    def test_clips_of_two_rates_name_the_draw_call(
            self, walk, stubbed, tmp_path, draw):
        faster = walk.resample(60)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draw([walk, faster], tmp_path / "clip.gif")
        assert _files_named(caught, "Frame rates differ") == {__file__}

    def test_a_clip_without_a_rate_names_the_play_call(self, walk, stubbed):
        unset = walk.copy()
        unset.frame_time = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bvhplot.play([walk, unset], backend="matplotlib", fps=10)
        assert _files_named(caught, "frame_time 0 (unset)") == {__file__}

    def test_clips_of_two_up_axes_name_the_play_call(self, walk, stubbed):
        turned = walk.copy()
        turned.world_up = "+y" if walk.world_up != "+y" else "+z"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bvhplot.play([walk, turned], backend="matplotlib")
        assert _files_named(caught, "different world_up") == {__file__}

    @pytest.mark.parametrize("draw", [
        lambda clip, path: bvhplot.render(clip, path),
        lambda clip, path: clip.render(path),
    ], ids=["bvhplot.render", "Bvh.render"])
    def test_a_render_without_opencv_names_the_render_call(
            self, walk, stubbed, without, tmp_path, draw):
        without("cv2")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draw(walk, tmp_path / "clip.mp4")
        assert _files_named(caught, "OpenCV not found") == {__file__}

    @pytest.mark.parametrize("play", [
        lambda clip: bvhplot.play(clip),
        lambda clip: clip.play(),
    ], ids=["bvhplot.play", "Bvh.play"])
    def test_a_slow_fallback_names_the_play_call(
            self, walk, stubbed, without, monkeypatch, play):
        without("vedo")
        monkeypatch.setattr(bvhplot, "_resolve_play_backend",
                            lambda requested: ("matplotlib", 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            play(walk)
        assert _files_named(caught, "No interactive backend") == {__file__}
        assert _files_named(caught, "OpenCV not found") == {__file__}

    @pytest.mark.parametrize("draw", [
        lambda clip, path: bvhplot.render(clip, path, backend="matplotlib"),
        lambda clip, path: clip.render(path, backend="matplotlib"),
    ], ids=["bvhplot.render", "Bvh.render"])
    def test_a_video_without_ffmpeg_names_the_render_call(
            self, walk, monkeypatch, tmp_path, draw):
        from matplotlib import animation
        monkeypatch.setattr(animation.writers, "is_available",
                            lambda name: False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draw(walk, tmp_path / "clip.mp4")
        assert _files_named(caught, "FFmpeg not found") == {__file__}
        assert (tmp_path / "clip.gif").exists()
