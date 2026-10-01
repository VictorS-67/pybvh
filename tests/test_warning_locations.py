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
from pybvh import bvhplot, read_bvh_directory, read_bvh_file
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
