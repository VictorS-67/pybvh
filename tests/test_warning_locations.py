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
from pybvh import read_bvh_directory, read_bvh_file

BVH_DIR = Path(__file__).parent.parent / "bvh_data"
TEST3 = BVH_DIR / "bvh_test3.bvh"   # rest pose and first frame disagree on up
DISAGREEMENT = "Rest pose suggests world up"


def _line_after_this_one() -> int:
    """The line number following the caller's line."""
    caller = inspect.currentframe().f_back  # type: ignore[union-attr]
    return caller.f_lineno + 1  # type: ignore[union-attr]


def _the_warning(caught, text):
    """The one warning in *caught* whose message contains *text*."""
    (found,) = [w for w in caught if text in str(w.message)]
    return found


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
