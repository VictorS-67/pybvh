"""The backend-choice module: which backend runs, from the request and a
test-built environment, so that every tier and every missing package is
reached on any machine. The k3d widget and the vedo window cannot run in
the suite; the paths of play() that open no window are also tested through
play() itself, in test_plot.py."""

from __future__ import annotations

import sys

import pytest

from pybvh.bvhplot import _backends
from pybvh.bvhplot._backends import Environment

NO_INTERACTIVE = (
    "No interactive backend (k3d, vedo) found. Install with: pip install pybvh[interactive]"
)
SLOW_PLAY = (
    "OpenCV not found for fast rendering. Install with: pip install pybvh[opencv]. "
    "Falling back to matplotlib (slow for long clips)."
)


def machine(*, notebook=False, display=False, importable=()):
    return Environment(in_notebook=notebook, has_display=display, importable=frozenset(importable))


class TestPlayTiers:
    """play()'s auto choice: the best backend the session and the installed
    packages allow, and a warning for each fallback naming what to install."""

    @pytest.mark.parametrize(
        "environment, backend, warned",
        [
            pytest.param(
                machine(notebook=True, importable={"k3d", "cv2", "vedo"}),
                "k3d",
                (),
                id="notebook-k3d",
            ),
            pytest.param(
                machine(notebook=True, importable={"cv2"}),
                "opencv_notebook",
                (NO_INTERACTIVE,),
                id="notebook-opencv",
            ),
            pytest.param(
                machine(notebook=True, importable={"cv2", "vedo"}),
                "opencv_notebook",
                (),
                id="notebook-opencv-vedo-installed",
            ),
            pytest.param(
                machine(notebook=True),
                "matplotlib",
                (NO_INTERACTIVE, SLOW_PLAY),
                id="notebook-nothing",
            ),
            pytest.param(
                machine(notebook=True, importable={"vedo"}),
                "matplotlib",
                (SLOW_PLAY,),
                id="notebook-vedo-only",
            ),
            pytest.param(
                machine(display=True, importable={"vedo", "k3d", "cv2"}),
                "vedo",
                (),
                id="window-vedo",
            ),
            pytest.param(
                machine(display=True, importable={"k3d", "cv2"}),
                "matplotlib",
                (NO_INTERACTIVE,),
                id="window-no-vedo",
            ),
            pytest.param(
                machine(importable={"vedo"}), "matplotlib", (), id="headless-vedo-installed"
            ),
            pytest.param(machine(), "matplotlib", (NO_INTERACTIVE,), id="headless-nothing"),
        ],
    )
    def test_auto_picks_the_best_backend_the_session_allows(self, environment, backend, warned):
        choice = _backends.choose_play_backend("auto", environment)
        assert choice.backend == backend
        assert choice.warnings == warned

    @pytest.mark.parametrize("named", ["k3d", "vedo", "matplotlib"])
    def test_a_named_backend_runs_without_a_warning(self, named):
        choice = _backends.choose_play_backend(named, machine(importable={"k3d", "vedo", "cv2"}))
        assert (choice.backend, choice.warnings) == (named, ())

    def test_opencv_names_the_notebook_video(self):
        choice = _backends.choose_play_backend("opencv", machine(notebook=True, importable={"cv2"}))
        assert (choice.backend, choice.warnings) == ("opencv_notebook", ())

    def test_opencv_outside_a_notebook_raises(self):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend("opencv", machine(display=True, importable={"cv2"}))
        assert str(raised.value) == (
            "backend='opencv' plays an inline video and only works inside a Jupyter "
            "notebook. In a script, use backend='vedo' (interactive window) or render() "
            "to a file instead."
        )

    def test_opencv_without_opencv_raises_naming_the_install(self):
        with pytest.raises(ImportError) as raised:
            _backends.choose_play_backend("opencv", machine(notebook=True))
        assert str(raised.value) == (
            "The opencv play backend requires opencv-python. "
            "Install with: pip install pybvh[opencv]"
        )

    def test_an_unknown_backend_raises(self):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend("opencvv", machine())
        assert str(raised.value) == (
            "Unknown backend 'opencvv'. Choose from: "
            "['auto', 'k3d', 'matplotlib', 'opencv', 'vedo']"
        )


class TestRenderChoice:
    """render()'s choice: by the file's extension and the installed
    packages under auto; a named backend wins."""

    @pytest.mark.parametrize("suffix", [".html", ".webp", ".apng", ".gif"])
    @pytest.mark.parametrize("importable", [{"cv2"}, set()], ids=["opencv", "no-opencv"])
    def test_auto_sends_what_opencv_cannot_write_to_matplotlib(self, suffix, importable):
        """Without the missing-OpenCV warning: OpenCV would not have
        written the file either."""
        choice = _backends.choose_render_backend("auto", suffix, machine(importable=importable))
        assert (choice.backend, choice.warnings) == ("matplotlib", ())

    @pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi"])
    def test_auto_writes_video_with_opencv_when_it_imports(self, suffix):
        choice = _backends.choose_render_backend("auto", suffix, machine(importable={"cv2"}))
        assert (choice.backend, choice.warnings) == ("opencv", ())

    @pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi"])
    def test_auto_falls_back_to_matplotlib_without_opencv(self, suffix):
        choice = _backends.choose_render_backend("auto", suffix, machine(importable={"vedo"}))
        assert choice.backend == "matplotlib"
        assert choice.warnings == (
            "OpenCV not found for fast rendering. Install with: pip install pybvh[opencv]. "
            "Falling back to matplotlib (slower).",
        )

    @pytest.mark.parametrize(
        "named, suffix",
        [("matplotlib", ".mp4"), ("opencv", ".gif"), ("vedo", ".mp4")],
    )
    def test_a_named_backend_wins_over_the_extension(self, named, suffix):
        """vedo is never chosen by auto: it is a look, not a fallback."""
        choice = _backends.choose_render_backend(named, suffix, machine())
        assert (choice.backend, choice.warnings) == (named, ())

    def test_an_unknown_backend_raises(self):
        with pytest.raises(ValueError) as raised:
            _backends.choose_render_backend("opencvv", ".mp4", machine())
        assert str(raised.value) == (
            "Unknown backend 'opencvv'. Choose from: ['auto', 'matplotlib', 'opencv', 'vedo']"
        )


class TestReadEnvironment:
    """The one function that reads the machine."""

    def test_a_package_counts_when_importing_it_succeeds(self, tmp_path, monkeypatch):
        """A package that is installed but fails to import (an OpenCV
        missing its system libraries) does not count."""
        (tmp_path / "pybvh_test_broken.py").write_text("raise ImportError('libGL.so.1')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        importable = _backends.read_environment().importable
        assert "json" in importable
        assert "pybvh_test_broken" not in importable
        assert "pybvh_test_not_installed" not in importable

    @pytest.mark.parametrize(
        "shell, in_notebook",
        [("ZMQInteractiveShell", True), ("TerminalInteractiveShell", False)],
    )
    def test_a_notebook_is_a_jupyter_kernel(self, monkeypatch, shell, in_notebook):
        ipython = pytest.importorskip("IPython")
        monkeypatch.setattr(ipython, "get_ipython", lambda: type(shell, (), {})())
        assert _backends.read_environment().in_notebook is in_notebook

    def test_a_plain_session_is_not_a_notebook(self):
        assert _backends.read_environment().in_notebook is False

    def test_reading_the_session_does_not_import_ipython(self, monkeypatch):
        """A script that never imported IPython is no notebook, and is not
        made to import it."""
        monkeypatch.delitem(sys.modules, "IPython", raising=False)
        assert _backends.read_environment().in_notebook is False
        assert "IPython" not in sys.modules

    @pytest.mark.parametrize(
        "variables, has_display",
        [
            ({}, False),
            ({"DISPLAY": ":0"}, True),
            ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ],
        ids=["none", "x11", "wayland"],
    )
    def test_linux_has_a_display_when_one_is_named(self, monkeypatch, variables, has_display):
        monkeypatch.setattr(sys, "platform", "linux")
        for name in ("DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        assert _backends.read_environment().has_display is has_display

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_macos_and_windows_always_have_a_display(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _backends.read_environment().has_display is True
