"""The backend-choice module: which backend runs, from the request and an
environment the conftest fixture builds, so that every tier and every
missing package is reached on any machine. The k3d widget and the vedo
window cannot run in the suite; the paths of play() that open no window
are also tested through play() itself, in test_plot.py."""

from __future__ import annotations

import sys

import pytest

from pybvh.bvhplot import _backends

NO_INTERACTIVE = (
    "No interactive backend (k3d, vedo) found. Install with: pip install pybvh[interactive]"
)
SLOW_PLAY = (
    "OpenCV not found for fast rendering. Install with: pip install pybvh[opencv]. "
    "Falling back to matplotlib (slow for long clips)."
)


class TestPlayTiers:
    """play()'s auto choice: the best backend the session and the installed
    packages allow, and a warning for each fallback naming what to install."""

    @pytest.mark.parametrize(
        "offered, backend, warned",
        [
            pytest.param(
                dict(notebook=True, importable={"k3d", "cv2", "vedo"}),
                "k3d",
                (),
                id="notebook-k3d",
            ),
            pytest.param(
                dict(notebook=True, importable={"cv2"}),
                "opencv_notebook",
                (NO_INTERACTIVE,),
                id="notebook-opencv",
            ),
            pytest.param(
                dict(notebook=True, importable={"cv2", "vedo"}),
                "opencv_notebook",
                (),
                id="notebook-opencv-vedo-installed",
            ),
            pytest.param(
                dict(notebook=True),
                "matplotlib",
                (NO_INTERACTIVE, SLOW_PLAY),
                id="notebook-nothing",
            ),
            pytest.param(
                dict(notebook=True, importable={"vedo"}),
                "matplotlib",
                (SLOW_PLAY,),
                id="notebook-vedo-only",
            ),
            pytest.param(
                dict(display=True, importable={"vedo", "k3d", "cv2"}),
                "vedo",
                (),
                id="window-vedo",
            ),
            pytest.param(
                dict(display=True, importable={"k3d", "cv2"}),
                "matplotlib",
                (NO_INTERACTIVE,),
                id="window-no-vedo",
            ),
            pytest.param(dict(importable={"vedo"}), "matplotlib", (), id="headless-vedo-installed"),
            pytest.param(dict(), "matplotlib", (NO_INTERACTIVE,), id="headless-nothing"),
        ],
    )
    def test_auto_picks_the_best_backend_the_session_allows(
        self, environment, offered, backend, warned
    ):
        choice = _backends.choose_play_backend("auto", environment(**offered))
        assert choice.backend == backend
        assert choice.warnings == warned

    @pytest.mark.parametrize("named", ["k3d", "vedo", "matplotlib"])
    def test_a_named_backend_runs_without_a_warning(self, environment, named):
        choice = _backends.choose_play_backend(
            named, environment(importable={"k3d", "vedo", "cv2"})
        )
        assert (choice.backend, choice.warnings) == (named, ())

    def test_opencv_names_the_notebook_video(self, environment):
        choice = _backends.choose_play_backend(
            "opencv", environment(notebook=True, importable={"cv2"})
        )
        assert (choice.backend, choice.warnings) == ("opencv_notebook", ())

    def test_opencv_outside_a_notebook_raises(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend("opencv", environment(display=True, importable={"cv2"}))
        assert str(raised.value) == (
            "backend='opencv' plays an inline video and only works inside a Jupyter "
            "notebook. In a script, use backend='vedo' (interactive window) or render() "
            "to a file instead."
        )

    def test_opencv_without_opencv_raises_naming_the_install(self, environment):
        with pytest.raises(ImportError) as raised:
            _backends.choose_play_backend("opencv", environment(notebook=True))
        assert str(raised.value) == (
            "The opencv play backend requires opencv-python. "
            "Install with: pip install pybvh[opencv]"
        )

    def test_an_unknown_backend_raises(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend("opencvv", environment())
        assert str(raised.value) == (
            "Unknown backend 'opencvv'. Choose from: "
            "['auto', 'k3d', 'matplotlib', 'opencv', 'vedo']"
        )


class TestPlayRateCap:
    """Under fps=None, the k3d widget and matplotlib play at most 30 fps,
    by keeping every step-th frame: the smallest whole step that brings
    the clip's rate to 30 or below."""

    @pytest.mark.parametrize(
        "backend, rate, step",
        [
            ("k3d", 120.0, 4),
            ("matplotlib", 100.0, 4),  # 100 / 4 = 25; 100 / 3 > 30
            ("matplotlib", 31.0, 2),
            ("matplotlib", 30.0, 1),
            ("opencv_notebook", 120.0, 1),  # a video player keeps up
            ("vedo", 120.0, 1),  # vedo's timer keeps up
        ],
    )
    def test_the_step_brings_the_rate_to_thirty_or_below(self, backend, rate, step):
        assert _backends.play_frame_step(backend, None, rate) == step

    @pytest.mark.parametrize("backend", ["k3d", "matplotlib"])
    def test_a_rate_given_is_never_capped(self, backend):
        assert _backends.play_frame_step(backend, 120.0, 120.0) == 1


class TestRenderChoice:
    """render()'s choice: by the file's extension and the installed
    packages under auto; a named backend wins."""

    @pytest.mark.parametrize("suffix", [".html", ".webp", ".apng", ".gif"])
    @pytest.mark.parametrize("importable", [{"cv2"}, set()], ids=["opencv", "no-opencv"])
    def test_auto_sends_html_and_animated_images_to_matplotlib(
        self, environment, suffix, importable
    ):
        """Without the missing-OpenCV warning: auto would not have chosen
        OpenCV for the file either."""
        choice = _backends.choose_render_backend("auto", suffix, environment(importable=importable))
        assert (choice.backend, choice.warnings) == ("matplotlib", ())

    @pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi", ".mkv"])
    def test_auto_gives_videos_and_unknown_extensions_to_opencv(self, environment, suffix):
        """An extension no backend writes (``.mkv``) goes the same way,
        and OpenCV's refusal names the extensions it writes."""
        choice = _backends.choose_render_backend("auto", suffix, environment(importable={"cv2"}))
        assert (choice.backend, choice.warnings) == ("opencv", ())

    @pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi", ".mkv"])
    def test_auto_gives_them_to_matplotlib_without_opencv(self, environment, suffix):
        choice = _backends.choose_render_backend("auto", suffix, environment(importable={"vedo"}))
        assert choice.backend == "matplotlib"
        assert choice.warnings == (
            "OpenCV not found for fast rendering. Install with: pip install pybvh[opencv]. "
            "Falling back to matplotlib (slower).",
        )

    @pytest.mark.parametrize(
        "named, suffix",
        [("matplotlib", ".mp4"), ("opencv", ".gif"), ("vedo", ".mp4")],
    )
    def test_a_named_backend_wins_over_the_extension(self, environment, named, suffix):
        """vedo is never chosen by auto: it is a look, not a fallback."""
        choice = _backends.choose_render_backend(named, suffix, environment())
        assert (choice.backend, choice.warnings) == (named, ())

    def test_an_unknown_backend_raises(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_render_backend("opencvv", ".mp4", environment())
        assert str(raised.value) == (
            "Unknown backend 'opencvv'. Choose from: ['auto', 'matplotlib', 'opencv', 'vedo']"
        )


class TestFrameChoice:
    @pytest.mark.parametrize("named", ["matplotlib", "vedo"])
    def test_a_known_backend_runs_as_named(self, named):
        assert _backends.choose_frame_backend(named) == named

    def test_an_unknown_backend_raises(self):
        with pytest.raises(ValueError) as raised:
            _backends.choose_frame_backend("opencv")
        assert str(raised.value) == "Unknown backend 'opencv'. Choose from: ['matplotlib', 'vedo']"


class TestImportChecks:
    """Before each dispatch, a backend whose package does not import
    raises, naming what to install."""

    @pytest.mark.parametrize(
        "call, backend, message",
        [
            (
                "frame",
                "vedo",
                "vedo backend requires vedo. Install with: pip install pybvh[viewer]",
            ),
            (
                "render",
                "vedo",
                "vedo backend requires vedo. Install with: pip install pybvh[viewer]",
            ),
            (
                "render",
                "opencv",
                "OpenCV backend requires opencv-python. Install with: pip install pybvh[opencv]",
            ),
            (
                "play",
                "k3d",
                "k3d backend requires k3d and ipywidgets. "
                "Install with: pip install pybvh[interactive]",
            ),
            ("play", "vedo", "vedo backend requires vedo. Install with: pip install pybvh[viewer]"),
            (
                "play",
                "opencv_notebook",
                "The opencv play backend requires opencv-python. "
                "Install with: pip install pybvh[opencv]",
            ),
        ],
    )
    def test_a_missing_package_raises_naming_the_install(self, environment, call, backend, message):
        with pytest.raises(ImportError) as raised:
            _backends.check_importable(call, backend, environment(notebook=True, display=True))
        assert str(raised.value) == message

    @pytest.mark.parametrize(
        "call, backend, package",
        [
            ("frame", "vedo", "vedo"),
            ("render", "vedo", "vedo"),
            ("render", "opencv", "cv2"),
            ("play", "k3d", "k3d"),
            ("play", "vedo", "vedo"),
            ("play", "opencv_notebook", "cv2"),
        ],
    )
    def test_a_package_that_imports_passes(self, environment, call, backend, package):
        _backends.check_importable(call, backend, environment(importable={package}))

    def test_a_package_that_fails_to_load_raises_from_its_own_error(self, tmp_path, monkeypatch):
        """Read from the machine, the import's error is the cause, so the
        traceback says why an installed package did not load."""
        (tmp_path / "k3d.py").write_text("raise ImportError('No module named ipywidgets')\n")
        monkeypatch.delitem(sys.modules, "k3d", raising=False)
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ImportError, match="k3d backend requires") as raised:
            _backends.check_importable("play", "k3d", _backends.read_environment())
        assert str(raised.value.__cause__) == "No module named ipywidgets"

    @pytest.mark.parametrize("call", ["frame", "render", "play"])
    def test_matplotlib_needs_no_optional_package(self, environment, call):
        _backends.check_importable(call, "matplotlib", environment())


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

    def test_a_plain_process_is_not_a_notebook(self):
        assert _backends.read_environment().in_notebook is False

    def test_reading_the_machine_does_not_import_ipython(self, monkeypatch):
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
