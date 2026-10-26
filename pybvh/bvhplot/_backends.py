"""Which backend draws a call, and what the machine offers it.

The router asks this module which backend runs and calls it. The machine
is read at one point, :func:`read_environment`, into an
:class:`Environment`; every choice here is a function of the request and
that value, so a test builds the value instead of changing the machine.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from collections.abc import Collection, Container, Mapping
from dataclasses import dataclass
from typing import NamedTuple

# The extra of pybvh that installs each optional package a backend imports.
_EXTRAS = {"cv2": "opencv", "k3d": "interactive", "vedo": "viewer"}

FRAME_BACKENDS = frozenset({"matplotlib", "vedo"})
RENDER_BACKENDS = frozenset({"auto", "opencv", "matplotlib", "vedo"})
PLAY_BACKENDS = frozenset({"auto", "k3d", "vedo", "opencv", "matplotlib"})


@dataclass(frozen=True)
class Environment:
    """What the machine offers the backends, read once per call.

    Attributes
    ----------
    in_notebook : bool
        The call runs in a Jupyter notebook kernel.
    has_display : bool
        A window can open: macOS and Windows always, Linux when
        ``DISPLAY`` or ``WAYLAND_DISPLAY`` is set.
    importable : Container[str]
        The optional packages that import (``"cv2"``, ``"k3d"``,
        ``"vedo"``), asked with ``in``. A package counts when importing it
        succeeds, not when it is installed: an OpenCV that fails to load
        its system libraries does not count.
    has_ffmpeg : bool
        The ``ffmpeg`` executable is on PATH, so videos can be encoded
        as H.264.
    """

    in_notebook: bool
    has_display: bool
    importable: Container[str]
    has_ffmpeg: bool


@dataclass(frozen=True)
class Choice:
    """The backend a call runs, and the warnings the choice owes the
    caller (each a message, warned by the router at the user's line)."""

    backend: str
    warnings: tuple[str, ...] = ()


class _ImportableOnDemand:
    """The packages that import, each tried the first time it is asked
    about and remembered with the error of a failed import. Trying every
    optional package up front would import VTK, OpenCV and ipywidgets into
    a process that only draws a matplotlib figure."""

    def __init__(self) -> None:
        self._failures: dict[str, ImportError | None] = {}

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.failure(name) is None

    def failure(self, name: str) -> ImportError | None:
        """The error importing *name* raised, or ``None`` when it imports."""
        if name not in self._failures:
            try:
                __import__(name)
                self._failures[name] = None
            except ImportError as error:
                self._failures[name] = error
        return self._failures[name]


def _detect_notebook() -> bool:
    # A Jupyter kernel imports IPython before any user code runs, so a
    # process that has not imported it is no notebook; importing it here
    # would cost every render() call in a script.
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    return type(ipython.get_ipython()).__name__ == "ZMQInteractiveShell"


def _has_display() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True  # macOS/Windows always have a windowing system
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def read_environment() -> Environment:
    """Read the machine: the one place bvhplot asks it anything."""
    return Environment(
        in_notebook=_detect_notebook(),
        has_display=_has_display(),
        importable=_ImportableOnDemand(),
        has_ffmpeg=shutil.which("ffmpeg") is not None,
    )


def install_hint(package: str) -> str:
    """How to install the optional *package* (``"cv2"``, ``"k3d"``,
    ``"vedo"``) with pybvh, or the ``"ffmpeg"`` executable, which no pip
    extra provides."""
    if package == "ffmpeg":
        return "Install ffmpeg (e.g. apt install ffmpeg / conda install ffmpeg)"
    return f"Install with: pip install pybvh[{_EXTRAS[package]}]"


def _check_known(requested: str, valid: frozenset[str]) -> None:
    if requested not in valid:
        raise ValueError(f"Unknown backend {requested!r}. Choose from: {sorted(valid)}")


class _Requirement(NamedTuple):
    package: str  # the optional package the backend imports
    subject: str  # the backend, as its error names it
    what: str  # what the error says the backend requires


# The optional package each backend of each public call imports.
_REQUIREMENTS = {
    ("frame", "vedo"): _Requirement("vedo", "vedo backend", "vedo"),
    ("render", "vedo"): _Requirement("vedo", "vedo backend", "vedo"),
    ("render", "opencv"): _Requirement("cv2", "OpenCV backend", "opencv-python"),
    ("play", "k3d"): _Requirement("k3d", "k3d backend", "k3d and ipywidgets"),
    ("play", "vedo"): _Requirement("vedo", "vedo backend", "vedo"),
    ("play", "opencv_notebook"): _Requirement("cv2", "The opencv play backend", "opencv-python"),
}


def check_importable(call: str, backend: str, environment: Environment) -> None:
    """Raise ``ImportError`` naming the install when the package *backend*
    imports for the public *call* does not import; a backend that needs
    no optional package always passes."""
    requirement = _REQUIREMENTS.get((call, backend))
    if requirement is None:
        return
    package, subject, what = requirement
    if package in environment.importable:
        return
    # The import's own error says why an installed package failed to load
    # (OpenCV without libGL, k3d without ipywidgets).
    importable = environment.importable
    cause = importable.failure(package) if isinstance(importable, _ImportableOnDemand) else None
    raise ImportError(f"{subject} requires {what}. {install_hint(package)}") from cause


def choose_frame_backend(requested: str) -> str:
    """The backend ``frame(backend=requested)`` runs: always the one
    named, since ``frame`` has no auto."""
    _check_known(requested, FRAME_BACKENDS)
    return requested


_SLOW_RENDER = (
    f"OpenCV not found for fast rendering. {install_hint('cv2')}. "
    "Falling back to matplotlib (slower)."
)


# Every backend accepts these whether it reads them or not: a backend that
# does not read them draws each skeleton in its own panel, which already
# gives the arrangement they ask for.
_ACCEPTED_BY_EVERY_BACKEND = frozenset({"spacing", "match_size"})

# How a message names an option that is not a keyword of its own.
_SHOWN = {"turntable": "camera='turntable'"}


def _either(words: list[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


def _with_unread_options(
    call: str,
    requested: str,
    choice: Choice,
    given: Collection[str],
    reads: Mapping[str, frozenset[str]],
) -> Choice:
    """Apply the rule for an option the chosen backend does not read
    (ADR 0006) to *choice*: *given* are the options the caller passed
    (not ``None``), *reads* each backend's declaration of the options it
    reads. A named backend raises ``ValueError`` naming every given
    option it does not read; under ``"auto"`` the choice gains a warning
    naming them and the backend chosen. Either message offers the
    backends that read them all, when there are any."""
    unread = sorted(set(given) - reads[choice.backend] - _ACCEPTED_BY_EVERY_BACKEND)
    if not unread:
        return choice
    names = _either([_SHOWN.get(option, option) for option in unread])
    it = "it" if len(unread) == 1 else "them"
    readers = [repr(backend) for backend, read in sorted(reads.items()) if set(unread) <= read]
    instead = (
        f"use backend={_either(readers)}, which {'reads' if len(readers) == 1 else 'read'} {it}"
        if readers
        else ""
    )
    if requested == "auto":
        ignored = "it is ignored" if len(unread) == 1 else "they are ignored"
        hint = f" To draw {it}, {instead}." if readers else ""
        message = (
            f"{call}() chose backend={choice.backend!r}, which does not read {names}, "
            f"so {ignored}.{hint}"
        )
        return Choice(choice.backend, (*choice.warnings, message))
    hint = f", or {instead}" if readers else ""
    raise ValueError(
        f"{call}(backend={choice.backend!r}) does not read {names}. Leave {it} unset{hint}."
    )


def _render_reads() -> dict[str, frozenset[str]]:
    """The options of ``render`` each backend reads, as it declares them
    beside its entry function."""
    # Imported here: a backend module is loaded only when a call needs it
    # (matplotlib's imports pyplot).
    from . import _matplotlib, _opencv, _vedo_offscreen

    return {
        "matplotlib": _matplotlib.RENDER_READS,
        "opencv": _opencv.RENDER_READS,
        "vedo": _vedo_offscreen.RENDER_READS,
    }


def choose_render_backend(
    requested: str, suffix: str, environment: Environment, given: Collection[str] = ()
) -> Choice:
    """The backend ``render(filepath, backend=requested)`` runs, for a
    file ending in *suffix* (lower case), passed the options *given*: the
    keywords whose value is not ``None``, ``camera="turntable"`` as
    ``"turntable"``.

    A named backend runs as named. Under ``"auto"``, the extensions the
    video module's table gives to matplotlib
    (:func:`._video.matplotlib_under_auto`: ``.html`` and the animated
    images) go to matplotlib even when OpenCV imports, and without a
    warning when it does not; every other extension goes to OpenCV when
    it imports, and else to
    matplotlib with a warning naming the install. ``"vedo"`` (the
    shadowed capsule renderer) is never chosen by auto: it is a
    deliberate look, not a fallback.

    A given option the backend does not read raises ``ValueError`` when
    the backend is named and adds a warning under ``"auto"``;
    ``spacing`` and ``match_size`` are accepted by every backend.
    """
    # Imported here: the video module imports this one.
    from ._video import matplotlib_under_auto

    _check_known(requested, RENDER_BACKENDS)
    if requested != "auto":
        choice = Choice(requested)
    elif matplotlib_under_auto(suffix):
        choice = Choice("matplotlib")
    elif "cv2" in environment.importable:
        choice = Choice("opencv")
    else:
        choice = Choice("matplotlib", (_SLOW_RENDER,))
    if not given:
        return choice
    return _with_unread_options("render", requested, choice, given, _render_reads())


_NO_INTERACTIVE = f"No interactive backend (k3d, vedo) found. {install_hint('k3d')}"
_SLOW_PLAY = (
    f"OpenCV not found for fast rendering. {install_hint('cv2')}. "
    "Falling back to matplotlib (slow for long clips)."
)


def choose_play_backend(requested: str, environment: Environment) -> Choice:
    """The backend ``play(backend=requested)`` runs.

    A named backend runs as named; ``"opencv"`` is the user-facing name
    (matching ``render()``) of the notebook inline video, which runs as
    ``"opencv_notebook"`` and needs a notebook. ``"auto"`` takes the first
    tier the session and the installed packages allow:

    - tier 1, interactive: k3d in a notebook, the vedo window in a script
      with a display;
    - tier 2, the fast fallback: the OpenCV inline video in a notebook,
      matplotlib's window in a script;
    - tier 3, the slow fallback: matplotlib's inline player in a
      notebook.

    A fallback warns that no interactive backend imports (unless vedo
    does, and only the display is missing), and tier 3 also that OpenCV
    does not.
    """
    _check_known(requested, PLAY_BACKENDS)
    if requested == "opencv":
        check_importable("play", "opencv_notebook", environment)
        if not environment.in_notebook:
            raise ValueError(
                "backend='opencv' plays an inline video and only works "
                "inside a Jupyter notebook. In a script, use "
                "backend='vedo' (interactive window) or render() to a "
                "file instead."
            )
        return Choice("opencv_notebook")
    if requested != "auto":
        return Choice(requested)

    if environment.in_notebook:
        if "k3d" in environment.importable:
            return Choice("k3d")
        slow = "cv2" not in environment.importable
        backend = "matplotlib" if slow else "opencv_notebook"
    else:
        if environment.has_display and "vedo" in environment.importable:
            return Choice("vedo")
        slow, backend = False, "matplotlib"

    warned = []
    if "vedo" not in environment.importable:
        warned.append(_NO_INTERACTIVE)
    if slow:
        warned.append(_SLOW_PLAY)
    return Choice(backend, tuple(warned))


# Notebook widgets (k3d, matplotlib's jshtml player) and matplotlib windows
# cannot keep up with high frame rates (120 fps), so play() caps them here.
# The OpenCV notebook video plays any rate in the browser's player, and the
# vedo window keeps its actors and steps them on a timer.
PLAY_MAX_FPS = 30.0
_UNCAPPED_PLAY_BACKENDS = frozenset({"opencv_notebook", "vedo"})


def play_frame_step(backend: str, fps: float | None, rate: float) -> int:
    """Every how many frames *backend* plays a clip of *rate* fps, when
    ``play`` was given *fps* (``None`` for the clip's own rate).

    A rate given is played as given. Under ``None``, a backend that cannot
    keep up plays every step-th frame, the smallest whole step bringing
    the rate to :data:`PLAY_MAX_FPS` or below; the kept frames keep their
    moments, so the rate played is *rate* over the step.
    """
    if fps is not None or backend in _UNCAPPED_PLAY_BACKENDS or rate <= PLAY_MAX_FPS:
        return 1
    return math.ceil(rate / PLAY_MAX_FPS)
