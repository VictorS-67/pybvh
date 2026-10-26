"""Which backend draws a call, and what the machine offers it.

The router asks this module which backend runs and calls it. The machine
is read at one point, :func:`read_environment`, into an
:class:`Environment`; every choice here is a function of the request and
that value, so a test builds the value instead of changing the machine.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Container
from dataclasses import dataclass
from typing import NamedTuple

# The extra of pybvh that installs each optional package a backend imports.
_EXTRAS = {"cv2": "opencv", "k3d": "interactive", "vedo": "viewer"}

FRAME_BACKENDS = frozenset({"matplotlib", "vedo"})
RENDER_BACKENDS = frozenset({"auto", "opencv", "matplotlib", "vedo"})
PLAY_BACKENDS = frozenset({"auto", "k3d", "vedo", "opencv", "matplotlib"})

# Formats only the matplotlib/pillow pipeline can write — OpenCV's
# VideoWriter handles video containers only (its .gif support is a
# dedicated pillow-based path inside render_opencv).
_MPL_ONLY_EXTENSIONS = frozenset({".html", ".webp", ".apng", ".gif"})


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
    """

    in_notebook: bool
    has_display: bool
    importable: Container[str]


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
    )


def install_hint(package: str) -> str:
    """How to install the optional *package* (``"cv2"``, ``"k3d"``,
    ``"vedo"``) with pybvh."""
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


def choose_render_backend(requested: str, suffix: str, environment: Environment) -> Choice:
    """The backend ``render(filepath, backend=requested)`` runs, for a
    file ending in *suffix* (lower case).

    A named backend runs as named. Under ``"auto"``, the extensions
    OpenCV cannot write (``.html``, ``.webp``, ``.apng``, ``.gif``) go to
    matplotlib even when OpenCV imports; every other one goes to OpenCV
    when it imports, and else to matplotlib with a warning naming the
    install. ``"vedo"`` (the shadowed capsule renderer) is never chosen by
    auto: it is a deliberate look, not a fallback.
    """
    _check_known(requested, RENDER_BACKENDS)
    if requested != "auto":
        return Choice(requested)
    if suffix in _MPL_ONLY_EXTENSIONS:
        return Choice("matplotlib")
    if "cv2" in environment.importable:
        return Choice("opencv")
    return Choice("matplotlib", (_SLOW_RENDER,))


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
