"""Which backend draws a call, and what the machine offers it.

The router asks this module which backend runs and calls it. The machine
is read at one point, :func:`read_environment`, into an
:class:`Environment`; every choice here is a function of the request and
that value, so a test builds the value instead of changing the machine.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Container
from dataclasses import dataclass, field

# The extra of pybvh that installs each optional package a backend imports.
_EXTRAS = {"cv2": "opencv", "k3d": "interactive", "vedo": "viewer"}

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
    """

    in_notebook: bool
    has_display: bool
    importable: Container[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Choice:
    """The backend a call runs, and the warnings the choice owes the
    caller (each a message, warned by the router at the user's line)."""

    backend: str
    warnings: tuple[str, ...] = ()


class _ImportableOnDemand:
    """The packages that import, each tried the first time it is asked
    about and remembered. Trying every optional package up front would
    import VTK, OpenCV and ipywidgets into a process that only draws a
    matplotlib figure."""

    def __init__(self) -> None:
        self._known: dict[str, bool] = {}

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name not in self._known:
            try:
                __import__(name)
                self._known[name] = True
            except ImportError:
                self._known[name] = False
        return self._known[name]


def _detect_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


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


# (call, backend) -> (the package the backend imports, the backend as its
# error names it, what the error says to install)
_REQUIREMENTS = {
    ("play", "opencv_notebook"): ("cv2", "The opencv play backend", "opencv-python"),
}


def check_importable(call: str, backend: str, environment: Environment) -> None:
    """Raise ``ImportError`` naming the install when the package *backend*
    imports for the public *call* does not import; a backend that needs
    no optional package always passes."""
    requirement = _REQUIREMENTS.get((call, backend))
    if requirement is None:
        return
    package, subject, what = requirement
    if package not in environment.importable:
        raise ImportError(f"{subject} requires {what}. {install_hint(package)}")


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
        backend, tier = (
            ("opencv_notebook", 2) if "cv2" in environment.importable else ("matplotlib", 3)
        )
    else:
        if environment.has_display and "vedo" in environment.importable:
            return Choice("vedo")
        backend, tier = "matplotlib", 2

    warned = []
    if "vedo" not in environment.importable:
        warned.append(_NO_INTERACTIVE)
    if tier == 3:
        warned.append(_SLOW_PLAY)
    return Choice(backend, tuple(warned))
