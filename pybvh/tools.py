"""Signed axes and the input validators the package shares.

Axes are written as signed strings (``'+y'``, ``'-z'``) throughout pybvh;
:func:`parse_axis` turns one into an :class:`Axis` (column index, sign,
unit vector) and :func:`get_main_direction` snaps a vector to the one it
lies closest to. The orientation a skeleton's geometry implies is computed
from these in :mod:`pybvh._orientation`. The module also holds the input
validators the package shares (file paths, frame time, frame rate, axis
strings).
"""

from __future__ import annotations

import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import numpy.typing as npt


def _validate_bvh_path(filepath: str | Path) -> Path:
    """Validate that a filepath exists and points to a .bvh file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to validate.

    Returns
    -------
    filepath : Path
        The validated filepath as a Path object.

    Raises
    ------
    ValueError
        If the file does not have a ``.bvh`` extension.
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if filepath.suffix != ".bvh":
        raise ValueError(f"{filepath} is not a bvh file")
    elif not filepath.exists():
        raise FileNotFoundError(f"could not find the file {filepath}")
    return filepath


#: The library's one "same rate" tolerance: two frame times are the same
#: rate when they differ by at most this fraction of the larger one.
#: Relative, because the error a computation or a saved table leaves in a
#: frame time scales with the frame time, so one tolerance serves 24 fps
#: and 1000 fps alike. 1e-6 is far above float noise (about 1e-16), and
#: above the error of a frame time derived from a ``time`` column saved
#: at 6 decimals over one second or more. ``1 / 29.97`` and NTSC's
#: ``1001 / 30000`` differ by 1e-6 of the larger, on the boundary; in
#: floats the difference falls just outside.
_SAME_RATE_RTOL = 1e-6


def _validate_frame_time(value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a valid frame time.

    A frame time is ``0`` (unset) or a positive finite number of seconds.
    Shared by the :attr:`Bvh.frame_time` setter and the reader's check of
    a file's ``Frame Time`` line, so the two cannot drift apart.
    """
    # math.isfinite, not np.isfinite: it accepts Fraction, ints beyond
    # int64 and NumPy scalars, where np.isfinite raises TypeError.
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(
            f"frame_time must be 0 (unset) or a positive finite number of seconds, got {value}"
        )


def _validate_fps(value: float, name: str = "fps") -> None:
    """Raise ``ValueError`` unless ``value`` is a positive finite frame rate.

    Unlike a frame time, a rate has no "unset" value: ``1 / inf`` is ``0``,
    so an infinite rate would silently become an unset frame time.
    """
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"{name} must be a positive finite number of frames per second, got {value}"
        )


# --------------------------------------------------------------------------------------------

# Axis detection utilities
# These are used by bvhplot and by ML pipeline methods (foot contacts, root trajectory)


def get_main_direction(
    coord_array: npt.NDArray[np.float64],
    tol: float = 1e-6,
) -> str | None:
    """Return the signed axis string (e.g. ``'+y'``) for the dominant component.

    Parameters
    ----------
    coord_array : np.ndarray
        1-D array of length 3 representing an (x, y, z) vector.
    tol : float, optional
        Minimum vector norm to consider valid (default ``1e-6``).
        Vectors shorter than this return ``None``.

    Returns
    -------
    main_dir : str or None
        Signed axis label such as ``'+x'``, ``'-z'``, etc.
        Returns ``None`` if the vector norm is below *tol*.
    """
    if float(np.linalg.norm(coord_array)) < tol:
        return None

    main_direction_idx = int(np.argmax(np.abs(coord_array)))
    if coord_array[main_direction_idx] < 0:
        main_dir = "-"
    else:
        main_dir = "+"

    if main_direction_idx == 0:
        main_dir += "x"
    elif main_direction_idx == 1:
        main_dir += "y"
    elif main_direction_idx == 2:
        main_dir += "z"
    else:
        raise ValueError("Invalid index")

    return main_dir


def extract_sign(ax: str) -> bool:
    """Return ``True`` if the axis string has a ``'+'`` sign, ``False`` if ``'-'``.

    Parameters
    ----------
    ax : str
        Signed axis string, e.g. ``'+x'`` or ``'-z'``.

    Returns
    -------
    is_positive : bool
        ``True`` for positive, ``False`` for negative.
    """
    if ax[0] == "+":
        return True
    elif ax[0] == "-":
        return False
    else:
        raise ValueError("The sign of the axis should be either '+' or '-'.")


_VALID_AXIS_STRINGS = frozenset({"+x", "-x", "+y", "-y", "+z", "-z"})

_AXIS_CHAR_TO_IDX = {"x": 0, "y": 1, "z": 2}

Axis = namedtuple("Axis", ["index", "sign", "vector"])
Axis.__doc__ = """A signed coordinate axis parsed into machine-usable form.

The parsed counterpart of pybvh's signed axis strings (``'+y'``, ``'-z'``), produced by :func:`pybvh.parse_axis` and returned by :attr:`Bvh.world_up_axis`, :attr:`Bvh.rest_forward_axis` and :attr:`Bvh.rest_up_axis`.

Fields: ``index`` (int, 0/1/2 — the axis's column in any ``(..., 3)`` coordinate array), ``sign`` (float, ``+1.0`` / ``-1.0`` — multiply by it to make the coordinate positive along the axis direction), ``vector`` (float64 ``(3,)`` unit vector along the axis, sign included).

A plain named tuple: construct one directly for a synthetic skeleton or a test, unpack it as ``index, sign, vector = axis``, or read the fields by name.

Comparing two ``Axis`` values with ``==`` raises, because tuple equality reaches the ``vector`` field and NumPy will not reduce an array to a single truth value. Compare ``(index, sign)`` instead — those two fields determine the axis completely, and ``vector`` is derived from them.
"""


def _axis_to_vector(ax: str) -> npt.NDArray[np.float64]:
    """Convert a signed axis string ('+y') to a unit vector ([0, 1, 0])."""
    vec = np.zeros(3)
    vec[_AXIS_CHAR_TO_IDX[ax[1]]] = 1.0 if ax[0] == "+" else -1.0
    return vec


def _axis_index_sign(ax: str) -> tuple[int, float]:
    """Split a signed axis string ('-z') into ``(coordinate index, sign)``.

    The scalar companion of :func:`_axis_to_vector`: ``index`` selects the coordinate column, ``sign`` (``+1.0`` / ``-1.0``) restores the direction, so ``coords[..., index] * sign`` is the up-positive height. The single parsing implementation behind :attr:`Bvh.world_up_axis` and the internal up-axis consumers.
    """
    return _AXIS_CHAR_TO_IDX[ax[1]], 1.0 if ax[0] == "+" else -1.0


def parse_axis(axis: str, *, allow_unsigned: bool = False) -> Axis:
    """Parse a signed axis string into an :class:`Axis`.

    The public form of the parser behind :attr:`Bvh.world_up_axis` and friends,
    for code that holds an axis string of its own — a dataset convention
    from a config file, or one of the axis strings pybvh returns.

    Parameters
    ----------
    axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, ``'-z'``. The axis letter is case-insensitive.
    allow_unsigned : bool, optional
        If True, also accept a bare letter (``'x'``), reading it as the
        positive direction. Default False — a bare letter states no
        direction, and defaulting it to positive invents one the caller
        did not ask for. Use it only where the sign is genuinely
        irrelevant, as for ``mirror``'s ``lateral``.

    Returns
    -------
    Axis
        The parsed ``(index, sign, vector)`` triple.

    Raises
    ------
    ValueError
        If ``axis`` is not one of the accepted strings.

    Examples
    --------
    >>> parse_axis('-z').index, parse_axis('-z').sign
    (2, -1.0)
    >>> heights = coords[..., up.index] * up.sign   # up = parse_axis('-y')
    """
    normalized = _validate_axis_string(axis, allow_unsigned=allow_unsigned)
    index, sign = _axis_index_sign(normalized)
    return Axis(index, sign, _axis_to_vector(normalized))


def _validate_axis_string(value: object, *, allow_unsigned: bool = False) -> str:
    """Validate and normalize a signed axis string.

    Accepts '+x'/'-x'/'+y'/'-y'/'+z'/'-z' (case-insensitive in the
    axis char). With ``allow_unsigned=True``, a bare axis char
    ('x'/'y'/'z') is also accepted and normalized to its positive form —
    for parameters where the sign is irrelevant (e.g. ``mirror``'s
    ``lateral``). Returns the normalized form. Raises ValueError
    on any other input.
    """
    normalized = value.lower() if isinstance(value, str) else value
    if allow_unsigned and isinstance(normalized, str) and normalized in _AXIS_CHAR_TO_IDX:
        normalized = "+" + normalized
    if normalized not in _VALID_AXIS_STRINGS:
        expected = sorted(_VALID_AXIS_STRINGS)
        if allow_unsigned:
            expected = expected + sorted(_AXIS_CHAR_TO_IDX)
        raise ValueError(f"Axis must be one of {expected}, got {value!r}")
    return normalized  # type: ignore[return-value]
