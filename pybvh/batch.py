"""Batch file loading and numpy export utilities for BVH datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt

from .read_bvh_file import read_bvh_file
from .bvh import Bvh
from .bvhnode import BvhJoint


def read_bvh_directory(
    dirpath: str | Path,
    pattern: str = "*.bvh",
    sort: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
) -> list[Bvh]:
    """Load all BVH files from a directory.

    Parameters
    ----------
    dirpath : str or Path
        Directory to search for BVH files.
    pattern : str, optional
        Glob pattern to filter files (default ``"*.bvh"``).
    sort : bool, optional
        If True (default), sort files alphabetically for
        deterministic ordering.
    parallel : bool, optional
        If True, load files in parallel using threads.
    max_workers : int or None, optional
        Maximum number of threads when ``parallel=True``.
        None defers to the ``ThreadPoolExecutor`` default.

    Returns
    -------
    list of Bvh
        One Bvh object per file found.

    Raises
    ------
    FileNotFoundError
        If ``dirpath`` does not exist.
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirpath}")

    files = list(dirpath.glob(pattern))
    if sort:
        files.sort()

    if not files:
        return []

    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(read_bvh_file, files))

    return [read_bvh_file(f) for f in files]


def batch_to_numpy(
    bvh_list: list[Bvh],
    representation: str = "euler",
    include_root_pos: bool = True,
    pad: bool = False,
    pad_value: float = 0.0,
) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
    """Convert a list of Bvh objects to NumPy arrays.

    All Bvh objects must share the same skeleton topology (joint
    names and rotation orders).

    Parameters
    ----------
    bvh_list : list of Bvh
        BVH objects to convert.
    representation : str, optional
        Rotation representation: ``'euler'`` (default), ``'6d'``,
        ``'quaternion'``, ``'axisangle'``, or ``'rotmat'``.
    include_root_pos : bool, optional
        If True (default), prepend root position (3 columns) to
        the rotation data.
    pad : bool, optional
        If True, zero-pad shorter sequences to the maximum length
        and return a single 3-D array ``(B, F_max, D)``.
        If False (default), return a list of 2-D arrays.
    pad_value : float, optional
        Value to use for padding (default ``0.0``).

    Returns
    -------
    ndarray or list of ndarray
        If ``pad=True``: array of shape ``(B, F_max, D)``.
        If ``pad=False``: list of arrays, each ``(F_i, D)``.

    Raises
    ------
    ValueError
        If skeletons are incompatible or representation is unknown.
    """
    if not bvh_list:
        raise ValueError("bvh_list is empty.")

    valid_reps = {"euler", "6d", "quaternion", "axisangle", "rotmat"}
    if representation not in valid_reps:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {sorted(valid_reps)}.")

    # Validate skeleton compatibility
    ref = bvh_list[0]
    ref_names = ref.joint_names
    ref_channels = [n.rot_channels for n in ref.nodes if isinstance(n, BvhJoint)]
    for i, bvh in enumerate(bvh_list[1:], start=1):
        names = bvh.joint_names
        if names != ref_names:
            first_diff = next(
                (j for j, (a, b) in enumerate(zip(ref_names, names)) if a != b),
                min(len(ref_names), len(names)),
            )
            raise ValueError(
                f"Skeleton mismatch at index {i}: joint {first_diff} is "
                f"'{ref_names[first_diff] if first_diff < len(ref_names) else 'N/A'}' "
                f"vs '{names[first_diff] if first_diff < len(names) else 'N/A'}'.")
        channels = [n.rot_channels for n in bvh.nodes if isinstance(n, BvhJoint)]
        if channels != ref_channels:
            raise ValueError(
                f"Rotation order mismatch at index {i}.")

    arrays: list[npt.NDArray[np.float64]] = []
    for bvh in bvh_list:
        arr = _bvh_to_flat(bvh, representation, include_root_pos)
        arrays.append(arr)

    if pad:
        max_len = max(a.shape[0] for a in arrays)
        dim = arrays[0].shape[1]
        result = np.full((len(arrays), max_len, dim), pad_value,
                         dtype=np.float64)
        for i, a in enumerate(arrays):
            result[i, :a.shape[0]] = a
        return result

    return arrays


def _bvh_to_flat(
    bvh: Bvh,
    representation: str,
    include_root_pos: bool,
) -> npt.NDArray[np.float64]:
    """Convert a single Bvh to a flat 2-D array ``(F, D)``."""
    if representation == "euler":
        # (F, J, 3) → (F, J*3)
        rot = bvh.joint_angles.reshape(bvh.frame_count, -1)
    elif representation == "6d":
        _, rot_raw, _ = bvh.get_frames_as_6d()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "quaternion":
        _, rot_raw, _ = bvh.get_frames_as_quaternion()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "axisangle":
        _, rot_raw, _ = bvh.get_frames_as_axisangle()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "rotmat":
        _, rot_raw, _ = bvh.get_frames_as_rotmat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if include_root_pos:
        return np.concatenate([bvh.root_pos, rot], axis=1)
    return rot
