"""Batch file loading and numpy export utilities for BVH datasets."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from .io import read_bvh_file
from .bvh import Bvh


def read_bvh_directory(
    dirpath: str | Path,
    pattern: str = "*.bvh",
    sort: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
    world_up: str = "auto",
    lr_mapping: dict[str, str] | None = None,
    skip_errors: bool = False,
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
    world_up : str, optional
        World vertical axis applied to every loaded file.
        ``"auto"`` (default) auto-detects per file.  Pass e.g.
        ``"+y"`` to override all files uniformly.
    lr_mapping : dict or None, optional
        Explicit left/right joint pair mapping applied to every loaded
        file. Useful when a whole dataset shares an unusual naming
        convention the auto-detect heuristic can't parse.
    skip_errors : bool, optional
        If True, files that fail to load emit a ``UserWarning`` and are
        skipped. If False (default), the first failure propagates as
        the original exception. Use True when robustness against
        occasional corrupt files matters more than strict verification.

    Returns
    -------
    list of Bvh
        One Bvh object per successfully loaded file. Shorter than the
        set of matched files when ``skip_errors=True`` and some failed.

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

    from functools import partial
    reader = partial(read_bvh_file, world_up=world_up, lr_mapping=lr_mapping)

    if skip_errors:
        def safe_reader(path: Path) -> Bvh | None:
            try:
                return reader(path)
            except Exception as e:
                warnings.warn(
                    f"read_bvh_directory: skipping {path} "
                    f"({type(e).__name__}: {e})",
                    stacklevel=2)
                return None

        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                maybe_results = list(pool.map(safe_reader, files))
        else:
            maybe_results = [safe_reader(f) for f in files]
        return [r for r in maybe_results if r is not None]

    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(reader, files))
    return [reader(f) for f in files]


def harmonize(
    clips: list[Bvh],
    *,
    reference: Bvh | None = None,
    target_fps: float | None = None,
    target_world_up: str | None = None,
    target_rest_up: str | None = None,
    target_rest_forward: str | None = None,
    on_incompatible: Literal["drop", "raise"] = "drop",
    verbose: bool = True,
) -> list[Bvh]:
    """Apply dataset-level harmonization to a list of clips.

    For each clip, applies in order:

    1. **Topology check vs ``reference``** (if provided). On mismatch,
       the clip is dropped or raises per ``on_incompatible``.
    2. **Bone-proportion retargeting** to ``reference`` (if provided).
    3. **Frame-rate resampling** to ``target_fps`` (if provided and the
       clip's current fps differs by more than ``0.01``).
    4. **World-up reorientation** to ``target_world_up`` (if provided
       and ``bvh.world_up != target_world_up``).  Rotates the entire
       scene — affects offsets, root_pos, and the ``world_up`` flag.
    5. **Rest-up reorientation** to ``target_rest_up`` (if provided and
       ``bvh.rest_up != target_rest_up``).  Rotates only the rest-pose
       offsets and compensates joint rotations so FK positions are
       unchanged; the world frame is untouched.
    6. **Rest-forward reorientation** to ``target_rest_forward`` (if
       provided and ``bvh.rest_forward != target_rest_forward``).
       Rotates rest-pose offsets around the vertical axis so the
       skeleton's rest-pose facing matches.

    The ordering matters: world-up is "heaviest" (touches everything),
    rest-up modifies only rest-pose offsets (leaving world frame intact),
    and rest-forward is a further rotation of those same offsets around
    the vertical.

    Any of the ``reference`` / ``target_*`` kwargs may be ``None`` to
    skip that stage. Passing all as ``None`` returns a shallow copy of
    ``clips`` (no-op).

    Parameters
    ----------
    clips : list of Bvh
        Input clips.
    reference : Bvh or None
        If provided, every clip must match ``reference.matches_topology``.
        Kept clips are then retargeted to ``reference``'s bone offsets.
    target_fps : float or None
        Target frame rate in Hz. Clips whose fps differs by more than
        ``0.01`` are resampled via quaternion SLERP.
    target_world_up : str or None
        Signed-axis string (``'+y'``, ``'-z'``, ...). Clips whose
        ``world_up`` differs are rotated via ``reorient_world_up``.
    target_rest_up : str or None
        Signed-axis string. Clips whose ``rest_up`` differs are
        corrected via ``reorient_rest_up``. Typically used to fix files
        whose rest pose and animation disagree on the up axis.
    target_rest_forward : str or None
        Signed-axis string. Clips whose ``rest_forward`` differs are
        rotated via ``reorient_rest_forward`` so the rest pose faces a
        consistent direction across the dataset.
    on_incompatible : {"drop", "raise"}, optional
        Behavior on topology mismatch with ``reference``.
        ``"drop"`` (default) skips the clip; ``"raise"`` raises
        ``ValueError`` at the first mismatch.
    verbose : bool, optional
        If True (default), emit a ``UserWarning`` per dropped clip
        identifying its index. Set to False for silent dropping.

    Returns
    -------
    list of Bvh
        Harmonized clips. May be shorter than ``clips`` if any were
        dropped.

    Raises
    ------
    ValueError
        If ``on_incompatible`` is not one of the accepted values, or if
        ``on_incompatible='raise'`` and a clip mismatches ``reference``.
    """
    if on_incompatible not in ("drop", "raise"):
        raise ValueError(
            f"on_incompatible must be 'drop' or 'raise', got {on_incompatible!r}")

    out: list[Bvh] = []
    for i, b in enumerate(clips):
        if reference is not None and not reference.matches_topology(b):
            if on_incompatible == "raise":
                raise ValueError(
                    f"Clip at index {i} has incompatible topology with reference.")
            if verbose:
                warnings.warn(
                    f"harmonize: dropping clip at index {i} "
                    f"(topology mismatch with reference)",
                    stacklevel=2)
            continue

        if reference is not None:
            b = b.retarget(reference)
        if target_fps is not None and abs(1.0 / b.frame_time - target_fps) > 1e-2:
            b = b.resample(target_fps)
        if target_world_up is not None and b.world_up != target_world_up:
            b = b.reorient_world_up(target_world_up)
        if target_rest_up is not None and b.rest_up != target_rest_up:
            b = b.reorient_rest_up(target_rest_up)
        if target_rest_forward is not None and b.rest_forward != target_rest_forward:
            b = b.reorient_rest_forward(target_rest_forward)

        out.append(b)

    return out


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
    for i, bvh in enumerate(bvh_list[1:], start=1):
        if ref.matches_topology(bvh):
            continue
        # Topology mismatch — report the first divergent joint / channel
        ref_names = ref.joint_names
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
        _, rot_raw = bvh.to_6d()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "quaternion":
        _, rot_raw = bvh.to_quaternions()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "axisangle":
        _, rot_raw = bvh.to_axisangle()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    elif representation == "rotmat":
        _, rot_raw = bvh.to_rotmat()
        rot = rot_raw.reshape(bvh.frame_count, -1)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if include_root_pos:
        return np.concatenate([bvh.root_pos, rot], axis=1)
    return rot


# =========================================================================
# Normalization utilities
# =========================================================================

def compute_normalization_stats(
    bvh_list: list[Bvh],
    representation: str = "euler",
    include_root_pos: bool = True,
) -> dict[str, np.ndarray]:
    """Compute per-channel mean and std across a dataset of BVH objects.

    Concatenates all frames from all clips, then computes mean and
    standard deviation per feature channel. Compatible with the
    ``Mean.npy`` / ``Std.npy`` convention used by HumanML3D and MDM.

    Parameters
    ----------
    bvh_list : list of Bvh
        Dataset of BVH objects (must share the same skeleton topology).
    representation : str, optional
        Rotation representation: ``'euler'`` (default), ``'6d'``,
        ``'quaternion'``, ``'axisangle'``, or ``'rotmat'``.
    include_root_pos : bool, optional
        If True (default), include root position in the features.

    Returns
    -------
    dict
        ``{"mean": ndarray (D,), "std": ndarray (D,),
        "constant_channels": ndarray of bool (D,)}``.

        ``constant_channels[i]`` is True when the raw standard deviation
        for channel ``i`` was below ``1e-8`` and the guard replaced it
        with ``1.0``. Normalized values on these channels are identically
        zero rather than ~N(0, 1) — use this mask to exclude them from
        per-channel diagnostics.

    Notes
    -----
    Save/load stats with ``np.savez("stats.npz", **stats)`` and
    ``dict(np.load("stats.npz"))``. Bool arrays round-trip cleanly
    through ``.npz``.
    """
    arrays = batch_to_numpy(
        bvh_list, representation=representation,
        include_root_pos=include_root_pos, pad=False)

    # arrays is list[ndarray (F_i, D)]
    all_frames = np.concatenate(arrays, axis=0)  # type: ignore[arg-type]

    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)

    # Guard against zero-std channels; record which channels were guarded
    constant_channels = std < 1e-8
    std = std.copy()
    std[constant_channels] = 1.0

    return {
        "mean": mean,
        "std": std,
        "constant_channels": constant_channels,
    }


def normalize_array(
    data: npt.NDArray[np.float64],
    stats: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Apply z-score normalization: ``(data - mean) / std``.

    Parameters
    ----------
    data : ndarray
        Data to normalize. Last dimension must match ``stats["mean"]``.
    stats : dict
        ``{"mean": ndarray (D,), "std": ndarray (D,)}`` from
        :func:`compute_normalization_stats`.

    Returns
    -------
    ndarray
        Normalized data, same shape as input.
    """
    return (data - stats["mean"]) / stats["std"]


def denormalize_array(
    data: npt.NDArray[np.float64],
    stats: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Reverse z-score normalization: ``data * std + mean``.

    Parameters
    ----------
    data : ndarray
        Normalized data to denormalize.
    stats : dict
        ``{"mean": ndarray (D,), "std": ndarray (D,)}`` from
        :func:`compute_normalization_stats`.

    Returns
    -------
    ndarray
        Denormalized data, same shape as input.
    """
    return data * stats["std"] + stats["mean"]
