"""Batch file loading and numpy export utilities for BVH datasets."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from .io import read_bvh_file
from .bvh import Bvh
from .features import to_feature_array


@dataclass
class HarmonizeReport:
    """Per-call summary of what :func:`harmonize` did to each clip.

    All fields use JSON-native types so the report can be serialized
    directly with ``json.dumps(dataclasses.asdict(report))`` and embedded
    as audit metadata alongside a preprocessed dataset.

    Attributes
    ----------
    kept_indices : list of int
        Indices (into the input ``clips`` list) of clips that survived
        the topology gate.
    kept_sources : list of str or None
        ``source_path`` of each kept clip, aligned with ``kept_indices``.
    dropped_indices : list of int
        Indices of clips that were dropped by the topology gate.
    dropped_sources : list of str or None
        ``source_path`` of each dropped clip, aligned with ``dropped_indices``.
    drop_reasons : list of str
        One human-readable reason per dropped clip, aligned with
        ``dropped_indices``.
    applied_stages : list of dict
        One dict per kept clip, aligned with ``kept_indices``. Each dict
        records which harmonization stages ran for that clip, with
        before→after where meaningful. Possible keys:
        ``"retarget"``, ``"resample"``, ``"world_up"``, ``"rest_up"``,
        ``"rest_forward"``, ``"euler_order"``. Empty dict means the clip
        passed the gate without needing any transformation.
    """
    kept_indices: list[int] = field(default_factory=list)
    kept_sources: list[str | None] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)
    dropped_sources: list[str | None] = field(default_factory=list)
    drop_reasons: list[str] = field(default_factory=list)
    applied_stages: list[dict[str, str]] = field(default_factory=list)


def _natural_sort_key(path: Path) -> tuple:
    """Sort key comparing embedded digit runs numerically, case-insensitively.

    ``(is_digit, value)`` pairs keep number/text comparisons type-safe at
    run boundaries (a digit run never compares against a text run).
    """
    return tuple(
        (True, int(token)) if token.isdigit() else (False, token.lower())
        for token in re.split(r"(\d+)", str(path)))


def read_bvh_directory(
    dirpath: str | Path,
    pattern: str = "*.bvh",
    sort: bool | str = True,
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
    sort : bool or {"lexicographic", "natural"}, optional
        File ordering. ``True`` (default) and ``"lexicographic"`` sort
        by full path string — the same order Python's ``sorted()`` over
        paths produces, so a list built elsewhere with ``sorted()``
        (labels, split manifests) stays index-aligned; note it puts
        ``file10.bvh`` before ``file2.bvh``. ``"natural"`` compares
        embedded digit runs numerically (``file2`` before ``file10``,
        case-insensitive) — opt-in, because silently diverging from the
        ecosystem's lexicographic default would misalign such parallel
        lists. ``False`` keeps the filesystem's glob order
        (non-deterministic across platforms).
    parallel : bool, optional
        If True, load files in parallel using threads. Parsing is
        CPU-bound and GIL-limited, so this mainly helps on slow storage
        (network filesystems, cold disks); expect little speedup on a
        warm local disk.
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
    if isinstance(sort, str) and sort not in ("lexicographic", "natural"):
        raise ValueError(
            f"sort must be a bool, 'lexicographic' or 'natural', got {sort!r}")

    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirpath}")

    files = list(dirpath.glob(pattern))
    if sort == "natural":
        files.sort(key=_natural_sort_key)
    elif sort:
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


@overload
def harmonize(
    clips: list[Bvh],
    *,
    reference: Bvh | None = ...,
    target_fps: float | None = ...,
    target_world_up: str | None = ...,
    target_rest_up: str | None = ...,
    target_rest_forward: str | None = ...,
    target_euler_order: str | None = ...,
    on_incompatible: Literal["drop", "raise"] = ...,
    verbose: bool = ...,
    return_report: Literal[False] = ...,
) -> list[Bvh]: ...
@overload
def harmonize(
    clips: list[Bvh],
    *,
    reference: Bvh | None = ...,
    target_fps: float | None = ...,
    target_world_up: str | None = ...,
    target_rest_up: str | None = ...,
    target_rest_forward: str | None = ...,
    target_euler_order: str | None = ...,
    on_incompatible: Literal["drop", "raise"] = ...,
    verbose: bool = ...,
    return_report: Literal[True],
) -> tuple[list[Bvh], HarmonizeReport]: ...
def harmonize(
    clips: list[Bvh],
    *,
    reference: Bvh | None = None,
    target_fps: float | None = None,
    target_world_up: str | None = None,
    target_rest_up: str | None = None,
    target_rest_forward: str | None = None,
    target_euler_order: str | None = None,
    on_incompatible: Literal["drop", "raise"] = "drop",
    verbose: bool = True,
    return_report: bool = False,
) -> list[Bvh] | tuple[list[Bvh], HarmonizeReport]:
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
    7. **Euler-order re-expression** to ``target_euler_order`` (if
       provided and any per-joint Euler order differs). Re-expresses
       each joint's stored Euler angles in the target order while
       preserving the underlying rotations.

    The ordering matters: world-up is "heaviest" (touches everything),
    rest-up modifies only rest-pose offsets (leaving world frame intact),
    and rest-forward is a further rotation of those same offsets around
    the vertical. Euler-order re-expression runs last because it only
    rewrites channel layout, not geometry.

    Any of the ``reference`` / ``target_*`` kwargs may be ``None`` to
    skip that stage. Passing all as ``None`` returns a shallow copy of
    ``clips`` (no-op).

    Parameters
    ----------
    clips : list of Bvh
        Input clips.
    reference : Bvh or None
        If provided, every clip must match ``reference.matches_hierarchy``
        (same joints, same parent structure — rest offsets are allowed
        to differ since retargeting will overwrite them next).
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
    target_euler_order : str or None
        Three-character order like ``'XYZ'`` / ``'ZYX'``. When set,
        clips with any joint whose Euler order differs are re-expressed
        in the target order via :meth:`Bvh.change_euler_order`. This is
        orientation-preserving: underlying rotations are unchanged; only
        the channel layout is rewritten. Numerical drift can occur
        on gimbal-lock-adjacent rotations — for bit-exact round-trips
        across the conversion, prefer rotation-invariant representations
        (``'6d'`` / ``'quat'``) downstream.
    on_incompatible : {"drop", "raise"}, optional
        Behavior on topology mismatch with ``reference``.
        ``"drop"`` (default) skips the clip; ``"raise"`` raises
        ``ValueError`` at the first mismatch.
    verbose : bool, optional
        If True (default), emit a single ``UserWarning`` at end of call
        when one or more clips were dropped, summarizing how many were
        dropped and identifying the first few. Set to False to silence
        the summary entirely.
    return_report : bool, optional
        If True, return ``(clips, report)`` where ``report`` is a
        :class:`HarmonizeReport` describing every stage applied to every
        kept clip plus per-clip drop reasons. Default ``False`` keeps
        the return type as a plain ``list[Bvh]``.

    Returns
    -------
    list of Bvh, or (list of Bvh, HarmonizeReport)
        Harmonized clips (and optional report). The list may be shorter
        than ``clips`` if any were dropped.

    Raises
    ------
    ValueError
        If ``on_incompatible`` is not one of the accepted values, or if
        ``on_incompatible='raise'`` and a clip mismatches ``reference``.
    """
    if on_incompatible not in ("drop", "raise"):
        raise ValueError(
            f"on_incompatible must be 'drop' or 'raise', got {on_incompatible!r}")

    report = HarmonizeReport()
    out: list[Bvh] = []
    for i, b in enumerate(clips):
        if reference is not None and not reference.matches_hierarchy(b, match_offsets=False):
            reason = "topology mismatch with reference"
            if on_incompatible == "raise":
                raise ValueError(
                    f"Clip at index {i} has incompatible topology with reference.")
            report.dropped_indices.append(i)
            report.dropped_sources.append(b.source_path)
            report.drop_reasons.append(reason)
            continue

        stages: dict[str, str] = {}
        if reference is not None:
            b = b.retarget(reference)
            stages["retarget"] = "applied"
        if target_fps is not None and abs(1.0 / b.frame_time - target_fps) > 1e-2:
            old_fps = 1.0 / b.frame_time
            b = b.resample(target_fps)
            stages["resample"] = f"{old_fps:.4g}→{target_fps:.4g}"
        if target_world_up is not None and b.world_up != target_world_up:
            old = b.world_up
            b = b.reorient_world_up(target_world_up)
            stages["world_up"] = f"{old}→{target_world_up}"
        if target_rest_up is not None and b.rest_up != target_rest_up:
            old = b.rest_up
            b = b.reorient_rest_up(target_rest_up)
            stages["rest_up"] = f"{old}→{target_rest_up}"
        if target_rest_forward is not None and b.rest_forward != target_rest_forward:
            old = b.rest_forward
            b = b.reorient_rest_forward(target_rest_forward)
            stages["rest_forward"] = f"{old}→{target_rest_forward}"
        if target_euler_order is not None and any(
                order != target_euler_order for order in b.euler_orders):
            b = b.change_euler_order(target_euler_order)
            stages["euler_order"] = f"→{target_euler_order}"

        report.kept_indices.append(i)
        report.kept_sources.append(b.source_path)
        report.applied_stages.append(stages)
        out.append(b)

    if verbose and report.dropped_indices:
        warnings.warn(_harmonize_summary(report, len(clips)), stacklevel=2)

    if return_report:
        return out, report
    return out


def _harmonize_summary(report: HarmonizeReport, total: int) -> str:
    """Build the end-of-call UserWarning text describing the drops."""
    n_drop = len(report.dropped_indices)
    preview_n = min(5, n_drop)
    preview: list[str] = []
    for k in range(preview_n):
        idx = report.dropped_indices[k]
        src = report.dropped_sources[k]
        preview.append(f"'{src}'" if src is not None else f"index {idx}")
    more = "" if n_drop <= preview_n else f", +{n_drop - preview_n} more"
    return (
        f"harmonize: dropped {n_drop}/{total} clips (topology mismatch with "
        f"reference). First divergent: {', '.join(preview)}{more}. "
        f"Pass return_report=True for the full drop list with reasons.")


def _clip_label(bvh: Bvh, idx: int) -> str:
    """Identify a clip by source_path when available, falling back to index."""
    if bvh.source_path is not None:
        return f"index {idx} ('{bvh.source_path}')"
    return f"index {idx}"


def _hierarchy_mismatch_message(ref: Bvh, bvh: Bvh, ref_idx: int, idx: int) -> str:
    """Build a diagnostic for a matches_hierarchy failure."""
    ref_label = _clip_label(ref, ref_idx)
    div_label = _clip_label(bvh, idx)

    if len(ref.nodes) != len(bvh.nodes):
        return (
            f"Skeleton hierarchy mismatch between {ref_label} and "
            f"{div_label}: node count {len(ref.nodes)} vs {len(bvh.nodes)}.")

    for j, (n1, n2) in enumerate(zip(ref.nodes, bvh.nodes)):
        if n1.name != n2.name:
            return (
                f"Skeleton hierarchy mismatch between {ref_label} and "
                f"{div_label}: node {j} is '{n1.name}' vs '{n2.name}'.")
        p1 = n1.parent.name if n1.parent is not None else None
        p2 = n2.parent.name if n2.parent is not None else None
        if p1 != p2:
            return (
                f"Skeleton hierarchy mismatch between {ref_label} and "
                f"{div_label}: node '{n1.name}' parent is "
                f"{p1!r} vs {p2!r}.")
        if not np.allclose(n1.offset, n2.offset, atol=1e-6):
            return (
                f"Skeleton hierarchy mismatch between {ref_label} and "
                f"{div_label}: rest offset for '{n1.name}' differs "
                f"({list(n1.offset)} vs {list(n2.offset)}). "
                f"Pre-harmonize bone proportions via "
                f"harmonize(clips, reference=ref).")

    return (
        f"Skeleton hierarchy mismatch between {ref_label} and {div_label}.")


def _channel_mismatch_message(ref: Bvh, bvh: Bvh, ref_idx: int, idx: int,
                               representation: str) -> str:
    """Build a diagnostic for a matches_channels failure."""
    ref_label = _clip_label(ref, ref_idx)
    div_label = _clip_label(bvh, idx)

    ref_orders = ref.euler_orders
    orders = bvh.euler_orders
    joint_names = ref.joint_names

    first_diff = next(
        (j for j, (a, b) in enumerate(zip(ref_orders, orders)) if a != b),
        None,
    )

    if first_diff is None:
        # Channel mismatch but not in Euler orders — must be root position channels.
        return (
            f"Root position-channel mismatch between {ref_label} and "
            f"{div_label}: {ref.root.pos_channels} vs "
            f"{bvh.root.pos_channels}.")

    return (
        f"Rotation-channel mismatch between {ref_label} and {div_label} "
        f"(joint '{joint_names[first_diff]}': '{ref_orders[first_diff]}' vs "
        f"'{orders[first_diff]}'). For representation='{representation}', "
        f"mismatched Euler orders corrupt the concatenated tensor's channel "
        f"layout — pre-harmonize the dataset with "
        f"harmonize(clips, target_euler_order='<ORDER>'). For "
        f"representation='6d' / 'quat' / 'rotmat', the tensor is "
        f"order-agnostic and this check is skipped.")


def batch_to_numpy(
    bvh_list: list[Bvh],
    representation: str = "euler",
    include_root_pos: bool = True,
    pad: bool = False,
    pad_value: float = 0.0,
) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
    """Convert a list of Bvh objects to NumPy arrays.

    All Bvh objects must share the same skeleton hierarchy. For
    representations whose channel layout depends on the source Euler
    order (``'euler'``, ``'axisangle'``), all clips must additionally
    share the same per-joint Euler orders. For rotation-invariant
    representations (``'6d'``, ``'quat'``, ``'rotmat'``) the
    Euler-order check is skipped.

    Parameters
    ----------
    bvh_list : list of Bvh
        BVH objects to convert.
    representation : str, optional
        Rotation representation: ``'euler'`` (default), ``'6d'``,
        ``'quat'``, ``'axisangle'``, or ``'rotmat'``.
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

    channel_layout_matters = representation in ("euler", "axisangle")

    # Validate skeleton compatibility
    ref = bvh_list[0]
    for i, bvh in enumerate(bvh_list[1:], start=1):
        if not ref.matches_hierarchy(bvh):
            raise ValueError(_hierarchy_mismatch_message(ref, bvh, 0, i))
        if channel_layout_matters and not ref.matches_channels(bvh):
            raise ValueError(
                _channel_mismatch_message(ref, bvh, 0, i, representation))

    # Per-clip extraction delegates to the features module — one place
    # owns the flat (F, D) layout, the valid-representation set, and its
    # error message.
    arrays: list[npt.NDArray[np.float64]] = [
        to_feature_array(
            bvh, representation=representation,
            include_root_pos=include_root_pos)
        for bvh in bvh_list
    ]

    if pad:
        max_len = max(a.shape[0] for a in arrays)
        dim = arrays[0].shape[1]
        result = np.full((len(arrays), max_len, dim), pad_value,
                         dtype=np.float64)
        for i, a in enumerate(arrays):
            result[i, :a.shape[0]] = a
        return result

    return arrays
