"""Contact detection on plain arrays.

The detector behind :func:`pybvh.analysis.foot_contacts` and :func:`pybvh.analysis.ground_contacts`: the tested joints' positions in, a per-frame contact label per joint out. It takes the joints' positions, their rest positions and the root's (the skeleton scale of the default thresholds), the frame time, the up axis and a :class:`_ContactSettings`, and imports no :class:`~pybvh.bvh.Bvh`: the two detectors read those from the clip, resolve the joints, run the checks whose messages name the clip (its frame time, its ``world_up``) and delegate here. Below the entry point are the kernels it calls: the speed and the clearance signals, the threshold resolution (the skeleton scale, Otsu's split, the stance calibration), the thresholding with hysteresis, the run filters, the confidence and the diagnostics.

The contact reference is estimated here, over the tested joints, with :func:`pybvh.geometry._floor_level`; the scene ground (``Bvh.floor_height``) is never read (ADR 0002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from . import geometry
from .signal import box_filter_smooth

# The defaults of the settings that some method does not read: the entry
# points' keywords default to None, so that a setting given is told from one
# left out, and gather() puts these in place.
_DEFAULT_VEL_SMOOTH_DURATION = 1.0 / 30.0
_DEFAULT_FLOOR = "auto"

_METHODS_READING_VELOCITY = ("velocity", "combined")
_METHODS_READING_HEIGHT = ("height", "combined")


@dataclass(frozen=True, kw_only=True)
class _ContactSettings:
    """The settings of one contact detection, gathered once at the detector's entry point.

    :func:`pybvh.analysis.foot_contacts` and :func:`pybvh.analysis.ground_contacts` declare these settings as keywords and document them; they build this object from them with :meth:`gather`, which validates the call and puts the defaults in place, and what lies below the entry points receives it instead of the keywords. The fields are the entry points' parameters of the same names, with every default resolved.
    """

    method: str
    vel_threshold: float | None
    vel_smooth_duration: float
    height_threshold: float | None
    floor: float | str
    min_contact_duration: float
    min_gap_duration: float
    hysteresis: float
    adaptive: bool
    height_reference: str

    @classmethod
    def gather(
        cls,
        *,
        method: str,
        vel_threshold: float | None,
        vel_smooth_duration: float | None,
        height_threshold: float | None,
        floor: float | str | None,
        min_contact_duration: float,
        min_gap_duration: float,
        hysteresis: float,
        adaptive: bool,
        height_reference: str | None,
        default_height_reference: str,
    ) -> _ContactSettings:
        """The settings of one call, checked, with each setting left at ``None`` given its default.

        ``default_height_reference`` is the detector's own default for ``height_reference``: ``"velocity"`` for :func:`pybvh.analysis.foot_contacts`, ``"floor"`` for :func:`pybvh.analysis.ground_contacts`.
        """
        if method not in ("velocity", "height", "combined"):
            raise ValueError(
                f"Unknown method {method!r}. Choose 'combined', 'velocity', or 'height'."
            )
        reads_velocity = method in _METHODS_READING_VELOCITY
        reads_height = method in _METHODS_READING_HEIGHT
        if not reads_height:
            for name, value in (
                ("height_threshold", height_threshold),
                ("floor", floor),
                ("height_reference", height_reference),
            ):
                if value is not None:
                    raise ValueError(
                        f"{name} is not read by method={method!r}, which runs no height test; "
                        "drop it, or choose method='combined' or 'height'."
                    )
        if not reads_velocity:
            for name, value in (
                ("vel_threshold", vel_threshold),
                ("vel_smooth_duration", vel_smooth_duration),
            ):
                if value is not None:
                    raise ValueError(
                        f"{name} is not read by method={method!r}, which runs no speed test; "
                        "drop it, or choose method='combined' or 'velocity'."
                    )
            if height_reference is not None:
                raise ValueError(
                    f"height_reference is not read by method={method!r}: it anchors the "
                    "height threshold to the speed test, which only method='combined' runs."
                )
        if height_reference is not None and height_threshold is not None:
            raise ValueError(
                f"height_reference is not read by method={method!r} with a height_threshold "
                "given: it anchors only a height threshold the detector derives."
            )
        if height_reference is None:
            height_reference = default_height_reference
        if height_reference not in ("velocity", "floor"):
            raise ValueError(
                f"height_reference must be 'velocity' or 'floor', got {height_reference!r}"
            )
        if floor is None:
            floor = _DEFAULT_FLOOR
        if isinstance(floor, str) and floor not in ("auto", "min"):
            raise ValueError(f"floor must be 'auto', 'min' or a float, got {floor!r}")
        if vel_smooth_duration is None:
            vel_smooth_duration = _DEFAULT_VEL_SMOOTH_DURATION
        for name, duration in (
            ("vel_smooth_duration", vel_smooth_duration),
            ("min_contact_duration", min_contact_duration),
            ("min_gap_duration", min_gap_duration),
        ):
            if not duration >= 0:
                raise ValueError(f"{name} must be >= 0 (seconds), got {duration}")
        if not 0 <= hysteresis < 1:
            raise ValueError(f"hysteresis must be in [0, 1), got {hysteresis}")
        for name, threshold in (
            ("vel_threshold", vel_threshold),
            ("height_threshold", height_threshold),
        ):
            if threshold is None:
                continue
            values = np.asarray(threshold, dtype=np.float64)
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"{name} must be finite and > 0, got {threshold}")
        if adaptive:
            # Under combined with height_reference="velocity" the stance
            # calibration, not Otsu's split, derives the height threshold.
            stance_calibrated = method == "combined" and height_reference == "velocity"
            splits_speed = reads_velocity and vel_threshold is None
            splits_height = reads_height and height_threshold is None and not stance_calibrated
            if not (splits_speed or splits_height):
                reasons = []
                if reads_velocity:
                    reasons.append("vel_threshold is given")
                if reads_height:
                    reasons.append(
                        "height_threshold is given"
                        if height_threshold is not None
                        else "the height threshold is calibrated on the stance frames "
                        "(height_reference='velocity'), not split"
                    )
                raise ValueError(
                    f"adaptive=True derives no threshold under method={method!r}: "
                    f"{' and '.join(reasons)}; drop adaptive=True."
                )
        return cls(
            method=method,
            vel_threshold=vel_threshold,
            vel_smooth_duration=vel_smooth_duration,
            height_threshold=height_threshold,
            floor=floor,
            min_contact_duration=min_contact_duration,
            min_gap_duration=min_gap_duration,
            hysteresis=hysteresis,
            adaptive=adaptive,
            height_reference=height_reference,
        )

    @property
    def reads_velocity(self) -> bool:
        """Whether the method tests the joints' speed."""
        return self.method in _METHODS_READING_VELOCITY

    @property
    def reads_height(self) -> bool:
        """Whether the method tests the joints' clearance above the contact reference."""
        return self.method in _METHODS_READING_HEIGHT


@dataclass(frozen=True, kw_only=True, eq=False)
class ContactReport:
    """What one contact detection applied, and the signals it compared.

    Returned beside the contacts by :func:`pybvh.analysis.foot_contacts` and :func:`pybvh.analysis.ground_contacts` (and their ``Bvh`` wrappers) when ``return_info=True``. Every field is always present: a field the call's method does not use is ``None``. A per-joint value is ``(n_joints,)`` in the order of ``joints``, the order of the contacts' columns, whether one joint is tested or several; nothing is averaged over the joints. Frozen, and its arrays are read-only; ``==`` compares identity, since the fields hold arrays.

    Attributes
    ----------
    joints : tuple of str
        The tested joints' names, in column order (``ground_contacts`` maps node indices back to names).
    method : {"combined", "velocity", "height"}
        The method applied.
    vel_smooth_duration : float or None
        The span (seconds) the speed is smoothed over; ``None`` under ``method="height"``, which reads no speed.
    vel_smooth_frames : int or None
        That span in frames, the window applied. ``None`` where no speed was measured: under ``method="height"``, and on a one-frame clip.
    height_reference : {"velocity", "floor"} or None
        How the height threshold was anchored; ``None`` where it anchored nothing: under ``method="velocity"`` and ``method="height"``, and with a ``height_threshold`` given.
    min_contact_duration, min_gap_duration : float
        The duration filters applied, in seconds.
    hysteresis : float
        The hysteresis band applied, as a fraction of each threshold.
    skeleton_scale : float or None
        The scale the default thresholds are written in (the mean rest distance from the root to the tested joints); ``None`` where no threshold was derived from it, every threshold the method reads being given.
    vel_threshold : ndarray, shape (n_joints,), or None
        The speed threshold applied to each joint, in units per second; ``None`` under ``method="height"``.
    adaptive_used_velocity : ndarray of bool, shape (n_joints,), or None
        With ``adaptive=True`` and a derived ``vel_threshold``, whether each joint's threshold is Otsu's split (``False``: the signal was not bimodal and the scaled default stands in). ``None`` where no split was attempted.
    speed : ndarray, shape (F, n_joints), or None
        Each joint's speed, in units per second, the signal compared against ``vel_threshold`` (frame 0 repeats frame 1). ``None`` under ``method="height"``, and on a one-frame clip, which has no speed to measure: there the speed test passes every frame, so ``method="combined"`` labels by the height test alone.
    height_threshold : ndarray, shape (n_joints,), or None
        The clearance threshold applied to each joint, in world units; ``None`` under ``method="velocity"``.
    adaptive_used_height : ndarray of bool, shape (n_joints,), or None
        As ``adaptive_used_velocity``, for a derived ``height_threshold`` under ``method="height"``.
    floor : float or None
        The contact reference, along the raw ``world_up`` axis: the level the clearance is measured from, estimated over the tested joints (``floor="auto"`` or ``"min"``) or given. Never the scene ground, :attr:`Bvh.floor_height <pybvh.bvh.Bvh.floor_height>` (ADR 0002). ``None`` under ``method="velocity"``.
    clearance : ndarray, shape (F, n_joints), or None
        Each joint's height above ``floor``, positive up, the signal compared against ``height_threshold``. ``None`` under ``method="velocity"``.
    confidence : ndarray, shape (n_joints,)
        Each joint's detection decisiveness in ``[0, 1]`` (not a probability): how far, on average over its contact frames, its signals sit below their thresholds, as a fraction of each threshold, and under ``method="combined"`` the geometric mean of that with the fraction of frames where the two tests agree. ``0.0`` for a joint never in contact.
    foot_skate_mean, foot_skate_max : ndarray, shape (n_joints,)
        The mean and the largest, over each joint's contact runs of two frames or more, of how far the joint drifts horizontally within the run, divided by the skeleton scale; a planted joint should be near ``0``. ``0.0`` for a joint with no such run.
    airborne_fraction : float
        The fraction of frames in which no tested joint is in contact.
    height_at_contact : ndarray, shape (n_joints,), or None
        Each joint's mean ``clearance`` over its contact frames, ``nan`` for a joint never in contact. ``None`` under ``method="velocity"``, which measures no clearance.
    """

    joints: tuple[str, ...]
    method: str
    vel_smooth_duration: float | None
    vel_smooth_frames: int | None
    height_reference: str | None
    min_contact_duration: float
    min_gap_duration: float
    hysteresis: float
    skeleton_scale: float | None
    vel_threshold: npt.NDArray[np.float64] | None
    adaptive_used_velocity: npt.NDArray[np.bool_] | None
    speed: npt.NDArray[np.float64] | None
    height_threshold: npt.NDArray[np.float64] | None
    adaptive_used_height: npt.NDArray[np.bool_] | None
    floor: float | None
    clearance: npt.NDArray[np.float64] | None
    confidence: npt.NDArray[np.float64]
    foot_skate_mean: npt.NDArray[np.float64]
    foot_skate_max: npt.NDArray[np.float64]
    airborne_fraction: float
    height_at_contact: npt.NDArray[np.float64] | None


def _contacts_from_positions(
    joint_positions: npt.NDArray[np.float64],
    joint_names: list[str],
    joint_rest: npt.NDArray[np.float64],
    root_rest: npt.NDArray[np.float64],
    *,
    frame_time: float,
    up_idx: int,
    up_sign: float,
    settings: _ContactSettings,
    return_info: bool,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ContactReport]:
    """Per-frame contact labels of the tested joints, from their positions.

    ``joint_positions`` is ``(F, n_joints, 3)``, in world coordinates; ``joint_names`` names its joints, for the report; ``joint_rest`` is their rest positions ``(n_joints, 3)`` and ``root_rest`` the root's ``(3,)``, which give the skeleton scale the default thresholds are written in. The up axis is ``up_idx`` signed by ``up_sign``. Returns the ``(F, n_joints)`` labels (1.0 in contact), with the :class:`ContactReport` when ``return_info``.

    The caller has checked the inputs whose messages name the clip: at least one joint, and a ``frame_time`` above zero whenever the method reads the speed or a duration filter is on.
    """
    num_joints = len(joint_names)
    F = joint_positions.shape[0]

    needs_scale = (settings.reads_velocity and settings.vel_threshold is None) or (
        settings.reads_height and settings.height_threshold is None
    )

    scale: float | None = None
    if needs_scale:
        scale = _skeleton_scale(joint_rest, root_rest)

    # ---- Signals (F, num_joints), threshold resolution ----
    speed = None
    clearance = None
    # The thresholds in use: the caller's, or a per-foot (nf,) array resolved here.
    vel_thr_used: float | npt.NDArray[np.float64] | None = None
    height_thr_used: float | npt.NDArray[np.float64] | None = None
    floor_raw = None
    vel_adaptive_used = None
    height_adaptive_used = None
    vel_smooth_frames = None

    if settings.reads_velocity:
        if F < 2:
            # No motion info — treat as "no velocity evidence against contact"
            # (speed 0 < any positive threshold) so combined falls back to height.
            speed = np.zeros((F, num_joints))
        else:
            disp = joint_positions[1:] - joint_positions[:-1]  # (F-1, nj, 3)
            # Condition the speed estimator over a fixed physical time span
            # so detection does not depend on the capture rate: box-averaging
            # the displacement *vectors* (norm-of-mean) cancels the
            # high-frequency jitter that adjacent-frame differencing picks up
            # at high fps.  Window is 1 at <= 30 fps — the raw signal, exactly.
            vel_smooth_frames = min(F - 1, max(1, round(settings.vel_smooth_duration / frame_time)))
            if vel_smooth_frames > 1:
                disp = box_filter_smooth(disp, vel_smooth_frames, axis=0)
            sp = np.linalg.norm(disp, axis=-1) / frame_time  # (F-1, nj), u/s
            speed = np.concatenate([sp[0:1], sp], axis=0)  # frame-0 propagated
        if settings.vel_threshold is not None:
            vel_thr_used = settings.vel_threshold
        else:
            assert scale is not None
            base = 0.12 * scale  # 12% of root-to-foot rest distance per second
            if settings.adaptive and F >= 2:
                vel_thr_used, vel_adaptive_used = _resolve_adaptive(speed, base)
            else:
                vel_thr_used = base

    if settings.reads_height:
        heights_signed = joint_positions[:, :, up_idx] * up_sign  # up-positive
        if isinstance(settings.floor, str):
            # "auto"/"min": the reference is estimated from the coords in
            # use, over the joints being tested — never from the scene
            # ground (Bvh.floor_height), which is measured over ALL nodes
            # and would sit an end site's length below these joints.
            percentile = 0.0 if settings.floor == "min" else 2.0
            floor_raw = geometry._floor_level(
                joint_positions, up_idx, up_sign, percentile=percentile
            )
        else:
            floor_raw = float(settings.floor)
        floor_signed = floor_raw * up_sign
        clearance = heights_signed - floor_signed
        if settings.height_threshold is not None:
            height_thr_used = settings.height_threshold
        else:
            assert scale is not None
            base = 0.013 * scale  # ~1.3% of root-to-foot rest distance above floor
            # A speed signal exists exactly when method is "combined" here.
            if speed is not None and settings.height_reference == "velocity":
                # Calibrate the height threshold per foot to its own stance
                # level (handles retargeting hover); reduces to `base` on rigs
                # where the foot reaches the floor.
                assert vel_thr_used is not None
                height_thr_used = _velocity_informed_height(clearance, speed, vel_thr_used, base)
            elif settings.adaptive:
                height_thr_used, height_adaptive_used = _resolve_adaptive(clearance, base)
            else:
                height_thr_used = base

    mask, confidence = _detect_contacts(
        speed,
        clearance,
        method=settings.method,
        vel_threshold=vel_thr_used,
        height_threshold=height_thr_used,
        hysteresis=settings.hysteresis,
    )

    # ---- Morphological duration filters (time → frames) ----
    # The caller refused frame_time == 0 with nonzero durations.
    dt = frame_time
    min_contact_frames = (
        max(1, round(settings.min_contact_duration / dt))
        if settings.min_contact_duration > 0
        else 1
    )
    min_gap_frames = (
        max(1, round(settings.min_gap_duration / dt)) if settings.min_gap_duration > 0 else 1
    )

    if min_contact_frames > 1:
        mask = _filter_short_runs(mask, min_contact_frames, value=True)
    if min_gap_frames > 1:
        mask = _filter_short_runs(mask, min_gap_frames, value=False)

    contacts = mask.astype(np.float64)

    if not return_info:
        return contacts

    diag_scale = scale if scale is not None else _skeleton_scale(joint_rest, root_rest)
    skate_mean, skate_max, airborne, height_at_contact = _contact_diagnostics(
        mask, joint_positions, clearance, up_idx, diag_scale
    )
    height_reference_applied = settings.method == "combined" and settings.height_threshold is None
    report = ContactReport(
        joints=tuple(joint_names),
        method=settings.method,
        vel_smooth_duration=(
            float(settings.vel_smooth_duration) if settings.reads_velocity else None
        ),
        vel_smooth_frames=vel_smooth_frames,
        height_reference=settings.height_reference if height_reference_applied else None,
        min_contact_duration=float(settings.min_contact_duration),
        min_gap_duration=float(settings.min_gap_duration),
        hysteresis=float(settings.hysteresis),
        skeleton_scale=None if scale is None else float(scale),
        vel_threshold=None if vel_thr_used is None else _per_joint(vel_thr_used, num_joints),
        adaptive_used_velocity=_optional_read_only(vel_adaptive_used),
        # The zeros a one-frame clip is tested with are no measured speed.
        speed=_optional_read_only(speed if F >= 2 else None),
        height_threshold=None
        if height_thr_used is None
        else _per_joint(height_thr_used, num_joints),
        adaptive_used_height=_optional_read_only(height_adaptive_used),
        floor=None if floor_raw is None else float(floor_raw),
        clearance=_optional_read_only(clearance),
        confidence=_read_only(confidence),
        foot_skate_mean=_read_only(skate_mean),
        foot_skate_max=_read_only(skate_max),
        airborne_fraction=airborne,
        height_at_contact=_optional_read_only(height_at_contact),
    )
    return contacts, report


_A = TypeVar("_A", bound=np.generic)


def _per_joint(
    threshold: float | npt.NDArray[np.float64], num_joints: int
) -> npt.NDArray[np.float64]:
    """A threshold applied, scalar or per joint, as the report's read-only ``(n_joints,)``."""
    return _read_only(np.broadcast_to(np.asarray(threshold, dtype=np.float64), (num_joints,)))


def _read_only(array: npt.NDArray[_A]) -> npt.NDArray[_A]:
    """A read-only copy of ``array``, for a field of the frozen :class:`ContactReport`."""
    frozen = np.array(array)
    frozen.flags.writeable = False
    return frozen


def _optional_read_only(array: npt.NDArray[_A] | None) -> npt.NDArray[_A] | None:
    """:func:`_read_only`, passing ``None`` through."""
    return None if array is None else _read_only(array)


def _rest_distance_to_root(
    joint_rest: npt.NDArray[np.float64],
    root_rest: npt.NDArray[np.float64],
) -> float | None:
    """Mean rest-pose distance from the root to the joints, or None when it measures nothing.

    ``joint_rest`` is ``(n_joints, 3)``, ``root_rest`` ``(3,)``. The distance is Euclidean, so it does not depend on the rest pose's orientation, and only the joints given contribute, so finger or spine subdivision leaves it alone: on the feet it is roughly half a humanoid's height, and it scales linearly with the whole skeleton. ``None`` when no joint is given or every joint sits on the root. The one measurement behind :func:`pybvh.analysis.skeleton_size`, which raises on ``None``, and the skeleton scale of the contact detectors' default thresholds, which use ``1.0`` in its place.
    """
    dists = [float(np.linalg.norm(position - root_rest)) for position in joint_rest]
    if not dists or max(dists) <= 0:
        return None
    return float(np.mean(dists))


def _skeleton_scale(
    joint_rest: npt.NDArray[np.float64],
    root_rest: npt.NDArray[np.float64],
) -> float:
    """Scale reference for the contact detectors' default thresholds.

    The mean rest-pose distance from the root to the tested joints, :func:`pybvh.analysis.skeleton_size`'s measurement. Where it measures nothing (every joint sits on the root) the scale is ``1.0``, so the thresholds don't collapse to zero: the public ``skeleton_size`` raises instead of fabricating, but a threshold scale must exist even where a measurement does not.
    """
    distance = _rest_distance_to_root(joint_rest, root_rest)
    return 1.0 if distance is None else distance


def _run_extents(
    mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Per-row ``[start, end)`` extent of the True run covering that row.

    Fully vectorized over a 2-D boolean array ``(F, M)``. Returns
    ``(starts, ends)``, each ``(F, M)``: for a row inside a True run, the
    run's start (inclusive) and end (exclusive); values on rows outside
    any run are sentinels (``< 0`` / ``> F``) — callers mask them out by
    ANDing with the run membership itself.
    """
    F, M = mask.shape
    # Pad with False top/bottom so a run starting at row 0 or ending at
    # row F-1 still produces ±1 transitions in `diffs`.
    padded = np.zeros((F + 2, M), dtype=np.int8)
    padded[1:-1] = mask
    diffs = np.diff(padded, axis=0)  # (F+1, M); +1 at run starts, -1 just past run ends

    pos_col = np.arange(F + 1)[:, None]
    # Most-recent run start position at or before each row (inclusive).
    start_idx = np.where(diffs == 1, pos_col, -1)
    start_pos = np.maximum.accumulate(start_idx, axis=0)
    # Next run end position strictly after each row.
    end_idx = np.where(diffs == -1, pos_col, F + 2)
    end_pos = np.minimum.accumulate(end_idx[::-1], axis=0)[::-1]
    return start_pos[:F], end_pos[1 : F + 1]


def _filter_short_runs(
    mask: npt.NDArray[np.bool_],
    min_run: int,
    value: bool = True,
) -> npt.NDArray[np.bool_]:
    """Remove runs of ``value`` shorter than ``min_run`` frames, per column.

    With ``value=True`` (default): zeroes out short True runs
    (morphological open — removes contact jitter).  With
    ``value=False``: fills in short False runs (morphological close —
    bridges short gaps in an otherwise continuous contact phase).

    Runs touching frame 0 or the last frame are exempt: the clip truncates them,
    so their observed length is only a lower bound and cannot be judged "short".

    Returns the input unchanged when ``min_run <= 1``.
    """
    if min_run <= 1:
        return mask
    if mask.ndim != 2:
        raise ValueError("_filter_short_runs expects a 2-D boolean array")

    F = mask.shape[0]
    # Search for runs of True in `m`; flip up front to also cover the
    # value=False case (filling short False gaps in `mask`).
    m = mask if value else ~mask
    starts, ends = _run_extents(m)
    run_length = ends - starts
    # A run touching frame 0 or the last frame is "open": the clip cut it off, so
    # its observed length is only a lower bound — we can't conclude it is short.
    # Exempt it, so the offline detector never extrapolates across the clip edge
    # (don't remove a truncated contact; don't fill a truncated gap, e.g. a foot
    # lifting at the very end is not "closed" back into a held stance).
    open_run = (starts == 0) | (ends == F)
    short_run_mask = m & (run_length < min_run) & ~open_run

    if value:
        return mask & ~short_run_mask
    return mask | short_run_mask


def _true_runs(col: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` ranges of contiguous True runs in a 1-D mask."""
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _hysteresis_mask(
    signal: npt.NDArray[np.float64],
    low_thr: npt.NDArray[np.float64] | float,
    high_thr: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.bool_]:
    """Schmitt-trigger threshold for "contact = LOW signal".

    A frame is *weak* (maybe-contact) when ``signal < high_thr`` and *strong*
    (definitely-contact) when ``signal < low_thr`` (``low_thr <= high_thr``,
    since being below the lower threshold is stronger evidence of contact). A
    weak run is kept only if it contains at least one strong frame — so an
    isolated dip below ``high_thr`` that never reaches ``low_thr`` is rejected,
    killing boundary flicker. Fully vectorized per column, reusing the
    run-machinery of :func:`_run_extents`. ``low_thr == high_thr`` reduces
    to ``signal < high_thr``. Thresholds may be scalar or per-foot ``(nf,)``.
    """
    weak = signal < high_thr
    strong = signal < low_thr
    F, M = signal.shape
    starts, ends = _run_extents(weak)
    s = np.clip(starts, 0, F)  # run start per row (F, M)
    e = np.clip(ends, 0, F)  # run end per row (F, M)
    strong_cum = np.concatenate(
        [np.zeros((1, M), dtype=np.int64), np.cumsum(strong.astype(np.int64), axis=0)], axis=0
    )  # (F+1, M)
    strong_in_run = np.take_along_axis(strong_cum, e, axis=0) - np.take_along_axis(
        strong_cum, s, axis=0
    )
    return weak & (strong_in_run > 0)


def _release_open_runs(
    mask: npt.NDArray[np.bool_],
    raw: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Trim hysteresis-filled contact runs that reach a clip boundary back to
    their raw single-threshold support on the open side.

    A contact run touching frame 0 or the last frame is *open*: what the foot
    does past the clip edge is unknown. Hysteresis fills the weak band around a
    run's strong frames, but an open run never crosses back out of the band, so
    the fill extrapolates contact all the way to the edge — a foot cut off
    mid-toe-off stays labelled "planted" to the last frame. The detector is
    offline (the whole clip is in hand), so that extrapolation is unjustified:
    on the open side, contact must not extend past the last frame the raw
    threshold (``signal < thr``) supports. Interior runs — closed on both ends,
    where the band genuinely re-opens — are left untouched.

    ``raw`` is the single-threshold mask ``signal < thr``. Every filled run
    contains a strong frame and every strong frame is raw (``low_thr < thr``),
    so an open run always has raw support to trim back to.
    """
    F = mask.shape[0]
    out = mask.copy()
    for j in range(mask.shape[1]):
        runs = _true_runs(out[:, j])
        if not runs:
            continue
        s, e = runs[0]
        if s == 0:  # left-open: drop leading band
            support = np.flatnonzero(raw[s:e, j])
            out[s : s + support[0], j] = False
        s, e = runs[-1]
        if e == F:  # right-open: drop trailing band
            support = np.flatnonzero(raw[s:e, j])
            out[s + support[-1] + 1 : e, j] = False
    return out


def _otsu_threshold(
    values: npt.NDArray[np.float64],
    *,
    nbins: int = 64,
    valley_ratio: float = 0.6,
    mass_min: float = 0.1,
) -> tuple[float | None, float]:
    """Otsu bimodal threshold of a 1-D sample, gated by a valley-depth test.

    Returns ``(threshold, strength)`` — ``threshold`` maximizes between-class
    variance (centred in any flat valley) and ``strength`` is Otsu's η
    (between/total variance). Returns ``(None, strength)`` — caller should fall
    back to a fixed threshold — when the distribution is not convincingly
    bimodal: the histogram density at the split is not clearly below both
    surrounding peaks (``> valley_ratio × min(peak)``), either side carries less
    than ``mass_min`` of the mass, fewer than 8 samples, or zero variance.
    η alone cannot tell bimodal from unimodal (a Gaussian split at its mean has
    η ≈ 0.6), so the *valley depth* is the real discriminator.
    """
    v = values[np.isfinite(values)]
    if v.size < 8:
        return None, 0.0
    vmin, vmax = float(v.min()), float(v.max())
    total_var = float(v.var())
    if vmax <= vmin or total_var <= 0:
        return None, 0.0
    # A sample whose whole range spans only a few float ULPs is constant for
    # our purposes; np.histogram (>= 2.2) raises when the bin edges collapse.
    edges = np.linspace(vmin, vmax, nbins + 1)
    if np.any(edges[:-1] >= edges[1:]):
        return None, 0.0
    hist, edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    cum_mean = np.cumsum(p * centers)
    mu_t = cum_mean[-1]
    valid = (w0 > 0) & (w1 > 0)
    sigma_b2 = np.zeros_like(p)
    sigma_b2[valid] = (mu_t * w0[valid] - cum_mean[valid]) ** 2 / (w0[valid] * w1[valid])
    # Centre the split in any flat plateau (a wide empty valley makes σ_b²
    # constant across it; argmax alone would snap to its left edge).
    plateau = np.flatnonzero(sigma_b2 >= sigma_b2.max() - 1e-12)
    k = int(plateau[len(plateau) // 2])
    if not valid[k]:
        return None, 0.0
    thr = float(centers[k])
    eta = float(sigma_b2[k] / total_var)
    # Smooth the histogram before the valley test — raw bin noise in a unimodal
    # distribution would otherwise fake a dip next to the mode.
    sm = np.convolve(hist.astype(np.float64), np.ones(5) / 5.0, mode="same")
    left_peak = float(sm[: k + 1].max())
    right_peak = float(sm[k:].max())
    valley = float(sm[max(0, k - 1) : k + 2].min())
    peak = min(left_peak, right_peak)
    valley_ok = peak > 0 and valley <= valley_ratio * peak
    mass_ok = (w0[k] >= mass_min) and (w1[k] >= mass_min)
    if not valley_ok or not mass_ok:
        return None, eta
    return thr, eta


def _resolve_adaptive(
    signal: npt.NDArray[np.float64],
    base: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-foot adaptive thresholds (Otsu) with fallback to ``base`` and clamp.

    Returns ``(thresholds (nf,), used (nf,) bool)``. A foot whose signal is not
    convincingly bimodal falls back to ``base`` (``used=False``); accepted
    thresholds are clamped to ``[0.25*base, 4*base]`` to bound the blast radius.
    """
    nf = signal.shape[1]
    thr = np.full(nf, float(base))
    used = np.zeros(nf, dtype=bool)
    lo, hi = 0.25 * base, 4.0 * base
    for i in range(nf):
        t, _ = _otsu_threshold(signal[:, i])
        if t is not None:
            thr[i] = float(np.clip(t, lo, hi))
            used[i] = True
    return thr, used


def _velocity_informed_height(
    clearance: npt.NDArray[np.float64],  # (F, nf), up-positive height above floor
    speed: npt.NDArray[np.float64],  # (F, nf), per-frame foot speed
    vel_threshold: float | npt.NDArray[np.float64],  # scalar or (nf,) — the stance velocity gate
    margin: float,  # 0.013*scale, the fixed default reused as a margin
) -> npt.NDArray[np.float64]:
    """Per-foot height threshold calibrated to each foot's own stance level.

    Retargeted mocap leaves the foot hovering a clip-dependent amount above the
    floor, so a fixed height threshold misses real stance. Here the threshold is
    pinned to where each foot actually sits when it is *slow* (a stance
    candidate). Per foot ``i``::

        slow_i      = speed_i < vel_threshold
        contact_h   = median(clearance_i[slow_i])      # this foot's stance level
        swing_high  = p90(clearance_i)                 # swing apex
        thr_i = contact_h + margin   if swing_high > contact_h + 2*margin
              = margin               otherwise          # guard

    The *additive* margin makes ``thr_i`` reduce to ``margin`` exactly when the
    foot reaches the floor (``contact_h ≈ 0``) — identical to the old fixed
    behaviour. The guard requires a clear swing above the contact level, so a
    slow-but-airborne *held* foot (no swing) falls back to the fixed threshold
    and is correctly rejected. Returns ``(nf,)``.
    """
    nf = clearance.shape[1]
    thr = np.full(nf, float(margin))
    vt = np.broadcast_to(np.atleast_1d(np.asarray(vel_threshold, dtype=np.float64)), (nf,))
    for i in range(nf):
        slow = speed[:, i] < vt[i]
        if not slow.any():
            continue  # no stance candidate -> fixed
        contact_h = float(np.median(clearance[slow, i]))
        swing_high = float(np.percentile(clearance[:, i], 90))
        if swing_high > contact_h + 2.0 * margin:  # a clear swing exists
            thr[i] = contact_h + margin
    return thr


def _contact_confidence(
    speed: npt.NDArray[np.float64] | None,
    clearance: npt.NDArray[np.float64] | None,
    vel_mask: npt.NDArray[np.bool_] | None,
    height_mask: npt.NDArray[np.bool_] | None,
    mask: npt.NDArray[np.bool_],
    method: str,
    vel_threshold: float | npt.NDArray[np.float64] | None,
    height_threshold: float | npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    """Per-foot detection confidence in ``[0, 1]`` (decisiveness, not probability).

    ``margin`` = mean over contact frames of how far each active signal sits
    below its threshold; ``agreement`` (combined only) = fraction of frames the
    velocity and height masks concur. ``confidence = sqrt(margin*agreement)``
    for combined, ``margin`` otherwise. Zero when a foot never contacts.
    """
    margins = []
    if vel_mask is not None:
        assert speed is not None and vel_threshold is not None
        margins.append(np.clip((vel_threshold - speed) / vel_threshold, 0.0, 1.0))
    if height_mask is not None:
        assert clearance is not None and height_threshold is not None
        margins.append(np.clip((height_threshold - clearance) / height_threshold, 0.0, 1.0))
    margin_frame = np.mean(margins, axis=0) if margins else np.zeros_like(mask, dtype=np.float64)
    contact_count = mask.sum(axis=0)
    safe = np.where(contact_count > 0, contact_count, 1)
    margin = np.where(contact_count > 0, (margin_frame * mask).sum(axis=0) / safe, 0.0)
    if method == "combined" and vel_mask is not None and height_mask is not None:
        agreement = np.mean(vel_mask == height_mask, axis=0)
        return np.sqrt(margin * agreement)
    return margin


def _detect_contacts(
    speed: npt.NDArray[np.float64] | None,
    clearance: npt.NDArray[np.float64] | None,
    *,
    method: str,
    vel_threshold: float | npt.NDArray[np.float64] | None,
    height_threshold: float | npt.NDArray[np.float64] | None,
    hysteresis: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Pure contact-detection core on precomputed ``(F, nf)`` signals.

    Returns ``(mask, confidence)`` — the raw (pre-morphology) contact mask and a
    per-foot confidence. Thresholds may be scalar or per-foot ``(nf,)``.
    """

    def thresholded(
        sig: npt.NDArray[np.float64] | None, thr: float | npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.bool_]:
        # The method reads this signal, so the caller computed it and its threshold.
        assert sig is not None and thr is not None
        if hysteresis and hysteresis > 0:
            mask = _hysteresis_mask(sig, thr * (1.0 - hysteresis), thr * (1.0 + hysteresis))
            # Don't let the band latch extrapolate a contact across an open clip
            # edge (e.g. a foot cut off mid-swing held "planted" to the last frame).
            return _release_open_runs(mask, sig < thr)
        return sig < thr

    if method == "velocity":
        vel_mask = thresholded(speed, vel_threshold)
        height_mask = None
        mask = vel_mask
    elif method == "height":
        vel_mask = None
        height_mask = thresholded(clearance, height_threshold)
        mask = height_mask
    else:
        vel_mask = thresholded(speed, vel_threshold)
        height_mask = thresholded(clearance, height_threshold)
        mask = vel_mask & height_mask

    confidence = _contact_confidence(
        speed, clearance, vel_mask, height_mask, mask, method, vel_threshold, height_threshold
    )
    return mask, confidence


def _contact_diagnostics(
    mask: npt.NDArray[np.bool_],
    joint_positions: npt.NDArray[np.float64],
    clearance: npt.NDArray[np.float64] | None,
    up_idx: int,
    scale: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64] | None]:
    """Unsupervised detection-quality diagnostics (no ground truth needed).

    Returns ``(foot_skate_mean, foot_skate_max, airborne_fraction, height_at_contact)``, the :class:`ContactReport` fields of those names:

    - the foot skate: horizontal drift of each joint within its detected contact
      runs, normalized by skeleton scale (a planted joint should be ~0), its
      mean and its largest over the runs;
    - ``airborne_fraction``: fraction of frames with no joint in contact;
    - ``height_at_contact``: mean clearance over each joint's contact frames,
      ``None`` without a ``clearance`` (the velocity method measures none).
    """
    horiz_axes = [a for a in range(3) if a != up_idx]
    horiz = joint_positions[:, :, horiz_axes]  # (F, nf, 2)
    nf = mask.shape[1]
    skate_mean = np.zeros(nf)
    skate_max = np.zeros(nf)
    for i in range(nf):
        drifts = []
        for s, e in _true_runs(mask[:, i]):
            if e - s >= 2:
                p = horiz[s:e, i, :]
                drifts.append(float(np.linalg.norm(p - p[0], axis=1).max()))
        if drifts:
            skate_mean[i] = np.mean(drifts) / scale
            skate_max[i] = np.max(drifts) / scale
    airborne = float(np.mean(mask.sum(axis=1) == 0))
    if clearance is None:
        return skate_mean, skate_max, airborne, None
    cc = mask.sum(axis=0)
    height_at_contact = np.where(
        cc > 0, (np.where(mask, clearance, 0.0)).sum(axis=0) / np.where(cc > 0, cc, 1), np.nan
    )
    return skate_mean, skate_max, airborne, height_at_contact
