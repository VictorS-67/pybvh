"""Contact detection on plain arrays.

The detector behind :func:`pybvh.analysis.foot_contacts` and :func:`pybvh.analysis.ground_contacts`: the tested joints' positions in, a per-frame contact label per joint out. It takes the joints' positions, their rest positions and the root's (the skeleton scale of the default thresholds), the frame time, the up axis and a :class:`_ContactSettings`, and imports no :class:`~pybvh.bvh.Bvh`: the two detectors read those from the clip, resolve the joints, run the checks whose messages name the clip (its frame time, its ``world_up``) and delegate here. Below the entry point are the kernels it calls: the speed and the clearance signals, the threshold resolution (the skeleton scale, Otsu's split, the stance calibration), the thresholding with hysteresis, the run filters, the confidence and the diagnostics.

The contact reference is estimated here, over the tested joints, with :func:`pybvh.geometry._floor_level`; the scene ground (``Bvh.floor_height``) is never read (ADR 0002).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeVar, cast, get_args

import numpy as np
import numpy.typing as npt

from . import geometry
from .signal import box_filter_smooth

# The choices of the settings that name one, as the entry points document them.
_Method = Literal["combined", "velocity", "height"]
_HeightReference = Literal["velocity", "floor"]
_FloorEstimate = Literal["auto", "min"]

# The defaults of the settings that some method does not read: the entry
# points' keywords default to None, so that a setting given is told from one
# left out, and _ContactSettings puts these in place.
_DEFAULT_VEL_SMOOTH_DURATION = 1.0 / 30.0
_DEFAULT_FLOOR: _FloorEstimate = "auto"

# The default thresholds, as fractions of the skeleton scale: the speed, per
# second, and the clearance margin above the contact reference.
_VEL_THRESHOLD_FRACTION = 0.12
_HEIGHT_MARGIN_FRACTION = 0.013
# Otsu's split is kept within these multiples of the default it replaces.
_SPLIT_LOWER_BOUND, _SPLIT_UPPER_BOUND = 0.25, 4.0
# The percentile of the per-frame lowest joint the contact reference is read
# at, by floor estimate.
_FLOOR_PERCENTILE: dict[_FloorEstimate, float] = {"auto": 2.0, "min": 0.0}

_METHODS_READING_VELOCITY = ("velocity", "combined")
_METHODS_READING_HEIGHT = ("height", "combined")

# A threshold as a caller gives it: one value for every joint, or one per joint.
_ThresholdArg = float | Sequence[float] | npt.NDArray[np.float64]

# Where a joint's threshold came from, as the report names it.
_SplitSource = Literal["otsu", "otsu_clamped"]
_VelocitySource = Literal["given", "scale", "otsu", "otsu_clamped"]
_HeightSource = Literal["given", "margin", "stance", "otsu", "otsu_clamped"]
_Fallback = TypeVar("_Fallback", Literal["scale"], Literal["margin"])


class _ContactSettings:
    """The settings of one contact detection, checked once at the detector's entry point.

    :func:`pybvh.analysis.foot_contacts` and :func:`pybvh.analysis.ground_contacts` declare these settings as keywords and document them; they pass them on here, and what lies below the entry points receives this object instead of the keywords. Construction checks the call and puts the defaults in place: each attribute is the entry points' parameter of the same name with its default resolved, and it is :data:`~typing.Final`, so nothing below the entry points rebinds it.

    ``default_height_reference`` is the detector's own default for ``height_reference``: ``"velocity"`` for :func:`pybvh.analysis.foot_contacts`, ``"floor"`` for :func:`pybvh.analysis.ground_contacts`. ``n_joints`` is the number of joints tested, the length a threshold given per joint must have; a threshold given is kept as a float or as a read-only ``(n_joints,)`` array.
    """

    def __init__(
        self,
        *,
        method: str,
        vel_threshold: _ThresholdArg | None,
        vel_smooth_duration: float | None,
        height_threshold: _ThresholdArg | None,
        floor: float | str | None,
        min_contact_duration: float,
        min_gap_duration: float,
        hysteresis: float,
        adaptive: bool,
        height_reference: str | None,
        default_height_reference: _HeightReference,
        n_joints: int,
    ) -> None:
        self.method: Final = _checked_method(method)
        _refuse_unread_settings(
            self.method,
            reads_velocity=self.reads_velocity,
            reads_height=self.reads_height,
            vel_threshold=vel_threshold,
            vel_smooth_duration=vel_smooth_duration,
            height_threshold=height_threshold,
            floor=floor,
            height_reference=height_reference,
        )
        self.height_reference: Final = _checked_height_reference(
            default_height_reference if height_reference is None else height_reference
        )
        self.floor: Final = _checked_floor(_DEFAULT_FLOOR if floor is None else floor)
        self.vel_smooth_duration: Final = (
            _DEFAULT_VEL_SMOOTH_DURATION if vel_smooth_duration is None else vel_smooth_duration
        )
        self.min_contact_duration: Final = min_contact_duration
        self.min_gap_duration: Final = min_gap_duration
        _check_durations(
            vel_smooth_duration=self.vel_smooth_duration,
            min_contact_duration=min_contact_duration,
            min_gap_duration=min_gap_duration,
        )
        if not 0 <= hysteresis < 1:
            raise ValueError(f"hysteresis must be in [0, 1), got {hysteresis}")
        self.hysteresis: Final = hysteresis
        self.vel_threshold: Final = _checked_threshold("vel_threshold", vel_threshold, n_joints)
        self.height_threshold: Final = _checked_threshold(
            "height_threshold", height_threshold, n_joints
        )
        if adaptive:
            self._check_adaptive_derives_a_threshold()
        self.adaptive: Final = adaptive

    @property
    def reads_velocity(self) -> bool:
        """Whether the method tests the joints' speed."""
        return self.method in _METHODS_READING_VELOCITY

    @property
    def reads_height(self) -> bool:
        """Whether the method tests the joints' clearance above the contact reference."""
        return self.method in _METHODS_READING_HEIGHT

    def _check_adaptive_derives_a_threshold(self) -> None:
        """Refuse ``adaptive=True`` where every threshold the method reads is given or stance-calibrated."""
        # Under combined with height_reference="velocity" the stance
        # calibration, not Otsu's split, derives the height threshold.
        stance_calibrated = self.method == "combined" and self.height_reference == "velocity"
        splits_speed = self.reads_velocity and self.vel_threshold is None
        splits_height = (
            self.reads_height and self.height_threshold is None and not stance_calibrated
        )
        if splits_speed or splits_height:
            return
        reasons = []
        if self.reads_velocity:
            reasons.append("vel_threshold is given")
        if self.reads_height:
            reasons.append(
                "height_threshold is given"
                if self.height_threshold is not None
                else "the height threshold is calibrated on the stance frames "
                "(height_reference='velocity'), not split"
            )
        raise ValueError(
            f"adaptive=True derives no threshold under method={self.method!r}: "
            f"{' and '.join(reasons)}; drop adaptive=True."
        )


def _checked_method(method: str) -> _Method:
    """``method``, refused unless it is one of the detectors' methods."""
    if method not in get_args(_Method):
        raise ValueError(f"Unknown method {method!r}. Choose 'combined', 'velocity', or 'height'.")
    return cast(_Method, method)


def _checked_height_reference(height_reference: str) -> _HeightReference:
    """``height_reference``, refused unless it is ``"velocity"`` or ``"floor"``."""
    if height_reference not in get_args(_HeightReference):
        raise ValueError(
            f"height_reference must be 'velocity' or 'floor', got {height_reference!r}"
        )
    return cast(_HeightReference, height_reference)


def _checked_floor(floor: float | str) -> float | _FloorEstimate:
    """``floor``, refused unless it is ``"auto"``, ``"min"`` or a finite float."""
    if isinstance(floor, str):
        if floor in get_args(_FloorEstimate):
            return cast(_FloorEstimate, floor)
    elif np.isfinite(floor):
        return floor
    raise ValueError(f"floor must be 'auto', 'min' or a finite float, got {floor!r}")


def _refuse_unread_settings(
    method: _Method,
    *,
    reads_velocity: bool,
    reads_height: bool,
    vel_threshold: _ThresholdArg | None,
    vel_smooth_duration: float | None,
    height_threshold: _ThresholdArg | None,
    floor: float | str | None,
    height_reference: str | None,
) -> None:
    """Refuse a setting given that ``method`` does not read: its default ``None`` tells it left out."""
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


def _check_durations(**durations: float) -> None:
    """Refuse a duration (seconds) not finite or below zero; each keyword names its setting."""
    for name, duration in durations.items():
        if not (np.isfinite(duration) and duration >= 0):
            raise ValueError(f"{name} must be finite and >= 0 (seconds), got {duration}")


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
    adaptive : bool
        Whether Otsu's split was asked for (``adaptive=True``); the threshold sources say where it stood.
    floor_setting : {"auto", "min"} or float, or None
        The ``floor`` setting applied: ``"auto"`` (the default) or ``"min"``, the estimate the contact reference ``floor`` came from, or the level given. ``None`` under ``method="velocity"``.
    skeleton_scale : float
        The scale the default thresholds are written in and the foot skate is divided by: the mean rest distance from the root to the tested joints. Reported whether or not a threshold was derived from it, since the diagnostics always use it.
    skeleton_scale_measured : bool
        Whether ``skeleton_scale`` was measured: ``False`` where every tested joint sits on the root, which leaves no distance to measure, and ``1.0`` stands in (``ground_contacts(["Hips"])`` on a clip rooted at the hips).
    vel_threshold : ndarray, shape (n_joints,), or None
        The speed threshold applied to each joint, in units per second; ``None`` under ``method="height"``.
    vel_threshold_source : tuple of str, or None
        Where each joint's ``vel_threshold`` came from: ``"given"`` (the caller's), ``"scale"`` (the default, ``0.12 × skeleton_scale``), ``"otsu"`` (``adaptive=True``: Otsu's split of the joint's speed), ``"otsu_clamped"`` (the split fell outside ``[0.25, 4]`` × the default, and that bound stands in). A split rejected, the speed not being bimodal, reads ``"scale"``. ``None`` where ``vel_threshold`` is, and on a one-frame clip, which compares no speed against it.
    speed : ndarray, shape (F, n_joints), or None
        Each joint's speed, in units per second, the signal compared against ``vel_threshold`` (frame 0 repeats frame 1). ``None`` under ``method="height"``, and on a one-frame clip, which has no speed to measure: there the speed test passes every frame, so ``method="combined"`` labels by the height test alone.
    height_threshold : ndarray, shape (n_joints,), or None
        The clearance threshold applied to each joint, in world units; ``None`` under ``method="velocity"``.
    height_threshold_source : tuple of str, or None
        Where each joint's ``height_threshold`` came from: ``"given"``, ``"margin"`` (the default, ``0.013 × skeleton_scale``), ``"stance"`` (``height_reference="velocity"``: the joint's clearance over its slow frames, plus the margin), ``"otsu"`` and ``"otsu_clamped"`` (``adaptive=True`` where the height threshold is split, the bounds as for the speed). A split rejected, or a stance calibration that finds no swing above the stance level, reads ``"margin"``. ``None`` where ``height_threshold`` is.
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
    adaptive: bool
    floor_setting: float | _FloorEstimate | None
    skeleton_scale: float
    skeleton_scale_measured: bool
    vel_threshold: npt.NDArray[np.float64] | None
    vel_threshold_source: tuple[_VelocitySource, ...] | None
    speed: npt.NDArray[np.float64] | None
    height_threshold: npt.NDArray[np.float64] | None
    height_threshold_source: tuple[_HeightSource, ...] | None
    floor: float | None
    clearance: npt.NDArray[np.float64] | None
    confidence: npt.NDArray[np.float64]
    foot_skate_mean: npt.NDArray[np.float64]
    foot_skate_max: npt.NDArray[np.float64]
    airborne_fraction: float
    height_at_contact: npt.NDArray[np.float64] | None


def _checked_threshold(
    name: str, threshold: _ThresholdArg | None, n_joints: int
) -> float | npt.NDArray[np.float64] | None:
    """A threshold given, checked: a float, a read-only ``(n_joints,)`` array, or ``None`` when not given."""
    if threshold is None:
        return None
    values = np.array(threshold, dtype=np.float64)
    if values.ndim != 0 and values.shape != (n_joints,):
        raise ValueError(
            f"{name} must be a scalar or one value per joint ({n_joints},), "
            f"got shape {values.shape}"
        )
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"{name} must be finite and > 0, got {threshold}")
    if values.ndim == 0:
        return float(values)
    values.flags.writeable = False  # np.array copied it: the caller's array stays theirs
    return values


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
    n_joints = len(joint_names)
    F = joint_positions.shape[0]

    # The scale the default thresholds are written in and the report's foot
    # skate is divided by.
    scale, scale_measured = _skeleton_scale(joint_rest, root_rest)

    # ---- Signals (F, n_joints), threshold resolution ----
    speed = None
    clearance = None
    # The thresholds in use: the caller's, or a per-joint (n_joints,) array resolved here.
    vel_thr_used: float | npt.NDArray[np.float64] | None = None
    height_thr_used: float | npt.NDArray[np.float64] | None = None
    floor_raw = None
    vel_source: tuple[_VelocitySource, ...] | None = None
    height_source: tuple[_HeightSource, ...] | None = None
    vel_smooth_frames = None

    if settings.reads_velocity:
        if F < 2:
            # No motion info — treat as "no velocity evidence against contact"
            # (speed 0 < any positive threshold) so combined falls back to height.
            speed = np.zeros((F, n_joints))
        else:
            disp = joint_positions[1:] - joint_positions[:-1]  # (F-1, n_joints, 3)
            # Condition the speed estimator over a fixed physical time span
            # so detection does not depend on the capture rate: box-averaging
            # the displacement *vectors* (norm-of-mean) cancels the
            # high-frequency jitter that adjacent-frame differencing picks up
            # at high fps.  Window is 1 at <= 30 fps — the raw signal, exactly.
            vel_smooth_frames = min(F - 1, max(1, round(settings.vel_smooth_duration / frame_time)))
            if vel_smooth_frames > 1:
                disp = box_filter_smooth(disp, vel_smooth_frames, axis=0)
            sp = np.linalg.norm(disp, axis=-1) / frame_time  # (F-1, n_joints), u/s
            speed = np.concatenate([sp[0:1], sp], axis=0)  # frame-0 propagated
        if settings.vel_threshold is not None:
            vel_thr_used = settings.vel_threshold
            vel_source = _each_joint("given", n_joints)
        else:
            base = _VEL_THRESHOLD_FRACTION * scale  # of the rest distance, per second
            if settings.adaptive and F >= 2:
                vel_thr_used, vel_source = _resolve_adaptive(speed, base, "scale")
            else:
                vel_thr_used = base
                vel_source = _each_joint("scale", n_joints)
        if F < 2:
            vel_source = None  # no speed was measured, so no threshold was compared against one

    if settings.reads_height:
        heights_signed = joint_positions[:, :, up_idx] * up_sign  # up-positive
        if isinstance(settings.floor, str):
            # "auto"/"min": the reference is estimated from the coords in
            # use, over the joints being tested — never from the scene
            # ground (Bvh.floor_height), which is measured over ALL nodes
            # and would sit an end site's length below these joints.
            floor_raw = geometry._floor_level(
                joint_positions, up_idx, up_sign, percentile=_FLOOR_PERCENTILE[settings.floor]
            )
        else:
            floor_raw = float(settings.floor)
        floor_signed = floor_raw * up_sign
        clearance = heights_signed - floor_signed
        if settings.height_threshold is not None:
            height_thr_used = settings.height_threshold
            height_source = _each_joint("given", n_joints)
        else:
            base = _HEIGHT_MARGIN_FRACTION * scale  # of the rest distance, above the floor
            # A speed signal exists exactly when method is "combined" here.
            if speed is not None and settings.height_reference == "velocity":
                # Calibrate the height threshold per joint to its own stance
                # level (handles retargeting hover); reduces to `base` on rigs
                # where the foot reaches the floor.
                assert vel_thr_used is not None
                height_thr_used, height_source = _velocity_informed_height(
                    clearance, speed, vel_thr_used, base
                )
            elif settings.adaptive:
                height_thr_used, height_source = _resolve_adaptive(clearance, base, "margin")
            else:
                height_thr_used = base
                height_source = _each_joint("margin", n_joints)

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

    skate_mean, skate_max, airborne, height_at_contact = _contact_diagnostics(
        mask, joint_positions, clearance, up_idx, scale
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
        adaptive=settings.adaptive,
        floor_setting=settings.floor if settings.reads_height else None,
        skeleton_scale=float(scale),
        skeleton_scale_measured=scale_measured,
        vel_threshold=None if vel_thr_used is None else _per_joint(vel_thr_used, n_joints),
        vel_threshold_source=vel_source,
        # The zeros a one-frame clip is tested with are no measured speed.
        speed=_optional_read_only(speed if F >= 2 else None),
        height_threshold=None if height_thr_used is None else _per_joint(height_thr_used, n_joints),
        height_threshold_source=height_source,
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
_S = TypeVar("_S", bound=str)


def _each_joint(source: _S, n_joints: int) -> tuple[_S, ...]:
    """One threshold source for every joint, as the report's per-joint tuple."""
    return (source,) * n_joints


def _per_joint(
    threshold: float | npt.NDArray[np.float64], n_joints: int
) -> npt.NDArray[np.float64]:
    """A threshold applied, scalar or per joint, as the report's read-only ``(n_joints,)``."""
    return _read_only(np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n_joints,)))


def _read_only(array: npt.NDArray[_A]) -> npt.NDArray[_A]:
    """A read-only copy of ``array``, for a field of the frozen :class:`ContactReport`."""
    frozen = np.array(array)
    frozen.flags.writeable = False
    return frozen


def _optional_read_only(array: npt.NDArray[_A] | None) -> npt.NDArray[_A] | None:
    """:func:`_read_only`, passing ``None`` through."""
    return None if array is None else _read_only(array)


def _skeleton_scale(
    joint_rest: npt.NDArray[np.float64],
    root_rest: npt.NDArray[np.float64],
) -> tuple[float, bool]:
    """Scale reference for the contact detectors' default thresholds, and whether it was measured.

    The mean rest-pose distance from the root to the tested joints, :func:`pybvh.analysis.skeleton_size`'s measurement, with ``True``. Where it measures nothing (every joint sits on the root) the scale is ``1.0``, with ``False``, so the thresholds don't collapse to zero: the public ``skeleton_size`` raises instead of fabricating, but a threshold scale must exist even where a measurement does not, and the report says it stood in.
    """
    distance = geometry._mean_distance_to_root(joint_rest, root_rest)
    return (1.0, False) if distance is None else (distance, True)


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
    to ``signal < high_thr``. Thresholds may be scalar or per joint ``(n_joints,)``.
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
    fallback: _Fallback,
) -> tuple[npt.NDArray[np.float64], tuple[_Fallback | _SplitSource, ...]]:
    """Per-joint adaptive thresholds (Otsu) with fallback to ``base`` and clamp.

    Returns ``(thresholds (n_joints,), sources (n_joints,))``, each joint's source as the
    report names it. A joint whose signal is not convincingly bimodal falls back
    to ``base``, its source ``fallback``; an accepted split is clamped to
    ``[0.25*base, 4*base]`` to bound the blast radius: ``"otsu"`` within the
    bounds, ``"otsu_clamped"`` where the bound stands in for it.
    """
    n_joints = signal.shape[1]
    thr = np.full(n_joints, float(base))
    sources: list[_Fallback | _SplitSource] = [fallback] * n_joints
    lo, hi = _SPLIT_LOWER_BOUND * base, _SPLIT_UPPER_BOUND * base
    for i in range(n_joints):
        t, _ = _otsu_threshold(signal[:, i])
        if t is not None:
            thr[i] = float(np.clip(t, lo, hi))
            sources[i] = "otsu" if lo <= t <= hi else "otsu_clamped"
    return thr, tuple(sources)


def _velocity_informed_height(
    clearance: npt.NDArray[np.float64],  # (F, n_joints), up-positive height above floor
    speed: npt.NDArray[np.float64],  # (F, n_joints), per-frame speed
    # Scalar or (n_joints,): the stance velocity gate.
    vel_threshold: float | npt.NDArray[np.float64],
    margin: float,  # 0.013*scale, the fixed default reused as a margin
) -> tuple[npt.NDArray[np.float64], tuple[Literal["stance", "margin"], ...]]:
    """Per-joint height threshold calibrated to each joint's own stance level.

    Retargeted mocap leaves the foot hovering a clip-dependent amount above the
    floor, so a fixed height threshold misses real stance. Here the threshold is
    pinned to where each joint actually sits when it is *slow* (a stance
    candidate). Per joint ``i``::

        slow_i      = speed_i < vel_threshold
        contact_h   = median(clearance_i[slow_i])      # this joint's stance level
        swing_high  = p90(clearance_i)                 # swing apex
        thr_i = contact_h + margin   if swing_high > contact_h + 2*margin
              = margin               otherwise          # guard

    The *additive* margin makes ``thr_i`` reduce to ``margin`` exactly when the
    foot reaches the floor (``contact_h ≈ 0``) — identical to the old fixed
    behaviour. The guard requires a clear swing above the contact level, so a
    slow-but-airborne *held* foot (no swing) falls back to the fixed threshold
    and is correctly rejected. Returns the ``(n_joints,)`` thresholds and each
    joint's source as the report names it: ``"stance"`` where it was
    calibrated, ``"margin"`` where the margin stands in.
    """
    n_joints = clearance.shape[1]
    thr = np.full(n_joints, float(margin))
    sources: list[Literal["stance", "margin"]] = ["margin"] * n_joints
    vt = np.broadcast_to(np.atleast_1d(np.asarray(vel_threshold, dtype=np.float64)), (n_joints,))
    for i in range(n_joints):
        slow = speed[:, i] < vt[i]
        if not slow.any():
            continue  # no stance candidate -> fixed
        contact_h = float(np.median(clearance[slow, i]))
        swing_high = float(np.percentile(clearance[:, i], 90))
        if swing_high > contact_h + 2.0 * margin:  # a clear swing exists
            thr[i] = contact_h + margin
            sources[i] = "stance"
    return thr, tuple(sources)


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
    """Per-joint detection confidence in ``[0, 1]`` (decisiveness, not probability).

    ``margin`` = mean over contact frames of how far each active signal sits
    below its threshold; ``agreement`` (combined only) = fraction of frames the
    velocity and height masks concur. ``confidence = sqrt(margin*agreement)``
    for combined, ``margin`` otherwise. Zero when a joint never contacts.
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
    """Pure contact-detection core on precomputed ``(F, n_joints)`` signals.

    Returns ``(mask, confidence)`` — the raw (pre-morphology) contact mask and a
    per-joint confidence. Thresholds may be scalar or per joint ``(n_joints,)``.
    """

    def thresholded(
        sig: npt.NDArray[np.float64] | None, thr: float | npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.bool_]:
        # The method reads this signal, so the caller computed it and its threshold.
        assert sig is not None and thr is not None
        if hysteresis > 0:
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
    horiz = joint_positions[:, :, horiz_axes]  # (F, n_joints, 2)
    n_joints = mask.shape[1]
    skate_mean = np.zeros(n_joints)
    skate_max = np.zeros(n_joints)
    for i in range(n_joints):
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
