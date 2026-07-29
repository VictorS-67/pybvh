from __future__ import annotations

import re
import warnings
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .rotations import _elementary_rotmat

if TYPE_CHECKING:
    from .bvh import Bvh



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
    if filepath.suffix != '.bvh':
        raise ValueError(f'{filepath} is not a bvh file')
    elif not filepath.exists():
        raise FileNotFoundError(f'could not find the file {filepath}')
    return filepath

#--------------------------------------------------------------------------------------------

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
    if ax[0] == '+':
        return True
    elif ax[0] == '-':
        return False
    else:
        raise ValueError("The sign of the axis should be either '+' or '-'.")


# ===========================================================================
# ORIENTATION HELPERS (internal)
# ===========================================================================
#
# This block replaces the old monolithic `get_forw_up_axis()` function with
# a set of single-purpose helpers. They cover three distinct concepts that
# the old function conflated:
#
#   1. WORLD UP — gravity axis of the BVH coordinate system. Constant per
#      file. Auto-detected from the first animation frame with rest-pose
#      topology as fallback.
#   2. TOPOLOGICAL LATERAL/UP — the character's own L/R and vertical axes
#      in rest pose. Used by mirror/retarget; independent of the animation.
#   3. FRAME-SPECIFIC FORWARD — the direction the character is *actually*
#      facing in world space at a given frame, derived from live joint
#      positions. Used for camera placement in visualizations.
#
# Layer diagram (→ = "used by", top = primitives, bottom = public API):
#
#     _axis_to_vector                _validate_axis_string
#           │                               │
#           ├──────────┐                    │
#           ▼          ▼                    ▼
#     rest_pose_   _resolve_          [Bvh.world_up setter]
#      coords      lr_pairs
#           │          │
#           ├──────────┤
#           ▼          ▼
#       _rest_upward, _rest_leftward
#           │          │       │
#           │          │       └──→ used by transforms.mirror()
#           │          ▼
#           │    _world_leftward_units ── the ONLY place the per-frame
#           │      │        │       │     leftward geometry is computed
#           │      │        │       └──→ bvhplot follow-mode azimuths
#           │      │        ▼            (render_mpl/opencv)
#           │      │   _world_leftward_unit_at_frame (1-frame view)
#           │      │        │
#           │      │        ▼
#           │      │   _compute_forward_at, ──→ Bvh.forward_at /
#           │      │   _compute_left_at         Bvh.left_at
#           │      ▼     (axis-string snap)
#           │  _facing_basis ──→ analysis.facing_frame
#           │     (+ _fallback_forward_vector, shared with the
#           ▼      axis-string snappers)
#       _infer_world_up ──→ Bvh.world_up (property)
#
#   (_signed_rotation_delta_around_axis is the scalar angle helper the
#    follow camera's vectorized delta math mirrors.)
#
# Public surface: `Axis` / `parse_axis` (re-exported as `pybvh.Axis` /
# `pybvh.parse_axis`), `Bvh.world_up` (property + setter), the parsed
# `Bvh.up_axis` / `Bvh.forward_axis` / `Bvh.rest_up_axis` properties,
# `Bvh.forward_at(frame=0)` / `Bvh.left_at(frame=0)` (methods), and — via
# `analysis.facing_frame` — the continuous vector basis. Everything else is
# underscore-prefixed and intended for internal use by pybvh modules.
# ---------------------------------------------------------------------------

_VALID_AXIS_STRINGS = frozenset(
    {'+x', '-x', '+y', '-y', '+z', '-z'})

_AXIS_CHAR_TO_IDX = {'x': 0, 'y': 1, 'z': 2}

Axis = namedtuple("Axis", ["index", "sign", "vector"])
Axis.__doc__ = """A signed coordinate axis parsed into machine-usable form.

The parsed counterpart of pybvh's signed axis strings (``'+y'``, ``'-z'``), produced by :func:`pybvh.parse_axis` and returned by :attr:`Bvh.up_axis`, :attr:`Bvh.forward_axis` and :attr:`Bvh.rest_up_axis`.

Fields: ``index`` (int, 0/1/2 — the axis's column in any ``(..., 3)`` coordinate array), ``sign`` (float, ``+1.0`` / ``-1.0`` — multiply by it to make the coordinate positive along the axis direction), ``vector`` (float64 ``(3,)`` unit vector along the axis, sign included).

A plain named tuple: construct one directly for a synthetic skeleton or a test, unpack it as ``index, sign, vector = axis``, or read the fields by name.

Comparing two ``Axis`` values with ``==`` raises, because tuple equality reaches the ``vector`` field and NumPy will not reduce an array to a single truth value. Compare ``(index, sign)`` instead — those two fields determine the axis completely, and ``vector`` is derived from them.
"""

# Arbitrary-but-stable horizontal forward per up axis, used when a
# skeleton carries no L/R orientation information at all.
_FALLBACK_FORWARD = {'y': '+z', 'z': '+x', 'x': '+y'}


def _axis_to_vector(ax: str) -> npt.NDArray[np.float64]:
    """Convert a signed axis string ('+y') to a unit vector ([0, 1, 0])."""
    vec = np.zeros(3)
    vec[_AXIS_CHAR_TO_IDX[ax[1]]] = 1.0 if ax[0] == '+' else -1.0
    return vec


def _axis_index_sign(ax: str) -> tuple[int, float]:
    """Split a signed axis string ('-z') into ``(coordinate index, sign)``.

    The scalar companion of :func:`_axis_to_vector`: ``index`` selects the coordinate column, ``sign`` (``+1.0`` / ``-1.0``) restores the direction, so ``coords[..., index] * sign`` is the up-positive height. The single parsing implementation behind :attr:`Bvh.up_axis` and the internal up-axis consumers.
    """
    return _AXIS_CHAR_TO_IDX[ax[1]], 1.0 if ax[0] == '+' else -1.0


def parse_axis(axis: str, *, allow_unsigned: bool = False) -> Axis:
    """Parse a signed axis string into an :class:`Axis`.

    The public form of the parser behind :attr:`Bvh.up_axis` and friends,
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
        irrelevant, as for ``mirror``'s ``lateral_axis``.

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


def _rest_upward(bvh: Bvh) -> str | None:
    """Infer the skeleton's topological up axis from the rest pose.

    Uses the same heuristic the old get_forw_up_axis used: iterate named
    body parts ("head", "neck", "chest", "spine") in priority order and
    check which axis their rest-pose cumulative offset dominates. Falls
    back to the axis with the largest spread across all joints.

    Parameters
    ----------
    bvh : Bvh
        The skeleton.

    Returns
    -------
    str or None
        Signed axis string (e.g. '+z'), or ``None`` when the rest pose
        carries no directional information (single-node skeletons or
        all-zero offsets).
    """
    if len(bvh.nodes) < 2:
        return None

    rest = bvh.rest_pose_positions()  # (N, 3)
    local_coord = rest - rest[0]  # root at origin

    up_body_parts = ["head", "neck", "chest", "spine"]
    for part_name in up_body_parts:
        for joint in bvh.nodes:
            if joint.name.lower() == part_name:
                coord = local_coord[bvh.node_index[joint.name]]
                direction = get_main_direction(coord)
                if direction is not None:
                    return direction

    # Fallback: axis with largest spread across all joints
    spread = np.ptp(local_coord, axis=0)
    if float(np.max(spread)) < 1e-6:
        return None  # degenerate rest pose: all joints coincide
    up_idx_fallback = int(np.argmax(spread))
    # Use the mean to determine sign (positive mean = positive axis direction)
    sign = '+' if np.mean(local_coord[:, up_idx_fallback]) >= 0 else '-'
    return sign + 'xyz'[up_idx_fallback]


# ---------------------------------------------------------------------------
# L/R name-detection heuristics (pure string logic on joint names)
# ---------------------------------------------------------------------------

_NUMBER_SUFFIX_RE = re.compile(r'\.\d+$')


def _strip_number_suffix(name: str) -> str:
    """Strip trailing ``.NNN`` suffix like Blender's ``.001`` duplicates."""
    return _NUMBER_SUFFIX_RE.sub('', name)


def _strip_namespace_prefix(name: str) -> tuple[str, str]:
    """Strip a ``foo:`` namespace prefix (Mixamo and similar).

    Returns ``(prefix, base)``. If no namespace, prefix is empty.
    """
    if ':' in name:
        prefix, _, base = name.rpartition(':')
        return prefix + ':', base
    return '', name


_SUFFIX_LR_TABLE = [
    # (left_suffix, right_suffix) — matched against the trailing substring
    # of the base name (after namespace and number-suffix strip). Ordered
    # most-specific-first so e.g. ".Left" / ".Right" wins over ".L" / ".R"
    # if both would parse.
    ('.Left', '.Right'), ('_Left', '_Right'),
    ('.left', '.right'), ('_left', '_right'),
    ('.L', '.R'), ('_L', '_R'),
    ('.l', '.r'), ('_l', '_r'),
]


def _try_suffix_partner(base: str, joint_names: set[str], prefix: str,
                        number_suffix: str) -> str | None:
    """If ``base`` matches a known L/R suffix, return the partner's full name."""
    for left_suf, right_suf in _SUFFIX_LR_TABLE:
        if base.endswith(left_suf):
            partner_base = base[: -len(left_suf)] + right_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
        elif base.endswith(right_suf):
            partner_base = base[: -len(right_suf)] + left_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
    return None


def _detect_lr_mapping_by_names(bvh: Bvh) -> dict[str, str]:
    """Internal: detect L/R joint pairs by extended name heuristics.

    Called at ``Bvh.__init__`` to populate ``bvh.lr_mapping`` eagerly.
    Rules tried most-specific-first so delimited suffixes win over bare
    substring matches:

    1. Delimited suffix — ``arm.L`` / ``arm.R``, ``arm_L`` / ``arm_R``,
       and their lowercase and ``.Left`` / ``_Left`` variants. Names are
       normalized by stripping a ``foo:`` namespace prefix (Mixamo's
       ``mixamorig:``) and a trailing ``.NNN`` number suffix (Blender's
       ``.001`` duplicates) before the suffix match; the partner is
       rebuilt with the same prefix/number suffix.
    2. Substring ``left`` / ``right`` (case-insensitive) — handles
       ``LeftArm`` / ``RightArm``, ``leftArm`` / ``rightArm``, and bare
       cases like ``LeftEye`` / ``RightEye`` with no delimiter. Namespace
       prefix is stripped before matching.
    3. Prefix ``L*`` / ``R*`` followed by an uppercase letter — handles
       ``LArm`` / ``RArm``.

    Mutual-match requirement: both halves must exist as joint names.
    Singletons are not paired.

    Returns ``{}`` if no pairs detected. See ``bvh.lr_mapping`` for the
    public API that wraps this function.
    """
    joint_names = set(bvh.joint_names)
    mapping: dict[str, str] = {}
    seen: set[str] = set()

    for name in bvh.joint_names:
        if name in seen:
            continue

        partner: str | None = None

        # Decompose name for suffix rule: strip namespace and number suffix
        ns_prefix, after_ns = _strip_namespace_prefix(name)
        base = _strip_number_suffix(after_ns)
        number_suffix = after_ns[len(base):]  # e.g. '.001' or ''

        # Strategy 1: delimited suffix (most specific)
        partner = _try_suffix_partner(base, joint_names, ns_prefix, number_suffix)

        # Strategy 2: "left"/"right" substring (case-insensitive)
        # Work on the post-namespace base (no number suffix) so Mixamo
        # names like "mixamorig:LeftArm" match against "LeftArm".
        if partner is None:
            lower = base.lower()
            if "left" in lower:
                partner_base = (base
                                .replace("Left", "Right")
                                .replace("left", "right")
                                .replace("LEFT", "RIGHT"))
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate
            elif "right" in lower:
                partner_base = (base
                                .replace("Right", "Left")
                                .replace("right", "left")
                                .replace("RIGHT", "LEFT"))
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate

        # Strategy 3: "L" / "R" prefix followed by uppercase
        if partner is None and len(base) >= 2:
            if base[0] == "L" and base[1].isupper():
                partner_base = "R" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate
            elif base[0] == "R" and base[1].isupper():
                partner_base = "L" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in joint_names:
                    partner = candidate

        if partner is not None and partner not in seen:
            # Normalise: always store the "Left" / "L" version as key
            left, right = _order_lr_pair(name, partner)
            mapping[left] = right
            seen.add(left)
            seen.add(right)

    return mapping


def _order_lr_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(left_name, right_name)``."""
    # Prefer substring form first (most common)
    for kw in ("Left", "left", "LEFT"):
        if kw in a:
            return (a, b)
        if kw in b:
            return (b, a)
    # Normalize for suffix check: strip namespace + number suffix
    _, a_after = _strip_namespace_prefix(a)
    _, b_after = _strip_namespace_prefix(b)
    a_base = _strip_number_suffix(a_after)
    b_base = _strip_number_suffix(b_after)
    for left_suf, _right_suf in _SUFFIX_LR_TABLE:
        if a_base.endswith(left_suf):
            return (a, b)
        if b_base.endswith(left_suf):
            return (b, a)
    # Fallback: "L" prefix → left
    a_prefix_check = _strip_number_suffix(a_after)
    b_prefix_check = _strip_number_suffix(b_after)
    if a_prefix_check.startswith("L") and (len(a_prefix_check) < 2 or a_prefix_check[1].isupper()):
        return (a, b)
    if b_prefix_check.startswith("L") and (len(b_prefix_check) < 2 or b_prefix_check[1].isupper()):
        return (b, a)
    return (a, b)


def _iter_unique_lr_pairs(mapping: dict[str, str]):
    """Yield each ``(name_a, name_b)`` pair from an L/R mapping exactly once.

    Tolerates symmetric mappings (the public form of ``Bvh.lr_mapping``,
    which contains both directions of every pair) without double-counting.
    """
    seen: set[frozenset[str]] = set()
    for a, b in mapping.items():
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        yield a, b


def _resolve_lr_pairs(
    mapping: dict[str, str] | None,
    name2idx: dict[str, int],
    *,
    strict: bool = False,
) -> list[tuple[int, int]]:
    """Resolve an L/R name mapping to ``(left_idx, right_idx)`` pairs.

    The single pair-resolution loop shared by every L/R consumer
    (``mirror``, ``auto_detect_lr_pairs``, the orientation heuristics).
    Symmetric mappings (the public form of ``Bvh.lr_mapping``) are
    deduplicated to one tuple per pair via :func:`_iter_unique_lr_pairs`.

    Parameters
    ----------
    mapping : dict or None
        L/R name mapping.  ``None`` or empty resolves to ``[]``.
    name2idx : dict
        Name → index lookup defining the target index space (e.g.
        ``bvh.joint_index`` or ``bvh.node_index``).
    strict : bool, optional
        If True, raise ``ValueError`` listing every mapped name missing
        from ``name2idx`` (used to surface typos in explicitly passed
        mappings).  Default False: pairs with unknown names are skipped.
    """
    if not mapping:
        return []
    pairs: list[tuple[int, int]] = []
    unknown: set[str] = set()
    for left_name, right_name in _iter_unique_lr_pairs(mapping):
        if left_name in name2idx and right_name in name2idx:
            pairs.append((name2idx[left_name], name2idx[right_name]))
        else:
            unknown.update(
                n for n in (left_name, right_name) if n not in name2idx)
    if strict and unknown:
        raise ValueError(
            f"lr_mapping references unknown joint names: "
            f"{sorted(unknown)}. Check against `bvh.joint_names`.")
    return pairs


def _rest_leftward(
    bvh: Bvh, mapping: dict[str, str] | None = None,
) -> str | None:
    """Infer the skeleton's leftward axis from rest-pose L/R symmetry.

    Averages the left-minus-right rest-pose positions across all
    matching L/R joint pairs, projects onto the horizontal plane, and
    returns the dominant signed axis — a vector pointing from the
    character's right toward their left. Convention matches the public
    :meth:`Bvh.left_at` method (``up × forward``) via the right-hand
    rule. Used by transforms that need a stable topological orientation
    reference (mirror, retarget).

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    mapping : dict or None
        If provided, pair up joints according to this explicit mapping
        instead of reading ``bvh.lr_mapping``. Useful when the caller
        wants to compute leftward for a mapping that isn't cached on
        the object.

    Returns
    -------
    str or None
        Signed axis string (e.g. ``'-x'``) pointing toward the
        character's left, or ``None`` if no L/R pairs are available or
        the averaged offset is degenerate.
    """
    up_ax = _rest_upward(bvh)
    if up_ax is None:
        return None
    up_idx = _AXIS_CHAR_TO_IDX[up_ax[1]]

    if mapping is None:
        mapping = bvh.lr_mapping
    pairs = _resolve_lr_pairs(mapping, bvh.node_index)
    if not pairs:
        return None

    rest = bvh.rest_pose_positions()  # (N, 3)
    left_idx = [li for li, _ in pairs]
    right_idx = [ri for _, ri in pairs]
    avg_leftward = np.mean(rest[left_idx] - rest[right_idx], axis=0)
    avg_leftward[up_idx] = 0.0  # project to ground plane

    leftward_ax = get_main_direction(avg_leftward)
    if leftward_ax is None or leftward_ax[1] == up_ax[1]:
        return None
    return leftward_ax


def _infer_world_up(bvh: Bvh, warn: bool = True) -> str:
    """Infer the world vertical axis from the first animation frame.

    Uses frame 0's head-above-hips direction as the primary source (this
    reflects the actual world-space orientation at playback time, not
    just the rest-pose topology). Falls back to rest-pose topology if:
      - there are no animation frames
      - the first frame's head-hips direction is ambiguous (no dominant
        component clearly larger than the others)
      - the relevant joints are not present

    Issues a UserWarning when the first-frame inference succeeds but
    disagrees with the rest-pose topology (i.e. a different dominant
    axis) — this is a strong signal that the BVH file authored its rest
    pose in one convention and animates in another.

    Parameters
    ----------
    bvh : Bvh
        The skeleton with populated animation data.

    Returns
    -------
    str
        Signed axis string (e.g. '+y').
    """
    # Compute rest-pose topology once; used both as fallback and as the
    # reference to compare against when emitting the disagreement warning.
    rest_up = _rest_upward(bvh)

    def fallback() -> str:
        """The rest-pose axis, or the house default when there is none."""
        if rest_up is not None:
            return rest_up
        # Neither the animation nor the rest pose said anything, so '+y'
        # is invented, not measured, and the returned string cannot say
        # so — same silent substitution _fallback_forward_vector warns
        # about, same treatment.
        warnings.warn(
            "Could not infer a world up axis: this skeleton has no "
            "recognizable head/neck/chest/spine joint to read the first "
            "frame from, and its rest pose is degenerate (single node, or "
            "all offsets coincident). Falling back to '+y', the house "
            "default — anything derived from up (floor_height, "
            "foot_contacts, facing, ground-plane projections) rests on a "
            "guess here. Set `bvh.world_up = '<axis>'` to fix it.",
            UserWarning, stacklevel=3)
        return '+y'

    # Need animation frames to do first-frame inference
    if bvh.frame_count == 0:
        return fallback()

    # Try to find a "head-ish" and "hips-ish" joint in the skeleton.
    # Priority: head -> neck -> last joint in spine chain.
    name_lookup = {n.name.lower(): i for i, n in enumerate(bvh.nodes)}
    head_idx = None
    for candidate in ('head', 'neck', 'chest', 'spine'):
        if candidate in name_lookup:
            head_idx = name_lookup[candidate]
            break
    # Hips = the root (index 0)
    hips_idx = 0

    if head_idx is None or head_idx == hips_idx:
        return fallback()

    frame0 = bvh.node_positions(frame=0)
    head_hips = frame0[head_idx] - frame0[hips_idx]

    # Check for a clear dominant axis: the largest component must be
    # strictly larger than the second-largest (ratio > 2). Otherwise
    # the pose is ambiguous (character crouched / lying / leaning).
    abs_components = np.abs(head_hips)
    sorted_abs = np.sort(abs_components)
    if sorted_abs[-1] < 2.0 * sorted_abs[-2] or sorted_abs[-1] < 1e-6:
        return fallback()

    # Clear winner from animation data
    frame_up = get_main_direction(head_hips)
    if frame_up is None:
        return fallback()

    # Warn if rest-pose topology disagrees (different dominant axis)
    if warn and rest_up is not None and rest_up[1] != frame_up[1]:
        warnings.warn(
            f"Rest pose suggests world up is {rest_up!r} but the first "
            f"animation frame's head-hips direction is closer to "
            f"{frame_up!r}. Using {frame_up!r} from the animation data. "
            f"If this is wrong for your file, set it explicitly via "
            f"`bvh.world_up = '<axis>'`.",
            UserWarning,
            stacklevel=2,
        )

    return frame_up


def _world_leftward_units(
    bvh: Bvh,
    coords: npt.NDArray[np.float64],
    world_up: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-frame world-space leftward **unit vectors** for a whole clip.

    The single implementation of the facing construction's first stage:
    average ``(left_pos - right_pos)`` across matching L/R joint pairs
    in world space, project onto the plane perpendicular to
    ``world_up``, normalize. Continuous (not snapped to a signed axis);
    matches the ``up × forward`` right-hand-rule convention used by
    :meth:`Bvh.left_at`. Every consumer of the leftward geometry —
    :func:`_facing_basis`, the single-frame axis-string snappers, the
    bvhplot follow camera — routes through here.

    Parameters
    ----------
    bvh : Bvh
        The skeleton (used for L/R pairs).
    coords : np.ndarray
        Spatial coordinates of shape (F, N, 3).
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    leftward : ndarray of shape (F, 3)
        Leftward unit vector per frame. Rows where ``valid`` is False
        carry no usable direction and must not be consumed.
    valid : ndarray of shape (F,), bool
        False for every frame when no L/R pairs exist; per-frame False
        where the averaged leftward is degenerate (parallel to world
        up, or zero). Callers should handle invalid frames by falling
        back to the topological rest-pose direction.
    """
    num_frames = coords.shape[0]
    pairs = _resolve_lr_pairs(bvh.lr_mapping, bvh.node_index)
    if not pairs:
        return (np.zeros((num_frames, 3)),
                np.zeros(num_frames, dtype=bool))

    left_idx = [li for li, _ in pairs]
    right_idx = [ri for _, ri in pairs]
    up_vec = _axis_to_vector(world_up)

    leftward = np.mean(coords[:, left_idx] - coords[:, right_idx], axis=1)
    # Project onto plane perpendicular to world_up
    leftward = leftward - (leftward @ up_vec)[:, np.newaxis] * up_vec
    norms = np.linalg.norm(leftward, axis=1)
    valid = norms >= 1e-6
    leftward[valid] /= norms[valid, np.newaxis]
    return leftward, valid


def _world_leftward_unit_at_frame(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> npt.NDArray[np.float64] | None:
    """Single-frame view of :func:`_world_leftward_units`.

    Takes (N, 3) coordinates for one frame; returns the leftward unit
    vector, or ``None`` for the degenerate cases where the vectorized
    form flags the frame invalid (no L/R pairs, or leftward parallel to
    world up / zero).
    """
    leftward, valid = _world_leftward_units(
        bvh, frame_coords[np.newaxis], world_up)
    if not valid[0]:
        return None
    return leftward[0]


def _rest_forward_from_topology(
    bvh: Bvh,
    world_up: str,
) -> npt.NDArray[np.float64] | None:
    """Forward measured from the topological rest leftward, or ``None``.

    The middle step of the facing fallback chain: :func:`_rest_leftward`
    (the L/R axis measured against the *rest-pose* up) crossed with
    ``world_up``. Returns ``None`` when there is no usable rest leftward,
    or when it is (anti)parallel to ``world_up`` — possible when
    rest-pose topology and ``world_up`` disagree — i.e. exactly the
    cases where the caller must fall through to the arbitrary
    ``_FALLBACK_FORWARD`` default. Shared by
    :func:`_fallback_forward_vector` (which does the falling through)
    and :func:`_facing_is_measured` (which reports it), so the warning
    and the predicate cannot disagree about what counts as measured.
    """
    rest_left = _rest_leftward(bvh)
    if rest_left is None:
        return None
    forward = np.cross(_axis_to_vector(rest_left), _axis_to_vector(world_up))
    # Cross of two distinct signed axes is exactly a unit axis vector;
    # it is zero only in the (anti)parallel case described above.
    if float(np.linalg.norm(forward)) < 1e-6:
        return None
    return forward


def _facing_is_measured(bvh: Bvh, world_up: str) -> bool:
    """Whether the rest-pose facing chain measures this skeleton.

    ``True`` exactly when :attr:`Bvh.rest_forward` derives from the
    skeleton's L/R geometry — either the rest-pose L/R offsets give a
    usable leftward against ``world_up`` directly, or the topological
    rest leftward crossed with ``world_up`` does. ``False`` exactly when
    that chain bottoms out in the warned ``_FALLBACK_FORWARD`` house
    default. The implementation walks the same two measuring steps
    :func:`_compute_forward_at` walks on the rest pose, so the predicate
    tracks the fallback by construction rather than by parallel logic.
    """
    rest_leftward = _world_leftward_unit_at_frame(
        bvh, bvh.rest_pose_positions(), world_up)
    if rest_leftward is not None:
        return True
    return _rest_forward_from_topology(bvh, world_up) is not None


def _fallback_forward_vector(
    bvh: Bvh,
    world_up: str,
) -> npt.NDArray[np.float64]:
    """Unit forward vector for frames carrying no usable L/R direction.

    The vector form of the fallback chain documented on
    :meth:`Bvh.forward_at`: the topological rest-pose leftward crossed
    with ``world_up``; when the skeleton has no usable rest-pose L/R
    geometry either (or it is parallel to ``world_up``), the
    arbitrary-but-stable ``_FALLBACK_FORWARD`` axis for this up axis.
    Always axis-aligned and perpendicular to ``world_up``, so snapping
    it reproduces the string fallback exactly.
    """
    forward = _rest_forward_from_topology(bvh, world_up)
    if forward is not None:
        return forward
    fallback = _FALLBACK_FORWARD[world_up[1]]
    # The returned axis is a house default, not a measurement, and the
    # return type cannot say so — mirroring `_infer_world_up`, which warns
    # on the same class of silent substitution. Python's default filter
    # shows this once per call site, so per-frame consumers stay quiet.
    warnings.warn(
        f"No usable left/right geometry on this skeleton, so its facing "
        f"direction cannot be measured; falling back to {fallback!r}, the "
        f"arbitrary-but-stable horizontal axis for world_up={world_up!r}. "
        f"Anything derived from facing (rest_forward, forward_axis, "
        f"forward_at, left_at, facing_frame) is a default here, not a "
        f"property of this file. Set `bvh.lr_mapping = {{...}}` to fix it, "
        f"or check `bvh.has_lr_geometry` to detect it.",
        UserWarning, stacklevel=2)
    return _axis_to_vector(fallback)


def _facing_basis(
    bvh: Bvh,
    coords: npt.NDArray[np.float64],
    world_up: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Vectorized per-frame facing basis — the geometry under ``facing_frame``.

    Builds the right-handed orthonormal triple from the leftward
    geometry: ``forward = leftward × up``, ``left = up × forward``,
    ``up`` = the exact ``world_up`` unit vector on every frame. Frames
    flagged invalid by :func:`_world_leftward_units` receive the
    constant :func:`_fallback_forward_vector` basis instead — the same
    fallback chain the axis-string snappers use — and are reported as
    such in ``valid``, which is passed through unchanged so the report
    tracks the fallback by construction.

    Parameters
    ----------
    bvh : Bvh
        The skeleton (used for L/R pairs and the rest-pose fallback).
    coords : np.ndarray
        Spatial coordinates of shape (F, N, 3).
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    forward, left, up : ndarray of shape (F, 3) each
        Unit vectors forming a right-handed basis per frame
        (``forward × left = up``).
    valid : ndarray of shape (F,), bool
        False on the rows that carry the constant fallback basis.
    """
    up_vec = _axis_to_vector(world_up)
    leftward, valid = _world_leftward_units(bvh, coords, world_up)

    # leftward is unit and perpendicular to up on valid rows, so both
    # crosses below give unit vectors and `left` re-orthogonalizes to
    # leftward itself (up × (leftward × up) = leftward).
    forward = np.cross(leftward, up_vec)
    left = np.cross(up_vec, forward)
    if not valid.all():
        fallback_forward = _fallback_forward_vector(bvh, world_up)
        forward[~valid] = fallback_forward
        left[~valid] = np.cross(up_vec, fallback_forward)

    up = np.tile(up_vec, (coords.shape[0], 1))
    return forward, left, up, valid


def _signed_rotation_delta_around_axis(
    v_from: npt.NDArray[np.float64],
    v_to: npt.NDArray[np.float64],
    axis_vec: npt.NDArray[np.float64],
) -> float:
    """Signed angle (degrees) rotating ``v_from`` to ``v_to`` around ``axis_vec``.

    Positive = counter-clockwise when viewed from the direction of
    ``axis_vec``. Both input vectors are assumed unit-length and already
    in the plane perpendicular to ``axis_vec``.
    """
    cos_a = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(v_from, v_to), axis_vec))
    return float(np.degrees(np.arctan2(sin_a, cos_a)))


def _compute_forward_at(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> str:
    """Compute the character's world-space forward direction at a given frame.

    Uses the actual joint positions at the frame (not rest-pose offsets),
    so the result tracks root rotation, hip twist, and shoulder rotation
    as the character moves.

    Algorithm:
      1. Get the continuous world-space leftward vector via
         :func:`_world_leftward_unit_at_frame`.
      2. ``forward = cross(leftward, world_up_vec)`` — follows the
         ``up × forward = leftward`` right-hand-rule convention.
      3. Snap to nearest signed axis via :func:`get_main_direction`.
      4. Fallback: if leftward is degenerate (parallel to world_up) or
         no L/R pairs found, use :func:`_fallback_forward_vector`
         (rest-pose leftward crossed with world_up, then the arbitrary
         per-up-axis default).

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    frame_coords : np.ndarray
        Spatial coordinates of shape (N, 3) for a single frame.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    str
        Signed axis string (e.g. '-z') giving the character's forward
        direction in world space at the given frame.
    """
    leftward_vec = _world_leftward_unit_at_frame(bvh, frame_coords, world_up)

    if leftward_vec is None:
        forward_vec = _fallback_forward_vector(bvh, world_up)
    else:
        forward_vec = np.cross(leftward_vec, _axis_to_vector(world_up))

    # Both branches yield a unit vector perpendicular to world_up, so
    # the snap never degenerates and never lands on the up axis.
    forward_ax = get_main_direction(forward_vec)
    assert forward_ax is not None
    return forward_ax


def _compute_left_at(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> str:
    """Compute the character's world-space leftward direction at a given frame.

    Companion of :func:`_compute_forward_at` — implements
    :meth:`Bvh.left_at`.  Snaps the continuous world-space leftward
    vector (averaged left-minus-right over L/R joint pairs) to the
    nearest signed axis, falling back to the rest-pose leftward when the
    frame's leftward is degenerate, and finally to a left axis derived
    from the same arbitrary horizontal forward `_compute_forward_at`
    uses — so the (up, forward, left) triple stays consistent even
    without any L/R information.

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    frame_coords : np.ndarray
        Spatial coordinates of shape (N, 3) for a single frame.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    str
        Signed axis string (e.g. '-x') pointing toward the character's
        left in world space at the given frame.
    """
    left_vec = _world_leftward_unit_at_frame(bvh, frame_coords, world_up)

    if left_vec is not None:
        # Unit and perpendicular to world_up by construction, so the
        # snap never degenerates and never lands on the up axis.
        left_ax = get_main_direction(left_vec)
        assert left_ax is not None
        return left_ax

    rest_left = _rest_leftward(bvh)
    if rest_left is not None:
        return rest_left

    # No usable L/R information — derive left from the fallback forward
    # via the right-hand rule (leftward = up × forward). Routed through
    # _fallback_forward_vector so this path emits the same warning: a
    # defaulted "left" is exactly as unmeasured as a defaulted forward.
    fwd_vec = _fallback_forward_vector(bvh, world_up)
    up_vec = _axis_to_vector(world_up)
    left_ax = get_main_direction(np.cross(up_vec, fwd_vec))
    assert left_ax is not None  # axis-aligned cross never degenerate
    return left_ax


def _validate_axis_string(value: object, *, allow_unsigned: bool = False) -> str:
    """Validate and normalize a signed axis string.

    Accepts '+x'/'-x'/'+y'/'-y'/'+z'/'-z' (case-insensitive in the
    axis char). With ``allow_unsigned=True``, a bare axis char
    ('x'/'y'/'z') is also accepted and normalized to its positive form —
    for parameters where the sign is irrelevant (e.g. ``mirror``'s
    ``lateral_axis``). Returns the normalized form. Raises ValueError
    on any other input.
    """
    normalized = value.lower() if isinstance(value, str) else value
    if (allow_unsigned and isinstance(normalized, str)
            and normalized in _AXIS_CHAR_TO_IDX):
        normalized = '+' + normalized
    if normalized not in _VALID_AXIS_STRINGS:
        expected = sorted(_VALID_AXIS_STRINGS)
        if allow_unsigned:
            expected = expected + sorted(_AXIS_CHAR_TO_IDX)
        raise ValueError(
            f"Axis must be one of {expected}, got {value!r}")
    return normalized  # type: ignore[return-value]


def _axis_aligned_rotation(
    from_ax: str,
    to_ax: str,
) -> npt.NDArray[np.float64]:
    """3x3 rotation matrix mapping signed axis ``from_ax`` to ``to_ax``.

    Only works for axis-aligned unit vectors.  Returns exact matrices
    (entries are 0, 1, or -1) for lossless coordinate transforms.

    Parameters
    ----------
    from_ax, to_ax : str
        Signed axis strings, e.g. ``'+y'``, ``'-z'``.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Orthogonal rotation matrix with ``det = +1``.
    """
    from_vec = _axis_to_vector(from_ax)
    to_vec = _axis_to_vector(to_ax)

    if np.allclose(from_vec, to_vec):
        return np.eye(3, dtype=np.float64)

    if np.allclose(from_vec, -to_vec):
        # 180-degree rotation around a perpendicular axis
        from_idx = _AXIS_CHAR_TO_IDX[from_ax[1]]
        perp_axis = 'XYZ'[(from_idx + 1) % 3]
        R = _elementary_rotmat(np.pi, perp_axis)
        return np.round(R).astype(np.float64)

    # 90-degree rotation: cross product gives the rotation axis and sign
    cross = np.cross(from_vec, to_vec)
    rot_idx = int(np.argmax(np.abs(cross)))
    rot_sign = np.sign(cross[rot_idx])
    R = _elementary_rotmat(rot_sign * np.pi / 2, 'XYZ'[rot_idx])
    return np.round(R).astype(np.float64)
